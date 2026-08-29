"""Bounded INT8-ConvRot safetensors reader for the H3 TP backbone.

The released ``minimax_h3_fl2va_pruned_int8_convrot`` checkpoint stores the
50 DiT blocks as row-wise INT8 matrices, FP32 per-output-row scales, and a
small JSON ``comfy_quant`` marker.  This module reads only the header and the
rank-local byte ranges; it never maps the 20 GiB payload or builds a full
FP16 CPU state dict.

V100 (SM70) does not have the IMMA instructions required by ComfyUI's native
INT8 linear kernel.  The runtime therefore keeps the INT8 bytes resident and
materialises one rotated FP16 matrix at a time (W8A16).  ``rotate_activation``
implements the matching online Hadamard transform used by the official
ConvRot converter.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from custom_nodes.NoHostMMap.safetensors import resolve_no_host_path


CONVROT_GROUP_SIZE = 256
INT8_DTYPE = torch.int8
SCALE_DTYPE = torch.float32
MatrixKind = Literal["qkv", "out_proj", "fc1", "fc2"]
_MAX_HEADER_BYTES = 64 << 20
_MAX_MARKER_BYTES = 4096
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


_DTYPES: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "U8": torch.uint8,
}


@dataclass(frozen=True)
class Int8TensorSpec:
    """One safetensors payload range with its logical dtype and shape."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    data_offset: int
    n_bytes: int


@dataclass
class Int8MatrixShard:
    """Rank-local INT8 matrix and its output-row scales."""

    qdata: torch.Tensor
    scale: torch.Tensor
    out_features: int
    in_features: int
    source_name: str
    kind: MatrixKind
    rank: int
    world_size: int
    convrot: bool = True
    convrot_groupsize: int = CONVROT_GROUP_SIZE
    # ``full_in_features`` is useful for diagnostics and for checking that an
    # input-column shard starts/ends on a ConvRot group boundary.
    full_in_features: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features

    @property
    def input_start(self) -> int:
        if self.kind in {"out_proj", "fc2"}:
            return self.rank * self.in_features
        return 0


def _read_header(path: os.PathLike[str] | str) -> tuple[str, dict[str, Any], int, int]:
    """Read and validate only a safetensors header.

    Returns ``(resolved_path, header, data_base_offset, file_size)``.  All
    offsets are checked against the actual file size so a truncated download
    fails before any CUDA allocation.
    """

    resolved = resolve_no_host_path(path)
    with open(resolved, "rb", buffering=0) as handle:
        file_size = os.fstat(handle.fileno()).st_size
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"incomplete safetensors header: {resolved}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > _MAX_HEADER_BYTES:
            raise ValueError(
                f"safetensors header is unreasonably large ({header_size} bytes): {resolved}"
            )
        if header_size > file_size - 8:
            raise ValueError(
                f"safetensors header exceeds file size ({header_size} bytes): {resolved}"
            )
        raw = handle.read(header_size)
        if len(raw) != header_size:
            raise ValueError(f"incomplete safetensors header: {resolved}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header: {resolved}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {resolved}")

    data_base = 8 + int(header_size)
    max_end = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict) or "data_offsets" not in info:
            raise ValueError(f"malformed safetensors entry {name!r}")
        try:
            start, stop = (int(value) for value in info["data_offsets"])
        except (TypeError, ValueError):
            raise ValueError(f"malformed data_offsets for {name!r}") from None
        if start < 0 or stop < start:
            raise ValueError(f"invalid data_offsets for {name!r}: {start}, {stop}")
        max_end = max(max_end, stop)
        if data_base + stop > file_size:
            raise ValueError(
                f"tensor {name!r} exceeds file size: end={data_base + stop}, size={file_size}"
            )
    # A complete safetensors file has no trailing payload bytes.  Requiring
    # this catches a partially resumed download whose header is already valid.
    if data_base + max_end != file_size:
        raise ValueError(
            f"safetensors payload is incomplete: expected {data_base + max_end} bytes, "
            f"got {file_size} ({resolved})"
        )
    return resolved, header, data_base, file_size


def _spec_from_info(
    name: str,
    info: dict[str, Any],
    data_base: int,
) -> Int8TensorSpec:
    dtype = _DTYPES.get(str(info.get("dtype")))
    if dtype is None:
        raise ValueError(f"unsupported safetensors dtype for {name}: {info.get('dtype')!r}")
    try:
        shape = tuple(int(value) for value in info["shape"])
        start, stop = (int(value) for value in info["data_offsets"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed safetensors entry {name!r}") from exc
    if any(value < 0 for value in shape):
        raise ValueError(f"negative shape for {name!r}: {shape}")
    expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if stop - start != expected:
        raise ValueError(
            f"safetensors byte geometry mismatch for {name}: {stop - start} != {expected}"
        )
    return Int8TensorSpec(
        name=name,
        dtype=dtype,
        shape=shape,
        data_offset=data_base + start,
        n_bytes=stop - start,
    )


def _lookup(header: dict[str, Any], name: str) -> tuple[str, dict[str, Any]] | None:
    """Find a canonical key, accepting the optional diffusion_model prefix."""

    info = header.get(name)
    if info is not None:
        return name, info
    prefixed = f"diffusion_model.{name}"
    info = header.get(prefixed)
    if info is not None:
        return prefixed, info
    return None


def _marker_config(
    path: str,
    marker: Int8TensorSpec,
) -> dict[str, Any]:
    if marker.dtype != torch.uint8 or marker.n_bytes > _MAX_MARKER_BYTES:
        raise ValueError(f"invalid ConvRot marker tensor {marker.name}")
    with open(path, "rb", buffering=0) as handle:
        handle.seek(marker.data_offset)
        raw = handle.read(marker.n_bytes)
    if len(raw) != marker.n_bytes:
        raise OSError(f"short read for ConvRot marker {marker.name}")
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ConvRot marker JSON for {marker.name}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"ConvRot marker is not an object for {marker.name}")
    if config.get("format") != "int8_tensorwise":
        raise ValueError(f"unsupported quantization for {marker.name}: {config}")
    if not bool(config.get("convrot", False)):
        raise ValueError(f"{marker.name} is INT8 but not ConvRot")
    # Early official H3 exports omit the optional ``weight_rotated`` and
    # ``per_row`` fields; their presence is only a compatibility annotation.
    # ConvRot itself already means the serialized matrix was rotated by the
    # converter, so default those fields to the documented values while still
    # rejecting an explicit contradictory marker.
    if "weight_rotated" in config and not bool(config["weight_rotated"]):
        raise ValueError(f"{marker.name} does not mark pre-rotated weights")
    if "per_row" in config and not bool(config["per_row"]):
        raise ValueError(f"{marker.name} is not per-output-row INT8")
    group_size = int(config.get("convrot_groupsize", CONVROT_GROUP_SIZE))
    if group_size != CONVROT_GROUP_SIZE:
        raise ValueError(
            f"unsupported H3 ConvRot group size {group_size}; expected {CONVROT_GROUP_SIZE}"
        )
    config["convrot_groupsize"] = group_size
    config.setdefault("weight_rotated", True)
    config.setdefault("per_row", True)
    return config


def inspect_int8_matrices(
    path: os.PathLike[str] | str,
    names: set[str] | list[str] | tuple[str, ...],
) -> tuple[dict[str, dict[str, Int8TensorSpec]], dict[str, Any]]:
    """Inspect core matrices and validate their ConvRot metadata."""

    resolved, header, data_base, file_size = _read_header(path)
    requested = set(names)
    result: dict[str, dict[str, Int8TensorSpec]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for name in sorted(requested):
        found = _lookup(header, name)
        scale_found = _lookup(header, f"{name}_scale")
        marker_found = _lookup(header, f"{name[:-len('.weight')]}.comfy_quant") if name.endswith(".weight") else None
        if found is None or scale_found is None or marker_found is None:
            raise KeyError(f"INT8 ConvRot checkpoint is missing {name}, scale, or marker")
        actual, weight_info = found
        scale_actual, scale_info = scale_found
        marker_actual, marker_info = marker_found
        weight = _spec_from_info(actual, weight_info, data_base)
        scale = _spec_from_info(scale_actual, scale_info, data_base)
        marker = _spec_from_info(marker_actual, marker_info, data_base)
        if weight.dtype != INT8_DTYPE or scale.dtype != SCALE_DTYPE:
            raise ValueError(
                f"{name} must be I8 + F32 scale, got {weight.dtype} + {scale.dtype}"
            )
        if len(weight.shape) != 2 or scale.shape != (weight.shape[0], 1):
            raise ValueError(
                f"{name} shape/scale mismatch: {weight.shape} / {scale.shape}"
            )
        configs[name] = _marker_config(resolved, marker)
        result[name] = {"weight": weight, "scale": scale, "marker": marker}
    metadata = {
        "path": resolved,
        "file_size": file_size,
        "header_bytes": data_base - 8,
        "data_offset": data_base,
        "tensor_count": sum(key != "__metadata__" for key in header),
        "configs": configs,
        "metadata": header.get("__metadata__", {}),
    }
    return result, metadata


def inspect_dense_specs(
    path: os.PathLike[str] | str,
    names: set[str] | list[str] | tuple[str, ...],
) -> tuple[dict[str, Int8TensorSpec], dict[str, Any]]:
    """Inspect non-quantized H3 tensors used by the persistent worker."""

    resolved, header, data_base, file_size = _read_header(path)
    result: dict[str, Int8TensorSpec] = {}
    for name in set(names):
        found = _lookup(header, name)
        if found is None:
            raise KeyError(f"INT8 checkpoint is missing tensor {name}")
        actual, info = found
        spec = _spec_from_info(actual, info, data_base)
        if spec.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
            raise ValueError(f"dense H3 tensor {name} has unsupported dtype {spec.dtype}")
        result[name] = spec
    return result, {
        "path": resolved,
        "file_size": file_size,
        "header_bytes": data_base - 8,
        "data_offset": data_base,
        "tensor_count": sum(key != "__metadata__" for key in header),
    }


class Int8DiskReader:
    """Read safetensors slices through one bounded CPU staging allocation."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        device: torch.device | str,
        staging_bytes: int = 4 << 20,
    ) -> None:
        if staging_bytes <= 0:
            raise ValueError("staging_bytes must be positive")
        self.path = resolve_no_host_path(path)
        self.device = torch.device(device)
        self.file = open(self.path, "rb", buffering=0)
        self.staging = torch.empty(staging_bytes, dtype=torch.uint8, device="cpu")

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

    def __enter__(self) -> "Int8DiskReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _advise(self, offset: int, size: int, advice: int | None) -> None:
        if advice is None or not hasattr(os, "posix_fadvise"):
            return
        try:
            os.posix_fadvise(self.file.fileno(), offset, size, advice)
        except OSError:
            pass

    def _read_cpu(self, offset: int, size: int) -> torch.Tensor:
        if size > self.staging.numel():
            raise ValueError(
                f"read {size} exceeds staging buffer {self.staging.numel()}"
            )
        target = self.staging[:size]
        view_type = ctypes.c_ubyte * size
        view = memoryview(view_type.from_address(target.data_ptr()))
        try:
            self.file.seek(offset)
            done = 0
            while done < size:
                count = self.file.readinto(view[done:])
                if count is None or count <= 0:
                    raise OSError(f"short read from {self.path} at {offset + done}")
                done += count
        finally:
            view.release()
        return target

    def _copy(self, destination: torch.Tensor, destination_byte: int, source: int, size: int) -> None:
        flat = destination.view(torch.uint8).reshape(-1)
        copied = 0
        while copied < size:
            count = min(self.staging.numel(), size - copied)
            value = self._read_cpu(source + copied, count)
            flat[destination_byte + copied : destination_byte + copied + count].copy_(value)
            self._advise(source + copied, count, getattr(os, "POSIX_FADV_DONTNEED", None))
            copied += count

    def read_full(self, spec: Int8TensorSpec, target_dtype: torch.dtype | None = None) -> torch.Tensor:
        value = torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
        self._copy(value, 0, spec.data_offset, spec.n_bytes)
        if target_dtype is not None and target_dtype != value.dtype:
            value = value.to(dtype=target_dtype)
        return value

    def _read_output_shard(
        self,
        spec: Int8TensorSpec,
        parts: int,
        rank: int,
        world_size: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(spec.shape) != 2:
            raise ValueError(f"expected a matrix for output shard: {spec.name}")
        out_features, in_features = spec.shape
        if out_features % parts or (out_features // parts) % world_size:
            raise ValueError(
                f"{spec.name} output rows do not divide parts={parts}, TP={world_size}"
            )
        part_rows = out_features // parts
        local_rows = part_rows // world_size
        item_size = torch.empty((), dtype=spec.dtype).element_size()
        row_bytes = in_features * item_size
        result = torch.empty((parts * local_rows, in_features), dtype=spec.dtype, device=self.device)
        dst = 0
        for part in range(parts):
            start = part * part_rows + rank * local_rows
            size = local_rows * row_bytes
            self._copy(result, dst * row_bytes, spec.data_offset + start * row_bytes, size)
            dst += local_rows
        return result if result.dtype == target_dtype else result.to(dtype=target_dtype)

    def _read_input_shard(
        self,
        spec: Int8TensorSpec,
        rank: int,
        world_size: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(spec.shape) != 2:
            raise ValueError(f"expected a matrix for input shard: {spec.name}")
        out_features, in_features = spec.shape
        if in_features % world_size:
            raise ValueError(f"{spec.name} input width does not divide TP={world_size}")
        local_in = in_features // world_size
        item_size = torch.empty((), dtype=spec.dtype).element_size()
        row_bytes = in_features * item_size
        local_row_bytes = local_in * item_size
        byte_start = rank * local_row_bytes
        byte_stop = byte_start + local_row_bytes
        result = torch.empty((out_features, local_in), dtype=spec.dtype, device=self.device)
        # Keep the selected columns in the bounded CPU staging allocation.  A
        # full source row is read in chunks, never retained for the checkpoint.
        rows_per_chunk = max(1, self.staging.numel() // row_bytes)
        selected = torch.empty((rows_per_chunk, local_row_bytes), dtype=torch.uint8, device="cpu")
        for row_start in range(0, out_features, rows_per_chunk):
            rows = min(rows_per_chunk, out_features - row_start)
            full = self._read_cpu(spec.data_offset + row_start * row_bytes, rows * row_bytes).view(rows, row_bytes)
            selected[:rows].copy_(full[:, byte_start:byte_stop])
            result.view(torch.uint8).reshape(out_features, local_row_bytes)[row_start : row_start + rows].copy_(selected[:rows])
            self._advise(spec.data_offset + row_start * row_bytes, rows * row_bytes, getattr(os, "POSIX_FADV_DONTNEED", None))
        return result if result.dtype == target_dtype else result.to(dtype=target_dtype)

    def read_matrix_shard(
        self,
        specs: dict[str, Int8TensorSpec],
        role: MatrixKind,
        rank: int,
        world_size: int,
        *,
        convrot: bool = True,
        convrot_groupsize: int = CONVROT_GROUP_SIZE,
    ) -> Int8MatrixShard:
        weight_spec = specs["weight"]
        scale_spec = specs["scale"]
        if role in {"qkv", "fc1"}:
            parts = 3 if role == "qkv" else 2
            qdata = self._read_output_shard(weight_spec, parts, rank, world_size, INT8_DTYPE)
            scale = self._read_output_shard(scale_spec, parts, rank, world_size, SCALE_DTYPE)
        else:
            qdata = self._read_input_shard(weight_spec, rank, world_size, INT8_DTYPE)
            scale = self.read_full(scale_spec, SCALE_DTYPE)
        local_out, local_in = (int(value) for value in qdata.shape)
        full_in = int(weight_spec.shape[1])
        if convrot:
            if convrot_groupsize <= 0 or local_in % convrot_groupsize:
                raise ValueError(
                    f"{weight_spec.name} local input {local_in} is not divisible by ConvRot group {convrot_groupsize}"
                )
            if role in {"out_proj", "fc2"} and (rank * local_in) % convrot_groupsize:
                raise ValueError(f"{weight_spec.name} TP split is not ConvRot-group aligned")
        return Int8MatrixShard(
            qdata=qdata,
            scale=scale,
            out_features=local_out,
            in_features=local_in,
            source_name=weight_spec.name,
            kind=role,
            rank=rank,
            world_size=world_size,
            convrot=bool(convrot),
            convrot_groupsize=int(convrot_groupsize),
            full_in_features=full_in,
        )


def _fallback_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (int(size), str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    if size < 4 or (size & (size - 1)) != 0 or (math.log(size, 4) % 1) != 0:
        raise ValueError(f"ConvRot Hadamard size must be a power of four, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / math.sqrt(size)
    _HADAMARD_CACHE[key] = h
    return h


def hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the canonical normalized ConvRot Hadamard matrix.

    The helper is intentionally implemented locally instead of importing
    Comfy-Kitchen on every rank.  Kitchen's helper currently builds the same
    regular matrix and its optional CUDA extension can take tens of seconds to
    initialize on an SM70 process.  Keeping this tiny 256x256 matrix local
    makes worker startup deterministic; the matrix multiplication below still
    uses PyTorch's V100 FP16 path.
    """

    return _fallback_hadamard(size, device, dtype)


def rotate_activation(
    value: torch.Tensor,
    group_size: int = CONVROT_GROUP_SIZE,
) -> torch.Tensor:
    """Apply the online ConvRot transform ``x @ H`` to the last dimension."""

    if value.shape[-1] % group_size:
        raise ValueError(
            f"activation width {value.shape[-1]} is not divisible by ConvRot group {group_size}"
        )
    h = hadamard(group_size, value.device, value.dtype)
    groups = value.reshape(-1, value.shape[-1] // group_size, group_size)
    return torch.matmul(groups, h).reshape_as(value)


def dequantize_int8(
    matrix: Int8MatrixShard,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Materialise one matrix using the official ConvRot decode semantics.

    The safetensors payload stores ``W_rot = W @ H.T``.  Kitchen's eager
    implementation first multiplies the INT8 values by the FP32 row scale,
    rotates the result back to the ordinary weight basis, and only then casts
    to the requested compute dtype.  Doing the scale multiplication in FP16
    and rotating the activation online is algebraically equivalent for exact
    arithmetic, but it is not numerically equivalent after 50 residual blocks
    (and it also differs from the reference backend).  Keep the materialised
    per-layer buffer bounded while matching the reference order exactly.
    """

    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported INT8 output dtype: {dtype}")
    if matrix.qdata.dtype != torch.int8 or matrix.scale.dtype != torch.float32:
        raise ValueError(f"invalid INT8 shard dtypes for {matrix.source_name}")
    if matrix.scale.shape != (matrix.out_features, 1):
        raise ValueError(
            f"invalid INT8 scale shape for {matrix.source_name}: {matrix.scale.shape}"
        )
    # Decode in FP32 first.  The largest local shard is ~77M elements, so this
    # temporary is bounded to one layer (about 294 MiB) rather than the whole
    # checkpoint.  In particular, do not multiply a FP16 q tensor by a FP16
    # scale: the official row scales are FP32 and can be subnormal on V100.
    result = matrix.qdata.to(device=matrix.qdata.device, dtype=torch.float32)
    result.mul_(matrix.scale.to(device=result.device, dtype=torch.float32))
    if matrix.convrot:
        h = hadamard(
            matrix.convrot_groupsize,
            result.device,
            torch.float32,
        )
        groups = result.reshape(
            -1,
            result.shape[-1] // matrix.convrot_groupsize,
            matrix.convrot_groupsize,
        )
        # The regular H3 Hadamard is symmetric/orthonormal, so the same
        # transform used by Kitchen's _rotate_weight maps W_rot back to W.
        result = torch.matmul(groups, h.T).reshape_as(result)
    return result.to(dtype=dtype)


def dequantize_int8_w8a16(
    matrix: Int8MatrixShard,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Materialise a ConvRot shard for the SM70 W8A16 online path.

    The official V100 fallback does not rotate the weight back to the
    ordinary basis.  It rotates the activation online and multiplies the
    stored INT8 values by their per-output-row scale in the compute dtype
    immediately before the FP16 Tensor-Core GEMM.  Keeping this operation
    separate from :func:`dequantize_int8` makes the two numerical orders
    explicit and lets TP A/B tests select either one without changing the
    on-disk reader.
    """

    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported INT8 output dtype: {dtype}")
    if matrix.qdata.dtype != torch.int8 or matrix.scale.dtype != torch.float32:
        raise ValueError(f"invalid INT8 shard dtypes for {matrix.source_name}")
    if matrix.scale.shape != (matrix.out_features, 1):
        raise ValueError(
            f"invalid INT8 scale shape for {matrix.source_name}: {matrix.scale.shape}"
        )
    # This is deliberately scale-in-compute-dtype, matching the SM70 VAE
    # fallback and the official eager W8A16 order.  All released H3 scales
    # are normal FP16 values; retaining FP32 here would create a third,
    # otherwise accidental arithmetic variant.
    result = matrix.qdata.to(dtype=dtype)
    result.mul_(matrix.scale.to(device=result.device, dtype=dtype))
    return result


__all__ = [
    "CONVROT_GROUP_SIZE",
    "INT8_DTYPE",
    "Int8DiskReader",
    "Int8MatrixShard",
    "Int8TensorSpec",
    "dequantize_int8",
    "dequantize_int8_w8a16",
    "hadamard",
    "inspect_dense_specs",
    "inspect_int8_matrices",
    "rotate_activation",
]
