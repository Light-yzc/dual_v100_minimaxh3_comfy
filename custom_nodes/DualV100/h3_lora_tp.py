"""Bounded no-mmap Turbo LoRA shard reader for H3 tensor parallelism."""

from __future__ import annotations

import ctypes
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from custom_nodes.NoHostMMap.safetensors import resolve_no_host_path


DEFAULT_STAGING_BYTES = 4 << 20
SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}


@dataclass(frozen=True)
class SafeTensorSpec:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    data_offset: int
    n_bytes: int


@dataclass
class LoRALinearShard:
    a: torch.Tensor
    b: torch.Tensor
    role: str

    @property
    def rank(self) -> int:
        return self.a.shape[0]


def inspect_safetensors(
    path: os.PathLike[str] | str,
    names: set[str] | list[str] | tuple[str, ...],
    *,
    optional_names: set[str] | list[str] | tuple[str, ...] = (),
) -> tuple[dict[str, SafeTensorSpec], dict[str, object]]:
    path = resolve_no_host_path(path)
    with open(path, "rb", buffering=0) as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"incomplete safetensors header: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"incomplete safetensors header: {path}")
    header = json.loads(header_bytes.decode("utf-8"))
    base_offset = 8 + header_size
    requested = set(names)
    optional = set(optional_names)
    if not optional.issubset(requested):
        raise ValueError("optional_names must be a subset of names")
    specs: dict[str, SafeTensorSpec] = {}
    for name in requested:
        info = header.get(name)
        actual_name = name
        # LightX2V's ComfyUI export keeps the generic LoRA prefix in the
        # safetensors keys (diffusion_model.blocks.*), while the older H3
        # runtime LoRA stores blocks.*.  Keep the caller-facing canonical name
        # stable and only substitute the on-disk header entry here.
        if info is None and not name.startswith("diffusion_model."):
            prefixed = "diffusion_model." + name
            info = header.get(prefixed)
            if info is not None:
                actual_name = prefixed
        if info is None:
            continue
        dtype = SAFETENSORS_DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported LoRA dtype {info['dtype']} for {name}")
        shape = tuple(int(value) for value in info["shape"])
        start, stop = (int(value) for value in info["data_offsets"])
        expected = 1
        for value in shape:
            expected *= value
        expected *= torch.empty((), dtype=dtype).element_size()
        if stop - start != expected:
            raise ValueError(
                f"safetensors byte geometry mismatch for {name}: {stop-start} != {expected}"
            )
        specs[name] = SafeTensorSpec(
            name=actual_name,
            dtype=dtype,
            shape=shape,
            data_offset=base_offset + start,
            n_bytes=stop - start,
        )
    missing = requested.difference(specs)
    required_missing = missing.difference(optional)
    if required_missing:
        raise KeyError(
            f"LoRA is missing requested tensors: {sorted(required_missing)}"
        )
    metadata = {
        "file_size": Path(path).stat().st_size,
        "header_bytes": header_size,
        "tensor_count": sum(name != "__metadata__" for name in header),
        "metadata": header.get("__metadata__", {}),
    }
    return specs, metadata


class SafeTensorDiskReader:
    """Read dense safetensors values through fixed-size ordinary staging."""

    def __init__(self, path, device, staging_bytes: int = DEFAULT_STAGING_BYTES):
        if staging_bytes <= 0:
            raise ValueError("staging_bytes must be positive")
        self.path = resolve_no_host_path(path)
        self.device = torch.device(device)
        self.file = open(self.path, "rb", buffering=0)
        self.staging = torch.empty(staging_bytes, dtype=torch.uint8, device="cpu")
        self._advise(0, 0, getattr(os, "POSIX_FADV_SEQUENTIAL", None))

    def _advise(self, offset: int, size: int, advice: int | None) -> None:
        if advice is None or not hasattr(os, "posix_fadvise"):
            return
        try:
            os.posix_fadvise(self.file.fileno(), offset, size, advice)
        except OSError:
            pass

    def _drop_cache(self, offset: int, size: int) -> None:
        self._advise(offset, size, getattr(os, "POSIX_FADV_DONTNEED", None))

    def close(self):
        if not self.file.closed:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _read_cpu(self, source_offset: int, size: int) -> torch.Tensor:
        if size > self.staging.numel():
            raise ValueError(f"read {size} exceeds staging buffer {self.staging.numel()}")
        target = self.staging[:size]
        view_type = ctypes.c_ubyte * size
        view = memoryview(view_type.from_address(target.data_ptr()))
        try:
            self.file.seek(source_offset)
            done = 0
            while done < size:
                count = self.file.readinto(view[done:])
                if count is None or count <= 0:
                    raise OSError(
                        f"short read from {self.path} at {source_offset + done}"
                    )
                done += count
        finally:
            view.release()
        return target

    def _copy_contiguous(self, destination, destination_offset, source_offset, size):
        flat = destination.view(torch.uint8).reshape(-1)
        copied = 0
        while copied < size:
            count = min(self.staging.numel(), size - copied)
            source = self._read_cpu(source_offset + copied, count)
            flat[destination_offset + copied : destination_offset + copied + count].copy_(
                source
            )
            self._drop_cache(source_offset + copied, count)
            copied += count

    def read_full(self, spec: SafeTensorSpec, target_dtype: torch.dtype) -> torch.Tensor:
        source = torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
        self._copy_contiguous(source, 0, spec.data_offset, spec.n_bytes)
        return source if source.dtype == target_dtype else source.to(target_dtype)

    def read_output_segments(
        self,
        spec: SafeTensorSpec,
        parts: int,
        rank: int,
        world_size: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(spec.shape) != 2:
            raise ValueError(f"expected matrix for output shard: {spec.name}")
        out_features, in_features = spec.shape
        if out_features % parts or (out_features // parts) % world_size:
            raise ValueError(f"output rows do not divide parts={parts}, TP={world_size}")
        part_rows = out_features // parts
        local_rows = part_rows // world_size
        row_bytes = in_features * torch.empty((), dtype=spec.dtype).element_size()
        source = torch.empty(
            (parts * local_rows, in_features), dtype=spec.dtype, device=self.device
        )
        destination_offset = 0
        for part in range(parts):
            start = part * part_rows + rank * local_rows
            size = local_rows * row_bytes
            self._copy_contiguous(
                source,
                destination_offset,
                spec.data_offset + start * row_bytes,
                size,
            )
            destination_offset += size
        return source if source.dtype == target_dtype else source.to(target_dtype)

    def read_input_shard(
        self,
        spec: SafeTensorSpec,
        rank: int,
        world_size: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(spec.shape) != 2:
            raise ValueError(f"expected matrix for input shard: {spec.name}")
        out_features, in_features = spec.shape
        if in_features % world_size:
            raise ValueError(f"input columns do not divide TP={world_size}")
        item_size = torch.empty((), dtype=spec.dtype).element_size()
        local_in = in_features // world_size
        row_bytes = in_features * item_size
        local_row_bytes = local_in * item_size
        byte_start = rank * local_row_bytes
        byte_stop = byte_start + local_row_bytes
        source = torch.empty(
            (out_features, local_in), dtype=spec.dtype, device=self.device
        )
        source_bytes = source.view(torch.uint8).reshape(out_features, local_row_bytes)
        rows_per_chunk = max(1, self.staging.numel() // row_bytes)
        selected = torch.empty(
            (rows_per_chunk, local_row_bytes), dtype=torch.uint8, device="cpu"
        )
        for row_start in range(0, out_features, rows_per_chunk):
            rows = min(rows_per_chunk, out_features - row_start)
            full_rows = self._read_cpu(
                spec.data_offset + row_start * row_bytes, rows * row_bytes
            ).view(rows, row_bytes)
            selected[:rows].copy_(full_rows[:, byte_start:byte_stop])
            source_bytes[row_start : row_start + rows].copy_(selected[:rows])
            self._drop_cache(
                spec.data_offset + row_start * row_bytes,
                rows * row_bytes,
            )
        return source if source.dtype == target_dtype else source.to(target_dtype)


def h3_lora_names(block: int) -> dict[str, tuple[str, str]]:
    prefix = f"blocks.{block}"
    modules = {
        "qkv": f"{prefix}.attn.qkv_proj",
        "out_proj": f"{prefix}.attn.out_proj",
        "fc1": f"{prefix}.mlp.fc1",
        "fc2": f"{prefix}.mlp.fc2",
    }
    return {
        role: (f"{module}.lora_A.weight", f"{module}.lora_B.weight")
        for role, module in modules.items()
    }


def _validate_pair(a_spec: SafeTensorSpec, b_spec: SafeTensorSpec, role: str):
    if len(a_spec.shape) != 2 or len(b_spec.shape) != 2:
        raise ValueError(f"{role} LoRA factors must be matrices")
    rank, in_features = a_spec.shape
    out_features, b_rank = b_spec.shape
    if rank != b_rank:
        raise ValueError(f"{role} LoRA rank mismatch: A={a_spec.shape}, B={b_spec.shape}")
    return out_features, in_features, rank


def load_h3_lora_tp_shards(
    path,
    block: int,
    rank: int,
    world_size: int,
    device,
    staging_bytes: int = DEFAULT_STAGING_BYTES,
) -> tuple[dict[str, LoRALinearShard], dict[str, object]]:
    names = h3_lora_names(block)
    specs, metadata = inspect_safetensors(
        path, {name for pair in names.values() for name in pair}
    )
    result = {}
    with SafeTensorDiskReader(path, device, staging_bytes) as reader:
        for role, (a_name, b_name) in names.items():
            a_spec, b_spec = specs[a_name], specs[b_name]
            _validate_pair(a_spec, b_spec, role)
            target_dtype = torch.float16 if role in {"qkv", "fc1"} else torch.float32
            if role == "qkv":
                a = reader.read_full(a_spec, target_dtype)
                b = reader.read_output_segments(b_spec, 3, rank, world_size, target_dtype)
            elif role == "fc1":
                a = reader.read_full(a_spec, target_dtype)
                b = reader.read_output_segments(b_spec, 2, rank, world_size, target_dtype)
            else:
                a = reader.read_input_shard(a_spec, rank, world_size, target_dtype)
                b = reader.read_full(b_spec, target_dtype)
            result[role] = LoRALinearShard(a=a, b=b, role=role)
    return result, metadata


def load_h3_lora_full(
    path,
    block: int,
    device,
    staging_bytes: int = DEFAULT_STAGING_BYTES,
) -> tuple[dict[str, LoRALinearShard], dict[str, object]]:
    names = h3_lora_names(block)
    specs, metadata = inspect_safetensors(
        path, {name for pair in names.values() for name in pair}
    )
    result = {}
    with SafeTensorDiskReader(path, device, staging_bytes) as reader:
        for role, (a_name, b_name) in names.items():
            a_spec, b_spec = specs[a_name], specs[b_name]
            _validate_pair(a_spec, b_spec, role)
            target_dtype = torch.float16 if role in {"qkv", "fc1"} else torch.float32
            result[role] = LoRALinearShard(
                a=reader.read_full(a_spec, target_dtype),
                b=reader.read_full(b_spec, target_dtype),
                role=role,
            )
    return result, metadata


def lora_delta(x: torch.Tensor, shard: LoRALinearShard) -> torch.Tensor:
    if x.dtype != shard.a.dtype or x.dtype != shard.b.dtype:
        raise ValueError(
            f"{shard.role} LoRA dtype mismatch: x={x.dtype}, "
            f"A={shard.a.dtype}, B={shard.b.dtype}"
        )
    return torch.nn.functional.linear(
        torch.nn.functional.linear(x, shard.a), shard.b
    )


__all__ = [
    "LoRALinearShard",
    "SafeTensorDiskReader",
    "SafeTensorSpec",
    "h3_lora_names",
    "inspect_safetensors",
    "load_h3_lora_full",
    "load_h3_lora_tp_shards",
    "lora_delta",
]
