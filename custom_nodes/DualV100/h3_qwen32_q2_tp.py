"""Header-only Qwen3-VL-32B Q2 output-row tensor parallel primitives.

The Qwen checkpoint used by MiniMax H3 is a 50-layer GGUF file.  This module
contains the storage-side part of the Q2 route: it parses only the GGUF
header, describes contiguous output-row shards, and reads compressed bytes
through a small CPU staging buffer directly into the owning CUDA device.  The
language blocks use output-row all-gather semantics; no partial dot products
are reduced across ranks.

The module deliberately does not register ComfyUI nodes or start a process
group.  ``H3TPRuntime`` owns those concerns.  It can therefore be imported by
the P0 CPU audit without importing the ComfyUI application.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gguf
    from gguf.gguf_reader import GGML_QUANT_SIZES, GGMLQuantizationType
except ImportError as exc:  # pragma: no cover - a clear error is better than a late NameError
    gguf = None  # type: ignore[assignment]
    GGML_QUANT_SIZES = {}  # type: ignore[assignment]
    GGMLQuantizationType = None  # type: ignore[assignment,misc]
    _GGUF_IMPORT_ERROR = exc
else:
    _GGUF_IMPORT_ERROR = None

try:
    from custom_nodes.NoHostMMap.gguf_reader import NoMmapGGUFReader
except ImportError:  # standalone import from the DualV100 directory
    from ..NoHostMMap.gguf_reader import NoMmapGGUFReader  # type: ignore


QWEN32_HIDDEN_SIZE = 5120
QWEN32_INTERMEDIATE_SIZE = 25600
QWEN32_NUM_LAYERS = 50
QWEN32_NUM_HEADS = 64
QWEN32_NUM_KV_HEADS = 8
QWEN32_HEAD_DIM = 128
QWEN32_Q_DIM = QWEN32_NUM_HEADS * QWEN32_HEAD_DIM
QWEN32_KV_DIM = QWEN32_NUM_KV_HEADS * QWEN32_HEAD_DIM
QWEN32_LOCAL_HEADS = QWEN32_NUM_HEADS // 2
QWEN32_LOCAL_KV_HEADS = QWEN32_NUM_KV_HEADS // 2
QWEN32_LOCAL_Q_DIM = QWEN32_Q_DIM // 2
QWEN32_LOCAL_KV_DIM = QWEN32_KV_DIM // 2
DEFAULT_STAGING_MIB = 4
MIN_STAGING_BYTES = 64 * 1024
MIB = 1 << 20

MATRIX_ROLES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
NORM_ROLES = (
    "input_layernorm",
    "post_attention_layernorm",
    "q_norm",
    "k_norm",
)


def _require_gguf() -> None:
    if gguf is None:
        raise ImportError("the gguf package is required for Qwen32 Q2 loading") from _GGUF_IMPORT_ERROR


def _qtype_name(qtype: Any) -> str:
    return str(getattr(qtype, "name", qtype))


def _qtype_value(qtype: Any) -> Any:
    """Normalize an enum, integer, or qtype name to GGML's enum."""

    _require_gguf()
    if isinstance(qtype, GGMLQuantizationType):
        return qtype
    if isinstance(qtype, str):
        try:
            return GGMLQuantizationType[qtype]
        except KeyError as exc:
            raise ValueError(f"unknown GGML qtype {qtype!r}") from exc
    try:
        return GGMLQuantizationType(int(qtype))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown GGML qtype {qtype!r}") from exc


def qtype_geometry(qtype: Any) -> tuple[int, int]:
    """Return ``(block_elements, block_bytes)`` from GGUF's authoritative table."""

    qtype = _qtype_value(qtype)
    try:
        block_elements, block_bytes = GGML_QUANT_SIZES[qtype]
    except KeyError as exc:
        raise ValueError(f"GGUF has no storage geometry for {_qtype_name(qtype)}") from exc
    return int(block_elements), int(block_bytes)


def quantized_nbytes(shape: Sequence[int], qtype: Any) -> int:
    """Compute GGML storage bytes without constructing a tensor."""

    block_elements, block_bytes = qtype_geometry(qtype)
    elements = math.prod(int(value) for value in shape)
    if elements % block_elements:
        raise ValueError(
            f"shape {tuple(shape)} has {elements} elements, not aligned to "
            f"{block_elements}-element {_qtype_name(qtype)} blocks"
        )
    return elements // block_elements * block_bytes


def row_storage_bytes(columns: int, qtype: Any) -> int:
    """Return compressed bytes for one contiguous output row.

    Q/K quantizers store complete blocks per row.  Scalar qtypes have a block
    size of one and therefore use the same function.  Callers must pass a
    block-aligned row width for quantized matrices; silently rounding would
    make file ranges overlap.
    """

    columns = int(columns)
    block_elements, block_bytes = qtype_geometry(qtype)
    if columns <= 0 or columns % block_elements:
        raise ValueError(
            f"row width {columns} is not aligned to {block_elements} elements "
            f"for {_qtype_name(qtype)}"
        )
    return columns // block_elements * block_bytes


def _logical_shape(reader_tensor: Any) -> tuple[int, ...]:
    # GGUF stores dimensions in little-endian/column-major order.  Comfy's
    # loader reverses them before exposing a logical torch shape.
    return tuple(int(value) for value in reversed(tuple(reader_tensor.shape)))


def _metadata_original_shape(reader: Any, tensor_name: str) -> tuple[int, ...] | None:
    field = reader.get_field(f"comfy.gguf.orig_shape.{tensor_name}")
    if field is None:
        return None
    # Match ComfyUI-GGUF's metadata convention without importing the node.
    try:
        return tuple(int(field.parts[index][0]) for index in field.data)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TensorSpec:
    """Scalar-only description of one GGUF tensor.

    ``shape`` is the logical torch shape (output rows first for language
    matrices).  No payload bytes are retained by this object.
    """

    name: str
    qtype: Any
    shape: tuple[int, ...]
    data_offset: int
    n_bytes: int
    row_bytes: int
    block_elements: int
    block_bytes: int
    gguf_shape: tuple[int, ...] = ()
    original_shape: tuple[int, ...] | None = None
    path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "gguf_shape", tuple(int(value) for value in self.gguf_shape))
        object.__setattr__(self, "data_offset", int(self.data_offset))
        object.__setattr__(self, "n_bytes", int(self.n_bytes))
        object.__setattr__(self, "row_bytes", int(self.row_bytes))
        object.__setattr__(self, "block_elements", int(self.block_elements))
        object.__setattr__(self, "block_bytes", int(self.block_bytes))

    @property
    def qtype_name(self) -> str:
        return _qtype_name(self.qtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def output_rows(self) -> int:
        if not self.shape:
            return 1
        return int(self.shape[0])

    @property
    def out_features(self) -> int:
        return self.output_rows

    @property
    def in_features(self) -> int:
        if self.ndim < 2:
            return 1
        return int(self.shape[1])

    @property
    def matrix(self) -> bool:
        return self.ndim == 2

    @property
    def shape_2d(self) -> tuple[int, int]:
        if not self.matrix:
            raise ValueError(f"{self.name} is not a 2D matrix: {self.shape}")
        return self.out_features, self.in_features

    @property
    def end_offset(self) -> int:
        return self.data_offset + self.n_bytes

    @property
    def bytes(self) -> int:
        return self.n_bytes

    def validate(self) -> None:
        if self.data_offset < 0 or self.n_bytes < 0:
            raise ValueError(f"invalid offset/size for {self.name}")
        expected = quantized_nbytes(self.gguf_shape or tuple(reversed(self.shape)), self.qtype)
        if expected != self.n_bytes:
            raise ValueError(
                f"{self.name} storage mismatch: geometry={expected}, file={self.n_bytes}"
            )
        if self.matrix:
            expected_row = row_storage_bytes(self.in_features, self.qtype)
            if expected_row != self.row_bytes or self.row_bytes * self.out_features != self.n_bytes:
                raise ValueError(
                    f"{self.name} row geometry mismatch: row={self.row_bytes}, "
                    f"expected={expected_row}, bytes={self.n_bytes}"
                )


@dataclass(frozen=True)
class TensorShardDescriptor:
    """One contiguous output-row range in a GGUF tensor."""

    tensor_name: str
    qtype: Any
    original_shape: tuple[int, ...]
    rank: int
    world_size: int
    first_output_row: int
    output_row_count: int
    data_offset: int
    n_bytes: int
    row_bytes: int
    path: str = ""

    @property
    def name(self) -> str:
        return self.tensor_name

    @property
    def qtype_name(self) -> str:
        return _qtype_name(self.qtype)

    @property
    def shape(self) -> tuple[int, int]:
        return self.output_row_count, int(self.original_shape[1])

    @property
    def out_features(self) -> int:
        return self.output_row_count

    @property
    def in_features(self) -> int:
        return int(self.original_shape[1])

    @property
    def byte_start(self) -> int:
        return self.data_offset

    @property
    def byte_end(self) -> int:
        return self.data_offset + self.n_bytes

    @property
    def end_offset(self) -> int:
        return self.byte_end

    @property
    def bytes(self) -> int:
        return self.n_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "qtype": self.qtype_name,
            "original_shape": list(self.original_shape),
            "rank": self.rank,
            "world_size": self.world_size,
            "first_output_row": self.first_output_row,
            "output_row_count": self.output_row_count,
            "data_offset": self.data_offset,
            "n_bytes": self.n_bytes,
            "row_bytes": self.row_bytes,
            "path": self.path,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
        }


@dataclass(frozen=True)
class GGUFLayout:
    """Header-only view of a Qwen GGUF file."""

    path: str
    file_size: int
    data_offset: int
    header_prefix_bytes: int
    tensors: tuple[TensorSpec, ...]
    qtype_counts: Mapping[str, int]
    language_layers: Mapping[int, tuple[TensorSpec, ...]]

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def payload_bytes(self) -> int:
        return sum(item.n_bytes for item in self.tensors)

    @property
    def language_layer_count(self) -> int:
        return len(self.language_layers)

    def tensor(self, name: str) -> TensorSpec:
        for item in self.tensors:
            if item.name == name:
                return item
        raise KeyError(name)

    def matrix_specs(self, layer: int | None = None) -> tuple[TensorSpec, ...]:
        values: Iterable[TensorSpec]
        if layer is None:
            values = self.tensors
        else:
            values = self.language_layers.get(int(layer), ())
        return tuple(item for item in values if item.matrix and _matrix_role(item.name) is not None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_size": self.file_size,
            "data_offset": self.data_offset,
            "header_prefix_bytes": self.header_prefix_bytes,
            "tensor_count": self.tensor_count,
            "payload_bytes": self.payload_bytes,
            "qtype_counts": dict(self.qtype_counts),
            "language_layers": {
                str(layer): [tensor_to_dict(item) for item in tensors]
                for layer, tensors in self.language_layers.items()
            },
            "tensors": [tensor_to_dict(item) for item in self.tensors],
        }


def tensor_to_dict(spec: TensorSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "qtype": spec.qtype_name,
        "shape": list(spec.shape),
        "gguf_shape": list(spec.gguf_shape),
        "original_shape": list(spec.original_shape) if spec.original_shape else None,
        "data_offset": spec.data_offset,
        "n_bytes": spec.n_bytes,
        "row_bytes": spec.row_bytes,
        "block_elements": spec.block_elements,
        "block_bytes": spec.block_bytes,
        "path": spec.path,
    }


_LANGUAGE_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _language_layer(name: str) -> int | None:
    match = _LANGUAGE_LAYER_RE.search(name)
    if match is None:
        return None
    # The GGUF includes a vision tower under visual.blocks.*; only language
    # tensors have the exact `layers.N` component.
    return int(match.group(1))


def _matrix_role(name: str) -> str | None:
    for role in MATRIX_ROLES:
        if name.endswith(f".{role}.weight"):
            return role
    return None


def _norm_role(name: str) -> str | None:
    for role in NORM_ROLES:
        if name.endswith(f".{role}.weight"):
            return role
    return None


def inspect_gguf(path: os.PathLike[str] | str) -> GGUFLayout:
    """Parse a GGUF header and return scalar tensor/shard geometry.

    The function never calls the upstream ``gguf.GGUFReader`` and therefore
    never creates a payload ``numpy.memmap``.  Only the bounded header prefix
    retained by :class:`NoMmapGGUFReader` is read.
    """

    _require_gguf()
    path = os.fspath(path)
    file_size = os.path.getsize(path)
    reader = NoMmapGGUFReader(path)
    specs: list[TensorSpec] = []
    qtypes: Counter[str] = Counter()
    language_layers: dict[int, list[TensorSpec]] = {}
    for tensor in reader.tensors:
        qtype = _qtype_value(tensor.tensor_type)
        shape = _logical_shape(tensor)
        block_elements, block_bytes = qtype_geometry(qtype)
        n_bytes = int(tensor.n_bytes)
        # For a matrix, the first logical dimension is the output-row count.
        # For vectors and higher-rank tensors we still expose a useful
        # contiguous row geometry by flattening all but the last dimension.
        if len(shape) >= 1:
            row_count = math.prod(shape[:-1]) if len(shape) > 1 else 1
            columns = int(shape[-1])
            if columns % block_elements == 0:
                calculated_row_bytes = columns // block_elements * block_bytes
            else:
                # Non-matrix tensors may not have a block-aligned final axis;
                # retain an exact per-row byte stride where possible.  Matrix
                # tensors fail below rather than receiving a rounded range.
                calculated_row_bytes = n_bytes // max(1, row_count)
        else:
            row_count = 1
            calculated_row_bytes = n_bytes
        if len(shape) == 2:
            calculated_row_bytes = row_storage_bytes(shape[1], qtype)
            if calculated_row_bytes * shape[0] != n_bytes:
                raise ValueError(
                    f"{tensor.name} matrix geometry mismatch: "
                    f"{calculated_row_bytes}*{shape[0]} != {n_bytes}"
                )
        original_shape = _metadata_original_shape(reader, tensor.name)
        spec = TensorSpec(
            name=str(tensor.name),
            qtype=qtype,
            shape=shape,
            data_offset=int(tensor.data_offset),
            n_bytes=n_bytes,
            row_bytes=int(calculated_row_bytes),
            block_elements=block_elements,
            block_bytes=block_bytes,
            gguf_shape=tuple(int(value) for value in tensor.shape),
            original_shape=original_shape,
            path=path,
        )
        # Validate exact byte geometry for all normal tensors.  A few exotic
        # higher-rank tensors can have a non-aligned final axis; their total
        # GGML storage is still checked, while output-row sharding remains
        # intentionally restricted to 2D language matrices.
        expected = quantized_nbytes(spec.gguf_shape, qtype)
        if expected != n_bytes:
            raise ValueError(f"{spec.name} storage mismatch: {expected} != {n_bytes}")
        specs.append(spec)
        qtypes[spec.qtype_name] += 1
        layer = _language_layer(spec.name)
        if layer is not None:
            language_layers.setdefault(layer, []).append(spec)
    return GGUFLayout(
        path=path,
        file_size=file_size,
        data_offset=int(reader.data_offset),
        header_prefix_bytes=len(getattr(reader, "_header_bytes", b"")),
        tensors=tuple(specs),
        qtype_counts=dict(qtypes),
        language_layers={key: tuple(value) for key, value in sorted(language_layers.items())},
    )


def build_output_row_shards(
    spec: TensorSpec,
    world_size: int = 2,
    *,
    rank: int | None = None,
) -> tuple[TensorShardDescriptor, ...] | TensorShardDescriptor:
    """Split a 2D tensor into contiguous, complete output-row ranges.

    The split is deliberately independent of tensor qtype.  Input columns are
    never sliced, so Q2_K/Q3_K blocks remain intact and every output element
    retains its complete dot product.  ``rank`` is a convenience for runtime
    callers; the audit uses the all-ranks tuple.
    """

    if not isinstance(spec, TensorSpec):
        raise TypeError(f"expected TensorSpec, got {type(spec).__name__}")
    if not spec.matrix:
        raise ValueError(f"output-row sharding requires a 2D matrix: {spec.name} {spec.shape}")
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    rows, columns = spec.shape_2d
    if rows < world_size:
        raise ValueError(f"{spec.name} has {rows} rows, fewer than TP={world_size}")
    # Balanced contiguous ranges also work for synthetic odd-sized tests.  A
    # real Qwen32 matrix has even row counts for TP=2.
    base, remainder = divmod(rows, world_size)
    descriptors: list[TensorShardDescriptor] = []
    first = 0
    for current_rank in range(world_size):
        count = base + (1 if current_rank < remainder else 0)
        start = first
        first += count
        descriptors.append(
            TensorShardDescriptor(
                tensor_name=spec.name,
                qtype=spec.qtype,
                original_shape=spec.shape,
                rank=current_rank,
                world_size=world_size,
                first_output_row=start,
                output_row_count=count,
                data_offset=spec.data_offset + start * spec.row_bytes,
                n_bytes=count * spec.row_bytes,
                row_bytes=spec.row_bytes,
                path=spec.path,
            )
        )
    if sum(item.n_bytes for item in descriptors) != spec.n_bytes:
        raise AssertionError(f"shard bytes do not cover {spec.name}")
    if descriptors[-1].byte_end != spec.end_offset:
        raise AssertionError(f"shard ranges do not end at {spec.end_offset} for {spec.name}")
    if rank is None:
        return tuple(descriptors)
    if not 0 <= int(rank) < world_size:
        raise ValueError(f"rank {rank} outside world size {world_size}")
    return descriptors[int(rank)]


def language_matrix_specs(layout: GGUFLayout) -> dict[int, dict[str, TensorSpec]]:
    """Return ``layer -> role -> TensorSpec`` for the seven language matrices."""

    result: dict[int, dict[str, TensorSpec]] = {}
    for layer, tensors in layout.language_layers.items():
        roles: dict[str, TensorSpec] = {}
        for spec in tensors:
            role = _matrix_role(spec.name)
            if role is not None:
                if not spec.matrix:
                    raise ValueError(f"language matrix {spec.name} is not 2D: {spec.shape}")
                roles[role] = spec
        if roles:
            result[int(layer)] = roles
    return result


class Qwen32Q2DiskReader:
    """Bounded-staging direct reader for one GGUF file.

    The reader owns no model-sized CPU buffer.  A compressed shard is copied in
    chunks from a normal file descriptor to a uint8 CUDA tensor.  Every
    consumed range receives best-effort ``POSIX_FADV_DONTNEED`` so the service
    cgroup/page cache cannot retain the 7.9 GiB payload.
    """

    def __init__(
        self,
        path: os.PathLike[str] | str,
        *,
        staging_mib: int = DEFAULT_STAGING_MIB,
        staging_bytes: int | None = None,
        pin_memory: bool = True,
    ) -> None:
        self.path = os.fspath(path)
        if staging_bytes is None:
            staging_bytes = int(staging_mib) * MIB
        staging_bytes = int(staging_bytes)
        if staging_bytes < MIN_STAGING_BYTES:
            raise ValueError(f"staging buffer must be >= {MIN_STAGING_BYTES} bytes")
        self.staging_bytes = staging_bytes
        self.file = open(self.path, "rb", buffering=0)
        self._lock = threading.RLock()
        try:
            self.staging = torch.empty(
                self.staging_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=bool(pin_memory),
            )
        except (RuntimeError, TypeError):
            # CPU-only audits and containers without a pin-memory allocator
            # still preserve bounded staging and exact read semantics.
            self.staging = torch.empty(self.staging_bytes, dtype=torch.uint8, device="cpu")
        self.bytes_read = 0
        self.read_ops = 0
        self.fadvise_bytes = 0
        self.started = time.perf_counter()
        self.closed = False
        self._advise(0, 0, getattr(os, "POSIX_FADV_SEQUENTIAL", None))

    def _advise(self, offset: int, size: int, advice: int | None) -> None:
        if advice is None or not hasattr(os, "posix_fadvise"):
            return
        try:
            os.posix_fadvise(self.file.fileno(), int(offset), int(size), advice)
            if advice == getattr(os, "POSIX_FADV_DONTNEED", object()):
                self.fadvise_bytes += max(0, int(size))
        except OSError:
            pass

    def _drop_cache(self, offset: int, size: int) -> None:
        self._advise(offset, size, getattr(os, "POSIX_FADV_DONTNEED", None))

    def _check_open(self) -> None:
        if self.closed or self.file.closed:
            raise RuntimeError("Qwen32Q2DiskReader is closed")

    def _read_cpu(self, source_offset: int, size: int) -> torch.Tensor:
        self._check_open()
        if size < 0 or size > self.staging.numel():
            raise ValueError(f"read size {size} exceeds staging {self.staging.numel()}")
        target = self.staging[:size]
        view_type = ctypes.c_ubyte * int(size)
        view = memoryview(view_type.from_address(target.data_ptr()))
        try:
            self.file.seek(int(source_offset))
            done = 0
            while done < size:
                count = self.file.readinto(view[done:])
                if count is None or count <= 0:
                    raise OSError(f"short read from {self.path} at {source_offset + done}")
                done += int(count)
        finally:
            view.release()
        self.bytes_read += int(size)
        self.read_ops += 1
        return target

    def read_cpu_bytes(self, offset: int, size: int, *, drop_cache: bool = True) -> bytes:
        """Read a bounded range for CPU audits/tests without model-sized copies."""

        if size < 0:
            raise ValueError("size must be non-negative")
        if size > self.staging_bytes:
            raise ValueError(
                f"CPU byte reads are capped at the bounded staging size "
                f"({size} > {self.staging_bytes}); use read_raw() for direct "
                "final-owner streaming"
            )
        output = bytearray(size)
        with self._lock:
            done = 0
            while done < size:
                count = min(self.staging.numel(), size - done)
                source = self._read_cpu(offset + done, count)
                view_type = ctypes.c_ubyte * int(count)
                view = memoryview(view_type.from_address(source.data_ptr()))
                try:
                    output[done : done + count] = view
                finally:
                    view.release()
                if drop_cache:
                    self._drop_cache(offset + done, count)
                done += count
        return bytes(output)

    def _copy_to(self, destination: torch.Tensor, destination_offset: int, source_offset: int, size: int, *, stream: torch.cuda.Stream | None = None, non_blocking: bool = False) -> None:
        flat = destination.reshape(-1)
        done = 0
        while done < size:
            count = min(self.staging.numel(), size - done)
            source = self._read_cpu(source_offset + done, count)
            if stream is not None and destination.is_cuda:
                with torch.cuda.stream(stream):
                    flat[destination_offset + done : destination_offset + done + count].copy_(
                        source, non_blocking=non_blocking
                    )
                # The single staging slot may be reused only after this copy
                # completes.  This synchronization is local to the copy stream;
                # it never synchronizes the caller's compute stream.
                if non_blocking:
                    event = torch.cuda.Event()
                    with torch.cuda.stream(stream):
                        event.record()
                    event.synchronize()
            else:
                flat[destination_offset + done : destination_offset + done + count].copy_(
                    source, non_blocking=non_blocking
                )
            self._drop_cache(source_offset + done, count)
            done += count

    def read_raw(
        self,
        spec_or_shard: TensorSpec | TensorShardDescriptor,
        *,
        device: torch.device | str = "cpu",
        stream: torch.cuda.Stream | None = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Materialize compressed bytes for one tensor/range on ``device``."""

        if isinstance(spec_or_shard, TensorSpec):
            offset, size = spec_or_shard.data_offset, spec_or_shard.n_bytes
        elif isinstance(spec_or_shard, TensorShardDescriptor):
            offset, size = spec_or_shard.data_offset, spec_or_shard.n_bytes
        else:
            raise TypeError(type(spec_or_shard).__name__)
        target = torch.device(device)
        with self._lock:
            output = torch.empty(size, dtype=torch.uint8, device=target)
            self._copy_to(output, 0, offset, size, stream=stream, non_blocking=non_blocking)
        return output

    def read_shard(self, descriptor: TensorShardDescriptor, **kwargs: Any) -> torch.Tensor:
        return self.read_raw(descriptor, **kwargs)

    def read_tensor(self, spec: TensorSpec, **kwargs: Any) -> torch.Tensor:
        return self.read_raw(spec, **kwargs)

    def stats(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "staging_bytes": int(self.staging_bytes),
            "bytes_read": int(self.bytes_read),
            "read_ops": int(self.read_ops),
            "fadvise_dontneed_bytes": int(self.fadvise_bytes),
            "elapsed_seconds": time.perf_counter() - self.started,
            "closed": bool(self.closed or self.file.closed),
        }

    def close(self) -> None:
        with self._lock:
            if not self.file.closed:
                self.file.close()
            self.closed = True
            staging = self.staging
            self.staging = torch.empty(0, dtype=torch.uint8)
            del staging

    def __enter__(self) -> "Qwen32Q2DiskReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


# Compatibility aliases used by early integration prototypes.
Qwen32Q2ShardReader = Qwen32Q2DiskReader
Qwen32Q2DirectReader = Qwen32Q2DiskReader


_DEQUANT_MODULE: Any | bool | None = None


def _load_dequant_module() -> Any | None:
    global _DEQUANT_MODULE
    if _DEQUANT_MODULE is False:
        return None
    if _DEQUANT_MODULE is not None:
        return _DEQUANT_MODULE
    candidates = (
        "custom_nodes.ComfyUI-GGUF.dequant",
        "ComfyUI-GGUF.dequant",
    )
    for module_name in candidates:
        try:
            _DEQUANT_MODULE = importlib.import_module(module_name)
            return _DEQUANT_MODULE
        except Exception:
            continue
    _DEQUANT_MODULE = False
    return None


def _to_uint32(values: torch.Tensor) -> torch.Tensor:
    values = values.view(torch.uint8).to(torch.int32)
    return (
        values[:, 0]
        | values[:, 1] << 8
        | values[:, 2] << 16
        | values[:, 3] << 24
    ).unsqueeze(1)


def _split_block_dims(blocks: torch.Tensor, *dims: int) -> tuple[torch.Tensor, ...]:
    sizes = list(dims) + [blocks.shape[1] - sum(dims)]
    return torch.split(blocks, sizes, dim=1)


def _dequantize_q2_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = _split_block_dims(blocks, 16, 64, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0x0F)).reshape(n_blocks, 16, 1)
    ml = (dmin * (scales >> 4)).reshape(n_blocks, 16, 1)
    shifts = torch.tensor((0, 2, 4, 6), dtype=torch.uint8, device=blocks.device).reshape(1, 1, 4, 1)
    values = (qs.reshape(n_blocks, -1, 1, 32) >> shifts) & 3
    values = values.reshape(n_blocks, 16, 16)
    return (dl * values - ml).reshape(n_blocks, 256)


def _dequantize_q3_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = _split_block_dims(blocks, 32, 64, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lshifts = torch.tensor((0, 4), dtype=torch.uint8, device=blocks.device).reshape(1, 2, 1)
    hshifts = torch.tensor((0, 2, 4, 6), dtype=torch.uint8, device=blocks.device).reshape(1, 4, 1)
    lscales = (lscales.reshape(n_blocks, 1, 8) >> lshifts).reshape(n_blocks, 16)
    hscales = (hscales.reshape(n_blocks, 1, 4) >> hshifts).reshape(n_blocks, 16)
    scales = ((lscales & 0x0F) | ((hscales & 0x03) << 4)).to(torch.int8) - 32
    dl = (d * scales).reshape(n_blocks, 16, 1)
    ql = qs.reshape(n_blocks, -1, 1, 32) >> hshifts.reshape(1, 1, 4, 1)
    bit_shifts = torch.arange(8, dtype=torch.uint8, device=blocks.device).reshape(1, 1, 8, 1)
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> bit_shifts
    ql = ql.reshape(n_blocks, 16, 16) & 3
    qh = (qh.reshape(n_blocks, 16, 16) & 1) ^ 1
    values = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * values).reshape(n_blocks, 256)


def _scalar_view(raw: torch.Tensor, qtype: Any, shape: Sequence[int], dtype: torch.dtype | None) -> torch.Tensor | None:
    qtype = _qtype_value(qtype)
    scalar_dtypes = {
        GGMLQuantizationType.F32: torch.float32,
        GGMLQuantizationType.F16: torch.float16,
        GGMLQuantizationType.BF16: torch.bfloat16,
        GGMLQuantizationType.F64: torch.float64,
        GGMLQuantizationType.I8: torch.int8,
        GGMLQuantizationType.I16: torch.int16,
        GGMLQuantizationType.I32: torch.int32,
        GGMLQuantizationType.I64: torch.int64,
    }
    base = scalar_dtypes.get(qtype)
    if base is None:
        return None
    values = raw.view(base).reshape(tuple(int(value) for value in shape))
    if dtype is not None and values.is_floating_point() and values.dtype != dtype:
        values = values.to(dtype)
    return values


@torch.inference_mode()
def dequantize_ggml(
    raw: torch.Tensor,
    qtype: Any,
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize one bounded compressed tensor on its current device.

    The repository's ComfyUI-GGUF torch implementations are preferred for
    Q2_K/Q3_K and other K-quants.  A NumPy fallback is retained for CPU-only
    audits and unusual qtypes; it is never used by the descriptor/reader path
    unless the caller explicitly asks to dequantize such a qtype.
    """

    if raw.dtype != torch.uint8:
        raise TypeError(f"compressed GGML storage must be uint8, got {raw.dtype}")
    shape = tuple(int(value) for value in shape)
    scalar = _scalar_view(raw, qtype, shape, dtype)
    if scalar is not None:
        return scalar
    qtype = _qtype_value(qtype)
    block_elements, block_bytes = qtype_geometry(qtype)
    expected = quantized_nbytes(shape, qtype)
    if raw.numel() != expected:
        raise ValueError(
            f"{_qtype_name(qtype)} storage mismatch: got {raw.numel()} bytes, expected {expected}"
        )
    blocks = raw.contiguous().reshape(-1, block_bytes)
    if qtype == GGMLQuantizationType.Q2_K:
        return _dequantize_q2_k(blocks, dtype).reshape(shape)
    if qtype == GGMLQuantizationType.Q3_K:
        return _dequantize_q3_k(blocks, dtype).reshape(shape)
    module = _load_dequant_module()
    if module is not None and hasattr(module, "dequantize"):
        # The helper accepts a flat/last-dimension byte tensor and preserves the
        # output shape.  Ensure a contiguous uint8 view so custom kernels do
        # not accidentally interpret a strided shard.
        return module.dequantize(raw.contiguous(), qtype, shape, dtype=dtype)
    if raw.is_cuda:
        # gguf.quants is NumPy-only.  This fallback is intentionally explicit:
        # it copies only this one requested shard, never the whole checkpoint.
        cpu = raw.detach().to("cpu")
    else:
        cpu = raw
    try:
        values = gguf.quants.dequantize(cpu.numpy(), qtype)
    except Exception as exc:
        raise RuntimeError(f"no torch dequantizer for {_qtype_name(qtype)}") from exc
    output = torch.from_numpy(values).reshape(shape)
    return output.to(device=raw.device, dtype=dtype)


@torch.inference_mode()
def dequantize_shard(
    raw: torch.Tensor,
    descriptor: TensorShardDescriptor,
    *,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return dequantize_ggml(raw, descriptor.qtype, descriptor.shape, dtype=dtype)


class Qwen32Q2SelectedEmbedding(nn.Module):
    """Dequantize only the token rows used by one conditioning request.

    Qwen32's complete embedding would occupy roughly 1.45 GiB in FP16.  The
    checkpoint stores every logical row in a contiguous GGML byte range, so a
    request can coalesce adjacent unique token IDs, read those row spans with
    the same bounded staging reader, and scatter the small result back to the
    original batch/sequence shape.  No full vocabulary tensor is constructed
    on either host or device.
    """

    def __init__(
        self,
        layout_or_path: GGUFLayout | os.PathLike[str] | str,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
        staging_mib: int = DEFAULT_STAGING_MIB,
        reader: Qwen32Q2DiskReader | None = None,
        tensor_name: str = "model.embed_tokens.weight",
    ) -> None:
        super().__init__()
        self.layout = (
            layout_or_path
            if isinstance(layout_or_path, GGUFLayout)
            else inspect_gguf(layout_or_path)
        )
        self.spec = self.layout.tensor(tensor_name)
        if not self.spec.matrix or self.spec.in_features != QWEN32_HIDDEN_SIZE:
            raise ValueError(
                f"unexpected Qwen32 embedding geometry {self.spec.shape}"
            )
        self.device = torch.device(device)
        self.compute_dtype = dtype
        self.reader = reader or Qwen32Q2DiskReader(
            self.layout.path, staging_mib=staging_mib
        )
        self._owns_reader = reader is None
        self.lookup_count = 0
        self.requested_tokens = 0
        self.unique_tokens = 0
        self.payload_bytes_read = 0
        self.dequant_seconds = 0.0

    @staticmethod
    def _runs(token_ids: Sequence[int]) -> Iterator[tuple[int, int]]:
        if not token_ids:
            return
        start = previous = int(token_ids[0])
        for value in token_ids[1:]:
            value = int(value)
            if value != previous + 1:
                yield start, previous + 1
                start = value
            previous = value
        yield start, previous + 1

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(input_ids):
            raise TypeError("input_ids must be a torch.Tensor")
        ids = input_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        flat = ids.reshape(-1)
        if flat.numel() == 0:
            return torch.empty(
                tuple(ids.shape) + (self.spec.in_features,),
                device=self.device,
                dtype=out_dtype or self.compute_dtype,
            )
        minimum = int(flat.min().item())
        maximum = int(flat.max().item())
        if minimum < 0 or maximum >= self.spec.out_features:
            raise IndexError(
                f"Qwen token ID range [{minimum}, {maximum}] is outside "
                f"[0, {self.spec.out_features})"
            )

        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        unique_values = [int(value) for value in unique.tolist()]
        dense = torch.empty(
            (len(unique_values), self.spec.in_features),
            device=self.device,
            dtype=out_dtype or self.compute_dtype,
        )
        cursor = 0
        started = time.perf_counter()
        for first, stop in self._runs(unique_values):
            count = stop - first
            descriptor = TensorShardDescriptor(
                tensor_name=self.spec.name,
                qtype=self.spec.qtype,
                original_shape=self.spec.shape,
                rank=0,
                world_size=1,
                first_output_row=first,
                output_row_count=count,
                data_offset=self.spec.data_offset + first * self.spec.row_bytes,
                n_bytes=count * self.spec.row_bytes,
                row_bytes=self.spec.row_bytes,
                path=self.spec.path,
            )
            raw = self.reader.read_shard(descriptor, device=self.device)
            rows = dequantize_shard(
                raw, descriptor, dtype=out_dtype or self.compute_dtype
            )
            dense[cursor : cursor + count].copy_(rows)
            cursor += count
            self.payload_bytes_read += descriptor.n_bytes
            del raw, rows
        self.dequant_seconds += time.perf_counter() - started
        inverse = inverse.to(device=self.device)
        result = dense.index_select(0, inverse).reshape(
            tuple(ids.shape) + (self.spec.in_features,)
        )
        self.lookup_count += 1
        self.requested_tokens += int(flat.numel())
        self.unique_tokens += len(unique_values)
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "tensor": self.spec.name,
            "device": str(self.device),
            "lookup_count": self.lookup_count,
            "requested_tokens": self.requested_tokens,
            "unique_tokens": self.unique_tokens,
            "payload_bytes_read": self.payload_bytes_read,
            "full_fp16_bytes_avoided": self.spec.out_features
            * self.spec.in_features
            * torch.empty((), dtype=torch.float16).element_size(),
            "dequant_seconds": self.dequant_seconds,
            "reader": self.reader.stats(),
        }

    def close(self) -> None:
        if self._owns_reader:
            self.reader.close()


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match Comfy's Qwen ``RMSNorm`` arithmetic and output dtype.

    The stock MiniMax/Qwen path calls ``torch.nn.functional.rms_norm`` rather
    than a hand-rolled reduction.  Keeping that operator matters over 50
    residual blocks: changing the reduction/rounding sequence can accumulate
    into materially different conditioning even when every quantized row is
    decoded correctly.
    """

    return F.rms_norm(
        x,
        (x.shape[-1],),
        weight=weight.to(device=x.device, dtype=x.dtype),
        eps=eps,
    )


def _apply_rope_fallback(q: torch.Tensor, k: torch.Tensor, freqs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if freqs is None:
        return q, k
    cos, sin, nsin = freqs
    q_out = q * cos
    q_half = q_out.shape[-1] // 2
    q_out[..., :q_half].addcmul_(q[..., q_half:], nsin)
    q_out[..., q_half:].addcmul_(q[..., :q_half], sin)
    k_out = k * cos
    k_half = k_out.shape[-1] // 2
    k_out[..., :k_half].addcmul_(k[..., k_half:], nsin)
    k_out[..., k_half:].addcmul_(k[..., :k_half], sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    # Comfy's attention selector knows the V100-safe backend.  Keep a direct
    # PyTorch fallback for CPU tests and standalone use.
    try:
        from comfy.ldm.modules.attention import optimized_attention_for_device
    except (ImportError, ModuleNotFoundError):
        optimized_attention_for_device = None
    kwargs = {"enable_gqa": True} if q.shape[1] != k.shape[1] else {}
    if optimized_attention_for_device is not None:
        op = optimized_attention_for_device(q.device, mask=mask is not None, small_input=True)
        # Backend/runtime failures are not swallowed: both ranks must fail
        # together instead of silently changing the numerical implementation.
        # ``skip_reshape`` tells Comfy that Q/K/V are already [B,H,S,D], but
        # does not by itself preserve that layout at the output.  The default
        # Comfy attention path otherwise returns [B,S,H*D]; treating that as
        # [B,H,S,D] below silently permutes token/head features and destroys
        # Qwen conditioning semantics.  Keep the native attention layout so
        # this function has the same contract as torch SDPA.
        return op(
            q,
            k,
            v,
            q.shape[1],
            mask=mask,
            skip_reshape=True,
            skip_output_reshape=True,
            **kwargs,
        )
    try:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, **kwargs)
    except TypeError:
        # Compatibility for older torch versions that predate enable_gqa.
        if q.shape[1] != k.shape[1]:
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)


class Qwen32Q2OutputLinear(nn.Module):
    """Lazy compressed output-row linear owned by one TP rank."""

    def __init__(
        self,
        descriptor: TensorShardDescriptor,
        reader: Qwen32Q2DiskReader | None = None,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_dequantized: bool = False,
    ) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.reader = reader
        self.device = torch.device(device)
        self.compute_dtype = dtype
        self.cache_dequantized = bool(cache_dequantized)
        self.raw: torch.Tensor | None = None
        self.dequantized: torch.Tensor | None = None
        self.load_seconds = 0.0
        self.dequant_seconds = 0.0
        self.forward_count = 0

    @property
    def resident_bytes(self) -> int:
        total = 0
        if self.raw is not None:
            total += int(self.raw.numel())
        if self.dequantized is not None:
            total += int(self.dequantized.numel() * self.dequantized.element_size())
        return total

    @property
    def loaded(self) -> bool:
        return self.raw is not None

    def load(self, *, stream: torch.cuda.Stream | None = None, non_blocking: bool = False) -> None:
        if self.raw is not None:
            return
        if self.reader is None:
            raise RuntimeError(f"no disk reader for {self.descriptor.tensor_name}")
        started = time.perf_counter()
        self.raw = self.reader.read_shard(
            self.descriptor,
            device=self.device,
            stream=stream,
            non_blocking=non_blocking,
        )
        self.load_seconds += time.perf_counter() - started

    @torch.inference_mode()
    def weight(self) -> torch.Tensor:
        self.load()
        if self.raw is None:
            raise RuntimeError("raw shard disappeared during load")
        if self.dequantized is None:
            started = time.perf_counter()
            self.dequantized = dequantize_shard(
                self.raw,
                self.descriptor,
                dtype=self.compute_dtype,
            ).contiguous()
            self.dequant_seconds += time.perf_counter() - started
            if not self.cache_dequantized:
                # Keep compressed bytes as the residency owner.  The temporary
                # dequantized matrix remains alive until F.linear returns.
                pass
        return self.dequantized

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight()
        output = F.linear(x, weight)
        self.forward_count += 1
        if not self.cache_dequantized:
            self.dequantized = None
            del weight
        return output

    def clear(self) -> None:
        raw, dequantized = self.raw, self.dequantized
        self.raw = None
        self.dequantized = None
        del raw, dequantized

    def stats(self) -> dict[str, Any]:
        return {
            "tensor": self.descriptor.tensor_name,
            "rank": self.descriptor.rank,
            "compressed_bytes": self.descriptor.n_bytes if self.raw is not None else 0,
            "dequantized_bytes": (
                int(self.dequantized.numel() * self.dequantized.element_size())
                if self.dequantized is not None
                else 0
            ),
            "load_seconds": self.load_seconds,
            "dequant_seconds": self.dequant_seconds,
            "forward_count": self.forward_count,
        }


class Qwen32Q2RankLocalBlock(nn.Module):
    """One rank's output-row Qwen block.

    ``gather`` must concatenate rank-local tensors in rank order.  It may be a
    callback backed by ``dist.all_gather_into_tensor`` or a test callback using
    ``torch.cat``.  Four gathers are performed in the documented order:
    local attention heads, local O rows, local SwiGLU rows, and local Down
    rows.  No collective performs a numerical sum.
    """

    def __init__(
        self,
        rank: int,
        matrices: Mapping[str, Qwen32Q2OutputLinear],
        norms: Mapping[str, torch.Tensor],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.device = torch.device(device)
        self.compute_dtype = dtype
        self.eps = float(eps)
        missing = set(MATRIX_ROLES).difference(matrices)
        if missing:
            raise ValueError(f"missing Qwen32 matrices for rank block: {sorted(missing)}")
        self.matrices = nn.ModuleDict(dict(matrices))
        self.norms: dict[str, torch.Tensor] = {
            key: value.to(device=self.device, dtype=dtype).detach()
            for key, value in norms.items()
            if key in {"input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"}
        }
        for key in ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"):
            if key not in self.norms:
                raise ValueError(f"missing norm {key}")
        self.forward_count = 0
        self.gather_count = 0
        self.last_stats: dict[str, Any] = {}

    def _gather(self, value: torch.Tensor, gather: Callable[..., torch.Tensor] | None, label: str) -> torch.Tensor:
        if gather is None:
            # A single-rank/CPU oracle can pass a tensor that is already full
            # width.  For a real two-rank block, fail closed rather than using
            # a silent partial result.
            expected = {
                "attention": QWEN32_Q_DIM,
                "o_rows": QWEN32_HIDDEN_SIZE,
                "mlp": QWEN32_INTERMEDIATE_SIZE,
                "down": QWEN32_HIDDEN_SIZE,
            }[label]
            if value.shape[-1] != expected:
                raise RuntimeError(f"{label} requires output-row all-gather (got {value.shape[-1]}, expected {expected})")
            return value
        try:
            result = gather(value, label=label, rank=self.rank)
        except TypeError:
            try:
                result = gather(value, label)
            except TypeError:
                result = gather(value)
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"gather callback returned {type(result).__name__}")
        self.gather_count += 1
        return result

    @torch.inference_mode()
    def forward_attention_local(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: Any = None,
    ) -> torch.Tensor:
        b, s, hidden = x.shape
        if hidden != QWEN32_HIDDEN_SIZE:
            raise ValueError(f"Qwen32 hidden width must be {QWEN32_HIDDEN_SIZE}, got {hidden}")
        h = _rms_norm(x, self.norms["input_layernorm"], self.eps)
        q = self.matrices["q_proj"](h).view(b, s, QWEN32_LOCAL_HEADS, QWEN32_HEAD_DIM).transpose(1, 2)
        k = self.matrices["k_proj"](h).view(b, s, QWEN32_LOCAL_KV_HEADS, QWEN32_HEAD_DIM).transpose(1, 2)
        v = self.matrices["v_proj"](h).view(b, s, QWEN32_LOCAL_KV_HEADS, QWEN32_HEAD_DIM).transpose(1, 2)
        q = _rms_norm(q, self.norms["q_norm"], self.eps)
        k = _rms_norm(k, self.norms["k_norm"], self.eps)
        try:
            from comfy.text_encoders.llama import apply_rope
        except (ImportError, ModuleNotFoundError):
            apply_rope = None
        if freqs_cis is not None and apply_rope is not None:
            q, k = apply_rope(q, k, freqs_cis)
        elif freqs_cis is not None:
            q, k = _apply_rope_fallback(q, k, freqs_cis)
        attn = _attention(q, k, v, attention_mask)
        return attn.transpose(1, 2).reshape(b, s, QWEN32_LOCAL_Q_DIM)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: Any = None,
        gather: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        local_attn = self.forward_attention_local(
            x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
        )
        full_attn = self._gather(local_attn, gather, "attention")
        residual = x
        local_o = self.matrices["o_proj"](full_attn)
        full_o = self._gather(local_o, gather, "o_rows")
        x = residual + full_o

        residual = x
        h = _rms_norm(x, self.norms["post_attention_layernorm"], self.eps)
        gate = self.matrices["gate_proj"](h)
        up = self.matrices["up_proj"](h)
        local_mlp = F.silu(gate) * up
        full_mlp = self._gather(local_mlp, gather, "mlp")
        local_down = self.matrices["down_proj"](full_mlp)
        full_down = self._gather(local_down, gather, "down")
        x = residual + full_down
        self.forward_count += 1
        self.last_stats = {
            "forward_count": self.forward_count,
            "gather_count": self.gather_count,
            "finite": bool(torch.isfinite(x).all().item()),
            "shape": list(x.shape),
        }
        return x

    def clear(self) -> None:
        for matrix in self.matrices.values():
            matrix.clear()
        norms = self.norms
        self.norms = {}
        del norms
        gc.collect()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def stats(self) -> dict[str, Any]:
        matrix_stats = {key: value.stats() for key, value in self.matrices.items()}
        norm_bytes = sum(
            int(value.numel() * value.element_size()) for value in self.norms.values()
        )
        compressed_bytes = sum(item["compressed_bytes"] for item in matrix_stats.values())
        dequantized_bytes = sum(item["dequantized_bytes"] for item in matrix_stats.values())
        return {
            "rank": self.rank,
            "resident_compressed_bytes": compressed_bytes,
            "resident_dequantized_bytes": dequantized_bytes,
            "resident_dense_bytes": norm_bytes,
            "resident_bytes": compressed_bytes + dequantized_bytes + norm_bytes,
            "forward_count": self.forward_count,
            "gather_count": self.gather_count,
            "matrices": matrix_stats,
            "last": dict(self.last_stats),
        }


# Short aliases make the integration layer less verbose and preserve names
# used by the design document's code sketches.
Qwen32Q2TPBlock = Qwen32Q2RankLocalBlock
Qwen32Q2Block = Qwen32Q2RankLocalBlock


def all_gather_output_rows(
    value: torch.Tensor,
    *,
    rank: int,
    world_size: int = 2,
    group: Any = None,
    label: str | None = None,
) -> torch.Tensor:
    """NCCL/Gloo output-row gather callback for :class:`Qwen32Q2RankLocalBlock`."""

    return _gather_last_dim(
        value,
        rank=rank,
        world_size=world_size,
        group=group,
        label=label,
    )


def _gather_last_dim(value: torch.Tensor, *, rank: int, world_size: int = 2, group: Any = None, label: str | None = None) -> torch.Tensor:
    """Correctly reshape all-gather output for tensors whose last dim is rows."""

    del rank, label
    if int(world_size) == 1:
        return value
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("output-row gather requested without an initialized process group")
    local = value.contiguous()
    flat = local.reshape(-1, local.shape[-1])
    gathered = torch.empty((int(world_size) * flat.shape[0], flat.shape[1]), dtype=local.dtype, device=local.device)
    torch.distributed.all_gather_into_tensor(gathered, flat, group=group)
    # all_gather_into_tensor places rank chunks along dim 0; restore the batch
    # dimensions and concatenate the feature/row dimension.
    chunks = gathered.reshape((int(world_size),) + tuple(local.shape))
    return chunks.movedim(0, -2).reshape(
        tuple(local.shape[:-1]) + (local.shape[-1] * int(world_size),)
    )


class Qwen32Q2TPBackbone(nn.Module):
    """Lazy rank-local 50-layer Qwen language backbone.

    This class intentionally starts in ``META_ONLY``.  ``load_layer`` reads
    only the requested layer's compressed shards.  The default ``evict`` mode
    clears each layer after its forward, while ``partial``/``full`` retain the
    configured layer set.  Header descriptors remain available after
    ``clear()`` so the next conditioning request can reload from disk.
    """

    def __init__(
        self,
        layout_or_path: GGUFLayout | os.PathLike[str] | str,
        *,
        rank: int = 0,
        world_size: int = 2,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
        staging_mib: int = DEFAULT_STAGING_MIB,
        residency: str = "evict",
        keep_layers: int | Sequence[int] = 0,
        cache_dequantized: bool = False,
        reader: Qwen32Q2DiskReader | None = None,
    ) -> None:
        super().__init__()
        if isinstance(layout_or_path, GGUFLayout):
            layout = layout_or_path
        else:
            layout = inspect_gguf(layout_or_path)
        self.layout = layout
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} outside world size {self.world_size}")
        self.device = torch.device(device)
        self.compute_dtype = dtype
        self.residency = str(residency).strip().lower()
        if self.residency not in {"evict", "partial", "full"}:
            raise ValueError("residency must be evict, partial, or full")
        if isinstance(keep_layers, int):
            self.keep_layer_ids = set(range(max(0, int(keep_layers))))
        else:
            self.keep_layer_ids = {int(item) for item in keep_layers}
        if self.residency == "full":
            self.keep_layer_ids = set(layout.language_layers)
        self.reader = reader or Qwen32Q2DiskReader(layout.path, staging_mib=staging_mib)
        self._owns_reader = reader is None
        self.layer_specs = language_matrix_specs(layout)
        if not self.layer_specs:
            raise ValueError("GGUF contains no Qwen language matrix specs")
        self.blocks: dict[int, Qwen32Q2RankLocalBlock] = {}
        self.state = "META_ONLY"
        self.forward_count = 0
        self.layer_load_seconds = 0.0
        self.layer_forward_seconds = 0.0
        self._cache_dequantized = bool(cache_dequantized)

    def _norm_specs(self, layer: int) -> dict[str, TensorSpec]:
        return {
            role: spec
            for spec in self.layout.language_layers.get(layer, ())
            if (role := _norm_role(spec.name)) is not None
        }

    def _norm_values(self, layer: int) -> dict[str, torch.Tensor]:
        # Norms are tiny F32 tensors.  They are read through the same bounded
        # reader, then converted once to compute dtype.
        result: dict[str, torch.Tensor] = {}
        for role, spec in self._norm_specs(layer).items():
            raw = self.reader.read_tensor(spec, device=self.device)
            values = dequantize_ggml(raw, spec.qtype, spec.shape, dtype=self.compute_dtype).reshape(spec.shape)
            result[role] = values
            del raw
        return result

    def load_layer(self, layer: int) -> Qwen32Q2RankLocalBlock:
        layer = int(layer)
        cached = self.blocks.get(layer)
        if cached is not None:
            return cached
        roles = self.layer_specs.get(layer)
        if roles is None:
            raise KeyError(f"Qwen language layer {layer} is absent")
        started = time.perf_counter()
        matrices: dict[str, Qwen32Q2OutputLinear] = {}
        for role in MATRIX_ROLES:
            spec = roles.get(role)
            if spec is None:
                raise ValueError(f"layer {layer} is missing matrix {role}")
            descriptor = build_output_row_shards(spec, self.world_size, rank=self.rank)
            assert isinstance(descriptor, TensorShardDescriptor)
            matrices[role] = Qwen32Q2OutputLinear(
                descriptor,
                self.reader,
                device=self.device,
                dtype=self.compute_dtype,
                cache_dequantized=self._cache_dequantized,
            )
        norms = self._norm_values(layer)
        block = Qwen32Q2RankLocalBlock(
            self.rank,
            matrices,
            norms,
            device=self.device,
            dtype=self.compute_dtype,
        )
        self.blocks[layer] = block
        self.layer_load_seconds += time.perf_counter() - started
        self.state = "ENCODING"
        return block

    def trim(self, keep_layers: int | Sequence[int] = 0) -> None:
        if isinstance(keep_layers, int):
            keep = set(range(max(0, int(keep_layers))))
        else:
            keep = {int(item) for item in keep_layers}
        for layer in tuple(self.blocks):
            if layer not in keep:
                self.blocks[layer].clear()
                del self.blocks[layer]
        self.keep_layer_ids = keep
        self.state = "DIT_READY" if not self.blocks else "ENCODING"
        gc.collect()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def forward_hidden(
        self,
        hidden: torch.Tensor,
        *,
        layer_ids: Sequence[int] | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: Any = None,
        gather: Callable[..., torch.Tensor] | None = None,
        deepstack_embeds: Sequence[torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        after_layer: Callable[[int, torch.Tensor], torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if layer_ids is None:
            layer_ids = tuple(sorted(self.layer_specs))
        current = hidden
        started = time.perf_counter()
        for layer in layer_ids:
            layer = int(layer)
            block = self.load_layer(layer)
            current = block(
                current,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                gather=gather,
            )
            # Match stock Qwen3-VL: the first DeepStack tensors are added after
            # their corresponding decoder blocks at visual-token positions.
            # Runtime integration must broadcast these small inputs so both
            # ranks execute the same mutation and collective schedule.
            if deepstack_embeds is not None and layer < len(deepstack_embeds):
                if visual_pos_masks is None:
                    raise ValueError("deepstack_embeds requires visual_pos_masks")
                current[visual_pos_masks] = (
                    current[visual_pos_masks]
                    + deepstack_embeds[layer].to(device=current.device, dtype=current.dtype)
                )
            if after_layer is not None:
                replacement = after_layer(layer, current)
                if replacement is not None:
                    if replacement.shape != current.shape:
                        raise ValueError(
                            f"after_layer changed shape {tuple(current.shape)} to "
                            f"{tuple(replacement.shape)}"
                        )
                    current = replacement
            if self.residency == "evict" or (
                self.residency == "partial" and layer not in self.keep_layer_ids
            ):
                block.clear()
                self.blocks.pop(layer, None)
        self.forward_count += 1
        self.layer_forward_seconds += time.perf_counter() - started
        self.state = "DIT_READY" if self.residency == "evict" else "ENCODING"
        return current

    # Integration-friendly aliases.  Embedding/vision assembly remains in the
    # stock Qwen outer model; this method consumes already assembled hidden.
    forward = forward_hidden
    encode_hidden = forward_hidden

    def clear(self) -> None:
        for block in self.blocks.values():
            block.clear()
        self.blocks.clear()
        gc.collect()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        self.state = "META_ONLY"

    def stats(self) -> dict[str, Any]:
        block_stats = {str(layer): block.stats() for layer, block in self.blocks.items()}
        allocated = reserved = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
        return {
            "state": self.state,
            "rank": self.rank,
            "world_size": self.world_size,
            "device": str(self.device),
            "residency": self.residency,
            "keep_layers": sorted(self.keep_layer_ids),
            "loaded_layers": sorted(self.blocks),
            "descriptor_bytes": 0,
            "header_tensor_count": self.layout.tensor_count,
            "resident_compressed_bytes": sum(item["resident_compressed_bytes"] for item in block_stats.values()),
            "resident_dequantized_bytes": sum(item["resident_dequantized_bytes"] for item in block_stats.values()),
            "resident_dense_bytes": sum(item["resident_dense_bytes"] for item in block_stats.values()),
            "resident_bytes": sum(item["resident_bytes"] for item in block_stats.values()),
            "cuda_allocated_bytes": allocated,
            "cuda_reserved_bytes": reserved,
            "forward_count": self.forward_count,
            "layer_load_seconds": self.layer_load_seconds,
            "layer_forward_seconds": self.layer_forward_seconds,
            "reader": self.reader.stats(),
            "blocks": block_stats,
        }

    get_stats = stats

    def close(self) -> None:
        self.clear()
        if self._owns_reader:
            self.reader.close()

    def __enter__(self) -> "Qwen32Q2TPBackbone":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


Qwen32Q2Backbone = Qwen32Q2TPBackbone
Qwen32Q2TPModel = Qwen32Q2TPBackbone


def payload_mmap_hits(path: os.PathLike[str] | str, *, pid: int | None = None) -> list[str]:
    """Return process-map lines containing ``path`` (normally an empty list)."""

    if pid is None:
        pid = os.getpid()
    maps_path = Path(f"/proc/{int(pid)}/maps")
    if not maps_path.is_file():
        return []
    needle = os.path.realpath(os.fspath(path))
    try:
        lines = maps_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line for line in lines if needle in line]


__all__ = [
    "DEFAULT_STAGING_MIB",
    "GGUFLayout",
    "MATRIX_ROLES",
    "QWEN32_HIDDEN_SIZE",
    "QWEN32_INTERMEDIATE_SIZE",
    "QWEN32_NUM_LAYERS",
    "Qwen32Q2Backbone",
    "Qwen32Q2Block",
    "Qwen32Q2DiskReader",
    "Qwen32Q2DirectReader",
    "Qwen32Q2OutputLinear",
    "Qwen32Q2RankLocalBlock",
    "Qwen32Q2SelectedEmbedding",
    "Qwen32Q2ShardReader",
    "Qwen32Q2TPBackbone",
    "Qwen32Q2TPBlock",
    "Qwen32Q2TPModel",
    "TensorShardDescriptor",
    "TensorSpec",
    "all_gather_output_rows",
    "build_output_row_shards",
    "dequantize_ggml",
    "dequantize_shard",
    "inspect_gguf",
    "language_matrix_specs",
    "payload_mmap_hits",
    "quantized_nbytes",
    "qtype_geometry",
    "row_storage_bytes",
    "tensor_to_dict",
]
