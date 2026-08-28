"""No-mmap Q4_0 shard reader for two-way MiniMax H3 tensor parallelism.

This module is intentionally narrow: it handles the four matrix layouts used
by H3 attention/MLP blocks and standard GGML Q4_0 storage.  It never maps the
GGUF payload and never constructs a full CPU tensor.  Disk bytes are copied
through a bounded ordinary CPU staging tensor into rank-local CUDA storage.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gguf
import torch

from custom_nodes.NoHostMMap.gguf_reader import NoMmapGGUFReader


Q4_BLOCK_ELEMENTS = 32
Q4_BLOCK_BYTES = 18
DEFAULT_STAGING_BYTES = 4 << 20
ShardKind = Literal["qkv", "out_proj", "fc1", "fc2"]
_Q4_NIBBLE_SHIFTS: dict[torch.device, torch.Tensor] = {}


@dataclass(frozen=True)
class Q4MatrixSpec:
    name: str
    out_features: int
    in_features: int
    data_offset: int
    n_bytes: int
    row_bytes: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features


@dataclass
class Q4MatrixShard:
    raw: torch.Tensor
    out_features: int
    in_features: int
    source_name: str
    kind: ShardKind | Literal["full"]
    rank: int | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features


def inspect_q4_matrices(
    path: os.PathLike[str] | str,
    names: set[str] | tuple[str, ...] | list[str],
) -> tuple[dict[str, Q4MatrixSpec], dict[str, int]]:
    """Read a bounded GGUF header and return validated Q4_0 matrix specs."""

    path = os.fspath(path)
    requested = set(names)
    reader = NoMmapGGUFReader(path)
    specs: dict[str, Q4MatrixSpec] = {}
    for tensor in reader.tensors:
        if tensor.name not in requested:
            continue
        if tensor.tensor_type != gguf.GGMLQuantizationType.Q4_0:
            raise ValueError(
                f"{tensor.name} is {tensor.tensor_type.name}, expected standard Q4_0"
            )
        shape = tuple(int(value) for value in reversed(tensor.shape))
        if len(shape) != 2:
            raise ValueError(f"{tensor.name} is not a matrix: {shape}")
        out_features, in_features = shape
        if in_features % Q4_BLOCK_ELEMENTS:
            raise ValueError(
                f"{tensor.name} input width {in_features} splits a Q4_0 block"
            )
        row_bytes = in_features // Q4_BLOCK_ELEMENTS * Q4_BLOCK_BYTES
        if row_bytes * out_features != int(tensor.n_bytes):
            raise ValueError(
                f"{tensor.name} byte geometry mismatch: "
                f"{row_bytes}*{out_features} != {tensor.n_bytes}"
            )
        specs[tensor.name] = Q4MatrixSpec(
            name=tensor.name,
            out_features=out_features,
            in_features=in_features,
            data_offset=int(tensor.data_offset),
            n_bytes=int(tensor.n_bytes),
            row_bytes=row_bytes,
        )
    missing = requested.difference(specs)
    if missing:
        raise KeyError(f"GGUF is missing requested tensors: {sorted(missing)}")
    metadata = {
        "file_size": Path(path).stat().st_size,
        "header_prefix_bytes": len(reader._header_bytes),
        "data_offset": int(reader.data_offset),
        "tensor_count": len(reader.tensors),
    }
    return specs, metadata


def output_row_segments(
    spec: Q4MatrixSpec,
    kind: Literal["qkv", "fc1"],
    rank: int,
    world_size: int,
) -> list[tuple[int, int]]:
    parts = 3 if kind == "qkv" else 2
    if spec.out_features % parts:
        raise ValueError(f"{spec.name} output width is incompatible with {kind}")
    part_rows = spec.out_features // parts
    if part_rows % world_size:
        raise ValueError(f"{spec.name} {kind} segment rows do not divide TP={world_size}")
    local_rows = part_rows // world_size
    return [
        (part * part_rows + rank * local_rows, part * part_rows + (rank + 1) * local_rows)
        for part in range(parts)
    ]


class Q4DiskReader:
    """Reusable bounded-staging reader for one GGUF file and one device."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        device: torch.device | str,
        staging_bytes: int = DEFAULT_STAGING_BYTES,
    ) -> None:
        if staging_bytes <= 0:
            raise ValueError("staging_bytes must be positive")
        self.path = os.fspath(path)
        self.device = torch.device(device)
        self.file = open(self.path, "rb", buffering=0)
        self.staging = torch.empty(staging_bytes, dtype=torch.uint8, device="cpu")
        self._advise(
            0,
            0,
            getattr(os, "POSIX_FADV_SEQUENTIAL", None),
        )

    def _advise(self, offset: int, size: int, advice: int | None) -> None:
        """Best-effort page-cache control for the low-RAM service cgroup."""

        if advice is None or not hasattr(os, "posix_fadvise"):
            return
        try:
            os.posix_fadvise(self.file.fileno(), offset, size, advice)
        except OSError:
            # Correctness never depends on fadvise support (some filesystems and
            # containers reject it); bounded userspace staging remains intact.
            pass

    def _drop_cache(self, offset: int, size: int) -> None:
        self._advise(offset, size, getattr(os, "POSIX_FADV_DONTNEED", None))

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

    def __enter__(self) -> "Q4DiskReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
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

    def _copy_contiguous(
        self,
        destination: torch.Tensor,
        destination_offset: int,
        source_offset: int,
        size: int,
    ) -> None:
        flat = destination.reshape(-1)
        copied = 0
        while copied < size:
            count = min(self.staging.numel(), size - copied)
            source = self._read_cpu(source_offset + copied, count)
            flat[destination_offset + copied : destination_offset + copied + count].copy_(
                source
            )
            self._drop_cache(source_offset + copied, count)
            copied += count

    def read_full(self, spec: Q4MatrixSpec) -> Q4MatrixShard:
        raw = torch.empty(
            (spec.out_features, spec.row_bytes), dtype=torch.uint8, device=self.device
        )
        self._copy_contiguous(raw, 0, spec.data_offset, spec.n_bytes)
        return Q4MatrixShard(
            raw=raw,
            out_features=spec.out_features,
            in_features=spec.in_features,
            source_name=spec.name,
            kind="full",
            rank=None,
        )

    def read_output_shard(
        self,
        spec: Q4MatrixSpec,
        kind: Literal["qkv", "fc1"],
        rank: int,
        world_size: int,
    ) -> Q4MatrixShard:
        segments = output_row_segments(spec, kind, rank, world_size)
        total_rows = sum(stop - start for start, stop in segments)
        raw = torch.empty(
            (total_rows, spec.row_bytes), dtype=torch.uint8, device=self.device
        )
        destination_offset = 0
        for start, stop in segments:
            size = (stop - start) * spec.row_bytes
            self._copy_contiguous(
                raw,
                destination_offset,
                spec.data_offset + start * spec.row_bytes,
                size,
            )
            destination_offset += size
        return Q4MatrixShard(
            raw=raw,
            out_features=total_rows,
            in_features=spec.in_features,
            source_name=spec.name,
            kind=kind,
            rank=rank,
        )

    def read_input_shard(
        self,
        spec: Q4MatrixSpec,
        kind: Literal["out_proj", "fc2"],
        rank: int,
        world_size: int,
    ) -> Q4MatrixShard:
        if spec.in_features % world_size:
            raise ValueError(f"{spec.name} input width does not divide TP={world_size}")
        local_in = spec.in_features // world_size
        if local_in % Q4_BLOCK_ELEMENTS:
            raise ValueError(
                f"{spec.name} local input width {local_in} splits a Q4_0 block"
            )
        local_row_bytes = local_in // Q4_BLOCK_ELEMENTS * Q4_BLOCK_BYTES
        byte_start = rank * local_row_bytes
        byte_stop = byte_start + local_row_bytes
        raw = torch.empty(
            (spec.out_features, local_row_bytes), dtype=torch.uint8, device=self.device
        )

        rows_per_chunk = max(1, self.staging.numel() // spec.row_bytes)
        # The selected half is materialized as a bounded contiguous CPU tensor;
        # its maximum size is <= the ordinary full-row staging allocation.
        selected = torch.empty(
            (rows_per_chunk, local_row_bytes), dtype=torch.uint8, device="cpu"
        )
        for row_start in range(0, spec.out_features, rows_per_chunk):
            rows = min(rows_per_chunk, spec.out_features - row_start)
            count = rows * spec.row_bytes
            full_rows = self._read_cpu(
                spec.data_offset + row_start * spec.row_bytes, count
            ).view(rows, spec.row_bytes)
            selected[:rows].copy_(full_rows[:, byte_start:byte_stop])
            raw[row_start : row_start + rows].copy_(selected[:rows])
            self._drop_cache(
                spec.data_offset + row_start * spec.row_bytes,
                count,
            )
        return Q4MatrixShard(
            raw=raw,
            out_features=spec.out_features,
            in_features=local_in,
            source_name=spec.name,
            kind=kind,
            rank=rank,
        )

    def read_tp_shard(
        self,
        spec: Q4MatrixSpec,
        kind: ShardKind,
        rank: int,
        world_size: int,
    ) -> Q4MatrixShard:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        if kind in {"qkv", "fc1"}:
            return self.read_output_shard(spec, kind, rank, world_size)
        return self.read_input_shard(spec, kind, rank, world_size)


def dequantize_q4_0(
    matrix: Q4MatrixShard,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize standard GGML Q4_0 bytes on their current device."""

    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported Q4_0 output dtype: {dtype}")
    out_features, in_features = matrix.shape
    if in_features % Q4_BLOCK_ELEMENTS:
        raise ValueError(f"Q4_0 matrix width is not block aligned: {matrix.shape}")
    expected = out_features * in_features // Q4_BLOCK_ELEMENTS * Q4_BLOCK_BYTES
    if matrix.raw.dtype != torch.uint8 or matrix.raw.numel() != expected:
        raise ValueError(
            f"Q4_0 raw storage mismatch for {matrix.source_name}: "
            f"got {matrix.raw.dtype}/{matrix.raw.numel()}, expected uint8/{expected}"
        )

    if dtype == torch.float16:
        try:
            from . import h3_v100_q4_ops
        except ImportError:
            try:
                import h3_v100_q4_ops
            except ImportError:
                h3_v100_q4_ops = None
        if h3_v100_q4_ops is not None:
            def eager(_matrix):
                return _dequantize_q4_0_eager(_matrix)

            return h3_v100_q4_ops.dequantize_q4_0_with_fallback(matrix, eager)

    return _dequantize_q4_0_eager(matrix, dtype)


def _dequantize_q4_0_eager(
    matrix: Q4MatrixShard,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference Q4_0 implementation used by the production fallback."""

    out_features, in_features = matrix.shape
    blocks = matrix.raw.reshape(-1, Q4_BLOCK_BYTES)
    scales = blocks[:, :2].contiguous().view(torch.float16).to(dtype)
    quants = blocks[:, 2:]
    device = torch.device(matrix.raw.device)
    shifts = _Q4_NIBBLE_SHIFTS.get(device)
    if shifts is None:
        shifts = torch.tensor([0, 4], dtype=torch.uint8, device=device).reshape(
            1, 1, 2, 1
        )
        _Q4_NIBBLE_SHIFTS[device] = shifts
    quants = quants.reshape(-1, 1, 1, Q4_BLOCK_ELEMENTS // 2)
    quants = ((quants >> shifts) & 0x0F).reshape(-1, Q4_BLOCK_ELEMENTS)
    quants = quants.to(torch.int8).sub_(8)
    values = scales * quants
    return values.reshape(out_features, in_features)


__all__ = [
    "DEFAULT_STAGING_BYTES",
    "Q4DiskReader",
    "Q4MatrixShard",
    "Q4MatrixSpec",
    "dequantize_q4_0",
    "inspect_q4_matrices",
    "output_row_segments",
]
