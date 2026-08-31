"""Standalone layer/pipeline parallel backend for the Qwen32B Q2 encoder.

This module is intentionally separate from :mod:`h3_qwen32_q2_tp`.  The
shared H3 runtime imports it lazily only after layer-MP is selected, so node
discovery does not create CUDA state or change an active output-row TP route.
It provides the ``qwen_forward``-style backend used by that switch while the
same shared runtime continues to own the later H3 DiT TP service.

The checkpoint remains header-only and direct-owner: each language layer's
compressed tensors are read from the bounded GGUF reader onto the GPU that owns
that layer.  A request crosses the two devices once at the selected layer
boundary.  No NCCL process group, rank-1 worker, or payload mmap is created by
this module.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # package import used by ComfyUI
    from . import h3_qwen32_q2_tp as qwen
except ImportError:  # pragma: no cover - source-tree/contract-test import
    import h3_qwen32_q2_tp as qwen  # type: ignore[no-redef]


QWEN32_MP_MODE = "mp"
QWEN32_TP_MODE = "tp"
QWEN32_MODE_ENV = "H3_QWEN32_Q2_MODE"
DEFAULT_MP_STAGING_MIB = qwen.DEFAULT_STAGING_MIB
DEFAULT_MP_DTYPE = torch.float32
DEFAULT_MP_SAFETY_FRACTION = 0.86
MIB = qwen.MIB
QWEN32_MP_PREFETCH_ENV = "H3_QWEN32_MP_PREFETCH"
QWEN32_MP_PREFETCH_ALIAS_ENV = "H3_QWEN32_PREFETCH"
QWEN32_MP_PREFETCH_CUDA_STREAM_ENV = "H3_QWEN32_MP_PREFETCH_CUDA_STREAM"
DEFAULT_MP_PREFETCH_MAX_MIB = 256
QWEN32_FINITE_TRACE_ENV = "H3_QWEN32_FINITE_TRACE"


def _finite_trace_enabled() -> bool:
    """Return whether expensive per-op numerical diagnostics are enabled.

    The checks synchronize CUDA streams, so they are deliberately opt-in and
    only used while investigating a failed conditioning request.
    """

    value = os.environ.get(QWEN32_FINITE_TRACE_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on", "enable", "enabled"}


def _trace_tensor_finite(label: str, value: Any) -> bool:
    """Log a compact finite/range report for one Qwen intermediate."""

    if not torch.is_tensor(value):
        logging.error("[H3 Qwen finite] %s is %s, expected tensor", label, type(value).__name__)
        return False
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    total = int(value.numel())
    if finite_count:
        finite_values = value.masked_select(finite)
        minimum = float(finite_values.min().item())
        maximum = float(finite_values.max().item())
        del finite_values
    else:
        minimum = maximum = float("nan")
    ok = finite_count == total
    logging.error(
        "[H3 Qwen finite] %s: finite=%s nonfinite=%d/%d min=%+.6e max=%+.6e "
        "shape=%s dtype=%s device=%s",
        label,
        ok,
        total - finite_count,
        total,
        minimum,
        maximum,
        tuple(value.shape),
        value.dtype,
        value.device,
    )
    return ok


def _dtype_element_size(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _canonical_device(value: torch.device | str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        # An explicit index makes reports and owner comparisons stable even
        # when the caller changes the current CUDA device between requests.
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def _env_bool(name: str, default: bool = False, *, alias: str | None = None) -> bool:
    """Read a strict boolean environment setting.

    The MP prefetch path is deliberately opt-in.  Accepting a short alias is
    useful for shell experiments while keeping one canonical name in reports
    and documentation.
    """

    value = os.environ.get(name)
    if value is None and alias:
        value = os.environ.get(alias)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    raise ValueError(f"{name} must be 0/1, got {value!r}")


def _prefetch_from_env(default: bool = False) -> bool:
    return _env_bool(
        QWEN32_MP_PREFETCH_ENV,
        default,
        alias=QWEN32_MP_PREFETCH_ALIAS_ENV,
    )


def _prefetch_cuda_stream_from_env(default: bool = False) -> bool:
    """Whether the experimental reader may allocate on an auxiliary stream.

    V100 + PyTorch's caching allocator has shown intermittent use-after-reuse
    when a layer allocated on a worker stream is handed to the compute stream,
    even with an event dependency and ``record_stream``.  The safe default is
    synchronous CUDA copies (the SSD read still runs in the worker); an
    auxiliary stream remains an explicit opt-in for future driver/allocator
    combinations that have passed the numerical gate.
    """

    return _env_bool(QWEN32_MP_PREFETCH_CUDA_STREAM_ENV, default)


def normalize_mp_devices(
    devices: Sequence[torch.device | str] | None = None,
    *,
    require_cuda: bool = False,
    check_peer_access: bool = True,
) -> tuple[torch.device, torch.device]:
    """Validate and canonicalize the two layer-MP devices.

    Planning is also useful on CPU-only machines, so callers may pass two
    ``cpu`` entries for a dry run.  A real CUDA execution requires distinct
    devices and peer access; the latter check is skipped when CUDA is not
    available so header-only audits remain side-effect free.
    """

    if isinstance(devices, str):
        devices = tuple(item.strip() for item in devices.split(",") if item.strip())
    if devices is None:
        configured = os.environ.get("H3_QWEN32_MP_DEVICES")
        if configured is None and not torch.cuda.is_available():
            devices = ("cpu", "cpu")
        else:
            raw = configured or "cuda:0,cuda:1"
            devices = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(devices) != 2:
        raise ValueError("Qwen32 layer-MP requires exactly two devices")
    first, second = (_canonical_device(item) for item in devices)
    if require_cuda and (first.type != "cuda" or second.type != "cuda"):
        raise RuntimeError("Qwen32 layer-MP requires two CUDA devices")
    if first.type == "cuda" or second.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for Qwen32 layer-MP")
        if first.type != "cuda" or second.type != "cuda":
            raise ValueError("Qwen32 MP devices must both be CUDA or both be CPU")
        if first.index == second.index:
            raise ValueError("Qwen32 MP devices must be different")
        if first.index is None or second.index is None:
            raise ValueError("Qwen32 MP CUDA devices need explicit indices")
        if max(first.index, second.index) >= torch.cuda.device_count():
            raise ValueError(
                f"Qwen32 MP device index outside visible range: {first}, {second}"
            )
        if check_peer_access:
            try:
                if not torch.cuda.can_device_access_peer(first.index, second.index):
                    raise RuntimeError(f"no CUDA peer access {first} -> {second}")
                if not torch.cuda.can_device_access_peer(second.index, first.index):
                    raise RuntimeError(f"no CUDA peer access {second} -> {first}")
            except RuntimeError:
                raise
            except Exception as exc:  # pragma: no cover - driver-specific
                raise RuntimeError("unable to verify CUDA peer access") from exc
    elif first != second:
        # CPU planning can use two logical labels (for example cpu and meta),
        # but actual CPU execution on distinct devices is not meaningful.
        # Keep the validation permissive for dry-run planners.
        pass
    return first, second


def _norm_role(name: str) -> str | None:
    for role in qwen.NORM_ROLES:
        if name.endswith(f".{role}.weight"):
            return role
    return None


@dataclass(frozen=True)
class Qwen32MPLayerCost:
    """Header-only memory estimate for one complete language layer."""

    layer: int
    compressed_bytes: int
    dense_bytes: int
    norm_bytes: int
    largest_dense_bytes: int

    @property
    def evict_peak_bytes(self) -> int:
        """Compressed layer plus the largest one-at-a-time dense workspace."""

        return int(self.compressed_bytes + self.norm_bytes + self.largest_dense_bytes)

    @property
    def resident_bytes(self) -> int:
        """Resident model bytes when compressed tensors are kept on a GPU."""

        return int(self.compressed_bytes + self.norm_bytes)

    @property
    def cache_dequantized_bytes(self) -> int:
        return int(self.resident_bytes + self.dense_bytes)

    def as_dict(self) -> dict[str, int]:
        return {
            "layer": int(self.layer),
            "compressed_bytes": int(self.compressed_bytes),
            "dense_bytes": int(self.dense_bytes),
            "norm_bytes": int(self.norm_bytes),
            "largest_dense_bytes": int(self.largest_dense_bytes),
            "evict_peak_bytes": int(self.evict_peak_bytes),
            "resident_bytes": int(self.resident_bytes),
            "cache_dequantized_bytes": int(self.cache_dequantized_bytes),
        }


@dataclass(frozen=True)
class Qwen32MPSplitPlan:
    """A contiguous two-device layer assignment and its memory estimate."""

    devices: tuple[str, str]
    layer_count: int
    split: int
    first_layers: tuple[int, ...]
    second_layers: tuple[int, ...]
    residency: str
    cache_dequantized: bool
    dtype: str
    strategy: str
    layer_costs: tuple[Qwen32MPLayerCost, ...]
    baseline_bytes: tuple[int, int]
    capacity_bytes: tuple[int | None, int | None]
    assigned_bytes: tuple[int, int]
    estimated_peak_bytes: tuple[int, int]
    fits_capacity: tuple[bool | None, bool | None]

    def owner_index(self, layer: int) -> int:
        layer = int(layer)
        if layer < 0 or layer >= self.layer_count:
            raise IndexError(f"layer {layer} outside [0, {self.layer_count})")
        return 0 if layer < self.split else 1

    def owner(self, layer: int) -> str:
        return self.devices[self.owner_index(layer)]

    @property
    def balanced_ratio(self) -> float:
        low = min(self.estimated_peak_bytes)
        high = max(self.estimated_peak_bytes)
        return float(high / low) if low else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "devices": list(self.devices),
            "layer_count": int(self.layer_count),
            "split": int(self.split),
            "first_layers": list(self.first_layers),
            "second_layers": list(self.second_layers),
            "residency": self.residency,
            "cache_dequantized": bool(self.cache_dequantized),
            "dtype": self.dtype,
            "strategy": self.strategy,
            "baseline_bytes": list(self.baseline_bytes),
            "capacity_bytes": list(self.capacity_bytes),
            "assigned_bytes": list(self.assigned_bytes),
            "estimated_peak_bytes": list(self.estimated_peak_bytes),
            "fits_capacity": list(self.fits_capacity),
            "balanced_ratio": self.balanced_ratio,
            "layer_costs": [item.as_dict() for item in self.layer_costs],
        }


@dataclass
class _PrefetchedLayer:
    """A bounded, ready-to-consume layer loaded by the I/O worker.

    Only compressed matrix bytes and the four small norm vectors are retained.
    Dequantised matrices are intentionally materialised on the caller's
    compute path, so the worker cannot create a second multi-gigabyte dense
    model copy while it overlaps disk I/O with the previous layer.
    """

    layer: int
    device: torch.device
    raw: dict[str, torch.Tensor]
    norms: dict[str, torch.Tensor]
    compressed_bytes: int
    elapsed_seconds: float
    ready_event: torch.cuda.Event | None = None

    def wait_on_current_stream(self) -> None:
        """Order the consumer stream after asynchronous H2D/dequant work."""

        if self.ready_event is None or self.device.type != "cuda":
            return
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device).wait_event(self.ready_event)

    def close(self) -> None:
        if self.ready_event is not None:
            # ``clear``/``close`` may run without a consumer.  Do not recycle
            # storage whose low-priority stream is still reading/dequantising.
            self.ready_event.synchronize()
            self.ready_event = None
        values = tuple(self.raw.values()) + tuple(self.norms.values())
        self.raw.clear()
        self.norms.clear()
        for value in values:
            del value


def _as_int_pair(
    value: Sequence[int] | Mapping[Any, int] | None,
    devices: tuple[torch.device, torch.device],
    *,
    default: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, Mapping):
        result = []
        for device in devices:
            result.append(int(value.get(str(device), value.get(device, 0))))
        return result[0], result[1]
    if len(value) != 2:
        raise ValueError("memory byte pairs must contain two values")
    return int(value[0]), int(value[1])


def _cuda_memory_pair(
    devices: tuple[torch.device, torch.device],
    safety_fraction: float,
) -> tuple[tuple[int, int], tuple[int | None, int | None]]:
    baseline = [0, 0]
    capacity: list[int | None] = [None, None]
    if not torch.cuda.is_available():
        return (0, 0), (None, None)
    for index, device in enumerate(devices):
        if device.type != "cuda":
            continue
        try:
            free, total = torch.cuda.mem_get_info(device)
        except (RuntimeError, ValueError):  # pragma: no cover - driver edge
            continue
        baseline[index] = max(0, int(total) - int(free))
        capacity[index] = max(0, int(int(total) * safety_fraction))
    return (baseline[0], baseline[1]), (capacity[0], capacity[1])


def _layer_cost(
    layer: int,
    roles: Mapping[str, qwen.TensorSpec],
    norm_specs: Iterable[qwen.TensorSpec],
    dtype: torch.dtype,
) -> Qwen32MPLayerCost:
    matrices = [roles[role] for role in qwen.MATRIX_ROLES if role in roles]
    if len(matrices) != len(qwen.MATRIX_ROLES):
        missing = sorted(set(qwen.MATRIX_ROLES).difference(roles))
        raise ValueError(f"layer {layer} is missing matrices: {missing}")
    dense_sizes = [math.prod(int(value) for value in spec.shape) * _dtype_element_size(dtype) for spec in matrices]
    return Qwen32MPLayerCost(
        layer=int(layer),
        compressed_bytes=sum(int(spec.n_bytes) for spec in matrices),
        dense_bytes=sum(dense_sizes),
        norm_bytes=sum(int(spec.n_bytes) for spec in norm_specs),
        largest_dense_bytes=max(dense_sizes),
    )


def _resolve_split_request(split: int | str | None) -> tuple[int | None, str]:
    requested = split
    if requested is None:
        requested = os.environ.get("H3_QWEN32_MP_SPLIT", "auto")
    if isinstance(requested, str):
        value = requested.strip().lower()
        if value in {"", "auto", "balanced", "default"}:
            return None, "auto"
        try:
            requested = int(value)
        except ValueError as exc:
            raise ValueError(
                "H3_QWEN32_MP_SPLIT must be auto or an integer layer boundary"
            ) from exc
    return int(requested), "explicit"


def resolve_qwen32_mode(value: str | None = None, *, default: str = QWEN32_MP_MODE) -> str:
    """Normalize the future backend switch without changing the live TP path."""

    requested = os.environ.get(QWEN32_MODE_ENV, default) if value is None else value
    normalized = str(requested).strip().lower().replace("-", "_")
    if normalized in {"mp", "layer_mp", "layer_parallel", "layerpipeline"}:
        return QWEN32_MP_MODE
    if normalized in {"tp", "output_row_tp", "outputrow_tp"}:
        return QWEN32_TP_MODE
    raise ValueError(
        f"{QWEN32_MODE_ENV} must be mp or tp, got {requested!r}"
    )


def plan_layer_split(
    layout_or_path: qwen.GGUFLayout | os.PathLike[str] | str,
    *,
    devices: Sequence[torch.device | str] | None = None,
    split: int | str | None = None,
    residency: str = "evict",
    keep_layers: int | Sequence[int] = 0,
    cache_dequantized: bool = False,
    dtype: torch.dtype = DEFAULT_MP_DTYPE,
    baseline_bytes: Sequence[int] | Mapping[Any, int] | None = None,
    capacity_bytes: Sequence[int] | Mapping[Any, int] | None = None,
    safety_fraction: float = DEFAULT_MP_SAFETY_FRACTION,
) -> Qwen32MPSplitPlan:
    """Choose a contiguous layer boundary using header geometry and VRAM.

    ``auto`` minimizes the larger normalized load after adding the current
    per-device baseline.  For ``evict`` the assigned load is compressed traffic
    (the steady resident peak is one layer); for ``partial``/``full`` it is the
    compressed residency implied by the policy.  Explicit ``split`` remains a
    hard override, which is useful when GPU0 also owns a DiT shard.
    """

    if devices is None and not torch.cuda.is_available():
        # Keep the header-only planner usable in CI/CPU containers.  Runtime
        # construction still validates the caller's explicit CUDA devices.
        devices = ("cpu", "cpu")
    layout = (
        layout_or_path
        if isinstance(layout_or_path, qwen.GGUFLayout)
        else qwen.inspect_gguf(layout_or_path)
    )
    pair = normalize_mp_devices(devices, check_peer_access=False)
    residency = str(residency).strip().lower()
    if residency not in {"evict", "partial", "full"}:
        raise ValueError("Qwen32 MP residency must be evict, partial, or full")
    if not 0.0 < float(safety_fraction) <= 1.0:
        raise ValueError("safety_fraction must be in (0, 1]")
    matrix_specs = qwen.language_matrix_specs(layout)
    layers = tuple(sorted(int(layer) for layer in matrix_specs))
    if not layers:
        raise ValueError("GGUF contains no Qwen language matrix layers")
    if len(layers) < 2:
        raise ValueError("Qwen32 layer-MP requires at least two language layers")
    expected = tuple(range(len(layers)))
    if layers != expected:
        raise ValueError(f"Qwen language layers must be contiguous 0..N-1, got {layers[:8]}...")
    costs = tuple(
        _layer_cost(
            layer,
            matrix_specs[layer],
            (
                spec
                for spec in layout.language_layers.get(layer, ())
                if _norm_role(spec.name) is not None
            ),
            dtype,
        )
        for layer in layers
    )
    layer_count = len(costs)
    requested_split, strategy = _resolve_split_request(split)
    if requested_split is not None and not 1 <= requested_split < layer_count:
        raise ValueError(
            f"Qwen32 MP split must be between 1 and {layer_count - 1}, got {requested_split}"
        )

    detected_baseline, detected_capacity = _cuda_memory_pair(pair, float(safety_fraction))
    baseline = _as_int_pair(baseline_bytes, pair, default=detected_baseline)
    capacity_values = _as_int_pair(capacity_bytes, pair, default=(0, 0)) if capacity_bytes is not None else detected_capacity
    # ``partial`` keeps only explicitly requested layers; an integer follows
    # the existing TP route convention and means layers [0, N).
    if isinstance(keep_layers, int):
        keep_set = set(range(max(0, int(keep_layers))))
    else:
        keep_set = {int(item) for item in keep_layers}
    if residency == "full":
        keep_set = set(layers)
    unknown_keep = sorted(item for item in keep_set if item not in layers)
    if unknown_keep:
        raise ValueError(f"Qwen32 MP keep_layers contains unknown IDs: {unknown_keep[:8]}")

    def unit_cost(cost: Qwen32MPLayerCost) -> int:
        if cache_dequantized:
            return cost.cache_dequantized_bytes
        # A partial route still benefits from a balanced potential assignment:
        # layers not retained now will be loaded on demand later.
        return cost.resident_bytes

    def peak_cost(indices: Sequence[int]) -> int:
        selected = [costs[index] for index in indices]
        if not selected:
            return 0
        if residency == "evict":
            return max(
                item.cache_dequantized_bytes if cache_dequantized else item.evict_peak_bytes
                for item in selected
            )
        if any(item.layer in keep_set for item in selected):
            return sum(
                (item.cache_dequantized_bytes if cache_dequantized else item.resident_bytes)
                for item in selected
                if item.layer in keep_set
            )
        return 0

    def score(boundary: int) -> tuple[float, float, int, int]:
        first_indices = tuple(range(boundary))
        second_indices = tuple(range(boundary, layer_count))
        first_load = sum(unit_cost(costs[index]) for index in first_indices)
        second_load = sum(unit_cost(costs[index]) for index in second_indices)
        totals = (baseline[0] + first_load, baseline[1] + second_load)
        normalized = []
        for total, cap in zip(totals, capacity_values):
            normalized.append(float(total) / float(cap) if cap else float(total))
        imbalance = abs(float(totals[0]) - float(totals[1]))
        return max(normalized), imbalance, abs(boundary - layer_count / 2), boundary

    if requested_split is None:
        split_value = min(range(1, layer_count), key=score)
    else:
        split_value = requested_split

    first_indices = tuple(range(split_value))
    second_indices = tuple(range(split_value, layer_count))
    assigned = (
        baseline[0] + sum(unit_cost(costs[index]) for index in first_indices),
        baseline[1] + sum(unit_cost(costs[index]) for index in second_indices),
    )
    # For full/partial residency, include retained bytes plus a conservative
    # one-layer workspace.  The latter matters during the first pass: a layer
    # is dequantized while its neighbouring compressed layers are already
    # resident.  For evict, only the largest transient layer is resident.
    if residency == "evict":
        estimated = (
            baseline[0] + peak_cost(first_indices),
            baseline[1] + peak_cost(second_indices),
        )
    else:
        def retained(indices: Sequence[int]) -> int:
            return sum(
                (costs[index].cache_dequantized_bytes if cache_dequantized else costs[index].resident_bytes)
                for index in indices
                if costs[index].layer in keep_set
            )

        def workspace(indices: Sequence[int]) -> int:
            if not indices:
                return 0
            return max(
                item.evict_peak_bytes for item in (costs[index] for index in indices)
            )

        estimated = (
            baseline[0] + retained(first_indices) + workspace(first_indices),
            baseline[1] + retained(second_indices) + workspace(second_indices),
        )
    fits: tuple[bool | None, bool | None] = tuple(
        None if cap is None else bool(value <= cap)
        for value, cap in zip(estimated, capacity_values)
    )  # type: ignore[assignment]
    if requested_split is None and strategy == "auto":
        strategy = "auto-memory-weighted" if any(capacity_values) else "auto-balanced"
    return Qwen32MPSplitPlan(
        devices=(str(pair[0]), str(pair[1])),
        layer_count=layer_count,
        split=int(split_value),
        first_layers=tuple(layers[:split_value]),
        second_layers=tuple(layers[split_value:]),
        residency=residency,
        cache_dequantized=bool(cache_dequantized),
        dtype=str(dtype).removeprefix("torch."),
        strategy=strategy,
        layer_costs=costs,
        baseline_bytes=baseline,
        capacity_bytes=capacity_values,
        assigned_bytes=assigned,
        estimated_peak_bytes=estimated,
        fits_capacity=fits,
    )


def _full_descriptor(spec: qwen.TensorSpec) -> qwen.TensorShardDescriptor:
    descriptor = qwen.build_output_row_shards(spec, world_size=1, rank=0)
    if not isinstance(descriptor, qwen.TensorShardDescriptor):  # pragma: no cover
        raise TypeError("world_size=1 did not return one descriptor")
    return descriptor


def _move_tree(value: Any, device: torch.device) -> Any:
    """Move request metadata while preserving tuple/list/mapping structure."""

    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


class Qwen32Q2MPLinear(qwen.Qwen32Q2OutputLinear):
    """Full output-row linear used by one layer-MP owner.

    The parent implementation already provides bounded direct reads and
    lazy dequantisation; a world-size-one descriptor means no input columns or
    output rows are sliced.
    """

    def attach_raw(self, raw: torch.Tensor) -> None:
        """Attach compressed bytes produced by the asynchronous prefetcher."""

        if not torch.is_tensor(raw) or raw.dtype != torch.uint8:
            raise TypeError("prefetched Qwen matrix must be a uint8 tensor")
        if raw.device != self.device:
            raise ValueError(
                f"prefetched Qwen matrix is on {raw.device}, expected {self.device}"
            )
        if int(raw.numel()) != int(self.descriptor.n_bytes):
            raise ValueError(
                f"prefetched {self.descriptor.tensor_name} has {raw.numel()} bytes; "
                f"expected {self.descriptor.n_bytes}"
            )
        if self.raw is not None:
            raise RuntimeError(f"{self.descriptor.tensor_name} is already loaded")
        self.raw = raw.contiguous()

    def ensure_raw(self) -> None:
        """Load compressed bytes without paying dequantisation cost yet."""

        self.load()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        try:
            return super().forward(x)
        finally:
            self.forward_seconds = getattr(self, "forward_seconds", 0.0) + (
                time.perf_counter() - started
            )

    def stats(self) -> dict[str, Any]:
        result = super().stats()
        result["forward_seconds"] = float(getattr(self, "forward_seconds", 0.0))
        return result


class Qwen32Q2MPLayerBlock(nn.Module):
    """A complete Qwen32 language block resident on one device."""

    def __init__(
        self,
        layer: int,
        matrices: Mapping[str, Qwen32Q2MPLinear],
        norms: Mapping[str, torch.Tensor],
        *,
        device: torch.device | str,
        dtype: torch.dtype = DEFAULT_MP_DTYPE,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        missing = set(qwen.MATRIX_ROLES).difference(matrices)
        if missing:
            raise ValueError(f"layer {layer} is missing matrices: {sorted(missing)}")
        self.layer = int(layer)
        self.device = _canonical_device(device)
        self.compute_dtype = dtype
        self.eps = float(eps)
        self.matrices = nn.ModuleDict(dict(matrices))
        self.norms: dict[str, torch.Tensor] = {
            key: value.to(device=self.device, dtype=dtype).detach()
            for key, value in norms.items()
            if key in set(qwen.NORM_ROLES)
        }
        missing_norms = set(qwen.NORM_ROLES).difference(self.norms)
        if missing_norms:
            raise ValueError(f"layer {layer} is missing norms: {sorted(missing_norms)}")
        self.forward_count = 0
        self.forward_seconds = 0.0
        self.raw_load_seconds = 0.0
        self.last_timing: dict[str, float] = {}
        self.last_stats: dict[str, Any] = {}
        self._prefetch_ready_event: torch.cuda.Event | None = None
        self._prefetch_consumer_stream: torch.cuda.Stream | None = None

    @property
    def resident_bytes(self) -> int:
        matrix_bytes = sum(item.resident_bytes for item in self.matrices.values())
        norm_bytes = sum(int(value.numel() * value.element_size()) for value in self.norms.values())
        return int(matrix_bytes + norm_bytes)

    @property
    def compressed_bytes(self) -> int:
        return int(sum(item.raw.numel() for item in self.matrices.values() if item.raw is not None))

    def ensure_raw(self) -> None:
        """Materialise all seven compressed matrices before compute starts.

        Loading the current layer as one bounded burst gives the prefetch
        worker an uncontended reader window while this layer executes.  It
        does not dequantise or retain dense matrices.
        """

        started = time.perf_counter()
        for matrix in self.matrices.values():
            matrix.ensure_raw()
        self.raw_load_seconds += time.perf_counter() - started

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: Any = None,
    ) -> torch.Tensor:
        started = time.perf_counter()
        # A prefetch worker copies compressed matrices and dequantises the
        # small RMSNorm vectors on its own CUDA stream.  ``consume()`` also
        # queues a wait, but the compute stream is not guaranteed to remain
        # the same between layer hand-offs (ComfyUI and attention backends
        # may change the current stream).  Every layer must therefore enforce
        # its dependency at the point where the payload is first consumed.
        # ``wait_event`` is stream-local and does not synchronize the device;
        # falling back to a device-wide sync here would erase the overlap that
        # prefetch is intended to provide.
        ready_event = self._prefetch_ready_event
        if (
            ready_event is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            with torch.cuda.device(self.device):
                compute_stream = torch.cuda.current_stream(self.device)
                compute_stream.wait_event(ready_event)
                self._prefetch_consumer_stream = compute_stream
                # The payload was allocated on the worker's copy stream but
                # is consumed (and later released) on this compute stream.
                # Waiting on the event orders the kernels; record_stream is
                # the separate allocator-lifetime requirement.  Without it,
                # evict-mode cleanup can return a raw layer buffer to the
                # caching allocator while a queued dequant/GEMM still reads
                # it.  The resulting use-after-reuse is timing-dependent and
                # showed up as NaN/Inf at larger sequence lengths.  Register
                # every cross-stream tensor before the first matrix use.
                for matrix in self.matrices.values():
                    if matrix.raw is not None:
                        matrix.raw.record_stream(compute_stream)
                for value in self.norms.values():
                    value.record_stream(compute_stream)
        trace = _finite_trace_enabled()
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.input", x)
        before = {
            "load": sum(float(item.load_seconds) for item in self.matrices.values()),
            "dequant": sum(float(item.dequant_seconds) for item in self.matrices.values()),
            "gemm": sum(float(getattr(item, "forward_seconds", 0.0)) for item in self.matrices.values()),
        }
        if x.device != self.device:
            x = x.to(self.device)
        if x.is_floating_point() and x.dtype != self.compute_dtype:
            x = x.to(dtype=self.compute_dtype)
        if x.ndim != 3 or x.shape[-1] != qwen.QWEN32_HIDDEN_SIZE:
            raise ValueError(
                f"Qwen32 MP block expects [B,S,{qwen.QWEN32_HIDDEN_SIZE}], got {tuple(x.shape)}"
            )
        b, sequence, _ = x.shape
        h = qwen._rms_norm(x, self.norms["input_layernorm"], self.eps)
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.input_layernorm", h)
        q_value = self.matrices["q_proj"](h)
        k_value = self.matrices["k_proj"](h)
        v_value = self.matrices["v_proj"](h)
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.q_proj", q_value)
            _trace_tensor_finite(f"layer{self.layer}.k_proj", k_value)
            _trace_tensor_finite(f"layer{self.layer}.v_proj", v_value)
        q_value = q_value.view(b, sequence, qwen.QWEN32_NUM_HEADS, qwen.QWEN32_HEAD_DIM).transpose(1, 2)
        k_value = k_value.view(b, sequence, qwen.QWEN32_NUM_KV_HEADS, qwen.QWEN32_HEAD_DIM).transpose(1, 2)
        v_value = v_value.view(b, sequence, qwen.QWEN32_NUM_KV_HEADS, qwen.QWEN32_HEAD_DIM).transpose(1, 2)
        q_value = qwen._rms_norm(q_value, self.norms["q_norm"], self.eps)
        k_value = qwen._rms_norm(k_value, self.norms["k_norm"], self.eps)
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.q_norm", q_value)
            _trace_tensor_finite(f"layer{self.layer}.k_norm", k_value)
        try:
            from comfy.text_encoders.llama import apply_rope
        except (ImportError, ModuleNotFoundError):  # pragma: no cover - CPU contract path
            apply_rope = None
        if freqs_cis is not None and apply_rope is not None:
            q_value, k_value = apply_rope(q_value, k_value, freqs_cis)
        elif freqs_cis is not None:
            q_value, k_value = qwen._apply_rope_fallback(q_value, k_value, freqs_cis)
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.q_rope", q_value)
            _trace_tensor_finite(f"layer{self.layer}.k_rope", k_value)
        attention = qwen._attention(q_value, k_value, v_value, attention_mask)
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.attention", attention)
        if attention.ndim != 4:
            raise RuntimeError(
                f"Qwen attention backend returned {tuple(attention.shape)}; expected [B,H,S,D]"
            )
        attention = attention.transpose(1, 2).reshape(b, sequence, qwen.QWEN32_Q_DIM)
        residual = x
        o_value = self.matrices["o_proj"](attention)
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.o_proj", o_value)
        x = residual + o_value
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.attention_residual", x)

        residual = x
        h = qwen._rms_norm(x, self.norms["post_attention_layernorm"], self.eps)
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.post_attention_layernorm", h)
        gate = self.matrices["gate_proj"](h)
        up = self.matrices["up_proj"](h)
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.gate_proj", gate)
            _trace_tensor_finite(f"layer{self.layer}.up_proj", up)
        mlp = F.silu(gate) * up
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.mlp", mlp)
        down = self.matrices["down_proj"](mlp)
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.down_proj", down)
        x = residual + down
        if trace:
            _trace_tensor_finite(f"layer{self.layer}.output", x)
        self.forward_count += 1
        self.forward_seconds += time.perf_counter() - started
        after = {
            "load": sum(float(item.load_seconds) for item in self.matrices.values()),
            "dequant": sum(float(item.dequant_seconds) for item in self.matrices.values()),
            "gemm": sum(float(getattr(item, "forward_seconds", 0.0)) for item in self.matrices.values()),
        }
        self.last_timing = {
            "matrix_load_seconds": max(0.0, after["load"] - before["load"]),
            "matrix_dequant_seconds": max(0.0, after["dequant"] - before["dequant"]),
            "matrix_gemm_seconds": max(0.0, after["gemm"] - before["gemm"]),
            "block_wall_seconds": time.perf_counter() - started,
        }
        self.last_stats = {
            "layer": self.layer,
            "forward_count": self.forward_count,
            "finite": bool(torch.isfinite(x).all().item()),
            "shape": list(x.shape),
            "device": str(self.device),
            "forward_seconds": self.forward_seconds,
            **self.last_timing,
        }
        return x

    def clear(self, *, release_cache: bool = True, collect: bool = True) -> None:
        """Release layer payload, optionally retaining allocator cache.

        The prefetch route skips collection/cache release between adjacent
        layers so it does not synchronize away the background read.  The
        enclosing backbone still empties both devices at request clear.
        """

        ready_event = self._prefetch_ready_event
        self._prefetch_ready_event = None
        consumer_stream = self._prefetch_consumer_stream
        self._prefetch_consumer_stream = None
        if ready_event is not None:
            ready_event.synchronize()
        if consumer_stream is not None:
            # ``record_stream`` prevents premature allocator reuse, but a
            # stream-local synchronization is still required before dropping
            # the last Python references: CUDA work queued by F.linear may
            # outlive the block call, and the next prefetch allocation can
            # otherwise race that read on V100's caching allocator.  This is
            # paid only by the opt-in prefetch route and still overlaps SSD
            # reads for the next layer with the current layer's compute.
            consumer_stream.synchronize()
        for matrix in self.matrices.values():
            matrix.clear()
        self.norms.clear()
        if collect:
            gc.collect()
        if release_cache and self.device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def stats(self) -> dict[str, Any]:
        matrix_stats = {key: value.stats() for key, value in self.matrices.items()}
        return {
            "layer": self.layer,
            "device": str(self.device),
            "resident_bytes": self.resident_bytes,
            "compressed_bytes": self.compressed_bytes,
            "forward_count": self.forward_count,
            "forward_seconds": self.forward_seconds,
            "raw_load_seconds": self.raw_load_seconds,
            "last_timing": dict(self.last_timing),
            "matrices": matrix_stats,
            "last": dict(self.last_stats),
        }


class _Qwen32MPLayerPrefetcher:
    """One-slot asynchronous compressed-layer reader.

    The worker is intentionally tiny and conservative: a single bounded
    future reads at most one complete layer, using a per-device CUDA copy
    stream when available.  The main forward thread consumes the future at
    the next layer boundary.  This overlaps SSD/read/decode of layer *N+1*
    with GEMMs for layer *N* without retaining dense weights or changing
    collective/activation ordering.
    """

    def __init__(
        self,
        backbone: "Qwen32Q2LayerMPBackbone",
        *,
        max_bytes: int = DEFAULT_MP_PREFETCH_MAX_MIB * MIB,
        use_cuda_stream: bool | None = None,
    ) -> None:
        self.backbone = backbone
        self.max_bytes = max(0, int(max_bytes))
        self.use_cuda_stream = (
            _prefetch_cuda_stream_from_env(False)
            if use_cuda_stream is None
            else bool(use_cuda_stream)
        )
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="h3-qwen-prefetch",
        )
        self._lock = threading.RLock()
        self._pending: dict[int, Future[_PrefetchedLayer]] = {}
        self._staged: dict[int, _PrefetchedLayer] = {}
        self._streams: dict[str, torch.cuda.Stream] = {}
        self.submitted = 0
        self.completed = 0
        self.consumed = 0
        self.hits = 0
        self.misses = 0
        self.skipped_capacity = 0
        self.fallbacks = 0
        self.errors = 0
        self.prefetch_bytes = 0
        self.prefetch_seconds = 0.0
        self.wait_seconds = 0.0
        self.max_staged_bytes = 0
        self.last_error: str | None = None
        self.closed = False

    def _stream_for(self, device: torch.device) -> torch.cuda.Stream | None:
        if (
            not self.use_cuda_stream
            or device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return None
        key = str(device)
        stream = self._streams.get(key)
        if stream is None:
            # Stream creation is kept in the worker thread.  Explicit device
            # context avoids accidentally allocating a stream on cuda:0 when
            # the next owner is cuda:1.
            with torch.cuda.device(device):
                try:
                    # Positive priorities are accepted by recent CUDA builds
                    # as a low-priority hint, but older V100 runtimes expose
                    # only priority 0.  Fall back without disabling prefetch.
                    stream = torch.cuda.Stream(device=device, priority=1)
                except (RuntimeError, ValueError):  # pragma: no cover - driver-specific
                    stream = torch.cuda.Stream(device=device, priority=0)
            self._streams[key] = stream
        return stream

    def _read_layer(self, layer: int) -> _PrefetchedLayer:
        started = time.perf_counter()
        device = self.backbone._owner(layer)
        stream = self._stream_for(device)
        raw: dict[str, torch.Tensor] = {}
        norms: dict[str, torch.Tensor] = {}
        ready_event: torch.cuda.Event | None = None
        try:
            context = torch.cuda.device(device) if device.type == "cuda" else None
            if context is None:
                # Keep one code path below while avoiding a null context
                # manager on CPU-only contract tests.
                from contextlib import nullcontext

                context = nullcontext()
            with context:
                roles = self.backbone.layer_specs.get(layer)
                if roles is None:
                    raise KeyError(f"Qwen language layer {layer} is absent")
                compressed = 0
                for role in qwen.MATRIX_ROLES:
                    spec = roles.get(role)
                    if spec is None:
                        raise ValueError(f"layer {layer} is missing matrix {role}")
                    compressed += int(spec.n_bytes)
                    raw[role] = self.backbone.reader.read_tensor(
                        spec,
                        device=device,
                        stream=stream,
                        non_blocking=stream is not None,
                    )
                # Norms are tiny but are included in the future so the main
                # thread never contends for the shared reader at consumption.
                for spec in self.backbone.layout.language_layers.get(layer, ()):
                    role = _norm_role(spec.name)
                    if role is None:
                        continue
                    compressed += int(spec.n_bytes)
                    raw_norm = self.backbone.reader.read_tensor(
                        spec,
                        device=device,
                        stream=stream,
                        non_blocking=stream is not None,
                    )
                    try:
                        if stream is not None:
                            with torch.cuda.stream(stream):
                                norms[role] = qwen.dequantize_ggml(
                                    raw_norm,
                                    spec.qtype,
                                    spec.shape,
                                    dtype=self.backbone.compute_dtype,
                                ).reshape(spec.shape)
                        else:
                            norms[role] = qwen.dequantize_ggml(
                                raw_norm,
                                spec.qtype,
                                spec.shape,
                                dtype=self.backbone.compute_dtype,
                            ).reshape(spec.shape)
                    finally:
                        del raw_norm
                expected = set(qwen.NORM_ROLES)
                if set(norms) != expected:
                    missing = sorted(expected.difference(norms))
                    raise ValueError(f"layer {layer} is missing norms: {missing}")
                if compressed > self.max_bytes:
                    raise MemoryError(
                        f"prefetch layer {layer} needs {compressed} bytes, "
                        f"cap is {self.max_bytes}"
                    )
                if stream is not None:
                    with torch.cuda.stream(stream):
                        ready_event = torch.cuda.Event()
                        ready_event.record(stream)
                elif device.type == "cuda" and torch.cuda.is_available():
                    # With synchronous copies the small norm dequantisation
                    # can still be queued on the worker thread's current
                    # stream.  Join that stream before publishing the layer;
                    # this is local to the worker and avoids exposing an
                    # unfinished norm tensor to the compute path.
                    torch.cuda.current_stream(device).synchronize()
            elapsed = time.perf_counter() - started
            return _PrefetchedLayer(
                layer=int(layer),
                device=device,
                raw=raw,
                norms=norms,
                compressed_bytes=int(compressed),
                elapsed_seconds=float(elapsed),
                ready_event=ready_event,
            )
        except BaseException:
            if ready_event is not None:
                ready_event.synchronize()
            for value in tuple(raw.values()) + tuple(norms.values()):
                del value
            raw.clear()
            norms.clear()
            gc.collect()
            if device.type == "cuda" and torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            raise

    def submit(self, layer: int) -> bool:
        layer = int(layer)
        with self._lock:
            if self.closed:
                return False
            if layer in self._pending or layer in self._staged:
                return True
            cost = self.backbone.plan.layer_costs[layer]
            if int(cost.compressed_bytes + cost.norm_bytes) > self.max_bytes:
                self.skipped_capacity += 1
                return False
            self._pending[layer] = self.executor.submit(self._read_layer, layer)
            self.submitted += 1
            return True

    def consume(self, layer: int) -> _PrefetchedLayer | None:
        layer = int(layer)
        with self._lock:
            future = self._pending.pop(layer, None)
            staged = self._staged.pop(layer, None)
        if staged is not None:
            self.hits += 1
            self.consumed += 1
            return staged
        if future is None:
            self.misses += 1
            return None
        wait_started = time.perf_counter()
        try:
            staged = future.result()
        except BaseException as exc:
            # A prefetch fault must not make an experimental switch take the
            # service down.  Drop the half-built layer and let load_layer()
            # perform the ordinary synchronous no-mmap read instead.
            self.errors += 1
            self.fallbacks += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            logging.warning(
                "[H3 Qwen MP] prefetch layer %d failed; falling back to sync load: %s",
                layer,
                self.last_error,
            )
            self.misses += 1
            self.wait_seconds += time.perf_counter() - wait_started
            return None
        self.wait_seconds += time.perf_counter() - wait_started
        self.completed += 1
        self.consumed += 1
        self.prefetch_bytes += int(staged.compressed_bytes)
        self.prefetch_seconds += float(staged.elapsed_seconds)
        self.hits += 1
        # Do not block the Python thread here: enqueue a dependency on the
        # caller's compute stream.  The first matrix/norm use then waits for
        # the worker's H2D/dequant sequence while subsequent CPU bookkeeping
        # remains synchronous and deterministic.
        staged.wait_on_current_stream()
        return staged

    def reset(self) -> None:
        """Cancel pending work and release staged tensors between requests."""

        with self._lock:
            futures = tuple(self._pending.values())
            self._pending.clear()
            staged = tuple(self._staged.values())
            self._staged.clear()
        # A worker may currently be inside a bounded file read.  Waiting here
        # keeps the shared reader valid and makes qwen_clear deterministic.
        for future in futures:
            try:
                value = future.result()
                if value is not None:
                    value.close()
            except BaseException:
                pass
        for value in staged:
            value.close()
        gc.collect()
        for device in self.backbone.devices:
            if device.type == "cuda" and torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            staged_bytes = sum(item.compressed_bytes for item in self._staged.values())
            pending = sorted(self._pending)
            staged = sorted(self._staged)
        self.max_staged_bytes = max(self.max_staged_bytes, int(staged_bytes))
        return {
            "enabled": True,
            "max_bytes": int(self.max_bytes),
            "cuda_stream": bool(self.use_cuda_stream),
            "submitted": int(self.submitted),
            "completed": int(self.completed),
            "consumed": int(self.consumed),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "skipped_capacity": int(self.skipped_capacity),
            "fallbacks": int(self.fallbacks),
            "errors": int(self.errors),
            "prefetch_bytes": int(self.prefetch_bytes),
            "prefetch_seconds": float(self.prefetch_seconds),
            "wait_seconds": float(self.wait_seconds),
            "staged_bytes": int(staged_bytes),
            "max_staged_bytes": int(self.max_staged_bytes),
            "pending_layers": pending,
            "staged_layers": staged,
            "last_error": self.last_error,
        }

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
        self.reset()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self._streams.clear()


class Qwen32Q2LayerMPBackbone(nn.Module):
    """Lazy complete-layer Qwen32 backbone with one activation handoff."""

    def __init__(
        self,
        layout_or_path: qwen.GGUFLayout | os.PathLike[str] | str,
        *,
        devices: Sequence[torch.device | str] | None = None,
        layer_split: int | str | None = None,
        dtype: torch.dtype = DEFAULT_MP_DTYPE,
        staging_mib: int = DEFAULT_MP_STAGING_MIB,
        residency: str = "evict",
        keep_layers: int | Sequence[int] = 0,
        cache_dequantized: bool = False,
        reader: qwen.Qwen32Q2DiskReader | None = None,
        output_device: torch.device | str | None = None,
        check_peer_access: bool = True,
        enforce_capacity: bool = True,
        baseline_bytes: Sequence[int] | Mapping[Any, int] | None = None,
        capacity_bytes: Sequence[int] | Mapping[Any, int] | None = None,
        safety_fraction: float = DEFAULT_MP_SAFETY_FRACTION,
        prefetch: bool | None = None,
        prefetch_max_mib: int | None = None,
    ) -> None:
        super().__init__()
        self.layout = (
            layout_or_path
            if isinstance(layout_or_path, qwen.GGUFLayout)
            else qwen.inspect_gguf(layout_or_path)
        )
        self.devices = normalize_mp_devices(
            devices,
            require_cuda=False,
            check_peer_access=check_peer_access,
        )
        self.first_device, self.second_device = self.devices
        self.compute_dtype = dtype
        self.residency = str(residency).strip().lower()
        if self.residency not in {"evict", "partial", "full"}:
            raise ValueError("Qwen32 MP residency must be evict, partial, or full")
        self._cache_dequantized = bool(cache_dequantized)
        requested_prefetch = (
            _prefetch_from_env(False) if prefetch is None else bool(prefetch)
        )
        requested_prefetch_max_mib = (
            int(
                os.environ.get(
                    "H3_QWEN32_MP_PREFETCH_MAX_MIB",
                    os.environ.get(
                        "H3_QWEN32_PREFETCH_MAX_MIB",
                        str(DEFAULT_MP_PREFETCH_MAX_MIB),
                    ),
                )
            )
            if prefetch_max_mib is None
            else int(prefetch_max_mib)
        )
        if requested_prefetch_max_mib <= 0:
            raise ValueError("Qwen32 MP prefetch_max_mib must be positive")
        # Keeping compressed data from two adjacent layers is only useful for
        # evict mode.  Partial/full residency already retains payload and
        # would turn a prefetch into an avoidable duplicate allocation.
        if requested_prefetch and self.residency != "evict":
            logging.info(
                "[H3 Qwen MP] prefetch requested with residency=%s; disabling "
                "prefetch to avoid duplicate resident payload",
                self.residency,
            )
            requested_prefetch = False
        self.prefetch_enabled = bool(requested_prefetch)
        self.prefetch_max_mib = int(requested_prefetch_max_mib)
        self.layer_specs = qwen.language_matrix_specs(self.layout)
        if not self.layer_specs:
            raise ValueError("GGUF contains no Qwen language matrix specs")
        self.keep_layer_ids = self._normalize_keep_layers(keep_layers)
        self.plan = plan_layer_split(
            self.layout,
            devices=self.devices,
            split=layer_split,
            residency=self.residency,
            keep_layers=keep_layers,
            cache_dequantized=self._cache_dequantized,
            dtype=dtype,
            baseline_bytes=baseline_bytes,
            capacity_bytes=capacity_bytes,
            safety_fraction=safety_fraction,
        )
        # A one-slot prefetch holds the next layer's compressed payload while
        # the current layer is computing.  Include that bounded overlap in
        # the capacity gate; otherwise a nearly-full service could pass the
        # ordinary evict estimate and OOM only at a layer boundary.
        self.prefetch_capacity_extra_bytes = 0
        if requested_prefetch:
            largest_prefetch = max(
                int(item.compressed_bytes + item.norm_bytes)
                for item in self.plan.layer_costs
            )
            self.prefetch_capacity_extra_bytes = min(
                largest_prefetch,
                requested_prefetch_max_mib * MIB,
            )
        self.prefetch_estimated_peak_bytes = tuple(
            int(value) + int(self.prefetch_capacity_extra_bytes)
            for value in self.plan.estimated_peak_bytes
        )
        capacity_failed = any(
            cap is not None and value > cap
            for value, cap in zip(
                self.prefetch_estimated_peak_bytes,
                self.plan.capacity_bytes,
            )
        )
        if enforce_capacity and (
            any(item is False for item in self.plan.fits_capacity)
            or capacity_failed
        ):
            raise MemoryError(
                "Qwen32 MP split does not fit the current device capacity: "
                f"estimated={self.prefetch_estimated_peak_bytes}, "
                f"capacity={self.plan.capacity_bytes}; choose evict or an explicit split"
            )
        self.enforce_capacity = bool(enforce_capacity)
        self.output_device = (
            _canonical_device(output_device)
            if output_device is not None
            else self.second_device
        )
        if self.output_device not in self.devices:
            raise ValueError("Qwen32 MP output_device must be one of the MP devices")
        self.reader = reader or qwen.Qwen32Q2DiskReader(
            self.layout.path,
            staging_mib=int(staging_mib),
        )
        self._owns_reader = reader is None
        self.blocks: dict[int, Qwen32Q2MPLayerBlock] = {}
        self.state = "META_ONLY"
        self.forward_count = 0
        self.layer_load_seconds = 0.0
        self.layer_raw_load_seconds = 0.0
        self.layer_forward_seconds = 0.0
        self.matrix_load_seconds = 0.0
        self.matrix_dequant_seconds = 0.0
        self.matrix_gemm_seconds = 0.0
        self.layer_compute_seconds = 0.0
        self.layer_clear_seconds = 0.0
        self.handoff_count = 0
        self.handoff_bytes = 0
        self._peak_activation_bytes = [0, 0]
        self._closed = False
        self._lock = threading.RLock()
        self._pending_prefetched_layer: _PrefetchedLayer | None = None
        self._prefetcher: _Qwen32MPLayerPrefetcher | None = None
        if self.prefetch_enabled:
            self._prefetcher = _Qwen32MPLayerPrefetcher(
                self,
                max_bytes=self.prefetch_max_mib * MIB,
            )
        logging.info(
            "[H3 Qwen MP] active: layers=%d split=%d/%d devices=%s,%s "
            "residency=%s prefetch=%s cap=%d MiB; no Qwen rank-1/NCCL",
            self.plan.layer_count,
            self.plan.split,
            self.plan.layer_count - self.plan.split,
            self.first_device,
            self.second_device,
            self.residency,
            self.prefetch_enabled,
            self.prefetch_max_mib,
        )

    def _normalize_keep_layers(self, keep_layers: int | Sequence[int]) -> set[int]:
        if isinstance(keep_layers, int):
            keep = set(range(max(0, int(keep_layers))))
        else:
            keep = {int(item) for item in keep_layers}
        unknown = sorted(item for item in keep if item not in self.layer_specs)
        if unknown:
            raise ValueError(f"Qwen32 MP keep_layers contains unknown IDs: {unknown[:8]}")
        if self.residency == "full":
            keep = set(self.layer_specs)
        return keep

    def _owner_index(self, layer: int) -> int:
        return self.plan.owner_index(layer)

    def _owner(self, layer: int) -> torch.device:
        return self.devices[self._owner_index(layer)]

    def _norm_values(self, layer: int, device: torch.device) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        for spec in self.layout.language_layers.get(layer, ()):
            role = _norm_role(spec.name)
            if role is None:
                continue
            raw = self.reader.read_tensor(spec, device=device)
            try:
                values[role] = qwen.dequantize_ggml(
                    raw,
                    spec.qtype,
                    spec.shape,
                    dtype=self.compute_dtype,
                ).reshape(spec.shape)
            finally:
                del raw
        missing = set(qwen.NORM_ROLES).difference(values)
        if missing:
            raise ValueError(f"layer {layer} is missing norms: {sorted(missing)}")
        return values

    def load_layer(self, layer: int) -> Qwen32Q2MPLayerBlock:
        with self._lock:
            if self._closed:
                raise RuntimeError("Qwen32 MP backbone is closed")
            layer = int(layer)
            cached = self.blocks.get(layer)
            if cached is not None:
                return cached
            roles = self.layer_specs.get(layer)
            if roles is None:
                raise KeyError(f"Qwen language layer {layer} is absent")
            device = self._owner(layer)
            prefetched = getattr(self, "_pending_prefetched_layer", None)
            if prefetched is not None and (
                prefetched.layer != layer or prefetched.device != device
            ):
                prefetched.close()
                raise RuntimeError(f"prefetched layer ownership mismatch for {layer}")
            started = time.perf_counter()
            matrices: dict[str, Qwen32Q2MPLinear] = {}
            for role in qwen.MATRIX_ROLES:
                spec = roles.get(role)
                if spec is None:
                    raise ValueError(f"layer {layer} is missing matrix {role}")
                matrix = Qwen32Q2MPLinear(
                    _full_descriptor(spec),
                    self.reader,
                    device=device,
                    dtype=self.compute_dtype,
                    cache_dequantized=self._cache_dequantized,
                )
                if prefetched is not None:
                    raw = prefetched.raw.pop(role, None)
                    if raw is None:
                        prefetched.close()
                        raise RuntimeError(
                            f"prefetched layer {layer} is missing matrix {role}"
                        )
                    matrix.attach_raw(raw)
                matrices[role] = matrix
            if prefetched is not None:
                norms = prefetched.norms
                prefetched.norms = {}
                prefetched.raw.clear()
            else:
                norms = self._norm_values(layer, device)
            block = Qwen32Q2MPLayerBlock(
                layer,
                matrices,
                norms,
                device=device,
                dtype=self.compute_dtype,
            )
            if prefetched is not None:
                block._prefetch_ready_event = prefetched.ready_event
                prefetched.ready_event = None
                prefetched.close()
            self.blocks[layer] = block
            self.layer_load_seconds += time.perf_counter() - started
            self.state = "ENCODING"
            return block

    @staticmethod
    def _layer_ids_default(layer_specs: Mapping[int, Any]) -> tuple[int, ...]:
        return tuple(sorted(int(layer) for layer in layer_specs))

    @torch.inference_mode()
    def forward_hidden(
        self,
        hidden: torch.Tensor,
        *,
        layer_ids: Sequence[int] | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: Any = None,
        deepstack_embeds: Sequence[torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        after_layer: Callable[[int, torch.Tensor], torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        with self._lock:
            if self._closed:
                raise RuntimeError("Qwen32 MP backbone is closed")
            if not torch.is_tensor(hidden):
                raise TypeError("Qwen32 MP hidden must be a tensor")
            if hidden.is_floating_point() and hidden.dtype != self.compute_dtype:
                hidden = hidden.to(dtype=self.compute_dtype)
            requested = (
                self._layer_ids_default(self.layer_specs)
                if layer_ids is None
                else tuple(int(layer) for layer in layer_ids)
            )
            if not requested:
                raise ValueError("Qwen32 MP forward requires at least one layer")
            if requested != tuple(sorted(requested)) or len(set(requested)) != len(requested):
                raise ValueError("Qwen32 MP layer_ids must be strictly increasing")
            unknown = sorted(set(requested).difference(self.layer_specs))
            if unknown:
                raise ValueError(f"Qwen32 MP forward contains unknown layers: {unknown}")
            if deepstack_embeds is not None and visual_pos_masks is None:
                raise ValueError("deepstack_embeds requires visual_pos_masks")
            current = hidden
            mask = attention_mask
            freqs = freqs_cis
            visual_mask = visual_pos_masks
            deepstack = deepstack_embeds
            started = time.perf_counter()
            first_nonfinite_layer: int | None = None
            first_nonfinite_device: torch.device | None = None
            first_target = self._owner(requested[0])
            # Callers normally assemble Qwen inputs on the first GPU, but the
            # backend is also usable as a standalone adapter.  Normalize all
            # metadata before the first block so a CPU/other-device mask never
            # reaches the attention kernel by accident.
            mask = _move_tree(mask, first_target)
            freqs = _move_tree(freqs, first_target)
            visual_mask = _move_tree(visual_mask, first_target)
            deepstack = _move_tree(deepstack, first_target)
            if _finite_trace_enabled():
                _trace_tensor_finite("forward.hidden", current)
                if torch.is_tensor(mask):
                    _trace_tensor_finite("forward.attention_mask", mask)
                if torch.is_tensor(visual_mask):
                    _trace_tensor_finite("forward.visual_mask", visual_mask)
                if isinstance(freqs, (tuple, list)):
                    for index, item in enumerate(freqs):
                        if torch.is_tensor(item):
                            _trace_tensor_finite(f"forward.freqs.{index}", item)
                if isinstance(deepstack, (tuple, list)):
                    for index, item in enumerate(deepstack):
                        if torch.is_tensor(item):
                            _trace_tensor_finite(f"forward.deepstack.{index}", item)
            prefetcher = self._prefetcher
            for position, layer in enumerate(requested):
                target = self._owner(layer)
                if current.device != target:
                    self.handoff_count += 1
                    self.handoff_bytes += int(current.numel() * current.element_size())
                    current = current.to(target, non_blocking=target.type == "cuda")
                    mask = _move_tree(mask, target)
                    freqs = _move_tree(freqs, target)
                    visual_mask = _move_tree(visual_mask, target)
                    deepstack = _move_tree(deepstack, target)
                prefetched = (
                    prefetcher.consume(layer) if prefetcher is not None else None
                )
                # Keep the public/mocked ``load_layer(layer)`` shape intact
                # for adapters that replace it in CPU tests.  The temporary
                # handoff slot is private to this forward invocation.
                if prefetched is not None:
                    self._pending_prefetched_layer = prefetched
                try:
                    block = self.load_layer(layer)
                finally:
                    self._pending_prefetched_layer = None
                # Read the complete compressed current layer before compute.
                # This is also the non-prefetch baseline: the only A/B change
                # is whether the next layer is read concurrently below.
                ensure_raw = getattr(block, "ensure_raw", None)
                if ensure_raw is not None:
                    raw_before = float(getattr(block, "raw_load_seconds", 0.0))
                    ensure_raw()
                    self.layer_raw_load_seconds += max(
                        0.0,
                        float(getattr(block, "raw_load_seconds", 0.0)) - raw_before,
                    )
                # Give the worker an uncontended reader window while this
                # layer's dequant/GEMMs execute.
                if prefetcher is not None and position + 1 < len(requested):
                    prefetcher.submit(requested[position + 1])
                current = block(
                    current,
                    attention_mask=mask,
                    freqs_cis=freqs,
                )
                # Qwen32Q2MPLayerBlock already performs this finite reduction
                # for its per-layer stats.  Preserve the first failing layer
                # so a request error identifies the actual numerical boundary
                # instead of reporting only the final aggregate output.
                if first_nonfinite_layer is None and not bool(
                    getattr(block, "last_stats", {}).get("finite", True)
                ):
                    first_nonfinite_layer = int(layer)
                    first_nonfinite_device = target
                    logging.error(
                        "[H3 Qwen MP] first non-finite layer=%d device=%s "
                        "shape=%s; subsequent layers are not numerically usable",
                        layer,
                        target,
                        tuple(current.shape),
                    )
                timing = getattr(block, "last_timing", {})
                self.matrix_load_seconds += float(
                    timing.get("matrix_load_seconds", 0.0)
                )
                self.matrix_dequant_seconds += float(
                    timing.get("matrix_dequant_seconds", 0.0)
                )
                self.matrix_gemm_seconds += float(
                    timing.get("matrix_gemm_seconds", 0.0)
                )
                self.layer_compute_seconds += float(
                    timing.get("block_wall_seconds", 0.0)
                )
                owner_index = self._owner_index(layer)
                self._peak_activation_bytes[owner_index] = max(
                    self._peak_activation_bytes[owner_index],
                    int(current.numel() * current.element_size()),
                )
                if deepstack is not None and layer < len(deepstack):
                    if visual_mask is None:
                        raise ValueError("deepstack_embeds requires visual_pos_masks")
                    current[visual_mask] = current[visual_mask] + deepstack[layer].to(
                        device=current.device,
                        dtype=current.dtype,
                    )
                    if _finite_trace_enabled():
                        _trace_tensor_finite(
                            f"layer{layer}.after_deepstack", current
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
                    clear_started = time.perf_counter()
                    if isinstance(block, Qwen32Q2MPLayerBlock):
                        # ``empty_cache`` after every layer serializes the
                        # allocator and can cost more than the bounded read.
                        # Drop Python/GPU payload references immediately but
                        # retain the allocator pool until qwen_clear(); this
                        # is the same policy for baseline and prefetch A/B.
                        block.clear(release_cache=False, collect=False)
                    else:
                        block.clear()
                    self.layer_clear_seconds += time.perf_counter() - clear_started
                    self.blocks.pop(layer, None)
            if current.device != self.output_device:
                self.handoff_count += 1
                self.handoff_bytes += int(current.numel() * current.element_size())
                current = current.to(self.output_device, non_blocking=self.output_device.type == "cuda")
            self.forward_count += 1
            self.layer_forward_seconds += time.perf_counter() - started
            self.state = "DIT_READY" if self.residency == "evict" else "ENCODING"
            if first_nonfinite_layer is not None:
                raise RuntimeError(
                    "Qwen32 MP output produced NaN/Inf at layer "
                    f"{first_nonfinite_layer} on {first_nonfinite_device}"
                )
            return current

    forward = forward_hidden
    encode_hidden = forward_hidden

    def trim(self, keep_layers: int | Sequence[int] = 0) -> None:
        with self._lock:
            if self._prefetcher is not None:
                self._prefetcher.reset()
            keep = self._normalize_keep_layers(keep_layers)
            for layer in tuple(self.blocks):
                if layer not in keep:
                    self.blocks[layer].clear()
                    self.blocks.pop(layer, None)
            self.keep_layer_ids = keep
            self.state = "ENCODING" if self.blocks else "DIT_READY"
            gc.collect()
            for device in self.devices:
                if device.type == "cuda" and torch.cuda.is_available():
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()

    def clear(self) -> None:
        with self._lock:
            if self._prefetcher is not None:
                self._prefetcher.reset()
            for block in tuple(self.blocks.values()):
                block.clear()
            self.blocks.clear()
            gc.collect()
            for device in self.devices:
                if device.type == "cuda" and torch.cuda.is_available():
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
            self.state = "META_ONLY"

    def stats(self) -> dict[str, Any]:
        block_stats = {str(layer): block.stats() for layer, block in self.blocks.items()}
        resident = [0, 0]
        for layer, block in self.blocks.items():
            resident[self._owner_index(layer)] += block.resident_bytes
        memory: list[dict[str, int | None]] = []
        for device in self.devices:
            allocated = reserved = None
            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    allocated = int(torch.cuda.memory_allocated(device))
                    reserved = int(torch.cuda.memory_reserved(device))
                except RuntimeError:  # pragma: no cover - driver edge
                    pass
            memory.append({"allocated_bytes": allocated, "reserved_bytes": reserved})
        prefetch = (
            self._prefetcher.stats()
            if self._prefetcher is not None
            else {
                "enabled": False,
                "max_bytes": int(self.prefetch_max_mib * MIB),
            }
        )
        return {
            "mode": QWEN32_MP_MODE,
            "state": self.state,
            "devices": [str(item) for item in self.devices],
            "output_device": str(self.output_device),
            "residency": self.residency,
            "prefetch": prefetch,
            "enforce_capacity": self.enforce_capacity,
            "keep_layers": sorted(self.keep_layer_ids),
            "layer_split": self.plan.as_dict(),
            "prefetch_capacity_extra_bytes": int(self.prefetch_capacity_extra_bytes),
            "prefetch_estimated_peak_bytes": list(self.prefetch_estimated_peak_bytes),
            "loaded_layers": sorted(self.blocks),
            "resident_bytes_by_device": resident,
            "resident_bytes": sum(resident),
            "peak_activation_bytes_by_device": list(self._peak_activation_bytes),
            "cuda_memory": memory,
            "handoff_count": self.handoff_count,
            "handoff_bytes": self.handoff_bytes,
            "forward_count": self.forward_count,
            "layer_load_seconds": self.layer_load_seconds,
            "layer_raw_load_seconds": self.layer_raw_load_seconds,
            "layer_forward_seconds": self.layer_forward_seconds,
            "matrix_load_seconds": self.matrix_load_seconds,
            "matrix_dequant_seconds": self.matrix_dequant_seconds,
            "matrix_gemm_seconds": self.matrix_gemm_seconds,
            "layer_compute_seconds": self.layer_compute_seconds,
            "layer_clear_seconds": self.layer_clear_seconds,
            "reader": self.reader.stats(),
            "payload_mmap_hits": qwen.payload_mmap_hits(self.layout.path),
            "blocks": block_stats,
        }

    get_stats = stats

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.clear()
            if self._prefetcher is not None:
                self._prefetcher.close()
                self._prefetcher = None
            if self._owns_reader:
                self.reader.close()
            self._closed = True

    def __enter__(self) -> "Qwen32Q2LayerMPBackbone":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class Qwen32Q2LayerMPRuntime:
    """Small runtime-compatible adapter for a future MP switch.

    Its public methods intentionally mirror the Qwen subset of
    ``H3TPRuntime``.  It has no child process and never initializes
    ``torch.distributed``.  A failed request clears the local backbone and
    leaves ``qwen_clear`` safe to call, so the CLIP facade can reuse its normal
    fail-closed lifecycle.
    """

    mode = QWEN32_MP_MODE

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        devices: Sequence[torch.device | str] | None = None,
        layer_split: int | str | None = None,
        staging_mib: int = DEFAULT_MP_STAGING_MIB,
        residency: str = "evict",
        keep_layers: int | Sequence[int] = 0,
        cache_dequantized: bool = False,
        dtype: torch.dtype = DEFAULT_MP_DTYPE,
        output_device: torch.device | str | None = None,
        check_peer_access: bool = True,
        enforce_capacity: bool = True,
        baseline_bytes: Sequence[int] | Mapping[Any, int] | None = None,
        capacity_bytes: Sequence[int] | Mapping[Any, int] | None = None,
        safety_fraction: float = DEFAULT_MP_SAFETY_FRACTION,
        prefetch: bool | None = None,
        prefetch_max_mib: int | None = None,
    ) -> None:
        self.model_path = "" if model_path is None else str(Path(model_path).resolve())
        self.devices = normalize_mp_devices(
            devices,
            require_cuda=False,
            check_peer_access=check_peer_access,
        )
        self.layer_split = layer_split
        self.staging_mib = int(staging_mib)
        self.residency = str(residency)
        self.keep_layers = keep_layers
        self.cache_dequantized = bool(cache_dequantized)
        self.dtype = dtype
        self.output_device = output_device
        self.check_peer_access = bool(check_peer_access)
        self.enforce_capacity = bool(enforce_capacity)
        self.baseline_bytes = baseline_bytes
        self.capacity_bytes = capacity_bytes
        self.safety_fraction = float(safety_fraction)
        self.prefetch = prefetch
        self.prefetch_max_mib = prefetch_max_mib
        self.qwen_backbone: Qwen32Q2LayerMPBackbone | None = None
        self.last_qwen_profile: dict[str, Any] | None = None
        self.child = None  # explicit marker: MP never owns a rank-1 process
        self.process_started = False
        self.started = False
        self.closed = False
        self.failed = False
        self.lock = threading.RLock()

    def ensure_process_started(self) -> None:
        """Compatibility no-op: MP deliberately has no worker process."""

        with self.lock:
            if self.closed:
                raise RuntimeError("Qwen32 MP runtime was closed")
            self.process_started = True

    def configure_qwen(
        self,
        model_path: str,
        *,
        staging_mib: int | None = None,
        residency: str | None = None,
        keep_layers: int | Sequence[int] | None = None,
        cache_dequantized: bool | None = None,
        layer_split: int | str | None = None,
    ) -> None:
        with self.lock:
            self.ensure_process_started()
            requested = str(Path(model_path).resolve())
            if self.qwen_backbone is not None and requested != self.model_path:
                raise RuntimeError("cannot change Qwen model after MP execution started")
            if self.qwen_backbone is not None:
                values = (
                    self.staging_mib if staging_mib is None else int(staging_mib),
                    self.residency if residency is None else str(residency),
                    self.keep_layers if keep_layers is None else keep_layers,
                    self.cache_dequantized if cache_dequantized is None else bool(cache_dequantized),
                    self.layer_split if layer_split is None else layer_split,
                )
                current = (
                    self.staging_mib,
                    self.residency,
                    self.keep_layers,
                    self.cache_dequantized,
                    self.layer_split,
                )
                if values != current:
                    raise RuntimeError("cannot change Qwen MP options after execution started")
                return
            self.model_path = requested
            if staging_mib is not None:
                self.staging_mib = int(staging_mib)
            if residency is not None:
                self.residency = str(residency)
            if keep_layers is not None:
                self.keep_layers = keep_layers
            if cache_dequantized is not None:
                self.cache_dequantized = bool(cache_dequantized)
            if layer_split is not None:
                self.layer_split = layer_split

    def _ensure_backbone(self) -> Qwen32Q2LayerMPBackbone:
        if not self.model_path:
            raise RuntimeError("Qwen32 MP model_path is not configured")
        if self.qwen_backbone is None:
            self.qwen_backbone = Qwen32Q2LayerMPBackbone(
                self.model_path,
                devices=self.devices,
                layer_split=self.layer_split,
                dtype=self.dtype,
                staging_mib=self.staging_mib,
                residency=self.residency,
                keep_layers=self.keep_layers,
                cache_dequantized=self.cache_dequantized,
                output_device=self.output_device,
                check_peer_access=self.check_peer_access,
                enforce_capacity=self.enforce_capacity,
                baseline_bytes=self.baseline_bytes,
                capacity_bytes=self.capacity_bytes,
                safety_fraction=self.safety_fraction,
                prefetch=self.prefetch,
                prefetch_max_mib=self.prefetch_max_mib,
            )
        return self.qwen_backbone

    def qwen_forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: Any = None,
        deepstack_embeds: Sequence[torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self.lock:
            self.ensure_process_started()
            if self.failed:
                raise RuntimeError("Qwen32 MP runtime is failed; call qwen_clear() first")
            # Keep the same Qwen→DiT/VAE lifecycle gate as the TP runtime:
            # release any decoder payload before the first layer allocates,
            # and only let qwen_clear() reopen the gate after success.
            try:
                from . import h3_async_vae_bridge
            except ImportError:  # pragma: no cover - standalone smoke path
                h3_async_vae_bridge = None
            if h3_async_vae_bridge is not None:
                prepare = getattr(
                    h3_async_vae_bridge, "prepare_active_vae_for_qwen", None
                )
                if prepare is not None:
                    prepare()
            backbone = self._ensure_backbone()
            try:
                output = backbone.forward_hidden(
                    hidden,
                    attention_mask=attention_mask,
                    freqs_cis=freqs_cis,
                    deepstack_embeds=deepstack_embeds,
                    visual_pos_masks=visual_pos_masks,
                )
                self.started = True
                self.last_qwen_profile = {
                    "mode": QWEN32_MP_MODE,
                    "shape": list(output.shape),
                    "finite": bool(torch.isfinite(output).all().item()),
                    "backbone": backbone.stats(),
                }
                return output
            except BaseException:
                self.failed = True
                try:
                    backbone.clear()
                except BaseException:
                    logging.exception("[H3 Qwen32 MP] failed to clear after forward error")
                raise

    def qwen_clear(self, *, notify_vae: bool = True) -> dict[str, Any]:
        with self.lock:
            self.ensure_process_started()
            # Do not call _ensure_backbone here: a failed/empty request must
            # never create a new model merely to clear it.
            if self.qwen_backbone is not None:
                self.qwen_backbone.clear()
            self.failed = False
            notified = False
            if notify_vae:
                try:
                    from . import h3_async_vae_bridge
                except ImportError:  # pragma: no cover - standalone use
                    h3_async_vae_bridge = None
                if h3_async_vae_bridge is not None:
                    notify = getattr(h3_async_vae_bridge, "notify_qwen_cleared", None)
                    if notify is not None:
                        notify()
                        notified = True
            return {
                "mode": QWEN32_MP_MODE,
                "configured": bool(self.model_path),
                "rank0": None if self.qwen_backbone is None else self.qwen_backbone.stats(),
                "rank1": None,
                "vae_notified": notified,
            }

    def qwen_trim(self, keep_layers: int | Sequence[int] = 0) -> dict[str, Any]:
        with self.lock:
            self.ensure_process_started()
            backbone = self._ensure_backbone()
            backbone.trim(keep_layers)
            self.keep_layers = keep_layers
            return {"mode": QWEN32_MP_MODE, "rank0": backbone.stats(), "rank1": None}

    def qwen_stats(self) -> dict[str, Any]:
        with self.lock:
            self.ensure_process_started()
            return {
                "mode": QWEN32_MP_MODE,
                "configured": bool(self.model_path),
                "rank0": None if self.qwen_backbone is None else self.qwen_backbone.stats(),
                "rank1": None,
                "last_profile": self.last_qwen_profile,
                "failed": self.failed,
            }

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            if self.qwen_backbone is not None:
                self.qwen_backbone.close()
                self.qwen_backbone = None
            self.process_started = False
            self.started = False
            self.closed = True


class Qwen32Q2MPRuntimeHandle:
    """Duck-compatible workflow handle for the existing Qwen CLIP facade.

    Keeping this handle in the MP module means a future node can replace the
    TP runtime handle with two lines while the active TP node remains untouched.
    ``qwen_clip`` storage is deliberately local to the handle, just like the
    current TP wrapper.
    """

    TYPE = "H3_QWEN32_MP_RUNTIME"

    def __init__(
        self,
        runtime: Qwen32Q2LayerMPRuntime | None = None,
        *,
        qwen_model_path: str | os.PathLike[str] | None = None,
        qwen_staging_mib: int = DEFAULT_MP_STAGING_MIB,
    ) -> None:
        self.runtime = (
            runtime
            if runtime is not None
            else Qwen32Q2LayerMPRuntime(
                qwen_model_path,
                staging_mib=qwen_staging_mib,
            )
        )
        configured = qwen_model_path or getattr(self.runtime, "model_path", "")
        self.qwen_model_path = str(Path(configured).resolve()) if configured else ""
        self.qwen_staging_mib = int(qwen_staging_mib)
        self._qwen_clip: Any | None = None
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def configure_qwen(
        self,
        *,
        staging_mib: int,
        residency: str = "evict",
        keep_layers: int | Sequence[int] = 0,
        cache_dequantized: bool = False,
        layer_split: int | str | None = None,
    ) -> Any:
        if not self.qwen_model_path:
            raise RuntimeError("Qwen32 MP handle has no model path")
        return self.runtime.configure_qwen(
            self.qwen_model_path,
            staging_mib=int(staging_mib),
            residency=str(residency),
            keep_layers=keep_layers,
            cache_dequantized=bool(cache_dequantized),
            layer_split=layer_split,
        )

    def qwen_clip(self) -> Any | None:
        with self._lock:
            return self._qwen_clip

    def set_qwen_clip(self, clip: Any) -> None:
        with self._lock:
            self._qwen_clip = clip


def build_qwen32_mp_clip(
    runtime_handle: Qwen32Q2MPRuntimeHandle,
    *,
    qwen_path: str | os.PathLike[str] | None = None,
    staging_mib: int | None = None,
    residency: str = "evict",
    keep_layers: int | Sequence[int] = 0,
    cache_dequantized: bool = False,
) -> Any:
    """Build the existing CLIP-like facade on top of the MP adapter lazily.

    The import is intentionally inside the function: importing this module by
    itself remains safe in a service that currently runs the TP route, and no
    ComfyUI node discovery or CUDA allocation is triggered.
    """

    if not isinstance(runtime_handle, Qwen32Q2MPRuntimeHandle):
        raise TypeError("build_qwen32_mp_clip requires Qwen32Q2MPRuntimeHandle")
    requested_path = qwen_path or runtime_handle.qwen_model_path
    if not requested_path:
        raise ValueError("Qwen32 MP clip requires a model path")
    path = str(Path(requested_path).resolve())
    if runtime_handle.qwen_model_path and path != runtime_handle.qwen_model_path:
        raise ValueError(
            "Qwen32 MP clip path differs from the model path on its runtime handle"
        )
    if not runtime_handle.qwen_model_path:
        runtime_handle.qwen_model_path = path
    if staging_mib is not None:
        runtime_handle.qwen_staging_mib = int(staging_mib)
    runtime_handle.configure_qwen(
        staging_mib=(
            int(staging_mib)
            if staging_mib is not None
            else int(runtime_handle.qwen_staging_mib)
        ),
        residency=str(residency),
        keep_layers=keep_layers,
        cache_dequantized=bool(cache_dequantized),
    )
    try:
        from .h3_qwen32_tp_node import _Qwen32TPClip
    except ImportError:  # pragma: no cover - standalone source-tree import
        from h3_qwen32_tp_node import _Qwen32TPClip  # type: ignore
    return _Qwen32TPClip(
        runtime_handle,
        qwen_path=path,
        staging_mib=(
            int(staging_mib)
            if staging_mib is not None
            else int(runtime_handle.qwen_staging_mib)
        ),
        residency=str(residency),
        keep_layers=keep_layers,
        cache_dequantized=bool(cache_dequantized),
    )


def create_qwen32_backend(mode: str | None = None, **kwargs: Any) -> Qwen32Q2LayerMPRuntime:
    """Factory used by the eventual two-line runtime switch.

    TP intentionally is not imported or changed here.  Passing ``tp`` gives a
    clear instruction to keep using the existing ``H3TPRuntime`` instead of
    silently constructing a different backend.
    """

    normalized = resolve_qwen32_mode(mode)
    if normalized == QWEN32_MP_MODE:
        return Qwen32Q2LayerMPRuntime(**kwargs)
    if normalized == QWEN32_TP_MODE:
        raise RuntimeError(
            "output-row TP remains owned by h3_tp_runtime; use that runtime explicitly"
        )
    raise ValueError(f"unsupported Qwen32 backend mode {mode!r}; use mp or tp")


# Short names make the eventual runtime replacement easy to read and mirror
# the aliases exposed by the existing TP module.
Qwen32Q2MPBackbone = Qwen32Q2LayerMPBackbone
Qwen32Q2MPModel = Qwen32Q2LayerMPBackbone


__all__ = [
    "DEFAULT_MP_DTYPE",
    "DEFAULT_MP_PREFETCH_MAX_MIB",
    "DEFAULT_MP_SAFETY_FRACTION",
    "DEFAULT_MP_STAGING_MIB",
    "QWEN32_MP_MODE",
    "QWEN32_MODE_ENV",
    "QWEN32_MP_PREFETCH_ALIAS_ENV",
    "QWEN32_MP_PREFETCH_ENV",
    "QWEN32_TP_MODE",
    "Qwen32MPLayerCost",
    "Qwen32MPSplitPlan",
    "Qwen32Q2LayerMPBackbone",
    "Qwen32Q2MPBackbone",
    "Qwen32Q2MPModel",
    "Qwen32Q2LayerMPRuntime",
    "Qwen32Q2MPRuntimeHandle",
    "Qwen32Q2MPLayerBlock",
    "Qwen32Q2MPLinear",
    "build_qwen32_mp_clip",
    "create_qwen32_backend",
    "normalize_mp_devices",
    "plan_layer_split",
    "resolve_qwen32_mode",
]
