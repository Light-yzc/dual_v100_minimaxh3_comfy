"""Bounded standard Q4_0 storage for H3 residual-cache tensors.

The H3 residual stream is FP32 and can approach one GiB per rank at the
largest packed layouts.  Cache tensors are quantized in row chunks to the
standard GGML Q4_0 byte layout and are dequantized only into a bounded CUDA
staging tensor when a metric or residual add needs them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch

try:  # Package import in ComfyUI.
    from . import h3_q4_tp as q4_tp
except ImportError:  # Standalone rank-1 import from this directory.
    import h3_q4_tp as q4_tp


Q4_BLOCK_ELEMENTS = 32
Q4_BLOCK_BYTES = 18
Q4_FORMAT = "ggml_q4_0"
Q4_FP16_SCALE_MAX = 65504.0
Q4_MAX_FINITE_SOURCE_ABS = Q4_FP16_SCALE_MAX * 8.0
# Q4-vs-Q4 comparisons have an exact zero floor for an unchanged input.  A
# small non-zero default catches minor step-to-step changes without treating
# the unavoidable FP32->Q4 reconstruction error as movement.
DEFAULT_GROUP_THRESHOLD = 0.005
DEFAULT_CACHE_CHUNK_ROWS = 256
DEFAULT_SIGNATURE_MAX_TOKENS = 2048
DEFAULT_SIGNATURE_HIDDEN_SAMPLES = 32


def normalize_feature_mode(value: str | None) -> str:
    """Normalize the bounded input-feature representation.

    ``q4`` is the historical, byte-exact comparison path.  ``signature`` is
    an opt-in research path: only a deterministic, stratified sample of the
    group input is retained for the decision.  Neither mode changes the Q4
    residual used to produce the model output.
    """

    normalized = "q4" if value is None else str(value).strip().lower()
    if normalized in {"q4", "q4_0", "ggml_q4_0"}:
        return "q4"
    if normalized in {"signature", "sketch", "bounded_signature"}:
        return "signature"
    raise ValueError(
        "H3 group feature_mode must be q4 or signature, "
        f"got {value!r}"
    )


def normalize_q4_format(value: str | None) -> str:
    """Accept only the standard GGML Q4_0 cache representation."""

    normalized = Q4_FORMAT if value is None else str(value).strip().lower()
    if normalized in {"q4_0", "ggml_q4_0"}:
        return Q4_FORMAT
    raise ValueError(
        f"H3 cache format must be standard GGML Q4_0, got {value!r}"
    )


def _normalize_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized == "auto":
        normalized = "cpu"
    if normalized not in {"cpu", "gpu"}:
        raise ValueError(f"H3 Q4 cache policy must be cpu/gpu, got {policy!r}")
    return normalized


def _validate_tensor(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.ndim != 2 or not tensor.is_floating_point():
        raise ValueError(
            f"H3 Q4 cache expects a floating 2D tensor, got {tensor.shape}/{tensor.dtype}"
        )
    rows, cols = (int(value) for value in tensor.shape)
    if rows <= 0 or cols <= 0 or cols % Q4_BLOCK_ELEMENTS:
        raise ValueError(
            f"H3 Q4 cache shape must have a positive 32-aligned width, got {tensor.shape}"
        )
    return rows, cols


def _signature_ranges(
    rows: int,
    ranges: Any = None,
) -> tuple[tuple[str, int, int], ...]:
    """Validate the token ranges used by a bounded input signature.

    H3's packed layout is passed as ``[start, stop, modulation_row]`` while
    the scalar-stat helper uses ``[label, start, stop]``.  Accept both forms,
    but require a contiguous full cover so no modality can silently disappear
    from the decision feature.
    """

    rows = int(rows)
    if rows <= 0:
        raise ValueError("H3 signature rows must be positive")
    if ranges is None:
        return (("packed", 0, rows),)
    normalized: list[tuple[str, int, int]] = []
    expected = 0
    for index, item in enumerate(ranges):
        if len(item) == 3 and isinstance(item[0], str):
            label, start, stop = str(item[0]), int(item[1]), int(item[2])
        elif len(item) >= 2:
            label = f"segment_{index}"
            start, stop = int(item[0]), int(item[1])
        else:
            raise ValueError(f"invalid H3 signature range {item!r}")
        if not label or not 0 <= start < stop <= rows or start != expected:
            raise ValueError(
                f"H3 signature ranges must be contiguous in [0, {rows}), got {item!r}"
            )
        normalized.append((label, start, stop))
        expected = stop
    if expected != rows:
        raise ValueError(
            f"H3 signature ranges cover [0, {expected}), expected [0, {rows})"
        )
    return tuple(normalized)


def _allocate_signature_counts(
    lengths: Sequence[int],
    max_tokens: int,
) -> tuple[int, ...]:
    """Allocate a deterministic token budget proportionally across ranges."""

    lengths = tuple(int(value) for value in lengths)
    max_tokens = int(max_tokens)
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("H3 signature ranges must all be non-empty")
    if max_tokens <= 0:
        raise ValueError("H3 signature max_tokens must be positive")
    total = min(sum(lengths), max_tokens)
    if total < len(lengths):
        # A useful signature must retain at least one sample from each packed
        # modality.  Failing closed is preferable to silently dropping audio
        # or video when a caller supplies an unrealistically tiny budget.
        raise ValueError(
            "H3 signature max_tokens must be at least the number of ranges: "
            f"{max_tokens} < {len(lengths)}"
        )
    counts = [1] * len(lengths)
    remaining = total - len(lengths)
    capacities = [length - 1 for length in lengths]
    capacity_total = sum(capacities)
    if remaining <= 0 or capacity_total <= 0:
        return tuple(counts)

    # Largest-remainder allocation keeps the result stable without a tensor or
    # numpy allocation.  The explicit capacity clamp handles very short media
    # segments cleanly.
    raw = [remaining * capacity / capacity_total for capacity in capacities]
    extras = [min(capacity, int(math.floor(value))) for capacity, value in zip(capacities, raw)]
    used = sum(extras)
    leftover = remaining - used
    order = sorted(
        range(len(lengths)),
        key=lambda index: (raw[index] - math.floor(raw[index]), capacities[index]),
        reverse=True,
    )
    while leftover > 0:
        progressed = False
        for index in order:
            if extras[index] < capacities[index]:
                extras[index] += 1
                leftover -= 1
                progressed = True
                if leftover == 0:
                    break
        if not progressed:
            break
    return tuple(count + extra for count, extra in zip(counts, extras))


def _evenly_spaced_indices(
    start: int,
    stop: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return integer indices including both ends when sampling a range."""

    start, stop, count = int(start), int(stop), int(count)
    length = stop - start
    if not 0 <= start < stop or not 1 <= count <= length:
        raise ValueError(
            f"invalid H3 signature index request [{start}, {stop})/{count}"
        )
    values = torch.arange(count, dtype=torch.long, device=device)
    if count == 1:
        return values.mul(0).add_((start + stop - 1) // 2)
    return start + (values * (length - 1)) // (count - 1)


@torch.inference_mode()
def deterministic_input_signature(
    tensor: torch.Tensor,
    *,
    max_tokens: int = DEFAULT_SIGNATURE_MAX_TOKENS,
    hidden_samples: int = DEFAULT_SIGNATURE_HIDDEN_SAMPLES,
    ranges: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a small, CPU-resident signature without copying the full input.

    The hidden projection is selected before the token gather.  At the 1 MP
    shape this bounds the largest temporary to roughly ``sequence ×
    hidden_samples`` (about 4.6 MiB for FP32 and 32 channels at the current
    1 MP sequence), rather than a
    sequence-sized FP32 clone.  Token samples are allocated proportionally per
    packed range so a large context prefix cannot hide changes in audio/video.
    """

    rows, cols = _validate_tensor(tensor)
    max_tokens = int(max_tokens)
    hidden_samples = int(hidden_samples)
    if max_tokens <= 0 or hidden_samples <= 0:
        raise ValueError("H3 signature max_tokens/hidden_samples must be positive")
    normalized_ranges = _signature_ranges(rows, ranges)
    counts = _allocate_signature_counts(
        [stop - start for _label, start, stop in normalized_ranges], max_tokens
    )
    hidden_count = min(cols, hidden_samples)
    hidden_indices = _evenly_spaced_indices(0, cols, hidden_count, tensor.device)
    token_parts = [
        _evenly_spaced_indices(start, stop, count, tensor.device)
        for (_label, start, stop), count in zip(normalized_ranges, counts)
    ]
    token_indices = torch.cat(token_parts, dim=0)

    # Selecting columns first is intentional: it caps the transient GPU
    # allocation at sequence * hidden_samples instead of max_tokens * hidden.
    selected_hidden = tensor.index_select(1, hidden_indices)
    sampled = selected_hidden.index_select(0, token_indices)
    cpu_signature = sampled.detach().to(device="cpu", dtype=torch.float32).contiguous()
    del selected_hidden, sampled, token_indices, hidden_indices, token_parts

    offset = 0
    range_report: list[dict[str, int | str]] = []
    for (label, start, stop), count in zip(normalized_ranges, counts):
        range_report.append(
            {
                "label": label,
                "token_start": start,
                "token_stop": stop,
                "sample_start": offset,
                "sample_stop": offset + count,
                "samples": count,
            }
        )
        offset += count
    metadata: dict[str, Any] = {
        "estimator": "stratified_uniform_hidden_sample",
        "shape": [rows, cols],
        "max_tokens": max_tokens,
        "hidden_samples": hidden_count,
        "token_samples": int(cpu_signature.shape[0]),
        "sample_elements": int(cpu_signature.numel()),
        "sample_bytes": int(cpu_signature.numel() * cpu_signature.element_size()),
        "ranges": range_report,
        "layout_key": tuple(
            (item["label"], item["token_start"], item["token_stop"], item["samples"])
            for item in range_report
        ),
    }
    return cpu_signature, metadata


def _signature_segment_error(
    current: torch.Tensor,
    reference: torch.Tensor,
    metric: str,
    epsilon: float,
) -> float:
    """Compute one small signature segment's normalized error."""

    delta = current - reference
    if metric == "relative_l1":
        numerator = float(delta.abs().sum(dtype=torch.float64).item())
        denominator = float(reference.abs().sum(dtype=torch.float64).item())
        count = max(1, int(reference.numel()))
        value = (numerator / count) / (denominator / count + epsilon)
    elif metric == "relative_l2":
        numerator = float(torch.sum(delta * delta, dtype=torch.float64).item())
        denominator = float(torch.sum(reference * reference, dtype=torch.float64).item())
        value = math.sqrt(numerator) / (math.sqrt(denominator) + epsilon)
    elif metric == "cosine":
        reference_sq = float(torch.sum(reference * reference, dtype=torch.float64).item())
        current_sq = float(torch.sum(current * current, dtype=torch.float64).item())
        dot = float(torch.sum(current * reference, dtype=torch.float64).item())
        cosine = dot / max(math.sqrt(reference_sq * current_sq), epsilon)
        value = 1.0 - max(-1.0, min(1.0, cosine))
    else:
        raise ValueError(f"unsupported H3 signature metric {metric!r}")
    del delta
    return float(value) if math.isfinite(value) else float("nan")


@torch.inference_mode()
def signature_difference(
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    metric: str,
    epsilon: float = 1e-6,
    current_metadata: dict[str, Any] | None = None,
    reference_metadata: dict[str, Any] | None = None,
    aggregation: str = "weighted",
) -> tuple[float, dict[str, Any]]:
    """Compare two bounded input signatures and expose per-range errors."""

    if current.ndim != 2 or reference.ndim != 2 or current.shape != reference.shape:
        raise ValueError(
            "H3 signature shape mismatch: "
            f"{tuple(current.shape)}/{tuple(reference.shape)}"
        )
    metric = str(metric).strip().lower()
    if metric not in {"relative_l1", "relative_l2", "cosine"}:
        raise ValueError(f"unsupported H3 signature metric {metric!r}")
    aggregation = str(aggregation).strip().lower()
    if aggregation not in {"weighted", "max_segment"}:
        raise ValueError(
            "H3 signature aggregation must be weighted or max_segment, "
            f"got {aggregation!r}"
        )
    if (current_metadata is None) != (reference_metadata is None):
        raise ValueError("H3 signature metadata is missing on one side of the comparison")
    if current_metadata is not None and reference_metadata is not None:
        if current_metadata.get("layout_key") != reference_metadata.get("layout_key"):
            raise ValueError("H3 signature sampling layout changed within one cache")
        if current_metadata.get("shape") != reference_metadata.get("shape"):
            raise ValueError("H3 signature source shape changed within one cache")
    ranges = (
        current_metadata.get("ranges")
        if current_metadata is not None
        else None
    )
    if not ranges:
        ranges = [
            {
                "sample_start": 0,
                "sample_stop": int(current.shape[0]),
                "label": "packed",
            }
        ]
    errors: list[dict[str, Any]] = []
    expected_start = 0
    weighted_num = 0.0
    weighted_den = 0.0
    weighted_count = 0
    for item in ranges:
        start = int(item["sample_start"])
        stop = int(item["sample_stop"])
        if (
            not 0 <= start < stop <= int(current.shape[0])
            or start != expected_start
        ):
            raise ValueError(f"invalid H3 signature sample range {item!r}")
        expected_start = stop
        current_part = current[start:stop]
        reference_part = reference[start:stop]
        value = _signature_segment_error(
            current_part, reference_part, metric, float(epsilon)
        )
        errors.append(
            {
                "label": str(item.get("label", "segment")),
                "sample_start": start,
                "sample_stop": stop,
                "samples": stop - start,
                "error": value,
            }
        )
        # Reconstruct a weighted aggregate in the same metric domain.  For
        # cosine, the segment error is already bounded and sample weighting is
        # the least surprising approximation for a tiny signature.
        if metric == "relative_l1":
            delta = current_part - reference_part
            weighted_num += float(delta.abs().sum(dtype=torch.float64).item())
            weighted_den += float(reference_part.abs().sum(dtype=torch.float64).item())
            weighted_count += int(reference_part.numel())
            del delta
        elif metric == "relative_l2":
            delta = current_part - reference_part
            weighted_num += float(torch.sum(delta * delta, dtype=torch.float64).item())
            weighted_den += float(torch.sum(reference_part * reference_part, dtype=torch.float64).item())
            del delta
        else:
            weighted_count += int(reference_part.numel())
        del current_part, reference_part
    if expected_start != int(current.shape[0]):
        raise ValueError(
            "H3 signature sample ranges do not cover the complete signature"
        )

    if metric == "relative_l1":
        error = (weighted_num / max(1, weighted_count)) / (
            weighted_den / max(1, weighted_count) + float(epsilon)
        )
    elif metric == "relative_l2":
        error = math.sqrt(weighted_num) / (math.sqrt(weighted_den) + float(epsilon))
    else:
        # Compute cosine over the complete compact signature rather than
        # averaging per-range angles.
        error = _signature_segment_error(
            current, reference, "cosine", float(epsilon)
        )
    finite_values = [item["error"] for item in errors] + [error]
    finite = all(math.isfinite(float(value)) for value in finite_values)
    if aggregation == "max_segment":
        finite_errors = [float(item["error"]) for item in errors if math.isfinite(float(item["error"]))]
        error = max(finite_errors) if finite_errors else float("nan")
    return float(error), {
        "available": finite,
        "metric": metric,
        "metric_domain": "fp32_bounded_signature",
        "aggregation": aggregation,
        "epsilon": float(epsilon),
        "signature_shape": [int(value) for value in current.shape],
        "sample_elements": int(current.numel()),
        "segment_errors": errors,
        "max_segment_error": (
            max(float(item["error"]) for item in errors)
            if errors and all(math.isfinite(float(item["error"])) for item in errors)
            else float("nan")
        ),
        "finite": finite and math.isfinite(float(error)),
    }


@dataclass
class Q4Tensor:
    raw: torch.Tensor
    rows: int
    cols: int
    policy: str
    source_dtype: str
    restore_scale_exponent: int = 0
    quantize_report: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    @property
    def row_bytes(self) -> int:
        return self.cols // Q4_BLOCK_ELEMENTS * Q4_BLOCK_BYTES

    @property
    def bytes(self) -> int:
        return int(self.raw.numel() * self.raw.element_size())

    @property
    def device(self) -> torch.device:
        return self.raw.device

    def validate(self) -> None:
        expected_shape = (self.rows, self.row_bytes)
        if self.raw.dtype != torch.uint8 or tuple(self.raw.shape) != expected_shape:
            raise ValueError(
                "H3 Q4 cache raw storage mismatch: "
                f"got {self.raw.dtype}/{tuple(self.raw.shape)}, "
                f"expected uint8/{expected_shape}"
            )
        if not 0 <= int(self.restore_scale_exponent) <= 126:
            raise ValueError(
                "H3 Q4 cache restore exponent must be in [0, 126], got "
                f"{self.restore_scale_exponent}"
            )

    @torch.inference_mode()
    def iter_dequantized(
        self,
        device: torch.device | str,
        *,
        chunk_rows: int,
        dtype: torch.dtype = torch.float32,
    ) -> Iterator[tuple[int, int, torch.Tensor, int]]:
        self.validate()
        if chunk_rows <= 0:
            raise ValueError("H3 Q4 cache chunk_rows must be positive")
        target = torch.device(device)
        for start in range(0, self.rows, chunk_rows):
            stop = min(start + chunk_rows, self.rows)
            raw = self.raw[start:stop]
            transfer_bytes = 0
            if raw.device != target:
                raw = raw.to(target)
                transfer_bytes = int(raw.numel())
            shard = q4_tp.Q4MatrixShard(
                raw=raw,
                out_features=stop - start,
                in_features=self.cols,
                source_name="h3_q4_cache",
                kind="full",
                rank=None,
            )
            values = q4_tp.dequantize_q4_0(shard, dtype=dtype)
            if self.restore_scale_exponent:
                values.mul_(math.ldexp(1.0, int(self.restore_scale_exponent)))
            yield start, stop, values, transfer_bytes
            del values, shard
            if raw.data_ptr() != self.raw[start:stop].data_ptr():
                del raw


@torch.inference_mode()
def quantize_q4_0(
    tensor: torch.Tensor,
    *,
    policy: str = "cpu",
    chunk_rows: int = DEFAULT_CACHE_CHUNK_ROWS,
    measure: bool = False,
) -> Q4Tensor:
    """Quantize a 2D tensor to a bounded standard GGML Q4_0 payload.

    Q4_0 stores each block scale as FP16. H3's FP32 residual stream can exceed
    the corresponding finite source range (65504 * 8), especially in later
    groups. A tensor-wide power-of-two pre-scale keeps every Q4_0 block
    standard and finite; the integer exponent is restored while decoding.
    """

    rows, cols = _validate_tensor(tensor)
    if chunk_rows <= 0:
        raise ValueError("H3 Q4 cache chunk_rows must be positive")
    normalized_policy = _normalize_policy(policy)
    if measure and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    started = time.perf_counter()
    source_max_abs = float(
        torch.linalg.vector_norm(tensor.reshape(-1), ord=float("inf")).item()
    )
    if not math.isfinite(source_max_abs):
        raise ValueError("H3 Q4 cache cannot quantize a tensor containing NaN/Inf")
    restore_scale_exponent = 0
    if source_max_abs > Q4_MAX_FINITE_SOURCE_ABS:
        restore_scale_exponent = int(
            math.ceil(math.log2(source_max_abs / Q4_MAX_FINITE_SOURCE_ABS))
        )
    scaled_max_abs = math.ldexp(source_max_abs, -restore_scale_exponent)
    while scaled_max_abs > Q4_MAX_FINITE_SOURCE_ABS:
        restore_scale_exponent += 1
        scaled_max_abs *= 0.5
    if restore_scale_exponent > 126:
        raise ValueError(
            "H3 Q4 cache source range requires an unsupported restore exponent: "
            f"max_abs={source_max_abs}, exponent={restore_scale_exponent}"
        )
    encode_multiplier = math.ldexp(1.0, -restore_scale_exponent)

    storage_device = tensor.device if normalized_policy == "gpu" else torch.device("cpu")
    row_bytes = cols // Q4_BLOCK_ELEMENTS * Q4_BLOCK_BYTES
    raw = torch.empty((rows, row_bytes), dtype=torch.uint8, device=storage_device)
    copied_bytes = 0
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        if restore_scale_exponent:
            source = tensor[start:stop].mul(encode_multiplier)
        else:
            source = tensor[start:stop].contiguous()
        blocks = source.reshape(-1, Q4_BLOCK_ELEMENTS)
        absolute = blocks.abs()
        imax = absolute.argmax(dim=1, keepdim=True)
        maximum = blocks.gather(1, imax)
        scale = maximum.mul(-0.125)
        inverse = torch.where(scale == 0, torch.zeros_like(scale), scale.reciprocal())
        quants = torch.trunc(blocks * inverse + 8.5).clamp_(0, 15).to(torch.uint8)
        packed = quants[:, :16] | (quants[:, 16:] << 4)
        scale_bytes = scale.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, 2)
        encoded = torch.empty(
            (blocks.shape[0], Q4_BLOCK_BYTES),
            dtype=torch.uint8,
            device=tensor.device,
        )
        encoded[:, :2].copy_(scale_bytes)
        encoded[:, 2:].copy_(packed)
        encoded_rows = encoded.reshape(stop - start, row_bytes)
        raw[start:stop].copy_(encoded_rows)
        if raw.device.type == "cpu":
            copied_bytes += int(encoded_rows.numel())
        del source, blocks, absolute, imax, maximum, scale, inverse
        del quants, packed, scale_bytes, encoded, encoded_rows

    if measure and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = Q4Tensor(
        raw=raw,
        rows=rows,
        cols=cols,
        policy=normalized_policy,
        source_dtype=str(tensor.dtype),
        restore_scale_exponent=restore_scale_exponent,
        quantize_report={
            "format": Q4_FORMAT,
            "measured": bool(measure),
            "quantize_ms": elapsed_ms if measure else None,
            "d2h_bytes": copied_bytes,
            "stored_bytes": int(raw.numel()),
            "source_bytes": int(tensor.numel() * tensor.element_size()),
            "compression_ratio": float(
                tensor.numel() * tensor.element_size() / max(1, raw.numel())
            ),
            "policy": normalized_policy,
            "chunk_rows": int(chunk_rows),
            "source_max_abs": source_max_abs,
            "scaled_max_abs": scaled_max_abs,
            "restore_scale_exponent": restore_scale_exponent,
            "restore_scale": math.ldexp(1.0, restore_scale_exponent),
        },
    )
    result.validate()
    return result


@torch.inference_mode()
def add_q4_to_(
    output: torch.Tensor,
    cached: Q4Tensor,
    *,
    chunk_rows: int = DEFAULT_CACHE_CHUNK_ROWS,
    measure: bool = False,
) -> dict[str, Any]:
    rows, cols = _validate_tensor(output)
    if cached.shape != (rows, cols):
        raise ValueError(
            f"H3 Q4 residual shape mismatch: cache={cached.shape}, output={(rows, cols)}"
        )
    if measure and output.is_cuda:
        torch.cuda.synchronize(output.device)
    started = time.perf_counter()
    transfer_bytes = 0
    for start, stop, values, copied in cached.iter_dequantized(
        output.device, chunk_rows=chunk_rows, dtype=output.dtype
    ):
        output[start:stop].add_(values)
        transfer_bytes += copied
        del values
    if measure and output.is_cuda:
        torch.cuda.synchronize(output.device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "kind": "q4_cache_add",
        "format": Q4_FORMAT,
        "measured": bool(measure),
        "total_ms": elapsed_ms if measure else None,
        "h2d_bytes": transfer_bytes,
        "cache_bytes": cached.bytes,
        "cache_device": cached.policy,
        "chunk_rows": int(chunk_rows),
        "restore_scale_exponent": int(cached.restore_scale_exponent),
    }


@torch.inference_mode()
def relative_difference(
    current: Q4Tensor,
    previous: Q4Tensor,
    *,
    metric: str,
    device: torch.device | str,
    chunk_rows: int = DEFAULT_CACHE_CHUNK_ROWS,
    epsilon: float = 1e-6,
    measure: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Compare two Q4 tensors after bounded dequantization on one rank."""

    if current.shape != previous.shape:
        raise ValueError(
            f"H3 Q4 feature shape mismatch: {current.shape} vs {previous.shape}"
        )
    metric = str(metric).lower()
    if metric not in {"relative_l1", "relative_l2", "cosine"}:
        raise ValueError(f"unsupported H3 group-cache metric {metric!r}")
    target = torch.device(device)
    if measure and target.type == "cuda":
        torch.cuda.synchronize(target)
    started = time.perf_counter()
    # Keep reductions on the accelerator for the whole metric.  Calling
    # ``.item()`` once per row chunk would force hundreds of host/device
    # synchronizations at 1 MP and can erase the block-skipping benefit.
    accum = torch.zeros(4, dtype=torch.float64, device=target)
    transfer_bytes = 0
    current_iter = current.iter_dequantized(
        target, chunk_rows=chunk_rows, dtype=torch.float32
    )
    previous_iter = previous.iter_dequantized(
        target, chunk_rows=chunk_rows, dtype=torch.float32
    )
    for current_item, previous_item in zip(current_iter, previous_iter, strict=True):
        start, stop, current_values, current_copied = current_item
        pstart, pstop, previous_values, previous_copied = previous_item
        if (start, stop) != (pstart, pstop):
            raise RuntimeError("H3 Q4 feature chunk iteration diverged")
        delta = current_values - previous_values
        if metric == "relative_l1":
            accum[0].add_(delta.abs().sum(dtype=torch.float64))
            accum[1].add_(previous_values.abs().sum(dtype=torch.float64))
        elif metric == "relative_l2":
            accum[0].add_(torch.sum(delta * delta, dtype=torch.float64))
            accum[1].add_(
                torch.sum(previous_values * previous_values, dtype=torch.float64)
            )
        else:
            accum[0].add_(
                torch.sum(previous_values * previous_values, dtype=torch.float64)
            )
            accum[1].add_(
                torch.sum(current_values * current_values, dtype=torch.float64)
            )
            accum[2].add_(
                torch.sum(current_values * previous_values, dtype=torch.float64)
            )
        transfer_bytes += current_copied + previous_copied
        del current_values, previous_values, delta

    reduced = accum.cpu()
    count = max(1, current.rows * current.cols)
    if metric == "relative_l1":
        delta_l1, previous_l1 = (float(value) for value in reduced[:2])
        error = (delta_l1 / count) / (previous_l1 / count + epsilon)
    elif metric == "relative_l2":
        delta_sq, previous_sq = (float(value) for value in reduced[:2])
        error = math.sqrt(delta_sq) / (math.sqrt(previous_sq) + epsilon)
    else:
        previous_sq, current_sq, dot = (float(value) for value in reduced[:3])
        cosine = dot / max(math.sqrt(previous_sq * current_sq), epsilon)
        error = 1.0 - max(-1.0, min(1.0, cosine))
    if measure and target.type == "cuda":
        torch.cuda.synchronize(target)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return float(error), {
        "metric": metric,
        "metric_domain": "q4_dequantized_pair",
        "epsilon": float(epsilon),
        "measured": bool(measure),
        "metric_ms": elapsed_ms if measure else None,
        "h2d_bytes": transfer_bytes,
        "finite": bool(math.isfinite(error)),
    }


@torch.inference_mode()
def q4_tensor_error(
    cached: Q4Tensor,
    reference: torch.Tensor,
    *,
    chunk_rows: int = DEFAULT_CACHE_CHUNK_ROWS,
) -> dict[str, float | bool]:
    """Measure Q4 reconstruction error without a full dequantized tensor."""

    rows, cols = _validate_tensor(reference)
    if cached.shape != (rows, cols):
        raise ValueError(
            f"H3 Q4/reference shape mismatch: {cached.shape} vs {(rows, cols)}"
        )
    delta_l1 = reference_l1 = delta_sq = reference_sq = got_sq = dot = 0.0
    max_abs = 0.0
    finite = True
    for start, stop, values, _copied in cached.iter_dequantized(
        reference.device, chunk_rows=chunk_rows, dtype=torch.float32
    ):
        wanted = reference[start:stop].float()
        delta = values - wanted
        delta_l1 += float(delta.abs().sum(dtype=torch.float64).item())
        reference_l1 += float(wanted.abs().sum(dtype=torch.float64).item())
        delta_sq += float(torch.sum(delta * delta, dtype=torch.float64).item())
        reference_sq += float(torch.sum(wanted * wanted, dtype=torch.float64).item())
        got_sq += float(torch.sum(values * values, dtype=torch.float64).item())
        dot += float(torch.sum(wanted * values, dtype=torch.float64).item())
        max_abs = max(
            max_abs,
            float(torch.linalg.vector_norm(delta, ord=float("inf")).item()),
        )
        finite = finite and math.isfinite(max_abs)
        del values, wanted, delta
    count = max(1, rows * cols)
    rms = math.sqrt(delta_sq / count)
    reference_rms = math.sqrt(reference_sq / count)
    return {
        "delta_l2": math.sqrt(delta_sq),
        "reference_l2": math.sqrt(reference_sq),
        "mean_relative_l1": (delta_l1 / count)
        / (reference_l1 / count + 1e-30),
        "rms": rms,
        "relative_l2": math.sqrt(delta_sq) / max(math.sqrt(reference_sq), 1e-30),
        "relative_rms": rms / max(reference_rms, 1e-30),
        "cosine": dot / max(math.sqrt(reference_sq * got_sq), 1e-30),
        "max_abs": max_abs,
        "finite": finite,
    }


@torch.inference_mode()
def cached_residual_ground_truth_error(
    cached_residual: Q4Tensor,
    true_residual: torch.Tensor,
    true_output: torch.Tensor,
    *,
    chunk_rows: int = DEFAULT_CACHE_CHUNK_ROWS,
    epsilon: float = 1e-6,
) -> dict[str, float | bool]:
    report = q4_tensor_error(
        cached_residual,
        true_residual,
        chunk_rows=chunk_rows,
    )
    true_output_l2 = float(torch.linalg.vector_norm(true_output.reshape(-1)).item())
    report["output_relative_l2"] = float(report["delta_l2"]) / (
        true_output_l2 + epsilon
    )
    report["residual_relative_l2"] = float(report["relative_l2"])
    return report


@torch.inference_mode()
def cached_group_ground_truth_error(
    predicted: torch.Tensor,
    true_output: torch.Tensor,
    cached_residual: Q4Tensor,
    *,
    chunk_rows: int = DEFAULT_CACHE_CHUNK_ROWS,
    epsilon: float = 1e-6,
) -> dict[str, float | bool]:
    """Compare a cached group output and residual with the true group run.

    The original group input need not be retained.  Since
    ``predicted = original_input + cached_residual``, the true residual is
    ``true_output - predicted + cached_residual``.
    """

    if predicted.shape != true_output.shape or cached_residual.shape != tuple(predicted.shape):
        raise ValueError(
            "H3 group ground-truth shape mismatch: "
            f"predicted={tuple(predicted.shape)}, true={tuple(true_output.shape)}, "
            f"cache={cached_residual.shape}"
        )
    output_delta_sq = true_output_sq = true_residual_sq = 0.0
    max_abs = 0.0
    finite = True
    for start, stop, cached_values, _copied in cached_residual.iter_dequantized(
        predicted.device, chunk_rows=chunk_rows, dtype=torch.float32
    ):
        got = predicted[start:stop].float()
        wanted = true_output[start:stop].float()
        delta = got - wanted
        true_residual = wanted - got + cached_values
        output_delta_sq += float(torch.sum(delta * delta, dtype=torch.float64).item())
        true_output_sq += float(torch.sum(wanted * wanted, dtype=torch.float64).item())
        true_residual_sq += float(
            torch.sum(true_residual * true_residual, dtype=torch.float64).item()
        )
        max_abs = max(
            max_abs,
            float(torch.linalg.vector_norm(delta, ord=float("inf")).item()),
        )
        finite = finite and math.isfinite(max_abs)
        del cached_values, got, wanted, delta, true_residual
    numerator = math.sqrt(output_delta_sq)
    return {
        "output_relative_l2": numerator / (math.sqrt(true_output_sq) + epsilon),
        "residual_relative_l2": numerator / (math.sqrt(true_residual_sq) + epsilon),
        "max_abs": max_abs,
        "finite": finite,
    }


@torch.inference_mode()
def tensor_error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_rows: int = DEFAULT_CACHE_CHUNK_ROWS,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Full-output and token error statistics with bounded FP32 temporaries."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError(
            f"expected matching 2D tensors, got {reference.shape}/{candidate.shape}"
        )
    delta_l1 = reference_l1 = delta_sq = reference_sq = candidate_sq = dot = 0.0
    max_abs = 0.0
    token_errors: list[torch.Tensor] = []
    finite = True
    for start in range(0, int(reference.shape[0]), chunk_rows):
        stop = min(start + chunk_rows, int(reference.shape[0]))
        wanted = reference[start:stop].float()
        got = candidate[start:stop].float()
        delta = got - wanted
        delta_l1 += float(delta.abs().sum(dtype=torch.float64).item())
        reference_l1 += float(wanted.abs().sum(dtype=torch.float64).item())
        delta_sq += float(torch.sum(delta * delta, dtype=torch.float64).item())
        reference_sq += float(torch.sum(wanted * wanted, dtype=torch.float64).item())
        candidate_sq += float(torch.sum(got * got, dtype=torch.float64).item())
        dot += float(torch.sum(wanted * got, dtype=torch.float64).item())
        max_abs = max(
            max_abs,
            float(torch.linalg.vector_norm(delta, ord=float("inf")).item()),
        )
        per_token = torch.linalg.vector_norm(delta, dim=1) / (
            torch.linalg.vector_norm(wanted, dim=1) + epsilon
        )
        token_errors.append(per_token.float().cpu())
        finite = finite and math.isfinite(max_abs)
        del wanted, got, delta, per_token
    tokens = torch.cat(token_errors) if token_errors else torch.empty(0)
    count = max(1, int(reference.numel()))
    quantile_levels = torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float32)
    quantiles = (
        torch.quantile(tokens, quantile_levels)
        if tokens.numel()
        else torch.full((4,), float("nan"))
    )
    return {
        "mean_relative_l1": (delta_l1 / count)
        / (reference_l1 / count + epsilon),
        "relative_l2": math.sqrt(delta_sq) / (math.sqrt(reference_sq) + epsilon),
        "cosine": dot / max(math.sqrt(reference_sq * candidate_sq), epsilon),
        "max_abs": max_abs,
        "token_error": {
            "mean": float(tokens.mean().item()) if tokens.numel() else None,
            "median": float(quantiles[0].item()),
            "p90": float(quantiles[1].item()),
            "p95": float(quantiles[2].item()),
            "p99": float(quantiles[3].item()),
            "max": float(tokens.max().item()) if tokens.numel() else None,
        },
        "finite": finite,
    }


@dataclass
class GroupCacheEntry:
    group_id: int
    start_block: int
    end_block: int
    previous_input: Q4Tensor | None = None
    residual: Q4Tensor | None = None
    # ``input_signature`` is the optional bounded replacement for
    # ``previous_input``.  It is a CPU FP32 sample used only by the feature
    # gate; the generated stream always uses ``residual`` above.
    input_signature: torch.Tensor | None = None
    input_signature_metadata: dict[str, Any] | None = None
    # A six-by-hidden AdaLN summary.  It is only populated by the opt-in
    # calibration collector and deliberately lives on CPU: unlike an input
    # activation it is independent of sequence length (a few hundred KiB per
    # group with the bounded channel sample).
    condition_signature: torch.Tensor | None = None
    condition_segments: tuple[tuple[int, int, int], ...] | None = None
    residual_q_floor: float | None = None
    cache_count: int = 0
    full_count: int = 0
    hit_count: int = 0

    @property
    def ready(self) -> bool:
        return (
            self.residual is not None
            and (self.previous_input is not None or self.input_signature is not None)
        )

    @property
    def bytes(self) -> int:
        return sum(
            value.bytes
            for value in (self.previous_input, self.residual)
            if value is not None
        ) + (
            0
            if self.condition_signature is None
            else int(
                self.condition_signature.numel()
                * self.condition_signature.element_size()
            )
        ) + (
            0
            if self.input_signature is None
            else int(
                self.input_signature.numel() * self.input_signature.element_size()
            )
        )

    def clear(self) -> None:
        previous_input = self.previous_input
        residual = self.residual
        input_signature = self.input_signature
        input_signature_metadata = self.input_signature_metadata
        condition_signature = self.condition_signature
        condition_segments = self.condition_segments
        self.previous_input = None
        self.residual = None
        self.input_signature = None
        self.input_signature_metadata = None
        self.condition_signature = None
        self.condition_segments = None
        self.residual_q_floor = None
        self.cache_count = 0
        self.full_count = 0
        self.hit_count = 0
        del (
            previous_input,
            residual,
            input_signature,
            input_signature_metadata,
            condition_signature,
            condition_segments,
        )


class GroupResidualCache:
    def __init__(self, policy: str = "cpu") -> None:
        self.policy = _normalize_policy(policy)
        self.feature_mode = "q4"
        self.signature_max_tokens = DEFAULT_SIGNATURE_MAX_TOKENS
        self.signature_hidden_samples = DEFAULT_SIGNATURE_HIDDEN_SAMPLES
        self.config_key: tuple[Any, ...] | None = None
        self.entries: list[GroupCacheEntry] = []

    @staticmethod
    def partition(warm_blocks: int, num_groups: int, block_count: int) -> list[tuple[int, int]]:
        warm_blocks = int(warm_blocks)
        num_groups = int(num_groups)
        block_count = int(block_count)
        remaining = block_count - warm_blocks
        if not 0 <= warm_blocks < block_count:
            raise ValueError(
                f"warm_blocks must be in [0, {block_count - 1}], got {warm_blocks}"
            )
        if not 1 <= num_groups <= remaining:
            raise ValueError(
                f"num_groups must be in [1, {remaining}], got {num_groups}"
            )
        base = remaining // num_groups
        boundaries = []
        start = warm_blocks
        for group_id in range(num_groups):
            stop = block_count if group_id == num_groups - 1 else start + base
            boundaries.append((start, stop))
            start = stop
        return boundaries

    def configure(
        self,
        *,
        warm_blocks: int,
        num_groups: int,
        block_count: int,
        policy: str,
        shape: tuple[int, int],
        feature_mode: str = "q4",
        signature_max_tokens: int = DEFAULT_SIGNATURE_MAX_TOKENS,
        signature_hidden_samples: int = DEFAULT_SIGNATURE_HIDDEN_SAMPLES,
    ) -> bool:
        normalized_policy = _normalize_policy(policy)
        normalized_feature_mode = normalize_feature_mode(feature_mode)
        signature_max_tokens = int(signature_max_tokens)
        signature_hidden_samples = int(signature_hidden_samples)
        if signature_max_tokens <= 0 or signature_hidden_samples <= 0:
            raise ValueError(
                "H3 group signature max_tokens/hidden_samples must be positive"
            )
        ranges = self.partition(warm_blocks, num_groups, block_count)
        key = (
            normalized_policy,
            tuple(shape),
            tuple(ranges),
            normalized_feature_mode,
            signature_max_tokens,
            signature_hidden_samples,
        )
        changed = key != self.config_key
        if changed:
            self.clear()
            self.policy = normalized_policy
            self.feature_mode = normalized_feature_mode
            self.signature_max_tokens = signature_max_tokens
            self.signature_hidden_samples = signature_hidden_samples
            self.config_key = key
            self.entries = [
                GroupCacheEntry(group_id=index, start_block=start, end_block=stop)
                for index, (start, stop) in enumerate(ranges)
            ]
        return changed

    def clear(self) -> None:
        entries = self.entries
        self.entries = []
        self.config_key = None
        for entry in entries:
            entry.clear()
        del entries

    @property
    def ready(self) -> bool:
        return bool(self.entries) and all(entry.ready for entry in self.entries)

    @property
    def bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries)

    def summary(self) -> dict[str, Any]:
        return {
            "format": Q4_FORMAT,
            "policy": self.policy,
            "feature_mode": self.feature_mode,
            "signature_max_tokens": self.signature_max_tokens,
            "signature_hidden_samples": self.signature_hidden_samples,
            "bytes": self.bytes,
            "ready": self.ready,
            "groups": [
                {
                    "group_id": entry.group_id,
                    "start_block": entry.start_block,
                    "end_block": entry.end_block,
                    "ready": entry.ready,
                    "cache_count": entry.cache_count,
                    "full_count": entry.full_count,
                    "hit_count": entry.hit_count,
                    "bytes": entry.bytes,
                    "condition_signature_bytes": (
                        0
                        if entry.condition_signature is None
                        else int(
                            entry.condition_signature.numel()
                            * entry.condition_signature.element_size()
                        )
                    ),
                    "condition_signature_shape": (
                        None
                        if entry.condition_signature is None
                        else [int(value) for value in entry.condition_signature.shape]
                    ),
                    "condition_anchor_available": (
                        entry.condition_signature is not None
                        and entry.condition_segments is not None
                    ),
                    "input_signature_bytes": (
                        0
                        if entry.input_signature is None
                        else int(
                            entry.input_signature.numel()
                            * entry.input_signature.element_size()
                        )
                    ),
                    "input_signature_shape": (
                        None
                        if entry.input_signature is None
                        else [int(value) for value in entry.input_signature.shape]
                    ),
                    "input_signature_estimator": (
                        None
                        if entry.input_signature_metadata is None
                        else entry.input_signature_metadata.get("estimator")
                    ),
                    "residual_q_floor": entry.residual_q_floor,
                }
                for entry in self.entries
            ],
        }


__all__ = [
    "DEFAULT_CACHE_CHUNK_ROWS",
    "DEFAULT_GROUP_THRESHOLD",
    "DEFAULT_SIGNATURE_HIDDEN_SAMPLES",
    "DEFAULT_SIGNATURE_MAX_TOKENS",
    "GroupCacheEntry",
    "GroupResidualCache",
    "Q4_FORMAT",
    "Q4Tensor",
    "add_q4_to_",
    "cached_group_ground_truth_error",
    "cached_residual_ground_truth_error",
    "normalize_q4_format",
    "normalize_feature_mode",
    "quantize_q4_0",
    "q4_tensor_error",
    "relative_difference",
    "deterministic_input_signature",
    "signature_difference",
    "tensor_error_metrics",
]
