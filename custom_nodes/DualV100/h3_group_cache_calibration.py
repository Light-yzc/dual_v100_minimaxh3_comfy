"""Bounded conditioning features for the opt-in H3 group-cache calibration.

The production cache stores a Q4 residual.  This module deliberately keeps
the additional calibration state independent of sequence length: a fixed
hidden-channel sample of AdaLN modulation vectors, stored on CPU.  It is
importable by the rank-1 worker without importing ComfyUI.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch


# ``H3TPBackbone._adaln`` returns this order after chunking the six modulation
# vectors.  Keep the names and indices in one place so the logger and the
# offline fitter agree about what a feature means.
CONDITION_COMPONENTS = (
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
)
GATE_INDICES = (2, 5)
AFFINE_INDICES = (0, 1, 3, 4)
ALL_INDICES = tuple(range(len(CONDITION_COMPONENTS)))
CONDITION_SIGNATURE_HIDDEN_SAMPLES = 256


def _normalized_segments(
    sequence: int,
    segments: Sequence[Sequence[int]],
    row_count: int,
) -> tuple[tuple[int, int, int], ...]:
    """Validate a packed row map and retain its tiny canonical description."""

    sequence = int(sequence)
    row_count = int(row_count)
    if sequence <= 0 or row_count <= 0:
        raise ValueError("H3 calibration sequence and row_count must be positive")
    covered = 0
    expected_start = 0
    normalized: list[tuple[int, int, int]] = []
    for segment in segments:
        if len(segment) != 3:
            raise ValueError(f"invalid H3 calibration segment: {segment!r}")
        start, stop, row = (int(value) for value in segment)
        if not 0 <= start < stop <= sequence:
            raise ValueError(
                f"invalid H3 calibration segment {segment!r} for S={sequence}"
            )
        if start != expected_start:
            raise ValueError(
                "H3 calibration segments must be contiguous from token zero; "
                f"expected start {expected_start}, got {start}"
            )
        if not 0 <= row < row_count:
            raise ValueError(
                f"H3 calibration row {row} is outside [0, {row_count})"
            )
        covered += stop - start
        expected_start = stop
        normalized.append((start, stop, row))
    if covered != sequence:
        raise ValueError(
            f"H3 calibration segments cover {covered} tokens, expected {sequence}"
        )
    return tuple(normalized)


@torch.inference_mode()
def sampled_hidden_indices(
    hidden: int,
    *,
    max_samples: int = CONDITION_SIGNATURE_HIDDEN_SAMPLES,
    device: torch.device | str,
) -> torch.Tensor:
    """Select deterministic evenly-spaced hidden channels for a signature."""

    hidden = int(hidden)
    max_samples = int(max_samples)
    if hidden <= 0 or max_samples <= 0:
        raise ValueError("H3 calibration hidden/sample count must be positive")
    sample_count = min(hidden, max_samples)
    # Integer arithmetic avoids a floating linspace round-trip and gives
    # stable samples on both NCCL ranks.
    values = torch.arange(sample_count, dtype=torch.long, device=device)
    return (values * (hidden - 1)) // max(1, sample_count - 1)


@torch.inference_mode()
def sampled_modulation_signature(
    modulation: Sequence[torch.Tensor],
    hidden_indices: torch.Tensor,
) -> torch.Tensor:
    """Take a bounded [component, AdaLN-row, hidden-sample] signature."""

    if len(modulation) != len(CONDITION_COMPONENTS):
        raise ValueError(
            "H3 calibration expects six AdaLN modulation components, got "
            f"{len(modulation)}"
        )
    if hidden_indices.ndim != 1 or hidden_indices.dtype != torch.long:
        raise ValueError("H3 calibration hidden indices must be a 1-D long tensor")
    outputs: list[torch.Tensor] = []
    row_count: int | None = None
    for value in modulation:
        if value.ndim != 2:
            raise ValueError(
                "H3 calibration modulation must be 2-D, got "
                f"{tuple(value.shape)}"
            )
        if row_count is None:
            row_count = int(value.shape[0])
        elif int(value.shape[0]) != row_count:
            raise ValueError("H3 calibration modulation rows disagree between components")
        if value.device != hidden_indices.device:
            raise ValueError("H3 calibration modulation/hidden-index device mismatch")
        outputs.append(value.float().index_select(1, hidden_indices))
    return torch.stack(outputs, dim=0).contiguous()


def _relative_l2(
    current: torch.Tensor,
    reference: torch.Tensor,
    indices: Sequence[int],
    current_segments: Sequence[Sequence[int]],
    reference_segments: Sequence[Sequence[int]],
    epsilon: float,
) -> float:
    # Weighted by token counts, but compare rows via the fixed packed-segment
    # boundaries rather than raw row IDs.  ComfyUI may renumber AdaLN rows
    # between forwards while retaining the same token layout.
    numerator_sq = 0.0
    denominator_sq = 0.0
    for current_segment, reference_segment in zip(
        current_segments, reference_segments, strict=True
    ):
        start, stop, current_row = (int(value) for value in current_segment)
        reference_start, reference_stop, reference_row = (
            int(value) for value in reference_segment
        )
        if (start, stop) != (reference_start, reference_stop):
            raise ValueError(
                "H3 calibration signatures have different packed segment boundaries"
            )
        if not 0 <= current_row < current.shape[-2]:
            raise ValueError("H3 calibration current AdaLN row is out of range")
        if not 0 <= reference_row < reference.shape[-2]:
            raise ValueError("H3 calibration reference AdaLN row is out of range")
        current_values = current[:, list(indices), current_row, :].float()
        reference_values = reference[:, list(indices), reference_row, :].float()
        delta = current_values - reference_values
        token_weight = float(stop - start)
        numerator_sq += token_weight * float(
            torch.sum(delta * delta, dtype=torch.float64).item()
        )
        denominator_sq += token_weight * float(
            torch.sum(reference_values * reference_values, dtype=torch.float64).item()
        )
        del current_values, reference_values, delta
    value = math.sqrt(numerator_sq) / (math.sqrt(denominator_sq) + float(epsilon))
    return float(value) if math.isfinite(value) else float("nan")


@torch.inference_mode()
def signature_difference(
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    current_segments: Sequence[Sequence[int]],
    reference_segments: Sequence[Sequence[int]],
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Measure gate, affine, and combined AdaLN signature changes."""

    if (
        current.ndim != 4
        or reference.ndim != 4
        or current.shape[0] != reference.shape[0]
        or current.shape[1] != reference.shape[1]
        or current.shape[-1] != reference.shape[-1]
    ):
        raise ValueError(
            "H3 calibration signatures must be matching 4-D tensors, got "
            f"{tuple(current.shape)}/{tuple(reference.shape)}"
        )
    if current.shape[1] != len(CONDITION_COMPONENTS):
        raise ValueError("H3 calibration signature has an unexpected component count")
    current_layout = _normalized_segments(
        int(sum(int(stop) - int(start) for start, stop, _row in current_segments)),
        current_segments,
        int(current.shape[-2]),
    )
    reference_layout = _normalized_segments(
        int(sum(int(stop) - int(start) for start, stop, _row in reference_segments)),
        reference_segments,
        int(reference.shape[-2]),
    )
    if len(current_layout) != len(reference_layout):
        raise ValueError("H3 calibration segment count changed within one cache")
    current = current.float()
    reference = reference.float()
    values = {
        "gate_relative_l2": _relative_l2(
            current, reference, GATE_INDICES, current_layout, reference_layout, epsilon
        ),
        "affine_relative_l2": _relative_l2(
            current,
            reference,
            AFFINE_INDICES,
            current_layout,
            reference_layout,
            epsilon,
        ),
        "all_relative_l2": _relative_l2(
            current, reference, ALL_INDICES, current_layout, reference_layout, epsilon
        ),
    }
    values.update(
        {
            "available": all(math.isfinite(float(value)) for value in values.values()),
            "epsilon": float(epsilon),
            "components": list(CONDITION_COMPONENTS),
            "signature_shape": [int(value) for value in current.shape],
            "metric": "token_weighted_relative_l2",
            "hidden_samples": int(current.shape[-1]),
            "segment_count": len(current_layout),
            "row_id_invariant": True,
        }
    )
    return values


def selected_condition_error(
    report: dict[str, Any],
    condition_metric: str,
) -> float | None:
    """Select the scalar used for logging/validation, never for default gating."""

    normalized = str(condition_metric).strip().lower()
    if normalized == "none":
        return None
    if normalized == "gates":
        key = "gate_relative_l2"
    elif normalized == "all_adaln":
        key = "all_relative_l2"
    else:
        raise ValueError(f"unsupported H3 calibration condition metric {condition_metric!r}")
    value = report.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def signature_bytes(signature: torch.Tensor | None) -> int:
    """Return the bounded persistent size of one CPU/GPU signature."""

    if signature is None:
        return 0
    return int(signature.numel() * signature.element_size())


__all__ = [
    "AFFINE_INDICES",
    "ALL_INDICES",
    "CONDITION_COMPONENTS",
    "CONDITION_SIGNATURE_HIDDEN_SAMPLES",
    "GATE_INDICES",
    "sampled_hidden_indices",
    "sampled_modulation_signature",
    "selected_condition_error",
    "signature_bytes",
    "signature_difference",
]
