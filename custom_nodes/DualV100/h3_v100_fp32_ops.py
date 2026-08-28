"""Fused FP32-residual operations for MiniMax H3 on NVIDIA Volta.

H3 keeps its replicated residual stream in FP32 on V100 to avoid the overflow
seen with the original BF16-trained activation range.  The attention and MLP
branches are still FP16.  Eager PyTorch materializes an FP32 RMSNorm output,
casts it, and then launches one modulation operation per packed modality span.

This module keeps the stable FP32 residual semantics while fusing RMSNorm,
FP16 materialization, and per-token AdaLN modulation into one SM70 kernel.  It
also fuses the FP32 gated residual update.  A compact row-index tensor maps each
packed token to its AdaLN row and is reused by all 50 blocks in one forward.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in non-CUDA installs.
    triton = None
    tl = None


H3_HIDDEN = 5376
H3_LOCAL_FFN = 7168
FP16_SCALE_TARGET = 32752.0


if triton is not None:

    @triton.jit
    def _h3_fp32_rms_mod_sm70_kernel(
        residual_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        mod_rows_ptr,
        output_ptr,
        epsilon,
        residual_stride,
        modulation_stride,
        output_stride,
        hidden: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden

        residual = tl.load(
            residual_ptr + row * residual_stride + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(residual * residual, axis=0)
        inv_rms = tl.rsqrt(square_sum / hidden + epsilon)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Preserve the production rounding boundaries: FP32 RMSNorm is first
        # materialized as FP16, then FP16 scale/multiply and shift/add follow.
        normalized = (residual * inv_rms * weight).to(tl.float16)
        mod_row = tl.load(mod_rows_ptr + row).to(tl.int64)
        mod_base = mod_row * modulation_stride + offsets
        scale = tl.load(scale_ptr + mod_base, mask=mask, other=0.0).to(tl.float16)
        shift = tl.load(shift_ptr + mod_base, mask=mask, other=0.0).to(tl.float16)
        one_plus_scale = (1.0 + scale).to(tl.float16)
        modulated = (normalized * one_plus_scale).to(tl.float16)
        modulated = (modulated + shift).to(tl.float16)
        tl.store(
            output_ptr + row * output_stride + offsets,
            modulated,
            mask=mask,
        )

    @triton.jit
    def _h3_fp32_gate_residual_sm70_kernel(
        residual_ptr,
        update_ptr,
        gate_ptr,
        mod_rows_ptr,
        residual_stride,
        update_stride,
        modulation_stride,
        hidden: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden
        mod_row = tl.load(mod_rows_ptr + row).to(tl.int64)

        residual_offsets = row * residual_stride + offsets
        update_offsets = row * update_stride + offsets
        gate_offsets = mod_row * modulation_stride + offsets
        residual = tl.load(residual_ptr + residual_offsets, mask=mask).to(tl.float32)
        update = tl.load(update_ptr + update_offsets, mask=mask).to(tl.float32)
        gate = tl.load(gate_ptr + gate_offsets, mask=mask).to(tl.float32)
        tl.store(residual_ptr + residual_offsets, residual + update * gate, mask=mask)

    @triton.jit
    def _h3_swiglu_scale_sm70_kernel(
        packed_ptr,
        swiglu_ptr,
        safe_ptr,
        row_scale_ptr,
        packed_stride,
        swiglu_stride,
        safe_stride,
        target,
        ffn: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < ffn
        packed_base = row * packed_stride
        gate = tl.load(packed_ptr + packed_base + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(packed_ptr + packed_base + ffn + offsets, mask=mask, other=0.0).to(tl.float32)
        swiglu = gate * tl.sigmoid(gate) * up
        maximum = tl.max(tl.abs(swiglu), axis=0)
        ratio = tl.maximum(maximum / target, 1.0)
        scale = tl.exp2(tl.ceil(tl.log2(ratio)))
        safe = (swiglu / scale).to(tl.float16)
        tl.store(swiglu_ptr + row * swiglu_stride + offsets, swiglu, mask=mask)
        tl.store(safe_ptr + row * safe_stride + offsets, safe, mask=mask)
        tl.store(row_scale_ptr + row, scale)


def _require_sm70(tensor: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for H3 fused FP32 residual operations")
    if not tensor.is_cuda:
        raise ValueError("H3 fused FP32 residual operations require CUDA tensors")
    if torch.cuda.get_device_capability(tensor.device) != (7, 0):
        raise ValueError("H3 fused FP32 residual operations require SM70")


def make_modulation_rows(
    sequence: int,
    segments: Sequence[Sequence[int]],
    device: torch.device | str,
) -> torch.Tensor:
    """Build the token-to-AdaLN-row map shared by all backbone blocks."""

    if sequence <= 0:
        raise ValueError(f"H3 modulation sequence must be positive, got {sequence}")
    rows = torch.empty(sequence, dtype=torch.int32, device=device)
    covered = torch.zeros(sequence, dtype=torch.bool, device="cpu")
    for segment in segments:
        if len(segment) != 3:
            raise ValueError(f"invalid H3 modulation segment: {segment}")
        start, stop, row = (int(value) for value in segment)
        if not 0 <= start < stop <= sequence or row < 0:
            raise ValueError(f"invalid H3 modulation segment: {segment}")
        if bool(covered[start:stop].any()):
            raise ValueError(f"overlapping H3 modulation segment: {segment}")
        rows[start:stop].fill_(row)
        covered[start:stop] = True
    if not bool(covered.all()):
        missing = int((~covered).sum().item())
        raise ValueError(f"H3 modulation segments leave {missing} tokens uncovered")
    return rows


def h3_fp32_rms_mod_sm70(
    residual: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    mod_rows: torch.Tensor,
    *,
    epsilon: float = 1e-5,
    num_warps: int = 8,
) -> torch.Tensor:
    """Produce one FP16 H3 branch from an FP32 residual in a single kernel."""

    _require_sm70(residual)
    if residual.dtype != torch.float32 or residual.ndim != 2:
        raise ValueError(f"expected 2D FP32 H3 residual, got {residual.shape}/{residual.dtype}")
    if residual.shape[1] != H3_HIDDEN:
        raise ValueError(f"expected H3 hidden={H3_HIDDEN}, got {residual.shape[1]}")
    if residual.stride(1) != 1:
        raise ValueError("H3 residual hidden dimension must be contiguous")
    if weight.numel() != H3_HIDDEN or weight.device != residual.device:
        raise ValueError("H3 RMSNorm weight has an incompatible shape/device")
    if shift.shape != scale.shape or shift.ndim != 2 or shift.shape[1] != H3_HIDDEN:
        raise ValueError("H3 AdaLN shift/scale must have equal [rows, 5376] shapes")
    if shift.device != residual.device or scale.device != residual.device:
        raise ValueError("H3 AdaLN shift/scale must be on the residual device")
    if mod_rows.shape != (residual.shape[0],) or mod_rows.dtype != torch.int32:
        raise ValueError("H3 modulation row map must be contiguous int32 [sequence]")
    if mod_rows.device != residual.device or not mod_rows.is_contiguous():
        raise ValueError("H3 modulation row map must be contiguous on the residual device")
    if not 1 <= num_warps <= 32 or num_warps & (num_warps - 1):
        raise ValueError(f"num_warps must be a power of two in [1,32], got {num_warps}")

    output = torch.empty_like(residual, dtype=torch.float16)
    block = triton.next_power_of_2(H3_HIDDEN)
    _h3_fp32_rms_mod_sm70_kernel[(residual.shape[0],)](
        residual,
        weight,
        shift,
        scale,
        mod_rows,
        output,
        float(epsilon),
        residual.stride(0),
        shift.stride(0),
        output.stride(0),
        hidden=H3_HIDDEN,
        block=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def h3_fp32_gate_residual_sm70_(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    mod_rows: torch.Tensor,
    *,
    num_warps: int = 8,
) -> torch.Tensor:
    """Apply the per-token FP32 AdaLN gate to a replicated residual in-place."""

    _require_sm70(residual)
    if residual.dtype != torch.float32 or residual.ndim != 2:
        raise ValueError("H3 gated residual must be a 2D FP32 tensor")
    if update.shape != residual.shape or update.dtype != torch.float32:
        raise ValueError("H3 gated update must match the FP32 residual")
    if update.device != residual.device or residual.stride(1) != 1 or update.stride(1) != 1:
        raise ValueError("H3 residual/update must be contiguous on one device")
    if gate.ndim != 2 or gate.shape[1] != residual.shape[1] or gate.device != residual.device:
        raise ValueError("H3 AdaLN gate has an incompatible shape/device")
    if mod_rows.shape != (residual.shape[0],) or mod_rows.dtype != torch.int32:
        raise ValueError("H3 modulation row map must be int32 [sequence]")
    if mod_rows.device != residual.device or not mod_rows.is_contiguous():
        raise ValueError("H3 modulation row map must be contiguous on the residual device")
    if not 1 <= num_warps <= 32 or num_warps & (num_warps - 1):
        raise ValueError(f"num_warps must be a power of two in [1,32], got {num_warps}")

    block = triton.next_power_of_2(residual.shape[1])
    _h3_fp32_gate_residual_sm70_kernel[(residual.shape[0],)](
        residual,
        update,
        gate,
        mod_rows,
        residual.stride(0),
        update.stride(0),
        gate.stride(0),
        hidden=residual.shape[1],
        block=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return residual


def h3_swiglu_scale_sm70(
    packed: torch.Tensor,
    *,
    target: float = FP16_SCALE_TARGET,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse FP32 SwiGLU, row scaling, and safe FP16 materialization.

    The FP32 output remains available to the existing FP32 FC2 LoRA path.  The
    second output is the exact scaled FP16 input consumed by the base FC2
    Tensor Core GEMM, and the final output is one FP32 power-of-two per token.
    """

    _require_sm70(packed)
    if packed.dtype != torch.float16 or packed.ndim != 2:
        raise ValueError(f"expected 2D FP16 packed FC1 output, got {packed.shape}/{packed.dtype}")
    if packed.shape[1] != 2 * H3_LOCAL_FFN or packed.stride(1) != 1:
        raise ValueError(
            f"expected contiguous H3 local FC1 width {2 * H3_LOCAL_FFN}, got {packed.shape}"
        )
    if not 0.0 < float(target) <= torch.finfo(torch.float16).max:
        raise ValueError(f"invalid FP16 scaling target {target}")
    if not 1 <= num_warps <= 32 or num_warps & (num_warps - 1):
        raise ValueError(f"num_warps must be a power of two in [1,32], got {num_warps}")

    shape = (packed.shape[0], H3_LOCAL_FFN)
    swiglu = torch.empty(shape, dtype=torch.float32, device=packed.device)
    safe = torch.empty(shape, dtype=torch.float16, device=packed.device)
    scale = torch.empty((packed.shape[0], 1), dtype=torch.float32, device=packed.device)
    block = triton.next_power_of_2(H3_LOCAL_FFN)
    _h3_swiglu_scale_sm70_kernel[(packed.shape[0],)](
        packed,
        swiglu,
        safe,
        scale,
        packed.stride(0),
        swiglu.stride(0),
        safe.stride(0),
        float(target),
        ffn=H3_LOCAL_FFN,
        block=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return swiglu, safe, scale


__all__ = [
    "H3_HIDDEN",
    "H3_LOCAL_FFN",
    "h3_fp32_gate_residual_sm70_",
    "h3_fp32_rms_mod_sm70",
    "h3_swiglu_scale_sm70",
    "make_modulation_rows",
]
