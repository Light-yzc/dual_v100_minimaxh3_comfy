"""SM70 kernels for H3's standard GGML Q4_0 storage.

The production H3 TP path normally uses the simple PyTorch dequantizer.  This
module provides an opt-in Triton implementation that writes the dequantized
FP16 matrix directly, avoiding the intermediate nibble/scales tensors created
by the eager expression.  It is intentionally a dequantizer, not a guessed
Q4 GEMM: the result is byte-layout compatible with the existing Linear path and
can be gated against the reference before use.
"""

from __future__ import annotations

import logging
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - non-CUDA/syntax-only environments
    triton = None
    tl = None


Q4_BLOCK_ELEMENTS = 32
Q4_BLOCK_BYTES = 18
_WARNED = False


if triton is not None:

    @triton.jit
    def _q4_0_dequant_kernel(
        raw_ptr,
        output_ptr,
        total_blocks,
        row_bytes,
        in_features,
        blocks_per_program: tl.constexpr,
        values_per_block: tl.constexpr,
    ):
        program = tl.program_id(0)
        block_ids = program * blocks_per_program + tl.arange(0, blocks_per_program)
        valid_blocks = block_ids < total_blocks
        row_ids = block_ids // (in_features // values_per_block)
        col_blocks = block_ids % (in_features // values_per_block)

        raw_base = row_ids * row_bytes + col_blocks * 18
        scale_lo = tl.load(raw_ptr + raw_base, mask=valid_blocks, other=0).to(tl.uint16)
        scale_hi = tl.load(raw_ptr + raw_base + 1, mask=valid_blocks, other=0).to(tl.uint16)
        scale_bits = scale_lo | (scale_hi << 8)
        scale = scale_bits.to(tl.float16, bitcast=True)

        value_ids = tl.arange(0, values_per_block)
        # GGML Q4_0 stores all low nibbles first, followed by all high
        # nibbles.  It is not the usual interleaved [lo, hi] value order.
        packed_ids = value_ids % 16
        shifts = (value_ids // 16) * 4
        packed = tl.load(
            raw_ptr
            + raw_base[:, None]
            + 2
            + packed_ids[None, :],
            mask=valid_blocks[:, None],
            other=0,
        )
        quants = ((packed >> shifts[None, :]) & 0x0F).to(tl.float16) - 8.0
        values = scale[:, None] * quants

        output_base = row_ids * in_features + col_blocks * values_per_block
        output_offsets = output_base[:, None] + value_ids[None, :]
        tl.store(
            output_ptr + output_offsets,
            values,
            mask=valid_blocks[:, None],
        )


def _validate(matrix) -> tuple[torch.Tensor, int, int, int]:
    raw = matrix.raw
    out_features = int(matrix.out_features)
    in_features = int(matrix.in_features)
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not raw.is_cuda or raw.dtype != torch.uint8:
        raise ValueError("H3 Q4 Triton dequantization requires CUDA uint8 raw storage")
    if not raw.is_contiguous():
        raise ValueError("H3 Q4 raw storage must be contiguous")
    if in_features <= 0 or out_features <= 0 or in_features % Q4_BLOCK_ELEMENTS:
        raise ValueError(f"invalid Q4 matrix shape: {(out_features, in_features)}")
    row_bytes = in_features // Q4_BLOCK_ELEMENTS * Q4_BLOCK_BYTES
    expected = out_features * row_bytes
    if raw.numel() != expected:
        raise ValueError(
            f"Q4 raw storage mismatch: got {raw.numel()} bytes, expected {expected}"
        )
    if torch.cuda.get_device_capability(raw.device) != (7, 0):
        raise ValueError("H3 Q4 Triton dequantization is specialized for SM70")
    return raw, out_features, in_features, row_bytes


def dequantize_q4_0_sm70(
    matrix,
    *,
    blocks_per_program: int = 4,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Dequantize Q4_0 raw bytes to an FP16 matrix on the current GPU."""

    raw, out_features, in_features, row_bytes = _validate(matrix)
    if blocks_per_program not in (1, 2, 4, 8, 16):
        raise ValueError(f"unsupported blocks_per_program={blocks_per_program}")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError(f"unsupported num_warps={num_warps}")
    output = torch.empty(
        (out_features, in_features), dtype=torch.float16, device=raw.device
    )
    total_blocks = out_features * (in_features // Q4_BLOCK_ELEMENTS)
    grid = (triton.cdiv(total_blocks, blocks_per_program),)
    _q4_0_dequant_kernel[grid](
        raw,
        output,
        total_blocks,
        row_bytes,
        in_features,
        blocks_per_program=blocks_per_program,
        values_per_block=Q4_BLOCK_ELEMENTS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def enabled_from_env() -> bool:
    return os.environ.get("H3_TP_Q4_DEQUANT", "eager").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "triton",
        "sm70",
    }


def dequantize_q4_0_with_fallback(matrix, eager):
    """Use the experimental kernel when requested, retaining eager fallback."""

    global _WARNED
    if not enabled_from_env():
        return eager(matrix)
    try:
        return dequantize_q4_0_sm70(matrix)
    except Exception as error:
        if os.environ.get("H3_TP_Q4_DEQUANT_STRICT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise
        if not _WARNED:
            logging.exception("H3 Q4 Triton dequantization failed; using eager fallback: %s", error)
            _WARNED = True
        return eager(matrix)


__all__ = [
    "dequantize_q4_0_sm70",
    "dequantize_q4_0_with_fallback",
    "enabled_from_env",
]
