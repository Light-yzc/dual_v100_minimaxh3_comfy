"""Narrow MiniMax-H3 global-attention kernel for NVIDIA Volta (SM70).

The production fallback remains ComfyUI's PyTorch SDPA implementation.  This
module is deliberately limited to H3's inference shape: FP16, non-causal,
unmasked self attention with head_dim=128.  It never materializes the S x S
score matrix and writes the result directly in [B, S, H, D] order so H3's
following output projection does not need a separate transpose/copy.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-Triton installs
    triton = None
    tl = None


_LOG2_E = 1.4426950408889634
_FALLBACK: Callable | None = None
_FALLBACK_WARNED = False
_INSTALLED = False


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


if triton is not None:

    @triton.jit
    def _h3_attn_fwd_sm70(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        scale_log2e,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        n_heads: tl.constexpr,
        n_ctx,
        head_dim: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        """FlashAttention-style FP16 forward with FP32 online-softmax state."""

        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // n_heads
        head = batch_head - batch * n_heads

        offs_m = query_block * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, head_dim)

        q_offsets = (
            batch * stride_qb
            + head * stride_qh
            + offs_m[:, None] * stride_qs
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=offs_m[:, None] < n_ctx, other=0.0)

        row_max = tl.full((block_m,), -float("inf"), tl.float32)
        row_sum = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, head_dim), tl.float32)

        for start_n in tl.range(0, n_ctx, block_n):
            key_rows = start_n + offs_n
            k_offsets = (
                batch * stride_kb
                + head * stride_kh
                + key_rows[:, None] * stride_ks
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(
                k_ptr + k_offsets,
                mask=key_rows[:, None] < n_ctx,
                other=0.0,
            )

            # Keep score/value accumulation in FP32 for stable long-context
            # softmax.  Values are converted to log2 space once so exp2 can
            # be used in the online-softmax recurrence.
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale_log2e
            scores = tl.where(key_rows[None, :] < n_ctx, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            correction = tl.exp2(row_max - new_max)
            probabilities = tl.exp2(scores - new_max[:, None])

            acc *= correction[:, None]
            v_offsets = (
                batch * stride_vb
                + head * stride_vh
                + key_rows[:, None] * stride_vs
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(
                v_ptr + v_offsets,
                mask=key_rows[:, None] < n_ctx,
                other=0.0,
            )
            acc = tl.dot(
                probabilities.to(tl.float16), v, acc=acc, out_dtype=tl.float32
            )
            row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
            row_max = new_max

        acc /= row_sum[:, None]
        o_offsets = (
            batch * stride_ob
            + offs_m[:, None] * stride_os
            + head * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptr + o_offsets, acc.to(tl.float16), mask=offs_m[:, None] < n_ctx)


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("H3 SM70 attention requires CUDA Q/K/V tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"Q/K/V devices differ: {q.device}, {k.device}, {v.device}")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"H3 SM70 attention requires FP16 Q/K/V, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"expected equal [B,H,S,D] tensors, got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[-1] != 128:
        raise ValueError(f"H3 SM70 kernel is specialized for head_dim=128, got {q.shape[-1]}")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the head dimension must be contiguous")
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (7, 0):
        raise ValueError(f"H3 SM70 kernel requires compute capability 7.0, got {capability}")


def h3_attention_sm70(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Return H3 attention in projection-ready contiguous [B,S,H,D] order."""

    _validate_inputs(q, k, v)
    batch, heads, sequence, head_dim = q.shape
    if sequence == 0:
        return torch.empty((batch, 0, heads, head_dim), dtype=q.dtype, device=q.device)

    block_m = block_m or _env_int("H3_V100_ATTN_BLOCK_M", 32)
    block_n = block_n or _env_int("H3_V100_ATTN_BLOCK_N", 32)
    num_warps = num_warps or _env_int("H3_V100_ATTN_WARPS", 4)
    num_stages = num_stages or _env_int("H3_V100_ATTN_STAGES", 2)
    if block_m not in (16, 32, 64, 128, 256) or block_n not in (16, 32, 64, 128, 256):
        raise ValueError(f"unsupported SM70 tile {block_m}x{block_n}")
    if num_warps not in (4, 8):
        raise ValueError(f"unsupported warp count {num_warps}")

    output = torch.empty(
        (batch, sequence, heads, head_dim), dtype=q.dtype, device=q.device
    )
    softmax_scale = float(scale if scale is not None else head_dim ** -0.5)
    grid = (triton.cdiv(sequence, block_m), batch * heads)
    _h3_attn_fwd_sm70[grid](
        q,
        k,
        v,
        output,
        softmax_scale * _LOG2_E,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        n_heads=heads,
        n_ctx=sequence,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _can_use_h3_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask,
    skip_reshape: bool,
    enable_gqa: bool,
) -> bool:
    if triton is None or mask is not None or not skip_reshape or enable_gqa:
        return False
    if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
        return False
    return (
        q.is_cuda
        and q.dtype == torch.float16
        and q.ndim == 4
        and q.shape == k.shape == v.shape
        and q.shape[-1] == 128
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and torch.cuda.get_device_capability(q.device) == (7, 0)
    )


def attention_h3_sm70(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    """ComfyUI-compatible adapter with a strict PyTorch-SDPA fallback."""

    del attn_precision
    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if not _can_use_h3_kernel(
        q, k, v, mask=mask, skip_reshape=skip_reshape, enable_gqa=enable_gqa
    ):
        if _FALLBACK is None:
            raise RuntimeError("H3 SM70 attention fallback was not registered")
        return _FALLBACK(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    try:
        output = h3_attention_sm70(q, k, v, scale=kwargs.get("scale"))
    except Exception as error:
        if os.environ.get("H3_V100_ATTENTION_STRICT", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise
        global _FALLBACK_WARNED
        if not _FALLBACK_WARNED:
            logging.exception(
                "H3 SM70 attention failed; retaining PyTorch SDPA fallback: %s", error
            )
            _FALLBACK_WARNED = True
        if _FALLBACK is None:
            raise
        return _FALLBACK(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    if skip_output_reshape:
        return output.permute(0, 2, 1, 3)
    return output.reshape(output.shape[0], output.shape[1], heads * output.shape[-1])


def install_h3_attention() -> bool:
    """Patch only MiniMax H3's module-local attention function."""

    global _FALLBACK, _INSTALLED
    if _INSTALLED:
        return True
    if triton is None or not torch.cuda.is_available():
        logging.warning("H3 SM70 attention requested but Triton/CUDA is unavailable")
        return False
    if not any(
        torch.cuda.get_device_capability(index) == (7, 0)
        for index in range(torch.cuda.device_count())
    ):
        logging.warning("H3 SM70 attention requested but no compute-capability 7.0 GPU exists")
        return False

    from comfy.ldm.minimax import model as h3_model

    if h3_model.optimized_attention is attention_h3_sm70:
        _INSTALLED = True
        return True
    _FALLBACK = h3_model.optimized_attention
    h3_model.optimized_attention = attention_h3_sm70
    _INSTALLED = True
    logging.info(
        "H3-only SM70 Triton attention installed (tile=%sx%s, warps=%s, stages=%s)",
        _env_int("H3_V100_ATTN_BLOCK_M", 32),
        _env_int("H3_V100_ATTN_BLOCK_N", 32),
        _env_int("H3_V100_ATTN_WARPS", 4),
        _env_int("H3_V100_ATTN_STAGES", 2),
    )
    return True


def install_from_env() -> bool:
    mode = os.environ.get("H3_V100_ATTENTION", "pytorch").strip().lower()
    if mode in {"", "0", "off", "false", "pytorch", "sdpa"}:
        return False
    if mode not in {"1", "on", "true", "triton", "sm70"}:
        raise ValueError(
            f"unsupported H3_V100_ATTENTION={mode!r}; use pytorch or triton"
        )
    return install_h3_attention()
