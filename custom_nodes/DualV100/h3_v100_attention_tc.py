"""Experimental SM70 attention using short FP16 Tensor-Core dot products.

The existing ``h3_v100_attention.py`` kernel performs an online softmax but
keeps the whole ``head_dim=128`` QK dot in one operation.  On Volta that can
produce a very large register footprint.  This candidate splits QK into
16-wide dots, then accumulates the partial scores in FP32.  The split is
deliberate: it gives Triton a Volta-compatible ``m16n8k16`` opportunity while
keeping the softmax state and the PV reduction in FP32.

This module is benchmark-only for now.  It does not patch ComfyUI and has no
relationship with NCCL; a TP caller would invoke it separately on each local
head shard.  The PyTorch efficient-SDPA path remains the production fallback.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - only used on non-Triton installs
    triton = None
    tl = None


_LOG2_E = 1.4426950408889634
_HEAD_DIM = 128
_QK_CHUNK = 16


if triton is not None:

    @triton.jit
    def _h3_attn_fwd_sm70_tc(
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
        n_heads,
        n_ctx,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        qk_fp16: tl.constexpr,
        head_dim: tl.constexpr,
        qk_chunk: tl.constexpr,
    ):
        """Flash-style forward; all program-local QK pieces are 16-wide."""

        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // n_heads
        head = batch_head - batch * n_heads

        offs_m = query_block * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        q_valid = offs_m < n_ctx

        q_base = batch * stride_qb + head * stride_qh
        k_base = batch * stride_kb + head * stride_kh
        v_base = batch * stride_vb + head * stride_vh

        # The actual online loop owns K/V tiles.  Keeping the Q loads outside
        # that loop would require a full 128-wide Q register tile, defeating
        # the purpose of this candidate, so Q is reread from its compact
        # global layout for each key tile and each 16-wide K slice.
        row_max = tl.full((block_m,), -float("inf"), tl.float32)
        row_sum = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, head_dim), tl.float32)

        for start_n in tl.range(0, n_ctx, block_n):
            key_rows = start_n + offs_n
            scores = tl.zeros((block_m, block_n), dtype=tl.float32)
            for d_start in range(0, head_dim, qk_chunk):
                offs_d = d_start + tl.arange(0, qk_chunk)
                q_offsets = (
                    q_base
                    + offs_m[:, None] * stride_qs
                    + offs_d[None, :] * stride_qd
                )
                k_offsets = (
                    k_base
                    + key_rows[:, None] * stride_ks
                    + offs_d[None, :] * stride_kd
                )
                q_tile = tl.load(
                    q_ptr + q_offsets,
                    mask=q_valid[:, None],
                    other=0.0,
                )
                k_tile = tl.load(
                    k_ptr + k_offsets,
                    mask=key_rows[:, None] < n_ctx,
                    other=0.0,
                )
                if qk_fp16:
                    partial = tl.dot(
                        q_tile,
                        tl.trans(k_tile),
                        out_dtype=tl.float16,
                    )
                else:
                    partial = tl.dot(
                        q_tile,
                        tl.trans(k_tile),
                        out_dtype=tl.float32,
                    )
                scores += partial.to(tl.float32)

            scores = scores * scale_log2e
            scores = tl.where(key_rows[None, :] < n_ctx, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            correction = tl.exp2(row_max - new_max)
            probabilities = tl.exp2(scores - new_max[:, None])

            acc *= correction[:, None]
            # Keep the PV dot in the current key tile.  Triton does not
            # support indexing a register tile with a tensor or a dynamic
            # slice on the installed release; using the complete tile keeps
            # the implementation valid while still avoiding SxS storage.
            v_offsets = (
                v_base
                + key_rows[:, None] * stride_vs
                + tl.arange(0, head_dim)[None, :] * stride_vd
            )
            v_tile = tl.load(
                v_ptr + v_offsets,
                mask=key_rows[:, None] < n_ctx,
                other=0.0,
            )
            acc = tl.dot(
                probabilities.to(tl.float16),
                v_tile,
                acc=acc,
                out_dtype=tl.float32,
            )
            row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
            row_max = new_max

        acc /= row_sum[:, None]
        o_offsets = (
            batch * stride_ob
            + offs_m[:, None] * stride_os
            + head * stride_oh
            + tl.arange(0, head_dim)[None, :] * stride_od
        )
        tl.store(o_ptr + o_offsets, acc.to(tl.float16), mask=q_valid[:, None])


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("SM70 TC attention requires CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must share a CUDA device")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("SM70 TC attention requires FP16 Q/K/V")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"expected equal [B,H,S,D] tensors, got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[-1] != _HEAD_DIM:
        raise ValueError(f"candidate requires head_dim={_HEAD_DIM}")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the head dimension must be contiguous")
    if torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("candidate requires NVIDIA SM70")


def h3_attention_sm70_tc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    qk_fp16: bool = False,
) -> torch.Tensor:
    """Return [B,S,H,128] output from the experimental chunked kernel."""

    _validate(q, k, v)
    batch, heads, sequence, _ = q.shape
    if sequence == 0:
        return torch.empty(
            (batch, 0, heads, _HEAD_DIM), dtype=q.dtype, device=q.device
        )
    block_m = block_m or _env_int("H3_V100_TC_ATTN_BLOCK_M", 16)
    block_n = block_n or _env_int("H3_V100_TC_ATTN_BLOCK_N", 64)
    num_warps = num_warps or _env_int("H3_V100_TC_ATTN_WARPS", 4)
    num_stages = num_stages or _env_int("H3_V100_TC_ATTN_STAGES", 1)
    if block_m not in (16, 32, 64):
        raise ValueError("block_m must be one of 16, 32, 64")
    if block_n not in (32, 64, 128):
        raise ValueError("block_n must be one of 32, 64, 128")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")

    output = torch.empty(
        (batch, sequence, heads, _HEAD_DIM), dtype=q.dtype, device=q.device
    )
    softmax_scale = float(scale if scale is not None else _HEAD_DIM ** -0.5)
    grid = (triton.cdiv(sequence, block_m), batch * heads)
    _h3_attn_fwd_sm70_tc[grid](
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
        block_m=block_m,
        block_n=block_n,
        qk_fp16=qk_fp16,
        head_dim=_HEAD_DIM,
        qk_chunk=_QK_CHUNK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
