"""Experimental bounded-score attention for H3 on SM70.

This is a control candidate for the custom SM70 kernel.  It uses cuBLAS-backed
batched matmul for QK^T and PV, while only materialising ``block_m x S`` scores
per query chunk.  It is not a drop-in production path yet: its extra score
workspace and Python chunk loop must beat PyTorch efficient SDPA at the real
H3 sequence lengths before it can be considered.
"""

from __future__ import annotations

import torch


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("H3 chunked attention requires CUDA Q/K/V")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("H3 chunked attention requires FP16 Q/K/V")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"expected equal [B,H,S,D] tensors, got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[-1] != 128:
        raise ValueError(f"H3 chunked attention requires head_dim=128, got {q.shape[-1]}")
    if torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("H3 chunked attention is currently specialized for SM70")


def h3_attention_chunked_bmm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_m: int = 128,
    scale: float | None = None,
    fp32_softmax: bool = False,
) -> torch.Tensor:
    """Return ``[B,S,H,D]`` H3 attention using bounded cuBLAS score tiles.

    The score tile is ``[B,H,block_m,S]``.  No ``S x S`` tensor is retained.
    ``fp32_softmax`` is exposed because it is a useful precision/throughput
    gate on Volta; the production candidate must use whichever mode passes the
    H3 output comparison, not whichever mode is merely faster.
    """
    _validate(q, k, v)
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")

    batch, heads, sequence, head_dim = q.shape
    if sequence == 0:
        return torch.empty((batch, 0, heads, head_dim), dtype=q.dtype, device=q.device)

    # Transpose is a view.  torch.matmul dispatches the two contractions to
    # strided-batched cuBLAS without making a full K transpose copy.
    kt = k.transpose(-1, -2)
    value_scale = float(scale if scale is not None else head_dim ** -0.5)
    output = torch.empty_like(q)

    for start in range(0, sequence, block_m):
        stop = min(start + block_m, sequence)
        scores = torch.matmul(q[:, :, start:stop, :], kt).mul_(value_scale)
        if fp32_softmax:
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        else:
            probabilities = torch.softmax(scores, dim=-1)
        output[:, :, start:stop, :] = torch.matmul(probabilities, v)

    return output.permute(0, 2, 1, 3).contiguous()
