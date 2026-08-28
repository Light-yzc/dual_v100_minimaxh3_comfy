"""Fused H3 Q/K RMSNorm + partial split-half RoPE for Volta (SM70)."""

from __future__ import annotations

import logging
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


_ORIGINAL_FORWARD = None
_INSTALLED = False


if triton is not None:

    @triton.jit
    def _h3_qk_rms_rope_sm70_kernel(
        q_ptr,
        k_ptr,
        freqs_ptr,
        q_weight_ptr,
        k_weight_ptr,
        epsilon,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_fb,
        stride_fs,
        stride_fh,
        stride_fp,
        stride_fr,
        stride_fc,
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rot_dim: tl.constexpr,
        stabilize: tl.constexpr,
        pair_block: tl.constexpr,
        tail_block: tl.constexpr,
    ):
        sequence = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // n_heads
        head = batch_head - batch * n_heads

        pair = tl.arange(0, pair_block)
        pair_count = rot_dim // 2
        pair_mask = pair < pair_count
        tail = tl.arange(0, tail_block)
        tail_count = head_dim - rot_dim
        tail_mask = tail < tail_count

        q_base = q_ptr + batch * stride_qb + sequence * stride_qs + head * stride_qh
        k_base = k_ptr + batch * stride_kb + sequence * stride_ks + head * stride_kh

        first_offsets = pair
        second_offsets = pair + pair_count
        tail_offsets = tail + rot_dim

        q0 = tl.load(q_base + first_offsets * stride_qd, mask=pair_mask, other=0.0).to(tl.float32)
        q1 = tl.load(q_base + second_offsets * stride_qd, mask=pair_mask, other=0.0).to(tl.float32)
        qt = tl.load(q_base + tail_offsets * stride_qd, mask=tail_mask, other=0.0).to(tl.float32)
        k0 = tl.load(k_base + first_offsets * stride_kd, mask=pair_mask, other=0.0).to(tl.float32)
        k1 = tl.load(k_base + second_offsets * stride_kd, mask=pair_mask, other=0.0).to(tl.float32)
        kt = tl.load(k_base + tail_offsets * stride_kd, mask=tail_mask, other=0.0).to(tl.float32)

        if stabilize:
            q_abs_max = tl.maximum(
                tl.maximum(tl.max(tl.abs(q0), axis=0), tl.max(tl.abs(q1), axis=0)),
                tl.max(tl.abs(qt), axis=0),
            )
            k_abs_max = tl.maximum(
                tl.maximum(tl.max(tl.abs(k0), axis=0), tl.max(tl.abs(k1), axis=0)),
                tl.max(tl.abs(kt), axis=0),
            )
            q_stabilizer = tl.maximum(q_abs_max, 1.0)
            k_stabilizer = tl.maximum(k_abs_max, 1.0)
            # Match H3's existing in-place FP16 div_ before RMSNorm.  Without
            # this materialization the fused path is slightly more accurate,
            # but no longer provides a clean same-seed regression target.
            q0 = (q0 / q_stabilizer).to(tl.float16).to(tl.float32)
            q1 = (q1 / q_stabilizer).to(tl.float16).to(tl.float32)
            qt = (qt / q_stabilizer).to(tl.float16).to(tl.float32)
            k0 = (k0 / k_stabilizer).to(tl.float16).to(tl.float32)
            k1 = (k1 / k_stabilizer).to(tl.float16).to(tl.float32)
            kt = (kt / k_stabilizer).to(tl.float16).to(tl.float32)

        q_square_sum = (
            tl.sum(q0 * q0, axis=0)
            + tl.sum(q1 * q1, axis=0)
            + tl.sum(qt * qt, axis=0)
        )
        k_square_sum = (
            tl.sum(k0 * k0, axis=0)
            + tl.sum(k1 * k1, axis=0)
            + tl.sum(kt * kt, axis=0)
        )
        q_inv_rms = tl.rsqrt(q_square_sum / head_dim + epsilon)
        k_inv_rms = tl.rsqrt(k_square_sum / head_dim + epsilon)

        qw0 = tl.load(q_weight_ptr + first_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        qw1 = tl.load(q_weight_ptr + second_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        qwt = tl.load(q_weight_ptr + tail_offsets, mask=tail_mask, other=0.0).to(tl.float32)
        kw0 = tl.load(k_weight_ptr + first_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        kw1 = tl.load(k_weight_ptr + second_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        kwt = tl.load(k_weight_ptr + tail_offsets, mask=tail_mask, other=0.0).to(tl.float32)

        # RMSNorm materializes FP16 before the existing Kitchen RoPE path.
        q0 = (q0 * q_inv_rms * qw0).to(tl.float16)
        q1 = (q1 * q_inv_rms * qw1).to(tl.float16)
        qt = (qt * q_inv_rms * qwt).to(tl.float16)
        k0 = (k0 * k_inv_rms * kw0).to(tl.float16)
        k1 = (k1 * k_inv_rms * kw1).to(tl.float16)
        kt = (kt * k_inv_rms * kwt).to(tl.float16)

        # H3's table is [1, S, 1, rot/2, 2, 2]: batch and head are
        # broadcast dimensions, so indexing them with the real batch/head
        # would walk beyond the table even though their PyTorch stride is
        # non-zero.
        freq_base = freqs_ptr + sequence * stride_fs + pair * stride_fp
        f00 = tl.load(freq_base, mask=pair_mask, other=0.0)
        f01 = tl.load(freq_base + stride_fc, mask=pair_mask, other=0.0)
        f10 = tl.load(freq_base + stride_fr, mask=pair_mask, other=0.0)
        f11 = tl.load(freq_base + stride_fr + stride_fc, mask=pair_mask, other=0.0)

        # Eager Kitchen materializes both FP16 products before the FP16 add;
        # preserve that rounding boundary rather than letting Triton fuse an
        # FP32 FMA and drift from the established latent baseline.
        q_out0 = ((f00 * q0).to(tl.float16) + (f01 * q1).to(tl.float16)).to(tl.float16)
        q_out1 = ((f10 * q0).to(tl.float16) + (f11 * q1).to(tl.float16)).to(tl.float16)
        k_out0 = ((f00 * k0).to(tl.float16) + (f01 * k1).to(tl.float16)).to(tl.float16)
        k_out1 = ((f10 * k0).to(tl.float16) + (f11 * k1).to(tl.float16)).to(tl.float16)

        tl.store(q_base + first_offsets * stride_qd, q_out0, mask=pair_mask)
        tl.store(q_base + second_offsets * stride_qd, q_out1, mask=pair_mask)
        tl.store(q_base + tail_offsets * stride_qd, qt, mask=tail_mask)
        tl.store(k_base + first_offsets * stride_kd, k_out0, mask=pair_mask)
        tl.store(k_base + second_offsets * stride_kd, k_out1, mask=pair_mask)
        tl.store(k_base + tail_offsets * stride_kd, kt, mask=tail_mask)


def _validate(q, k, freqs, q_weight, k_weight, rot_dim: int) -> None:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not (q.is_cuda and k.is_cuda and freqs.is_cuda):
        raise ValueError("H3 RMS-RoPE requires CUDA tensors")
    if q.device != k.device or q.device != freqs.device:
        raise ValueError("Q/K/frequencies must share a CUDA device")
    if q.dtype != torch.float16 or k.dtype != q.dtype or freqs.dtype != q.dtype:
        raise ValueError("H3 RMS-RoPE is specialized for FP16")
    if q.ndim != 4 or q.shape != k.shape or q.shape[-1] != 128:
        raise ValueError(f"expected equal [B,S,H,128] Q/K, got {q.shape}, {k.shape}")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("Q/K head dimension must be contiguous")
    if rot_dim != 96:
        raise ValueError(f"H3 SM70 RMS-RoPE is specialized for rot_dim=96, got {rot_dim}")
    if freqs.ndim != 6 or freqs.shape[-3:] != (rot_dim // 2, 2, 2):
        raise ValueError(f"unexpected RoPE table shape {freqs.shape}")
    if q_weight.numel() != 128 or k_weight.numel() != 128:
        raise ValueError("Q/K RMSNorm weights must each have 128 elements")
    if torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("H3 RMS-RoPE kernel requires SM70")


def h3_qk_rms_rope_sm70_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-6,
    rot_dim: int = 96,
    stabilize: bool = True,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and rotate Q/K in-place without intermediate tensors."""

    if k_weight is None:
        k_weight = q_weight
    _validate(q, k, freqs, q_weight, k_weight, rot_dim)
    batch, sequence, heads, head_dim = q.shape
    grid = (sequence, batch * heads)
    _h3_qk_rms_rope_sm70_kernel[grid](
        q,
        k,
        freqs,
        q_weight,
        k_weight,
        float(epsilon),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        freqs.stride(0),
        freqs.stride(1),
        freqs.stride(2),
        freqs.stride(3),
        freqs.stride(4),
        freqs.stride(5),
        n_heads=heads,
        head_dim=head_dim,
        rot_dim=rot_dim,
        stabilize=stabilize,
        pair_block=64,
        tail_block=32,
        num_warps=num_warps,
        num_stages=1,
    )
    return q, k


def _can_fuse(q, k, freqs, q_weight, k_weight, rot_dim) -> bool:
    return (
        triton is not None
        and torch.is_tensor(q)
        and torch.is_tensor(k)
        and torch.is_tensor(freqs)
        and q.is_cuda
        and q.dtype == k.dtype == freqs.dtype == torch.float16
        and q.shape == k.shape
        and q.ndim == 4
        and q.shape[-1] == 128
        and rot_dim == 96
        and q_weight.numel() == 128
        and k_weight.numel() == 128
        and torch.cuda.get_device_capability(q.device) == (7, 0)
    )


def install_h3_rms_rope() -> bool:
    """Replace H3 Attention.forward while retaining all unsupported fallbacks."""

    global _ORIGINAL_FORWARD, _INSTALLED
    if _INSTALLED:
        return True
    if triton is None or not torch.cuda.is_available():
        logging.warning("H3 SM70 RMS-RoPE requested but Triton/CUDA is unavailable")
        return False

    import comfy.model_management
    from comfy.ldm.minimax import model as h3_model

    _ORIGINAL_FORWARD = h3_model.Attention.forward

    def fused_forward(self, x, rope_freqs=None, transformer_options={}):
        if rope_freqs is None:
            return _ORIGINAL_FORWARD(
                self, x, rope_freqs=rope_freqs, transformer_options=transformer_options
            )

        sequence = x.shape[0]
        qkv = h3_model._h3_check_finite("attn.qkv_proj", self.qkv_proj(x))
        q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
        v = v.view(sequence, self.heads, self.head_dim)
        q = q.view(1, sequence, self.heads, self.head_dim)
        k = k.view(1, sequence, self.heads, self.head_dim)
        q_weight = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        k_weight = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot_dim = rope_freqs.shape[-3] * 2

        if not _can_fuse(q, k, rope_freqs, q_weight, k_weight, rot_dim):
            return _ORIGINAL_FORWARD(
                self, x, rope_freqs=rope_freqs, transformer_options=transformer_options
            )

        h3_qk_rms_rope_sm70_(
            q,
            k,
            rope_freqs,
            q_weight,
            k_weight,
            epsilon=self.q_norm.eps,
            rot_dim=rot_dim,
            stabilize=True,
            num_warps=int(os.environ.get("H3_V100_RMS_ROPE_WARPS", "4")),
        )
        q = h3_model._h3_check_finite("attn.q", q[0])
        k = h3_model._h3_check_finite("attn.k", k[0])
        v = h3_model._h3_check_finite("attn.v", v)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = h3_model._h3_check_finite(
            "attn.sdp",
            h3_model.optimized_attention(
                q,
                k,
                v,
                self.heads,
                mask=None,
                skip_reshape=True,
                transformer_options=transformer_options,
            ),
        ).squeeze(0)
        if h3_model._H3_FP32_ATTN_OUT and out.dtype == torch.float16:
            out = out.float()
        return h3_model._h3_check_finite("attn.out_proj", self.out_proj(out))

    h3_model.Attention.forward = fused_forward
    _INSTALLED = True
    logging.info("H3-only fused SM70 Q/K RMSNorm + partial RoPE installed")
    return True


def install_from_env() -> bool:
    value = os.environ.get("H3_V100_RMS_ROPE", "pytorch").strip().lower()
    if value in {"", "0", "off", "false", "pytorch", "eager"}:
        return False
    if value not in {"1", "on", "true", "triton", "sm70"}:
        raise ValueError(f"unsupported H3_V100_RMS_ROPE={value!r}")
    return install_h3_rms_rope()
