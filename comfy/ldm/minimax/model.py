"""MiniMax H3 audio-video DiT.

Single-stream packed-token transformer denoising video (24ch, patch 1x2x2) and
stereo audio (32ch, 40 Hz) latents jointly, conditioned on Qwen3-VL layer-50 hidden states.
The packed sequence is:
[text | cond rows | audio | video] for t2va/fl2va
[text | reference blocks | audio | video] for ref2va

Timestep domain: the model receives the *video* sigma from the sampler and
derives per-token timesteps t = 1 - sigma internally; the audio stream runs on
its own shifted schedule (sigma_shift video 12.0 / audio 3.0), mapped from the
video sigma in closed form. The sampler carries the audio latent scaled onto the
video schedule (ModelSamplingAV); forward() undoes that scale and converts the
velocity back, so _forward only ever sees the stream's own latent.
"""

import math
import os
import time

import torch
import torch.nn as nn

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


# Opt-in numerical bisection for V100/H3 bring-up.  This deliberately raises at
# the *first* non-finite tensor instead of trying to hide it with nan_to_num.
# It is off in normal inference, so there is no synchronization/statistics cost.
_H3_FINITE_TRACE = os.environ.get("H3_FINITE_TRACE", "").lower() in {"1", "true", "yes", "on"}
_H3_FP32_RESIDUAL = os.environ.get("H3_FP32_RESIDUAL", "").lower() in {"1", "true", "yes", "on"}
_H3_FP32_MLP = os.environ.get("H3_FP32_MLP", "").lower() in {"1", "true", "yes", "on"}
_H3_FP32_MLP_CHUNK_ROWS = max(1, int(os.environ.get("H3_FP32_MLP_CHUNK_ROWS", "4096")))
_H3_FP32_ATTN_OUT = os.environ.get("H3_FP32_ATTN_OUT", "").lower() in {"1", "true", "yes", "on"}
_H3_PARALLEL_MODE = os.environ.get("H3_PARALLEL_MODE", "single").strip().lower()
_H3_PP_SPLIT = int(os.environ.get("H3_PP_SPLIT", "25"))
_H3_PP_DEVICE = os.environ.get("H3_PP_DEVICE", "cuda:1")
_H3_PARALLEL_PROFILE = os.environ.get("H3_PARALLEL_PROFILE", "").lower() in {"1", "true", "yes", "on"}
_H3_TRACE_BLOCK = "h3"


def _h3_check_finite(label, x):
    if not _H3_FINITE_TRACE or not torch.is_tensor(x) or not x.is_floating_point():
        return x
    finite = torch.isfinite(x)
    if bool(finite.all()):
        return x
    nan_count = int(torch.isnan(x).sum().item())
    inf_count = int(torch.isinf(x).sum().item())
    valid = x[finite]
    max_abs = float(valid.abs().max().item()) if valid.numel() else float("nan")
    message = (
        f"[H3 finite trace] first non-finite at {_H3_TRACE_BLOCK}.{label}; "
        f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device}; "
        f"nan={nan_count} inf={inf_count} finite_abs_max={max_abs:.6g}"
    )
    print(message, flush=True)
    raise RuntimeError(message)


def _safe_fp16_rms_norm(norm, x):
    """Keep RMSNorm in fp16 while preventing overflow in the square reduction."""
    if x.dtype != torch.float16:
        return norm(x)
    row_scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min_(1.0)
    return norm(x / row_scale)


def _h3_branch_rms_norm(norm, x):
    """Normalize an FP32 residual stream, then return FP16 branch activations."""
    if _H3_FP32_RESIDUAL and x.dtype == torch.float32:
        return norm(x).to(torch.float16)
    return _safe_fp16_rms_norm(norm, x)


def _h3_module_device(module):
    """Return a module's materialized parameter/buffer device, if any.

    GGUF Linear stores a quantized Parameter, so the normal ``parameters``
    iterator remains the reliable source of truth after a block is moved to a
    second CUDA device.
    """
    for value in module.parameters(recurse=True):
        return value.device
    for value in module.buffers(recurse=True):
        return value.device
    return None


def _h3_to_device(module, device):
    """Move a module while making the target CUDA device current.

    The context is necessary for the optional custom CUDA GGUF paths; explicit
    tensor devices alone are insufficient for a few third-party kernels.
    """
    with comfy.model_management.cuda_device_context(device):
        module.to(device)


def time_shift_sigma(sigma, from_shift, to_shift):
    # invert sigma = s*b/(1+(s-1)*b) to the base grid, re-apply the other shift
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def patchify_video(latent, patch_size=(1, 2, 2)):
    # [B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    # [B, C=32, ch=2, T] -> [ch*T, 32] channel-major (ch0 t0..T-1, ch1 t0..T-1)
    b, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    # linspace((1 - ratio) / 2, (1 + ratio) / 2, dim // patch, endpoint=False) * 32
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    # area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    # origin + exclusive cumsum
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _audio_grid(cursor, t, w_low, w_high):
    # channel-major stereo rows: t advances per latent frame, w pinned to the grid extremes per stereo channel, h stays 0
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden, out, dtype=None, device=None, operations=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = operations.Linear(freq_dim, hidden, bias=True, dtype=dtype, device=device)
        self.proj_out = operations.Linear(hidden, out, bias=True, dtype=dtype, device=device)

    def forward(self, t):
        # t: [M] in [0, 1]; fp32 throughout, cos before sin
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(nn.functional.silu(self.proj_in(emb)))


def rope_rotation_table(angles, dtype):
    """[S, rot_dim] pair angles -> [1, S, 1, rot_dim/2, 2, 2] rotation matrices."""
    half = angles.shape[-1] // 2
    ang = angles[:, :half]  # duplicated halves: [:, :half] == [:, half:]
    c, s = torch.cos(ang), torch.sin(ang)
    table = torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2)
    return table.to(dtype)


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = operations.Linear(hidden, inner * 3, bias=False, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = operations.Linear(inner, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        qkv = _h3_check_finite("attn.qkv_proj", self.qkv_proj(x))
        q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            # fused per-head RMSNorm + partial split-half rope, in place on the qkv buffer
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            # V100 fp16 can overflow while RMSNorm squares a large q/k value.
            # Per-head rescaling is mathematically cancelled by RMSNorm and keeps
            # the entire attention path in fp16.
            if q.dtype == torch.float16:
                q_scale = q.detach().abs().amax(dim=-1, keepdim=True).clamp_min_(1.0)
                k_scale = k.detach().abs().amax(dim=-1, keepdim=True).clamp_min_(1.0)
                q.div_(q_scale)
                k.div_(k_scale)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = _safe_fp16_rms_norm(self.q_norm, q.view(s, self.heads, self.head_dim))
            k = _safe_fp16_rms_norm(self.k_norm, k.view(s, self.heads, self.head_dim))
        _h3_check_finite("attn.q", q)
        _h3_check_finite("attn.k", k)
        _h3_check_finite("attn.v", v)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = _h3_check_finite(
            "attn.sdp",
            optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True,
                                transformer_options=transformer_options),
        )
        out = out.squeeze(0)
        # Keep SDPA and QKV in FP16, but use a wide accumulator/output for the
        # final attention projection.  H3 can exceed the FP16 result range at
        # video resolutions even when every Q/K/V and attention output is finite.
        if _H3_FP32_ATTN_OUT and out.dtype == torch.float16:
            out = out.float()
        return _h3_check_finite("attn.out_proj", self.out_proj(out))


class MLP(nn.Module):
    def __init__(self, hidden, ffn, dtype=None, device=None, operations=None):
        super().__init__()
        self.fc1 = operations.Linear(hidden, ffn * 2, bias=False, dtype=dtype, device=device)
        self.fc2 = operations.Linear(ffn, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        if _H3_FP32_MLP and x.dtype == torch.float16:
            # The down projection is where the BF16-trained H3 MLP first
            # exceeds FP16's output range.  Keep fc1/SiLU in FP16, then run
            # fc2 in FP32 and return a FP32 residual update.  Splitting rows
            # bounds the 1 MP activation peak; calling fc2.forward preserves
            # runtime Turbo LoRA adapters rather than bypassing them.
            out = torch.empty((x.shape[0], self.fc2.out_features), dtype=torch.float32, device=x.device)
            for start in range(0, x.shape[0], _H3_FP32_MLP_CHUNK_ROWS):
                stop = min(start + _H3_FP32_MLP_CHUNK_ROWS, x.shape[0])
                h = _h3_check_finite("mlp.fc1", self.fc1(x[start:stop]))
                # A finite pair of FP16 halves can still overflow during
                # SiLU(gate) * up.  Upcast before the nonlinearity, not only
                # before fc2, so the BF16-trained MLP's dynamic range survives.
                h = _h3_check_finite("mlp.swiglu", comfy.ops.INPUT_ACT_EAGER["swiglu"](h.float()))
                out[start:stop] = _h3_check_finite("mlp.fc2", self.fc2(h))
            return out
        h = _h3_check_finite("mlp.fc1", self.fc1(x))
        return _h3_check_finite("mlp.fc2", comfy.ops.linear_input_act(self.fc2, h, "swiglu"))


class AdalnProj(nn.Module):
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = operations.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype, device=device)

    def forward(self, t_emb):
        # [M, t_dim] -> expand tensors of [M*modalities, hidden]
        x = self.linear(nn.functional.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _mod_scale_shift(h, shift, scale, segments):
    # segments: [(start, stop, mod_row)] covering h contiguously.
    for a, b, row in segments:
        h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _mod_gate(x, gate, other, segments):
    # other is the fresh attn/mlp output: accumulate the gated residual into the stream in place, one fused kernel per segment
    for a, b, row in segments:
        if _H3_FP32_RESIDUAL and x.dtype == torch.float32:
            x[a:b].addcmul_(other[a:b].float(), gate[row].float())
        else:
            x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
    return x


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)

    def forward(self, x, transformer_options={}):
        # attn/mlp outputs are fresh: accumulate residuals in place
        _h3_check_finite("refiner.input", x)
        h = _h3_check_finite("refiner.norm1", _safe_fp16_rms_norm(self.norm1, x))
        x = _h3_check_finite(
            "refiner.residual_attn",
            self.attn(h, transformer_options=transformer_options).add_(x),
        )
        h = _h3_check_finite("refiner.norm2", _safe_fp16_rms_norm(self.norm2, x))
        return _h3_check_finite("refiner.residual_mlp", self.mlp(h).add_(x))


class TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps, dtype=dtype, device=device, operations=operations)
            for _ in range(num_layers)])
        self.final_norm = operations.RMSNorm(hidden, eps=final_eps, dtype=dtype, device=device)

    def forward(self, x, transformer_options={}):
        global _H3_TRACE_BLOCK
        for i, block in enumerate(self.blocks):
            previous_trace_block = _H3_TRACE_BLOCK
            _H3_TRACE_BLOCK = f"refiner[{i}]"
            try:
                x = block(x, transformer_options=transformer_options)
            finally:
                _H3_TRACE_BLOCK = previous_trace_block
        return _h3_check_finite("refiner.final_norm", _safe_fp16_rms_norm(self.final_norm, x))


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,
                 apply_silu=True, adaln_dtype=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                    device=device, operations=operations)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        for name, value in (("adaln.shift_msa", shift_msa), ("adaln.scale_msa", scale_msa),
                            ("adaln.gate_msa", gate_msa), ("adaln.shift_mlp", shift_mlp),
                            ("adaln.scale_mlp", scale_mlp), ("adaln.gate_mlp", gate_mlp)):
            _h3_check_finite(name, value)
        h = _h3_check_finite("norm1", _h3_branch_rms_norm(self.norm1, x))
        h = _h3_check_finite("modulate_attn", _mod_scale_shift(h, shift_msa, scale_msa, mod_segments))
        x = _h3_check_finite(
            "residual_attn",
            _mod_gate(x, gate_msa,
                      self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
                      mod_segments),
        )
        h = _h3_check_finite("norm2", _h3_branch_rms_norm(self.norm2, x))
        h = _h3_check_finite("modulate_mlp", _mod_scale_shift(h, shift_mlp, scale_mlp, mod_segments))
        return _h3_check_finite("residual_mlp", _mod_gate(x, gate_mlp, self.mlp(h), mod_segments))


class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True, adaln_dtype=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                    device=device, operations=operations)
        # output heads are the checkpoint's fp32 island; norm/adaln are stored at model dtype
        self.video_out = operations.Linear(hidden, video_dim, bias=True, dtype=torch.float32, device=device)
        self.audio_out = operations.Linear(hidden, audio_dim, bias=True, dtype=torch.float32, device=device)

    def forward(self, x, t_emb, video_seg, audio_seg):
        # video_seg / audio_seg: (start, stop, timestep_row) of the target streams
        shift, scale = self.adaln_proj(t_emb)
        va, vb, vrow = video_seg
        aa, ab, arow = audio_seg
        hv = (_safe_fp16_rms_norm(self.norm, x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]).to(torch.float32)
        ha = (_safe_fp16_rms_norm(self.norm, x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]).to(torch.float32)
        return self.video_out(hv), self.audio_out(ha)


class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]  # (kind, n_rows)
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]  # per segment: [n, 3] float64 (t, h, w)

        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        cursor = text_len
        row = text_len

        if keyframes:
            # fl2va: keyframe cond rows right after text, sharing the target spatial grid
            for kf in keyframes:
                pixel_index = kf["resolved_frame_index"]
                if pixel_index == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and pixel_index == frame_count - 1:
                    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                segments.append(("cond", frame_rows))
                pos.append(g)
                img_pos.append(torch.arange(row, row + frame_rows))
                img_update.append(torch.zeros(frame_rows, dtype=torch.bool))
                row += frame_rows

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    # the block's audio rows pack immediately before its video
                    # rows, both sharing the cursor origin
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(_video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += max(float(rt), sum(_video_t_spans(vt)))

        # target audio then target video, always the last two segments
        segments.append(("audio", audio_t * 2))
        pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(_video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)  # [S, 3] float64
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        # contiguous segment table (start, stop, kind)
        # kinds: text / cond / ref_img / ref_audio / audio / video
        # the packed sequence is uniform per segment in (modality tag, timestep class),
        # except the text span (tag runs resolved at forward time from the presentation tags)
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


class MiniMaxH3Model(nn.Module):
    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
                 rope_inv_freq_len=16, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
                 sigma_shift_video=12.0, sigma_shift_audio=3.0,
                 adaln_curve_grid=None,
                 image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        self.use_adaln_curves = adaln_curve_grid is not None
        # curve-form checkpoints replace the time embedder and full-width adaln weights with a small shared basis of the time-embedding curve
        curve = {"apply_silu": not self.use_adaln_curves,
                 "adaln_dtype": torch.float32 if self.use_adaln_curves else dtype}
        video_patch_dim = latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = operations.Linear(video_patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.audio_patch_proj = operations.Linear(audio_latents_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.condition_proj = operations.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32))
        else:
            self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim,
                                              dtype=torch.float32, device=device, operations=operations)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32))
        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,
                                          final_norm_eps, dtype=dtype, device=device, operations=operations)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                     time_embed_dim, norm_eps, qk_norm_eps, **curve, dtype=dtype, device=device, operations=operations)
            for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, time_embed_dim, video_patch_dim, audio_latents_dim,
                                      final_norm_eps, **curve, dtype=dtype, device=device, operations=operations)

    def _prepare_pipeline_parallel(self, source_device):
        """Place contiguous H3 blocks on two CUDA devices for a PP forward.

        This deliberately is *not* tensor parallelism: a complete residual
        stream crosses NVLink once at the block boundary.  Keeping each GGUF
        block intact matters because its compressed rows are dequantized on the
        device of the input activation.  Existing generic MultiGPU placement
        cannot express that ownership relation.
        """
        if _H3_PARALLEL_MODE == "single":
            return None
        if _H3_PARALLEL_MODE != "pp":
            raise RuntimeError(
                f"Unsupported H3_PARALLEL_MODE={_H3_PARALLEL_MODE!r}; "
                "supported modes are single and pp."
            )
        if source_device.type != "cuda" or source_device.index is None:
            raise RuntimeError("H3 pipeline parallel requires a concrete CUDA source device")
        peer_device = torch.device(_H3_PP_DEVICE)
        if peer_device.type != "cuda" or peer_device.index is None or peer_device == source_device:
            raise RuntimeError(
                f"H3 pipeline peer must be a different CUDA device; got source={source_device}, "
                f"peer={peer_device}"
            )
        if not torch.cuda.can_device_access_peer(source_device, peer_device):
            raise RuntimeError(f"H3 pipeline requires CUDA P2P: {source_device} -> {peer_device} is unavailable")
        if not torch.cuda.can_device_access_peer(peer_device, source_device):
            raise RuntimeError(f"H3 pipeline requires CUDA P2P: {peer_device} -> {source_device} is unavailable")
        if not 0 < _H3_PP_SPLIT < len(self.blocks):
            raise RuntimeError(
                f"H3_PP_SPLIT must be within 1..{len(self.blocks) - 1}; got {_H3_PP_SPLIT}"
            )

        # Text encoding has already produced ``context`` by the time this model
        # is entered.  Reclaim GPU1 before moving the second DiT stage there;
        # otherwise the static 32B Qwen GGUF consumes the capacity the PP stage
        # needs.  The diffusion patcher itself is registered on source_device,
        # so it is intentionally not a candidate for this eviction.
        peer_models_loaded = any(
            loaded.device == peer_device and loaded.model is not None
            for loaded in comfy.model_management.current_loaded_models
        )
        if peer_models_loaded:
            released = comfy.model_management.free_memory(1e30, peer_device)
            print(
                f"[H3PP] released {len(released)} completed model(s) from {peer_device} before sampling",
                flush=True,
            )

        # A model may have been reloaded by ComfyUI since the prior prompt; use
        # the real parameter device rather than a sticky boolean to keep the
        # partition correct in both warm and reload paths.
        for index, block in enumerate(self.blocks):
            target = source_device if index < _H3_PP_SPLIT else peer_device
            if _h3_module_device(block) != target:
                _h3_to_device(block, target)
        if _h3_module_device(self.final_layer) != peer_device:
            _h3_to_device(self.final_layer, peer_device)

        print(
            f"[H3PP] active: blocks 0-{_H3_PP_SPLIT - 1} on {source_device}, "
            f"blocks {_H3_PP_SPLIT}-{len(self.blocks) - 1} + final on {peer_device}",
            flush=True,
        )
        return peer_device

    def _refine_text_states(self, text_states, transformer_options={}):
        """Project/refine [L, text_dim] states with a narrow V100 FP32 island.

        The 5120 -> 5376 projection can legitimately exceed FP16's 65504
        output range for a small number of Qwen tokens.  It is only evaluated
        once per prompt (tens of rows), so retaining the projection and two
        refiner blocks in FP32 costs negligible runtime/VRAM while preserving
        the Q4+FP16 main DiT path.
        """
        input_dtype = text_states.dtype
        fp32_text = text_states.float() if input_dtype == torch.float16 else text_states
        projected = _h3_check_finite("preprocess_text.condition_proj", self.condition_proj(fp32_text))
        refined = _h3_check_finite(
            "preprocess_text.refined_fp32",
            self.token_refiner(projected, transformer_options=transformer_options),
        )
        refined = _h3_check_finite("preprocess_text.output", refined.to(input_dtype))
        return refined

    def preprocess_text_embeds(self, text_states):
        """[B, L, text_dim] Qwen states -> [B, L, hidden] refined text embeds."""
        _h3_check_finite("preprocess_text.input", text_states)
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self._refine_text_states(text_states[0]).unsqueeze(0)

    def rope_freqs(self, position_ids, device):
        # [S, 3] float64 -> [S, 96] fp32
        pos = position_ids.to(torch.float32).to(device)
        inv = comfy.model_management.cast_to(self.rope.inv_freq, device=device)
        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)      # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)              # [S, 48]
        return torch.cat((half, half), dim=-1)                 # [S, 96]

    def _cond_video_rows(self, payload, device):
        """Concatenated visual condition rows (normalized latents -> patchified), with condition noise augmentation."""
        rows = []
        aug = payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP)
        seed = int(payload.get("seed", 0))
        # every condition intentionally restarts the same RNG stream
        for z in payload.get("cond_video_latents", []):
            r = patchify_video(z.to(torch.float32), self.patch_size)
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    def _cond_audio_rows(self, payload, device):
        rows = []
        aug = payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP)
        seed = int(payload.get("seed", 0)) + 1
        for z in payload.get("cond_audio_latents", []):
            r = pack_audio(z.to(torch.float32))
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        # the sampler carries the audio as (sigma_v / sigma_a) * x_audio; undo it outside
        # the wrappers so they and the network see the stream's own latent and velocity
        scale = float((minimax_payload or {}).get("audio_scale", 1.0))
        audio_src = x[1]
        if scale != 1.0:
            shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
            shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
            sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
            carry = (sigma_a / sigma_v).to(audio_src.dtype)
            x = [x[0], audio_src * carry]

        out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

        if scale != 1.0:
            # d/d(sigma_v) of the carried variable
            out[1] = ((1.0 - scale) * (audio_src * carry)
                      + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1])
        return out

    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        video_x, audio_x = x[0], x[1]
        _h3_check_finite("input.video", video_x)
        _h3_check_finite("input.audio", audio_x)
        _h3_check_finite("input.context", context)
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype  # compute dtype
        pp_peer_device = self._prepare_pipeline_parallel(device)
        if _H3_PARALLEL_PROFILE:
            # Partition moves are intentionally outside the denoiser timing: the
            # first warmup forward may establish the placement, while the metric
            # below reports the steady-state work done for each diffusion step.
            torch.cuda.synchronize(device)
            if pp_peer_device is not None:
                torch.cuda.synchronize(pp_peer_device)
                torch.cuda.reset_peak_memory_stats(pp_peer_device)
            torch.cuda.reset_peak_memory_stats(device)
            profile_started = time.perf_counter()
        else:
            profile_started = None

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]
        # extra_conds prebuilds the layout once per sampling run
        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                  keyframes=payload.get("keyframes"),
                                  refs=payload.get("refs"),
                                  frame_count=payload.get("frame_count"))

        # model_base passes model_sampling.timestep(sigma) = sigma * 1000
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

        # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
        has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
        seg_t = {"text": t_v, "video": t_v, "audio": t_a,
                 "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                 "ref_audio": max(t_a, aud_aug)}
        unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set())
                          | ({seg_t["ref_audio"]} if has_aud_cond else set()))
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                # the presentation text span mixes tags (vision pads carry the video modality) split into tag runs
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        # embed
        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        cond_video_rows = self._cond_video_rows(payload, device)
        cond_audio_rows = self._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = _h3_check_finite("embed.video", self.video_patch_proj(all_video_rows).to(dtype))
        audio_embed = _h3_check_finite("embed.audio", self.audio_patch_proj(all_audio_rows).to(dtype))
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self._refine_text_states(text_states, transformer_options=transformer_options)

        # segments are contiguous: assemble by slices, embed rows follow segment order
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:  # ref_audio / audio
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n

        # The patch/conditioning embeddings have been copied into the packed
        # residual stream above.  They are large at 720p (the video embedding
        # alone is hundreds of MiB), but no later stage reads them.  Drop all
        # scratch references before the first DiT block so the QKV temporary,
        # NCCL buffers, and FP32 residual do not overlap these dead tensors.
        # This is lifetime-only: values and dtypes are unchanged.
        del (
            img_update,
            audio_update,
            video_rows,
            audio_rows,
            cond_video_rows,
            cond_audio_rows,
            all_video_rows,
            all_audio_rows,
            video_embed,
            audio_embed,
            text_states,
            context,
        )
        # H3 was trained with a wide BF16 residual stream.  On V100 the
        # branch kernels stay FP16, but retaining the residual in FP32 avoids
        # legitimate residual additions overflowing FP16 before the next norm.
        if _H3_FP32_RESIDUAL:
            h = h.float()
        _h3_check_finite("embed.packed", h)

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            # adaln projections consume interpolated coordinates of the time-embedding curve
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)     # t in [0,1] -> fractional grid index, out-of-range t clamps to the curve ends
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)   # lower grid row, max-clamp keeps t=1.0 on the last interval instead of reading past the table
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))  # blend the two rows by the fractional part
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)
        _h3_check_finite("time_embed", t_emb)

        # rotation table computed once per forward, consumed by the kitchen split-half rope
        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
        _h3_check_finite("rope", rope_freqs)

        # blocks
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        # ComfyUI's generic queue assumes every block is on ``device``.  In
        # pipeline mode it would pull stage 2 back to GPU0, so keep the already
        # resident GGUF blocks in place instead.
        prefetch_queue = None if pp_peer_device is not None else comfy.model_prefetch.make_prefetch_queue(
            list(self.blocks), device, transformer_options)
        block_device = device
        block_t_emb = t_emb
        block_rope_freqs = rope_freqs
        pp_boundary_seconds = None
        global _H3_TRACE_BLOCK
        for i, block in enumerate(self.blocks):
            if pp_peer_device is not None and i == _H3_PP_SPLIT:
                # This is the one full-residual PP handoff per denoise forward.
                # Explicit P2P copies preserve the FP32 V100 stability stream;
                # no CPU staging or lossy FP16 cast is involved.
                if _H3_PARALLEL_PROFILE:
                    torch.cuda.synchronize(device)
                    pp_boundary_started = time.perf_counter()
                h = h.to(pp_peer_device, non_blocking=True)
                block_t_emb = t_emb.to(pp_peer_device, non_blocking=True)
                block_rope_freqs = rope_freqs.to(pp_peer_device, non_blocking=True)
                block_device = pp_peer_device
                if _H3_PARALLEL_PROFILE:
                    torch.cuda.synchronize(pp_peer_device)
                    torch.cuda.synchronize(device)
                    pp_boundary_seconds = time.perf_counter() - pp_boundary_started
            previous_trace_block = _H3_TRACE_BLOCK
            _H3_TRACE_BLOCK = f"dit[{i}]"
            try:
                # Some GGUF/rope kernels inspect torch.cuda.current_device(),
                # not just their tensor arguments.  The enclosing sampler stays
                # on GPU0, so explicitly enter the owning stage for every block.
                with comfy.model_management.cuda_device_context(block_device):
                    _h3_check_finite("block_input", h)
                    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, block_device, block)
                    if ("double_block", i) in blocks_replace:
                        def block_wrap(args):
                            return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                                 transformer_options=args["transformer_options"])}
                        h = blocks_replace[("double_block", i)](
                            {"img": h, "t_emb": block_t_emb, "mod_segments": mod_segments, "rope_freqs": block_rope_freqs,
                             "transformer_options": transformer_options},
                            {"original_block": block_wrap})["img"]
                    else:
                        h = block(h, block_t_emb, mod_segments, block_rope_freqs, transformer_options=transformer_options)
                    _h3_check_finite("block_output", h)
            finally:
                _H3_TRACE_BLOCK = previous_trace_block
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, block_device, None)

        # target streams are single contiguous segments (audio then video, last two)
        video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
        audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
        with comfy.model_management.cuda_device_context(block_device):
            v, a = self.final_layer(h, block_t_emb, video_seg, audio_seg)
        _h3_check_finite("final.video", v)
        _h3_check_finite("final.audio", a)

        if pp_peer_device is not None:
            # Sampler state lives on GPU0, so return only the small predicted
            # video/audio latent heads after stage 2.  This is tens of MiB, not
            # a second full hidden stream.
            v = v.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)

        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(a)

        if profile_started is not None:
            torch.cuda.synchronize(device)
            if pp_peer_device is not None:
                torch.cuda.synchronize(pp_peer_device)
            elapsed = time.perf_counter() - profile_started
            mem0 = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            mem1 = (torch.cuda.max_memory_allocated(pp_peer_device) / (1024 ** 3)
                    if pp_peer_device is not None else 0.0)
            boundary = f"{pp_boundary_seconds:.4f}s" if pp_boundary_seconds is not None else "n/a"
            print(
                f"[H3PROFILE mode={_H3_PARALLEL_MODE}] denoise_forward={elapsed:.3f}s "
                f"pp_boundary={boundary} peak_alloc_gib=({mem0:.2f},{mem1:.2f})",
                flush=True,
            )

        return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
