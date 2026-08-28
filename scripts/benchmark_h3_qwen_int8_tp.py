#!/usr/bin/env python3
"""One-layer real-weight Qwen3-VL-4B INT8 ConvRot TP gate.

This is a bounded test: it reads only language-model layer 0 from the
header-only safetensors reader, never maps the checkpoint and never builds the
full encoder.  It compares one complete quantized block on one V100 with a
two-GPU tensor-parallel copy of the same block:

  Q/K/V and gate/up: output-row shards
  O and down: input-column shards
  attention and FFN partials: NCCL-backed in-process reduce + broadcast

The result is a real-kernel feasibility check, not a production loader.  A
full Qwen TP integration still needs direct sharded materialisation, vision
tower placement, deepstack handling, and a resident cache.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch
import torch.cuda.comm as cuda_comm
import torch.nn as nn
import torch.nn.functional as F

from custom_nodes.NoHostMMap.safetensors import _read_header, _read_into
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout


MIB = 2**20
DEV0 = torch.device("cuda:0")
DEV1 = torch.device("cuda:1")
BENCH_ATTN_MODE = os.environ.get("H3_QWEN_BENCH_ATTN", "causal")
BENCH_TP_FP32_PARTIAL = os.environ.get("H3_QWEN_BENCH_FP32_PARTIAL", "0").lower() in {
    "1", "true", "yes", "on"
}


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return x * inv.to(dtype=x.dtype) * weight


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask=None) -> torch.Tensor:
    if BENCH_ATTN_MODE == "masked":
        if attention_mask is None:
            size = q.shape[-2]
            attention_mask = torch.empty(
                (size, size), device=q.device, dtype=q.dtype
            ).fill_(torch.finfo(q.dtype).min / 4).triu_(1)
        import comfy.ops

        return comfy.ops.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    if BENCH_ATTN_MODE != "causal":
        raise ValueError(f"unsupported H3_QWEN_BENCH_ATTN={BENCH_ATTN_MODE!r}")
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class QuantLinear(nn.Module):
    def __init__(self, weight: QuantizedTensor, device: torch.device):
        super().__init__()
        self.weight = weight
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class FullQwenBlock(nn.Module):
    def __init__(self, weights: dict[str, tuple[torch.Tensor, torch.Tensor]], norms: dict[str, torch.Tensor], device: torch.device):
        super().__init__()
        self.hidden = 2560
        self.heads = 32
        self.kv_heads = 8
        self.head_dim = 128
        self.eps = 1e-6
        self.input_norm = nn.Parameter(norms["input"].to(device), requires_grad=False)
        self.post_norm = nn.Parameter(norms["post"].to(device), requires_grad=False)
        self.q_norm = nn.Parameter(norms["q"].to(device), requires_grad=False)
        self.k_norm = nn.Parameter(norms["k"].to(device), requires_grad=False)
        for name, (qdata, scale) in weights.items():
            qt = _qt(qdata.to(device), scale.to(device), tuple(qdata.shape))
            setattr(self, name, QuantLinear(qt, device))

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        b, s, _ = x.shape
        h = _rms(x, self.input_norm, self.eps)
        q = self.q_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = _rms(q, self.q_norm, self.eps)
        k = _rms(k, self.k_norm, self.eps)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        attn = _attention(q, k, v, attention_mask)
        attn = attn.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        x = x + self.o_proj(attn)
        h = _rms(x, self.post_norm, self.eps)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class LocalQwenBlock(nn.Module):
    def __init__(
        self,
        rank: int,
        weights: dict[str, tuple[torch.Tensor, torch.Tensor]],
        norms: dict[str, torch.Tensor],
        device: torch.device,
    ):
        super().__init__()
        self.rank = rank
        self.hidden = 2560
        self.local_heads = 16
        self.local_kv_heads = 4
        self.head_dim = 128
        self.local_inner = 2048
        self.local_kv_inner = 512
        self.local_ff = 4864
        self.eps = 1e-6
        self.input_norm = nn.Parameter(norms["input"].to(device), requires_grad=False)
        self.post_norm = nn.Parameter(norms["post"].to(device), requires_grad=False)
        self.q_norm = nn.Parameter(norms["q"].to(device), requires_grad=False)
        self.k_norm = nn.Parameter(norms["k"].to(device), requires_grad=False)
        for name, (qdata, scale) in weights.items():
            qt = _qt(qdata.to(device), scale.to(device), tuple(qdata.shape))
            setattr(self, name, QuantLinear(qt, device))

    def forward_attention(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        b, s, _ = x.shape
        h = _rms(x, self.input_norm, self.eps)
        q = self.q_proj(h).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.local_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.local_kv_heads, self.head_dim).transpose(1, 2)
        q = _rms(q, self.q_norm, self.eps)
        k = _rms(k, self.k_norm, self.eps)
        k = k.repeat_interleave(self.local_heads // self.local_kv_heads, dim=1)
        v = v.repeat_interleave(self.local_heads // self.local_kv_heads, dim=1)
        attn = _attention(q, k, v, attention_mask)
        attn = attn.transpose(1, 2).reshape(b, s, self.local_inner)
        output = self.o_proj(attn)
        return output.float() if BENCH_TP_FP32_PARTIAL else output

    def forward_ff(self, x: torch.Tensor) -> torch.Tensor:
        h = _rms(x, self.post_norm, self.eps)
        output = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return output.float() if BENCH_TP_FP32_PARTIAL else output


def _qt(qdata: torch.Tensor, scale: torch.Tensor, shape: tuple[int, int]) -> QuantizedTensor:
    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.float16,
        orig_shape=shape,
        is_weight=True,
        convrot=True,
        convrot_groupsize=256,
    )
    return QuantizedTensor(qdata, "TensorWiseINT8Layout", params)


def _read_cpu(file_ref, header: dict, base: int, key: str, dtype: torch.dtype) -> torch.Tensor:
    entry = header[key]
    start, end = entry["data_offsets"]
    element_size = torch.empty((), dtype=dtype).element_size()
    tensor = torch.empty((end - start) // element_size, dtype=dtype)
    _read_into(file_ref, base + start, tensor)
    return tensor.reshape(entry["shape"])


def _load_layer(path: Path, layer_index: int = 0) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, torch.Tensor], int]:
    file_ref, header, base = _read_header(str(path))
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    # All seven matrices in Qwen3-VL-4B's first language block are int8
    # ConvRot.  The names are intentionally explicit: a changed checkpoint
    # shape must fail loudly instead of silently benchmarking another model.
    prefixes = {
        "q_proj": f"model.language_model.layers.{layer_index}.self_attn.q_proj",
        "k_proj": f"model.language_model.layers.{layer_index}.self_attn.k_proj",
        "v_proj": f"model.language_model.layers.{layer_index}.self_attn.v_proj",
        "o_proj": f"model.language_model.layers.{layer_index}.self_attn.o_proj",
        "gate_proj": f"model.language_model.layers.{layer_index}.mlp.gate_proj",
        "up_proj": f"model.language_model.layers.{layer_index}.mlp.up_proj",
        "down_proj": f"model.language_model.layers.{layer_index}.mlp.down_proj",
    }
    for name, prefix in prefixes.items():
        q = _read_cpu(file_ref, header, base, prefix + ".weight", torch.int8)
        scale = _read_cpu(file_ref, header, base, prefix + ".weight_scale", torch.float32)
        weights[name] = (q, scale)
    norms = {
        "input": _read_cpu(file_ref, header, base, f"model.language_model.layers.{layer_index}.input_layernorm.weight", torch.bfloat16).to(torch.float16),
        "post": _read_cpu(file_ref, header, base, f"model.language_model.layers.{layer_index}.post_attention_layernorm.weight", torch.bfloat16).to(torch.float16),
        "q": _read_cpu(file_ref, header, base, f"model.language_model.layers.{layer_index}.self_attn.q_norm.weight", torch.bfloat16).to(torch.float16),
        "k": _read_cpu(file_ref, header, base, f"model.language_model.layers.{layer_index}.self_attn.k_norm.weight", torch.bfloat16).to(torch.float16),
    }
    try:
        os.posix_fadvise(file_ref.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass
    file_ref.close()
    return weights, norms, len(header)


def _shard_weights(weights: dict[str, tuple[torch.Tensor, torch.Tensor]], rank: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (q, scale) in weights.items():
        if name in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}:
            rows = q.shape[0] // 2
            sl = slice(rank * rows, (rank + 1) * rows)
            out[name] = (q[sl].contiguous(), scale[sl].contiguous())
        elif name in {"o_proj", "down_proj"}:
            cols = q.shape[1] // 2
            sl = slice(rank * cols, (rank + 1) * cols)
            out[name] = (q[:, sl].contiguous(), scale.contiguous())
        else:
            raise KeyError(name)
    return out


def _err(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    # ``tp_cross_gpu`` intentionally compares the two rank outputs.  PyTorch
    # does not insert a peer copy for arithmetic between cuda:0/cuda:1, so
    # make that boundary explicit.  This is only a small diagnostic output;
    # the measured TP path itself remains entirely device-local between
    # collectives.
    if candidate.device != reference.device:
        candidate = candidate.to(reference.device)
    delta = candidate.float() - reference.float()
    ref = reference.float()
    rms = torch.sqrt(delta.square().mean())
    denom = torch.sqrt(ref.square().mean()).clamp_min(1e-12)
    cosine = F.cosine_similarity(ref.reshape(1, -1), candidate.float().reshape(1, -1))[0]
    return {
        "max_abs": float(delta.abs().max().item()),
        "relative_rms": float((rms / denom).item()),
        "cosine": float(cosine.item()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def _sync() -> None:
    torch.cuda.synchronize(DEV0)
    torch.cuda.synchronize(DEV1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/GALAX/minimax-h3/models/text_encoders/qwen3vl_4b_int8_convrot.safetensors"),
    )
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("requires two CUDA devices")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    torch.cuda.set_device(DEV0)
    torch.manual_seed(20260825)
    weights, norms, header_count = _load_layer(args.model)
    x0 = torch.randn((1, args.sequence, 2560), device=DEV0, dtype=torch.float16)
    x1 = x0.to(DEV1)

    # Full block on GPU0: this is the per-layer baseline of the existing
    # layer/pipeline MP route.
    full = FullQwenBlock(weights, norms, DEV0).eval()
    with torch.inference_mode():
        for _ in range(args.warmup):
            full(x0)
        _sync()
        torch.cuda.reset_peak_memory_stats(DEV0)
        start = time.perf_counter()
        for _ in range(args.repetitions):
            full_out = full(x0)
        _sync()
        full_ms = (time.perf_counter() - start) * 1000.0 / args.repetitions
        full_out = full_out.detach()
    full_peak = torch.cuda.max_memory_allocated(DEV0) / MIB
    del full
    torch.cuda.empty_cache()
    _sync()

    local0 = LocalQwenBlock(0, _shard_weights(weights, 0), norms, DEV0).eval()
    local1 = LocalQwenBlock(1, _shard_weights(weights, 1), norms, DEV1).eval()

    def tp_forward(a0: torch.Tensor, a1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p0 = local0.forward_attention(a0)
        with torch.cuda.device(DEV1):
            p1 = local1.forward_attention(a1)
        with torch.cuda.device(DEV0):
            reduced = cuda_comm.reduce_add([p0, p1], destination=0)
        bcast = cuda_comm.broadcast(
            reduced,
            out=(torch.empty_like(reduced, device=DEV0), torch.empty_like(reduced, device=DEV1)),
        )
        a0 = a0 + bcast[0]
        a1 = a1 + bcast[1]
        p0 = local0.forward_ff(a0)
        with torch.cuda.device(DEV1):
            p1 = local1.forward_ff(a1)
        with torch.cuda.device(DEV0):
            reduced = cuda_comm.reduce_add([p0, p1], destination=0)
        bcast = cuda_comm.broadcast(
            reduced,
            out=(torch.empty_like(reduced, device=DEV0), torch.empty_like(reduced, device=DEV1)),
        )
        return a0 + bcast[0], a1 + bcast[1]

    with torch.inference_mode():
        for _ in range(args.warmup):
            tp_out0, tp_out1 = tp_forward(x0, x1)
        _sync()
        torch.cuda.reset_peak_memory_stats(DEV0)
        torch.cuda.reset_peak_memory_stats(DEV1)
        start = time.perf_counter()
        for _ in range(args.repetitions):
            tp_out0, tp_out1 = tp_forward(x0, x1)
        _sync()
        tp_ms = (time.perf_counter() - start) * 1000.0 / args.repetitions
    tp_peak = [torch.cuda.max_memory_allocated(DEV0) / MIB, torch.cuda.max_memory_allocated(DEV1) / MIB]

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hardware": "2x Tesla V100 expected",
        "model": str(args.model),
        "header_tensors": header_count,
        "checkpoint_scope": "language_model.layers.0 only",
        "host_mmap": False,
        "shape": [1, args.sequence, 2560],
        "full_layer_mp_ms": full_ms,
        "strict_tp_ms": tp_ms,
        "speedup_full_over_tp": full_ms / tp_ms if tp_ms else None,
        "full_peak_mib_gpu0": full_peak,
        "tp_peak_mib": tp_peak,
        "full_vs_tp": _err(full_out, tp_out0),
        "tp_cross_gpu": _err(tp_out0, tp_out1),
        "finite": bool(torch.isfinite(tp_out0).all().item() and torch.isfinite(tp_out1).all().item()),
        "numerically_qualified": bool(
            torch.isfinite(tp_out0).all().item()
            and torch.isfinite(tp_out1).all().item()
            and _err(full_out, tp_out0)["relative_rms"] <= 3e-3
            and _err(full_out, tp_out0)["cosine"] >= 0.9999
            and _err(tp_out0, tp_out1)["max_abs"] == 0.0
        ),
        "note": "real INT8 ConvRot weights; one block; no vision tower; no checkpoint mmap",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
