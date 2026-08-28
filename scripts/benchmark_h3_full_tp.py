#!/usr/bin/env python3
"""Safe feasibility benchmark for extending H3 tensor parallelism.

This benchmark does not open a checkpoint.  It uses the dimensions and
operator shapes of Qwen3-VL-4B and the H3 ViT video decoder, then compares:

* ``layer``: the current useful model-parallel route (full blocks, 12/24 by
  default, activation handoff once at the split), and
* ``tp``: head/column-parallel input projections, row-parallel output
  projections, and two NCCL all-reduces per transformer block.

The purpose is to answer whether *strict* TP is worth integrating.  The
synthetic weights make this a compute/communication gate, not a quality claim.
Use ``--correctness`` for a small exact full-vs-sharded numerical check before
running the production-sized synthetic timing case.

Examples:

  python scripts/benchmark_h3_full_tp.py --module qwen --route layer \
      --sequence 256 --output results/qwen_layer.json
  torchrun --standalone --nproc_per_node=2 scripts/benchmark_h3_full_tp.py \
      --module qwen --route tp --sequence 256 --output results/qwen_tp.json

No checkpoint, safetensors reader, mmap, or host staging buffer is used here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


WORLD = 2
MIB = 2**20


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Keep the reduction in FP32, matching the safety direction used by the
    # production V100 routes.  The result returns to the activation dtype.
    inv = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return x * inv.to(dtype=x.dtype) * weight


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return F.linear(x, weight, bias)


class FullQwenBlock(nn.Module):
    """A Qwen3-VL language block with the production 4B dimensions."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        intermediate: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        inner = heads * head_dim
        kv_inner = kv_heads * head_dim
        self.hidden = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.eps = 1e-6
        self.input_norm = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.post_norm = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.q_proj = nn.Linear(hidden, inner, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden, kv_inner, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden, kv_inner, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(inner, hidden, bias=False, device=device, dtype=dtype)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        h = rms_norm(x, self.input_norm, self.eps)
        q = self.q_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        x = x + self.o_proj(attn)
        h = rms_norm(x, self.post_norm, self.eps)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class TPQwenBlock(nn.Module):
    """Two-way Qwen TP block: head/column + row parallel."""

    def __init__(
        self,
        rank: int,
        hidden: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        intermediate: int,
        device: torch.device,
        dtype: torch.dtype,
        source: FullQwenBlock | None = None,
    ) -> None:
        super().__init__()
        if heads % WORLD or kv_heads % WORLD or intermediate % WORLD:
            raise ValueError("Qwen TP dimensions must divide by two")
        self.rank = rank
        self.hidden = hidden
        self.local_heads = heads // WORLD
        self.local_kv_heads = kv_heads // WORLD
        self.head_dim = head_dim
        self.local_inner = self.local_heads * head_dim
        self.local_kv_inner = self.local_kv_heads * head_dim
        self.eps = 1e-6
        self.input_norm = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.post_norm = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.q_proj = nn.Linear(hidden, self.local_inner, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden, self.local_kv_inner, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden, self.local_kv_inner, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.local_inner, hidden, bias=False, device=device, dtype=dtype)
        self.local_intermediate = intermediate // WORLD
        self.gate_proj = nn.Linear(hidden, self.local_intermediate, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden, self.local_intermediate, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(self.local_intermediate, hidden, bias=False, device=device, dtype=dtype)
        if source is not None:
            self._copy_from_full(source)

    @torch.no_grad()
    def _copy_from_full(self, source: FullQwenBlock) -> None:
        rank = self.rank
        heads = source.heads
        kv_heads = source.kv_heads
        intermediate = source.gate_proj.out_features
        h0 = rank * (heads // WORLD) * source.head_dim
        h1 = (rank + 1) * (heads // WORLD) * source.head_dim
        k0 = rank * (kv_heads // WORLD) * source.head_dim
        k1 = (rank + 1) * (kv_heads // WORLD) * source.head_dim
        f0 = rank * (intermediate // WORLD)
        f1 = (rank + 1) * (intermediate // WORLD)
        self.input_norm.copy_(source.input_norm)
        self.post_norm.copy_(source.post_norm)
        self.q_proj.weight.copy_(source.q_proj.weight[h0:h1])
        self.k_proj.weight.copy_(source.k_proj.weight[k0:k1])
        self.v_proj.weight.copy_(source.v_proj.weight[k0:k1])
        self.o_proj.weight.copy_(source.o_proj.weight[:, h0:h1])
        self.gate_proj.weight.copy_(source.gate_proj.weight[f0:f1])
        self.up_proj.weight.copy_(source.up_proj.weight[f0:f1])
        self.down_proj.weight.copy_(source.down_proj.weight[:, f0:f1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        h = rms_norm(x, self.input_norm, self.eps)
        q = self.q_proj(h).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.local_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.local_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.local_heads // self.local_kv_heads, dim=1)
        v = v.repeat_interleave(self.local_heads // self.local_kv_heads, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(b, s, self.local_inner)
        partial = self.o_proj(attn)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        x = x + partial
        h = rms_norm(x, self.post_norm, self.eps)
        partial = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return x + partial


class FullVAEBlock(nn.Module):
    """One H3 ViT3D decoder block (hidden=2048, 32 heads x 64)."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if hidden != heads * head_dim:
            raise ValueError("VAE hidden must equal heads * head_dim")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.eps = 1e-5
        self.norm1 = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.norm2 = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.scale1 = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.scale2 = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.q_proj = nn.Linear(hidden, hidden, bias=True, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden, hidden, bias=True, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden, hidden, bias=True, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden, hidden, bias=True, device=device, dtype=dtype)
        self.w1 = nn.Linear(hidden, hidden * 4 * 2, bias=True, device=device, dtype=dtype)
        self.w2 = nn.Linear(hidden * 4, hidden, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        h = rms_norm(x, self.norm1, self.eps)
        q = self.q_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, s, self.hidden)
        x = x + self.o_proj(attn) * self.scale1
        h = rms_norm(x, self.norm2, self.eps)
        gate, value = self.w1(h).chunk(2, dim=-1)
        return x + self.w2(F.silu(gate) * value) * self.scale2


class TPVAEBlock(nn.Module):
    """Two-way VAE TP block with head and FFN channel sharding."""

    def __init__(
        self,
        rank: int,
        hidden: int,
        heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        source: FullVAEBlock | None = None,
    ) -> None:
        super().__init__()
        if heads % WORLD or hidden % WORLD:
            raise ValueError("VAE TP dimensions must divide by two")
        self.rank = rank
        self.hidden = hidden
        self.local_heads = heads // WORLD
        self.head_dim = head_dim
        self.local_inner = self.local_heads * head_dim
        self.eps = 1e-5
        local_ff = hidden * 4 // WORLD
        self.local_ff = local_ff
        self.norm1 = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.norm2 = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.scale1 = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.scale2 = nn.Parameter(torch.ones(hidden, device=device, dtype=dtype))
        self.q_proj = nn.Linear(hidden, self.local_inner, bias=True, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden, self.local_inner, bias=True, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden, self.local_inner, bias=True, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.local_inner, hidden, bias=False, device=device, dtype=dtype)
        self.o_bias = nn.Parameter(torch.zeros(hidden, device=device, dtype=dtype))
        self.w1 = nn.Linear(hidden, local_ff * 2, bias=True, device=device, dtype=dtype)
        self.w2 = nn.Linear(local_ff, hidden, bias=False, device=device, dtype=dtype)
        self.w2_bias = nn.Parameter(torch.zeros(hidden, device=device, dtype=dtype))
        if source is not None:
            self._copy_from_full(source)

    @torch.no_grad()
    def _copy_from_full(self, source: FullVAEBlock) -> None:
        rank = self.rank
        h0 = rank * self.local_inner
        h1 = (rank + 1) * self.local_inner
        f0 = rank * self.local_ff
        f1 = (rank + 1) * self.local_ff
        self.norm1.copy_(source.norm1)
        self.norm2.copy_(source.norm2)
        self.scale1.copy_(source.scale1)
        self.scale2.copy_(source.scale2)
        for dst, src in ((self.q_proj, source.q_proj), (self.k_proj, source.k_proj), (self.v_proj, source.v_proj)):
            dst.weight.copy_(src.weight[h0:h1])
            dst.bias.copy_(src.bias[h0:h1])
        self.o_proj.weight.copy_(source.o_proj.weight[:, h0:h1])
        self.o_bias.copy_(source.o_proj.bias)
        # H3's gated FFN stores all gate rows followed by all value rows;
        # ``chunk(2, dim=-1)`` in forward splits the *output* dimension.  A
        # TP rank therefore needs the local gate slice and the matching local
        # value slice, concatenated in that same order.
        gate_rows = source.w1.weight[f0:f1]
        value_rows = source.w1.weight[source.w1.out_features // 2 + f0:source.w1.out_features // 2 + f1]
        gate_bias = source.w1.bias[f0:f1]
        value_bias = source.w1.bias[source.w1.out_features // 2 + f0:source.w1.out_features // 2 + f1]
        self.w1.weight.copy_(torch.cat((gate_rows, value_rows), dim=0))
        self.w1.bias.copy_(torch.cat((gate_bias, value_bias), dim=0))
        self.w2.weight.copy_(source.w2.weight[:, f0:f1])
        self.w2_bias.copy_(source.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        h = rms_norm(x, self.norm1, self.eps)
        q = self.q_proj(h).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.local_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, s, self.local_inner)
        partial = self.o_proj(attn)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        x = x + (partial + self.o_bias) * self.scale1
        h = rms_norm(x, self.norm2, self.eps)
        gate, value = self.w1(h).chunk(2, dim=-1)
        partial = self.w2(F.silu(gate) * value)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return x + (partial + self.w2_bias) * self.scale2


def _config(module: str, device: torch.device, dtype: torch.dtype) -> dict[str, int | torch.dtype]:
    if module == "qwen":
        return {
            "hidden": 2560,
            "heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "intermediate": 9728,
            "sequence": 256,
        }
    if module == "vae":
        return {
            "hidden": 2048,
            "heads": 32,
            "head_dim": 64,
            "sequence": 2245,
        }
    raise ValueError(f"unsupported module: {module}")


def _error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    delta = candidate.float() - reference.float()
    ref = reference.float()
    rms = torch.sqrt(delta.square().mean())
    ref_rms = torch.sqrt(ref.square().mean()).clamp_min(1e-12)
    cosine = F.cosine_similarity(ref.reshape(1, -1), candidate.float().reshape(1, -1))[0]
    return {
        "max_abs": float(delta.abs().max().item()),
        "rms": float(rms.item()),
        "relative_rms": float((rms / ref_rms).item()),
        "cosine": float(cosine.item()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def _make_full(module: str, cfg: dict[str, int | torch.dtype], device: torch.device, dtype: torch.dtype) -> nn.Module:
    if module == "qwen":
        return FullQwenBlock(
            int(cfg["hidden"]), int(cfg["heads"]), int(cfg["kv_heads"]),
            int(cfg["head_dim"]), int(cfg["intermediate"]), device, dtype,
        ).eval()
    return FullVAEBlock(int(cfg["hidden"]), int(cfg["heads"]), int(cfg["head_dim"]), device, dtype).eval()


def _make_tp(module: str, rank: int, cfg: dict[str, int | torch.dtype], device: torch.device, dtype: torch.dtype, source=None) -> nn.Module:
    if module == "qwen":
        return TPQwenBlock(
            rank, int(cfg["hidden"]), int(cfg["heads"]), int(cfg["kv_heads"]),
            int(cfg["head_dim"]), int(cfg["intermediate"]), device, dtype, source,
        ).eval()
    return TPVAEBlock(rank, int(cfg["hidden"]), int(cfg["heads"]), int(cfg["head_dim"]), device, dtype, source).eval()


def _synchronize_all(devices: tuple[torch.device, ...]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _run_correctness(module: str, device: torch.device, rank: int, dtype: torch.dtype) -> dict[str, object]:
    # Small shapes keep the reference on each rank and make this gate safe even
    # when the machine is already hosting another unrelated CUDA process.
    if module == "qwen":
        cfg = {"hidden": 128, "heads": 4, "kv_heads": 2, "head_dim": 16, "intermediate": 256, "sequence": 17}
    else:
        cfg = {"hidden": 128, "heads": 4, "head_dim": 32, "sequence": 37}
    torch.manual_seed(20260825)
    full = _make_full(module, cfg, device, dtype)
    tp = _make_tp(module, rank, cfg, device, dtype, source=full)
    x = torch.randn((1, int(cfg["sequence"]), int(cfg["hidden"])), device=device, dtype=dtype)
    dist.broadcast(x, src=0)
    with torch.inference_mode():
        reference = full(x)
        candidate = tp(x.clone())
    gathered = [torch.empty_like(candidate) for _ in range(WORLD)]
    dist.all_gather(gathered, candidate)
    rank_consistency = _error(gathered[0], gathered[1])
    result = {
        "module": module,
        "shape": list(x.shape),
        "full_vs_tp": _error(reference, candidate),
        "tp_rank_consistency": rank_consistency,
        "finite": bool(torch.isfinite(candidate).all().item()),
    }
    del gathered, candidate, reference, x, tp, full
    torch.cuda.empty_cache()
    return result


def _run_tp(args: argparse.Namespace) -> dict[str, object]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank or rank not in (0, 1):
        raise RuntimeError("this benchmark expects local rank 0/1")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != WORLD:
        raise RuntimeError("this benchmark requires exactly two NCCL ranks")
    dtype = torch.float16
    cfg = _config(args.module, device, dtype)
    cfg["sequence"] = args.sequence or int(cfg["sequence"])
    if args.module == "qwen":
        cfg["intermediate"] = int(cfg["intermediate"])

    correctness = _run_correctness(args.module, device, rank, dtype) if args.correctness else None
    blocks = nn.ModuleList([
        _make_tp(args.module, rank, cfg, device, dtype)
        for _ in range(args.layers)
    ]).eval()
    x = torch.randn((1, int(cfg["sequence"]), int(cfg["hidden"])), device=device, dtype=dtype)
    dist.broadcast(x, src=0)
    with torch.inference_mode():
        for _ in range(args.warmup):
            y = x.clone()
            for block in blocks:
                y = block(y)
        _synchronize_all((device,))
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repetitions):
            y = x.clone()
            for block in blocks:
                y = block(y)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / args.repetitions
    gathered = [torch.empty_like(y) for _ in range(WORLD)]
    dist.all_gather(gathered, y)
    consistency = _error(gathered[0], gathered[1])
    peak = torch.cuda.max_memory_allocated(device) / MIB
    params = sum(p.numel() * p.element_size() for p in blocks.parameters()) / MIB
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hardware": "2x Tesla V100 expected; synthetic weights only",
        "checkpoint_loaded": False,
        "host_mmap": False,
        "route": "strict_tensor_parallel",
        "module": args.module,
        "layers": args.layers,
        "shape": [1, int(cfg["sequence"]), int(cfg["hidden"])],
        "rank": rank,
        "local_parameters_mib": params,
        "forward_ms_per_rank": elapsed,
        "peak_allocated_mib": peak,
        "tp_rank_consistency": consistency,
        "correctness": correctness,
        "numerically_qualified": bool(
            consistency["finite"] and consistency["max_abs"] == 0.0
            and (correctness is None or (
                correctness["finite"]
                and correctness["full_vs_tp"]["finite"]
                and correctness["full_vs_tp"]["relative_rms"] <= 3e-3
                and correctness["full_vs_tp"]["cosine"] >= 0.9999
            ))
        ),
    }
    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"saved report: {output}", flush=True)
    dist.barrier()
    del gathered, y, x, blocks
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return report


def _run_layer(args: argparse.Namespace) -> dict[str, object]:
    if "RANK" in os.environ:
        raise RuntimeError("layer route must run as one ordinary Python process, not torchrun")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("layer route requires two visible CUDA devices")
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    for device in devices:
        torch.cuda.set_device(device)
    dtype = torch.float16
    cfg = _config(args.module, devices[0], dtype)
    cfg["sequence"] = args.sequence or int(cfg["sequence"])
    blocks: list[nn.Module] = []
    default_split = int(os.environ.get(
        "H3_QWEN_SPLIT" if args.module == "qwen" else "H3_VAE_SPLIT", "12"
    ))
    split = args.split if args.split is not None else default_split
    if not 1 <= split < args.layers:
        raise ValueError(f"split must be between 1 and layers-1, got {split}")
    for index in range(args.layers):
        blocks.append(_make_full(args.module, cfg, devices[0 if index < split else 1], dtype))
    x = torch.randn((1, int(cfg["sequence"]), int(cfg["hidden"])), device=devices[0], dtype=dtype)
    with torch.inference_mode():
        for _ in range(args.warmup):
            y = x.clone()
            current = devices[0]
            for index, block in enumerate(blocks):
                target = devices[0 if index < split else 1]
                if target != current:
                    y = y.to(target)
                    current = target
                y = block(y)
        _synchronize_all(devices)
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.repetitions):
            y = x.clone()
            current = devices[0]
            for index, block in enumerate(blocks):
                target = devices[0 if index < split else 1]
                if target != current:
                    y = y.to(target)
                    current = target
                y = block(y)
        _synchronize_all(devices)
        elapsed = (time.perf_counter() - start) * 1000.0 / args.repetitions
    peak = [torch.cuda.max_memory_allocated(device) / MIB for device in devices]
    params = [
        sum(p.numel() * p.element_size() for block in blocks[:split] for p in block.parameters()) / MIB,
        sum(p.numel() * p.element_size() for block in blocks[split:] for p in block.parameters()) / MIB,
    ]
    finite = bool(torch.isfinite(y).all().item())
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hardware": "2x Tesla V100 expected; synthetic weights only",
        "checkpoint_loaded": False,
        "host_mmap": False,
        "route": "layer_model_parallel",
        "module": args.module,
        "layers": args.layers,
        "split": [split, args.layers - split],
        "shape": list(x.shape),
        "forward_ms": elapsed,
        "peak_allocated_mib": peak,
        "resident_parameter_mib": params,
        "activation_handoff": "cuda:0 -> cuda:1 once",
        "finite": finite,
        "numerically_qualified": finite,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved report: {output}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=("qwen", "vae"), required=True)
    parser.add_argument("--route", choices=("layer", "tp"), required=True)
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument(
        "--split",
        type=int,
        help="layer-MP split; defaults to H3_QWEN_SPLIT/H3_VAE_SPLIT or 12",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.route == "tp":
        _run_tp(args)
    else:
        _run_layer(args)


if __name__ == "__main__":
    main()
