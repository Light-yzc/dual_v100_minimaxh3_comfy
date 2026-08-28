#!/usr/bin/env python3
"""Numerical and performance gate for a two-way H3 tensor-parallel block.

Run with exactly two NCCL ranks, for example::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      scripts/benchmark_h3_tp_block.py --profile small

The benchmark deliberately loads no model files.  It creates rank-local FP16
weights directly on each GPU, so host RAM never contains a full H3 block.  The
partition matches MiniMax H3's fused tensor ordering:

* QKV is column parallel, with a local slice from each Q, K and V segment.
* attention output projection is row parallel, followed by all-reduce.
* FC1 is column parallel, with a local slice from both gate and up segments.
* FC2 is row parallel, followed by all-reduce.

``small`` builds an independent dense reference and is the numerical gate.
``h3`` uses the real 5376/56x128/14336 dimensions and defaults to a short
sequence so it can be run without evicting the resident ComfyUI service.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


REPO_ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20


@dataclass(frozen=True)
class BlockShape:
    sequence: int
    hidden: int
    heads: int
    head_dim: int
    ffn: int

    @property
    def inner(self) -> int:
        return self.heads * self.head_dim

    def validate(self, world_size: int) -> None:
        if self.sequence <= 0 or self.hidden <= 0 or self.ffn <= 0:
            raise ValueError(f"invalid non-positive block shape: {self}")
        if self.heads * self.head_dim != self.inner:
            raise ValueError("invalid attention dimensions")
        if self.heads % world_size:
            raise ValueError(f"heads={self.heads} is not divisible by TP={world_size}")
        if self.ffn % world_size:
            raise ValueError(f"ffn={self.ffn} is not divisible by TP={world_size}")


@dataclass
class LocalWeights:
    qkv: torch.Tensor
    out: torch.Tensor
    fc1: torch.Tensor
    fc2: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("small", "h3"), default="small")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=512,
        help="VRAM which must remain beyond weights, activations and workspace",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def profile_shape(profile: str, sequence: int | None) -> BlockShape:
    if profile == "small":
        return BlockShape(sequence or 128, hidden=512, heads=8, head_dim=64, ffn=1024)
    return BlockShape(sequence or 128, hidden=5376, heads=56, head_dim=128, ffn=14336)


def rss_mib() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def local_storage_bytes(shape: BlockShape, world_size: int) -> int:
    local_inner = shape.inner // world_size
    local_ffn = shape.ffn // world_size
    elements = (
        3 * local_inner * shape.hidden
        + shape.hidden * local_inner
        + 2 * local_ffn * shape.hidden
        + shape.hidden * local_ffn
    )
    return elements * torch.empty((), dtype=torch.float16).element_size()


def activation_guard_bytes(shape: BlockShape, world_size: int) -> int:
    local_inner = shape.inner // world_size
    local_ffn = shape.ffn // world_size
    # Live tensors in the unfused reference plus a conservative SDPA/NCCL
    # workspace allowance.  This is intentionally an upper-bound guard, not a
    # claim about allocator-exact peak usage.
    live_elements = shape.sequence * (
        8 * shape.hidden + 6 * local_inner + 3 * local_ffn
    )
    return live_elements * 2 + 512 * MIB


def ensure_vram_budget(
    shape: BlockShape,
    world_size: int,
    device: torch.device,
    min_free_mib: int,
) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    weights = local_storage_bytes(shape, world_size)
    activations = activation_guard_bytes(shape, world_size)
    required = weights + activations + min_free_mib * MIB
    stats = {
        "free_before_mib": free_bytes / MIB,
        "total_mib": total_bytes / MIB,
        "local_weight_mib": weights / MIB,
        "activation_and_workspace_guard_mib": activations / MIB,
        "required_with_reserve_mib": required / MIB,
    }
    if free_bytes < required:
        raise RuntimeError(
            "VRAM safety guard refused TP block benchmark: "
            f"free={free_bytes / MIB:.0f} MiB, required={required / MIB:.0f} MiB "
            f"on {device}"
        )
    return stats


def make_weight(
    rows: int,
    columns: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    weight = torch.empty((rows, columns), dtype=torch.float16, device=device)
    weight.normal_(mean=0.0, std=1.0 / math.sqrt(columns), generator=generator)
    return weight


def make_local_weights(
    shape: BlockShape,
    world_size: int,
    rank: int,
    device: torch.device,
    seed: int,
) -> LocalWeights:
    local_inner = shape.inner // world_size
    local_ffn = shape.ffn // world_size
    generator = torch.Generator(device=device).manual_seed(seed + 1009 * rank)
    return LocalWeights(
        qkv=make_weight(3 * local_inner, shape.hidden, device=device, generator=generator),
        out=make_weight(shape.hidden, local_inner, device=device, generator=generator),
        fc1=make_weight(2 * local_ffn, shape.hidden, device=device, generator=generator),
        fc2=make_weight(shape.hidden, local_ffn, device=device, generator=generator),
    )


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return functional.rms_norm(x, (x.shape[-1],), weight=None, eps=1e-5)


def projected_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Inputs are [S, H, D]; output is projection-ready [S, H*D].
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        output = functional.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=False
        )
    return output.transpose(1, 2).reshape(q.shape[0], -1)


def local_attention(x: torch.Tensor, weight: torch.Tensor, local_heads: int, head_dim: int) -> torch.Tensor:
    qkv = functional.linear(x, weight)
    local_inner = local_heads * head_dim
    q, k, v = qkv.split(local_inner, dim=-1)
    q = rms_norm(q.view(x.shape[0], local_heads, head_dim))
    k = rms_norm(k.view(x.shape[0], local_heads, head_dim))
    v = v.view(x.shape[0], local_heads, head_dim)
    return projected_attention(q, k, v)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return functional.silu(gate).mul(up)


def tp_forward(
    x: torch.Tensor,
    weights: LocalWeights,
    shape: BlockShape,
    world_size: int,
) -> torch.Tensor:
    local_heads = shape.heads // world_size
    h = rms_norm(x)
    attention = local_attention(h, weights.qkv, local_heads, shape.head_dim)
    attention_partial = functional.linear(attention, weights.out)
    dist.all_reduce(attention_partial, op=dist.ReduceOp.SUM)
    residual = x + attention_partial

    h = rms_norm(residual)
    mlp_hidden = swiglu(functional.linear(h, weights.fc1))
    mlp_partial = functional.linear(mlp_hidden, weights.fc2)
    dist.all_reduce(mlp_partial, op=dist.ReduceOp.SUM)
    return residual + mlp_partial


def all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return gathered


def dense_weights(weights: LocalWeights) -> LocalWeights:
    """Reconstruct conceptual full matrices from correctly ordered TP shards."""

    qkv_ranks = all_gather(weights.qkv)
    q_parts, k_parts, v_parts = [], [], []
    for shard in qkv_ranks:
        q, k, v = shard.chunk(3, dim=0)
        q_parts.append(q)
        k_parts.append(k)
        v_parts.append(v)
    qkv = torch.cat(
        [torch.cat(q_parts, dim=0), torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)],
        dim=0,
    )

    out = torch.cat(all_gather(weights.out), dim=1)

    fc1_ranks = all_gather(weights.fc1)
    gate_parts, up_parts = [], []
    for shard in fc1_ranks:
        gate, up = shard.chunk(2, dim=0)
        gate_parts.append(gate)
        up_parts.append(up)
    fc1 = torch.cat(
        [torch.cat(gate_parts, dim=0), torch.cat(up_parts, dim=0)], dim=0
    )
    fc2 = torch.cat(all_gather(weights.fc2), dim=1)
    return LocalWeights(qkv=qkv, out=out, fc1=fc1, fc2=fc2)


def dense_forward(x: torch.Tensor, weights: LocalWeights, shape: BlockShape) -> torch.Tensor:
    h = rms_norm(x)
    attention = local_attention(h, weights.qkv, shape.heads, shape.head_dim)
    residual = x + functional.linear(attention, weights.out)
    h = rms_norm(residual)
    mlp_hidden = swiglu(functional.linear(h, weights.fc1))
    return residual + functional.linear(mlp_hidden, weights.fc2)


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    ref = reference.float()
    got = candidate.float()
    difference = got - ref
    ref_rms = torch.sqrt(torch.mean(ref.square()))
    diff_rms = torch.sqrt(torch.mean(difference.square()))
    ref_max = ref.abs().max()
    denominator = torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(got)
    cosine = torch.sum(ref * got) / denominator
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rms": float(diff_rms.item()),
        "reference_max_abs": float(ref_max.item()),
        "reference_rms": float(ref_rms.item()),
        "max_abs_over_reference_max": float((difference.abs().max() / ref_max.clamp_min(1e-30)).item()),
        "relative_rms": float((diff_rms / ref_rms.clamp_min(1e-30)).item()),
        "cosine": float(cosine.item()),
        "finite": bool(torch.isfinite(got).all().item()),
    }


def max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def timed_tp_forward(
    x: torch.Tensor,
    weights: LocalWeights,
    shape: BlockShape,
    world_size: int,
    warmup: int,
    repetitions: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    output = None
    for _ in range(warmup):
        output = tp_forward(x, weights, shape, world_size)
    torch.cuda.synchronize(device)
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        output = tp_forward(x, weights, shape, world_size)
    end.record()
    end.synchronize()
    local_ms = start.elapsed_time(end) / repetitions
    return output, max_across_ranks(local_ms, device)  # type: ignore[arg-type]


def component_profile(
    x: torch.Tensor,
    weights: LocalWeights,
    shape: BlockShape,
    world_size: int,
    device: torch.device,
) -> dict[str, float]:
    local_heads = shape.heads // world_size
    labels: list[str] = []
    events: list[torch.cuda.Event] = []

    def mark(label: str) -> None:
        labels.append(label)
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events.append(event)

    mark("start")
    h = rms_norm(x)
    mark("norm1")
    attention = local_attention(h, weights.qkv, local_heads, shape.head_dim)
    mark("qkv_norm_sdpa")
    attention_partial = functional.linear(attention, weights.out)
    mark("out_proj")
    dist.all_reduce(attention_partial)
    mark("attention_all_reduce")
    residual = x + attention_partial
    h = rms_norm(residual)
    mark("residual_norm2")
    mlp_hidden = swiglu(functional.linear(h, weights.fc1))
    mark("fc1_swiglu")
    mlp_partial = functional.linear(mlp_hidden, weights.fc2)
    mark("fc2")
    dist.all_reduce(mlp_partial)
    mark("mlp_all_reduce")
    _ = residual + mlp_partial
    mark("final_residual")
    events[-1].synchronize()

    result: dict[str, float] = {}
    for index in range(1, len(events)):
        milliseconds = events[index - 1].elapsed_time(events[index])
        result[labels[index]] = max_across_ranks(milliseconds, device)
    return result


def gather_rank_stats(device: torch.device) -> list[dict[str, float | int]]:
    free_bytes, _ = torch.cuda.mem_get_info(device)
    local = torch.tensor(
        [
            float(dist.get_rank()),
            torch.cuda.max_memory_allocated(device) / MIB,
            torch.cuda.max_memory_reserved(device) / MIB,
            free_bytes / MIB,
            rss_mib(),
        ],
        dtype=torch.float64,
        device=device,
    )
    values = all_gather(local)
    return [
        {
            "rank": int(value[0].item()),
            "peak_allocated_mib": float(value[1].item()),
            "peak_reserved_mib": float(value[2].item()),
            "free_after_mib": float(value[3].item()),
            "rss_mib": float(value[4].item()),
        }
        for value in values
    ]


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    world_size = dist.get_world_size()
    if world_size != 2:
        raise SystemExit("benchmark_h3_tp_block.py requires exactly two NCCL ranks")

    shape = profile_shape(args.profile, args.sequence)
    shape.validate(world_size)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise SystemExit(f"expected V100/SM70, got sm_{props.major}{props.minor} on rank {rank}")
    peer = 1 - local_rank
    if not torch.cuda.can_device_access_peer(local_rank, peer):
        raise SystemExit(f"CUDA P2P unavailable from local GPU {local_rank} to {peer}")

    budget = ensure_vram_budget(shape, world_size, device, args.min_free_mib)
    torch.cuda.reset_peak_memory_stats(device)
    weights = make_local_weights(shape, world_size, rank, device, args.seed)

    x = torch.empty((shape.sequence, shape.hidden), dtype=torch.float16, device=device)
    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        x.normal_(generator=generator)
    dist.broadcast(x, src=0)

    tp_output = tp_forward(x, weights, shape, world_size)
    rank_outputs = all_gather(tp_output)
    rank_error = tensor_error(rank_outputs[0], rank_outputs[1])
    dense_error = None
    if args.profile == "small":
        full_weights = dense_weights(weights)
        if rank == 0:
            dense_output = dense_forward(x, full_weights, shape)
            dense_error = tensor_error(dense_output, tp_output)
        del full_weights

    # Keep collective ordering identical: rank 0 publishes the dense result as
    # fixed-width numeric fields instead of using a pickled object collective.
    dense_vector = torch.zeros(5, dtype=torch.float64, device=device)
    if rank == 0 and dense_error is not None:
        dense_vector.copy_(torch.tensor([
            dense_error["max_abs"],
            dense_error["mean_abs"],
            dense_error["rms"],
            dense_error["cosine"],
            float(dense_error["finite"]),
        ], dtype=torch.float64, device=device))
    dist.broadcast(dense_vector, src=0)
    if args.profile == "small":
        dense_error = {
            "max_abs": float(dense_vector[0].item()),
            "mean_abs": float(dense_vector[1].item()),
            "rms": float(dense_vector[2].item()),
            "cosine": float(dense_vector[3].item()),
            "finite": bool(dense_vector[4].item()),
        }

    tp_output, block_ms = timed_tp_forward(
        x,
        weights,
        shape,
        world_size,
        args.warmup,
        args.repetitions,
        device,
    )
    components = component_profile(x, weights, shape, world_size, device)
    rank_stats = gather_rank_stats(device)

    numerical_qualified = bool(
        rank_error["finite"]
        and rank_error["max_abs"] == 0.0
        and (
            dense_error is None
            or (
                dense_error["finite"]
                and dense_error["cosine"] >= 0.999
                and dense_error["max_abs"] <= 0.05
            )
        )
    )

    report = {
        "created_unix": time.time(),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "backend": "torchrun/NCCL",
        "world_size": world_size,
        "profile": args.profile,
        "shape": asdict(shape) | {"inner": shape.inner},
        "partition": {
            "qkv": "column-parallel, matching local Q/K/V head rows",
            "out_proj": "row-parallel input columns + all-reduce",
            "fc1": "column-parallel, matching local gate/up rows",
            "fc2": "row-parallel input columns + all-reduce",
        },
        "budget_by_rank0": budget,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "p2p": True,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "milliseconds_per_block": block_ms,
        "component_milliseconds_max_rank": components,
        "rank_consistency": rank_error,
        "dense_reference_error": dense_error,
        "numerically_qualified": numerical_qualified,
        "rank_resources": rank_stats,
    }

    if rank == 0:
        output = args.output
        if output is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            output = REPO_ROOT / "results" / f"h3_tp_block_{args.profile}_{stamp}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"saved report: {output}", flush=True)

    del tp_output, rank_outputs, weights, x
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
