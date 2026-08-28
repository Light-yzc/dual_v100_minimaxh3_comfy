#!/usr/bin/env python3
"""Benchmark a real H3 Q4_0 block with two-way NCCL tensor parallelism.

Only four matrices from one selected block are read.  Each rank loads its Q4
shard directly from the GGUF on /mnt/GALAX through bounded CPU staging, then
dequantizes one local matrix at a time on its V100.  Rank 0 can additionally
run the complete Q4 block as an independent dense numerical/performance
reference; no full model and no VAE/text encoder are loaded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_h3_tp_block as fp16_bench  # noqa: E402


DEFAULT_MODEL = Path(
    "/mnt/GALAX/minimax-h3/models/diffusion_models/"
    "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf"
)
DEFAULT_LORA = Path(
    "/mnt/GALAX/minimax-h3/models/loras/"
    "minimax_h3_turbo_v4_step600_ema.safetensors"
)
MIB = 1 << 20
WIDE_LINEAR_MODE = "fp32"


def load_q4_tp_module():
    path = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_q4_tp.py"
    spec = importlib.util.spec_from_file_location("h3_q4_tp_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Q4 TP helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


q4_tp = load_q4_tp_module()


def load_lora_tp_module():
    path = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_lora_tp.py"
    spec = importlib.util.spec_from_file_location("h3_lora_tp_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import LoRA TP helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lora_tp = load_lora_tp_module()


def load_wide_linear_module():
    path = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_v100_fp32_linear.py"
    spec = importlib.util.spec_from_file_location("h3_v100_fp32_linear_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import H3 wide Linear helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wide_linear = load_wide_linear_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument(
        "--wide-linear",
        choices=("fp32", "fp16-fp32"),
        default="fp32",
        help=(
            "row-parallel out/FC2 path: full FP32 GEMM, or FP16 Tensor Core "
            "inputs with direct FP32 output"
        ),
    )
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--staging-mib", type=int, default=4)
    parser.add_argument("--min-free-mib", type=int, default=384)
    parser.add_argument(
        "--skip-dense-reference",
        action="store_true",
        help="skip rank-0 full-Q4 reference (use only after numerical gate passed)",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def matrix_names(block: int) -> dict[str, str]:
    prefix = f"blocks.{block}"
    return {
        "qkv": f"{prefix}.attn.qkv_proj.weight",
        "out_proj": f"{prefix}.attn.out_proj.weight",
        "fc1": f"{prefix}.mlp.fc1.weight",
        "fc2": f"{prefix}.mlp.fc2.weight",
    }


def validate_h3_specs(specs, names) -> fp16_bench.BlockShape:
    qkv = specs[names["qkv"]]
    out = specs[names["out_proj"]]
    fc1 = specs[names["fc1"]]
    fc2 = specs[names["fc2"]]
    hidden = qkv.in_features
    inner = qkv.out_features // 3
    ffn = fc1.out_features // 2
    expected = {
        "qkv": (3 * inner, hidden),
        "out_proj": (hidden, inner),
        "fc1": (2 * ffn, hidden),
        "fc2": (hidden, ffn),
    }
    actual = {
        "qkv": qkv.shape,
        "out_proj": out.shape,
        "fc1": fc1.shape,
        "fc2": fc2.shape,
    }
    if actual != expected:
        raise ValueError(f"inconsistent H3 block matrix shapes: {actual}")
    if inner % 128:
        raise ValueError(f"H3 attention inner width {inner} is not 128-aligned")
    return fp16_bench.BlockShape(
        sequence=1,
        hidden=hidden,
        heads=inner // 128,
        head_dim=128,
        ffn=ffn,
    )


def load_local_shards(path, specs, names, rank, world_size, device, staging_bytes):
    start = time.monotonic()
    rss_before = fp16_bench.rss_mib()
    with q4_tp.Q4DiskReader(path, device, staging_bytes) as reader:
        shards = {
            role: reader.read_tp_shard(specs[name], role, rank, world_size)
            for role, name in names.items()
        }
    torch.cuda.synchronize(device)
    return shards, {
        "seconds": time.monotonic() - start,
        "rss_before_mib": rss_before,
        "rss_after_mib": fp16_bench.rss_mib(),
        "compressed_mib": sum(shard.raw.numel() for shard in shards.values()) / MIB,
        "host_staging_mib": staging_bytes / MIB,
    }


def load_full_weights(path, specs, names, device, staging_bytes):
    start = time.monotonic()
    with q4_tp.Q4DiskReader(path, device, staging_bytes) as reader:
        weights = {role: reader.read_full(specs[name]) for role, name in names.items()}
    torch.cuda.synchronize(device)
    return weights, {
        "seconds": time.monotonic() - start,
        "compressed_mib": sum(weight.raw.numel() for weight in weights.values()) / MIB,
    }


power_of_two_fp16_scale = wide_linear.power_of_two_fp16_scale
tensor_core_fp32_output_linear = wide_linear.tensor_core_fp32_output_linear


def q4_linear(
    x: torch.Tensor,
    matrix,
    lora=None,
    *,
    wide: bool = False,
    scale_input: bool = False,
) -> torch.Tensor:
    if wide and WIDE_LINEAR_MODE == "fp16-fp32":
        weight = q4_tp.dequantize_q4_0(matrix, dtype=torch.float16)
        output = tensor_core_fp32_output_linear(
            x, weight, scale_input=scale_input
        )
    else:
        weight = q4_tp.dequantize_q4_0(matrix, dtype=x.dtype)
        output = functional.linear(x, weight)
    del weight
    if lora is not None:
        output.add_(lora_tp.lora_delta(x, lora))
    return output


def q4_attention(x: torch.Tensor, qkv, heads: int, head_dim: int, lora=None) -> torch.Tensor:
    packed = q4_linear(x, qkv, lora)
    inner = heads * head_dim
    q, k, v = packed.split(inner, dim=-1)
    q = fp16_bench.rms_norm(q.view(x.shape[0], heads, head_dim))
    k = fp16_bench.rms_norm(k.view(x.shape[0], heads, head_dim))
    v = v.view(x.shape[0], heads, head_dim)
    return fp16_bench.projected_attention(q, k, v)


def tp_forward(x, shards, loras, shape, world_size):
    local_heads = shape.heads // world_size
    # Match the V100 production stability islands: replicated residuals stay
    # FP32, attention/QKV and FC1 stay FP16, while row-parallel out_proj/FC2
    # produce and all-reduce FP32 partials.
    h = fp16_bench.rms_norm(x).to(torch.float16)
    attention = q4_attention(
        h, shards["qkv"], local_heads, shape.head_dim,
        None if loras is None else loras["qkv"],
    )
    attention_partial = q4_linear(
        attention.float(), shards["out_proj"],
        None if loras is None else loras["out_proj"],
        wide=True,
    )
    dist.all_reduce(attention_partial)
    residual = x + attention_partial
    h = fp16_bench.rms_norm(residual).to(torch.float16)
    mlp_hidden = fp16_bench.swiglu(q4_linear(
        h, shards["fc1"], None if loras is None else loras["fc1"]
    ).float())
    mlp_partial = q4_linear(
        mlp_hidden, shards["fc2"], None if loras is None else loras["fc2"],
        wide=True, scale_input=True,
    )
    dist.all_reduce(mlp_partial)
    return residual + mlp_partial


def dense_forward(x, weights, loras, shape):
    h = fp16_bench.rms_norm(x).to(torch.float16)
    attention = q4_attention(
        h, weights["qkv"], shape.heads, shape.head_dim,
        None if loras is None else loras["qkv"],
    )
    residual = x + q4_linear(
        attention.float(), weights["out_proj"],
        None if loras is None else loras["out_proj"],
    )
    h = fp16_bench.rms_norm(residual).to(torch.float16)
    mlp_hidden = fp16_bench.swiglu(q4_linear(
        h, weights["fc1"], None if loras is None else loras["fc1"]
    ).float())
    return residual + q4_linear(
        mlp_hidden, weights["fc2"], None if loras is None else loras["fc2"]
    )


def timed_tp(x, shards, loras, shape, world_size, warmup, repetitions, device):
    output = None
    for _ in range(warmup):
        output = tp_forward(x, shards, loras, shape, world_size)
    torch.cuda.synchronize(device)
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        output = tp_forward(x, shards, loras, shape, world_size)
    end.record()
    end.synchronize()
    milliseconds = start.elapsed_time(end) / repetitions
    return output, fp16_bench.max_across_ranks(milliseconds, device)


def timed_dense(x, weights, loras, shape, warmup, repetitions, device):
    output = None
    for _ in range(warmup):
        output = dense_forward(x, weights, loras, shape)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        output = dense_forward(x, weights, loras, shape)
    end.record()
    end.synchronize()
    return output, start.elapsed_time(end) / repetitions


def profile_components(x, shards, loras, shape, world_size, device):
    labels = []
    events = []

    def mark(label):
        labels.append(label)
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events.append(event)

    local_heads = shape.heads // world_size
    mark("start")
    h = fp16_bench.rms_norm(x).to(torch.float16)
    mark("norm1")
    qkv_weight = q4_tp.dequantize_q4_0(shards["qkv"], dtype=h.dtype)
    mark("qkv_dequant")
    packed = functional.linear(h, qkv_weight)
    del qkv_weight
    if loras is not None:
        packed.add_(lora_tp.lora_delta(h, loras["qkv"]))
    inner = local_heads * shape.head_dim
    q, k, v = packed.split(inner, dim=-1)
    q = fp16_bench.rms_norm(q.view(x.shape[0], local_heads, shape.head_dim))
    k = fp16_bench.rms_norm(k.view(x.shape[0], local_heads, shape.head_dim))
    attention = fp16_bench.projected_attention(
        q, k, v.view(x.shape[0], local_heads, shape.head_dim)
    )
    mark("qkv_gemm_norm_sdpa")
    out_dtype = torch.float16 if WIDE_LINEAR_MODE == "fp16-fp32" else torch.float32
    out_weight = q4_tp.dequantize_q4_0(shards["out_proj"], dtype=out_dtype)
    mark("out_dequant")
    if WIDE_LINEAR_MODE == "fp16-fp32":
        attention_partial = tensor_core_fp32_output_linear(
            attention.float(), out_weight, scale_input=False
        )
    else:
        attention_partial = functional.linear(attention.float(), out_weight)
    del out_weight
    if loras is not None:
        attention_partial.add_(lora_tp.lora_delta(
            attention.float(), loras["out_proj"]
        ))
    mark("out_gemm")
    dist.all_reduce(attention_partial)
    mark("attention_all_reduce")
    residual = x + attention_partial
    h = fp16_bench.rms_norm(residual).to(torch.float16)
    mark("residual_norm2")
    fc1_weight = q4_tp.dequantize_q4_0(shards["fc1"], dtype=h.dtype)
    mark("fc1_dequant")
    fc1_output = functional.linear(h, fc1_weight)
    del fc1_weight
    if loras is not None:
        fc1_output.add_(lora_tp.lora_delta(h, loras["fc1"]))
    mlp_hidden = fp16_bench.swiglu(fc1_output.float())
    mark("fc1_gemm_swiglu")
    fc2_dtype = torch.float16 if WIDE_LINEAR_MODE == "fp16-fp32" else torch.float32
    fc2_weight = q4_tp.dequantize_q4_0(shards["fc2"], dtype=fc2_dtype)
    mark("fc2_dequant")
    if WIDE_LINEAR_MODE == "fp16-fp32":
        mlp_partial = tensor_core_fp32_output_linear(
            mlp_hidden, fc2_weight, scale_input=True
        )
    else:
        mlp_partial = functional.linear(mlp_hidden, fc2_weight)
    del fc2_weight
    if loras is not None:
        mlp_partial.add_(lora_tp.lora_delta(mlp_hidden, loras["fc2"]))
    mark("fc2_gemm")
    dist.all_reduce(mlp_partial)
    mark("mlp_all_reduce")
    _ = residual + mlp_partial
    mark("final_residual")
    events[-1].synchronize()

    result = {}
    for index in range(1, len(events)):
        elapsed = events[index - 1].elapsed_time(events[index])
        result[labels[index]] = fp16_bench.max_across_ranks(elapsed, device)
    return result


def broadcast_dense_metrics(error, milliseconds, device):
    vector = torch.zeros(10, dtype=torch.float64, device=device)
    if dist.get_rank() == 0 and error is not None:
        vector.copy_(torch.tensor([
            error["max_abs"],
            error["mean_abs"],
            error["rms"],
            error["reference_max_abs"],
            error["reference_rms"],
            error["max_abs_over_reference_max"],
            error["relative_rms"],
            error["cosine"],
            float(error["finite"]),
            milliseconds,
        ], dtype=torch.float64, device=device))
    dist.broadcast(vector, src=0)
    return {
        "error": {
            "max_abs": float(vector[0].item()),
            "mean_abs": float(vector[1].item()),
            "rms": float(vector[2].item()),
            "reference_max_abs": float(vector[3].item()),
            "reference_rms": float(vector[4].item()),
            "max_abs_over_reference_max": float(vector[5].item()),
            "relative_rms": float(vector[6].item()),
            "cosine": float(vector[7].item()),
            "finite": bool(vector[8].item()),
        },
        "milliseconds": float(vector[9].item()),
    }


def main() -> None:
    global WIDE_LINEAR_MODE
    args = parse_args()
    WIDE_LINEAR_MODE = args.wide_linear
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    world_size = dist.get_world_size()
    if world_size != 2:
        raise SystemExit("real H3 Q4 benchmark requires exactly two NCCL ranks")
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise SystemExit(f"expected SM70, got sm_{props.major}{props.minor}")
    if not torch.cuda.can_device_access_peer(local_rank, 1 - local_rank):
        raise SystemExit("CUDA P2P is unavailable between the selected ranks")

    names = matrix_names(args.block)
    specs, gguf_metadata = q4_tp.inspect_q4_matrices(args.model, set(names.values()))
    base_shape = validate_h3_specs(specs, names)
    shape = fp16_bench.BlockShape(
        sequence=args.sequence,
        hidden=base_shape.hidden,
        heads=base_shape.heads,
        head_dim=base_shape.head_dim,
        ffn=base_shape.ffn,
    )
    shape.validate(world_size)

    free_before, total_bytes = torch.cuda.mem_get_info(device)
    local_compressed = sum(spec.n_bytes for spec in specs.values()) // world_size
    # Q4 dequantization briefly holds int8 nibbles plus an FP16 matrix.  FC1 is
    # the largest local matrix, so reserve 3 bytes per element there, plus
    # compressed shards, 256 MiB activation/workspace, and caller reserve.
    largest_local_elements = (shape.ffn // world_size) * 2 * shape.hidden
    required = (
        local_compressed
        + largest_local_elements * 3
        + (32 * MIB if not args.skip_lora else 0)
        + 256 * MIB
        + args.min_free_mib * MIB
    )
    if free_before < required:
        raise RuntimeError(
            f"VRAM safety guard: free={free_before / MIB:.0f} MiB, "
            f"required={required / MIB:.0f} MiB on rank {rank}"
        )

    torch.cuda.reset_peak_memory_stats(device)
    shards, load_stats = load_local_shards(
        args.model,
        specs,
        names,
        rank,
        world_size,
        device,
        args.staging_mib * MIB,
    )
    loras = None
    lora_metadata = None
    lora_load_stats = None
    if not args.skip_lora:
        lora_start = time.monotonic()
        loras, lora_metadata = lora_tp.load_h3_lora_tp_shards(
            args.lora,
            args.block,
            rank,
            world_size,
            device,
            args.staging_mib * MIB,
        )
        torch.cuda.synchronize(device)
        lora_load_stats = {
            "seconds": time.monotonic() - lora_start,
            "resident_mib": sum(
                value.a.numel() * value.a.element_size()
                + value.b.numel() * value.b.element_size()
                for value in loras.values()
            ) / MIB,
        }
    x = torch.empty((shape.sequence, shape.hidden), dtype=torch.float32, device=device)
    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        x.normal_(generator=generator)
    dist.broadcast(x, src=0)

    wide_reference_error = None
    if args.wide_linear == "fp16-fp32":
        WIDE_LINEAR_MODE = "fp32"
        fp32_tp_reference = tp_forward(x, shards, loras, shape, world_size)
        WIDE_LINEAR_MODE = args.wide_linear
        initial_tp = tp_forward(x, shards, loras, shape, world_size)
        wide_reference_error = fp16_bench.tensor_error(
            fp32_tp_reference, initial_tp
        )
        del fp32_tp_reference
    else:
        initial_tp = tp_forward(x, shards, loras, shape, world_size)
    rank_outputs = fp16_bench.all_gather(initial_tp)
    rank_error = fp16_bench.tensor_error(rank_outputs[0], rank_outputs[1])

    dense_load_stats = None
    dense_error = None
    dense_ms = 0.0
    full_weights = None
    full_loras = None
    dense_lora_load_stats = None
    if not args.skip_dense_reference and rank == 0:
        torch.cuda.empty_cache()
        full_weights, dense_load_stats = load_full_weights(
            args.model, specs, names, device, args.staging_mib * MIB
        )
        if not args.skip_lora:
            lora_start = time.monotonic()
            full_loras, _ = lora_tp.load_h3_lora_full(
                args.lora, args.block, device, args.staging_mib * MIB
            )
            torch.cuda.synchronize(device)
            dense_lora_load_stats = {
                "seconds": time.monotonic() - lora_start,
                "resident_mib": sum(
                    value.a.numel() * value.a.element_size()
                    + value.b.numel() * value.b.element_size()
                    for value in full_loras.values()
                ) / MIB,
            }
        dense_output, dense_ms = timed_dense(
            x, full_weights, full_loras, shape,
            args.warmup, args.repetitions, device
        )
        dense_error = fp16_bench.tensor_error(dense_output, initial_tp)
        del dense_output
        del full_weights
        full_weights = None
        if full_loras is not None:
            del full_loras
            full_loras = None
        torch.cuda.empty_cache()
    dense_metrics = None
    if not args.skip_dense_reference:
        dense_metrics = broadcast_dense_metrics(dense_error, dense_ms, device)

    tp_output, tp_ms = timed_tp(
        x,
        shards,
        loras,
        shape,
        world_size,
        args.warmup,
        args.repetitions,
        device,
    )
    component_samples = [
        profile_components(x, shards, loras, shape, world_size, device)
        for _ in range(3)
    ]
    components = {
        label: statistics.median(sample[label] for sample in component_samples)
        for label in component_samples[0]
    }
    rank_resources = fp16_bench.gather_rank_stats(device)

    qualified = bool(
        rank_error["finite"]
        and rank_error["max_abs"] == 0.0
        and (
            wide_reference_error is None
            or (
                wide_reference_error["finite"]
                and wide_reference_error["cosine"] >= 0.999999
                and wide_reference_error["relative_rms"] <= 2e-3
            )
        )
        and (
            dense_metrics is None
            or (
                dense_metrics["error"]["finite"]
                and dense_metrics["error"]["cosine"] >= 0.999999
                and dense_metrics["error"]["relative_rms"] <= 2e-3
            )
        )
    )
    report = {
        "created_unix": time.time(),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "backend": "real GGUF Q4_0 + Turbo LoRA shards + torchrun/NCCL",
        "numerical_path": {
            "residual": "FP32",
            "qkv_attention_fc1": "FP16",
            "out_proj_fc2_and_all_reduce": "FP32",
            "wide_linear": args.wide_linear,
            "fc2_input_scaling": (
                "per-token power-of-two to FP16 range"
                if args.wide_linear == "fp16-fp32" else None
            ),
        },
        "payload_mmap": False,
        "model": str(args.model),
        "lora": None if args.skip_lora else str(args.lora),
        "lora_scope": (
            None if args.skip_lora else
            "block backbone qkv/out_proj/fc1/fc2; AdaLN curve injection excluded"
        ),
        "block": args.block,
        "shape": {
            "sequence": shape.sequence,
            "hidden": shape.hidden,
            "heads": shape.heads,
            "head_dim": shape.head_dim,
            "ffn": shape.ffn,
        },
        "world_size": world_size,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gguf": gguf_metadata,
        "lora_metadata": lora_metadata,
        "vram_guard_rank0": {
            "free_before_mib": free_before / MIB,
            "total_mib": total_bytes / MIB,
            "estimated_required_mib": required / MIB,
            "minimum_post_run_reserve_mib": args.min_free_mib,
        },
        "rank0_local_load": load_stats,
        "rank0_local_lora_load": lora_load_stats,
        "rank0_dense_load": dense_load_stats,
        "rank0_dense_lora_load": dense_lora_load_stats,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "tp_milliseconds_per_block": tp_ms,
        "dense_milliseconds_per_block": (
            dense_metrics["milliseconds"] if dense_metrics is not None else None
        ),
        "speedup_vs_dense": (
            dense_metrics["milliseconds"] / tp_ms if dense_metrics is not None else None
        ),
        "component_milliseconds_max_rank": components,
        "component_profile": "median of 3 runs; each sample is max across ranks",
        "rank_consistency": rank_error,
        "mixed_wide_error_vs_fp32_tp": wide_reference_error,
        "dense_reference_error": (
            dense_metrics["error"] if dense_metrics is not None else None
        ),
        "numerically_qualified": qualified,
        "rank_resources": rank_resources,
    }

    if rank == 0:
        output = args.output
        if output is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            suffix = "q4" if args.skip_lora else "q4_lora"
            output = REPO_ROOT / "results" / f"h3_{suffix}_tp_block_{stamp}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"saved report: {output}", flush=True)

    del tp_output, initial_tp, rank_outputs, shards, x
    if loras is not None:
        del loras
    if full_weights is not None:
        del full_weights
    if full_loras is not None:
        del full_loras
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
