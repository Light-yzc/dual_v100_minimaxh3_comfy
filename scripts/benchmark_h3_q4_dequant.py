#!/usr/bin/env python3
"""Benchmark the eager and Triton Q4_0 dequantizers on real H3 shards."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MODEL = Path(
    "/mnt/GALAX/minimax-h3/models/diffusion_models/"
    "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def timed(function, device, warmup: int, repetitions: int):
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        result = function()
    end.record()
    end.synchronize()
    peak_extra = max(0, torch.cuda.max_memory_allocated(device) - baseline)
    return result, start.elapsed_time(end) / repetitions, peak_extra


def matrix_names(block: int):
    prefix = f"blocks.{block}"
    return {
        "qkv": f"{prefix}.attn.qkv_proj.weight",
        "out_proj": f"{prefix}.attn.out_proj.weight",
        "fc1": f"{prefix}.mlp.fc1.weight",
        "fc2": f"{prefix}.mlp.fc2.weight",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--matrices", nargs="+", choices=("qkv", "out_proj", "fc1", "fc2"),
        default=("qkv", "out_proj", "fc1", "fc2"),
    )
    parser.add_argument("--staging-mib", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(f"cuda:{args.rank}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise SystemExit(f"expected SM70, got {(props.major, props.minor)}")
    if not args.model.is_file():
        raise SystemExit(f"missing model: {args.model}")

    q4 = load_module("h3_q4_dequant_bench_q4", REPO_ROOT / "custom_nodes/DualV100/h3_q4_tp.py")
    ops = load_module("h3_q4_dequant_bench_ops", REPO_ROOT / "custom_nodes/DualV100/h3_v100_q4_ops.py")
    names = matrix_names(args.block)
    specs, metadata = q4.inspect_q4_matrices(args.model, set(names.values()))
    reader = q4.Q4DiskReader(args.model, device, args.staging_mib << 20)
    report = {
        "created_unix": time.time(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "model": str(args.model),
        "block": args.block,
        "rank": args.rank,
        "world_size": 2,
        "header": metadata,
        "results": [],
    }
    try:
        for role in args.matrices:
            shard = reader.read_tp_shard(specs[names[role]], role, args.rank, 2)
            eager = lambda: q4._dequantize_q4_0_eager(shard)
            triton = lambda: ops.dequantize_q4_0_sm70(shard, blocks_per_program=4)
            eager_output, eager_ms, eager_peak = timed(
                eager, device, args.warmup, args.repetitions
            )
            candidate, triton_ms, triton_peak = timed(
                triton, device, args.warmup, args.repetitions
            )
            diff = candidate.float() - eager_output.float()
            item = {
                "role": role,
                "shape": list(shard.shape),
                "raw_mib": shard.raw.numel() / 2**20,
                "eager_ms": eager_ms,
                "triton_ms": triton_ms,
                "speedup": eager_ms / triton_ms,
                "eager_peak_extra_mib": eager_peak / 2**20,
                "triton_peak_extra_mib": triton_peak / 2**20,
                "max_abs": float(diff.abs().max().item()),
                "rms": float(diff.square().mean().sqrt().item()),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        candidate.float().flatten(), eager_output.float().flatten(), dim=0
                    ).item()
                ),
                "finite": bool(torch.isfinite(candidate).all().item()),
            }
            report["results"].append(item)
            print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)
            del shard, eager_output, candidate, diff
            torch.cuda.empty_cache()
    finally:
        reader.close()

    output = args.output or REPO_ROOT / "results" / "h3_q4_dequant_sm70.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved report: {output}", flush=True)


if __name__ == "__main__":
    main()
