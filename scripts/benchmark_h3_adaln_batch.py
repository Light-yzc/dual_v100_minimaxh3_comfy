#!/usr/bin/env python3
"""Benchmark batched FP32 AdaLN preparation for all 50 H3 blocks."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path("/home/regen/minimax-h3/ComfyUI")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(REPO_ROOT / "custom_nodes" / "DualV100"))

import h3_lora_tp as lora_tp  # noqa: E402
import h3_q4_tp as q4_tp  # noqa: E402
import h3_tp_backbone as tp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=tp.DEFAULT_MODEL)
    parser.add_argument("--lora", type=Path, default=tp.DEFAULT_LORA)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--staging-mib", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2043)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "h3_adaln_batch_sm70_gate.json",
    )
    return parser.parse_args()


def load_values(args, device):
    dense_requested = {
        name
        for block in range(tp.LAYERS)
        for role, name in tp.dense_names(block).items()
        if role in {"adaln_weight", "adaln_bias"}
    }
    dense_specs = tp._inspect_dense_specs(args.model, dense_requested)
    weights, biases = [], []
    with q4_tp.Q4DiskReader(args.model, device, args.staging_mib << 20) as reader:
        for block in range(tp.LAYERS):
            names = tp.dense_names(block)
            weights.append(tp._read_dense(reader, dense_specs[names["adaln_weight"]], torch.float32))
            biases.append(tp._read_dense(reader, dense_specs[names["adaln_bias"]], torch.float32))

    lora_requested = {
        name for block in range(tp.LAYERS) for name in tp.adaln_lora_names(block)
    }
    lora_specs, metadata = lora_tp.inspect_safetensors(args.lora, lora_requested)
    lora_a, lora_b = [], []
    with lora_tp.SafeTensorDiskReader(args.lora, device, args.staging_mib << 20) as reader:
        for block in range(tp.LAYERS):
            a_name, b_name = tp.adaln_lora_names(block)
            lora_a.append(reader.read_full(lora_specs[a_name], torch.float32))
            lora_b.append(reader.read_full(lora_specs[b_name], torch.float32))
    return (
        torch.stack(weights),
        torch.stack(biases),
        torch.stack(lora_a),
        torch.stack(lora_b),
        metadata,
    )


def eager_all(t_emb, silu_temb, weights, biases, lora_a, lora_b, strength):
    outputs = []
    for block in range(weights.shape[0]):
        output = functional.linear(t_emb, weights[block], biases[block])
        delta = functional.linear(
            functional.linear(silu_temb, lora_a[block]), lora_b[block]
        )
        outputs.append(output.add_(delta, alpha=strength))
    return torch.stack(outputs)


def batched_all(t_emb, silu_temb, weights, biases, lora_a, lora_b, strength):
    blocks = weights.shape[0]
    t_batch = t_emb.unsqueeze(0).expand(blocks, -1, -1)
    silu_batch = silu_temb.unsqueeze(0).expand(blocks, -1, -1)
    output = torch.bmm(t_batch, weights.transpose(1, 2))
    output.add_(biases.unsqueeze(1))
    rank_output = torch.bmm(silu_batch, lora_a.transpose(1, 2))
    delta = torch.bmm(rank_output, lora_b.transpose(1, 2))
    return output.add_(delta, alpha=strength)


def tensor_core_all(t_emb, silu_temb, weights16, biases, lora_a16, lora_b16, strength):
    blocks = weights16.shape[0]
    t_batch = t_emb.half().unsqueeze(0).expand(blocks, -1, -1)
    silu_batch = silu_temb.half().unsqueeze(0).expand(blocks, -1, -1)
    output = torch.bmm(
        t_batch, weights16.transpose(1, 2), out_dtype=torch.float32
    )
    output.add_(biases.unsqueeze(1))
    rank_output = torch.bmm(
        silu_batch, lora_a16.transpose(1, 2), out_dtype=torch.float32
    )
    rank_half = rank_output.half()
    delta = torch.bmm(
        rank_half, lora_b16.transpose(1, 2), out_dtype=torch.float32
    )
    return output.add_(delta, alpha=strength)


def tensor_error(reference, actual):
    delta = actual - reference
    rms = delta.square().mean().sqrt()
    ref_rms = reference.square().mean().sqrt()
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rms": float(rms.item()),
        "reference_rms": float(ref_rms.item()),
        "relative_rms": float((rms / ref_rms.clamp_min(1e-30)).item()),
        "cosine": float(functional.cosine_similarity(reference.flatten(), actual.flatten(), dim=0).item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def timed(call, warmup, repetitions):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repetitions)
    return statistics.median(values)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise SystemExit("this benchmark requires SM70")
    torch.cuda.reset_peak_memory_stats(device)
    loaded_at = time.monotonic()
    weights, biases, lora_a, lora_b, metadata = load_values(args, device)
    torch.cuda.synchronize(device)
    load_seconds = time.monotonic() - loaded_at
    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_emb = torch.randn(
        (args.rows, tp.ADALN_INPUT), generator=generator, device=device, dtype=torch.float32
    )
    silu_temb = torch.randn(
        (args.rows, tp.ADALN_TIME_DIM), generator=generator, device=device, dtype=torch.float32
    )

    eager = eager_all(t_emb, silu_temb, weights, biases, lora_a, lora_b, args.strength)
    batched = batched_all(t_emb, silu_temb, weights, biases, lora_a, lora_b, args.strength)
    weights16, lora_a16, lora_b16 = weights.half(), lora_a.half(), lora_b.half()
    tensor_core = tensor_core_all(
        t_emb, silu_temb, weights16, biases, lora_a16, lora_b16, args.strength
    )
    batched_error = tensor_error(eager, batched)
    tensor_core_error = tensor_error(eager, tensor_core)
    eager_ms = timed(
        lambda: eager_all(t_emb, silu_temb, weights, biases, lora_a, lora_b, args.strength),
        args.warmup,
        args.repetitions,
    )
    batched_ms = timed(
        lambda: batched_all(t_emb, silu_temb, weights, biases, lora_a, lora_b, args.strength),
        args.warmup,
        args.repetitions,
    )
    tensor_core_ms = timed(
        lambda: tensor_core_all(
            t_emb, silu_temb, weights16, biases, lora_a16, lora_b16, args.strength
        ),
        args.warmup,
        args.repetitions,
    )
    report = {
        "created_unix": time.time(),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": "7.0",
        "blocks": tp.LAYERS,
        "rows": args.rows,
        "output_shape": list(batched.shape),
        "model": str(args.model),
        "lora": str(args.lora),
        "lora_metadata": metadata,
        "payload_mmap": False,
        "load_seconds": load_seconds,
        "eager_all_blocks_ms": eager_ms,
        "batched_fp32_all_blocks_ms": batched_ms,
        "batched_fp32_speedup": eager_ms / batched_ms,
        "batched_fp32_error": batched_error,
        "tensor_core_all_blocks_ms": tensor_core_ms,
        "tensor_core_speedup": eager_ms / tensor_core_ms,
        "tensor_core_error": tensor_core_error,
        "tensor_core_rank_max_abs": float(
            torch.bmm(
                silu_temb.half().unsqueeze(0).expand(tp.LAYERS, -1, -1),
                lora_a16.transpose(1, 2),
                out_dtype=torch.float32,
            ).abs().max().item()
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
        "numerically_qualified": bool(
            tensor_core_error["finite"]
            and tensor_core_error["cosine"] >= 0.99999
            and tensor_core_error["relative_rms"] <= 2e-3
            and tensor_core_ms < eager_ms
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved report: {args.output}")


if __name__ == "__main__":
    main()
