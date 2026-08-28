#!/usr/bin/env python3
"""Numerical/performance gate for fused H3 FP32-residual operations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[1]
OPS_PATH = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_v100_fp32_ops.py"


def load_ops():
    spec = importlib.util.spec_from_file_location("h3_v100_fp32_ops_benchmark", OPS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {OPS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "h3_fp32_ops_sm70_gate.json",
    )
    return parser.parse_args()


def tensor_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    ref = reference.float()
    got = actual.float()
    delta = got - ref
    ref_rms = ref.square().mean().sqrt()
    rms = delta.square().mean().sqrt()
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rms": float(rms.item()),
        "reference_rms": float(ref_rms.item()),
        "relative_rms": float((rms / ref_rms.clamp_min(1e-12)).item()),
        "cosine": float(functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()),
        "finite": bool(torch.isfinite(got).all().item()),
    }


def time_call(call, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repetitions)
    return statistics.median(samples)


def main() -> None:
    args = parse_args()
    ops = load_ops()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise SystemExit("this benchmark requires SM70")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sequence = args.sequence
    hidden = ops.H3_HIDDEN
    midpoint = sequence // 2
    segments = [[0, midpoint, 0], [midpoint, sequence, 3]]
    mod_rows = ops.make_modulation_rows(sequence, segments, device)

    residual = torch.randn(
        (sequence, hidden), generator=generator, device=device, dtype=torch.float32
    ).mul_(4096.0)
    weight = torch.randn((hidden,), generator=generator, device=device, dtype=torch.float16)
    shift = torch.randn((6, hidden), generator=generator, device=device, dtype=torch.float32)
    scale = torch.randn((6, hidden), generator=generator, device=device, dtype=torch.float32).mul_(0.25)
    gate = torch.randn((6, hidden), generator=generator, device=device, dtype=torch.float32).mul_(0.25)
    update = torch.randn(
        (sequence, hidden), generator=generator, device=device, dtype=torch.float32
    ).mul_(512.0)
    packed_fc1 = torch.randn(
        (sequence, 2 * ops.H3_LOCAL_FFN),
        generator=generator,
        device=device,
        dtype=torch.float16,
    ).mul_(128.0)

    def eager_rms_mod():
        output = functional.rms_norm(residual, (hidden,), weight=weight, eps=1e-5).half()
        for start, stop, row in segments:
            output[start:stop].mul_(1.0 + scale[row].half()).add_(shift[row].half())
        return output

    def fused_rms_mod():
        return ops.h3_fp32_rms_mod_sm70(
            residual, weight, shift, scale, mod_rows, epsilon=1e-5, num_warps=args.warps
        )

    eager_gate_buffer = residual.clone()
    fused_gate_buffer = residual.clone()

    def eager_gate():
        output = eager_gate_buffer
        for start, stop, row in segments:
            output[start:stop].addcmul_(update[start:stop].float(), gate[row].float())
        return output

    def fused_gate():
        output = fused_gate_buffer
        return ops.h3_fp32_gate_residual_sm70_(
            output, update, gate, mod_rows, num_warps=args.warps
        )

    def eager_swiglu_scale():
        packed_gate, packed_up = packed_fc1.chunk(2, dim=-1)
        swiglu = functional.silu(packed_gate.float()).mul_(packed_up.float())
        maximum = swiglu.detach().abs().amax(dim=-1, keepdim=True)
        ratio = (maximum / ops.FP16_SCALE_TARGET).clamp_min_(1.0)
        row_scale = torch.exp2(torch.ceil(torch.log2(ratio)))
        safe = (swiglu / row_scale).half()
        return swiglu, safe, row_scale

    def fused_swiglu_scale():
        return ops.h3_swiglu_scale_sm70(
            packed_fc1, target=ops.FP16_SCALE_TARGET, num_warps=args.warps
        )

    eager_rms = eager_rms_mod()
    fused_rms = fused_rms_mod()
    eager_residual = residual.clone()
    fused_residual = residual.clone()
    for start, stop, row in segments:
        eager_residual[start:stop].addcmul_(update[start:stop], gate[row])
    ops.h3_fp32_gate_residual_sm70_(
        fused_residual, update, gate, mod_rows, num_warps=args.warps
    )
    eager_swiglu, eager_safe, eager_scale = eager_swiglu_scale()
    fused_swiglu, fused_safe, fused_scale = fused_swiglu_scale()
    rms_error = tensor_error(eager_rms, fused_rms)
    gate_error = tensor_error(eager_residual, fused_residual)
    swiglu_error = tensor_error(eager_swiglu, fused_swiglu)
    safe_error = tensor_error(eager_safe, fused_safe)
    scale_error = tensor_error(eager_scale, fused_scale)
    eager_rms_ms = time_call(eager_rms_mod, args.warmup, args.repetitions)
    fused_rms_ms = time_call(fused_rms_mod, args.warmup, args.repetitions)
    eager_gate_ms = time_call(eager_gate, args.warmup, args.repetitions)
    fused_gate_ms = time_call(fused_gate, args.warmup, args.repetitions)
    eager_swiglu_ms = time_call(eager_swiglu_scale, args.warmup, args.repetitions)
    fused_swiglu_ms = time_call(fused_swiglu_scale, args.warmup, args.repetitions)

    torch.cuda.synchronize(device)
    report = {
        "created_unix": time.time(),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": "7.0",
        "shape": [sequence, hidden],
        "segments": segments,
        "warps": args.warps,
        "rms_mod": {
            "eager_ms": eager_rms_ms,
            "fused_ms": fused_rms_ms,
            "speedup": eager_rms_ms / fused_rms_ms,
            "error": rms_error,
            "output_sha256": hashlib.sha256(fused_rms.cpu().numpy().tobytes()).hexdigest(),
        },
        "gate_residual": {
            "eager_ms": eager_gate_ms,
            "fused_ms": fused_gate_ms,
            "speedup": eager_gate_ms / fused_gate_ms,
            "error": gate_error,
            "output_sha256": hashlib.sha256(fused_residual.cpu().numpy().tobytes()).hexdigest(),
        },
        "swiglu_scale": {
            "eager_ms": eager_swiglu_ms,
            "fused_ms": fused_swiglu_ms,
            "speedup": eager_swiglu_ms / fused_swiglu_ms,
            "swiglu_error": swiglu_error,
            "safe_fp16_error": safe_error,
            "scale_error": scale_error,
            "scale_min": float(fused_scale.min().item()),
            "scale_max": float(fused_scale.max().item()),
            "safe_max_abs": float(fused_safe.abs().max().item()),
        },
        "numerically_qualified": bool(
            rms_error["finite"]
            and rms_error["cosine"] >= 0.99999
            and rms_error["relative_rms"] <= 2e-3
            and gate_error["finite"]
            and gate_error["cosine"] >= 0.999999
            and gate_error["relative_rms"] <= 1e-6
            and swiglu_error["finite"]
            and swiglu_error["cosine"] >= 0.999999
            and swiglu_error["relative_rms"] <= 2e-4
            and safe_error["finite"]
            and safe_error["relative_rms"] <= 2e-4
            and scale_error["max_abs"] == 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved report: {args.output}")


if __name__ == "__main__":
    main()
