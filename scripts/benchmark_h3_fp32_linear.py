#!/usr/bin/env python3
"""Numerical, overflow, and speed gate for H3's SM70 FP32-output Linear."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "h3_v100_fp32_linear_gate.json"
MIB = 1 << 20


def load_wide_linear_module():
    path = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_v100_fp32_linear.py"
    spec = importlib.util.spec_from_file_location("h3_v100_fp32_linear_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import H3 FP32 Linear helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wide_linear = load_wide_linear_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2034)
    parser.add_argument("--min-free-mib", type=int, default=768)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    reference = reference.float()
    candidate = candidate.float()
    difference = candidate - reference
    reference_rms = torch.sqrt(torch.mean(reference.square()))
    difference_rms = torch.sqrt(torch.mean(difference.square()))
    cosine = functional.cosine_similarity(
        reference.reshape(1, -1), candidate.reshape(1, -1), dim=1
    )[0]
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rms": float(difference_rms.item()),
        "reference_rms": float(reference_rms.item()),
        "relative_rms": float((difference_rms / reference_rms).item()),
        "cosine": float(cosine.item()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def median_cuda_ms(function, warmup: int, repetitions: int, device: torch.device) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
        del output
    return statistics.median(samples)


def run_geometry(
    *,
    role: str,
    rows: int,
    in_features: int,
    out_features: int,
    scale_input: bool,
    generator: torch.Generator,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> dict[str, object]:
    weight = torch.empty(
        (out_features, in_features), dtype=torch.float16, device=device
    ).normal_(mean=0.0, std=0.006, generator=generator)
    x = torch.empty((rows, in_features), dtype=torch.float32, device=device).normal_(
        mean=0.0, std=8.0 if scale_input else 1.0, generator=generator
    )
    fp32_weight = weight.float()
    reference = functional.linear(x, fp32_weight)
    candidate = wide_linear.tensor_core_fp32_output_linear(
        x, weight, scale_input=scale_input
    )
    error = tensor_error(reference, candidate)

    fp32_ms = median_cuda_ms(
        lambda: functional.linear(x, fp32_weight), warmup, repetitions, device
    )
    tensor_core_ms = median_cuda_ms(
        lambda: wide_linear.tensor_core_fp32_output_linear(
            x, weight, scale_input=scale_input
        ),
        warmup,
        repetitions,
        device,
    )
    result = {
        "role": role,
        "shape": [rows, in_features, out_features],
        "scale_input": scale_input,
        "fp32_gemm_ms": fp32_ms,
        "tensor_core_fp32_output_ms": tensor_core_ms,
        "speedup": fp32_ms / tensor_core_ms,
        "error_vs_fp32_gemm": error,
    }
    del candidate, reference, fp32_weight, x, weight
    torch.cuda.empty_cache()
    return result


def run_overflow_gate(
    rows: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, object]:
    in_features, out_features = 2048, 1024
    x = torch.empty((rows, in_features), dtype=torch.float32, device=device).normal_(
        mean=0.0, std=100000.0, generator=generator
    )
    x[:, 0] = torch.linspace(1.0e6, 1.0e8, rows, device=device)
    weight = torch.empty(
        (out_features, in_features), dtype=torch.float16, device=device
    ).normal_(mean=0.0, std=1.0e-5, generator=generator)
    reference = functional.linear(x, weight.float())
    scale = wide_linear.power_of_two_fp16_scale(x)
    scaled = x / scale
    candidate = wide_linear.tensor_core_fp32_output_linear(
        x, weight, scale_input=True
    )
    scale_log2 = torch.log2(scale)
    result = {
        "shape": [rows, in_features, out_features],
        "unscaled_fp16_input_finite": bool(torch.isfinite(x.to(torch.float16)).all().item()),
        "scaled_fp16_input_finite": bool(torch.isfinite(scaled.to(torch.float16)).all().item()),
        "scaled_max_abs": float(scaled.abs().max().item()),
        "scale_min": float(scale.min().item()),
        "scale_max": float(scale.max().item()),
        "all_scales_power_of_two": bool(
            torch.equal(scale_log2, torch.round(scale_log2))
        ),
        "error_vs_fp32_gemm": tensor_error(reference, candidate),
    }
    del candidate, scaled, scale, reference, weight, x
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.warmup < 0 or args.repetitions <= 0:
        raise SystemExit("rows/repetitions must be positive and warmup non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise SystemExit(f"expected SM70, got sm_{capability[0]}{capability[1]}")
    free_before, total = torch.cuda.mem_get_info(device)
    if free_before < args.min_free_mib * MIB:
        raise RuntimeError(
            f"VRAM guard: only {free_before / MIB:.0f} MiB free on {device}; "
            f"need {args.min_free_mib} MiB"
        )

    torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    geometries = [
        run_geometry(
            role="attention_out",
            rows=args.rows,
            in_features=7168,
            out_features=5376,
            scale_input=False,
            generator=generator,
            device=device,
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        run_geometry(
            role="mlp_fc2",
            rows=args.rows,
            in_features=14336,
            out_features=5376,
            scale_input=True,
            generator=generator,
            device=device,
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
    ]
    overflow = run_overflow_gate(args.rows, generator, device)
    qualified = (
        all(
            item["error_vs_fp32_gemm"]["finite"]
            and item["error_vs_fp32_gemm"]["relative_rms"] <= 1.0e-3
            and item["error_vs_fp32_gemm"]["cosine"] >= 0.999999
            for item in geometries
        )
        and not overflow["unscaled_fp16_input_finite"]
        and overflow["scaled_fp16_input_finite"]
        and overflow["all_scales_power_of_two"]
        and overflow["error_vs_fp32_gemm"]["finite"]
        and overflow["error_vs_fp32_gemm"]["relative_rms"] <= 1.0e-3
    )

    result = {
        "created_unix": time.time(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "seed": args.seed,
        "fp16_scale_target": wide_linear.FP16_SCALE_TARGET,
        "geometries": geometries,
        "overflow_gate": overflow,
        "resources": {
            "free_before_mib": free_before / MIB,
            "total_vram_mib": total / MIB,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
            "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "numerically_qualified": qualified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not qualified:
        raise SystemExit("H3 FP32 Tensor Core numerical gate failed")


if __name__ == "__main__":
    main()
