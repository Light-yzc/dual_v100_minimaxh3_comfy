#!/usr/bin/env python3
"""Benchmark the H3-specific fused SM70 Q/K RMSNorm + partial RoPE kernel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_v100_rms_rope.py"


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_rotation_table(sequence: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(2031)
    angles = torch.randn((sequence, 48), device=device, generator=generator)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, sequence, 1, 48, 2, 2
    ).half()


def make_qk(sequence: int, heads: int, device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    packed = torch.randn(
        (1, sequence, 3 * heads * 128),
        dtype=torch.float16,
        device=device,
        generator=generator,
    ) * 40
    inner = heads * 128
    q = packed[:, :, :inner].view(1, sequence, heads, 128)
    k = packed[:, :, inner : 2 * inner].view(1, sequence, heads, 128)
    return packed, q, k


def reference_(q, k, freqs, q_weight, k_weight, epsilon, eager_rope):
    q_scale = q.detach().abs().amax(dim=-1, keepdim=True).clamp_min_(1.0)
    k_scale = k.detach().abs().amax(dim=-1, keepdim=True).clamp_min_(1.0)
    q.div_(q_scale)
    k.div_(k_scale)
    return eager_rope.rms_rope_split_half_(
        q, k, freqs, q_weight, k_weight, epsilon=epsilon, rot_dim=96
    )


def timed(function, device, warmup, repetitions):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function()
    end.record()
    end.synchronize()
    return (
        start.elapsed_time(end) / repetitions,
        max(0, torch.cuda.max_memory_allocated(device) - baseline),
    )


def errors(reference, candidate):
    delta = candidate.float() - reference.float()
    ref = reference.float()
    cand = candidate.float()
    cosine = torch.sum(ref * cand) / (
        torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cand)
    )
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rms": float(torch.sqrt(torch.mean(delta.square())).item()),
        "cosine": float(cosine.item()),
        "finite": bool(torch.isfinite(cand).all().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", nargs="+", type=int, default=[128, 2048, 8192])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise SystemExit("benchmark requires SM70")
    kernel = import_path("h3_v100_rms_rope_bench", KERNEL_PATH)
    from comfy_kitchen.backends.eager import rope as eager_rope

    q_weight = torch.randn(128, device=device, dtype=torch.float16) * 0.1 + 1
    k_weight = torch.randn(128, device=device, dtype=torch.float16) * 0.1 + 1
    epsilon = 1e-6
    report = {
        "created_unix": time.time(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": kernel.triton.__version__,
        "gpu": props.name,
        "heads": args.heads,
        "results": [],
    }

    for sequence in args.sequences:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        required = sequence * 3 * args.heads * 128 * 2 * 4 + 1024 * 2**20
        item = {
            "sequence": sequence,
            "free_before_mib": free_bytes / 2**20,
            "guard_required_mib": required / 2**20,
        }
        if free_bytes < required:
            item["skipped"] = "VRAM safety guard"
            report["results"].append(item)
            continue

        source, _, _ = make_qk(sequence, args.heads, device, 2033)
        reference_storage = source.clone()
        inner = args.heads * 128
        q_ref = reference_storage[:, :, :inner].view(1, sequence, args.heads, 128)
        k_ref = reference_storage[:, :, inner : 2 * inner].view(1, sequence, args.heads, 128)
        freqs = make_rotation_table(sequence, device)
        reference_(q_ref, k_ref, freqs, q_weight, k_weight, epsilon, eager_rope)

        candidates = []
        for warps in args.warps:
            candidate_storage = source.clone()
            q_candidate = candidate_storage[:, :, :inner].view(1, sequence, args.heads, 128)
            k_candidate = candidate_storage[:, :, inner : 2 * inner].view(1, sequence, args.heads, 128)
            kernel.h3_qk_rms_rope_sm70_(
                q_candidate,
                k_candidate,
                freqs,
                q_weight,
                k_weight,
                epsilon=epsilon,
                rot_dim=96,
                stabilize=True,
                num_warps=warps,
            )
            torch.cuda.synchronize(device)
            q_error = errors(q_ref, q_candidate)
            k_error = errors(k_ref, k_candidate)

            timing_storage = source.clone()
            q_timing = timing_storage[:, :, :inner].view(1, sequence, args.heads, 128)
            k_timing = timing_storage[:, :, inner : 2 * inner].view(1, sequence, args.heads, 128)
            milliseconds, peak = timed(
                lambda: kernel.h3_qk_rms_rope_sm70_(
                    q_timing,
                    k_timing,
                    freqs,
                    q_weight,
                    k_weight,
                    epsilon=epsilon,
                    rot_dim=96,
                    stabilize=True,
                    num_warps=warps,
                ),
                device,
                args.warmup,
                args.repetitions,
            )
            candidates.append(
                {
                    "warps": warps,
                    "milliseconds": milliseconds,
                    "peak_extra_mib": peak / 2**20,
                    "q_error": q_error,
                    "k_error": k_error,
                }
            )
            del candidate_storage, timing_storage

        timing_reference = source.clone()
        q_timing_ref = timing_reference[:, :, :inner].view(1, sequence, args.heads, 128)
        k_timing_ref = timing_reference[:, :, inner : 2 * inner].view(1, sequence, args.heads, 128)
        eager_ms, eager_peak = timed(
            lambda: reference_(
                q_timing_ref,
                k_timing_ref,
                freqs,
                q_weight,
                k_weight,
                epsilon,
                eager_rope,
            ),
            device,
            args.warmup,
            args.repetitions,
        )
        item["current_eager"] = {
            "milliseconds": eager_ms,
            "peak_extra_mib": eager_peak / 2**20,
        }
        for candidate in candidates:
            candidate["speedup_vs_current_eager"] = eager_ms / candidate["milliseconds"]
        item["sm70_candidates"] = candidates
        report["results"].append(item)
        print(json.dumps(item, indent=2), flush=True)
        del source, reference_storage, timing_reference, freqs
        torch.cuda.empty_cache()

    output = args.output or (
        REPO_ROOT / "results" / f"h3_rms_rope_sm70_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"saved report: {output}")


if __name__ == "__main__":
    main()
