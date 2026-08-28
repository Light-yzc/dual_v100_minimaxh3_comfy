#!/usr/bin/env python3
"""A/B benchmark for the experimental Qwen32 layer-MP prefetch path.

Run this script once with ``--prefetch 0`` and once with ``--prefetch 1`` in
separate processes.  It intentionally exercises only the 50 language layers
with a deterministic hidden state, so the result isolates read/dequant/GEMM
overlap from ComfyUI sampler and VAE work.  The script refuses to start when
the two GPUs do not have enough headroom for the bounded layer workspace.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DUAL_ROOT = REPO_ROOT / "custom_nodes" / "DualV100"
DEFAULT_MODEL = Path(
    "/mnt/GALAX/minimax-h3/models/text_encoders/"
    "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
)
MIB = 1 << 20


def _load_mp_module():
    package_name = "custom_nodes.DualV100"
    package = types.ModuleType(package_name)
    package.__path__ = [str(DUAL_ROOT)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.h3_qwen32_q2_mp")


def _cuda_snapshot() -> dict[str, list[int]]:
    values = {"allocated": [], "reserved": [], "free": []}
    for index in (0, 1):
        device = torch.device(f"cuda:{index}")
        free, _total = torch.cuda.mem_get_info(device)
        values["allocated"].append(int(torch.cuda.memory_allocated(device)))
        values["reserved"].append(int(torch.cuda.memory_reserved(device)))
        values["free"].append(int(free))
    return values


def _sync() -> None:
    for index in (0, 1):
        torch.cuda.synchronize(torch.device(f"cuda:{index}"))


def _error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    reference = reference.detach().double().cpu()
    candidate = candidate.detach().double().cpu()
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    delta = candidate - reference
    ref_norm = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    got_norm = torch.linalg.vector_norm(candidate).clamp_min(1e-12)
    return {
        "shape_match": True,
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_rms": float((torch.linalg.vector_norm(delta) / ref_norm).item()),
        "cosine": float((torch.sum(reference * candidate) / (ref_norm * got_norm)).item()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prefetch", type=int, choices=(0, 1), required=True)
    parser.add_argument("--prefetch-max-mib", type=int, default=256)
    parser.add_argument("--sequence", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional CPU .pt tensor from the other A/B run for exact comparison.",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        help="Optional path for the final conditioning tensor (CPU .pt).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("requires two CUDA devices")
    if args.sequence <= 0 or args.repetitions <= 0:
        raise ValueError("sequence and repetitions must be positive")

    before = _cuda_snapshot()
    # An evict layer needs roughly 685 MiB plus one optional 161 MiB
    # prefetched compressed layer.  The model planner itself retains an 86%
    # device-capacity safety ceiling, so require enough raw free memory to
    # satisfy that independent gate too.  This keeps the diagnostic process
    # from competing with an active H3 service only to fail after setup.
    total = []
    for index in (0, 1):
        _free, device_total = torch.cuda.mem_get_info(torch.device(f"cuda:{index}"))
        total.append(int(device_total))
    workspace = (1024 if args.prefetch else 768) * MIB
    planner_free = [
        max(0, int(value) - int(float(value) * 0.86)) + workspace
        for value in total
    ]
    required_free = max([workspace, *planner_free])
    if any(value < required_free for value in before["free"]):
        raise SystemExit(
            "insufficient GPU headroom for the benchmark; "
            f"free MiB={[value // MIB for value in before['free']]} "
            f"required>={required_free // MIB}"
        )

    mp = _load_mp_module()
    torch.manual_seed(20260827)
    for index in (0, 1):
        torch.cuda.reset_peak_memory_stats(torch.device(f"cuda:{index}"))
    runtime = mp.Qwen32Q2LayerMPRuntime(
        str(args.model),
        devices=("cuda:0", "cuda:1"),
        layer_split="auto",
        staging_mib=4,
        residency="evict",
        check_peer_access=True,
        enforce_capacity=True,
        prefetch=bool(args.prefetch),
        prefetch_max_mib=int(args.prefetch_max_mib),
    )
    hidden = torch.randn(
        (1, int(args.sequence), mp.qwen.QWEN32_HIDDEN_SIZE),
        device="cuda:0",
        dtype=torch.float32,
    )
    timings: list[float] = []
    checksums: list[float] = []
    started = time.perf_counter()
    try:
        for _ in range(int(args.repetitions)):
            call_started = time.perf_counter()
            output = runtime.qwen_forward(hidden)
            _sync()
            timings.append(time.perf_counter() - call_started)
            checksums.append(float(output.double().sum().item()))
            runtime.qwen_clear(notify_vae=False)
        elapsed = time.perf_counter() - started
        profile = runtime.qwen_stats().get("last_profile")
        rank0 = None if profile is None else profile.get("backbone")
        final_output = output.detach().float().cpu()
        report = {
            "variant": "prefetch" if args.prefetch else "baseline",
            "prefetch": bool(args.prefetch),
            "prefetch_max_mib": int(args.prefetch_max_mib),
            "model": str(args.model),
            "sequence": int(args.sequence),
            "repetitions": int(args.repetitions),
            "timings_seconds": timings,
            "mean_seconds": sum(timings) / len(timings),
            "total_seconds": elapsed,
            "checksums": checksums,
            "checksum_consistent": max(checksums) - min(checksums) <= 1e-4,
            "profile": profile,
            "rank0": rank0,
            "cuda_before": before,
            "cuda_after": _cuda_snapshot(),
            "cuda_peak_allocated": [
                int(torch.cuda.max_memory_allocated(torch.device(f"cuda:{i}")))
                for i in (0, 1)
            ],
            "cuda_peak_reserved": [
                int(torch.cuda.max_memory_reserved(torch.device(f"cuda:{i}")))
                for i in (0, 1)
            ],
        }
        if args.reference is not None:
            reference = torch.load(args.reference, map_location="cpu", weights_only=True)
            if not torch.is_tensor(reference):
                raise TypeError(f"reference must be a tensor, got {type(reference)!r}")
            report["vs_reference"] = _error_metrics(reference, final_output)
        if args.dump is not None:
            args.dump.parent.mkdir(parents=True, exist_ok=True)
            torch.save(final_output, args.dump)
            report["conditioning_dump"] = str(args.dump)
    finally:
        try:
            runtime.qwen_clear(notify_vae=False)
        except BaseException:
            pass
        runtime.close()
        del hidden, runtime
        gc.collect()
        for index in (0, 1):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
