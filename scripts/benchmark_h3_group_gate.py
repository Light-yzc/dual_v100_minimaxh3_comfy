#!/usr/bin/env python3
"""Measure the Q4 and bounded-signature group-cache decision gates.

This benchmark intentionally does not load an H3 checkpoint.  It allocates one
1 MP-sized residual stream and measures only the representation needed to
decide whether a stored group residual may be reused.  Run each mode in a
separate process so the reported host RSS and CUDA peaks are independent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
Q4_CACHE_PATH = REPO_ROOT / "custom_nodes/DualV100/h3_q4_cache.py"
Q4_TP_PATH = REPO_ROOT / "custom_nodes/DualV100/h3_q4_tp.py"
sys.path.insert(0, str(REPO_ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rss_mib() -> float:
    # Linux reports ru_maxrss in KiB.  Keep the fallback local and explicit so
    # this script remains usable without psutil.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def cuda_measure(device: torch.device, function):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    peak_extra = max(0, torch.cuda.max_memory_allocated(device) - baseline)
    return result, elapsed_ms, peak_extra / 2**20


def cpu_measure(function):
    started = time.perf_counter()
    result = function()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, elapsed_ms


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("q4", "signature"), required=True)
    result.add_argument("--rows", type=int, default=37746)
    result.add_argument("--cols", type=int, default=5376)
    result.add_argument("--max-tokens", type=int, default=2048)
    result.add_argument("--hidden-samples", type=int, default=32)
    result.add_argument("--chunk-rows", type=int, default=256)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=20260828)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (7, 0):
        raise SystemExit(
            f"expected SM70 for this benchmark, got {properties.major}.{properties.minor}"
        )
    if args.rows <= 0 or args.cols <= 0 or args.cols % 32:
        raise SystemExit("rows must be positive and cols must be a multiple of 32")

    # The cache module imports this sibling relatively in normal package mode;
    # load it under the package-local name for standalone execution first.
    if "h3_q4_tp" not in sys.modules:
        load_module("h3_q4_tp", Q4_TP_PATH)
    q4_cache = load_module("h3_group_gate_q4_cache", Q4_CACHE_PATH)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    # H3 residuals use FP32 on the V100 path.  Keep exactly one current stream
    # plus one reference stream; no model weights or full hidden history exist.
    reference = torch.randn(
        (args.rows, args.cols), generator=generator, dtype=torch.float32, device=device
    )
    reference.mul_(0.37)
    current = reference.clone()
    current[:, ::97].add_(0.0025)

    report = {
        "created_unix": time.time(),
        "mode": args.mode,
        "rows": args.rows,
        "cols": args.cols,
        "device": str(device),
        "gpu": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "chunk_rows": args.chunk_rows,
        "max_tokens": args.max_tokens,
        "hidden_samples": args.hidden_samples,
        "input_bytes_each": int(reference.numel() * reference.element_size()),
        "rss_before_mib": rss_mib(),
    }

    if args.mode == "q4":
        stored, stored_ms, stored_peak = cuda_measure(
            device,
            lambda: q4_cache.quantize_q4_0(
                reference,
                policy="cpu",
                chunk_rows=args.chunk_rows,
                measure=True,
            ),
        )
        # The stored representation is outside the timed current-input gate,
        # matching a cache-ready inference step.
        del stored_ms, stored_peak

        def timed_current():
            current_q4 = q4_cache.quantize_q4_0(
                current,
                policy="cpu",
                chunk_rows=args.chunk_rows,
                measure=True,
            )
            value, details = q4_cache.relative_difference(
                current_q4,
                stored,
                metric="relative_l2",
                device=device,
                chunk_rows=args.chunk_rows,
                measure=True,
            )
            return current_q4, value, details

        measured, gate_ms, gate_peak = cuda_measure(device, timed_current)
        current_q4, feature_error, metric_details = measured
        report.update(
            {
                "representation": "ggml_q4_0",
                "feature_error": feature_error,
                "gate_ms": gate_ms,
                "gate_peak_extra_mib": gate_peak,
                "stored_bytes": stored.bytes,
                "current_bytes": current_q4.bytes,
                "stored_quantize_report": stored.quantize_report,
                "current_quantize_report": current_q4.quantize_report,
                "metric_report": metric_details,
            }
        )
        del current_q4, stored
    else:
        ranges = ((0, args.rows // 2, 0), (args.rows // 2, args.rows, 1))
        stored, stored_ms = cpu_measure(
            lambda: q4_cache.deterministic_input_signature(
                reference,
                max_tokens=args.max_tokens,
                hidden_samples=args.hidden_samples,
                ranges=ranges,
            )
        )
        del stored_ms
        stored_signature, stored_metadata = stored

        def timed_current():
            current_signature, current_metadata = (
                q4_cache.deterministic_input_signature(
                    current,
                    max_tokens=args.max_tokens,
                    hidden_samples=args.hidden_samples,
                    ranges=ranges,
                )
            )
            value, details = q4_cache.signature_difference(
                current_signature,
                stored_signature,
                metric="relative_l2",
                current_metadata=current_metadata,
                reference_metadata=stored_metadata,
                aggregation="max_segment",
            )
            return current_signature, value, details

        measured, gate_ms = cpu_measure(timed_current)
        current_signature, feature_error, metric_details = measured
        report.update(
            {
                "representation": "fp32_bounded_signature",
                "feature_error": feature_error,
                "gate_ms": gate_ms,
                "gate_peak_extra_mib": 0.0,
                "stored_bytes": int(
                    stored_signature.numel() * stored_signature.element_size()
                ),
                "current_bytes": int(
                    current_signature.numel() * current_signature.element_size()
                ),
                "signature_metadata": stored_metadata,
                "metric_report": metric_details,
            }
        )
        del current_signature, stored_signature, stored_metadata

    del current, reference
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    report["rss_peak_mib"] = rss_mib()
    report["cuda_allocated_after_mib"] = torch.cuda.memory_allocated(device) / 2**20
    report["cuda_reserved_after_mib"] = torch.cuda.memory_reserved(device) / 2**20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
