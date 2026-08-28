#!/usr/bin/env python3
"""Test whether query-row chunking recovers SDPA efficiency at long sequence.

Efficient SDPA measures 39.1 TFLOPS at S=4096 but only 34.0 TFLOPS at
S=37746 for the H3 local-head shape, a 13% decline.  Softmax is per query
row, so splitting the query rows into chunks -- each chunk attending to the
*full* key/value sequence -- is mathematically exact and performs exactly the
same number of FLOPs.  The only question is whether the CUTLASS FMHA kernel
schedules a short-query launch better than one long-query launch.

This is not the discarded "sequence parallel" idea, which split query rows
*and* kept each rank's 28 local heads, thereby dropping half the required
head contributions.  Here every chunk sees every local head and every key.

No checkpoint is loaded and no ComfyUI service is touched.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

HEAD_DIM = 128
LOCAL_HEADS = 28


def timed(callable_, warmup: int, repetitions: int) -> float:
    """Return the median wall time in milliseconds of a CUDA callable."""
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    samples.sort()
    return samples[len(samples) // 2]


def attention_whole(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Production form: one SDPA launch over the whole query sequence."""
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        return functional.scaled_dot_product_attention(
            query, keys, values, attn_mask=None, dropout_p=0.0, is_causal=False
        )


def attention_chunked(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    chunk_rows: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Exact equivalent: query rows in chunks, full keys and values per chunk."""
    sequence = query.shape[2]
    for start in range(0, sequence, chunk_rows):
        stop = min(start + chunk_rows, sequence)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out[:, :, start:stop] = functional.scaled_dot_product_attention(
                query[:, :, start:stop],
                keys,
                values,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
    return out


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Absolute and relative agreement between two attention outputs."""
    left = reference.float()
    right = candidate.float()
    difference = (left - right).abs()
    denominator = left.norm().item()
    return {
        "max_abs": difference.max().item(),
        "rms": difference.pow(2).mean().sqrt().item(),
        "relative_rms": ((left - right).norm().item() / denominator) if denominator else 0.0,
        "cosine": functional.cosine_similarity(
            left.reshape(1, -1), right.reshape(1, -1)
        ).item(),
        "finite": bool(torch.isfinite(right).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence", type=int, default=37746)
    parser.add_argument(
        "--chunk-rows", type=int, nargs="+", default=[2048, 4096, 8192, 16384]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise SystemExit(f"expected SM70, got sm_{props.major}{props.minor}")

    shape = (1, LOCAL_HEADS, args.sequence, HEAD_DIM)
    generator = torch.Generator(device=device).manual_seed(20260828)
    query = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
    keys = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
    values = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
    flop = 4.0 * LOCAL_HEADS * args.sequence * args.sequence * HEAD_DIM

    torch.cuda.reset_peak_memory_stats(device)
    baseline_ms = timed(
        lambda: attention_whole(query, keys, values), args.warmup, args.repetitions
    )
    baseline_peak = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    reference = attention_whole(query, keys, values)
    print(
        f"whole   S={args.sequence}: {baseline_ms:.3f} ms  "
        f"{flop / (baseline_ms * 1e-3) / 1e12:.1f} TFLOPS  peak {baseline_peak:.0f} MiB",
        flush=True,
    )

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "checkpoint_loaded": False,
        "host_mmap": False,
        "sequence": args.sequence,
        "heads": LOCAL_HEADS,
        "head_dim": HEAD_DIM,
        "baseline": {
            "label": "single SDPA launch over whole query sequence",
            "ms": baseline_ms,
            "tflops": flop / (baseline_ms * 1e-3) / 1e12,
            "peak_allocated_mib": baseline_peak,
        },
        "candidates": [],
    }

    out = torch.empty_like(reference)
    for chunk_rows in args.chunk_rows:
        if chunk_rows >= args.sequence:
            continue
        torch.cuda.reset_peak_memory_stats(device)
        elapsed = timed(
            lambda rows=chunk_rows: attention_chunked(query, keys, values, rows, out),
            args.warmup,
            args.repetitions,
        )
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        candidate = attention_chunked(query, keys, values, chunk_rows, out.clone())
        error = tensor_error(reference, candidate)
        entry = {
            "chunk_rows": chunk_rows,
            "chunks": -(-args.sequence // chunk_rows),
            "ms": elapsed,
            "tflops": flop / (elapsed * 1e-3) / 1e12,
            "speedup_vs_whole": baseline_ms / elapsed,
            "peak_allocated_mib": peak_mib,
            "numerical_vs_whole": error,
            "numerically_qualified": bool(
                error["finite"] and error["relative_rms"] <= 1e-6
            ),
        }
        report["candidates"].append(entry)
        print(
            f"chunk {chunk_rows:>5}: {elapsed:.3f} ms  {entry['tflops']:.1f} TFLOPS  "
            f"{entry['speedup_vs_whole']:.4f}x  peak {peak_mib:.0f} MiB  "
            f"max_abs {error['max_abs']:.3e}  qualified {entry['numerically_qualified']}",
            flush=True,
        )
        del candidate

    best = max(report["candidates"], key=lambda entry: entry["speedup_vs_whole"], default=None)
    if best is not None:
        report["summary"] = {
            "best_chunk_rows": best["chunk_rows"],
            "best_speedup_vs_whole": best["speedup_vs_whole"],
            "best_numerically_qualified": best["numerically_qualified"],
            "attention_share_of_forward": 0.743,
            "projected_end_to_end_speedup": 1.0
            / (1.0 - 0.743 + 0.743 / best["speedup_vs_whole"]),
        }
        print(json.dumps(report["summary"], indent=2), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"saved report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
