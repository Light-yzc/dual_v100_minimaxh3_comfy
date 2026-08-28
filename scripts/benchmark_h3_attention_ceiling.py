#!/usr/bin/env python3
"""Measure the SM70 attention ceiling for the H3 DiT local-head shape.

No checkpoint is loaded and no ComfyUI service is touched.  The question this
answers is narrow: the production ``attention_sdpa`` stage runs at roughly
29 TFLOPS while the ordinary GEMMs in the same forward reach roughly
87 TFLOPS, so is the gap a fixed overhead, a tensor-core issue, or the
asymptotic efficiency of the CUTLASS FMHA kernel on Volta?

Three measurements, all on one device:

1.  ``gemm_peak``     -- large square cuBLAS FP16 GEMM, the practical TC ceiling.
2.  ``qk_bmm``        -- batched QK^T at the real head shape, which is what any
                         cuBLAS-based attention would have to pay, plus the HBM
                         traffic implied by materialising the scores.
3.  ``sdpa_scaling``  -- efficient SDPA swept over sequence length so the
                         effective TFLOPS curve is visible rather than inferred
                         from a single point.

The script allocates well under 2 GiB and never maps a model payload.
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
HBM_BYTES_PER_SECOND = 900.0e9


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


def gemm_peak(device: torch.device, size: int, warmup: int, repetitions: int) -> dict:
    """Large square FP16 GEMM: the demonstrated cuBLAS tensor-core ceiling."""
    left = torch.randn(size, size, device=device, dtype=torch.float16)
    right = torch.randn(size, size, device=device, dtype=torch.float16)
    out = torch.empty(size, size, device=device, dtype=torch.float16)
    elapsed = timed(lambda: torch.mm(left, right, out=out), warmup, repetitions)
    flop = 2.0 * size * size * size
    del left, right, out
    torch.cuda.empty_cache()
    return {
        "size": size,
        "ms": elapsed,
        "tflops": flop / (elapsed * 1e-3) / 1e12,
    }


def qk_bmm(
    device: torch.device,
    sequence: int,
    rows: int,
    warmup: int,
    repetitions: int,
) -> dict:
    """Batched QK^T for one query tile at the real head shape.

    ``rows`` query rows against the full ``sequence`` keys, batched over the
    local heads.  Extrapolating to the whole sequence gives the compute cost a
    cuBLAS attention would pay, and the score tensor size gives the HBM traffic
    it would pay on top -- the term the fused FMHA kernel avoids entirely.
    """
    query = torch.randn(LOCAL_HEADS, rows, HEAD_DIM, device=device, dtype=torch.float16)
    keys = torch.randn(LOCAL_HEADS, HEAD_DIM, sequence, device=device, dtype=torch.float16)
    scores = torch.empty(LOCAL_HEADS, rows, sequence, device=device, dtype=torch.float16)
    elapsed = timed(lambda: torch.bmm(query, keys, out=scores), warmup, repetitions)
    tile_flop = 2.0 * LOCAL_HEADS * rows * sequence * HEAD_DIM
    tiles = sequence / rows
    # QK^T and PV are the same shape, hence the factor of two for full attention.
    full_flop = 2.0 * tile_flop * tiles
    score_bytes = 2.0 * LOCAL_HEADS * sequence * sequence
    del query, keys, scores
    torch.cuda.empty_cache()
    return {
        "sequence": sequence,
        "rows": rows,
        "ms": elapsed,
        "tflops": tile_flop / (elapsed * 1e-3) / 1e12,
        "projected_full_attention_flop": full_flop,
        "projected_compute_ms": (full_flop / (tile_flop / (elapsed * 1e-3))) * 1e3,
        "score_write_read_bytes": 2.0 * score_bytes,
        "score_traffic_floor_ms": (2.0 * score_bytes / HBM_BYTES_PER_SECOND) * 1e3,
    }


def sdpa_scaling(
    device: torch.device,
    sequence: int,
    warmup: int,
    repetitions: int,
) -> dict:
    """Efficient SDPA at the production layout, one sequence length."""
    shape = (1, LOCAL_HEADS, sequence, HEAD_DIM)
    query = torch.randn(shape, device=device, dtype=torch.float16)
    keys = torch.randn(shape, device=device, dtype=torch.float16)
    values = torch.randn(shape, device=device, dtype=torch.float16)

    def run() -> None:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            functional.scaled_dot_product_attention(
                query, keys, values, attn_mask=None, dropout_p=0.0, is_causal=False
            )

    torch.cuda.reset_peak_memory_stats(device)
    elapsed = timed(run, warmup, repetitions)
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    flop = 4.0 * LOCAL_HEADS * sequence * sequence * HEAD_DIM
    del query, keys, values
    torch.cuda.empty_cache()
    return {
        "sequence": sequence,
        "ms": elapsed,
        "tflops": flop / (elapsed * 1e-3) / 1e12,
        "peak_allocated_mib": peak_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gemm-sizes", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument(
        "--sdpa-sequences",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384, 37746],
    )
    parser.add_argument("--bmm-rows", type=int, default=512)
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
        "heads": LOCAL_HEADS,
        "head_dim": HEAD_DIM,
        "sdpa_backends": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "gemm_peak": [],
        "qk_bmm": [],
        "sdpa_scaling": [],
    }

    for size in args.gemm_sizes:
        entry = gemm_peak(device, size, args.warmup, args.repetitions)
        report["gemm_peak"].append(entry)
        print(f"gemm {size}^3: {entry['ms']:.3f} ms  {entry['tflops']:.1f} TFLOPS", flush=True)

    for sequence in args.sdpa_sequences:
        entry = sdpa_scaling(device, sequence, args.warmup, args.repetitions)
        report["sdpa_scaling"].append(entry)
        print(
            f"sdpa  S={sequence}: {entry['ms']:.3f} ms  {entry['tflops']:.1f} TFLOPS"
            f"  peak {entry['peak_allocated_mib']:.0f} MiB",
            flush=True,
        )

    for sequence in args.sdpa_sequences:
        entry = qk_bmm(device, sequence, args.bmm_rows, args.warmup, args.repetitions)
        report["qk_bmm"].append(entry)
        print(
            f"qk    S={sequence}: {entry['ms']:.3f} ms  {entry['tflops']:.1f} TFLOPS"
            f"  projected compute {entry['projected_compute_ms']:.0f} ms"
            f"  score traffic floor {entry['score_traffic_floor_ms']:.0f} ms",
            flush=True,
        )

    peak_tflops = max(entry["tflops"] for entry in report["gemm_peak"])
    target = next(
        (entry for entry in report["sdpa_scaling"] if entry["sequence"] == 37746), None
    )
    if target is not None:
        report["summary"] = {
            "cublas_fp16_peak_tflops": peak_tflops,
            "sdpa_tflops_at_37746": target["tflops"],
            "sdpa_fraction_of_cublas_peak": target["tflops"] / peak_tflops,
            "sdpa_ms_at_37746": target["ms"],
            "ms_if_sdpa_matched_cublas_peak": target["ms"] * target["tflops"] / peak_tflops,
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
