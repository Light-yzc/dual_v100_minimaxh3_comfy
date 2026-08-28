#!/usr/bin/env python3
"""Benchmark the experimental bounded-score cuBLAS attention candidate.

No checkpoint is loaded.  This compares PyTorch efficient SDPA with the
candidate in ``custom_nodes/DualV100/h3_v100_attention_bmm.py`` at the actual
H3 local-head shape.  The candidate is intentionally kept out of the
production adapter until both speed and output gates pass.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_h3_attention import (
    backend_capabilities,
    make_h3_qkv,
    projected_sdpa,
    tensor_error,
    timed_cuda,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BMM_PATH = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_v100_attention_bmm.py"


def load_bmm_module():
    spec = importlib.util.spec_from_file_location("h3_v100_attention_bmm_bench", BMM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate from {BMM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", type=int, nargs="+", default=[2048, 37746])
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--softmax", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-free-mib", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise SystemExit(f"expected SM70, got sm_{props.major}{props.minor}")
    candidate_module = load_bmm_module()

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
        "heads": args.heads,
        "head_dim": args.head_dim,
        "softmax": args.softmax,
        "results": [],
    }

    for sequence in args.sequences:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        qkv_bytes = sequence * 3 * args.heads * args.head_dim * 2
        result = {
            "sequence": sequence,
            "free_before_mib": free_bytes / 2**20,
            "estimated_required_mib": (qkv_bytes + 2 * sequence * args.heads * args.head_dim * 2 + args.min_free_mib * 2**20) / 2**20,
        }
        if free_bytes < result["estimated_required_mib"] * 2**20:
            result["skipped"] = "VRAM safety guard"
            report["results"].append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            continue

        packed, q, k, v = make_h3_qkv(sequence, args.heads, args.head_dim, device, seed=2029)
        result["input_strides"] = {"q": list(q.stride()), "k": list(k.stride()), "v": list(v.stride())}
        result["sdpa_capabilities"] = backend_capabilities(q, k, v)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            reference, sdpa_ms, sdpa_peak = timed_cuda(
                lambda: projected_sdpa(q, k, v),
                warmup=args.warmup,
                repetitions=args.repetitions,
                device=device,
            )
        result["pytorch_efficient_sdpa"] = {
            "milliseconds": sdpa_ms,
            "peak_extra_mib": sdpa_peak / 2**20,
            "finite": bool(torch.isfinite(reference).all().item()),
        }

        candidates = []
        for block_m in args.blocks:
            try:
                output, milliseconds, peak = timed_cuda(
                    lambda bm=block_m: candidate_module.h3_attention_chunked_bmm(
                        q,
                        k,
                        v,
                        block_m=bm,
                        fp32_softmax=args.softmax == "fp32",
                    ).reshape(1, sequence, -1),
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    device=device,
                )
                errors = tensor_error(reference, output)
                qualified = (
                    errors["finite"]
                    and errors["cosine"] >= 0.999
                    and errors["max_abs"] <= 0.05
                )
                candidates.append({
                    "block_m": block_m,
                    "milliseconds": milliseconds,
                    "peak_extra_mib": peak / 2**20,
                    "speedup_vs_sdpa": sdpa_ms / milliseconds,
                    "error_vs_sdpa": errors,
                    "numerically_qualified": qualified,
                })
                del output
            except Exception as error:
                candidates.append({
                    "block_m": block_m,
                    "error": f"{type(error).__name__}: {error}",
                    "numerically_qualified": False,
                })
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        result["chunked_bmm_candidates"] = candidates
        report["results"].append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        del reference, packed, q, k, v
        torch.cuda.empty_cache()

    output = args.output or REPO_ROOT / "results" / f"h3_attention_bmm_sm70_{args.softmax}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved report: {output}", flush=True)


if __name__ == "__main__":
    main()
