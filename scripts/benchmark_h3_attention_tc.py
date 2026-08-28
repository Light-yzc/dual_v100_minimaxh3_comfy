#!/usr/bin/env python3
"""Benchmark the experimental chunked-FP16 SM70 attention kernel.

This is intentionally separate from the production attention benchmark.  It
uses synthetic tensors only, defaults to a conservative free-VRAM guard, and
does not install a ComfyUI hook.  The candidate is only interesting if its
long-sequence output and timing beat the Q-only PyTorch efficient-SDPA path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_h3_attention import make_h3_qkv, tensor_error, timed_cuda


REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_v100_attention_tc.py"


def load_kernel_module():
    spec = importlib.util.spec_from_file_location("h3_v100_attention_tc_bench", KERNEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import kernel from {KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_config(value: str) -> tuple[int, int, int, int]:
    fields = value.lower().replace("x", ",").split(",")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError("config must be BLOCK_MxBLOCK_NxWARPSxSTAGES")
    return tuple(int(field) for field in fields)  # type: ignore[return-value]


def make_layout(q, k, v, layout: str):
    if layout == "strided":
        return q, k, v
    if layout == "q":
        return q.contiguous(), k, v
    if layout == "all":
        return q.contiguous(), k.contiguous(), v.contiguous()
    raise ValueError(f"unsupported layout {layout!r}")


def raw_sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )


def pack_output(output):
    return output.transpose(1, 2).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", type=int, nargs="+", default=[128, 2048, 8192])
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--configs",
        type=parse_config,
        nargs="+",
        default=[
            parse_config("16x32x4x1"),
            parse_config("16x64x4x1"),
            parse_config("32x64x4x1"),
            parse_config("16x128x8x1"),
        ],
    )
    parser.add_argument("--layout", choices=("strided", "q", "all"), default="q")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise SystemExit(f"expected SM70, got sm_{props.major}{props.minor}")
    kernel = load_kernel_module()

    report = {
        "created_unix": time.time(),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": getattr(kernel.triton, "__version__", None),
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "heads": args.heads,
        "head_dim": args.head_dim,
        "layout": args.layout,
        "qk_variants": ["fp32_chunk", "fp16_chunk"],
        "results": [],
    }

    for sequence in args.sequences:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        qkv_bytes = sequence * 3 * args.heads * args.head_dim * 2
        output_bytes = sequence * args.heads * args.head_dim * 2
        # The candidate owns one output and the reference owns one output.  A
        # generous 1 GiB workspace allowance keeps this synthetic test from
        # competing with the resident H3 service if it was started by mistake.
        required = qkv_bytes + 2 * output_bytes + 1024 * 2**20 + args.min_free_mib * 2**20
        sequence_result = {
            "sequence": sequence,
            "free_before_mib": free_bytes / 2**20,
            "total_mib": total_bytes / 2**20,
            "estimated_required_mib": required / 2**20,
        }
        if free_bytes < required:
            sequence_result["skipped"] = "VRAM safety guard"
            report["results"].append(sequence_result)
            print(
                f"S={sequence}: skipped; free={free_bytes / 2**20:.0f} MiB, "
                f"guard requires={required / 2**20:.0f} MiB",
                flush=True,
            )
            continue

        packed, q0, k0, v0 = make_h3_qkv(
            sequence, args.heads, args.head_dim, device, seed=2029
        )
        q, k, v = make_layout(q0, k0, v0, args.layout)
        sequence_result["input_strides"] = {
            "q": list(q.stride()),
            "k": list(k.stride()),
            "v": list(v.stride()),
        }

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            reference, reference_ms, reference_peak = timed_cuda(
                lambda: raw_sdpa(q, k, v),
                warmup=args.warmup,
                repetitions=args.repetitions,
                device=device,
            )
        reference_packed = pack_output(reference)
        sequence_result["pytorch_efficient_sdpa"] = {
            "milliseconds": reference_ms,
            "peak_extra_mib": reference_peak / 2**20,
            "finite": bool(torch.isfinite(reference).all().item()),
        }

        candidates = []
        for block_m, block_n, warps, stages in args.configs:
            label = f"{block_m}x{block_n}-w{warps}-s{stages}"
            for qk_fp16 in (False, True):
                variant = "fp16_chunk" if qk_fp16 else "fp32_chunk"
                try:
                    candidate, kernel_ms, kernel_peak = timed_cuda(
                        lambda bm=block_m, bn=block_n, nw=warps, ns=stages, fp16=qk_fp16: kernel.h3_attention_sm70_tc(
                            q,
                            k,
                            v,
                            block_m=bm,
                            block_n=bn,
                            num_warps=nw,
                            num_stages=ns,
                            qk_fp16=fp16,
                        ),
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        device=device,
                    )
                except Exception as error:
                    candidates.append(
                        {
                            "config": label,
                            "variant": variant,
                            "error": f"{type(error).__name__}: {error}",
                            "numerically_qualified": False,
                        }
                    )
                    print(f"{label} {variant}: failed: {type(error).__name__}: {error}", flush=True)
                    continue

                errors = tensor_error(reference_packed, candidate)
                qualified = (
                    errors["finite"]
                    and errors["cosine"] >= 0.999
                    and errors["max_abs"] <= 0.05
                )
                candidates.append(
                    {
                        "config": label,
                        "variant": variant,
                        "milliseconds": kernel_ms,
                        "peak_extra_mib": kernel_peak / 2**20,
                        "speedup_vs_sdpa": reference_ms / kernel_ms,
                        "error_vs_sdpa": errors,
                        "numerically_qualified": qualified,
                    }
                )
                del candidate
        sequence_result["sm70_tc_candidates"] = candidates
        report["results"].append(sequence_result)
        print(json.dumps(sequence_result, ensure_ascii=False, indent=2), flush=True)
        del reference_packed, reference, packed, q0, k0, v0, q, k, v
        torch.cuda.empty_cache()

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = REPO_ROOT / "results" / f"h3_attention_sm70_tc_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"saved report: {output}", flush=True)


if __name__ == "__main__":
    main()
