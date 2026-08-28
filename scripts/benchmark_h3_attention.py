#!/usr/bin/env python3
"""Numerical and performance benchmark for H3's SM70 attention path.

The script loads no model files.  It allocates only synthetic Q/K/V tensors on
one selected GPU, checks free VRAM before every sequence length, and writes a
JSON report so failed/slow experiments are retained rather than hand-waved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_v100_attention.py"


def load_kernel_module():
    spec = importlib.util.spec_from_file_location("h3_v100_attention_bench", KERNEL_PATH)
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


def timed_cuda(function, *, warmup: int, repetitions: int, device: torch.device):
    result = None
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        result = function()
    end.record()
    end.synchronize()
    milliseconds = start.elapsed_time(end) / repetitions
    peak_extra = max(0, torch.cuda.max_memory_allocated(device) - baseline)
    return result, milliseconds, peak_extra


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_fp32 = reference.float()
    candidate_fp32 = candidate.float()
    difference = candidate_fp32 - reference_fp32
    ref_norm = torch.linalg.vector_norm(reference_fp32)
    cand_norm = torch.linalg.vector_norm(candidate_fp32)
    cosine = torch.sum(reference_fp32 * candidate_fp32) / (ref_norm * cand_norm)
    result = {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rms": float(torch.sqrt(torch.mean(difference.square())).item()),
        "cosine": float(cosine.item()),
        "finite": bool(torch.isfinite(candidate_fp32).all().item()),
    }
    del reference_fp32, candidate_fp32, difference
    return result


def make_h3_qkv(
    sequence: int,
    heads: int,
    head_dim: int,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    # Match H3's fused-qkv views: sequence stride is 3*H*D rather than D.
    packed = torch.randn(
        (1, sequence, 3 * heads * head_dim),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    inner = heads * head_dim
    q = packed[:, :, :inner].view(1, sequence, heads, head_dim).transpose(1, 2)
    k = packed[:, :, inner : 2 * inner].view(1, sequence, heads, head_dim).transpose(1, 2)
    v = packed[:, :, 2 * inner :].view(1, sequence, heads, head_dim).transpose(1, 2)
    return packed, q, k, v


def raw_sdpa(q, k, v):
    return functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )


def pack_attention_output(output):
    # H3's row-parallel out projection consumes [B, S, H*D].  The SDPA
    # backend returns [B, H, S, D], so this is a real materialization rather
    # than a metadata-only reshape.  Keep it separate in the benchmark to
    # avoid attributing a bandwidth-bound layout copy to the SDPA kernel.
    return output.transpose(1, 2).contiguous().view(
        output.shape[0], output.shape[2], -1
    )


def projected_sdpa(q, k, v):
    return pack_attention_output(raw_sdpa(q, k, v))


def compact_qkv(q, k, v, layout: str):
    if layout == "bhsd":
        return q.contiguous(), k.contiguous(), v.contiguous()
    if layout == "bshd":
        # Preserve H3's token-major physical order while removing the gaps
        # left by the fused [Q|K|V] projection.  Transposing back gives SDPA
        # the same logical [B,H,S,D] shape with a compact BSHD backing store.
        return tuple(
            tensor.transpose(1, 2).contiguous().transpose(1, 2)
            for tensor in (q, k, v)
        )
    if layout == "stack_bhsd":
        # One packed allocation with three standard BHSD planes.  The views
        # retain the packed owner, so no extra copies are made after stack.
        compact = torch.stack((q[0], k[0], v[0]), dim=0)
        return tuple(plane.unsqueeze(0) for plane in compact.unbind(0))
    if layout.endswith("_bhsd"):
        selected = set(layout.removesuffix("_bhsd"))
        if not selected or not selected <= {"q", "k", "v"}:
            raise ValueError(f"unsupported partial compact layout: {layout}")
        return tuple(
            tensor.contiguous() if name in selected else tensor
            for name, tensor in zip(("q", "k", "v"), (q, k, v), strict=True)
        )
    raise ValueError(f"unsupported compact layout: {layout}")


def compact_projected_sdpa(q, k, v, layout: str):
    compact = compact_qkv(q, k, v, layout)
    return projected_sdpa(*compact)


def backend_capabilities(q, k, v) -> dict[str, bool]:
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
    return {
        "flash": bool(torch.backends.cuda.can_use_flash_attention(params)),
        "cudnn": bool(torch.backends.cuda.can_use_cudnn_attention(params)),
        "efficient": bool(torch.backends.cuda.can_use_efficient_attention(params)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", type=int, nargs="+", default=[128, 2048])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument(
        "--configs",
        type=parse_config,
        nargs="+",
        default=[parse_config("32x32x4x2")],
    )
    parser.add_argument(
        "--skip-candidates",
        action="store_true",
        help="profile the PyTorch SDPA/layout baseline without running Triton candidates",
    )
    parser.add_argument(
        "--benchmark-compact-qkv",
        action="store_true",
        help="compare compact BHSD/BSHD QKV layouts, including their copy cost",
    )
    parser.add_argument(
        "--compact-layouts",
        nargs="+",
        default=["bshd", "bhsd", "stack_bhsd"],
        help="compact layouts to benchmark (also supports q/k/v/qk/qv/kv_bhsd)",
    )
    parser.add_argument("--min-free-mib", type=int, default=1024)
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
        "layout": "H3 fused-QKV strided views",
        "results": [],
    }

    for sequence in args.sequences:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        qkv_bytes = sequence * 3 * args.heads * args.head_dim * 2
        output_bytes = sequence * args.heads * args.head_dim * 2
        # SDPA workspace and allocator reserve vary by torch build.  Refuse a
        # run unless there is room for QKV, two outputs, 1 GiB workspace, and
        # the caller-requested post-run reserve.
        required = qkv_bytes + 2 * output_bytes + 1024 * 2**20 + args.min_free_mib * 2**20
        sequence_result = {
            "sequence": sequence,
            "free_before_mib": free_bytes / 2**20,
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

        packed, q, k, v = make_h3_qkv(
            sequence, args.heads, args.head_dim, device, args.seed
        )
        sequence_result["input_strides"] = {
            "q": list(q.stride()),
            "k": list(k.stride()),
            "v": list(v.stride()),
        }
        sequence_result["sdpa_capabilities"] = backend_capabilities(q, k, v)

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            raw_reference, raw_sdpa_ms, raw_sdpa_peak = timed_cuda(
                lambda: raw_sdpa(q, k, v),
                warmup=args.warmup,
                repetitions=args.repetitions,
                device=device,
            )
            packed_reference, pack_ms, pack_peak = timed_cuda(
                lambda: pack_attention_output(raw_reference),
                warmup=args.warmup,
                repetitions=args.repetitions,
                device=device,
            )
            reference, sdpa_ms, sdpa_peak = timed_cuda(
                lambda: projected_sdpa(q, k, v),
                warmup=args.warmup,
                repetitions=args.repetitions,
                device=device,
            )
        sequence_result["pytorch_efficient_sdpa"] = {
            "milliseconds": sdpa_ms,
            "peak_extra_mib": sdpa_peak / 2**20,
            "finite": bool(torch.isfinite(reference).all().item()),
            "kernel_only_milliseconds": raw_sdpa_ms,
            "kernel_only_peak_extra_mib": raw_sdpa_peak / 2**20,
            "output_pack_milliseconds": pack_ms,
            "output_pack_peak_extra_mib": pack_peak / 2**20,
            "split_sum_milliseconds": raw_sdpa_ms + pack_ms,
            "output_pack_fraction": pack_ms / sdpa_ms,
            "split_output_error": tensor_error(reference, packed_reference),
            "raw_output_stride": list(raw_reference.stride()),
            "packed_output_stride": list(packed_reference.stride()),
            "output_pack_aliases_raw": (
                packed_reference.untyped_storage().data_ptr()
                == raw_reference.untyped_storage().data_ptr()
            ),
        }

        compact_results = []
        if args.benchmark_compact_qkv:
            for layout in args.compact_layouts:
                compact, copy_ms, copy_peak = timed_cuda(
                    lambda selected=layout: compact_qkv(q, k, v, selected),
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    device=device,
                )
                compact_q, compact_k, compact_v = compact
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    compact_raw, compact_kernel_ms, compact_kernel_peak = timed_cuda(
                        lambda: raw_sdpa(compact_q, compact_k, compact_v),
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        device=device,
                    )
                    compact_e2e, compact_e2e_ms, compact_e2e_peak = timed_cuda(
                        lambda selected=layout: compact_projected_sdpa(
                            q, k, v, selected
                        ),
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        device=device,
                    )
                compact_packed = pack_attention_output(compact_raw)
                compact_results.append(
                    {
                        "layout": layout,
                        "q_stride": list(compact_q.stride()),
                        "k_stride": list(compact_k.stride()),
                        "v_stride": list(compact_v.stride()),
                        "copy_milliseconds": copy_ms,
                        "copy_peak_extra_mib": copy_peak / 2**20,
                        "kernel_only_milliseconds": compact_kernel_ms,
                        "kernel_only_peak_extra_mib": compact_kernel_peak / 2**20,
                        "kernel_speedup_vs_strided": raw_sdpa_ms / compact_kernel_ms,
                        "copy_plus_kernel_milliseconds": copy_ms + compact_kernel_ms,
                        "split_speedup_vs_strided": sdpa_ms
                        / (copy_ms + compact_kernel_ms),
                        "end_to_end_milliseconds": compact_e2e_ms,
                        "end_to_end_peak_extra_mib": compact_e2e_peak / 2**20,
                        "end_to_end_speedup_vs_strided": sdpa_ms / compact_e2e_ms,
                        "error_vs_strided": tensor_error(reference, compact_packed),
                        "end_to_end_error_vs_strided": tensor_error(
                            reference, compact_e2e
                        ),
                    }
                )
                del (
                    compact_e2e,
                    compact_packed,
                    compact_raw,
                    compact_q,
                    compact_k,
                    compact_v,
                    compact,
                )
        sequence_result["compact_qkv"] = compact_results

        candidates = []
        for block_m, block_n, warps, stages in (() if args.skip_candidates else args.configs):
            label = f"{block_m}x{block_n}-w{warps}-s{stages}"
            try:
                candidate, kernel_ms, kernel_peak = timed_cuda(
                    lambda bm=block_m, bn=block_n, nw=warps, ns=stages: kernel.h3_attention_sm70(
                        q,
                        k,
                        v,
                        block_m=bm,
                        block_n=bn,
                        num_warps=nw,
                        num_stages=ns,
                    ).reshape(1, sequence, -1),
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    device=device,
                )
            except Exception as error:
                candidates.append(
                    {
                        "config": label,
                        "error": f"{type(error).__name__}: {error}",
                        "numerically_qualified": False,
                    }
                )
                print(f"{label}: failed: {type(error).__name__}: {error}", flush=True)
                continue
            errors = tensor_error(reference, candidate)
            qualified = (
                errors["finite"]
                and errors["cosine"] >= 0.999
                and errors["max_abs"] <= 0.05
            )
            candidates.append(
                {
                    "config": label,
                    "milliseconds": kernel_ms,
                    "peak_extra_mib": kernel_peak / 2**20,
                    "speedup_vs_sdpa": sdpa_ms / kernel_ms,
                    "error_vs_sdpa": errors,
                    "numerically_qualified": qualified,
                }
            )
            del candidate
        sequence_result["sm70_candidates"] = candidates
        report["results"].append(sequence_result)
        del reference, packed_reference, raw_reference, packed, q, k, v
        torch.cuda.empty_cache()
        print(json.dumps(sequence_result, ensure_ascii=False, indent=2), flush=True)

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = REPO_ROOT / "results" / f"h3_attention_sm70_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"saved report: {output}", flush=True)


if __name__ == "__main__":
    main()
