#!/usr/bin/env python3
"""Compare two saved Qwen3-VL/ClipProj conditioning runs on CPU.

The encoder benchmarks run in separate protected processes.  This companion
keeps the comparison cheap and deterministic: it only reads the saved FP32
conditioning tensors and JSON reports, then writes a compact gate artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    # Conditioning contains an intentionally huge attention-sink coordinate.
    # Accumulating millions of products in FP32 can move cosine by ~1e-3 and
    # even round a nearly identical subset above 1.  Metrics are cheap on CPU,
    # so accumulate in FP64 and keep the saved conditioning itself unchanged.
    reference = reference.double()
    candidate = candidate.double()
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "finite": bool(torch.isfinite(candidate).all().item()),
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


def read_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q4-report", type=Path, required=True)
    parser.add_argument("--q4-tensor", type=Path, required=True)
    parser.add_argument("--int8-report", type=Path, required=True)
    parser.add_argument("--int8-tensor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    q4_report = read_report(args.q4_report)
    int8_report = read_report(args.int8_report)
    q4 = torch.load(args.q4_tensor, map_location="cpu", weights_only=True).float()
    int8 = torch.load(args.int8_tensor, map_location="cpu", weights_only=True).float()

    comparison: dict[str, object] = {
        "q4_report": str(args.q4_report),
        "int8_report": str(args.int8_report),
        "q4_tensor": str(args.q4_tensor),
        "int8_tensor": str(args.int8_tensor),
        "shape": list(q4.shape),
        "overall_q4_vs_int8": metrics(int8, q4),
        "finite": bool(torch.isfinite(q4).all().item() and torch.isfinite(int8).all().item()),
        "same_shape": bool(q4.shape == int8.shape),
        "tag_counts": {
            "q4": q4_report.get("conditioning_tag_counts"),
            "int8": int8_report.get("conditioning_tag_counts"),
        },
        "runtime": {
            "q4_load_seconds": q4_report.get("load_seconds"),
            "int8_load_seconds": int8_report.get("load_seconds"),
            "q4_projection_load_seconds": q4_report.get("projection_load_seconds"),
            "int8_projection_load_seconds": int8_report.get("projection_load_seconds"),
            "q4_cold_conditioning_ms": q4_report.get("cold_conditioning_ms"),
            "int8_cold_conditioning_ms": int8_report.get("cold_conditioning_ms"),
            "q4_warm_conditioning_ms": q4_report.get("warm_conditioning_ms"),
            "int8_warm_conditioning_ms": int8_report.get("warm_conditioning_ms"),
            "q4_warm_conditioning_mean_ms": q4_report.get("warm_conditioning_mean_ms"),
            "int8_warm_conditioning_mean_ms": int8_report.get("warm_conditioning_mean_ms"),
        },
        "memory": {
            "q4_rss_peak_mib": q4_report.get("rss_peak_mib"),
            "int8_rss_peak_mib": int8_report.get("rss_peak_mib"),
            "q4_cuda_after_load": q4_report.get("cuda_after_load"),
            "int8_cuda_after_load": int8_report.get("cuda_after_load"),
            "q4_cuda_after_conditioning": q4_report.get("cuda_after_conditioning"),
            "int8_cuda_after_conditioning": int8_report.get("cuda_after_conditioning"),
        },
        "mmap_audit": {
            "q4_model_map_hits": len(q4_report.get("model_map_hits", [])),
            "q4_projection_map_hits": len(q4_report.get("projection_map_hits", [])),
            "q4_payload_map_hits": len(q4_report.get("q4_payload_map_hits", [])),
            "int8_model_map_hits": len(int8_report.get("model_map_hits", [])),
            "int8_projection_map_hits": len(int8_report.get("projection_map_hits", [])),
        },
    }

    q4_tags = q4_report.get("conditioning_tags")
    int8_tags = int8_report.get("conditioning_tags")
    if isinstance(q4_tags, list) and isinstance(int8_tags, list) and q4.shape == int8.shape:
        if q4_tags == int8_tags and len(q4_tags) == q4.shape[-2]:
            tag_tensor = torch.tensor(q4_tags, dtype=torch.long)
            comparison["tagged_metrics"] = {}
            for name, value in (("text", 1), ("vision", 0)):
                mask = tag_tensor == value
                if not bool(mask.any().item()):
                    comparison["tagged_metrics"][name] = {
                        "tokens": 0,
                        "q4_vs_int8": None,
                    }
                    continue
                comparison["tagged_metrics"][name] = {
                    "tokens": int(mask.sum().item()),
                    "q4_vs_int8": metrics(int8[..., mask, :], q4[..., mask, :]),
                }
        else:
            comparison["tagged_metrics_error"] = "Q4 and INT8 token tags differ"
    else:
        comparison["tagged_metrics_error"] = "tag vectors are absent; rerun benchmark with tag reporting"

    comparison["gates"] = {
        "both_finite": comparison["finite"],
        "same_shape": comparison["same_shape"],
        "q4_no_payload_mmap": comparison["mmap_audit"]["q4_payload_map_hits"] == 0,
        "text_conditioning_gate": None,
        "image_conditioning_gate": None,
    }
    tagged = comparison.get("tagged_metrics", {})
    if isinstance(tagged, dict) and isinstance(tagged.get("text", {}).get("q4_vs_int8"), dict):
        text_metrics = tagged["text"]["q4_vs_int8"]
        comparison["gates"]["text_conditioning_gate"] = bool(
            text_metrics.get("finite")
            and text_metrics.get("cosine", 0.0) >= 0.995
            and text_metrics.get("relative_rms", float("inf")) <= 0.05
        )
    if isinstance(tagged, dict) and isinstance(tagged.get("vision", {}).get("q4_vs_int8"), dict):
        vision_metrics = tagged["vision"]["q4_vs_int8"]
        comparison["gates"]["image_conditioning_gate"] = bool(
            vision_metrics.get("finite")
            and vision_metrics.get("cosine", 0.0) >= 0.995
            and vision_metrics.get("relative_rms", float("inf")) <= 0.05
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
