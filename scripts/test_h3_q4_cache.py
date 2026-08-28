#!/usr/bin/env python3
"""CPU-only correctness and memory-geometry checks for H3 Q4_0 caches."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import gguf
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "custom_nodes/DualV100"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--rows", type=int, default=37)
    result.add_argument("--cols", type=int, default=5376)
    result.add_argument("--chunk-rows", type=int, default=7)
    result.add_argument("--seed", type=int, default=20260827)
    result.add_argument("--one-mp-sequence", type=int, default=37746)
    result.add_argument("--output", type=Path)
    return result


def dequantize(cache_module, value) -> torch.Tensor:
    chunks = [
        chunk
        for _start, _stop, chunk, _copied in value.iter_dequantized(
            "cpu", chunk_rows=5, dtype=torch.float32
        )
    ]
    return torch.cat(chunks, dim=0)


def main() -> None:
    args = parser().parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.cols % 32:
        raise SystemExit("rows must be positive and cols must be a positive multiple of 32")
    if args.chunk_rows <= 0 or args.one_mp_sequence <= 0:
        raise SystemExit("chunk rows and one-MP sequence must be positive")

    cache = load_module(
        "h3_q4_cache_test_module",
        REPO_ROOT / "custom_nodes/DualV100/h3_q4_cache.py",
    )
    if cache.normalize_q4_format("Q4_0") != cache.Q4_FORMAT:
        raise AssertionError("Q4_0 alias normalization failed")
    try:
        cache.normalize_q4_format("q5_0")
    except ValueError:
        pass
    else:
        raise AssertionError("non-Q4_0 cache format was accepted")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    source = torch.randn((args.rows, args.cols), generator=generator, dtype=torch.float32)
    source.mul_(0.37)
    source[0, :32] = 0.0

    encoded = cache.quantize_q4_0(
        source,
        policy="cpu",
        chunk_rows=args.chunk_rows,
        measure=True,
    )
    reference_raw = gguf.quantize(
        source.numpy().copy(), gguf.GGMLQuantizationType.Q4_0
    )
    byte_equal = bool(np.array_equal(encoded.raw.numpy(), reference_raw))
    if not byte_equal:
        mismatch = np.flatnonzero(encoded.raw.numpy().reshape(-1) != reference_raw.reshape(-1))
        raise AssertionError(f"Q4_0 byte mismatch at {mismatch[:8].tolist()}")

    decoded = dequantize(cache, encoded)
    reference_decoded = torch.from_numpy(
        gguf.dequantize(reference_raw, gguf.GGMLQuantizationType.Q4_0)
    )
    decode_max_abs = float((decoded - reference_decoded).abs().max().item())
    if decode_max_abs != 0.0:
        raise AssertionError(f"Q4_0 decode mismatch: {decode_max_abs}")

    large_source = source.mul(8_000_000.0)
    large_encoded = cache.quantize_q4_0(
        large_source,
        policy="cpu",
        chunk_rows=args.chunk_rows,
        measure=True,
    )
    if large_encoded.restore_scale_exponent <= 0:
        raise AssertionError("large Q4_0 cache input did not enable range scaling")
    scaled_large_source = large_source * math.ldexp(
        1.0, -large_encoded.restore_scale_exponent
    )
    large_reference_raw = gguf.quantize(
        scaled_large_source.numpy().copy(), gguf.GGMLQuantizationType.Q4_0
    )
    if not np.array_equal(large_encoded.raw.numpy(), large_reference_raw):
        raise AssertionError("scaled large-input Q4_0 bytes differ from gguf")
    large_decoded = dequantize(cache, large_encoded)
    large_reference_decoded = torch.from_numpy(
        gguf.dequantize(large_reference_raw, gguf.GGMLQuantizationType.Q4_0)
    ).mul_(math.ldexp(1.0, large_encoded.restore_scale_exponent))
    large_decode_max_abs = float(
        (large_decoded - large_reference_decoded).abs().max().item()
    )
    if large_decode_max_abs != 0.0 or not bool(torch.isfinite(large_decoded).all()):
        raise AssertionError(
            f"scaled large-input Q4_0 decode failed: max_abs={large_decode_max_abs}"
        )
    large_metric, large_metric_details = cache.relative_difference(
        large_encoded,
        large_encoded,
        metric="relative_l1",
        device="cpu",
        chunk_rows=args.chunk_rows,
    )
    if large_metric != 0.0 or not large_metric_details["finite"]:
        raise AssertionError(
            f"scaled identical Q4_0 metric must be finite zero, got {large_metric}"
        )

    base = torch.randn((args.rows, args.cols), generator=generator, dtype=torch.float32)
    candidate = base.clone()
    add_report = cache.add_q4_to_(
        candidate,
        encoded,
        chunk_rows=args.chunk_rows,
        measure=True,
    )
    add_max_abs = float((candidate - (base + reference_decoded)).abs().max().item())
    if add_max_abs != 0.0:
        raise AssertionError(f"chunked Q4 add mismatch: {add_max_abs}")

    metric_reports = {}
    for metric in ("relative_l1", "relative_l2", "cosine"):
        value, details = cache.relative_difference(
            encoded,
            encoded,
            metric=metric,
            device="cpu",
            chunk_rows=args.chunk_rows,
            measure=True,
        )
        if abs(value) > 1e-12:
            raise AssertionError(f"identical Q4 {metric} should be zero, got {value}")
        metric_reports[metric] = {"identical_error": value, **details}

    delta = source.clone()
    delta[:, ::97].add_(0.02)
    delta_q4 = cache.quantize_q4_0(
        delta,
        policy="cpu",
        chunk_rows=args.chunk_rows,
    )
    for metric in ("relative_l1", "relative_l2", "cosine"):
        value, details = cache.relative_difference(
            delta_q4,
            encoded,
            metric=metric,
            device="cpu",
            chunk_rows=args.chunk_rows,
        )
        if not math.isfinite(value) or value < 0.0:
            raise AssertionError(f"invalid Q4 {metric}: {value}")
        metric_reports[metric]["perturbed_error"] = value
        metric_reports[metric]["perturbed_details"] = details

    if args.rows < 3:
        raise SystemExit("rows must be at least 3 for the signature-range test")
    signature_cut_a = max(1, args.rows // 3)
    signature_cut_b = max(signature_cut_a + 1, (args.rows * 2) // 3)
    signature_cut_b = min(args.rows - 1, signature_cut_b)
    signature_ranges = (
        (0, signature_cut_a, 0),
        (signature_cut_a, signature_cut_b, 1),
        (signature_cut_b, args.rows, 2),
    )
    signature_tokens = min(args.rows, 23)
    signature, signature_metadata = cache.deterministic_input_signature(
        source,
        max_tokens=signature_tokens,
        hidden_samples=min(args.cols, 32),
        ranges=signature_ranges,
    )
    signature_delta = source.clone()
    signature_delta[signature_cut_b:, ::97].add_(0.5)
    signature_changed, signature_changed_metadata = cache.deterministic_input_signature(
        signature_delta,
        max_tokens=signature_tokens,
        hidden_samples=min(args.cols, 32),
        ranges=signature_ranges,
    )
    signature_reports = {}
    for metric in ("relative_l1", "relative_l2", "cosine"):
        identical, details = cache.signature_difference(
            signature,
            signature,
            metric=metric,
            current_metadata=signature_metadata,
            reference_metadata=signature_metadata,
        )
        if abs(identical) > 1e-12 or not details["finite"]:
            raise AssertionError(
                f"identical bounded signature {metric} should be finite zero, got {identical}"
            )
        weighted, weighted_details = cache.signature_difference(
            signature_changed,
            signature,
            metric=metric,
            current_metadata=signature_changed_metadata,
            reference_metadata=signature_metadata,
            aggregation="weighted",
        )
        maximum, maximum_details = cache.signature_difference(
            signature_changed,
            signature,
            metric=metric,
            current_metadata=signature_changed_metadata,
            reference_metadata=signature_metadata,
            aggregation="max_segment",
        )
        if not math.isfinite(weighted) or not math.isfinite(maximum):
            raise AssertionError(f"invalid bounded signature {metric} metric")
        if metric != "cosine" and maximum + 1e-12 < weighted:
            raise AssertionError(
                f"max-segment signature {metric} unexpectedly below weighted metric"
            )
        signature_reports[metric] = {
            "identical_error": identical,
            "weighted_perturbed_error": weighted,
            "max_segment_perturbed_error": maximum,
            "weighted_details": weighted_details,
            "max_segment_details": maximum_details,
        }
    invalid_metadata = dict(signature_metadata)
    invalid_metadata["layout_key"] = ()
    try:
        cache.signature_difference(
            signature_changed,
            signature,
            metric="relative_l2",
            current_metadata=signature_changed_metadata,
            reference_metadata=invalid_metadata,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("signature layout mismatch was not rejected")

    expected_ranges = [(8, 18), (18, 28), (28, 38), (38, 50)]
    actual_ranges = cache.GroupResidualCache.partition(8, 4, 50)
    if actual_ranges != expected_ranges:
        raise AssertionError(f"group partition mismatch: {actual_ranges}")

    group_cache = cache.GroupResidualCache("cpu")
    group_cache.configure(
        warm_blocks=8,
        num_groups=4,
        block_count=50,
        policy="cpu",
        shape=(args.rows, args.cols),
    )
    for entry in group_cache.entries:
        entry.previous_input = cache.quantize_q4_0(
            source, policy="cpu", chunk_rows=args.chunk_rows
        )
        entry.residual = cache.quantize_q4_0(
            delta - source, policy="cpu", chunk_rows=args.chunk_rows
        )
    q4_bytes_per_tensor = args.rows * (args.cols // 32) * 18
    expected_group_bytes = q4_bytes_per_tensor * 2 * 4
    if group_cache.bytes != expected_group_bytes:
        raise AssertionError(
            f"group cache byte mismatch: {group_cache.bytes} != {expected_group_bytes}"
        )
    group_cache.configure(
        warm_blocks=8,
        num_groups=4,
        block_count=50,
        policy="cpu",
        shape=(args.rows, args.cols),
        feature_mode="signature",
        signature_max_tokens=signature_tokens,
        signature_hidden_samples=min(args.cols, 32),
    )
    if group_cache.entries and group_cache.ready:
        raise AssertionError("changing feature mode must invalidate old Q4 state")

    signature_group_cache = cache.GroupResidualCache("cpu")
    signature_group_cache.configure(
        warm_blocks=8,
        num_groups=4,
        block_count=50,
        policy="cpu",
        shape=(args.rows, args.cols),
        feature_mode="signature",
        signature_max_tokens=signature_tokens,
        signature_hidden_samples=min(args.cols, 32),
    )
    for entry in signature_group_cache.entries:
        entry.input_signature = signature.clone()
        entry.input_signature_metadata = dict(signature_metadata)
        entry.residual = cache.quantize_q4_0(
            delta - source, policy="cpu", chunk_rows=args.chunk_rows
        )
    expected_signature_group_bytes = q4_bytes_per_tensor * 4 + signature.numel() * signature.element_size() * 4
    if signature_group_cache.bytes != expected_signature_group_bytes:
        raise AssertionError(
            "signature group cache byte mismatch: "
            f"{signature_group_cache.bytes} != {expected_signature_group_bytes}"
        )
    if not signature_group_cache.ready:
        raise AssertionError("signature group cache should be ready with residuals and signatures")
    if signature_group_cache.summary()["feature_mode"] != "signature":
        raise AssertionError("signature group cache did not retain its feature mode")

    one_mp_fp32_bytes = args.one_mp_sequence * args.cols * 4
    one_mp_q4_bytes = args.one_mp_sequence * (args.cols // 32) * 18
    one_mp_signature_bytes_per_group = (
        min(args.one_mp_sequence, cache.DEFAULT_SIGNATURE_MAX_TOKENS)
        * min(args.cols, cache.DEFAULT_SIGNATURE_HIDDEN_SAMPLES)
        * 4
    )
    report = {
        "format": "ggml_q4_0",
        "default_group_threshold": cache.DEFAULT_GROUP_THRESHOLD,
        "byte_compatible_with_gguf": byte_equal,
        "decode_max_abs_vs_gguf": decode_max_abs,
        "large_range": {
            "source_max_abs": float(large_source.abs().max().item()),
            "restore_scale_exponent": large_encoded.restore_scale_exponent,
            "raw_matches_scaled_gguf": True,
            "decode_max_abs_vs_scaled_gguf": large_decode_max_abs,
            "identical_relative_l1": large_metric,
            "finite": bool(torch.isfinite(large_decoded).all()),
            "quantize_report": large_encoded.quantize_report,
        },
        "chunked_add_max_abs": add_max_abs,
        "shape": [args.rows, args.cols],
        "raw_bytes": encoded.bytes,
        "expected_raw_bytes": q4_bytes_per_tensor,
        "source_fp32_bytes": source.numel() * source.element_size(),
        "compression_ratio_vs_fp32": (
            source.numel() * source.element_size() / encoded.bytes
        ),
        "metrics": metric_reports,
        "signature": {
            "metadata": signature_metadata,
            "metrics": signature_reports,
        },
        "partition": [list(item) for item in actual_ranges],
        "group_cache": {
            "persistent_q4_tensors": 8,
            "bytes": group_cache.bytes,
            "expected_bytes": expected_group_bytes,
        },
        "signature_group_cache": {
            "persistent_q4_residuals": 4,
            "persistent_input_signatures": 4,
            "bytes": signature_group_cache.bytes,
            "expected_bytes": expected_signature_group_bytes,
        },
        "one_mp_estimate_per_rank": {
            "sequence": args.one_mp_sequence,
            "fp32_tensor_bytes": one_mp_fp32_bytes,
            "q4_tensor_bytes": one_mp_q4_bytes,
            "whole_tail_one_residual_bytes": one_mp_q4_bytes,
            "group_four_previous_plus_four_residual_bytes": one_mp_q4_bytes * 8,
            "signature_bytes_per_group": one_mp_signature_bytes_per_group,
            "group_four_signature_plus_four_residual_bytes": (
                one_mp_q4_bytes * 4 + one_mp_signature_bytes_per_group * 4
            ),
            "compression_ratio_per_tensor": one_mp_fp32_bytes / one_mp_q4_bytes,
        },
        "quantize_report": encoded.quantize_report,
        "add_report": add_report,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
