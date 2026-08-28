#!/usr/bin/env python3
"""CPU-only contract checks for the opt-in H3 calibration feature path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("h3_group_cache_calibration_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.hidden <= 0 or args.samples <= 0:
        raise SystemExit("rows, hidden, and samples must be positive")
    calibration = load_module(
        REPO_ROOT / "custom_nodes/DualV100/h3_group_cache_calibration.py"
    )
    device = torch.device("cpu")
    indices = calibration.sampled_hidden_indices(
        args.hidden, max_samples=args.samples, device=device
    )
    if indices.dtype != torch.long or indices.ndim != 1:
        raise AssertionError("hidden sample indices have the wrong contract")
    if int(indices.numel()) != min(args.hidden, args.samples):
        raise AssertionError("hidden sample count is not bounded")
    if int(indices[0]) != 0 or int(indices[-1]) != args.hidden - 1:
        raise AssertionError("hidden sample must include both channel endpoints")
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise AssertionError("hidden samples must be unique")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    modulation = [
        torch.randn((args.rows, args.hidden), generator=generator, dtype=torch.float32)
        for _ in calibration.CONDITION_COMPONENTS
    ]
    signature = calibration.sampled_modulation_signature(modulation, indices).unsqueeze(0)
    if tuple(signature.shape) != (
        1,
        len(calibration.CONDITION_COMPONENTS),
        args.rows,
        int(indices.numel()),
    ):
        raise AssertionError(f"unexpected calibration signature shape: {signature.shape}")

    segments = tuple(
        (start, start + 1, start) for start in range(args.rows)
    )
    # Simulate ComfyUI renumbering AdaLN rows while retaining the same packed
    # token segments.  The helper must compare by segment position, not row id.
    permutation = torch.roll(torch.arange(args.rows), shifts=1)
    reordered = [value.index_select(0, permutation) for value in modulation]
    reordered_signature = calibration.sampled_modulation_signature(
        reordered, indices
    ).unsqueeze(0)
    inverse_permutation = torch.argsort(permutation)
    reordered_segments = tuple(
        (start, start + 1, int(inverse_permutation[start]))
        for start in range(args.rows)
    )
    identical = calibration.signature_difference(
        signature,
        reordered_signature,
        current_segments=segments,
        reference_segments=reordered_segments,
    )
    if not identical["available"] or any(
        abs(float(identical[key])) > 1e-12
        for key in ("gate_relative_l2", "affine_relative_l2", "all_relative_l2")
    ):
        raise AssertionError(f"row-id-invariant comparison failed: {identical}")

    perturbed = [value.clone() for value in modulation]
    perturbed[calibration.GATE_INDICES[0]][1].add_(0.75)
    perturbed_signature = calibration.sampled_modulation_signature(
        perturbed, indices
    ).unsqueeze(0)
    changed = calibration.signature_difference(
        perturbed_signature,
        signature,
        current_segments=segments,
        reference_segments=segments,
    )
    if not changed["available"] or not float(changed["gate_relative_l2"]) > 0.0:
        raise AssertionError(f"gate perturbation was not detected: {changed}")
    if calibration.selected_condition_error(changed, "gates") != changed[
        "gate_relative_l2"
    ]:
        raise AssertionError("gate metric selection is inconsistent")
    if calibration.selected_condition_error(changed, "all_adaln") != changed[
        "all_relative_l2"
    ]:
        raise AssertionError("all-AdaLN metric selection is inconsistent")
    if calibration.selected_condition_error(changed, "none") is not None:
        raise AssertionError("none condition metric should select no value")

    try:
        calibration.signature_difference(
            signature,
            reordered_signature,
            current_segments=segments,
            reference_segments=((0, args.rows, 0),),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched calibration segment count was accepted")

    report = {
        "ok": True,
        "device": str(device),
        "rows": args.rows,
        "hidden": args.hidden,
        "hidden_samples": int(indices.numel()),
        "signature_bytes": calibration.signature_bytes(signature),
        "identical_row_reordered": identical,
        "gate_perturbation": changed,
        "finite": all(
            math.isfinite(float(changed[key]))
            for key in ("gate_relative_l2", "affine_relative_l2", "all_relative_l2")
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
