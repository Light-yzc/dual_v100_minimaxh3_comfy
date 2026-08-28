#!/usr/bin/env python3
"""Fit a conservative, group-specific H3 cache-risk policy offline.

The script consumes forward reports emitted by the opt-in calibration mode.
It never edits a workflow and the runtime does not auto-load its output.  A
policy becomes a candidate only after its held-out error/coverage is reviewed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - deployment dependency guard
    raise SystemExit("fit_h3_group_cache_policy.py requires numpy") from error


FEATURE_NAMES = {
    "input": "input_feature_error",
    "gate": "adaln_gate_relative_l2",
    "affine": "adaln_affine_relative_l2",
    "all_adaln": "adaln_all_relative_l2",
    "q_floor": "q4_residual_floor_relative_l2",
    "age": "cache_age",
    "sigma": "sigma_delta",
}
DEFAULT_FEATURES = ("input", "gate", "affine", "q_floor", "age")


@dataclass(frozen=True)
class CalibrationRecord:
    group_id: int
    features: dict[str, float]
    target: float
    source: str
    sample_id: str
    decision: str | None
    target_kind: str | None
    feature_mode: str | None
    signature_aggregation: str | None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _target_from_row(row: dict[str, Any], target_name: str) -> tuple[float, str] | None:
    calibration = row.get("calibration")
    target = calibration.get("target") if isinstance(calibration, dict) else None
    if not isinstance(target, dict):
        ground_truth = row.get("ground_truth")
        if isinstance(ground_truth, dict):
            target = {
                "kind": "legacy_ground_truth",
                target_name: ground_truth.get(target_name),
            }
    if not isinstance(target, dict):
        return None
    value = _finite(target.get(target_name))
    if value is None or value < 0.0:
        return None
    return value, str(target.get("kind", "unknown"))


def _features_from_row(
    row: dict[str, Any],
    parent: dict[str, Any],
) -> dict[str, float]:
    calibration = row.get("calibration")
    calibration = calibration if isinstance(calibration, dict) else {}
    condition = calibration.get("condition")
    condition = condition if isinstance(condition, dict) else {}
    values: dict[str, Any] = {
        "input": calibration.get("input_feature_error", row.get("feature_error")),
        "gate": condition.get("gate_relative_l2"),
        "affine": condition.get("affine_relative_l2"),
        "all_adaln": condition.get("all_relative_l2"),
        "q_floor": calibration.get(
            "residual_q4_floor_relative_l2",
            (
                (row.get("residual_quantization_error") or {}).get("relative_l2")
                if isinstance(row.get("residual_quantization_error"), dict)
                else None
            ),
        ),
        "age": calibration.get(
            "cache_age", row.get("cache_count_before", parent.get("step", 0))
        ),
        "sigma": calibration.get("sigma_delta", parent.get("sigma_delta")),
    }
    result: dict[str, float] = {}
    for name, value in values.items():
        number = _finite(value)
        if number is not None and number >= 0.0:
            result[name] = number
    return result


def _feature_provenance(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return the feature representation so incompatible thresholds do not mix."""

    calibration = row.get("calibration")
    calibration = calibration if isinstance(calibration, dict) else {}
    mode = calibration.get("input_feature_mode", row.get("feature_mode"))
    aggregation = calibration.get(
        "signature_aggregation", row.get("signature_aggregation")
    )
    normalized_mode = None if mode is None else str(mode).strip().lower()
    normalized_aggregation = (
        None if aggregation is None else str(aggregation).strip().lower()
    )
    return normalized_mode, normalized_aggregation


def _rows_from_object(
    value: Any,
    source: str,
    inherited: dict[str, Any] | None = None,
) -> Iterable[tuple[dict[str, Any], dict[str, Any], str]]:
    """Yield ``(group row, parent, stable sample id)`` from report variants."""

    inherited = dict(inherited or {})
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _rows_from_object(item, source, {**inherited, "index": index})
        return
    if not isinstance(value, dict):
        return

    if isinstance(value.get("rank0"), dict):
        rank0 = value["rank0"]
        group = rank0.get("group_cache")
        if isinstance(group, dict):
            parent = {**value, **group}
            sample_id = ":".join(
                [
                    source,
                    str(group.get("generation_id", value.get("forward_index", "?"))),
                    str(group.get("step", value.get("forward_index", "?"))),
                ]
            )
            for row in group.get("groups", []):
                if isinstance(row, dict):
                    yield row, parent, sample_id
        return

    # A compact oracle summary may contain group rows directly.
    if isinstance(value.get("groups"), list):
        parent = {**inherited, **value}
        sample_id = ":".join(
            [
                source,
                str(value.get("generation_id", value.get("forward_index", "?"))),
                str(value.get("step", value.get("forward_index", "?"))),
            ]
        )
        for row in value["groups"]:
            if isinstance(row, dict):
                yield row, parent, sample_id
        return

    # Permit a directory-level ``records``/``entries`` wrapper without
    # interpreting arbitrary JSON metadata as a calibration row.
    for key in ("records", "entries", "forwards"):
        if isinstance(value.get(key), (list, dict)):
            yield from _rows_from_object(value[key], source, inherited)
            return


def read_records(paths: Sequence[Path], target_name: str) -> tuple[list[CalibrationRecord], list[str]]:
    records: list[CalibrationRecord] = []
    errors: list[str] = []
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            errors.append(f"missing input: {path}")
    seen: set[Path] = set()
    for path in files:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        for row, parent, sample_id in _rows_from_object(payload, str(path)):
            target = _target_from_row(row, target_name)
            if target is None:
                continue
            target_value, target_kind = target
            group_id = _finite(row.get("group_id", row.get("group")))
            if group_id is None:
                continue
            feature_mode, signature_aggregation = _feature_provenance(row)
            records.append(
                CalibrationRecord(
                    group_id=int(group_id),
                    features=_features_from_row(row, parent),
                    target=target_value,
                    source=str(path),
                    sample_id=sample_id,
                    decision=(
                        str(row.get("decision"))
                        if row.get("decision") is not None
                        else None
                    ),
                    target_kind=target_kind,
                    feature_mode=feature_mode,
                    signature_aggregation=signature_aggregation,
                )
            )
    return records, errors


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def robust_scales(records: Sequence[CalibrationRecord], features: Sequence[str]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for feature in features:
        values = [record.features[feature] for record in records if feature in record.features]
        if not values:
            scales[feature] = 1.0
            continue
        # A non-centred p90 scale preserves the non-negative interpretation of
        # every risk term and makes coefficients comparable across groups.
        p90 = _quantile(values, 0.90)
        median = statistics.median(values)
        scales[feature] = max(p90, median, 1e-12)
    return scales


def fit_nonnegative_ridge(
    records: Sequence[CalibrationRecord],
    features: Sequence[str],
    scales: dict[str, float],
    *,
    ridge: float,
    iterations: int = 5000,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot fit an empty calibration set")
    matrix = np.asarray(
        [
            [1.0] + [record.features[name] / scales[name] for name in features]
            for record in records
        ],
        dtype=np.float64,
    )
    target = np.asarray([record.target for record in records], dtype=np.float64)
    count = float(len(records))
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    # The gradient Lipschitz bound is the largest eigenvalue of X'X/n, not
    # the largest individual row norm.  Using the latter can make projected
    # gradient descent oscillate and repeatedly project the useful weights to
    # zero on correlated features.
    gram_bound = float(np.linalg.norm(matrix, ord=2) ** 2 / count)
    step = 1.0 / max(2.0 * (gram_bound + max(0.0, ridge)), 1e-12)
    for _ in range(max(1, int(iterations))):
        residual = matrix @ weights - target
        gradient = (2.0 / count) * (matrix.T @ residual)
        gradient[1:] += 2.0 * max(0.0, ridge) * weights[1:]
        weights -= step * gradient
        np.maximum(weights, 0.0, out=weights)
    prediction = matrix @ weights
    residual = target - prediction
    safety = max(0.0, _quantile(residual.tolist(), 0.95))
    conservative = prediction + safety
    return {
        "samples": len(records),
        "intercept": float(weights[0]),
        "weights_normalized": {
            name: float(weights[index + 1])
            for index, name in enumerate(features)
        },
        "weights_raw": {
            name: float(weights[index + 1] / scales[name])
            for index, name in enumerate(features)
        },
        "safety_margin_p95": float(safety),
        "target_mean": float(target.mean()),
        "target_p95": _quantile(target.tolist(), 0.95),
        "prediction_mean": float(prediction.mean()),
        "prediction_p95": _quantile(prediction.tolist(), 0.95),
        "conservative_coverage": float(np.mean(conservative >= target)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "max_underprediction_after_margin": float(
            max(0.0, float(np.max(target - conservative)))
        ),
    }


def _split_records(
    records: Sequence[CalibrationRecord],
    fraction: float,
) -> tuple[list[CalibrationRecord], list[CalibrationRecord]]:
    if fraction <= 0.0 or len(records) < 2:
        return list(records), []
    sample_ids = sorted({record.sample_id for record in records})
    holdout_count = max(1, int(math.ceil(len(sample_ids) * fraction)))
    # Stable, dependency-free split: lexical ordering is reproducible across
    # machines and keeps all groups from one forward in the same partition.
    holdout_ids = set(sample_ids[-holdout_count:])
    train = [record for record in records if record.sample_id not in holdout_ids]
    holdout = [record for record in records if record.sample_id in holdout_ids]
    if not train:
        return list(records), []
    return train, holdout


def _evaluate(
    records: Sequence[CalibrationRecord],
    fit: dict[str, Any],
    features: Sequence[str],
    scales: dict[str, float],
) -> dict[str, Any]:
    if not records:
        return {"samples": 0}
    normalized = fit["weights_normalized"]
    predictions = []
    targets = []
    for record in records:
        prediction = float(fit["intercept"])
        for name in features:
            prediction += float(normalized[name]) * record.features[name] / scales[name]
        predictions.append(prediction)
        targets.append(record.target)
    margin = float(fit.get("safety_margin", fit["safety_margin_p95"]))
    under = [target - prediction - margin for target, prediction in zip(targets, predictions)]
    return {
        "samples": len(records),
        "rmse": float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))),
        "conservative_coverage": float(
            np.mean(np.asarray(predictions) + margin >= np.asarray(targets))
        ),
        "max_underprediction_after_margin": max(0.0, float(max(under))),
    }


def fit_policy(
    records: Sequence[CalibrationRecord],
    features: Sequence[str],
    *,
    ridge: float,
    min_samples: int,
    holdout_fraction: float,
    safety_quantile: float,
) -> dict[str, Any]:
    if not 0.0 < safety_quantile < 1.0:
        raise ValueError("safety quantile must be between zero and one")
    missing = {
        name: sum(1 for record in records if name not in record.features)
        for name in features
    }
    usable = [
        record
        for record in records
        if all(name in record.features for name in features)
    ]
    if not usable:
        raise ValueError(
            "no records contain every requested feature; missing counts="
            f"{missing}"
        )
    train, holdout = _split_records(usable, holdout_fraction)
    scales = robust_scales(train, features)
    # Fit with the requested quantile after the generic solver; recalculate the
    # margin explicitly so 0.90/0.95 sweeps are reflected in the policy.
    fitted = fit_nonnegative_ridge(train, features, scales, ridge=ridge)
    matrix = np.asarray(
        [
            [1.0] + [record.features[name] / scales[name] for name in features]
            for record in train
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [fitted["intercept"]]
        + [fitted["weights_normalized"][name] for name in features],
        dtype=np.float64,
    )
    residual = np.asarray([record.target for record in train]) - matrix @ weights
    margin = max(0.0, _quantile(residual.tolist(), safety_quantile))
    fitted["safety_margin"] = float(margin)
    fitted["safety_quantile"] = float(safety_quantile)
    fitted["holdout"] = _evaluate(holdout, fitted, features, scales)

    groups: dict[str, Any] = {}
    global_fit = fitted
    for group_id in sorted({record.group_id for record in usable}):
        group_records = [record for record in train if record.group_id == group_id]
        if len(group_records) >= min_samples:
            group_fit = fit_nonnegative_ridge(
                group_records, features, scales, ridge=ridge
            )
            group_matrix = np.asarray(
                [
                    [1.0]
                    + [record.features[name] / scales[name] for name in features]
                    for record in group_records
                ],
                dtype=np.float64,
            )
            group_weights = np.asarray(
                [group_fit["intercept"]]
                + [group_fit["weights_normalized"][name] for name in features],
                dtype=np.float64,
            )
            group_residual = np.asarray(
                [record.target for record in group_records]
            ) - group_matrix @ group_weights
            group_fit["safety_margin"] = float(
                max(0.0, _quantile(group_residual.tolist(), safety_quantile))
            )
            group_fit["safety_quantile"] = float(safety_quantile)
            group_fit["fallback"] = None
            groups[str(group_id)] = group_fit
        else:
            groups[str(group_id)] = {
                "samples": len(group_records),
                "fallback": "global",
            }

    return {
        "version": 1,
        "target": "output_relative_l2",
        "features": list(features),
        "feature_scales": scales,
        "ridge": float(ridge),
        "min_samples_per_group": int(min_samples),
        "safety_quantile": float(safety_quantile),
        "samples": len(usable),
        "missing_feature_counts": missing,
        "global": global_fit,
        "groups": groups,
        "runtime_consumption": "manual_review_only; runtime does not auto-load this file",
        "decision_formula": (
            "risk = intercept + sum(weight_normalized[f] * feature[f] / "
            "feature_scales[f]) + safety_margin"
        ),
    }


def _self_test() -> None:
    rng = np.random.default_rng(20260827)
    records: list[CalibrationRecord] = []
    for index in range(160):
        x = rng.random()
        gate = rng.random() * 0.5
        affine = rng.random() * 0.3
        q_floor = 0.04 + rng.random() * 0.04
        age = float(index % 3)
        target = 0.02 + 0.8 * x + 0.35 * gate + 0.15 * affine + 0.2 * q_floor + 0.03 * age
        target += rng.normal(0.0, 0.005)
        records.append(
            CalibrationRecord(
                group_id=index % 4,
                features={
                    "input": float(x),
                    "gate": float(gate),
                    "affine": float(affine),
                    "q_floor": float(q_floor),
                    "age": age,
                },
                target=max(0.0, float(target)),
                source="self-test",
                sample_id=f"sample-{index}",
                decision="cache",
                target_kind="synthetic",
                feature_mode="signature",
                signature_aggregation="weighted",
            )
        )
    policy = fit_policy(
        records,
        DEFAULT_FEATURES,
        ridge=0.02,
        min_samples=16,
        holdout_fraction=0.2,
        safety_quantile=0.95,
    )
    weights = policy["global"]["weights_normalized"]
    if any(float(value) < -1e-10 for value in weights.values()):
        raise AssertionError(f"negative fitted weight: {weights}")
    if policy["global"]["conservative_coverage"] < 0.90:
        raise AssertionError("synthetic conservative coverage is unexpectedly low")
    print(json.dumps({"ok": True, "global": policy["global"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", choices=("output_relative_l2", "residual_relative_l2"), default="output_relative_l2")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--ridge", type=float, default=0.02)
    parser.add_argument("--min-samples", type=int, default=32)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--safety-quantile", type=float, default=0.95)
    parser.add_argument(
        "--feature-mode",
        choices=("auto", "any", "q4", "signature"),
        default="auto",
        help=(
            "Fit one input-feature representation. auto accepts one explicit "
            "mode and fails on mixed q4/signature data; any is an explicit override."
        ),
    )
    parser.add_argument(
        "--signature-aggregation",
        choices=("auto", "any", "weighted", "max_segment"),
        default="auto",
        help=(
            "When fitting signature data, keep weighted/max_segment separate; "
            "auto fails on mixed aggregation modes."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.inputs:
        raise SystemExit("provide forward-report files/directories or --self-test")
    features = tuple(item.strip() for item in str(args.features).split(",") if item.strip())
    unknown = [item for item in features if item not in FEATURE_NAMES]
    if unknown:
        raise SystemExit(f"unknown feature names: {unknown}; choices={sorted(FEATURE_NAMES)}")
    if args.ridge < 0.0 or args.min_samples < 1:
        raise SystemExit("ridge must be non-negative and min-samples must be positive")
    records, errors = read_records(args.inputs, args.target)
    explicit_modes = {
        record.feature_mode for record in records if record.feature_mode is not None
    }
    explicit_aggregations = {
        record.signature_aggregation
        for record in records
        if record.signature_aggregation is not None
    }
    if args.feature_mode == "auto":
        if len(explicit_modes) > 1:
            raise SystemExit(
                "mixed q4/signature calibration records detected; choose "
                "--feature-mode q4/signature or explicitly pass --feature-mode any"
            )
        selected_mode = next(iter(explicit_modes), None)
    else:
        selected_mode = None if args.feature_mode == "any" else args.feature_mode
    if args.signature_aggregation == "auto":
        if len(explicit_aggregations) > 1:
            raise SystemExit(
                "mixed signature aggregation records detected; choose "
                "--signature-aggregation weighted/max_segment or explicitly pass any"
            )
        selected_aggregation = next(iter(explicit_aggregations), None)
    else:
        selected_aggregation = (
            None
            if args.signature_aggregation == "any"
            else args.signature_aggregation
        )
    if selected_mode is not None:
        records = [
            record for record in records if record.feature_mode == selected_mode
        ]
    if selected_aggregation is not None:
        records = [
            record
            for record in records
            if record.signature_aggregation == selected_aggregation
        ]
    # The policy schema currently names the selected target explicitly.
    if not records:
        raise SystemExit("no calibration target records found\n" + "\n".join(errors))
    policy = fit_policy(
        records,
        features,
        ridge=args.ridge,
        min_samples=args.min_samples,
        holdout_fraction=args.holdout_fraction,
        safety_quantile=args.safety_quantile,
    )
    policy["target"] = args.target
    policy["feature_mode_filter"] = args.feature_mode
    policy["selected_feature_mode"] = selected_mode
    policy["signature_aggregation_filter"] = args.signature_aggregation
    policy["selected_signature_aggregation"] = selected_aggregation
    policy["feature_modes"] = dict(
        sorted(
            (
                str(mode),
                sum(1 for record in records if record.feature_mode == mode),
            )
            for mode in {record.feature_mode for record in records}
        )
    )
    policy["signature_aggregations"] = dict(
        sorted(
            (
                str(aggregation),
                sum(
                    1
                    for record in records
                    if record.signature_aggregation == aggregation
                ),
            )
            for aggregation in {record.signature_aggregation for record in records}
        )
    )
    policy["record_sources"] = sorted({record.source for record in records})
    policy["target_kinds"] = dict(
        sorted(
            (
                kind,
                sum(1 for record in records if record.target_kind == kind),
            )
            for kind in {record.target_kind for record in records}
        )
    )
    if errors:
        policy["read_warnings"] = errors
    rendered = json.dumps(policy, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
