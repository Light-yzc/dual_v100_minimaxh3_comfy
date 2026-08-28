#!/usr/bin/env python3
"""Audit Qwen3-VL-32B Q2 GGUF tensor and output-row TP geometry.

The audit is CPU-only.  It imports the production header-only descriptor
module, never maps the GGUF payload, and never materializes model weights.
The emitted JSON is the P0 gate described in ``docs/QWEN32B_Q2_TP_PLAN.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MODEL = Path(
    "/mnt/GALAX/minimax-h3/models/text_encoders/"
    "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "qwen32_q2_tp_layout.json"
LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
EXPECTED_QTYPE_COUNTS = {"F16": 2, "F32": 433, "Q2_K": 417, "Q3_K": 50}
EXPECTED_FILE_SIZE = 8_487_968_160
EXPECTED_DATA_OFFSET = 75_040
EXPECTED_LAYER_BYTES = 160_564_224
EXPECTED_LANGUAGE_BYTES = 8_028_211_200
EXPECTED_EMBED_BYTES = 255_252_480


def _load_qwen32_module():
    path = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_qwen32_q2_tp.py"
    spec = importlib.util.spec_from_file_location("h3_qwen32_q2_tp_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Qwen32 Q2 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _tensor_values(layout: Any) -> list[Any]:
    tensors = _field(layout, "tensors")
    if tensors is None:
        raise AttributeError("GGUFLayout does not expose tensors")
    if isinstance(tensors, dict):
        return list(tensors.values())
    return list(tensors)


def _shape(spec: Any) -> tuple[int, ...]:
    shape = _field(spec, "shape")
    if shape is not None:
        return tuple(int(value) for value in shape)
    out_features = _field(spec, "out_features")
    in_features = _field(spec, "in_features")
    if out_features is not None and in_features is not None:
        return int(out_features), int(in_features)
    raise AttributeError(f"tensor descriptor has no shape: {_jsonable(spec)}")


def _qtype_name(spec: Any) -> str:
    qtype = _field(spec, "qtype", _field(spec, "tensor_type"))
    if isinstance(qtype, Enum):
        return qtype.name
    name = getattr(qtype, "name", None)
    return str(name if name is not None else qtype)


def _descriptor_rows(descriptor: Any) -> tuple[int, int]:
    for first_name, count_name in (
        ("first_output_row", "output_row_count"),
        ("row_start", "row_count"),
        ("start_row", "row_count"),
    ):
        first = _field(descriptor, first_name)
        count = _field(descriptor, count_name)
        if first is not None and count is not None:
            return int(first), int(first) + int(count)
    start = _field(descriptor, "row_start", _field(descriptor, "start_row"))
    stop = _field(descriptor, "row_stop", _field(descriptor, "stop_row"))
    if start is not None and stop is not None:
        return int(start), int(stop)
    rows = _field(descriptor, "output_rows", _field(descriptor, "rows"))
    if rows is not None and len(rows) == 2:
        return int(rows[0]), int(rows[1])
    raise AttributeError(f"shard descriptor has no output-row range: {_jsonable(descriptor)}")


def _descriptor_bytes(descriptor: Any, spec: Any) -> int:
    value = _field(descriptor, "n_bytes", _field(descriptor, "byte_count"))
    if value is not None:
        return int(value)
    start, stop = _descriptor_rows(descriptor)
    return (stop - start) * int(_field(spec, "row_bytes"))


def _language_layers(specs: Iterable[Any]) -> dict[int, list[Any]]:
    layers: dict[int, list[Any]] = defaultdict(list)
    for spec in specs:
        match = LAYER_PATTERN.match(str(_field(spec, "name")))
        if match is not None:
            layers[int(match.group(1))].append(spec)
    return dict(layers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Header-only Qwen32 Q2 output-row TP layout audit"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-tensors", type=int, default=902)
    parser.add_argument("--expected-layers", type=int, default=50)
    parser.add_argument(
        "--allow-check-failures",
        action="store_true",
        help="write the report but return success when a P0 invariant fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.world_size < 1:
        raise ValueError("world-size must be positive")

    qwen32 = _load_qwen32_module()
    # Importing torch/gguf is a fixed process cost, not GGUF header residency.
    # Measure immediately around the actual header parse so the report proves
    # that the 7.9 GiB payload was not pulled into the service RSS.
    before_rss = _rss_mib()
    before_peak = _peak_rss_mib()
    started = time.perf_counter()
    layout = qwen32.inspect_gguf(args.model)
    inspect_seconds = time.perf_counter() - started
    specs = _tensor_values(layout)
    layers = _language_layers(specs)

    qtype_counts = Counter(_qtype_name(spec) for spec in specs)
    qtype_bytes = Counter()
    tensor_records: list[dict[str, Any]] = []
    for spec in specs:
        record = _jsonable(spec)
        if not isinstance(record, dict):
            raise TypeError(f"unexpected tensor descriptor: {record!r}")
        record.setdefault("name", str(_field(spec, "name")))
        record.setdefault("shape", list(_shape(spec)))
        record["qtype"] = _qtype_name(spec)
        record["n_bytes"] = int(_field(spec, "n_bytes"))
        qtype_bytes[record["qtype"]] += record["n_bytes"]
        tensor_records.append(record)

    layer_records = []
    all_shards: list[dict[str, Any]] = []
    shard_failures: list[dict[str, Any]] = []
    matrix_count = 0
    for layer_index in sorted(layers):
        layer_specs = layers[layer_index]
        layer_bytes = sum(int(_field(spec, "n_bytes")) for spec in layer_specs)
        matrix_names = []
        for spec in sorted(layer_specs, key=lambda item: str(_field(item, "name"))):
            shape = _shape(spec)
            if len(shape) != 2:
                continue
            matrix_count += 1
            matrix_names.append(str(_field(spec, "name")))
            descriptors = list(
                qwen32.build_output_row_shards(spec, world_size=args.world_size)
            )
            rows = [_descriptor_rows(descriptor) for descriptor in descriptors]
            shard_bytes = [_descriptor_bytes(descriptor, spec) for descriptor in descriptors]
            expected_rows = int(shape[0])
            expected_bytes = int(_field(spec, "n_bytes"))
            row_coverage_ok = (
                len(rows) == args.world_size
                and rows[0][0] == 0
                and rows[-1][1] == expected_rows
                and all(left[1] == right[0] for left, right in zip(rows, rows[1:]))
            )
            byte_sum_ok = sum(shard_bytes) == expected_bytes
            rank_order_ok = all(
                int(_field(descriptor, "rank", rank)) == rank
                for rank, descriptor in enumerate(descriptors)
            )
            file_ranges_ok = all(
                int(_field(descriptor, "data_offset"))
                == int(_field(spec, "data_offset")) + start * int(_field(spec, "row_bytes"))
                and _descriptor_bytes(descriptor, spec)
                == (stop - start) * int(_field(spec, "row_bytes"))
                for descriptor, (start, stop) in zip(descriptors, rows)
            )
            block_alignment_ok = (
                int(shape[1]) % int(_field(spec, "block_elements")) == 0
                and int(_field(spec, "row_bytes"))
                == int(shape[1])
                // int(_field(spec, "block_elements"))
                * int(_field(spec, "block_bytes"))
            )
            item = {
                "tensor": str(_field(spec, "name")),
                "layer": layer_index,
                "qtype": _qtype_name(spec),
                "shape": list(shape),
                "source_n_bytes": expected_bytes,
                "shard_n_bytes": shard_bytes,
                "row_coverage_ok": row_coverage_ok,
                "byte_sum_ok": byte_sum_ok,
                "rank_order_ok": rank_order_ok,
                "file_ranges_ok": file_ranges_ok,
                "block_alignment_ok": block_alignment_ok,
                "ranks": [_jsonable(descriptor) for descriptor in descriptors],
            }
            all_shards.append(item)
            if not (
                row_coverage_ok
                and byte_sum_ok
                and rank_order_ok
                and file_ranges_ok
                and block_alignment_ok
            ):
                shard_failures.append(item)
        layer_records.append(
            {
                "layer": layer_index,
                "tensor_count": len(layer_specs),
                "matrix_count": len(matrix_names),
                "n_bytes": layer_bytes,
                "matrix_names": matrix_names,
            }
        )

    mmap_hits = qwen32.payload_mmap_hits(args.model)
    mmap_hits = list(mmap_hits) if not isinstance(mmap_hits, int) else mmap_hits
    after_rss = _rss_mib()
    after_peak = _peak_rss_mib()
    layer_byte_values = {item["n_bytes"] for item in layer_records}
    layer_ids = sorted(layers)
    language_bytes = sum(item["n_bytes"] for item in layer_records)
    tensor_by_name = {str(_field(spec, "name")): spec for spec in specs}
    embedding = tensor_by_name.get("model.embed_tokens.weight")
    checks = {
        "tensor_count": len(specs) == args.expected_tensors,
        "known_file_size": args.model.stat().st_size == EXPECTED_FILE_SIZE,
        "known_data_offset": int(_field(layout, "data_offset")) == EXPECTED_DATA_OFFSET,
        "known_qtype_counts": dict(sorted(qtype_counts.items())) == EXPECTED_QTYPE_COUNTS,
        "language_layer_count": len(layers) == args.expected_layers,
        "language_layer_ids_contiguous": layer_ids == list(range(args.expected_layers)),
        "language_layer_bytes_uniform": layer_byte_values == {EXPECTED_LAYER_BYTES},
        "known_language_payload_bytes": language_bytes == EXPECTED_LANGUAGE_BYTES,
        "known_embedding_bytes": (
            embedding is not None and int(_field(embedding, "n_bytes")) == EXPECTED_EMBED_BYTES
        ),
        "language_matrix_count": matrix_count == args.expected_layers * 7,
        "all_output_row_shards_closed": not shard_failures,
        "payload_mmap_count_zero": (mmap_hits == 0 if isinstance(mmap_hits, int) else not mmap_hits),
        "layout_file_size_matches": int(_field(layout, "file_size")) == args.model.stat().st_size,
    }
    checks["passed"] = all(checks.values())

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": sys.version,
        "model": str(args.model.resolve()),
        "file_size": args.model.stat().st_size,
        "world_size": args.world_size,
        "header_only": True,
        "payload_materialized": False,
        "inspect_seconds": inspect_seconds,
        "rss_before_mib": before_rss,
        "rss_after_mib": after_rss,
        "rss_delta_mib": after_rss - before_rss,
        "rss_peak_before_mib": before_peak,
        "rss_peak_after_mib": after_peak,
        "payload_mmap_hits": mmap_hits,
        "layout": _jsonable(layout),
        "qtype_counts": dict(sorted(qtype_counts.items())),
        "qtype_bytes": dict(sorted(qtype_bytes.items())),
        "language_layer_bytes": language_bytes,
        "language_layers": layer_records,
        "output_row_shards": all_shards,
        "shard_failures": shard_failures,
        "tensors": tensor_records,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved report: {args.output}", flush=True)
    if not checks["passed"] and not args.allow_check_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
