#!/usr/bin/env python3
"""Inspect H3 Q4_0 matrix shards without mapping or reading model payloads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_q4_tp_module():
    path = REPO_ROOT / "custom_nodes" / "DualV100" / "h3_q4_tp.py"
    spec = importlib.util.spec_from_file_location("h3_q4_tp_inspector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Q4 TP helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


q4_tp = load_q4_tp_module()
Q4_BLOCK_BYTES = q4_tp.Q4_BLOCK_BYTES
Q4_BLOCK_ELEMENTS = q4_tp.Q4_BLOCK_ELEMENTS
inspect_q4_matrices = q4_tp.inspect_q4_matrices
output_row_segments = q4_tp.output_row_segments


DEFAULT_MODEL = Path(
    "/mnt/GALAX/minimax-h3/models/diffusion_models/"
    "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "results" / "h3_q4_tp_layout.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = f"blocks.{args.block}"
    roles = {
        "qkv": f"{prefix}.attn.qkv_proj.weight",
        "out_proj": f"{prefix}.attn.out_proj.weight",
        "fc1": f"{prefix}.mlp.fc1.weight",
        "fc2": f"{prefix}.mlp.fc2.weight",
    }
    specs, metadata = inspect_q4_matrices(args.model, set(roles.values()))
    matrices = {}
    for role, name in roles.items():
        spec = specs[name]
        item = asdict(spec)
        item["qtype"] = "Q4_0"
        item["block_elements"] = Q4_BLOCK_ELEMENTS
        item["block_bytes"] = Q4_BLOCK_BYTES
        item["ranks"] = []
        if role in {"qkv", "fc1"}:
            for rank in range(args.world_size):
                segments = output_row_segments(spec, role, rank, args.world_size)
                item["ranks"].append(
                    {
                        "rank": rank,
                        "mode": "output rows",
                        "shape": [
                            sum(stop - start for start, stop in segments),
                            spec.in_features,
                        ],
                        "row_segments": [list(segment) for segment in segments],
                        "file_ranges": [
                            [
                                spec.data_offset + start * spec.row_bytes,
                                spec.data_offset + stop * spec.row_bytes,
                            ]
                            for start, stop in segments
                        ],
                    }
                )
        else:
            local_in = spec.in_features // args.world_size
            if local_in % Q4_BLOCK_ELEMENTS:
                raise ValueError(f"{name} local input width splits a Q4 block")
            local_row_bytes = local_in // Q4_BLOCK_ELEMENTS * Q4_BLOCK_BYTES
            for rank in range(args.world_size):
                item["ranks"].append(
                    {
                        "rank": rank,
                        "mode": "input columns, strided once per output row",
                        "shape": [spec.out_features, local_in],
                        "input_columns": [rank * local_in, (rank + 1) * local_in],
                        "byte_window_in_each_row": [
                            rank * local_row_bytes,
                            (rank + 1) * local_row_bytes,
                        ],
                        "source_row_stride_bytes": spec.row_bytes,
                        "local_n_bytes": spec.out_features * local_row_bytes,
                    }
                )
        matrices[role] = item

    report = {
        "created_unix": time.time(),
        "model": str(args.model),
        "payload_mmap": False,
        "payload_read": False,
        "world_size": args.world_size,
        "block": args.block,
        "gguf": metadata,
        "matrices": matrices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved report: {args.output}")


if __name__ == "__main__":
    main()
