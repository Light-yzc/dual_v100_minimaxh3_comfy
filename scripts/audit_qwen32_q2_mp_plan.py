#!/usr/bin/env python3
"""Print a header-only Qwen32B Q2 layer-MP split and VRAM estimate.

The command never materializes a tensor payload and never starts NCCL.  It is
safe to run while the ComfyUI TP service is serving requests; CUDA memory is
only queried when the caller explicitly asks for CUDA devices.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MODEL = Path(
    "/mnt/GALAX/minimax-h3/models/text_encoders/"
    "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "qwen32_q2_mp_plan.json"


def _load_modules():
    # Import through the package when ComfyUI is available.  The fallback lets
    # this CPU audit run from a clean source checkout without node discovery.
    try:
        return importlib.import_module(
            "custom_nodes.DualV100.h3_qwen32_q2_mp"
        )
    except (ImportError, ModuleNotFoundError):
        dual = REPO_ROOT / "custom_nodes" / "DualV100"
        if str(dual) not in sys.path:
            sys.path.insert(0, str(dual))
        return importlib.import_module("h3_qwen32_q2_mp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Header-only Qwen32 Q2 layer-MP VRAM plan"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--devices",
        default="cpu,cpu",
        help="two devices, e.g. cpu,cpu for a non-CUDA audit or cuda:0,cuda:1",
    )
    parser.add_argument(
        "--split",
        default="auto",
        help="auto or the first layer assigned to device 1 (for example 12)",
    )
    parser.add_argument(
        "--residency",
        choices=("evict", "partial", "full"),
        default="evict",
    )
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument(
        "--baseline-mib",
        nargs=2,
        type=int,
        metavar=("GPU0", "GPU1"),
        help="optional existing allocation baseline for each device",
    )
    parser.add_argument(
        "--capacity-mib",
        nargs=2,
        type=int,
        metavar=("GPU0", "GPU1"),
        help="optional hard capacity for each device",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.dtype == "float16":
        import torch

        dtype = torch.float16
    else:
        import torch

        dtype = torch.float32
    mp = _load_modules()
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    if len(devices) != 2:
        raise ValueError("--devices must contain two comma-separated values")
    layout = mp.qwen.inspect_gguf(args.model)
    plan = mp.plan_layer_split(
        layout,
        devices=devices,
        split=args.split,
        residency=args.residency,
        dtype=dtype,
        baseline_bytes=(
            None
            if args.baseline_mib is None
            else tuple(int(value) * (1 << 20) for value in args.baseline_mib)
        ),
        capacity_bytes=(
            None
            if args.capacity_mib is None
            else tuple(int(value) * (1 << 20) for value in args.capacity_mib)
        ),
    )
    report = {
        "model": str(args.model.resolve()),
        "header_only": True,
        "payload_mmap_hits": mp.qwen.payload_mmap_hits(args.model),
        "tensor_count": layout.tensor_count,
        "language_layers": layout.language_layer_count,
        "payload_bytes": layout.payload_bytes,
        "plan": plan.as_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
