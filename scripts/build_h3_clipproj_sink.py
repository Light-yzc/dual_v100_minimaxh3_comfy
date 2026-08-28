#!/usr/bin/env python3
"""Write the small ClipProj attention-sink sidecar from one 32B calibration row.

The v3.1 residual-only matrix in circulation was calibrated with token 0
excluded.  A single 32B token-0 row is therefore enough to restore that fixed
attention sink; it is not a second text encoder and is only 5120 FP32 values.
"""

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="torch.save tensor or conditioning row")
    parser.add_argument("destination", type=Path, help="*.safetensors sidecar to create")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.destination.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing sidecar: {args.destination}")
    value = torch.load(args.source, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value.get("sink_out", value.get("tensor"))
    if not torch.is_tensor(value):
        raise SystemExit("source must contain a tensor")
    value = value.detach().float().reshape(-1)
    if value.numel() != 5120 or not bool(torch.isfinite(value).all()):
        raise SystemExit(f"source must be one finite [5120] vector, got {tuple(value.shape)}")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_name(args.destination.name + ".tmp")
    save_file({"sink_out": value.contiguous()}, str(temporary), metadata={"kind": "MiniMax-H3 token-0 sink"})
    os.replace(temporary, args.destination)
    print(f"Wrote {args.destination} ({args.destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
