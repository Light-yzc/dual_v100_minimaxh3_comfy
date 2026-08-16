#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def tensor_stats(tensor: torch.Tensor):
    finite = torch.isfinite(tensor)
    values = tensor[finite]
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(finite.all().item()),
        "nonfinite_count": int((~finite).sum().item()),
        "min": float(values.min().item()) if values.numel() else None,
        "max": float(values.max().item()) if values.numel() else None,
        "rms": float(values.float().square().mean().sqrt().item()) if values.numel() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("latent", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.latent, map_location="cpu", weights_only=True)
    if payload.get("kind") == "h3_av_nested":
        tensors = payload["tensors"]
    elif payload.get("kind") == "latent":
        tensors = [payload["samples"]]
    else:
        raise SystemExit(f"Unknown latent kind: {payload.get('kind')!r}")

    stats = [tensor_stats(tensor) for tensor in tensors]
    print(json.dumps({"kind": payload["kind"], "tensors": stats}, indent=2))
    if not all(item["finite"] for item in stats):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
