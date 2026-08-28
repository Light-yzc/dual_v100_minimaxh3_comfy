#!/usr/bin/env python3
"""Record finite/statistical checks for raw interleaved float32 audio."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    values = np.fromfile(args.audio, dtype=np.float32)
    if values.size % args.channels:
        raise ValueError("raw audio sample count is not divisible by channels")
    finite = np.isfinite(values)
    valid = values[finite]
    report = {
        "audio": str(args.audio),
        "sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "dtype": "float32",
        "channels": args.channels,
        "sample_rate": args.sample_rate,
        "frames": int(values.size // args.channels),
        "duration_seconds": float(values.size / args.channels / args.sample_rate),
        "finite": bool(finite.all()),
        "nonfinite_count": int((~finite).sum()),
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "rms": float(np.sqrt(np.mean(np.square(valid, dtype=np.float64)))) if valid.size else None,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["finite"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
