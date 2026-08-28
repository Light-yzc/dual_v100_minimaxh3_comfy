#!/usr/bin/env python3
"""Compare two saved MiniMax H3 audio/video latent payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as functional


def load(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") == "h3_av_nested":
        return payload["tensors"]
    if payload.get("kind") == "latent":
        return [payload["samples"]]
    raise ValueError(f"unsupported latent kind in {path}: {payload.get('kind')!r}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()


def compare(reference: torch.Tensor, candidate: torch.Tensor):
    if reference.shape != candidate.shape:
        raise ValueError(f"latent shape mismatch: {reference.shape} != {candidate.shape}")
    ref = reference.float()
    got = candidate.float()
    delta = got - ref
    rms = delta.square().mean().sqrt()
    ref_rms = ref.square().mean().sqrt()
    return {
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "finite": bool(torch.isfinite(got).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rms": float(rms.item()),
        "reference_rms": float(ref_rms.item()),
        "relative_rms": float((rms / ref_rms.clamp_min(1e-30)).item()),
        "cosine": float(
            functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = load(args.reference)
    candidate = load(args.candidate)
    if len(reference) != len(candidate):
        raise ValueError(f"latent tensor count mismatch: {len(reference)} != {len(candidate)}")
    labels = ["video", "audio"] if len(reference) == 2 else [str(i) for i in range(len(reference))]
    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_sha256": digest(args.reference),
        "candidate_sha256": digest(args.candidate),
        "tensors": {
            label: compare(ref, got)
            for label, ref, got in zip(labels, reference, candidate, strict=True)
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
