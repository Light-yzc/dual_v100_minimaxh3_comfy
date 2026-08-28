"""Bitwise + wall-time gate for the H3 VAE layer-MP tile pipeline.

The pipeline (``H3_VAE_MP_PIPELINE=1``) only changes how the two existing
decoder stages are scheduled across CUDA streams: every tile still traverses
all 36 blocks in the original order with the same weights, and the blend and
canvas write order is unchanged.  The acceptance gate is therefore ``max_abs
== 0.0``, not a cosine approximation.  A non-zero difference means the change
touched the numerical path and must be reverted.

Both paths are exercised against the *same* resident model object so the
comparison cannot be confounded by a second load.

Usage:
    /home/regen/minimax-h3/.venv/bin/python scripts/test_h3_vae_mp_pipeline.py \
        --width 448 --height 256 --frames 21 \
        --output results/vae_mp_pipeline_448x256.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

COMFY_ROOT = Path(os.environ.get("H3_COMFY_ROOT", "/home/regen/minimax-h3/ComfyUI"))


def _bootstrap() -> None:
    if not COMFY_ROOT.is_dir():
        raise SystemExit(f"ComfyUI root not found: {COMFY_ROOT}")
    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(COMFY_ROOT / "custom_nodes"))


def _resolve_vae_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"VAE checkpoint not found: {path}")
        return path
    candidates = sorted((COMFY_ROOT / "models" / "vae").glob("minimax_h3_video_vae_*.safetensors"))
    if not candidates:
        raise SystemExit("no minimax_h3_video_vae_*.safetensors under models/vae")
    # Prefer INT8 when it exists, otherwise the FP16 production checkpoint.
    for candidate in candidates:
        if "int8" in candidate.name:
            return candidate
    return candidates[0]


def _whole_card_used_mib() -> list[int]:
    import subprocess

    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except Exception:
        return []
    used = []
    for line in raw.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2:
            used.append(int(parts[1]))
    return used


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", default=None, help="video VAE checkpoint path")
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=21,
                        help="pixel frames; converted to latent tokens")
    parser.add_argument("--split", type=int, default=None,
                        help="decoder block split (default H3_VAE_SPLIT or 24)")
    parser.add_argument("--depth", type=int, default=2,
                        help="H3_VAE_MP_PIPELINE_DEPTH under test")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeat", type=int, default=2,
                        help="timed decodes per path after one warmup")
    parser.add_argument("--output", default=None, help="result JSON path")
    args = parser.parse_args()

    _bootstrap()

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("this gate requires two CUDA devices")

    from DualV100 import h3_model_parallel as mp

    vae_path = _resolve_vae_path(args.vae)

    # Load with the pipeline off; it is installed explicitly further down so
    # both measurements share one set of resident weights.
    os.environ["H3_VAE_MP_PIPELINE"] = "0"
    os.environ["H3_VAE_MP_PIPELINE_DEPTH"] = str(args.depth)

    print(f"[gate] loading {vae_path}", flush=True)
    vae = mp.load_h3_video_vae_parallel(str(vae_path), split=args.split)
    if vae is None:
        raise SystemExit("layer-MP VAE unavailable (H3_VAE_MP disabled or no P2P)")
    model = vae.first_stage_model
    report = dict(getattr(model, "_h3_parallel_report", {}))
    print(f"[gate] split={report.get('split')} "
          f"devices={report.get('decoder_devices')}", flush=True)

    latent_h = args.height // model.vae_ratio
    latent_w = args.width // model.vae_ratio
    latent_t = max(2, (args.frames - 5) // 17 * 5 + 2 if args.frames > 1 else 1)

    y_idx, _y_len, _y_ov = model.split_tiles(args.height)
    x_idx, _x_len, _x_ov = model.split_tiles(args.width)
    tiles = len(y_idx) * len(x_idx)
    group = max(1, int(getattr(model, "_h3_v100_int8_tile_batch", 1)))
    print(f"[gate] latent={latent_t}x{latent_h}x{latent_w} "
          f"tiles={len(y_idx)}x{len(x_idx)}={tiles} group_size={group}", flush=True)
    if tiles <= group:
        print("[gate] WARNING: one tile group only; the pipeline will fall back "
              "to the serial path and the comparison is vacuous", flush=True)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.randn(
        (1, model.latents_mean.numel(), latent_t, latent_h, latent_w),
        generator=generator, dtype=torch.float32,
    ).to(device=vae.device, dtype=torch.float16)

    def run(tag: str) -> tuple[object, list[float], list[int]]:
        torch.cuda.synchronize(vae.device)
        # warmup pass is discarded: it pays first-touch workspace costs
        out = vae.decode(latent)
        torch.cuda.synchronize(vae.device)
        times = []
        peak = _whole_card_used_mib()
        for _ in range(max(1, args.repeat)):
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
            start = time.perf_counter()
            out = vae.decode(latent)
            torch.cuda.synchronize(vae.device)
            times.append(time.perf_counter() - start)
            now = _whole_card_used_mib()
            peak = [max(a, b) for a, b in zip(peak, now)] if peak else now
        print(f"[gate] {tag}: {[round(x, 3) for x in times]} s "
              f"whole-card MiB={peak}", flush=True)
        return out, times, peak

    serial_out, serial_times, serial_peak = run("serial   ")

    pipeline_report = mp._install_h3_vae_mp_pipeline(model, vae.parallel_devices)
    if not pipeline_report.get("enabled"):
        # _enabled() reads the environment at install time.
        os.environ["H3_VAE_MP_PIPELINE"] = "1"
        pipeline_report = mp._install_h3_vae_mp_pipeline(model, vae.parallel_devices)
    if not pipeline_report.get("enabled"):
        raise SystemExit(f"pipeline did not install: {pipeline_report}")
    print(f"[gate] pipeline installed: {pipeline_report}", flush=True)

    pipeline_out, pipeline_times, pipeline_peak = run("pipeline ")

    if not getattr(model, "_h3_mp_pipeline_active", False):
        raise SystemExit("pipeline deactivated itself during decode; see the log")

    a = serial_out.float()
    b = pipeline_out.float()
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")
    max_abs = float((a - b).abs().max().item())
    finite = bool(torch.isfinite(b).all().item())
    serial_best = min(serial_times)
    pipeline_best = min(pipeline_times)

    result = {
        "vae": str(vae_path),
        "resolution": [args.width, args.height],
        "pixel_frames": args.frames,
        "latent_shape": list(latent.shape),
        "tiles": {"rows": len(y_idx), "cols": len(x_idx), "group_size": group},
        "split": report.get("split"),
        "pipeline": pipeline_report,
        "serial_seconds": serial_times,
        "pipeline_seconds": pipeline_times,
        "speedup_best": round(serial_best / pipeline_best, 4) if pipeline_best else None,
        "whole_card_peak_mib": {
            "serial": serial_peak, "pipeline": pipeline_peak,
        },
        "max_abs_difference": max_abs,
        "bitwise_identical": max_abs == 0.0,
        "pipeline_output_finite": finite,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[gate] wrote {out_path}", flush=True)

    print(json.dumps(result, indent=2), flush=True)

    if not finite:
        print("[gate] FAIL: pipeline output is not finite", flush=True)
        return 1
    if max_abs != 0.0:
        print(f"[gate] FAIL: not bitwise identical (max_abs={max_abs}); "
              "the scheduling change touched the numerical path", flush=True)
        return 1
    print(f"[gate] PASS: bitwise identical; best-case speedup "
          f"{serial_best / pipeline_best:.3f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
