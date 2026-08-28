#!/usr/bin/env python3
"""Measure the NVLink cost of the collectives a 2-way H3 tensor-parallel DiT needs.

This is deliberately a communication feasibility test, not a claim that the
current ComfyUI-GGUF loader is tensor-parallel.  H3 has 56 attention heads and
a 14336-wide MLP, both cleanly divisible by two.  A conventional 2-way TP
implementation would use one all-reduce after attention output projection and
one after the MLP output projection in each of 50 DiT layers.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist


def h3_sequence_length(width: int, height: int, frames: int, text_tokens: int) -> dict[str, int]:
    if width % 32 or height % 32:
        raise ValueError("H3 canvas dimensions must be multiples of 32")
    while frames % 17 != 5:
        frames += 1
    latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    video_rows = latent_t * (height // 32) * (width // 32)
    audio_t = round((frames / 24) * 40)
    audio_rows = audio_t * 2
    return {
        "frames": frames,
        "latent_t": latent_t,
        "video_rows": video_rows,
        "audio_rows": audio_rows,
        "sequence_length": text_tokens + audio_rows + video_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    if dist.get_world_size() != 2:
        raise SystemExit("This benchmark is specifically for a two-way H3 TP group")

    shape = h3_sequence_length(args.width, args.height, args.frames, args.text_tokens)
    sequence_length = shape["sequence_length"]
    payload_bytes = sequence_length * args.hidden * 2  # fp16 replicated hidden state
    collectives = args.layers * 2 * args.steps
    # Zeros keep the repeated all-reduce payload finite during a long benchmark.
    tensor = torch.zeros((sequence_length, args.hidden), dtype=torch.float16, device=local_rank)

    for _ in range(args.warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(collectives):
        dist.all_reduce(tensor)
    end.record()
    end.synchronize()
    elapsed_s = start.elapsed_time(end) / 1000.0

    # Matches NCCL's usual two-rank bus-bandwidth definition.
    algorithmic_gib_s = (payload_bytes * collectives / 2**30) / elapsed_s
    bus_gib_s = algorithmic_gib_s * (2 * (dist.get_world_size() - 1) / dist.get_world_size())
    if dist.get_rank() == 0:
        print("H3 two-way TP collective feasibility")
        print(f"canvas={args.width}x{args.height}, frames={shape['frames']}, latent_t={shape['latent_t']}")
        print(
            f"packed rows: text={args.text_tokens}, audio={shape['audio_rows']}, "
            f"video={shape['video_rows']}, total={sequence_length}"
        )
        print(f"replicated hidden payload per all-reduce: {payload_bytes / 2**20:.1f} MiB")
        print(f"collectives: {collectives} ({args.layers} layers x 2 x {args.steps} steps)")
        print(
            f"total communication time: {elapsed_s:.3f}s, "
            f"{elapsed_s / args.steps:.3f}s/denoise step, "
            f"{algorithmic_gib_s:.2f} GiB/s algorithmic, {bus_gib_s:.2f} GiB/s bus"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
