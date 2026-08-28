#!/usr/bin/env python3
"""Bounded full-language-stack gate for Qwen INT8 layer-MP vs strict TP.

This loads one language block at a time from the real ConvRot safetensors
checkpoint.  The CPU copy is discarded immediately after the block's weights
are placed on their owning GPU; no full state dict, mmap, or full CPU model is
created.  It deliberately measures the language blocks only: the Qwen
embedding, vision tower, deepstack merge, final norm, and ClipProj MLP are
outside this compute/placement gate.

The existing production route is layer-MP (12/24 by default).  Strict TP is measured here
as a candidate, with the same real quantized weights and explicit in-process
peer reductions.  A candidate is qualified only when both TP ranks agree and
the full-vs-TP error stays within the same gate used by the one-layer test.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import time
from pathlib import Path

import torch
import torch.cuda.comm as cuda_comm

from scripts.benchmark_h3_qwen_int8_tp import (
    DEV0,
    DEV1,
    FullQwenBlock,
    LocalQwenBlock,
    _err,
    _load_layer,
    _shard_weights,
)


MIB = 2**20


def _sync() -> None:
    torch.cuda.synchronize(DEV0)
    torch.cuda.synchronize(DEV1)


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    _sync()


def _load_mp(path: Path, layers: int, split: int):
    blocks = []
    header_count = None
    for index in range(layers):
        weights, norms, header_count = _load_layer(path, index)
        device = DEV0 if index < split else DEV1
        blocks.append(FullQwenBlock(weights, norms, device).eval())
        del weights, norms
        gc.collect()
    return blocks, header_count


def _load_tp(path: Path, layers: int):
    local0 = []
    local1 = []
    header_count = None
    for index in range(layers):
        weights, norms, header_count = _load_layer(path, index)
        shard0 = _shard_weights(weights, 0)
        shard1 = _shard_weights(weights, 1)
        local0.append(LocalQwenBlock(0, shard0, norms, DEV0).eval())
        local1.append(LocalQwenBlock(1, shard1, norms, DEV1).eval())
        del shard0, shard1, weights, norms
        gc.collect()
    return local0, local1, header_count


def _mp_forward(blocks, x, split: int):
    current = DEV0
    for index, block in enumerate(blocks):
        target = DEV0 if index < split else DEV1
        if target != current:
            x = x.to(target)
            current = target
        x = block(x)
    return x


def _tp_forward(local0, local1, x0, x1):
    a0, a1 = x0, x1
    for block0, block1 in zip(local0, local1):
        p0 = block0.forward_attention(a0)
        with torch.cuda.device(DEV1):
            p1 = block1.forward_attention(a1)
        with torch.cuda.device(DEV0):
            reduced = cuda_comm.reduce_add([p0, p1], destination=0)
        bcast = cuda_comm.broadcast(
            reduced,
            out=(torch.empty_like(reduced, device=DEV0), torch.empty_like(reduced, device=DEV1)),
        )
        a0 = a0 + bcast[0]
        a1 = a1 + bcast[1]

        p0 = block0.forward_ff(a0)
        with torch.cuda.device(DEV1):
            p1 = block1.forward_ff(a1)
        with torch.cuda.device(DEV0):
            reduced = cuda_comm.reduce_add([p0, p1], destination=0)
        bcast = cuda_comm.broadcast(
            reduced,
            out=(torch.empty_like(reduced, device=DEV0), torch.empty_like(reduced, device=DEV1)),
        )
        a0 = a0 + bcast[0]
        a1 = a1 + bcast[1]
    return a0, a1


def _peak() -> list[float]:
    return [torch.cuda.max_memory_allocated(d) / MIB for d in (DEV0, DEV1)]


def _allocated() -> list[float]:
    return [torch.cuda.memory_allocated(d) / MIB for d in (DEV0, DEV1)]


def _time_mp(blocks, x, split: int, warmup: int, repetitions: int):
    with torch.inference_mode():
        for _ in range(warmup):
            mp_out = _mp_forward(blocks, x, split)
        _sync()
        for d in (DEV0, DEV1):
            torch.cuda.reset_peak_memory_stats(d)
        start = time.perf_counter()
        for _ in range(repetitions):
            mp_out = _mp_forward(blocks, x, split)
        _sync()
        elapsed = (time.perf_counter() - start) * 1000.0 / repetitions
    return elapsed, mp_out.detach(), _peak(), _allocated()


def _time_tp(local0, local1, x0, x1, warmup: int, repetitions: int):
    with torch.inference_mode():
        for _ in range(warmup):
            tp0, tp1 = _tp_forward(local0, local1, x0, x1)
        _sync()
        for d in (DEV0, DEV1):
            torch.cuda.reset_peak_memory_stats(d)
        start = time.perf_counter()
        for _ in range(repetitions):
            tp0, tp1 = _tp_forward(local0, local1, x0, x1)
        _sync()
        elapsed = (time.perf_counter() - start) * 1000.0 / repetitions
    return elapsed, tp0.detach(), tp1.detach(), _peak(), _allocated()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/GALAX/minimax-h3/models/text_encoders/qwen3vl_4b_int8_convrot.safetensors"),
    )
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument(
        "--split",
        type=int,
        default=12,
        help="layer-MP split; production default is 12/24 (strict TP is separate)",
    )
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("requires two CUDA devices")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not 1 <= args.split < args.layers:
        raise ValueError("split must be inside the layer range")
    torch.cuda.set_device(DEV0)
    torch.manual_seed(20260825)
    x0 = torch.randn((1, args.sequence, 2560), device=DEV0, dtype=torch.float16)
    x1 = x0.to(DEV1)

    started = time.perf_counter()
    mp_blocks, header_count = _load_mp(args.model, args.layers, args.split)
    mp_load_seconds = time.perf_counter() - started
    mp_ms, mp_out, mp_peak, mp_allocated = _time_mp(
        mp_blocks, x0, args.split, args.warmup, args.repetitions
    )
    del mp_blocks
    _release_cuda()

    started = time.perf_counter()
    tp0_blocks, tp1_blocks, tp_header_count = _load_tp(args.model, args.layers)
    tp_load_seconds = time.perf_counter() - started
    if tp_header_count != header_count:
        raise RuntimeError("checkpoint header changed between MP and TP loads")
    tp_ms, tp_out0, tp_out1, tp_peak, tp_allocated = _time_tp(
        tp0_blocks, tp1_blocks, x0, x1, args.warmup, args.repetitions
    )

    full_vs_tp = _err(mp_out, tp_out0)
    tp_cross_gpu = _err(tp_out0, tp_out1)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hardware": "2x Tesla V100 expected",
        "model": str(args.model),
        "header_tensors": header_count,
        "checkpoint_scope": f"language_model.layers.0..{args.layers - 1}",
        "layers": args.layers,
        "split": [args.split, args.layers - args.split],
        "host_mmap": False,
        "bounded_cpu_load": "one language layer at a time; page-cache DONTNEED best effort",
        "shape": [1, args.sequence, 2560],
        "layer_mp_ms": mp_ms,
        "strict_tp_ms": tp_ms,
        "speedup_layer_mp_over_tp": mp_ms / tp_ms if tp_ms else None,
        "layer_mp_load_seconds": mp_load_seconds,
        "strict_tp_load_seconds": tp_load_seconds,
        "layer_mp_peak_mib": mp_peak,
        "strict_tp_peak_mib": tp_peak,
        "layer_mp_allocated_after_mib": mp_allocated,
        "strict_tp_allocated_after_mib": tp_allocated,
        "process_maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "full_vs_tp": full_vs_tp,
        "tp_cross_gpu": tp_cross_gpu,
        "finite": bool(
            full_vs_tp["finite"]
            and tp_cross_gpu["finite"]
            and torch.isfinite(mp_out).all().item()
            and torch.isfinite(tp_out0).all().item()
            and torch.isfinite(tp_out1).all().item()
        ),
        "numerically_qualified": bool(
            full_vs_tp["finite"]
            and full_vs_tp["relative_rms"] <= 3e-3
            and full_vs_tp["cosine"] >= 0.9999
            and tp_cross_gpu["finite"]
            and tp_cross_gpu["max_abs"] == 0.0
        ),
        "note": "real INT8 ConvRot language weights; Q/K RMSNorm included; no vision tower, embedding, or ClipProj MLP",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
