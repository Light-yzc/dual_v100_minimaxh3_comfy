#!/usr/bin/env python3
"""Low-memory block-level probe for TP-aware H3 TE-Speed.

One ``torchrun`` loads both persistent shards once and measures the explicit
``tail={4,8,12,16} x mcs={1,2}`` matrix.  It writes scalar JSON only.  This is
a synthetic diagnostic; final video/audio latent comparisons remain required.

  torchrun --standalone --nproc_per_node=2 \
    scripts/benchmark_h3_te_speed_tp.py --sequence 128 \
    --output results/h3_te_speed_block_matrix_s128.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path("/home/regen/minimax-h3/ComfyUI")
sys.path.insert(0, str(REPO_ROOT / "custom_nodes" / "DualV100"))
sys.path.insert(0, str(COMFY_ROOT))

import h3_tp_backbone as tp  # noqa: E402


DEFAULT_TAILS = (4, 8, 12, 16)
DEFAULT_MCS = (1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=tp.DEFAULT_MODEL)
    parser.add_argument("--lora", type=Path, default=tp.DEFAULT_LORA)
    parser.add_argument("--egrid", type=Path, default=tp.DEFAULT_EGRID)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--tail-blocks", type=int, nargs="+")
    parser.add_argument(
        "--warm-blocks",
        type=int,
        help="legacy single boundary; mutually exclusive with --tail-blocks",
    )
    parser.add_argument(
        "--mcs-values", type=int, nargs="+", default=list(DEFAULT_MCS)
    )
    parser.add_argument("--input-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--staging-mib", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument(
        "--max-hidden-mib",
        type=float,
        default=128.0,
        help="hard guard for each synthetic FP32 hidden stream",
    )
    parser.add_argument(
        "--exact-rank-gather-max-mib",
        type=float,
        default=8.0,
        help="use a bounded rank sample above this tensor size",
    )
    parser.add_argument("--metric-chunk-rows", type=int, default=256)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write the full JSON report but print only a compact completion line",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def unique_positive(values: list[int], name: str) -> list[int]:
    result: list[int] = []
    for raw in values:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}")
        if value not in result:
            result.append(value)
    return result


@torch.inference_mode()
def error_metrics_chunked(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    chunk_rows: int,
) -> dict[str, float | bool]:
    """Compare with one bounded FP32 delta chunk instead of a full copy."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError(
            f"expected matching 2D tensors, got {reference.shape}/{candidate.shape}"
        )
    if chunk_rows <= 0:
        raise ValueError("metric chunk rows must be positive")
    delta_sq = ref_sq = candidate_sq = dot = 0.0
    max_abs = 0.0
    finite = True
    count = int(reference.numel())
    for start in range(0, int(reference.shape[0]), chunk_rows):
        stop = min(start + chunk_rows, int(reference.shape[0]))
        ref = reference[start:stop].float()
        got = candidate[start:stop].float()
        delta = got - ref
        delta_sq += float(torch.sum(delta * delta, dtype=torch.float64).item())
        ref_sq += float(torch.sum(ref * ref, dtype=torch.float64).item())
        candidate_sq += float(torch.sum(got * got, dtype=torch.float64).item())
        dot += float(torch.sum(ref * got, dtype=torch.float64).item())
        max_abs = max(
            max_abs,
            float(torch.linalg.vector_norm(delta, ord=float("inf")).item()),
        )
        finite = finite and bool(torch.isfinite(got).all().item())
        del ref, got, delta
    rms = math.sqrt(delta_sq / max(1, count))
    ref_rms = math.sqrt(ref_sq / max(1, count))
    return {
        "max_abs": max_abs,
        "rms": rms,
        "reference_rms": ref_rms,
        "relative_rms": rms / max(ref_rms, 1e-30),
        "cosine": dot / max(math.sqrt(ref_sq * candidate_sq), 1e-30),
        "finite": finite,
    }


def make_inputs(backbone: tp.H3TPBackbone, sequence: int, seed: int):
    if sequence < 8:
        raise ValueError("sequence must be at least 8")
    device = backbone.device
    generator = torch.Generator(device=device).manual_seed(seed)
    residual = (
        torch.randn((sequence, tp.HIDDEN), generator=generator, device=device) * 0.2
    )
    t_emb = backbone.adaln_table[[0, 512]].clone()
    rope = torch.empty(
        (1, sequence, 1, 48, 2, 2), dtype=torch.float16, device=device
    )
    positions = torch.arange(sequence, device=device, dtype=torch.float32)
    frequencies = torch.arange(48, device=device, dtype=torch.float32).add_(1.0)
    angles = positions[:, None] * frequencies[None] * (1.0 / 8192.0)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    rope[0, :, 0, :, 0, 0] = cosine
    rope[0, :, 0, :, 0, 1] = -sine
    rope[0, :, 0, :, 1, 0] = sine
    rope[0, :, 0, :, 1, 1] = cosine
    context_stop = max(1, sequence // 4)
    audio_stop = max(context_stop + 1, sequence // 2)
    segments = [
        [0, context_stop, 1],
        [context_stop, audio_stop, 2],
        [audio_stop, sequence, 3],
    ]
    return residual, t_emb, rope, segments


def timed_forward(backbone, residual, t_emb, segments, rope, **kwargs):
    torch.cuda.synchronize(backbone.device)
    started = time.perf_counter()
    output, metrics = backbone.forward(
        residual, t_emb, segments, rope, **kwargs
    )
    torch.cuda.synchronize(backbone.device)
    return output, metrics, (time.perf_counter() - started) * 1000.0


@torch.inference_mode()
def rank_consistency(
    value: torch.Tensor,
    exact_max_bytes: int,
    metric_chunk_rows: int,
) -> dict[str, Any]:
    value_bytes = int(value.numel() * value.element_size())
    if value_bytes <= exact_max_bytes:
        compared = value
        exact = True
        sampling = None
    else:
        token_stride = max(1, math.ceil(int(value.shape[0]) / 2048))
        hidden_stride = max(1, math.ceil(int(value.shape[1]) / 32))
        compared = value[::token_stride, ::hidden_stride].contiguous()
        exact = False
        sampling = {
            "token_stride": token_stride,
            "hidden_stride": hidden_stride,
            "sample_elements": int(compared.numel()),
            "full_elements": int(value.numel()),
        }
    gathered = [torch.empty_like(compared) for _ in range(2)]
    dist.all_gather(gathered, compared)
    report = error_metrics_chunked(
        gathered[0], gathered[1], metric_chunk_rows
    )
    report.update(
        {
            "exact_full_tensor": exact,
            "compared_bytes_per_rank": int(
                compared.numel() * compared.element_size()
            ),
            "sampling": sampling,
        }
    )
    del gathered
    if compared is not value:
        del compared
    return report


def constant_input_delta(
    source_l2: float, count: int, delta: float
) -> dict[str, float]:
    delta_l2 = abs(float(delta)) * math.sqrt(count)
    return {
        "exact_constant_delta": float(delta),
        "l2_norm": delta_l2,
        "rms": abs(float(delta)),
        "relative_rms": delta_l2 / max(source_l2, 1e-30),
        "max_abs": abs(float(delta)),
    }


def resource_report(
    device: torch.device,
    local_peak_mib: float,
    cache_bytes: int,
) -> list[dict[str, Any]]:
    memory = tp.process_memory_stats()
    local = torch.tensor(
        [
            local_peak_mib,
            torch.cuda.memory_allocated(device) / tp.MIB,
            torch.cuda.memory_reserved(device) / tp.MIB,
            cache_bytes / tp.MIB,
            -1.0 if memory["rss_mib"] is None else memory["rss_mib"],
            -1.0 if memory["rss_peak_mib"] is None else memory["rss_peak_mib"],
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(local) for _ in range(2)]
    dist.all_gather(gathered, local)
    report = []
    for rank, values in enumerate(gathered):
        report.append(
            {
                "rank": rank,
                "peak_allocated_mib": float(values[0].item()),
                "allocated_mib": float(values[1].item()),
                "reserved_mib": float(values[2].item()),
                "cache_mib": float(values[3].item()),
                "rss_mib": None if values[4].item() < 0 else float(values[4].item()),
                "rss_peak_mib": (
                    None if values[5].item() < 0 else float(values[5].item())
                ),
            }
        )
    del gathered, local
    return report


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank not in (0, 1) or local_rank != rank:
        raise RuntimeError("benchmark requires local ranks 0 and 1")
    if args.tail_blocks is not None and args.warm_blocks is not None:
        raise ValueError("use either --tail-blocks or --warm-blocks")
    if args.warm_blocks is not None:
        if not 0 < args.warm_blocks < tp.LAYERS:
            raise ValueError("warm-blocks must be in [1, 49]")
        tails = [tp.LAYERS - int(args.warm_blocks)]
        tail_source = "legacy_warm_blocks_cli"
    else:
        tails = unique_positive(
            list(args.tail_blocks or DEFAULT_TAILS), "tail-blocks"
        )
        tail_source = "tail_blocks"
    if any(tail >= tp.LAYERS for tail in tails):
        raise ValueError(f"tail-blocks must be below {tp.LAYERS}: {tails}")
    mcs_values = unique_positive(list(args.mcs_values), "mcs-values")
    hidden_bytes = int(args.sequence * tp.HIDDEN * 4)
    if hidden_bytes > int(args.max_hidden_mib * tp.MIB):
        raise MemoryError(
            f"synthetic hidden is {hidden_bytes / tp.MIB:.1f} MiB/rank, above "
            f"the {args.max_hidden_mib:.1f} MiB guard; use the Comfy workflow "
            "matrix for real-resolution tests"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    backbone = tp.H3TPBackbone(
        rank=rank,
        device=device,
        model_path=args.model,
        lora_path=args.lora,
        egrid_path=args.egrid,
        lora_strength=args.strength,
        staging_bytes=args.staging_mib << 20,
        chunk_rows=args.chunk_rows,
    )
    source, t_emb, rope, segments = make_inputs(
        backbone, args.sequence, args.seed
    )
    stat_ranges = tp.infer_target_stat_ranges(args.sequence, segments)
    source_stats = tp.tensor_scalar_stats(source)["overall"]
    source_l2 = float(source_stats["l2_norm"])
    source_count = int(source.numel())
    configurations: list[dict[str, Any]] = []

    for tail_blocks in tails:
        boundary_block = tp.LAYERS - tail_blocks
        for mcs in mcs_values:
            cache = tp.H3TPResidualCache("cpu")
            backbone.clear_block_stats_state()
            anchor, anchor_metrics, anchor_ms = timed_forward(
                backbone,
                source.clone(),
                t_emb,
                segments,
                rope,
                start_block=0,
                end_block=tp.LAYERS,
                snapshot_at=boundary_block,
                collect_block_stats=True,
                stat_ranges=stat_ranges,
                reset_block_stats=True,
            )
            anchor_tail_stats = cache.store(
                anchor,
                backbone.take_snapshot(),
                stat_ranges=stat_ranges,
                collect_stats=True,
                retain=True,
            )
            anchor_cache_operation = dict(cache.last_operation)
            local_peak_mib = float(anchor_metrics["peak_allocated_mib"])
            del anchor
            cache_steps: list[dict[str, Any]] = []

            for cache_index in range(1, mcs + 1):
                delta = args.input_delta * cache_index
                exact, exact_metrics, exact_ms = timed_forward(
                    backbone,
                    source.clone().add_(delta),
                    t_emb,
                    segments,
                    rope,
                    start_block=0,
                    end_block=tp.LAYERS,
                    collect_block_stats=True,
                    stat_ranges=stat_ranges,
                )
                candidate, candidate_metrics, partial_ms = timed_forward(
                    backbone,
                    source.clone().add_(delta),
                    t_emb,
                    segments,
                    rope,
                    start_block=0,
                    end_block=boundary_block,
                    collect_block_stats=True,
                    stat_ranges=stat_ranges,
                )
                cache.add_to(candidate, measure=True)
                cache_operation = dict(cache.last_operation)
                approximation_error = error_metrics_chunked(
                    exact, candidate, args.metric_chunk_rows
                )
                ranks = rank_consistency(
                    candidate,
                    int(args.exact_rank_gather_max_mib * tp.MIB),
                    args.metric_chunk_rows,
                )
                local_peak_mib = max(
                    local_peak_mib,
                    float(exact_metrics["peak_allocated_mib"]),
                    float(candidate_metrics["peak_allocated_mib"]),
                    torch.cuda.max_memory_allocated(device) / tp.MIB,
                )
                cache_steps.append(
                    {
                        "cache_index": cache_index,
                        "input_delta_from_anchor": constant_input_delta(
                            source_l2, source_count, delta
                        ),
                        "timing_ms": {
                            "exact_full": exact_ms,
                            "partial_prefix": partial_ms,
                            "cache_operation": cache_operation,
                        },
                        "approximation_error": approximation_error,
                        "rank_consistency": ranks,
                        "exact_metrics": exact_metrics,
                        "candidate_metrics_before_cache_add": candidate_metrics,
                        "candidate_final_stats": tp.tensor_scalar_stats(
                            candidate, stat_ranges
                        ),
                    }
                )
                del exact, candidate

            configurations.append(
                {
                    "tail_blocks": tail_blocks,
                    "boundary_block": boundary_block,
                    "mcs": mcs,
                    "anchor_full_ms": anchor_ms,
                    "anchor_metrics": anchor_metrics,
                    "anchor_tail_residual": anchor_tail_stats,
                    "anchor_cache_operation": anchor_cache_operation,
                    "cache_steps": cache_steps,
                    "rank_resources": resource_report(
                        device, local_peak_mib, cache.bytes
                    ),
                }
            )
            del cache
            dist.barrier()

    if rank == 0:
        report = {
            "created_unix": time.time(),
            "host": platform.node(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "backend": "TP-aware TE-Speed block calibration synthetic probe",
            "world_size": 2,
            "shape": {
                "sequence": args.sequence,
                "hidden": tp.HIDDEN,
                "hidden_mib_per_rank": hidden_bytes / tp.MIB,
                "stat_ranges": [list(item) for item in stat_ranges],
            },
            "tail_source": tail_source,
            "tail_candidates": tails,
            "mcs_candidates": mcs_values,
            "model": str(args.model),
            "lora": str(args.lora),
            "payload_mmap": False,
            "model_loaded_once": True,
            "quality_gate_applied": False,
            "note": (
                "Synthetic hidden errors are data only; compare saved "
                "video/audio latents before any quality decision."
            ),
            "configurations": configurations,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.quiet:
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "configurations": len(configurations),
                        "tail_candidates": tails,
                        "mcs_candidates": mcs_values,
                        "finite": all(
                            step["approximation_error"]["finite"]
                            and step["rank_consistency"]["finite"]
                            for config in configurations
                            for step in config["cache_steps"]
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    del source, t_emb, rope, backbone
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
