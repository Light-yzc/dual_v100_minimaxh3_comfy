#!/usr/bin/env python3
"""Load all 50 real H3 Q4+Turbo shards and benchmark the NCCL TP backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path("/home/regen/minimax-h3/ComfyUI")
sys.path.insert(0, str(REPO_ROOT / "custom_nodes" / "DualV100"))
sys.path.insert(0, str(COMFY_ROOT))

import h3_tp_backbone as tp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=tp.DEFAULT_MODEL)
    parser.add_argument("--lora", type=Path, default=tp.DEFAULT_LORA)
    parser.add_argument("--egrid", type=Path, default=tp.DEFAULT_EGRID)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--staging-mib", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--compare-eager-fp32", action="store_true")
    parser.add_argument(
        "--stage-profile",
        action="store_true",
        help="record Q4/GEMM/LoRA/RoPE/SDPA stages for the measured forward",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def payload_map_count(path: Path) -> int:
    needle = str(path.resolve())
    try:
        return sum(needle in line for line in Path("/proc/self/maps").read_text().splitlines())
    except OSError:
        return -1


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    delta = candidate.float() - reference.float()
    ref = reference.float()
    rms = delta.square().mean().sqrt()
    ref_rms = ref.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        ref.reshape(1, -1), candidate.float().reshape(1, -1)
    )[0]
    return {
        "max_abs": float(delta.abs().max().item()),
        "rms": float(rms.item()),
        "relative_rms": float((rms / ref_rms.clamp_min(1e-30)).item()),
        "cosine": float(cosine.item()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


#: Accuracy gates for every comparison this report may carry, keyed by report
#: field.  ``numerically_qualified`` fails closed on any comparison missing from
#: this table, so adding a new comparison to the report without also declaring
#: its threshold turns the run unqualified instead of silently ignoring it.
COMPARISON_GATES: dict[str, dict[str, float]] = {
    "fused_vs_eager_fp32": {"min_cosine": 0.9999, "max_relative_rms": 2e-2},
    "sequence_parallel_compare": {"min_cosine": 0.9999, "max_relative_rms": 2e-2},
}


def comparison_verdicts(report: dict) -> dict[str, dict]:
    """Grade every output-vs-output comparison present in ``report``.

    ``rank_consistency`` is deliberately excluded: it compares the two ranks
    against each other, so it passes perfectly whenever both ranks make the
    same mistake and can never serve as a correctness gate.  Only comparisons
    against a reference output belong here.
    """
    verdicts: dict[str, dict] = {}
    for field, value in report.items():
        if field == "rank_consistency" or not isinstance(value, dict):
            continue
        if "cosine" not in value or "relative_rms" not in value:
            continue
        gate = COMPARISON_GATES.get(field)
        if gate is None:
            verdicts[field] = {
                "passed": False,
                "reason": f"no accuracy gate declared for {field!r} in COMPARISON_GATES",
            }
            continue
        failures = []
        if not value.get("finite", False):
            failures.append("non-finite output")
        if value["cosine"] < gate["min_cosine"]:
            failures.append(f"cosine {value['cosine']:.6f} < {gate['min_cosine']}")
        if value["relative_rms"] > gate["max_relative_rms"]:
            failures.append(
                f"relative_rms {value['relative_rms']:.6g} > {gate['max_relative_rms']}"
            )
        verdicts[field] = {
            "passed": not failures,
            "reason": "; ".join(failures) if failures else "within gate",
        }
    return verdicts


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank != rank or rank not in (0, 1):
        raise RuntimeError("benchmark requires one local rank on each cuda:0/cuda:1")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != 2:
        raise RuntimeError("benchmark requires exactly two NCCL ranks")

    def progress(stage: str, current: int, total: int) -> None:
        if current == 1 or current == total or current % 10 == 0:
            print(f"rank{rank} load {stage} {current}/{total}", flush=True)

    load_started = time.monotonic()
    backbone = tp.H3TPBackbone(
        rank=rank,
        device=device,
        model_path=args.model,
        lora_path=args.lora,
        egrid_path=args.egrid,
        lora_strength=args.strength,
        staging_bytes=args.staging_mib << 20,
        chunk_rows=args.chunk_rows,
        progress=progress,
    )
    load_wall = time.monotonic() - load_started

    residual = torch.empty(
        (args.sequence, tp.HIDDEN), dtype=torch.float32, device=device
    )
    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        residual.normal_(mean=0.0, std=0.2, generator=generator)
    dist.broadcast(residual, src=0)

    # Exact rows from the model's base curve avoid introducing a synthetic
    # interpolation difference while exercising two timestep/modality groups.
    t_emb = backbone.adaln_table[[0, 512]].clone()
    rope = torch.empty(
        (1, args.sequence, 1, 48, 2, 2), dtype=torch.float16, device=device
    )
    if rank == 0:
        positions = torch.arange(args.sequence, device=device, dtype=torch.float32)
        frequencies = torch.arange(48, device=device, dtype=torch.float32).add_(1.0)
        angles = positions[:, None] * frequencies[None] * (1.0 / 8192.0)
        cosine, sine = torch.cos(angles), torch.sin(angles)
        rope[0, :, 0, :, 0, 0] = cosine
        rope[0, :, 0, :, 0, 1] = -sine
        rope[0, :, 0, :, 1, 0] = sine
        rope[0, :, 0, :, 1, 1] = cosine
    dist.broadcast(t_emb, src=0)
    dist.broadcast(rope, src=0)
    midpoint = args.sequence // 2
    segments = [[0, midpoint, 0], [midpoint, args.sequence, 3]]

    initial = residual.clone()
    fused_vs_eager = None
    if args.compare_eager_fp32:
        backbone.fused_fp32_ops = False
        eager_output, _ = backbone.forward(
            initial.clone(), t_emb, segments, rope, profile=False
        )
        backbone.fused_fp32_ops = True
        fused_output, _ = backbone.forward(
            initial.clone(), t_emb, segments, rope, profile=False
        )
        fused_vs_eager = tensor_error(eager_output, fused_output)
        del eager_output, fused_output
    for _ in range(args.warmup):
        warm_output, _ = backbone.forward(
            initial.clone(), t_emb, segments, rope, profile=False
        )
        del warm_output
    output, metrics = backbone.forward(
        initial.clone(),
        t_emb,
        segments,
        rope,
        profile=True,
        stage_profile=args.stage_profile,
    )
    gathered = [torch.empty_like(output) for _ in range(2)]
    dist.all_gather(gathered, output)
    error = tensor_error(gathered[0], gathered[1])

    local_vector = torch.tensor(
        [
            load_wall,
            rss_mib(),
            metrics["total_ms"],
            metrics["collective_ms"],
            metrics["peak_allocated_mib"],
            float(payload_map_count(args.model)),
        ],
        dtype=torch.float64,
        device=device,
    )
    vectors = [torch.empty_like(local_vector) for _ in range(2)]
    dist.all_gather(vectors, local_vector)

    if rank == 0:
        output_cpu = output.cpu().contiguous()
        report = {
            "created_unix": time.time(),
            "host": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "backend": "full 50-layer persistent Q4+Turbo LoRA / torchrun+NCCL",
            "world_size": 2,
            "shape": {
                "sequence": args.sequence,
                "hidden": tp.HIDDEN,
                "layers": tp.LAYERS,
            },
            "model": str(args.model),
            "lora": str(args.lora),
            "payload_mmap": False,
            "rank_consistency": error,
            "fused_vs_eager_fp32": fused_vs_eager,
            "output_sha256": hashlib.sha256(output_cpu.numpy().tobytes()).hexdigest(),
            "rank_resources": [
                {
                    "rank": index,
                    "load_wall_seconds": float(vector[0].item()),
                    "max_rss_mib": float(vector[1].item()),
                    "forward_ms": float(vector[2].item()),
                    "collective_ms": float(vector[3].item()),
                    "peak_allocated_mib": float(vector[4].item()),
                    "model_payload_maps": int(vector[5].item()),
                }
                for index, vector in enumerate(vectors)
            ],
            "rank0_load": backbone.load_stats,
            "warmup": args.warmup,
            "rank0_profile": metrics,
        }
        verdicts = comparison_verdicts(report)
        report["comparison_verdicts"] = verdicts
        report["numerically_qualified"] = bool(
            error["finite"]
            and error["max_abs"] == 0.0
            and all(verdict["passed"] for verdict in verdicts.values())
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"saved report: {args.output}", flush=True)

    del gathered, output, residual, initial, t_emb, rope, backbone
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
