#!/usr/bin/env python3
"""Gate FP16 Tensor Core / FP32-output execution for H3 row LoRA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path("/home/regen/minimax-h3/ComfyUI")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(REPO_ROOT / "custom_nodes" / "DualV100"))

import h3_lora_tp as lora_tp  # noqa: E402


LORA_PATH = Path(
    "/mnt/GALAX/minimax-h3/models/loras/minimax_h3_turbo_v4_step600_ema.safetensors"
)
FP16_TARGET = 32752.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--staging-mib", type=int, default=4)
    parser.add_argument("--lora", type=Path, default=LORA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "h3_lora_fp32_tc_gate.json",
    )
    return parser.parse_args()


def power_scale(x: torch.Tensor) -> torch.Tensor:
    maximum = x.detach().abs().amax(dim=-1, keepdim=True)
    ratio = (maximum / FP16_TARGET).clamp_min_(1.0)
    return torch.exp2(torch.ceil(torch.log2(ratio)))


def fp32_lora(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    source = x.float() if x.dtype != torch.float32 else x
    return functional.linear(functional.linear(source, a), b)


def tc_lora(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16:
        source = x
        input_scale = None
    else:
        input_scale = power_scale(x)
        source = (x / input_scale).half()
    rank_output = torch.mm(source, a.t(), out_dtype=torch.float32)
    rank_scale = power_scale(rank_output)
    safe_rank = (rank_output / rank_scale).half()
    output = torch.mm(safe_rank, b.t(), out_dtype=torch.float32)
    output.mul_(rank_scale)
    if input_scale is not None:
        output.mul_(input_scale)
    return output


def tensor_error(reference: torch.Tensor, actual: torch.Tensor):
    delta = actual.float() - reference.float()
    ref = reference.float()
    rms = delta.square().mean().sqrt()
    ref_rms = ref.square().mean().sqrt()
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rms": float(rms.item()),
        "reference_rms": float(ref_rms.item()),
        "relative_rms": float((rms / ref_rms.clamp_min(1e-30)).item()),
        "cosine": float(functional.cosine_similarity(ref.flatten(), actual.float().flatten(), dim=0).item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def timed(call, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repetitions)
    return statistics.median(samples)


def load_row_loras(path: Path, device: torch.device, staging_bytes: int):
    names = lora_tp.h3_lora_names(0)
    selected = {name for role in ("out_proj", "fc2") for name in names[role]}
    specs, metadata = lora_tp.inspect_safetensors(path, selected)
    result = {}
    with lora_tp.SafeTensorDiskReader(path, device, staging_bytes) as reader:
        for role in ("out_proj", "fc2"):
            a_name, b_name = names[role]
            a = reader.read_input_shard(specs[a_name], 0, 2, torch.float32)
            b = reader.read_full(specs[b_name], torch.float32)
            result[role] = (a, b)
    return result, metadata


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise SystemExit("this benchmark requires SM70")
    loras, metadata = load_row_loras(args.lora, device, args.staging_mib << 20)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    attention = torch.randn(
        (args.sequence, 3584), generator=generator, device=device, dtype=torch.float16
    )
    gate = torch.randn(
        (args.sequence, 7168), generator=generator, device=device, dtype=torch.float16
    ).mul_(128.0)
    up = torch.randn(
        (args.sequence, 7168), generator=generator, device=device, dtype=torch.float16
    ).mul_(128.0)
    swiglu = functional.silu(gate.float()).mul_(up.float())

    cases = {}
    for role, source in (("out_proj", attention), ("fc2", swiglu)):
        a32, b32 = loras[role]
        # This is a numerical conversion from source BF16 values, never a
        # same-width byte reinterpretation.
        a16, b16 = a32.half(), b32.half()
        reference = fp32_lora(source, a32, b32)
        candidate = tc_lora(source, a16, b16)
        fp32_ms = timed(
            lambda: fp32_lora(source, a32, b32), args.warmup, args.repetitions
        )
        tc_ms = timed(
            lambda: tc_lora(source, a16, b16), args.warmup, args.repetitions
        )
        cases[role] = {
            "source_shape": list(source.shape),
            "source_dtype": str(source.dtype),
            "source_max_abs": float(source.abs().max().item()),
            "a_shape": list(a32.shape),
            "b_shape": list(b32.shape),
            "fp32_ms": fp32_ms,
            "tensor_core_fp32_output_ms": tc_ms,
            "speedup": fp32_ms / tc_ms,
            "error": tensor_error(reference, candidate),
            "fp32_weight_mib": (a32.numel() + b32.numel()) * 4 / (1 << 20),
            "fp16_weight_mib": (a16.numel() + b16.numel()) * 2 / (1 << 20),
        }
        del reference, candidate, a16, b16

    qualified = all(
        case["error"]["finite"]
        and case["error"]["cosine"] >= 0.99999
        and case["error"]["relative_rms"] <= 2e-3
        and case["speedup"] > 1.0
        for case in cases.values()
    )
    report = {
        "created_unix": time.time(),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": "7.0",
        "sequence": args.sequence,
        "lora": str(args.lora),
        "lora_metadata": metadata,
        "conversion": "BF16 source -> FP32 reference / numerical FP16 candidate",
        "cases": cases,
        "numerically_qualified": qualified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved report: {args.output}")


if __name__ == "__main__":
    main()
