#!/usr/bin/env python3
"""Bounded audit for the resident Qwen3-VL Q4 GGUF loader.

This intentionally stops after ``gguf_clip_loader`` returns.  It does not
construct a Qwen model or run a prompt.  The audit answers the important
loader question first: are the GGML payloads actually resident on the chosen
GPU, or did a surrounding ComfyUI path make a CPU copy?  It uses the same
header-only/no-host-mmap loader as ClipProj and records process RSS, CUDA
allocation, tensor-device counts, and model-file mappings.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch


MIB = 2**20
DEFAULT_MODEL = Path(
    "/mnt/GALAX/minimax-h3/experimental/qwen3vl_q4/"
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
)


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def allocated_mib(device: torch.device) -> float:
    return torch.cuda.memory_allocated(device) / MIB


def reserved_mib(device: torch.device) -> float:
    return torch.cuda.memory_reserved(device) / MIB


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    os.environ.setdefault("H3_NO_HOST_MMAP", "1")
    started = time.perf_counter()
    report: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": str(args.model),
        "device": str(device),
        "host_mmap": False,
        "rss_before_mib": rss_mib(),
        "cuda_allocated_before_mib": allocated_mib(device),
        "cuda_reserved_before_mib": reserved_mib(device),
    }

    # The hyphenated custom-node directory is intentionally imported the same
    # way as the production ClipProj path.
    loader = importlib.import_module("custom_nodes.ComfyUI-GGUF.loader")
    state = loader.gguf_clip_loader(
        str(args.model),
        dynamic=False,
        direct_device=device,
    )
    torch.cuda.synchronize(device)

    devices: dict[str, int] = {}
    quantized = 0
    cpu_tensors: list[str] = []
    meta_tensors: list[str] = []
    for key, value in state.items():
        value_device = getattr(value, "device", None)
        device_name = str(value_device)
        devices[device_name] = devices.get(device_name, 0) + 1
        if value_device is not None and value_device.type == "cpu":
            cpu_tensors.append(key)
        if getattr(value, "is_meta", False):
            meta_tensors.append(key)
        if loader.is_quantized(value):
            quantized += 1

    maps = Path(f"/proc/{os.getpid()}/maps").read_text(encoding="utf-8")
    model_map_hits = [line for line in maps.splitlines() if str(args.model) in line]
    report.update({
        "load_seconds": time.perf_counter() - started,
        "state_dict_keys": len(state),
        "quantized_values": quantized,
        "tensor_devices": devices,
        "cpu_tensor_count": len(cpu_tensors),
        "cpu_tensor_examples": cpu_tensors[:12],
        "meta_tensor_count": len(meta_tensors),
        "meta_tensor_examples": meta_tensors[:12],
        "model_map_hits": model_map_hits,
        "rss_after_mib": rss_mib(),
        "cuda_allocated_after_mib": allocated_mib(device),
        "cuda_reserved_after_mib": reserved_mib(device),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "finite_check": all(
            bool(torch.isfinite(value).all().item())
            for value in state.values()
            if isinstance(value, torch.Tensor)
            and value.device.type == "cuda"
            and value.is_floating_point()
            and value.numel() < 10_000_000
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    del state
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
