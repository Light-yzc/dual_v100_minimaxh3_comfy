#!/usr/bin/env python3
import json
import os
import time

import torch


def synchronize_pair() -> None:
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)


def peer_copy_bandwidth(src_index: int, dst_index: int, size_bytes: int, iterations: int) -> dict:
    elements = size_bytes // 2
    source = torch.ones(elements, dtype=torch.float16, device=f"cuda:{src_index}")
    target = torch.empty_like(source, device=f"cuda:{dst_index}")

    with torch.cuda.device(dst_index):
        for _ in range(3):
            target.copy_(source, non_blocking=True)
    synchronize_pair()

    start = time.perf_counter()
    with torch.cuda.device(dst_index):
        for _ in range(iterations):
            target.copy_(source, non_blocking=True)
    synchronize_pair()
    elapsed = time.perf_counter() - start

    return {
        "direction": f"cuda:{src_index} -> cuda:{dst_index}",
        "gib_per_s": (size_bytes * iterations / 2**30) / elapsed,
        "checksum": target[:1024].float().sum().item(),
    }


def full_duplex_bandwidth(size_bytes: int, iterations: int) -> dict:
    elements = size_bytes // 2
    source_0 = torch.ones(elements, dtype=torch.float16, device="cuda:0")
    target_0 = torch.empty_like(source_0)
    source_1 = torch.ones(elements, dtype=torch.float16, device="cuda:1")
    target_1 = torch.empty_like(source_1)
    stream_0 = torch.cuda.Stream(device=0)
    stream_1 = torch.cuda.Stream(device=1)

    def copy_both() -> None:
        with torch.cuda.device(0), torch.cuda.stream(stream_0):
            target_0.copy_(source_1, non_blocking=True)
        with torch.cuda.device(1), torch.cuda.stream(stream_1):
            target_1.copy_(source_0, non_blocking=True)

    for _ in range(3):
        copy_both()
    synchronize_pair()

    start = time.perf_counter()
    for _ in range(iterations):
        copy_both()
    synchronize_pair()
    elapsed = time.perf_counter() - start

    return {
        "direction": "full duplex cuda:0 <-> cuda:1",
        "aggregate_gib_per_s": (2 * size_bytes * iterations / 2**30) / elapsed,
        "checksum_cuda_0": target_0[:1024].float().sum().item(),
        "checksum_cuda_1": target_1[:1024].float().sum().item(),
    }


def main() -> None:
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "nccl_available": torch.distributed.is_nccl_available(),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if torch.cuda.device_count() < 2:
        raise SystemExit("Need two CUDA devices")

    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {props.name}, capability={props.major}.{props.minor}, "
            f"VRAM={props.total_memory / 2**30:.2f} GiB"
        )
    for src, dst in ((0, 1), (1, 0)):
        print(
            f"can_device_access_peer({src}, {dst}) = "
            f"{torch.cuda.can_device_access_peer(src, dst)}"
        )

    if not all(torch.cuda.can_device_access_peer(src, dst) for src, dst in ((0, 1), (1, 0))):
        raise SystemExit("CUDA peer access is not available in both directions")

    size_bytes = int(os.environ.get("PEER_COPY_BYTES", str(512 * 1024 * 1024)))
    iterations = int(os.environ.get("PEER_COPY_ITERATIONS", "20"))
    if size_bytes <= 0 or size_bytes % 2:
        raise SystemExit("PEER_COPY_BYTES must be a positive, even byte count")
    if iterations <= 0:
        raise SystemExit("PEER_COPY_ITERATIONS must be positive")

    results = [
        peer_copy_bandwidth(0, 1, size_bytes, iterations),
        peer_copy_bandwidth(1, 0, size_bytes, iterations),
        full_duplex_bandwidth(size_bytes, iterations),
    ]
    print(json.dumps({"size_bytes": size_bytes, "iterations": iterations, "results": results}, indent=2))


if __name__ == "__main__":
    main()
