#!/usr/bin/env python3
import json
import time

import torch


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

    size_bytes = 512 * 1024 * 1024
    elements = size_bytes // 2
    source = torch.ones(elements, dtype=torch.float16, device="cuda:0")
    target = torch.empty_like(source, device="cuda:1")

    for _ in range(3):
        target.copy_(source, non_blocking=True)
    torch.cuda.synchronize()

    iterations = 20
    start = time.perf_counter()
    for _ in range(iterations):
        target.copy_(source, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gib_s = (size_bytes * iterations / 2**30) / elapsed
    checksum = target[:1024].float().sum().item()
    print(f"cuda:0 -> cuda:1: {gib_s:.2f} GiB/s, checksum={checksum:.1f}")


if __name__ == "__main__":
    main()
