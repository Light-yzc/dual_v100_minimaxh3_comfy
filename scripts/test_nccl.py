#!/usr/bin/env python3
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    rank = dist.get_rank()
    world = dist.get_world_size()

    size_bytes = 256 * 1024 * 1024
    tensor = torch.full(
        (size_bytes // 2,),
        float(rank + 1),
        dtype=torch.float16,
        device=local_rank,
    )

    for _ in range(3):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    iterations = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    end.record()
    end.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0
    algorithmic_gib_s = (size_bytes * iterations / 2**30) / elapsed_s
    bus_gib_s = algorithmic_gib_s * (2.0 * (world - 1) / world)
    if rank == 0:
        print(
            f"NCCL all_reduce: {algorithmic_gib_s:.2f} GiB/s algorithmic, "
            f"{bus_gib_s:.2f} GiB/s bus, {elapsed_s:.4f}s"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
