#!/usr/bin/env python3
"""Exercise bidirectional CUDA peer copies for a sustained NVLink stability test."""

import argparse
import json
import os
import time

import torch


def synchronize_pair() -> None:
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seconds",
        type=float,
        default=float(os.environ.get("NVLINK_SOAK_SECONDS", "600")),
        help="Minimum test duration (default: NVLINK_SOAK_SECONDS or 600).",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=int(os.environ.get("NVLINK_SOAK_BYTES", str(128 * 1024 * 1024))),
        help="Bytes per direction per copy (default: 128 MiB).",
    )
    parser.add_argument(
        "--report-every",
        type=float,
        default=float(os.environ.get("NVLINK_SOAK_REPORT_SECONDS", "30")),
        help="Seconds between bandwidth/correctness reports.",
    )
    parser.add_argument(
        "--copies-per-batch",
        type=int,
        default=int(os.environ.get("NVLINK_SOAK_COPIES_PER_BATCH", "16")),
        help="Concurrent-copy batches between synchronizations.",
    )
    args = parser.parse_args()

    if args.seconds <= 0 or args.bytes <= 0 or args.bytes % 2:
        raise SystemExit("--seconds and --bytes must be positive; --bytes must be even for float16")
    if args.report_every <= 0 or args.copies_per_batch <= 0:
        raise SystemExit("--report-every and --copies-per-batch must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("Need two usable CUDA devices for the NVLink soak test")
    if not all(torch.cuda.can_device_access_peer(src, dst) for src, dst in ((0, 1), (1, 0))):
        raise SystemExit("CUDA peer access is not available in both directions")

    elements = args.bytes // 2
    source_0 = torch.full((elements,), 1.0, dtype=torch.float16, device="cuda:0")
    source_1 = torch.full((elements,), 2.0, dtype=torch.float16, device="cuda:1")
    target_0 = torch.empty_like(source_0)
    target_1 = torch.empty_like(source_1)
    stream_0 = torch.cuda.Stream(device=0)
    stream_1 = torch.cuda.Stream(device=1)

    def copy_full_duplex() -> None:
        with torch.cuda.device(0), torch.cuda.stream(stream_0):
            target_0.copy_(source_1, non_blocking=True)
        with torch.cuda.device(1), torch.cuda.stream(stream_1):
            target_1.copy_(source_0, non_blocking=True)

    for _ in range(4):
        copy_full_duplex()
    synchronize_pair()

    print(
        json.dumps(
            {
                "event": "start",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "seconds": args.seconds,
                "bytes_per_direction": args.bytes,
                "copies_per_batch": args.copies_per_batch,
                "gpu_0": torch.cuda.get_device_name(0),
                "gpu_1": torch.cuda.get_device_name(1),
            },
            indent=2,
        )
    )

    started = time.perf_counter()
    last_report = started
    last_report_bytes = 0
    transferred_bytes = 0
    batches = 0

    try:
        while True:
            for _ in range(args.copies_per_batch):
                copy_full_duplex()
            synchronize_pair()
            batches += 1
            transferred_bytes += 2 * args.bytes * args.copies_per_batch

            now = time.perf_counter()
            if now - last_report >= args.report_every or now - started >= args.seconds:
                expected_0 = 2.0
                expected_1 = 1.0
                got_0 = target_0[0].item()
                got_1 = target_1[0].item()
                if got_0 != expected_0 or got_1 != expected_1:
                    raise RuntimeError(
                        f"peer-copy mismatch: cuda:1 -> cuda:0={got_0}, "
                        f"cuda:0 -> cuda:1={got_1}"
                    )
                interval_s = now - last_report
                interval_bytes = transferred_bytes - last_report_bytes
                print(
                    json.dumps(
                        {
                            "event": "report",
                            "elapsed_seconds": round(now - started, 3),
                            "batches": batches,
                            "aggregate_gib_per_s": round((interval_bytes / 2**30) / interval_s, 3),
                            "cuda_1_to_cuda_0": got_0,
                            "cuda_0_to_cuda_1": got_1,
                        }
                    ),
                    flush=True,
                )
                last_report = now
                last_report_bytes = transferred_bytes
            if now - started >= args.seconds:
                break
    except RuntimeError as exc:
        elapsed = time.perf_counter() - started
        raise SystemExit(f"NVLink/P2P soak failed after {elapsed:.3f}s: {exc}") from exc

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "event": "pass",
                "elapsed_seconds": round(elapsed, 3),
                "batches": batches,
                "aggregate_gib_per_s": round((transferred_bytes / 2**30) / elapsed, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
