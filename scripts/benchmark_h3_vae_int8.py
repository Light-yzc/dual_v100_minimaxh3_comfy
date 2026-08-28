#!/usr/bin/env python3
"""A/B benchmark for the MiniMax-H3 video VAE on the dual V100 host.

Each invocation loads exactly one VAE in a fresh process.  The loader is the
resident two-way MP path, and H3_NO_HOST_MMAP is enabled by default.  A saved
CPU output from a previous invocation can be supplied with ``--reference`` so
the INT8 decode is compared with the FP16 decode from the identical latent.

This intentionally measures VAE decode only.  It does not claim that a
checkpoint's on-disk size is its VRAM footprint: the JSON reports all three
quantities separately, plus per-GPU allocated/reserved peaks and host RSS.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

# Keep host-side work bounded before importing torch/ComfyUI.
os.environ.setdefault("H3_NO_HOST_MMAP", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch


COMFY_ROOT = Path("/home/regen/minimax-h3/ComfyUI")
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

FP16_PATH = Path("/mnt/GALAX/minimax-h3/models/vae/minimax_h3_video_vae_fp16.safetensors")
INT8_PATH = Path(
    "/mnt/GALAX/minimax-h3/experimental/vae_int8/"
    "minimax_h3_video_vae_int8_convrot.safetensors"
)
MIB = 2**20


def _import_h3_mp():
    """Import the installed DualV100 module without running node discovery."""
    module_path = COMFY_ROOT / "custom_nodes/DualV100/h3_model_parallel.py"
    spec = importlib.util.spec_from_file_location("h3_vae_benchmark_mp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_video_latent(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        tensors = value.get("tensors")
        if isinstance(tensors, (list, tuple)) and tensors:
            return _extract_video_latent(tensors[0])
        samples = value.get("samples")
        if isinstance(samples, torch.Tensor):
            return samples
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _extract_video_latent(item)
            except TypeError:
                continue
    raise TypeError(f"no video latent tensor in {type(value).__name__}")


def _load_latent(path: Path | None, shape: tuple[int, int, int], device: torch.device):
    if path is None:
        generator = torch.Generator(device=device).manual_seed(20260826)
        return torch.randn((1, 24, *shape), device=device, dtype=torch.float16, generator=generator)
    value = torch.load(path, map_location="cpu", weights_only=False)
    latent = _extract_video_latent(value)
    if latent.ndim != 5 or latent.shape[1] != 24:
        raise ValueError(f"expected [B,24,T,H,W] latent, got {tuple(latent.shape)}")
    return latent.to(device=device, dtype=torch.float16, copy=True)


def _rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _current_rss_mib() -> float:
    try:
        for line in Path(f"/proc/{os.getpid()}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return _rss_mib()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a checkpoint without materialising it in host RAM."""
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _cuda_alloc(devices):
    return {str(d): torch.cuda.memory_allocated(d) / MIB for d in devices}


def _cuda_reserved(devices):
    return {str(d): torch.cuda.memory_reserved(d) / MIB for d in devices}


def _cuda_peak_alloc(devices):
    return {str(d): torch.cuda.max_memory_allocated(d) / MIB for d in devices}


def _cuda_peak_reserved(devices):
    return {str(d): torch.cuda.max_memory_reserved(d) / MIB for d in devices}


def _sync(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def _error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    """Legacy whole-tensor metric helper for small tensors only.

    Large video outputs must use ``_stream_error_metrics`` below.  Keeping this
    helper makes the numeric definition explicit and is useful for tiny unit
    probes, but the benchmark never invokes it for a large output.
    """
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    delta = candidate - reference
    ref_rms = torch.sqrt(torch.mean(reference.square())).clamp_min(1e-12)
    candidate_rms = torch.sqrt(torch.mean(candidate.square())).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        reference.reshape(1, -1), candidate.reshape(1, -1), dim=1
    )[0].clamp(-1.0, 1.0)
    rmse = torch.sqrt(torch.mean(delta.square()))
    return {
        "shape_match": True,
        "finite": bool(torch.isfinite(candidate).all().item()),
        "rmse": float(rmse.item()),
        "relative_rmse": float((rmse / ref_rms).item()),
        "mae": float(delta.abs().mean().item()),
        "max_abs": float(delta.abs().max().item()),
        "reference_rms": float(ref_rms.item()),
        "cosine": float(cosine.item()),
        "reference_min": float(reference.min().item()),
        "reference_max": float(reference.max().item()),
        "candidate_min": float(candidate.min().item()),
        "candidate_max": float(candidate.max().item()),
    }


def _frame_bounds(tensor: torch.Tensor, chunk_frames: int):
    """Yield frame ranges without flattening a non-contiguous GPU tensor."""
    if tensor.ndim < 2:
        raise ValueError(f"expected a video tensor with a frame dimension, got {tensor.shape}")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    for start in range(0, int(tensor.shape[1]), chunk_frames):
        yield start, min(start + chunk_frames, int(tensor.shape[1]))


def _cpu_frame_chunk(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Copy only one frame chunk to CPU; never reshape the full GPU output."""
    return tensor[:, start:end].detach().to(
        device="cpu", dtype=torch.float32, copy=True
    ).contiguous()


def _stream_tensor_stats(tensor: torch.Tensor, chunk_frames: int) -> dict[str, object]:
    """Calculate output statistics with at most one frame chunk on the CPU."""
    finite = True
    minimum = float("inf")
    maximum = float("-inf")
    for start, end in _frame_bounds(tensor, chunk_frames):
        chunk = _cpu_frame_chunk(tensor, start, end)
        flat = chunk.reshape(-1)
        finite = finite and bool(torch.isfinite(flat).all().item())
        minimum = min(minimum, float(flat.min().item()))
        maximum = max(maximum, float(flat.max().item()))
        del flat, chunk
    return {
        "finite": finite,
        "min": minimum,
        "max": maximum,
    }


def _stream_error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    chunk_frames: int,
) -> dict[str, object]:
    """Compare two videos while keeping only one frame chunk resident on CPU.

    ``candidate`` normally remains on its VAE output GPU.  A non-contiguous
    ``movedim`` output is sliced by frames instead of flattened, which avoids a
    hidden full-size contiguous GPU copy.  ``reference`` may be a file-backed
    tensor; only the accessed chunk is faulted into RAM.
    """
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }

    count = int(candidate.numel())
    finite = True
    diff_sq = 0.0
    diff_abs = 0.0
    ref_sq = 0.0
    candidate_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    ref_min = float("inf")
    ref_max = float("-inf")
    candidate_min = float("inf")
    candidate_max = float("-inf")

    for start, end in _frame_bounds(candidate, chunk_frames):
        candidate_chunk = _cpu_frame_chunk(candidate, start, end)
        reference_chunk = reference[:, start:end].detach().to(
            device="cpu", dtype=torch.float32, copy=False
        ).contiguous()
        candidate_flat = candidate_chunk.reshape(-1)
        reference_flat = reference_chunk.reshape(-1)
        delta = candidate_flat - reference_flat

        finite = finite and bool(
            torch.isfinite(candidate_flat).all().item()
            and torch.isfinite(reference_flat).all().item()
        )
        diff_sq += float(torch.sum(delta.square(), dtype=torch.float64).item())
        diff_abs += float(torch.sum(delta.abs(), dtype=torch.float64).item())
        ref_sq += float(torch.sum(reference_flat.square(), dtype=torch.float64).item())
        candidate_sq += float(torch.sum(candidate_flat.square(), dtype=torch.float64).item())
        dot += float(torch.sum(reference_flat * candidate_flat, dtype=torch.float64).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        ref_min = min(ref_min, float(reference_flat.min().item()))
        ref_max = max(ref_max, float(reference_flat.max().item()))
        candidate_min = min(candidate_min, float(candidate_flat.min().item()))
        candidate_max = max(candidate_max, float(candidate_flat.max().item()))

        del delta, candidate_flat, reference_flat, reference_chunk, candidate_chunk

    ref_rms = (ref_sq / max(count, 1)) ** 0.5
    candidate_rms = (candidate_sq / max(count, 1)) ** 0.5
    rmse = (diff_sq / max(count, 1)) ** 0.5
    denominator = max(ref_rms * candidate_rms * count, 1e-12)
    cosine = max(-1.0, min(1.0, dot / denominator))
    return {
        "shape_match": True,
        "finite": finite,
        "rmse": rmse,
        "relative_rmse": rmse / max(ref_rms, 1e-12),
        "mae": diff_abs / max(count, 1),
        "max_abs": max_abs,
        "reference_rms": ref_rms,
        "candidate_rms": candidate_rms,
        "cosine": cosine,
        "reference_min": ref_min,
        "reference_max": ref_max,
        "candidate_min": candidate_min,
        "candidate_max": candidate_max,
    }


def _reference_sidecar(path: Path) -> Path:
    return Path(f"{path}.json")


def _load_reference(path: Path):
    """Load a small reference normally or a large reference file-backed."""
    if path.suffix == ".f32" and _reference_sidecar(path).is_file():
        metadata = json.loads(_reference_sidecar(path).read_text(encoding="utf-8"))
        shape = tuple(int(x) for x in metadata["shape"])
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        expected = element_count * 4
        if path.stat().st_size != expected:
            raise ValueError(
                f"reference size mismatch for {path}: {path.stat().st_size} != {expected}"
            )
        mapped = torch.from_file(
            str(path), shared=False, size=element_count, dtype=torch.float32
        ).reshape(shape)
        return mapped, {"format": "raw_f32", "mmap": True, "shape": list(shape)}

    # PyTorch's mmap applies to the .pt reference storage only.  The VAE
    # checkpoint itself is still loaded by NoHostMMap and is never mmap'ed.
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=False)
    tensor = _extract_video_latent(value)
    return tensor, {"format": "torch", "mmap": True, "shape": list(tensor.shape)}


def _stream_save_raw_f32(path: Path, tensor: torch.Tensor, chunk_frames: int) -> None:
    """Save a video as raw float32 in frame chunks, with a shape sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb", buffering=1024 * 1024) as handle:
        # A C-order [B,T,...] tensor stores all frames of batch item 0 before
        # batch item 1.  Write batch-by-batch so the raw file can be mapped
        # back with its original shape even if a future benchmark uses B>1.
        for batch_index in range(int(tensor.shape[0])):
            for start, end in _frame_bounds(tensor, chunk_frames):
                chunk = _cpu_frame_chunk(
                    tensor[batch_index : batch_index + 1], start, end
                )
                handle.write(chunk.numpy().tobytes(order="C"))
                del chunk
        handle.flush()
        os.fsync(handle.fileno())
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    _reference_sidecar(path).write_text(
        json.dumps(
            {"dtype": "float32", "shape": list(tensor.shape), "order": "C"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _create_file_backed_output(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    """Create a shared raw-f32 CPU output buffer without anonymous-RAM backing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite disk output buffer: {path}")
    element_count = math.prod(shape)
    with path.open("wb") as handle:
        handle.truncate(element_count * 4)
    return torch.from_file(
        str(path), shared=True, size=element_count, dtype=torch.float32
    ).reshape(shape)


def _decode_to_buffer(vae, samples_in: torch.Tensor, output_buffer: torch.Tensor):
    """H3ParallelVAE.decode equivalent using a caller-owned CPU/disk buffer."""
    samples = samples_in.to(device=vae.device, dtype=vae.vae_dtype)
    vae.first_stage_model.decode(samples, output_buffer=output_buffer)
    return output_buffer.movedim(1, -1)


def _flush_file_backed_output(path: Path) -> None:
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    finally:
        os.close(descriptor)


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("fp16", "int8"), required=True)
    parser.add_argument("--latent", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--save-output", type=Path)
    parser.add_argument(
        "--stream-output",
        type=Path,
        help="save raw float32 output in frame chunks (creates <path>.json)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=int,
        default=24,
        help="decoder layer split on GPU0/GPU1 (production default: 24/12)",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--stream-chunk-frames",
        type=int,
        default=1,
        help="number of video frames per CPU transfer/metric chunk",
    )
    parser.add_argument(
        "--max-full-output-mib",
        type=float,
        default=256.0,
        help="refuse --save-output when its full CPU tensor exceeds this size",
    )
    parser.add_argument(
        "--output-device",
        choices=("gpu", "cpu", "disk"),
        default="gpu",
        help="where the VAE output buffer lives; disk is file-backed CPU memory",
    )
    parser.add_argument(
        "--disk-output-buffer",
        type=Path,
        help="raw BCTHW float32 backing file required by --output-device disk",
    )
    parser.add_argument(
        "--max-cpu-output-mib",
        type=float,
        default=256.0,
        help="refuse large CPU output buffers unless --allow-large-cpu-output is set",
    )
    parser.add_argument(
        "--allow-large-cpu-output",
        action="store_true",
        help="explicitly allow a full large video output buffer on host RAM",
    )
    parser.add_argument("--latent-shape", nargs=3, type=int, metavar=("T", "H", "W"), default=(5, 16, 28))
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0:
        raise SystemExit("warmup must be >= 0 and repetitions must be >= 1")
    if args.stream_chunk_frames < 1:
        raise SystemExit("--stream-chunk-frames must be positive")
    if args.save_output is not None and args.stream_output is not None:
        raise SystemExit("use either --save-output or --stream-output, not both")
    if args.output_device == "disk" and args.disk_output_buffer is None:
        raise SystemExit("--output-device disk requires --disk-output-buffer")
    if args.output_device != "disk" and args.disk_output_buffer is not None:
        raise SystemExit("--disk-output-buffer is only valid with --output-device disk")

    path = INT8_PATH if args.variant == "int8" else FP16_PATH
    if not path.is_file():
        raise SystemExit(f"missing VAE checkpoint: {path}")
    if args.latent is not None and not args.latent.is_file():
        raise SystemExit(f"missing latent: {args.latent}")
    if args.reference is not None and not args.reference.is_file():
        raise SystemExit(f"missing reference output: {args.reference}")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("two CUDA devices are required")
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    torch.cuda.set_device(devices[0])
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)

    report: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "variant": args.variant,
        "checkpoint": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": _sha256_file(path),
        "latent": str(args.latent) if args.latent is not None else "generated",
        "latent_shape_request": list(args.latent_shape),
        "devices": [str(x) for x in devices],
        "split": args.split,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "stream_chunk_frames": args.stream_chunk_frames,
        "max_full_output_mib": args.max_full_output_mib,
        "output_device_mode": args.output_device,
        "max_cpu_output_mib": args.max_cpu_output_mib,
        "host_mmap": False,
        "host_rss_before_mib": _current_rss_mib(),
        "results": {},
    }

    # Install the header-only reader before loading anything from the VAE.
    from custom_nodes.NoHostMMap.safetensors import install

    install()
    h3mp = _import_h3_mp()
    report["loader"] = "DualV100 resident H3 VAE MP + NoHostMMap direct owner"

    load_start = time.perf_counter()
    vae = h3mp.load_h3_video_vae_parallel(
        str(path), devices=tuple(str(x) for x in devices), split=args.split
    )
    _sync(devices)
    load_seconds = time.perf_counter() - load_start
    report["load_seconds"] = load_seconds
    report["load_peak_allocated_mib"] = _cuda_peak_alloc(devices)
    report["load_peak_reserved_mib"] = _cuda_peak_reserved(devices)
    report["allocated_after_load_mib"] = _cuda_alloc(devices)
    report["reserved_after_load_mib"] = _cuda_reserved(devices)
    report["host_rss_after_load_mib"] = _current_rss_mib()
    report["host_rss_peak_mib"] = _rss_mib()
    report["loader_report"] = getattr(vae, "_h3_parallel_report", {})

    latent = _load_latent(args.latent, tuple(args.latent_shape), devices[0])
    report["latent_shape"] = list(latent.shape)
    report["latent_dtype"] = str(latent.dtype)
    expected_output_shape = tuple(
        int(x) for x in vae.first_stage_model.decode_output_shape(latent.shape)
    )
    expected_output_mib = (
        math.prod(expected_output_shape) * torch.tensor([], dtype=torch.float32).element_size() / MIB
    )
    report["expected_output_shape"] = list(expected_output_shape)
    report["expected_output_mib"] = expected_output_mib
    file_backed_output = None
    if args.output_device == "cpu":
        if expected_output_mib > args.max_cpu_output_mib and not args.allow_large_cpu_output:
            raise SystemExit(
                f"refusing CPU output buffer ({expected_output_mib:.1f} MiB > "
                f"{args.max_cpu_output_mib:.1f} MiB); use --output-device gpu "
                "or explicitly add --allow-large-cpu-output"
            )
        vae.output_device = torch.device("cpu")
        decode_once = vae.decode
    elif args.output_device == "gpu":
        # ComfyUI's normal intermediate device is CPU.  The benchmark opts in
        # to a GPU output buffer so a 1 MP video does not occupy another 1.5 GiB
        # of host RAM; only one frame chunk is copied out for metrics/writes.
        vae.output_device = devices[0]
        decode_once = vae.decode
    else:
        file_backed_output = _create_file_backed_output(
            args.disk_output_buffer, expected_output_shape
        )
        decode_once = lambda value: _decode_to_buffer(
            vae, value, file_backed_output
        )
        report["disk_output_buffer"] = str(args.disk_output_buffer)
        report["disk_output_layout"] = "raw float32 BCTHW"
    report["actual_output_device"] = (
        f"file-backed-cpu:{args.disk_output_buffer}"
        if file_backed_output is not None
        else str(vae.output_device)
    )
    _sync(devices)

    with torch.inference_mode():
        for _ in range(args.warmup):
            warm_start = time.perf_counter()
            warm_output = decode_once(latent)
            _sync(devices)
            warm_seconds = time.perf_counter() - warm_start
            del warm_output
        gc.collect()

        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        allocated_baseline = _cuda_alloc(devices)
        reserved_baseline = _cuda_reserved(devices)
        call_seconds = []
        output = None
        for index in range(args.repetitions):
            start = time.perf_counter()
            output = decode_once(latent)
            _sync(devices)
            call_seconds.append(time.perf_counter() - start)
            if index + 1 < args.repetitions:
                del output
                output = None

    if output is None:
        raise RuntimeError("decode produced no output")
    # Keep the final output on its VAE GPU.  Copying the complete 1 MP video
    # to CPU here was the source of the previous RAM spike; all subsequent
    # statistics and optional writes operate one frame chunk at a time.
    output = output.detach()
    report["warmup_seconds"] = warm_seconds if args.warmup else None
    report["decode_seconds"] = call_seconds
    report["decode_mean_seconds"] = sum(call_seconds) / len(call_seconds)
    report["allocated_baseline_before_decode_mib"] = allocated_baseline
    report["reserved_baseline_before_decode_mib"] = reserved_baseline
    report["decode_peak_allocated_mib"] = _cuda_peak_alloc(devices)
    report["decode_peak_reserved_mib"] = _cuda_peak_reserved(devices)
    report["decode_peak_extra_allocated_mib"] = {
        key: _cuda_peak_alloc(devices)[key] - allocated_baseline[key]
        for key in allocated_baseline
    }
    report["decode_peak_extra_reserved_mib"] = {
        key: _cuda_peak_reserved(devices)[key] - reserved_baseline[key]
        for key in allocated_baseline
    }
    report["allocated_after_decode_mib"] = _cuda_alloc(devices)
    report["reserved_after_decode_mib"] = _cuda_reserved(devices)
    report["output_shape"] = list(output.shape)
    report["output_dtype"] = str(output.dtype)
    report["output_device"] = str(output.device)
    report["output_cpu_materialization"] = (
        "file-backed-full-buffer; frame-chunk metrics"
        if file_backed_output is not None
        else "frame_chunks_only"
    )
    report["host_rss_after_decode_mib"] = _current_rss_mib()
    maps = Path(f"/proc/{os.getpid()}/maps").read_text(errors="ignore")
    report["checkpoint_path_in_proc_maps"] = str(path) in maps

    output_stats = _stream_tensor_stats(output, args.stream_chunk_frames)
    report["output_finite"] = output_stats["finite"]
    report["output_min"] = output_stats["min"]
    report["output_max"] = output_stats["max"]

    if args.save_output is not None:
        full_output_mib = output.numel() * output.element_size() / MIB
        if full_output_mib > args.max_full_output_mib:
            raise SystemExit(
                f"refusing full CPU output ({full_output_mib:.1f} MiB > "
                f"{args.max_full_output_mib:.1f} MiB); use --stream-output"
            )
        _save_tensor(args.save_output, output)
        report["saved_output"] = str(args.save_output)
    if args.stream_output is not None:
        _stream_save_raw_f32(args.stream_output, output, args.stream_chunk_frames)
        report["stream_output"] = str(args.stream_output)
        report["stream_output_sidecar"] = str(_reference_sidecar(args.stream_output))

    if args.reference is not None:
        reference, reference_info = _load_reference(args.reference)
        report["reference"] = str(args.reference)
        report["reference_info"] = reference_info
        report["error_vs_reference"] = _stream_error_metrics(
            reference, output, args.stream_chunk_frames
        )
        del reference

    if file_backed_output is not None:
        _flush_file_backed_output(args.disk_output_buffer)
        report["disk_output_path_in_proc_maps"] = (
            str(args.disk_output_buffer) in Path(f"/proc/{os.getpid()}/maps").read_text(
                errors="ignore"
            )
        )

    report["host_rss_after_metrics_mib"] = _current_rss_mib()
    report["host_rss_peak_mib"] = _rss_mib()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
