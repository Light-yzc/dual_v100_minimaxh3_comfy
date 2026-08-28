#!/usr/bin/env python3
"""Measure resident Qwen3-VL conditioning with the production layer-MP route.

The script deliberately uses the real ClipProj loader, but does not start a
ComfyUI HTTP server or load H3/VAE.  It is intended for a protected
systemd-run cgroup.  Q4 and INT8 are run in separate processes so a previous
encoder cannot remain resident while the next one is loaded.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import platform
import resource
import sys
import time
import types
from pathlib import Path

import torch


COMFY_ROOT = Path("/home/regen/minimax-h3/ComfyUI")
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

MIB = 2**20
DEVICES = (torch.device("cuda:0"), torch.device("cuda:1"))
PROMPT = (
    "A green glass marble rolls through a shallow line of spilled coffee beans "
    "on a cafe counter, morning sunlight and gentle handheld parallax, "
    "cinematic close shot, coherent motion, no text."
)
Q4_NAME = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
INT8_NAME = "qwen3vl_4b_int8_convrot.safetensors"
PROJECTION_NAME = "mmh3-4b-ClipProj-v3.1.safetensors"


def rss_current_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return 0.0


def cuda_stats() -> dict[str, list[float]]:
    return {
        "allocated_mib": [torch.cuda.memory_allocated(d) / MIB for d in DEVICES],
        "reserved_mib": [torch.cuda.memory_reserved(d) / MIB for d in DEVICES],
        "peak_allocated_mib": [torch.cuda.max_memory_allocated(d) / MIB for d in DEVICES],
    }


def sync_all() -> None:
    for device in DEVICES:
        torch.cuda.synchronize(device)


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    # Accumulate metrics in FP64.  FP32 dot/norm rounding can report cosine
    # slightly above 1.0 for near-identical conditioning and obscures the tiny
    # direct-owner-vs-old-loader delta this harness is meant to gate.
    reference = reference.double().cpu()
    candidate = candidate.double().cpu()
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "finite": bool(torch.isfinite(candidate).all().item()),
        }
    delta = candidate - reference
    ref_norm = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    got_norm = torch.linalg.vector_norm(candidate).clamp_min(1e-12)
    return {
        "shape_match": True,
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_rms": float((torch.linalg.vector_norm(delta) / ref_norm).item()),
        "cosine": float((torch.sum(reference * candidate) / (ref_norm * got_norm)).item()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("q4", "int8"), default="q4")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--projection", default=PROJECTION_NAME)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional reference image. When set, exercise the Qwen3-VL vision/mmproj path.",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("requires two CUDA devices")

    # The INT8 safetensors route uses the production DynamicVRAM patcher: the
    # no-host-mmap reader first exposes meta tensors with file offsets and the
    # patcher materialises each owned parameter directly on its GPU.  A plain
    # standalone import leaves AIMDO/DynamicVRAM disabled, so load_state_dict
    # tries an ordinary meta-tensor copy and fails before the benchmark starts.
    # Q4 has its own direct GGUF loader and deliberately does not need this.
    saved_benchmark_argv = None
    if args.variant == "int8":
        import comfy.options

        # ComfyUI normally parses CLI flags only from main.py.  This script is
        # a standalone harness, so explicitly enable that parser before
        # importing ``nodes``; otherwise the flags below are intentionally
        # ignored by comfy.cli_args.
        comfy.options.args_parsing = True
        saved_benchmark_argv = sys.argv[:]
        sys.argv = [
            sys.argv[0],
            "--enable-dynamic-vram",
            "--fast-disk",
            "--disable-pinned-memory",
        ]
        # comfy_aimdo.host_buffer snapshots control.lib at import time. The
        # normal server initializes the native library before importing
        # nodes/model_patcher, so do the same in this harness.
        import comfy_aimdo.control

        try:
            comfy_aimdo.control.init()
        except TypeError:
            comfy_aimdo.control.init()

    # Register only the two SSD model folders needed by the standalone loader.
    # This keeps the benchmark independent of a running ComfyUI server.
    # Importing nodes first mirrors ComfyUI's startup order: DualV100's
    # optional MultiGPU patch reads the stock node mappings while importing.
    import nodes  # noqa: F401
    import folder_paths
    if saved_benchmark_argv is not None:
        sys.argv = saved_benchmark_argv

    if args.variant == "int8":
        # main.py normally performs this small AIMDO bootstrap before custom
        # nodes are loaded. Reproduce it here without starting an HTTP server,
        # so the file-slice state dict is assigned first and materialised by
        # ModelPatcherDynamic one tensor at a time.
        import comfy.memory_management
        import comfy.model_management
        import comfy.model_patcher
        import comfy_aimdo.control

        try:
            comfy_aimdo.control.init()
        except TypeError:
            # Older comfy-aimdo releases expose only the no-argument form.
            comfy_aimdo.control.init()
        devices = comfy.model_management.get_all_torch_devices()
        try:
            aimdo_initialized = comfy_aimdo.control.init_devices(
                (device.index, 0) for device in devices
            )
        except TypeError:
            aimdo_initialized = comfy_aimdo.control.init_devices(
                device.index for device in devices
            )
        if not aimdo_initialized:
            raise RuntimeError("comfy-aimdo could not initialize the INT8 benchmark devices")
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True

    # The standalone benchmark needs DualV100's pure MP helpers, not its node
    # registration side effects.  Supplying the package path skips
    # DualV100/__init__.py, whose MultiGPU registration expects ComfyUI's full
    # extra-node startup sequence (already present in a server).
    dual_package_name = "custom_nodes.DualV100"
    for module_name in list(sys.modules):
        if module_name == dual_package_name or module_name.startswith(dual_package_name + "."):
            del sys.modules[module_name]
    dual_package = types.ModuleType(dual_package_name)
    dual_package.__path__ = [
        str(COMFY_ROOT / "custom_nodes" / "DualV100")
    ]
    dual_package.__package__ = dual_package_name
    # ``nodes`` or another custom-node import may have populated the real
    # package already; replace it before importing ClipProj's pure helper.
    sys.modules[dual_package_name] = dual_package

    q4_root = "/mnt/GALAX/minimax-h3/experimental/qwen3vl_q4"
    model_root = "/mnt/GALAX/minimax-h3/models"
    folder_paths.add_model_folder_path("text_encoders", q4_root)
    folder_paths.add_model_folder_path("text_encoders", f"{model_root}/text_encoders")
    clipproj_module = importlib.import_module(
        "custom_nodes.ComfyUI-ClipProj.clipproj_nodes"
    )
    projection_module = importlib.import_module(
        "custom_nodes.ComfyUI-ClipProj.clipproj_projection"
    )
    projection_module.register_folder()
    projection_folder = folder_paths.folder_names_and_paths["clip_projections"][0]
    if f"{model_root}/clip_projections" not in projection_folder:
        projection_folder.append(f"{model_root}/clip_projections")

    if args.model is None:
        args.model = Path(q4_root if args.variant == "q4" else f"{model_root}/text_encoders") / (
            Q4_NAME if args.variant == "q4" else INT8_NAME
        )
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    for device in DEVICES:
        torch.cuda.set_device(device)
    torch.manual_seed(20260825)
    started = time.perf_counter()
    report: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hardware": "2x Tesla V100 expected",
        "variant": args.variant,
        "model": str(args.model),
        "projection": args.projection,
        "split": None,
        "route": "resident layer-model-parallel",
        "host_mmap": False,
        "prompt": args.prompt,
        "input_mode": "image+text" if args.image is not None else "text-only",
        "reference_image": str(args.image) if args.image is not None else None,
        "rss_before_mib": rss_current_mib(),
        "cuda_before": cuda_stats(),
    }

    # _load_encoder is the exact production ClipProj entry point.  It installs
    # Qwen's layer MP and keeps the encoder resident; no strict TP is involved.
    clip = clipproj_module._load_encoder(
        args.model.name,
        "auto",
        "cuda:0",
        "resident",
        "qwen-conditioning-benchmark",
    )
    sync_all()
    parallel_report = getattr(clip, "_h3_qwen_parallel_report", None)
    if parallel_report is not None:
        split = int(parallel_report["split"])
        report["split"] = [split, int(parallel_report["language_layers"]) - split]
    report["load_seconds"] = time.perf_counter() - started
    report["cuda_after_load"] = cuda_stats()
    report["rss_after_load_mib"] = rss_current_mib()

    projection_started = time.perf_counter()
    projected = clipproj_module._wrap(clip, args.projection)
    report["projection_load_seconds"] = time.perf_counter() - projection_started

    if args.image is not None:
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        from PIL import Image

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("--image requires numpy in the benchmark environment") from exc
        with Image.open(args.image) as image:
            image = image.convert("RGB")
            image_array = np.asarray(image).copy()
            image_tensor = torch.from_numpy(image_array).float().div(255.0).unsqueeze(0)
        # Keep the image tensor on CPU, matching ComfyUI's LoadImage output;
        # the Qwen visual tower performs its own bounded device transfer.
        encoded_tokens = projected.tokenize(args.prompt, images=[image_tensor])
        report["reference_image_shape"] = list(image_tensor.shape)
        report["reference_image_dtype"] = str(image_tensor.dtype)

        def encode_once():
            # _encode is the same implementation used by encode_from_tokens,
            # but also gives us MiniMax's text/vision position tags.  Keeping
            # the tags in the report lets the Q4-vs-INT8 comparison separate
            # visual-token drift from ordinary prompt-token drift.
            return projected._encode(encoded_tokens)

    else:

        def encode_once():
            return projected._encode(projected.tokenize(args.prompt))

    # Keep this as no_grad rather than inference_mode.  The production
    # ClipProj path already guards the Qwen forward with no_grad, while the
    # residual ClipProj v3.1 MLP is an ordinary nn.Sequential.  PyTorch's
    # inference tensors cannot be consumed by that Linear path on this
    # version ("Inference tensors do not track version counter"), which would
    # make the benchmark fail for the MLP projection only.
    with torch.no_grad():
        first_started = time.perf_counter()
        conditioning, tags = encode_once()
        sync_all()
        first_ms = (time.perf_counter() - first_started) * 1000.0
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(max(1, args.repetitions)):
            call_started = time.perf_counter()
            conditioning, tags = encode_once()
            sync_all()
            timings.append((time.perf_counter() - call_started) * 1000.0)

    if not isinstance(conditioning, torch.Tensor):
        raise TypeError(f"ProjectedCLIP returned {type(conditioning)!r}, expected Tensor")
    conditioning_cpu = conditioning.detach().float().cpu()
    tags_cpu = tags.detach().to(device="cpu", dtype=torch.long)
    finite = bool(torch.isfinite(conditioning_cpu).all().item())
    values = conditioning_cpu
    report.update({
        "conditioning_shape": list(values.shape),
        "conditioning_dtype": str(conditioning.dtype),
        "conditioning_device": str(conditioning.device),
        "conditioning_rms": float(values.square().mean().sqrt().item()),
        "conditioning_max_abs": float(values.abs().max().item()),
        "conditioning_finite": finite,
        "conditioning_tag_counts": {
            "text": int((tags_cpu == 1).sum().item()),
            "vision": int((tags_cpu == 0).sum().item()),
            "other": int(((tags_cpu != 0) & (tags_cpu != 1)).sum().item()),
        },
        "conditioning_tags": tags_cpu.tolist(),
        "cold_conditioning_ms": first_ms,
        "warm_conditioning_ms": timings,
        "warm_conditioning_mean_ms": sum(timings) / len(timings),
        "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "rss_after_conditioning_mib": rss_current_mib(),
        "cuda_after_conditioning": cuda_stats(),
    })

    if args.reference is not None and args.reference.is_file():
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        report["vs_reference"] = error_metrics(reference, conditioning_cpu)
    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        torch.save(conditioning_cpu, args.dump)
        report["conditioning_dump"] = str(args.dump)

    maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    report["model_map_hits"] = [line for line in maps.splitlines() if str(args.model) in line]
    projection_path = str(Path(model_root) / "clip_projections" / args.projection)
    report["projection_map_hits"] = [line for line in maps.splitlines() if projection_path in line]
    report["q4_payload_map_hits"] = [
        line for line in maps.splitlines()
        if args.variant == "q4" and q4_root in line
    ]
    report["qwen_parallel_report"] = parallel_report
    report["finite"] = finite
    report["numerically_qualified"] = finite and (
        "vs_reference" not in report
        or (
            report["vs_reference"].get("finite", False)
            and report["vs_reference"].get("relative_rms", float("inf")) <= 0.05
            and report["vs_reference"].get("cosine", 0.0) >= 0.995
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    del projected, clip, conditioning, conditioning_cpu, tags, tags_cpu
    gc.collect()
    for device in DEVICES:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
