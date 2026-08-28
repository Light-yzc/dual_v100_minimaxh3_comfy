#!/usr/bin/env python3
"""Low-memory torch.compile gate for MiniMax H3 VAE and nearby modules.

This benchmark deliberately does not load a checkpoint.  It instantiates the
same ComfyUI modules with synthetic weights, runs eager and ``torch.compile``
on one GPU, and records first-call compilation cost, steady-state CUDA-event
time, peak allocated VRAM, and numerical error.  It is intended to answer
whether a local VAE block is worth compiling; it must not be read as an
end-to-end VAE decode result.

The default cases are deliberately bounded.  ``--case`` can be used to run a
single larger shape in a fresh process, which is safer on 16 GiB cards than
accumulating several inductor graphs in one process.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable

# Keep Inductor's host-side compiler parallelism bounded.  These must be set
# before importing torch/ComfyUI, otherwise a compiler worker burst can make a
# small host look like a model-loading OOM.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("MAX_JOBS", "1")

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path("/home/regen/minimax-h3/ComfyUI")
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from comfy.ldm.minimax.audio_vae import AMPBlock1, AttnProjection, BigVGAN  # noqa: E402
from comfy.ldm.minimax.vae import ResnetBlock3D, ViT3DDecoder  # noqa: E402


MIB = 2**20


def _make_finite(module: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    """Move a synthetic module without changing non-parameter rope buffers."""
    module.to(device=device)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.data = parameter.data.to(device=device, dtype=dtype)
            if parameter.ndim == 1 and parameter.numel() > 0:
                parameter.data.fill_(1.0)
            else:
                parameter.data.normal_(mean=0.0, std=0.02)
        for buffer in module.buffers():
            if buffer.is_floating_point() and buffer.numel() > 0 and not bool(torch.isfinite(buffer).all().item()):
                # Empty/uninitialised checkpoint-only buffers are not read by
                # the tested forward.  Finite values make failures explicit if
                # a future code change starts using one, while preserving real
                # filters/rotary frequencies that affect the compiled graph.
                buffer.data = torch.zeros_like(buffer, device=device)
    module.eval()


def _finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


def _first_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"module returned no tensor: {type(value).__name__}")


def _error(reference: object, candidate: object) -> dict[str, object]:
    ref = _first_tensor(reference).float()
    got = _first_tensor(candidate).float()
    if ref.shape != got.shape:
        return {"shape_match": False, "reference": list(ref.shape), "candidate": list(got.shape)}
    delta = got - ref
    denom = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    cosine = torch.sum(ref * got) / (torch.linalg.vector_norm(got).clamp_min(1e-12) * denom)
    return {
        "shape_match": True,
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_rms": float((torch.sqrt(torch.mean(delta.square())) / denom).item()),
        "cosine": float(cosine.item()),
        "finite": bool(torch.isfinite(got).all().item()),
    }


def _call(module: torch.nn.Module, template: torch.Tensor) -> object:
    # Several H3 blocks intentionally use in-place residuals.  A fresh input
    # preserves the semantics of one independent inference call and prevents a
    # benchmark repetition from timing a progressively mutated tensor.
    return module(template.clone())


def _timed(
    function: Callable[[], object],
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> tuple[object, float, float]:
    result = None
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        result = function()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end) / repetitions
    extra = max(0, torch.cuda.max_memory_allocated(device) - baseline) / MIB
    return result, elapsed, extra


def _case_specs() -> dict[str, tuple[str, Callable[[], torch.nn.Module], tuple[int, ...], torch.dtype]]:
    # Decoder shapes are one transformer layer at the same 24-channel latent
    # and 32x64-head dimensions as H3.  448/256 and 832/480 are the production
    # smoke and medium resolutions; the full 36-layer decoder is not compiled
    # by default because its graph/cold-start cost is a separate product test.
    return {
        "video_resnet_128_t5_h64_w64": (
            "video_vae_encode_resnet",
            lambda: ResnetBlock3D(128),
            (1, 128, 5, 64, 64),
            torch.float16,
        ),
        "video_decoder_1layer_448_chunk": (
            "video_vae_decode_vit1",
            lambda: ViT3DDecoder(num_layers=1),
            (1, 24, 5, 16, 28),
            torch.float16,
        ),
        "video_decoder_1layer_832_chunk": (
            "video_vae_decode_vit1",
            lambda: ViT3DDecoder(num_layers=1),
            (1, 24, 5, 30, 52),
            torch.float16,
        ),
        "video_decoder_1layer_1mp_chunk": (
            "video_vae_decode_vit1",
            lambda: ViT3DDecoder(num_layers=1),
            (1, 24, 7, 48, 84),
            torch.float16,
        ),
        "video_decoder_full36_448_chunk": (
            "video_vae_decode_vit36",
            lambda: ViT3DDecoder(num_layers=36),
            (1, 24, 5, 16, 28),
            torch.float16,
        ),
        "video_decoder_full36_832_chunk": (
            "video_vae_decode_vit36",
            lambda: ViT3DDecoder(num_layers=36),
            (1, 24, 5, 30, 52),
            torch.float16,
        ),
        "video_decoder_full36_1mp_chunk": (
            "video_vae_decode_vit36",
            lambda: ViT3DDecoder(num_layers=36),
            (1, 24, 7, 48, 84),
            torch.float16,
        ),
        "audio_ampblock_512_len1000": (
            "audio_vae_bigvgan_ampblock",
            lambda: AMPBlock1(512),
            (1, 512, 1000),
            torch.float32,
        ),
        "audio_ampblock_32_len20000": (
            "audio_vae_bigvgan_ampblock",
            lambda: AMPBlock1(32),
            (1, 32, 20000),
            torch.float32,
        ),
        "audio_attn_projection_lat200": (
            "audio_vae_attn_projection",
            lambda: AttnProjection(2048, 32, num_heads=8),
            (1, 200, 2048),
            torch.float32,
        ),
        "audio_bigvgan_full_len200": (
            "audio_vae_bigvgan_decoder",
            lambda: BigVGAN(num_mels=2048, upsample_initial_channel=1024),
            (1, 2048, 200),
            torch.float32,
        ),
        "clipproj_like_mlp": (
            "clipproj_residual_like",
            lambda: torch.nn.Sequential(
                torch.nn.Linear(2560, 4096),
                torch.nn.GELU(),
                torch.nn.Linear(4096, 5120),
            ),
            (64, 2560),
            torch.float16,
        ),
    }


def run_case(
    name: str,
    spec: tuple[str, Callable[[], torch.nn.Module], tuple[int, ...], torch.dtype],
    device: torch.device,
    warmup: int,
    repetitions: int,
    min_free_mib: int,
    compile_mode: str,
) -> dict[str, object]:
    label, factory, shape, dtype = spec
    free_before, total = torch.cuda.mem_get_info(device)
    result: dict[str, object] = {
        "name": name,
        "label": label,
        "input_shape": list(shape),
        "input_dtype": str(dtype),
        "free_before_mib": free_before / MIB,
        "total_mib": total / MIB,
    }
    if free_before < (min_free_mib + 512) * MIB:
        result["skipped"] = f"free VRAM below safety guard ({min_free_mib}+512 MiB)"
        return result

    torch.manual_seed(20260825)
    module = factory()
    _make_finite(module, device, dtype)
    result["parameter_mib"] = sum(
        parameter.numel() * parameter.element_size() for parameter in module.parameters()
    ) / MIB
    generator = torch.Generator(device=device).manual_seed(20260825)
    template = torch.randn(shape, device=device, dtype=dtype, generator=generator)

    with torch.inference_mode():
        try:
            eager_reference = _call(module, template)
            result["eager_finite"] = _finite(eager_reference)
            _, eager_ms, eager_peak = _timed(
                lambda: _call(module, template), device, warmup, repetitions
            )
            result["eager"] = {"milliseconds": eager_ms, "peak_extra_mib": eager_peak}
        except Exception as error:
            result["eager_error"] = f"{type(error).__name__}: {error}"
            del module, template
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            return result

        compiled = None
        try:
            compile_start = time.perf_counter()
            compiled = torch.compile(module, mode=compile_mode, dynamic=False, fullgraph=False)
            # ``max-autotune`` enables CUDA Graphs on this torch build.  A
            # graph may return storage owned by the capture, so clone the
            # reference before the timed repetitions can overwrite it.
            compiled_reference = _call(compiled, template).clone()
            torch.cuda.synchronize(device)
            first_call_ms = (time.perf_counter() - compile_start) * 1000.0
            result["compile_first_call_ms"] = first_call_ms
            result["compiled_finite"] = _finite(compiled_reference)
            _, compiled_ms, compiled_peak = _timed(
                lambda: _call(compiled, template), device, warmup, repetitions
            )
            result["compiled"] = {
                "milliseconds": compiled_ms,
                "peak_extra_mib": compiled_peak,
                "speedup_vs_eager": eager_ms / compiled_ms if compiled_ms else None,
            }
            result["error_vs_eager"] = _error(eager_reference, compiled_reference)
            error_metrics = result["error_vs_eager"]
            result["numerically_qualified"] = bool(
                isinstance(error_metrics, dict)
                and error_metrics.get("shape_match")
                and error_metrics.get("finite")
                and float(error_metrics.get("relative_rms", float("inf"))) <= 1e-3
                and float(error_metrics.get("cosine", 0.0)) >= 0.999
            )
        except Exception as error:
            result["compile_error"] = f"{type(error).__name__}: {error}"
            result["numerically_qualified"] = False

    del compiled, eager_reference, module, template
    # Reset Dynamo's references before the next case.  This does not delete the
    # on-disk code cache, but releases the CUDA module/activation references.
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case", choices=sorted(_case_specs()), action="append")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument(
        "--mode",
        choices=("default", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise SystemExit(f"expected SM70, got sm_{props.major}{props.minor}")

    specs = _case_specs()
    names = args.case or [
        "video_resnet_128_t5_h64_w64",
        "video_decoder_1layer_448_chunk",
        "audio_ampblock_512_len1000",
        "audio_ampblock_32_len20000",
        "audio_attn_projection_lat200",
        "clipproj_like_mlp",
    ]
    report: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "checkpoint_loaded": False,
        "host_mmap": False,
        "compiler": {
            "backend": "torch.compile / TorchInductor",
            "mode": args.mode,
            "dynamic": False,
            "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
            "max_jobs": os.environ.get("MAX_JOBS"),
        },
        "safety": {
            "device": str(device),
            "min_free_mib": args.min_free_mib,
            "one_gpu_only": True,
            "model_root_requested": "/mnt/GALAX",
            "model_root_used": None,
        },
        "results": [],
    }

    for name in names:
        print(f"running {name}", flush=True)
        item = run_case(
            name,
            specs[name],
            device,
            args.warmup,
            args.repetitions,
            args.min_free_mib,
            args.mode,
        )
        report["results"].append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

    mode_suffix = args.mode.replace("-", "_")
    output = args.output or REPO_ROOT / "results" / f"h3_vae_compile_sm70_{mode_suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved report: {output}", flush=True)


if __name__ == "__main__":
    main()
