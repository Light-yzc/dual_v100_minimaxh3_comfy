"""Lifecycle bridge between Qwen, DiT sampling, and the asynchronous H3 VAE.

The loader remains opt-in.  Runtime code notifies this module only after both
Qwen ranks have cleared their CUDA payload; the sampler hook then overlaps a
capped VAE prefetch with DiT and materialises the deferred tail after sampling
has returned.  Keeping the finalisation outside individual model forwards is
important because one sigma can execute several conditioning chunks.
"""

from __future__ import annotations

import functools
import gc
import importlib
import logging
import os
import threading
from dataclasses import dataclass
from typing import Sequence

import torch

from .h3_async_vae import (
    AsyncVAEHandle,
    H3AsyncVAE,
    inspect_safetensors,
    load_h3_video_vae_async,
)


_TRUE = {"1", "true", "yes", "on"}
_SAMPLE_HOOK_MARKER = "_h3_async_vae_sample_hook"


def _enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE


@dataclass(frozen=True)
class _DITCycle:
    handle: AsyncVAEHandle
    generation: int
    sequence: int


class _ActiveVAERegistry:
    """Process-local lifecycle state for the VAE selected by the workflow."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.handle: AsyncVAEHandle | None = None
        self.generation = 0
        self.sequence = 0
        self.qwen_cleared = False
        self.cycle: _DITCycle | None = None

    def register(self, handle: AsyncVAEHandle) -> AsyncVAEHandle | None:
        with self.lock:
            if self.handle is handle:
                return None
            previous = self.handle
            self.handle = handle
            self.generation += 1
            self.qwen_cleared = False
            self.cycle = None
            return previous

    def prepare_for_qwen(self) -> tuple[AsyncVAEHandle | None, int]:
        with self.lock:
            if self.cycle is not None:
                raise RuntimeError("cannot start Qwen while an H3 DiT cycle is active")
            self.qwen_cleared = False
            return self.handle, self.generation

    def mark_qwen_cleared(self) -> bool:
        with self.lock:
            handle = self.handle
            if handle is None:
                return False
            if self.cycle is not None:
                raise RuntimeError("Qwen clear arrived during an active H3 DiT cycle")
            handle.mark_dit_ready()
            self.qwen_cleared = True
            return True

    def begin_dit(self) -> tuple[_DITCycle | None, bool]:
        with self.lock:
            if self.cycle is not None:
                return self.cycle, False
            handle = self.handle
            if handle is None or not self.qwen_cleared:
                return None, False
            self.sequence += 1
            cycle = _DITCycle(handle, self.generation, self.sequence)
            self.cycle = cycle
            self.qwen_cleared = False
            try:
                started = bool(handle.mark_denoising(start_prefetch=True))
            except BaseException:
                self.cycle = None
                raise
            return cycle, started

    def current_cycle(self) -> _DITCycle | None:
        with self.lock:
            return self.cycle

    def matches(self, cycle: _DITCycle) -> bool:
        with self.lock:
            return (
                self.cycle == cycle
                and self.handle is cycle.handle
                and self.generation == cycle.generation
            )

    def finish(self, cycle: _DITCycle) -> bool:
        with self.lock:
            if not self.matches(cycle):
                return False
            self.cycle = None
            return True

    def detach_cycle(self, cycle: _DITCycle) -> bool:
        with self.lock:
            if self.handle is not cycle.handle or self.generation != cycle.generation:
                return False
            self.handle = None
            self.generation += 1
            self.qwen_cleared = False
            self.cycle = None
            return True

    def clear(self) -> AsyncVAEHandle | None:
        with self.lock:
            handle = self.handle
            self.handle = None
            self.generation += 1
            self.qwen_cleared = False
            self.cycle = None
            return handle

    def snapshot(self) -> tuple[AsyncVAEHandle | None, dict[str, object]]:
        with self.lock:
            handle = self.handle
            lifecycle = {
                "active": handle is not None,
                "generation": self.generation,
                "qwen_cleared": self.qwen_cleared,
                "dit_active": self.cycle is not None,
                "dit_sequence": None if self.cycle is None else self.cycle.sequence,
            }
            return handle, lifecycle


_REGISTRY = _ActiveVAERegistry()
_HOOK_LOCK = threading.Lock()


def _cancel_and_release(handle: AsyncVAEHandle) -> None:
    """Best-effort cleanup which must never hide the original workflow error."""

    try:
        handle.cancel()
    except BaseException:
        logging.exception("[H3 async VAE] failed to cancel partial VAE load")
    try:
        handle.release_decoder(keep_encoder=False)
    except BaseException:
        logging.exception("[H3 async VAE] failed to release partial VAE payload")


def register_active_async_vae(handle: AsyncVAEHandle) -> None:
    """Make ``handle`` the lifecycle target and release any replaced payload."""

    previous = _REGISTRY.register(handle)
    if previous is not None:
        _cancel_and_release(previous)


def _validate_h3_vae_checkpoint(path: os.PathLike[str] | str) -> str:
    """Accept the two supported H3 video VAE layouts and name which one it is.

    Two layouts are production paths now:

    * all-FP16 -- the original checkpoint, loaded as ordinary tensors.
    * INT8-ConvRot -- I8 qdata plus F32 ``weight_scale`` plus a ``comfy_quant``
      JSON marker per quantized Linear, with an FP32 encoder and small
      parameters.  ``h3_async_vae_int8`` supplies the loader callables.

    Anything else fails closed rather than being materialised at a dtype the
    compute path does not expect.
    """
    specs, header = inspect_safetensors(path)
    checkpoint_metadata = header.get("metadata", {})
    if not isinstance(checkpoint_metadata, dict) or (
        "minimax_h3_video_vae" not in checkpoint_metadata
    ):
        raise ValueError("H3 async VAE requires a MiniMax H3 video VAE checkpoint")
    if not specs:
        raise ValueError("H3 async VAE checkpoint contains no tensors")

    if any(name.endswith(".comfy_quant") for name in specs):
        allowed = {torch.int8, torch.uint8, torch.float32, torch.bfloat16, torch.float16}
        unexpected = {
            spec.dtype for spec in specs.values() if spec.dtype not in allowed
        }
        if unexpected:
            raise ValueError(
                "H3 async VAE INT8-ConvRot checkpoint has unexpected dtypes: "
                f"{sorted(str(dtype) for dtype in unexpected)}"
            )
        return "int8_convrot"

    if any(spec.dtype != torch.float16 for spec in specs.values()):
        raise ValueError(
            "H3 async VAE supports an all-FP16 or an INT8-ConvRot checkpoint; "
            "this file is neither"
        )
    return "fp16"


def maybe_load_async_vae_facade(
    path: os.PathLike[str] | str,
    devices: Sequence[torch.device | str] | None = None,
    *,
    split: int | None = None,
    prefetch_limits: Sequence[int | None] | None = None,
) -> H3AsyncVAE | None:
    """Return and register the opt-in async facade, otherwise leave loading alone."""

    if not _enabled("H3_ASYNC_VAE_LOAD", False):
        return None
    checkpoint_format = _validate_h3_vae_checkpoint(path)
    facade = load_h3_video_vae_async(
        path,
        devices,
        split=split,
        prefetch_limits=prefetch_limits,
    )
    if facade is None:
        return None
    handle = getattr(facade, "async_handle", None)
    if handle is None:
        raise TypeError("H3 async VAE facade does not expose its lifecycle handle")
    # The loader node logs this.  Two routes with very different memory
    # profiles share one facade type, so the log has to name which one ran.
    handle.checkpoint_format = checkpoint_format
    register_active_async_vae(handle)
    return facade


def prepare_active_vae_for_qwen(*, keep_encoder: bool = True) -> bool:
    """Release the previous decoder before Qwen starts allocating its shards."""

    handle, generation = _REGISTRY.prepare_for_qwen()
    if handle is None:
        return False
    handle.prepare_for_qwen(keep_encoder=keep_encoder)
    current, lifecycle = _REGISTRY.snapshot()
    return current is handle and lifecycle["generation"] == generation


def notify_qwen_cleared() -> bool:
    """Open the DiT gate after both Qwen ranks have cleared and synchronized."""

    return _REGISTRY.mark_qwen_cleared()


def _begin_active_dit() -> tuple[_DITCycle | None, bool]:
    try:
        return _REGISTRY.begin_dit()
    except BaseException:
        handle, _ = _REGISTRY.snapshot()
        if handle is not None:
            _REGISTRY.clear()
            _cancel_and_release(handle)
        raise


def notify_dit_start() -> bool:
    """Start capped prefetch once, and only after :func:`notify_qwen_cleared`."""

    _cycle, started = _begin_active_dit()
    return started


def _release_dit_shards() -> None:
    """Return both DiT ranks' cached blocks before the decoder tail loads.

    ``gc.collect``/``empty_cache`` below only affect this process, so they free
    rank 0.  Rank 1 is a separate process whose allocator has to be told
    explicitly, and at 720p it holds ~14.5 GiB reserved -- more than enough to
    starve its half of a layer-MP decoder.

    Done here rather than relying on the DiT node's own post-sample hook: both
    hooks wrap ``KSAMPLER.sample``, and which one ends up outermost depends on
    the order ComfyUI happens to execute the two loader nodes in.  The graph
    does not constrain that order, so the release is invoked from the one place
    that is guaranteed to run immediately before the decoder is materialised.
    Releasing twice is harmless; releasing after the load would be useless.
    """
    try:
        from .h3_tp_runtime import active_runtime
    except ImportError:
        try:
            from custom_nodes.DualV100.h3_tp_runtime import active_runtime
        except ImportError:
            return
    runtime = active_runtime()
    if runtime is None:
        return
    try:
        runtime.release_cached_memory()
    except Exception:
        # Never fail a finished sample over cleanup.  The decode that follows
        # may still OOM, but the latent is already computed and saveable.
        logging.warning(
            "[H3 async VAE] DiT shard release before tail load failed; continuing",
            exc_info=True,
        )


def _settle_dit_cuda(handle: AsyncVAEHandle) -> None:
    """Make final-forward frees driver-visible before uncapped tail loading."""

    _release_dit_shards()
    gc.collect()
    if not torch.cuda.is_available():
        return
    seen: set[torch.device] = set()
    for raw_device in getattr(handle, "devices", ()):
        device = torch.device(raw_device)
        if device.type != "cuda" or device in seen:
            continue
        seen.add(device)
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def _abort_cycle(cycle: _DITCycle) -> None:
    _REGISTRY.detach_cycle(cycle)
    _cancel_and_release(cycle.handle)


def _finalize_cycle(cycle: _DITCycle) -> H3AsyncVAE | None:
    if not _REGISTRY.matches(cycle):
        return None
    _settle_dit_cuda(cycle.handle)
    facade = cycle.handle.finalize_tail()
    _REGISTRY.finish(cycle)
    return facade


def finalize_active_vae_after_dit() -> H3AsyncVAE | None:
    """Materialise the deferred tail after the complete sampler call returns."""

    cycle = _REGISTRY.current_cycle()
    if cycle is None:
        cycle, _started = _begin_active_dit()
    if cycle is None:
        return None
    try:
        return _finalize_cycle(cycle)
    except BaseException:
        _abort_cycle(cycle)
        raise


def active_async_vae_stats() -> dict[str, object]:
    """Return lifecycle state plus the loader's byte and memory ledger."""

    handle, lifecycle = _REGISTRY.snapshot()
    lifecycle["handle"] = None if handle is None else handle.stats()
    return lifecycle


get_active_vae_stats = active_async_vae_stats


def clear_active_async_vae() -> bool:
    """Unregister the active facade and release all of its CUDA payload."""

    handle = _REGISTRY.clear()
    if handle is None:
        return False
    _cancel_and_release(handle)
    return True


clear_active_vae = clear_active_async_vae


def install_turbo_sampler_hook() -> bool:
    """Idempotently wrap ``KSAMPLER.sample`` with the H3 VAE lifecycle."""

    samplers = importlib.import_module("comfy.samplers")
    sampler_class = samplers.KSAMPLER
    with _HOOK_LOCK:
        current = sampler_class.sample
        if getattr(current, _SAMPLE_HOOK_MARKER, False):
            return False

        @functools.wraps(current)
        def sample_with_async_vae(self, *args, **kwargs):
            cycle, _started = _begin_active_dit()
            try:
                result = current(self, *args, **kwargs)
            except BaseException:
                if cycle is not None:
                    _abort_cycle(cycle)
                raise
            if cycle is not None:
                try:
                    _finalize_cycle(cycle)
                except BaseException:
                    _abort_cycle(cycle)
                    raise
            return result

        setattr(sample_with_async_vae, _SAMPLE_HOOK_MARKER, True)
        sample_with_async_vae._h3_async_vae_original_sample = current
        sampler_class.sample = sample_with_async_vae
        return True


__all__ = [
    "active_async_vae_stats",
    "clear_active_async_vae",
    "clear_active_vae",
    "finalize_active_vae_after_dit",
    "get_active_vae_stats",
    "install_turbo_sampler_hook",
    "maybe_load_async_vae_facade",
    "notify_dit_start",
    "notify_qwen_cleared",
    "prepare_active_vae_for_qwen",
    "register_active_async_vae",
]
