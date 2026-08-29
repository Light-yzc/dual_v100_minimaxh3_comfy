"""INT8-ConvRot adapter for the asynchronous two-GPU H3 video VAE.

Why this exists
---------------
``h3_model_parallel.load_h3_video_vae_parallel`` loads the whole VAE when its
loader node runs, which in a ComfyUI graph is *before* the sampler.  The
weights therefore have to coexist with the DiT forward peak.  With the INT8
DiT that does not fit at 720p, and no decoder split changes it, because the
shortfall is a total-capacity problem rather than a balance problem:

    INT8 DiT 720p/243f measured peak reserved : r0 14912   r1 14528 MiB
    free at peak (16384 cap, ~400 ctx/NCCL)   : r0  1072   r1  1456 MiB
    total free at peak                        :       2528 MiB
    INT8 video VAE on disk                    :       3025 MiB
                                                -> short 497 MiB

``h3_async_vae`` already solves exactly this by deferring the decoder until the
sampler has returned, but its production entry validated an all-FP16
checkpoint and built the module tree with ``disable_weight_init``.  Neither
holds for the INT8 checkpoint, which is I8 qdata plus F32 scales plus a
``comfy_quant`` JSON marker per quantized Linear.

This module supplies the four callables ``AsyncVAEHandle`` already accepts --
``model_factory``, ``value_loader``, ``tensor_setter``, ``model_finalizer`` --
so the residency state machine, the bounded reader and the memory ledger are
reused unchanged.  With the decoder deferred the budget becomes:

    encoder (must coexist with the DiT peak) :  688 MiB  vs 1072 free  -> fits
    decoder (loaded after post-sample release): 2337 MiB  vs ~5.3 GiB   -> fits

Fails closed: an unexpected quantization format or a missing scale/marker
raises instead of silently materializing a wrong-dtype weight.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

try:
    from .h3_async_vae import BoundedSafeTensorReader, SafeTensorSpec, _CopyToken
except ImportError:
    from custom_nodes.DualV100.h3_async_vae import (
        BoundedSafeTensorReader,
        SafeTensorSpec,
        _CopyToken,
    )

MIB = 1024 * 1024

# Suffixes that are not standalone values but part of a quantized triplet.
_QUANT_MARKER_SUFFIX = ".comfy_quant"
_QUANT_SCALE_SUFFIX = ".weight_scale"

_TRUE = {"1", "true", "yes", "on"}


def _enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE


def is_int8_convrot_vae(specs: dict[str, SafeTensorSpec]) -> bool:
    """Detect the INT8-ConvRot layout from header names alone.

    Deliberately structural rather than filename-based: the production INT8
    checkpoint is reached through a symlink, and a name test would also accept
    a re-quantized file with a different internal layout.
    """
    return any(name.endswith(_QUANT_MARKER_SUFFIX) for name in specs)


def _read_marker_config(
    reader: BoundedSafeTensorReader,
    spec: SafeTensorSpec,
    device: torch.device,
) -> dict[str, Any]:
    """Read one ``comfy_quant`` JSON marker and parse it on the host.

    ``device`` must be one of the handle's two owner devices: the reader's
    staging rings and CUDA streams are keyed by exactly those, so passing
    ``cpu`` raises ``KeyError``.  The marker is a few dozen bytes, so the
    round trip to host for ``json.loads`` costs nothing and does not violate
    the bounded-memory contract that governs weight payloads.
    """
    value, token = reader.read(spec, device)
    token.wait()
    raw = value.detach().to(torch.uint8).cpu().numpy().tobytes()
    try:
        config = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"H3 INT8 VAE marker {spec.name!r} is not valid JSON"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(f"H3 INT8 VAE marker {spec.name!r} is not a JSON object")
    return config


class Int8AsyncVAEAdapter:
    """Bundle of injectable callables that teach the async handle INT8.

    One adapter instance belongs to one handle.  It owns the quantized-triplet
    bookkeeping that has to survive across ``_load_names`` calls, because the
    encoder pass, the capped DiT-overlap prefetch and the post-sample tail each
    invoke the loader separately and a weight may be read in a different pass
    than its scale.
    """

    def __init__(
        self,
        specs: dict[str, SafeTensorSpec],
        devices: Sequence[torch.device],
    ) -> None:
        self.specs = specs
        self.devices = tuple(torch.device(device) for device in devices)
        self._lock = threading.RLock()
        # weight name -> parsed marker config, populated lazily.  Markers are a
        # few dozen bytes of JSON and carry no device state, so unlike the
        # scales they are safe to cache across generations.
        self._marker_cache: dict[str, dict[str, Any]] = {}
        self.int8_count = 0
        self.ordinary_count = 0

        self._quantized_weights = {
            name[: -len(_QUANT_MARKER_SUFFIX)] + ".weight"
            for name in specs
            if name.endswith(_QUANT_MARKER_SUFFIX)
        }
        if not self._quantized_weights:
            raise ValueError("no INT8 quantized weights found in H3 VAE checkpoint")

    # ------------------------------------------------------------------
    # model_factory
    # ------------------------------------------------------------------
    def model_factory(self) -> nn.Module:
        """Build the VAE on meta with mixed-precision ops.

        ``disable_weight_init`` (the FP16 default) creates plain Linears that
        cannot host a ``QuantizedTensor``.  ``mixed_precision_ops`` installs the
        int8 dispatch the ConvRot weights need.
        """
        import comfy.ops
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        quant_ops = comfy.ops.mixed_precision_ops(
            {"int8_tensorwise": {}}, torch.float16
        )
        with torch.device("meta"):
            return MiniMaxH3VideoVAE(operations=quant_ops).eval()

    # ------------------------------------------------------------------
    # value_loader
    # ------------------------------------------------------------------
    def value_loader(
        self,
        reader: BoundedSafeTensorReader,
        spec: SafeTensorSpec,
        owner: torch.device,
    ) -> tuple[torch.Tensor, _CopyToken]:
        """Stream one checkpoint value to its owner, honouring INT8 triplets.

        Markers and scales are consumed as metadata for their weight rather
        than being set on the module, so they are returned as tiny CPU
        placeholders; ``tensor_setter`` drops them.
        """
        name = spec.name

        if name.endswith(_QUANT_MARKER_SUFFIX):
            weight_name = name[: -len(_QUANT_MARKER_SUFFIX)] + ".weight"
            config = _read_marker_config(reader, spec, owner)
            fmt = config.get("format")
            if fmt != "int8_tensorwise":
                raise ValueError(
                    f"unsupported H3 VAE quantization for {weight_name}: {config}"
                )
            with self._lock:
                self._marker_cache[weight_name] = config
            return torch.empty(0, device="cpu"), _CopyToken()

        if name.endswith(_QUANT_SCALE_SUFFIX):
            # Skipped rather than stashed.  A scale is device-bound state, and
            # keeping it across passes would survive ``release_decoder``: if the
            # split changed between generations the stale tensor would land on
            # the wrong card.  ``_load_quantized_weight`` re-reads it on demand
            # instead -- 4 bytes per Linear, 144 total.
            return torch.empty(0, device="cpu"), _CopyToken()

        if name in self._quantized_weights:
            return self._load_quantized_weight(reader, spec, owner)

        # Ordinary parameter/buffer.  Normalization constants stay F32; the VAE
        # casts them at their use sites.  Everything else computes in FP16.
        target_dtype = (
            torch.float32
            if name in {"latents_mean", "latents_std"}
            else torch.float16 if spec.dtype.is_floating_point else spec.dtype
        )
        value, token = reader.read(spec, owner)
        if value.dtype != target_dtype:
            token.wait()
            value = value.to(dtype=target_dtype)
            token = _CopyToken()
        with self._lock:
            self.ordinary_count += 1
        return value, token

    def _load_quantized_weight(
        self,
        reader: BoundedSafeTensorReader,
        spec: SafeTensorSpec,
        owner: torch.device,
    ) -> tuple[torch.Tensor, _CopyToken]:
        """Assemble one ``QuantizedTensor`` from qdata plus its scale and marker.

        The scale and the marker are always read here rather than being picked
        up from an earlier pass.  In this checkpoint every ``weight_scale``
        precedes its ``weight`` in file order (verified: 144/144), so the scale
        has normally already been visited and skipped by the time the qdata
        arrives; reading it again costs one 4-byte pread per Linear and keeps
        the loader free of cross-pass device state.
        """
        from comfy.quant_ops import QuantizedTensor, get_layout_class

        name = spec.name
        base = name[: -len(".weight")]

        with self._lock:
            config = self._marker_cache.get(name)
        if config is None:
            marker_spec = self.specs.get(base + _QUANT_MARKER_SUFFIX)
            if marker_spec is None:
                raise ValueError(f"H3 INT8 VAE weight {name} has no comfy_quant marker")
            config = _read_marker_config(reader, marker_spec, owner)
            if config.get("format") != "int8_tensorwise":
                raise ValueError(
                    f"unsupported H3 VAE quantization for {name}: {config}"
                )
            with self._lock:
                self._marker_cache[name] = config

        scale_spec = self.specs.get(base + _QUANT_SCALE_SUFFIX)
        if scale_spec is None:
            raise ValueError(f"H3 INT8 VAE weight {name} has no weight_scale")
        scale, scale_token = reader.read(scale_spec, owner)
        scale_token.wait()
        if scale.dtype != torch.float32:
            scale = scale.to(dtype=torch.float32)

        qdata, token = reader.read(spec, owner)
        token.wait()
        if qdata.dtype != torch.int8:
            qdata = qdata.view(torch.int8)

        layout = get_layout_class("TensorWiseINT8Layout")
        params = layout.Params(
            scale=scale,
            orig_dtype=torch.float16,
            orig_shape=tuple(spec.shape),
            is_weight=True,
            convrot=bool(config.get("convrot", False)),
            convrot_groupsize=int(config.get("convrot_groupsize", 256)),
        )
        with self._lock:
            self.int8_count += 1
        return (
            QuantizedTensor(qdata, "TensorWiseINT8Layout", params),
            _CopyToken(),
        )

    # ------------------------------------------------------------------
    # tensor_setter
    # ------------------------------------------------------------------
    def tensor_setter(
        self,
        model: nn.Module,
        name: str,
        value: torch.Tensor,
    ) -> None:
        """Attach a value, skipping the marker/scale placeholders.

        A ``QuantizedTensor`` cannot go through ``nn.Parameter``, so quantized
        weights use ``set_attr_param`` and additionally carry the per-module
        dispatch flags the int8 Linear path reads.
        """
        if name.endswith(_QUANT_MARKER_SUFFIX) or name.endswith(_QUANT_SCALE_SUFFIX):
            return

        import comfy.utils

        if name in self._quantized_weights:
            comfy.utils.set_attr_param(model, name, value)
            parent, _leaf = comfy.utils.resolve_attr(model, name)
            parent.quant_format = "int8_tensorwise"
            parent.layout_type = "TensorWiseINT8Layout"
            parent._full_precision_mm = False
            return

        buffer_names = getattr(model, "_h3_async_int8_buffer_names", None)
        if buffer_names is None:
            buffer_names = {key for key, _ in model.named_buffers()}
            model._h3_async_int8_buffer_names = buffer_names
        if name in buffer_names:
            comfy.utils.set_attr_buffer(model, name, value)
        else:
            comfy.utils.set_attr_param(model, name, value)

    # ------------------------------------------------------------------
    # model_finalizer
    # ------------------------------------------------------------------
    def model_finalizer(
        self,
        model: nn.Module,
        devices: Sequence[torch.device],
        split: int,
    ) -> nn.Module:
        """Install the V100 INT8 compute path and the layer-MP decoder.

        Mirrors the resident loader's compute tail -- SM70 W8A16 fallback, the
        batched tiled-decode wrapper, then the MP decoder -- so both routes
        produce numerically identical modules.

        It deliberately does *not* register an ``H3VAELayoutManager``.  That
        manager exists to move decoder blocks off cuda:0 before sampling and
        back before decode, which only matters when the decoder is resident
        during the DiT forward.  Here it is not: the tail runs after the sampler
        has returned, so there is no sampling layout to restore and the blocks
        can be placed in their decode-optimal positions immediately.  Adding one
        would register a rebalance that never fires, since the async facade's
        ``decode`` has no ``ensure_decode`` call.
        """
        from .h3_model_parallel import (
            H3ParallelViTDecoder,
            _install_h3_v100_int8_tile_batch,
            _install_h3_v100_int8_w8a16,
            _install_h3_vae_mp_pipeline,
        )

        first, second = self.devices
        source_decoder = model.decoder

        w8a16_count = 0
        backend = "comfy_kitchen_int8"
        sm70_pair = all(
            torch.cuda.get_device_capability(device) == (7, 0)
            for device in self.devices
        )
        if sm70_pair and _enabled("H3_VAE_INT8_SM70_W8A16", True):
            w8a16_count = _install_h3_v100_int8_w8a16(source_decoder)
            if w8a16_count:
                backend = "sm70_convrot_w8a16_fp16_tensorcore"
                logging.info(
                    "[H3 async VAE INT8] SM70 W8A16 installed on %d decoder Linears",
                    w8a16_count,
                )

        try:
            tile_batch = max(1, int(os.environ.get("H3_VAE_INT8_TILE_BATCH", "1")))
        except ValueError as exc:
            raise ValueError("H3_VAE_INT8_TILE_BATCH must be a positive integer") from exc
        tile_batch = _install_h3_v100_int8_tile_batch(model, tile_batch)

        model.decoder = H3ParallelViTDecoder(source_decoder, first, second, split)
        pipeline_report = _install_h3_vae_mp_pipeline(model, self.devices)
        model.eval()

        for device in self.devices:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.synchronize(device)

        model._h3_parallel_report = {
            "route": "async_int8_convrot",
            "split": split,
            "decoder_blocks": len(source_decoder.transformer_blocks),
            "quantized_linear_tensors": self.int8_count,
            "ordinary_tensors": self.ordinary_count,
            "quant_format": "int8_tensorwise + convrot",
            "compute_dtype": "torch.float16",
            "int8_compute_backend": backend,
            "sm70_w8a16_linears": w8a16_count,
            "int8_spatial_tile_batch": tile_batch,
            "mp_tile_pipeline": pipeline_report,
            "layout_manager": None,
        }
        logging.info(
            "[H3 async VAE INT8] finalized: %d int8 Linears, %d ordinary, "
            "split=%d backend=%s tile_batch=%d (no stage rebalance: decoder "
            "loads directly in its decode layout)",
            self.int8_count,
            self.ordinary_count,
            split,
            backend,
            tile_batch,
        )
        return model


def build_int8_async_kwargs(
    specs: dict[str, SafeTensorSpec],
    devices: Sequence[torch.device],
) -> dict[str, Callable[..., Any]]:
    """Return the ``AsyncVAEHandle`` kwargs that enable the INT8 route."""
    adapter = Int8AsyncVAEAdapter(specs, devices)
    return {
        "model_factory": adapter.model_factory,
        "value_loader": adapter.value_loader,
        "tensor_setter": adapter.tensor_setter,
        "model_finalizer": adapter.model_finalizer,
    }


__all__ = [
    "Int8AsyncVAEAdapter",
    "build_int8_async_kwargs",
    "is_int8_convrot_vae",
]
