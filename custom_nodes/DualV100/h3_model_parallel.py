"""Resident two-GPU model-parallel paths for the H3 VAE and Qwen3-VL-4B.

These paths are deliberately layer/pipeline parallel rather than pretending
that a generic ``device_map`` is tensor parallel.  The two models are
sequential networks: a VAE decoder and a Qwen language stack.  Keeping their
weights resident and transferring the activation once at the layer boundary
is the useful operation on the dual V100/NVLink machine.

The loader uses the deployment's header-only safetensors reader.  Parameters
are assigned as file-backed meta tensors, then materialised one tensor at a
time directly on their owning GPU.  No full CPU state dict and no mmap of a
multi-gigabyte checkpoint is created.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
import types
import weakref
from pathlib import Path

import torch
import torch.nn as nn


_TRUE = {"1", "true", "yes", "on"}
_VAE_CACHE: dict[tuple[str, str, str, int], object] = {}
# Decode wants the heavier half on cuda:0; DiT sampling wants cuda:0 clear.
# These are the two ends of the stage rebalance, not competing defaults.
_DEFAULT_VAE_SPLIT = 24
_DEFAULT_VAE_DIT_SPLIT = 18
# Set by the loader so the DiT runtime can restore the sampling layout without
# taking a dependency on whichever node happens to hold the VAE object.
_LAYOUT_MANAGERS: "weakref.WeakSet" = weakref.WeakSet()
_LAYOUT_LOCK = threading.Lock()


def register_layout_manager(manager) -> None:
    with _LAYOUT_LOCK:
        _LAYOUT_MANAGERS.add(manager)


def ensure_sampling_layout() -> list[dict]:
    """Restore every resident VAE to its sampling layout.

    Called from the DiT runtime before it allocates, so a previous decode's
    24/12 layout cannot strand ~1.5 GiB on cuda:0 and reproduce the rank-0 OOM.
    """
    with _LAYOUT_LOCK:
        managers = list(_LAYOUT_MANAGERS)
    reports = []
    for manager in managers:
        # Skip before touching CUDA at all: the common case is that the layout
        # already matches, and this runs on every sampler entry.
        if manager.current_split == manager.dit_split:
            continue
        try:
            manager.ensure_sampling()
            reports.append(manager.report())
        except Exception:
            logging.exception("[H3 VAE MP] failed to restore the sampling layout")
    return reports


def _enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE


def _cuda_devices(devices=None) -> tuple[torch.device, torch.device] | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None
    if devices is None:
        devices = ("cuda:0", "cuda:1")
    if len(devices) != 2:
        raise ValueError("H3 model parallel requires exactly two devices")
    result = tuple(torch.device(x) for x in devices)
    if any(x.type != "cuda" for x in result):
        raise ValueError("H3 model parallel devices must be CUDA devices")
    if result[0] == result[1]:
        raise ValueError("H3 model parallel devices must be different")
    if not torch.cuda.can_device_access_peer(result[0].index, result[1].index):
        raise RuntimeError(
            f"H3 model parallel requires CUDA P2P/NVLink: {result[0]} -> {result[1]}"
        )
    if not torch.cuda.can_device_access_peer(result[1].index, result[0].index):
        raise RuntimeError(
            f"H3 model parallel requires CUDA P2P/NVLink: {result[1]} -> {result[0]}"
        )
    return result


def _resolve_vae_output_device(devices: tuple[torch.device, torch.device]) -> torch.device:
    """Resolve the VAE output buffer without inheriting a mutable global device.

    ComfyUI normally keeps intermediate tensors on the CPU.  MultiGPU and
    ``--gpu-only`` can change that global policy while a graph is executing,
    though, and the H3 decoder's final temporary is large enough that an
    accidental GPU1 output buffer can exhaust the DiT rank-1 card.  Keep the
    production default explicit and allow a deliberate per-device override
    for small, isolated VAE benchmarks.
    """

    raw = os.environ.get("H3_VAE_OUTPUT_DEVICE", "cpu").strip().lower()
    if raw in {"", "cpu", "host"}:
        return torch.device("cpu")
    if raw == "auto":
        # ``auto`` intentionally means the safe host buffer, not
        # ``mm.intermediate_device()``; the latter may point at GPU1 after a
        # MultiGPU node changed its current-device guard.
        return torch.device("cpu")
    try:
        target = torch.device(raw)
    except RuntimeError as exc:
        raise ValueError(
            "H3_VAE_OUTPUT_DEVICE must be cpu, auto, or one of the MP CUDA devices"
        ) from exc
    if target not in devices:
        raise ValueError(
            "H3_VAE_OUTPUT_DEVICE must be cpu/auto or one of "
            f"{devices[0]}, {devices[1]}; got {raw!r}"
        )
    return target


def _resolve_vae_split_request(split) -> int:
    """Normalize the deployment split request before constructing the cache key."""

    requested = split
    if requested is None:
        requested = os.environ.get("H3_VAE_SPLIT", str(_DEFAULT_VAE_SPLIT))
    if isinstance(requested, str):
        value = requested.strip().lower()
        if value in {"auto", "balanced", "default"}:
            # The service loads the VAE before the DiT ranks.  A live free-
            # memory heuristic at that point sees two empty cards and picks
            # 18/18, but rank1 later retains the heavier NCCL worker.  24/12
            # is the empirically safe balanced split for this workload; users
            # can still select any explicit layer boundary.
            return _DEFAULT_VAE_SPLIT
        requested = value
    try:
        return max(1, int(requested))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "H3_VAE_SPLIT must be an integer, auto, balanced, or default"
        ) from exc


def _device_list_from_env(name: str = "H3_MP_DEVICES") -> tuple[torch.device, torch.device]:
    raw = os.environ.get(name, "cuda:0,cuda:1")
    values = tuple(x.strip() for x in raw.split(",") if x.strip())
    if len(values) != 2:
        raise ValueError(f"{name} must contain two comma-separated devices")
    return tuple(torch.device(x) for x in values)  # type: ignore[return-value]


def _cuda_context(device: torch.device):
    if device.type != "cuda":
        return None
    try:
        import comfy.model_management as mm

        return mm.cuda_device_context(device)
    except Exception:
        return torch.cuda.device(device)


def _move_module(module: nn.Module, device: torch.device) -> None:
    """Move a fully materialised module without relying on the current card."""
    context = _cuda_context(device)
    if context is None:
        module.to(device)
    else:
        with context:
            module.to(device)


def _materialize(module: nn.Module, device: torch.device) -> int:
    """Materialise file-backed parameters under *module* on *device*."""
    try:
        from custom_nodes.NoHostMMap.safetensors import _materialize_file_backed_model

        count = int(_materialize_file_backed_model(module, device))
    except ImportError:
        count = 0
    _move_module(module, device)
    return count


def _materialize_direct(parent: nn.Module, name: str, device: torch.device, buffer=False) -> int:
    value = getattr(parent, name)
    if not isinstance(value, torch.Tensor):
        return 0
    if getattr(value, "is_meta", False):
        try:
            from custom_nodes.NoHostMMap.safetensors import _file_slice_tensor_to_device
            import comfy.utils

            loaded = _file_slice_tensor_to_device(value, device, value.dtype)
            if loaded is None:
                raise RuntimeError(f"{parent.__class__.__name__}.{name} is meta without a file slice")
            if buffer:
                comfy.utils.set_attr_buffer(parent, name, loaded)
            else:
                comfy.utils.set_attr_param(parent, name, loaded)
            return 1
        except ImportError:
            pass
    if value.device != device:
        if buffer:
            parent._buffers[name] = value.to(device)
        else:
            parent._parameters[name] = nn.Parameter(value.to(device), requires_grad=value.requires_grad)
    return 0


def _direct_tensor_to_device(value: torch.Tensor, device: torch.device, dtype=None) -> torch.Tensor:
    """Read one checkpoint tensor straight to its final device.

    The H3 INT8 VAE contains ordinary F32 tensors as well as raw INT8 weights
    and per-row scales.  The regular ``load_state_dict`` path cannot consume
    the latter as Parameters, and converting the whole state dict to CPU would
    defeat the no-host-mmap memory bound.  NoHostMMap gives meta tensors a
    file-slice descriptor; consume that descriptor here one tensor at a time.
    """
    if getattr(value, "is_meta", False):
        from custom_nodes.NoHostMMap.safetensors import _file_slice_tensor_to_device

        loaded = _file_slice_tensor_to_device(value, device, dtype)
        if loaded is None:
            raise RuntimeError(
                f"file-backed tensor has no slice: shape={tuple(value.shape)} "
                f"dtype={value.dtype}"
            )
        return loaded

    loaded = value.to(device=device)
    if dtype is not None and loaded.dtype != dtype:
        loaded = loaded.to(dtype=dtype)
    return loaded


def _h3_int8_vae_state_dict(state_dict) -> bool:
    """Return whether a state dict is Kijai/Comfy INT8-ConvRot H3 VAE."""
    return any(
        key.startswith("decoder.transformer_blocks.")
        and key.endswith(".comfy_quant")
        for key in state_dict
    )


def _h3_v100_int8_w8a16_forward(self, input, *args, **kwargs):
    """SM70 weight-only INT8 fallback: ConvRot + FP16 Tensor Core GEMM.

    Comfy-Kitchen's native CUDA INT8 Linear requires SM75 and its Triton
    implementation requires SM80.  V100 therefore falls back to a generic
    path which rotates and dynamically quantizes every activation, performs an
    INT32 matmul, then dequantizes the output.  That path is substantially
    slower than FP16 Tensor Cores for the H3 decoder's large-M matrices.

    Keep the checkpoint weight resident as INT8.  For one Linear call, rotate
    the FP16 activation exactly as ConvRot requires, dequantize only that
    layer's weight into a bounded temporary, and run the GEMM on FP16 Tensor
    Cores.  The temporary dies after the call, so resident model storage stays
    quantized and peak scratch is bounded by the largest single matrix.
    """
    from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

    weight = self.weight
    if (
        input.dtype != torch.float16
        or not isinstance(weight, QuantizedTensor)
        or weight._layout_cls != "TensorWiseINT8Layout"
        or getattr(weight._params, "transposed", False)
        or len(getattr(self, "weight_function", ())) > 0
        or len(getattr(self, "bias_function", ())) > 0
    ):
        return self._h3_v100_int8_original_forward(input, *args, **kwargs)

    qdata, _scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    if qdata.device != input.device:
        return self._h3_v100_int8_original_forward(input, *args, **kwargs)

    params = weight._params
    if getattr(params, "convrot", False):
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

        group_size = int(getattr(params, "convrot_groupsize", 256))
        hadamard = _build_hadamard(
            group_size, device=input.device, dtype=input.dtype
        )
        input = _rotate_activation(input, hadamard, group_size)

    dequant_weight = torch.empty(
        qdata.shape, device=qdata.device, dtype=input.dtype
    )
    scale_fp16 = self._h3_v100_int8_scale_fp16
    torch.mul(qdata, scale_fp16, out=dequant_weight)

    bias = self.bias
    if bias is not None and (bias.device != input.device or bias.dtype != input.dtype):
        bias = bias.to(device=input.device, dtype=input.dtype)
    return torch.nn.functional.linear(input, dequant_weight, bias)


def _install_h3_v100_int8_w8a16(decoder: nn.Module) -> int:
    """Install the bounded W8A16 path on quantized decoder Linears."""
    from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

    count = 0
    for module in decoder.modules():
        weight = getattr(module, "weight", None)
        if (
            not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != "TensorWiseINT8Layout"
        ):
            continue
        if hasattr(module, "_h3_v100_int8_original_forward"):
            continue

        _qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        module._h3_v100_int8_scale_fp16 = scale.to(
            device=_qdata.device, dtype=torch.float16
        )
        module._h3_v100_int8_original_forward = module.forward
        module.forward = types.MethodType(_h3_v100_int8_w8a16_forward, module)
        count += 1
    return count


def _h3_v100_int8_batched_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
    """Decode several independent spatial tiles in one model batch.

    The stock H3 VAE visits all 36 transformer blocks once per spatial tile.
    On V100 our bounded W8A16 path therefore dequantizes the same 144 weights
    again for every tile. Grouping adjacent tiles on the batch dimension
    preserves attention isolation and the original overlap/blend order while
    amortizing each temporary weight conversion across multiple tiles.

    Only one small tile group is live at a time. This deliberately avoids a
    full-frame activation batch and keeps the quantized resident-weight saving.
    """
    tile_batch = max(1, int(self._h3_v100_int8_tile_batch))
    if tile_batch == 1:
        return self._h3_v100_int8_original_tiled_decode(z)

    height = int(z.shape[-2]) * int(self.vae_ratio)
    width = int(z.shape[-1]) * int(self.vae_ratio)
    y_idx, y_len, y_overlap = self.split_tiles(height)
    x_idx, x_len, x_overlap = self.split_tiles(width)
    input_batch = int(z.shape[0])

    canvas = None
    row_tails = []
    out_y = 0
    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
        zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
        new_tails = []
        left_tail = None
        out_x = 0

        for group_start in range(0, len(x_idx), tile_batch):
            group_end = min(group_start + tile_batch, len(x_idx))
            latent_tiles = []
            for j in range(group_start, group_end):
                zj = x_idx[j] // self.vae_ratio
                zw = x_len[j] // self.vae_ratio
                latent_tiles.append(z[..., zi : zi + zl, zj : zj + zw])

            if len(latent_tiles) == 1:
                latent_group = latent_tiles[0]
            else:
                latent_group = torch.cat(latent_tiles, dim=0)
            decoded_group = self._decode_pixels(latent_group)

            for local_index, j in enumerate(range(group_start, group_end)):
                batch_start = local_index * input_batch
                batch_end = batch_start + input_batch
                tile = decoded_group[batch_start:batch_end]

                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_overlap[i] :, :].clone())
                next_left_tail = (
                    tile[..., :, -x_overlap[j] :].clone()
                    if j < len(x_idx) - 1
                    else None
                )
                if i > 0:
                    tile = self.blend(
                        row_tails[j], tile, y_overlap[i - 1], dim=-2
                    )
                if j > 0:
                    tile = self.blend(
                        left_tail, tile, x_overlap[j - 1], dim=-1
                    )
                left_tail = next_left_tail
                if i < len(y_idx) - 1:
                    tile = tile[..., :-y_overlap[i], :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-x_overlap[j]]
                if canvas is None:
                    canvas = torch.empty(
                        *tile.shape[:-2],
                        height,
                        width,
                        dtype=tile.dtype,
                        device=tile.device,
                    )
                canvas[
                    ..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]
                ].copy_(tile)
                out_x += int(tile.shape[-1])

            del decoded_group, latent_group, latent_tiles

        row_tails = new_tails
        out_y += int(tile.shape[-2])

    if canvas is None:
        raise RuntimeError("H3 INT8 tiled decode produced no spatial tiles")
    return canvas


def _install_h3_v100_int8_tile_batch(model: nn.Module, tile_batch: int) -> int:
    """Install the bounded spatial tile batching path and return its size."""
    tile_batch = max(1, int(tile_batch))
    if tile_batch == 1:
        return 1
    if hasattr(model, "_h3_v100_int8_original_tiled_decode"):
        model._h3_v100_int8_tile_batch = tile_batch
        return tile_batch
    model._h3_v100_int8_tile_batch = tile_batch
    model._h3_v100_int8_original_tiled_decode = model.tiled_decode
    model.tiled_decode = types.MethodType(_h3_v100_int8_batched_tiled_decode, model)
    return tile_batch


def _h3_vae_owner(key: str, devices, split: int) -> torch.device:
    """Choose the permanent owner for one H3 VAE checkpoint tensor."""
    first, second = devices
    if key.startswith("decoder.transformer_blocks."):
        block_index = int(key.split(".")[2])
        return first if block_index < split else second
    if key.startswith("decoder.norm_out.") or key.startswith("decoder.proj_out."):
        return second
    return first


def _parameter_storage_bytes(parameter: torch.Tensor) -> int:
    """Count raw resident bytes, not a QuantizedTensor's logical FP16 size."""
    qdata = getattr(parameter, "_qdata", None)
    if isinstance(qdata, torch.Tensor):
        total = qdata.numel() * qdata.element_size()
        scale = getattr(getattr(parameter, "_params", None), "scale", None)
        if isinstance(scale, torch.Tensor):
            total += scale.numel() * scale.element_size()
        return int(total)
    return int(parameter.numel() * parameter.element_size())


def _first_parameter_device(module: nn.Module) -> torch.device | None:
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return None


def _move_freqs(freqs, device: torch.device):
    if isinstance(freqs, (tuple, list)):
        return type(freqs)(_move_freqs(x, device) for x in freqs)
    return freqs.to(device) if torch.is_tensor(freqs) and freqs.device != device else freqs


class H3ParallelViTDecoder(nn.Module):
    """Run the existing H3 ViT decoder with a single activation handoff."""

    def __init__(self, source: nn.Module, first_device: torch.device, second_device: torch.device, split: int):
        super().__init__()
        self.source = source
        self.first_device = first_device
        self.second_device = second_device
        self.split = split
        self.out_channels = source.out_channels
        self.patch_size = source.patch_size
        self.patch_size_t = source.patch_size_t
        self._h3_parallel = True

    def stage_first(self, x):
        """Run embedding plus the blocks owned by the first card.

        Split out of ``forward`` so an outer scheduler can overlap this stage
        for tile ``i+1`` with :meth:`stage_second` for tile ``i``.  The op
        sequence is identical to the fused path, so both produce the same
        values bit for bit.
        """
        d = self.source
        first = self.first_device
        if x.device != first:
            x = x.to(first)
        b, _, latent_t, latent_h, latent_w = x.shape

        h = d.x_embedder(x.flatten(2).transpose(1, 2))
        num_patches = h.shape[1]
        num_suffix = 1 + d.num_register_tokens
        h = torch.cat(
            [
                h,
                d.register_tokens.to(device=first, dtype=h.dtype).expand(b, -1, -1),
                torch.zeros_like(h[:, 0:1, :]),
            ],
            dim=1,
        )

        # The position table is small and is intentionally generated on the
        # first card.  It is copied only once when the block split is crossed.
        from comfy.ldm.minimax.vae import create_token_ids

        img_ids = create_token_ids((latent_t, latent_h, latent_w), x.device, x.dtype).expand(b, -1, -1)
        suffix_ids = torch.zeros((b, num_suffix, 3), device=first, dtype=img_ids.dtype)
        img_ids = torch.cat([img_ids, suffix_ids], dim=1)
        rotary_pos_emb = d.pos_embed(img_ids)

        for block in d.transformer_blocks[: self.split]:
            h = block(h, rotary_pos_emb)

        shape = (b, latent_t, latent_h, latent_w, num_patches)
        return h, rotary_pos_emb, shape

    def stage_second(self, h, rotary_pos_emb, shape):
        """Run the blocks owned by the second card and unpatchify."""
        d = self.source
        second = self.second_device
        b, latent_t, latent_h, latent_w, num_patches = shape

        if h.device != second:
            h = h.to(second)
        rotary_pos_emb = _move_freqs(rotary_pos_emb, second)

        for block in d.transformer_blocks[self.split :]:
            h = block(h, rotary_pos_emb)

        h = d.proj_out(d.norm_out(h))
        output = h[:, :num_patches, :]
        output = output.view(
            b, latent_t, latent_h, latent_w,
            self.out_channels, self.patch_size_t, self.patch_size, self.patch_size,
        )
        # Keep the temporal patch axis separate until the final reshape.  The
        # source H3 decoder has eight axes here:
        # [B, latent_T, latent_H, latent_W, C, patch_T, patch_H, patch_W].
        # Dropping the final patch-W axis silently produced a seven-axis
        # permute error on the first real decode.
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return output.reshape(
            b, self.out_channels,
            latent_t * self.patch_size_t,
            latent_h * self.patch_size,
            latent_w * self.patch_size,
        )

    def forward(self, x):
        h, rotary_pos_emb, shape = self.stage_first(x)
        return self.stage_second(h, rotary_pos_emb, shape)


_SAMPLER_HOOK_MARKER = "_h3_vae_layout_presample"
_SAMPLER_HOOK_LOCK = threading.Lock()


def install_presample_layout_hook() -> bool:
    """Restore the VAE sampling layout at the sampler entry, before activations.

    Placement matters more than it looks.  The rebalance has to complete while
    cuda:0 is still empty of DiT activations, so it cannot go in the TP
    ``forward`` (the residual is already resident by then) and it cannot go in
    ``ensure_started`` (that returns early from the second request onward).
    ``KSAMPLER.sample`` runs before the first denoise step allocates anything
    and is common to every H3 route, so a request that decoded at 24/12 is back
    at the sampling layout before the next sampler touches the card.

    Deliberately independent of the optional modules: no TE-Speed, Group Cache,
    async-VAE or Qwen route is imported or assumed here.  When no layer-MP VAE
    is resident the hook costs one empty set iteration.
    """
    try:
        samplers = __import__("comfy.samplers", fromlist=["KSAMPLER"])
    except Exception:
        logging.debug("[H3 VAE MP] comfy.samplers unavailable; pre-sample hook skipped")
        return False

    sampler_class = getattr(samplers, "KSAMPLER", None)
    if sampler_class is None:
        return False

    with _SAMPLER_HOOK_LOCK:
        current = sampler_class.sample
        if getattr(current, _SAMPLER_HOOK_MARKER, False):
            return False

        import functools

        @functools.wraps(current)
        def sample_with_sampling_layout(self, *args, **kwargs):
            try:
                ensure_sampling_layout()
            except Exception:
                # Never block sampling on the rebalance: staying in the decode
                # layout may OOM later, but failing here would break routes
                # that have no layer-MP VAE at all.
                logging.exception(
                    "[H3 VAE MP] pre-sample layout restore failed; continuing"
                )
            return current(self, *args, **kwargs)

        setattr(sample_with_sampling_layout, _SAMPLER_HOOK_MARKER, True)
        sample_with_sampling_layout._h3_vae_layout_original_sample = current
        sampler_class.sample = sample_with_sampling_layout
        logging.info(
            "[H3 VAE MP] pre-sample layout hook installed on KSAMPLER.sample"
        )
        return True


def _resolve_vae_stage_splits() -> tuple[int, int]:
    """Resolve the (sampling, decode) decoder splits.

    The two stages have opposite memory pressure and therefore want different
    layouts:

    * During DiT sampling the rank-0 backbone needs every spare byte on cuda:0.
      A decode-optimal 24/12 VAE layout parks ~3.4 GiB (FP16) of idle decoder
      weights there and is what pushed the 1280x736 QKV projection into OOM.
    * During decode the layer-MP decoder is serial, so a balanced 18/18 split
      leaves cuda:1 holding half the blocks while cuda:0 is otherwise free.

    ``H3_VAE_SPLIT`` remains the single-layout override for callers that do not
    want the rebalance at all; it seeds both stages when the stage-specific
    variables are unset.
    """
    base = os.environ.get("H3_VAE_SPLIT")
    dit_raw = os.environ.get("H3_VAE_DIT_SPLIT")
    decode_raw = os.environ.get("H3_VAE_DECODE_SPLIT")

    def parse(raw, default: int, name: str) -> int:
        if raw is None:
            return default
        value = str(raw).strip().lower()
        if value in {"auto", "balanced", "default"}:
            return default
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an integer, auto, balanced, or default"
            ) from exc
        if not 1 <= parsed <= 35:
            raise ValueError(f"{name} must be in [1, 35]; got {parsed}")
        return parsed

    seeded = _resolve_vae_split_request(base) if base is not None else None
    dit_split = parse(dit_raw, _DEFAULT_VAE_DIT_SPLIT if seeded is None else seeded,
                      "H3_VAE_DIT_SPLIT")
    decode_split = parse(decode_raw, _DEFAULT_VAE_SPLIT if seeded is None else seeded,
                         "H3_VAE_DECODE_SPLIT")
    return min(dit_split, 35), min(decode_split, 35)


def _module_resident_bytes(module: nn.Module) -> int:
    """Resident bytes of one module, counting INT8 storage rather than FP16."""
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(
        module.buffers(recurse=True)
    ):
        total += _parameter_storage_bytes(tensor)
    return total


def _move_quantized_aware(module: nn.Module, device: torch.device) -> None:
    """Move one decoder block to *device*, INT8 wrapper tensors included.

    ``QuantizedTensor`` is a wrapper subclass whose ``to()`` correctly follows
    ``_qdata`` and the per-tensor ``scale``, so ``nn.Module.to`` is sufficient
    for the weights themselves.  What it cannot know about is the FP16 scale
    copy that :func:`_install_h3_v100_int8_w8a16` caches on the module for the
    SM70 W8A16 path.  Leaving that cache behind produces a cross-device operand
    on the first Linear after a rebalance, so refresh it here.
    """
    _move_module(module, device)
    for sub in module.modules():
        cached = getattr(sub, "_h3_v100_int8_scale_fp16", None)
        if isinstance(cached, torch.Tensor) and cached.device != device:
            sub._h3_v100_int8_scale_fp16 = cached.to(device=device)


def _whole_card_free_bytes(device: torch.device) -> int:
    """Free bytes on the whole card, not just this process's allocator.

    ``mem_get_info`` is deliberate: the admission check has to account for the
    other rank's NCCL buffers and CUDA context, which a torch allocator query
    cannot see.
    """
    with torch.cuda.device(device):
        free, _total = torch.cuda.mem_get_info()
    return int(free)


class H3VAELayoutManager:
    """Move decoder blocks between the two cards as the stage changes.

    Only tensor residency changes.  Block order, weights, dtypes and the
    forward implementation are untouched, so a rebalance is numerically inert:
    the same block still runs on the same values, just on the other card.

    Blocks move one at a time with a synchronize and a source release between
    them, so the transient cost is one block rather than the whole span being
    duplicated.  Each target layout passes a whole-card admission check first;
    when cuda:0 cannot take the full decode layout the manager degrades toward
    the sampling layout instead of letting a later allocation fail inside a
    collective.
    """

    def __init__(self, model: nn.Module, devices, dit_split: int, decode_split: int):
        self.model = model
        self.devices = tuple(devices)
        self.dit_split = int(dit_split)
        self.decode_split = int(decode_split)
        self.current_split = int(dit_split)
        self.safety_bytes = _vae_rebalance_safety_bytes()
        self.history: list[dict] = []

    @property
    def decoder(self):
        decoder = getattr(self.model, "decoder", None)
        return decoder if isinstance(decoder, H3ParallelViTDecoder) else None

    def _candidates(self, target: int) -> list[int]:
        """Target split first, then successively less aggressive fallbacks."""
        if target == self.current_split:
            return [target]
        step = 4 if target > self.current_split else -4
        values = [target]
        value = target - step
        while (step > 0 and value > self.current_split) or (
            step < 0 and value < self.current_split
        ):
            values.append(value)
            value -= step
        values.append(self.current_split)
        seen = set()
        return [v for v in values if not (v in seen or seen.add(v))]

    def _move_bytes(self, split: int) -> tuple[int, int, list[int]]:
        """Bytes leaving each card, plus the block indices that move."""
        decoder = self.decoder
        blocks = decoder.source.transformer_blocks
        moving = list(
            range(min(split, self.current_split), max(split, self.current_split))
        )
        total = sum(_module_resident_bytes(blocks[i]) for i in moving)
        if split > self.current_split:
            return total, 0, moving  # cuda:1 -> cuda:0
        return 0, total, moving

    def _admit(self, split: int) -> tuple[bool, dict]:
        to_first, to_second, moving = self._move_bytes(split)
        first, second = self.devices
        free_first = _whole_card_free_bytes(first)
        free_second = _whole_card_free_bytes(second)
        # One block is duplicated while it is in flight, so require the
        # incoming span plus one block of headroom above the safety margin.
        blocks = self.decoder.source.transformer_blocks
        largest = max(
            (_module_resident_bytes(blocks[i]) for i in moving), default=0
        )
        need_first = to_first + largest if to_first else 0
        need_second = to_second + largest if to_second else 0
        ok = (
            (need_first == 0 or free_first >= need_first + self.safety_bytes)
            and (need_second == 0 or free_second >= need_second + self.safety_bytes)
        )
        detail = {
            "split": split,
            "blocks_moving": len(moving),
            "incoming_mib": {
                str(first): round(to_first / 1024**2, 1),
                str(second): round(to_second / 1024**2, 1),
            },
            "free_mib": {
                str(first): round(free_first / 1024**2, 1),
                str(second): round(free_second / 1024**2, 1),
            },
            "safety_mib": round(self.safety_bytes / 1024**2, 1),
            "admitted": ok,
        }
        return ok, detail

    def _apply(self, split: int) -> None:
        decoder = self.decoder
        first, second = self.devices
        blocks = decoder.source.transformer_blocks
        if split > self.current_split:
            moving, target = range(self.current_split, split), first
        else:
            moving, target = range(split, self.current_split), second
        for index in moving:
            _move_quantized_aware(blocks[index], target)
            # Synchronize and release per block: the point of moving one at a
            # time is that only one block is ever duplicated across the cards.
            with torch.cuda.device(target):
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
        decoder.split = split
        self.current_split = split

    def ensure(self, split: int, *, reason: str) -> int:
        """Move to *split*, degrading to a safe layout when cuda:0 is tight."""
        decoder = self.decoder
        if decoder is None:
            return self.current_split
        if split == self.current_split:
            return self.current_split

        attempts = []
        for candidate in self._candidates(split):
            if candidate == self.current_split:
                attempts.append({"split": candidate, "admitted": True,
                                 "note": "kept current layout"})
                logging.warning(
                    "[H3 VAE MP] %s: cannot admit split %d; keeping %d/%d",
                    reason, split, self.current_split,
                    len(decoder.source.transformer_blocks) - self.current_split,
                )
                break
            ok, detail = self._admit(candidate)
            attempts.append(detail)
            if not ok:
                continue
            started = time.perf_counter()
            try:
                self._apply(candidate)
            except torch.cuda.OutOfMemoryError:
                # Admission passed but the move still failed; the partially
                # moved span is consistent because ``decoder.split`` is only
                # advanced after every block has landed, so recover by
                # re-deriving the layout from the blocks themselves.
                self._resync_split()
                torch.cuda.empty_cache()
                logging.exception(
                    "[H3 VAE MP] %s: OOM while moving to split %d; layout is "
                    "now %d", reason, candidate, self.current_split,
                )
                continue
            elapsed = time.perf_counter() - started
            moved = detail["blocks_moving"]
            logging.info(
                "[H3 VAE MP] %s: layout %d/%d (%d blocks over NVLink in %.3fs)",
                reason, candidate,
                len(decoder.source.transformer_blocks) - candidate,
                moved, elapsed,
            )
            self.history.append(
                {"reason": reason, "requested": split, "applied": candidate,
                 "seconds": round(elapsed, 4), "attempts": attempts}
            )
            return candidate

        self.history.append(
            {"reason": reason, "requested": split, "applied": self.current_split,
             "attempts": attempts}
        )
        return self.current_split

    def _resync_split(self) -> None:
        """Re-derive the split from where the blocks actually live."""
        decoder = self.decoder
        first = self.devices[0]
        blocks = decoder.source.transformer_blocks
        split = 0
        for index, block in enumerate(blocks):
            if _first_parameter_device(block) == first:
                split = index + 1
            else:
                break
        decoder.split = split
        self.current_split = split

    def ensure_decode(self) -> int:
        return self.ensure(self.decode_split, reason="decode")

    def ensure_sampling(self) -> int:
        return self.ensure(self.dit_split, reason="sampling")

    def report(self) -> dict:
        decoder = self.decoder
        blocks = 0 if decoder is None else len(decoder.source.transformer_blocks)
        return {
            "dit_split": self.dit_split,
            "decode_split": self.decode_split,
            "current_split": self.current_split,
            "decoder_blocks": blocks,
            "safety_mib": round(self.safety_bytes / 1024**2, 1),
            "transitions": len(self.history),
            "last": self.history[-1] if self.history else None,
        }


def _vae_rebalance_safety_bytes() -> int:
    raw = os.environ.get("H3_VAE_REBALANCE_SAFETY_MIB", "1024").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "H3_VAE_REBALANCE_SAFETY_MIB must be an integer number of MiB"
        ) from exc
    if value < 0:
        raise ValueError("H3_VAE_REBALANCE_SAFETY_MIB must be >= 0")
    return value << 20


def _parallel_qwen_forward(self, x, attention_mask=None, embeds=None, num_tokens=None,
                           intermediate_output=None, final_layer_norm_intermediate=True,
                           dtype=None, position_ids=None, embeds_info=[],
                           past_key_values=None, input_ids=None, deepstack_embeds=None,
                           visual_pos_masks=None):
    """Layer-parallel equivalent of ComfyUI's ``Llama2_.forward``."""
    first = self._h3_mp_first
    second = self._h3_mp_second
    split = self._h3_mp_split

    if embeds is not None:
        x = embeds
    else:
        x = self.embed_tokens(x, out_dtype=dtype)
    if x.device != first:
        x = x.to(first)

    seq_len = x.shape[1]
    past_len = 0
    if past_key_values is not None and len(past_key_values) > 0:
        past_len = self.get_past_len(past_key_values)
    if position_ids is None:
        position_ids = torch.arange(past_len, past_len + seq_len, device=first).unsqueeze(0)
    else:
        position_ids = position_ids.to(first)

    freqs_cis = self.compute_freqs_cis(position_ids, first)
    mask = None
    if attention_mask is not None:
        attention_mask = attention_mask.to(first)
        mask = 1.0 - attention_mask.to(x.dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
        mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)
    if seq_len > 1:
        causal_mask = torch.empty(
            past_len + seq_len, past_len + seq_len,
            dtype=x.dtype, device=first,
        ).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
        mask = causal_mask if mask is None else mask + causal_mask

    from comfy.ldm.modules.attention import optimized_attention_for_device

    current = first
    intermediate = None
    all_intermediate = None
    only_layers = None
    if intermediate_output is not None:
        if isinstance(intermediate_output, list):
            all_intermediate = []
            only_layers = set(intermediate_output)
        elif intermediate_output == "all":
            all_intermediate = []
            intermediate_output = None
        elif intermediate_output < 0:
            intermediate_output = len(self.layers) + intermediate_output

    next_key_values = []
    visual_mask = visual_pos_masks.to(first) if visual_pos_masks is not None else None
    deepstack = deepstack_embeds

    for i, layer in enumerate(self.layers):
        target = first if i < split else second
        if target != current:
            x = x.to(target)
            mask = mask.to(target) if mask is not None else None
            freqs_cis = _move_freqs(freqs_cis, target)
            visual_mask = visual_mask.to(target) if visual_mask is not None else None
            current = target

        if all_intermediate is not None:
            if only_layers is None or i in only_layers:
                all_intermediate.append(x.unsqueeze(1).clone())

        past_kv = None
        if past_key_values is not None:
            past_kv = past_key_values[i] if len(past_key_values) > 0 else []

        optimized_attention = optimized_attention_for_device(
            target, mask=mask is not None, small_input=True
        )
        x, current_kv = layer(
            x=x,
            attention_mask=mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=past_kv,
        )
        if current_kv is not None:
            next_key_values.append(current_kv)

        if deepstack is not None and i < len(deepstack):
            x[visual_mask] = x[visual_mask] + deepstack[i].to(x)
        if i == intermediate_output:
            intermediate = x.clone()

    if self.norm is not None:
        if x.device != second:
            x = x.to(second)
        x = self.norm(x)

    if all_intermediate is not None:
        if only_layers is None or ((len(self.layers)) in only_layers):
            all_intermediate.append(x.unsqueeze(1).clone())
        all_intermediate = [v.to(x.device) for v in all_intermediate]
        intermediate = torch.cat(all_intermediate, dim=1)

    if intermediate is not None and final_layer_norm_intermediate and self.norm is not None:
        intermediate = self.norm(intermediate.to(second))

    if len(next_key_values) > 0:
        return x, intermediate, next_key_values
    return x, intermediate


def install_qwen_layer_parallel(sm, devices=None, split=None) -> dict:
    """Place a Qwen3-VL language/vision model on both V100s.

    ``sm`` is the inner ``Qwen3VLClipModel`` returned by ClipProj's loader.
    The initial full load is intentionally done by ComfyUI first, while the
    cards are empty; only then are the already materialised layers moved to
    their permanent owners.  If a caller offloads the encoder between prompts,
    a later encode restores those owners after ComfyUI brings it back.
    """
    if getattr(sm, "_h3_qwen_layer_parallel", False):
        # A pre-DiT ClipProj release moves the complete encoder to CPU.  The
        # next ``load_models_gpu`` call necessarily brings it back through its
        # single ``load_device`` first, so re-assert the permanent owners after
        # every reload instead of treating the installation as one-shot.
        _restore_qwen_layer_parallel(sm)
        return getattr(sm, "_h3_qwen_parallel_report", {})
    pair = _cuda_devices(devices)
    if pair is None:
        raise RuntimeError("Qwen layer parallel requires two CUDA devices")
    first, second = pair
    transformer = sm.transformer
    language = getattr(transformer, "model", None)
    layers = getattr(language, "layers", None)
    if language is None or layers is None or len(layers) < 2:
        raise TypeError("the selected CLIP is not a Qwen3-VL language stack")
    if split is None:
        # The language blocks are not the only resident payload: GPU0 also
        # owns H3 rank0 and the conditioning owner.  Twelve layers on the
        # first card leaves enough headroom for 1 MP keyframe rows while the
        # remaining 24 layers stay resident on GPU1.
        split = int(os.environ.get("H3_QWEN_SPLIT", "12"))
    split = max(1, min(int(split), len(layers) - 1))

    # The visual tower and token embedding are small relative to the 36
    # language blocks; keeping both on GPU0 avoids an additional handoff for
    # multimodal preprocessing and deepstack features.
    _move_module(language.embed_tokens, first)
    if getattr(transformer, "visual", None) is not None:
        _move_module(transformer.visual, first)
    for index, layer in enumerate(layers):
        _move_module(layer, first if index < split else second)
    if getattr(language, "norm", None) is not None:
        _move_module(language.norm, second)
    if getattr(language, "lm_head", None) is not None:
        _move_module(language.lm_head, second)

    language._h3_mp_first = first
    language._h3_mp_second = second
    language._h3_mp_split = split
    language.forward = types.MethodType(_parallel_qwen_forward, language)
    transformer._h3_qwen_layer_parallel = True
    sm._h3_qwen_layer_parallel = True
    sm._h3_qwen_parallel_report = {
        "devices": [str(first), str(second)],
        "language_layers": len(layers),
        "split": split,
        "vision_device": str(first),
        "activation_handoff": "cuda:0 -> cuda:1 once at language layer split",
    }
    logging.info(
        "[H3 Qwen MP] %d layers split %d/%d: embedding+vision=%s, tail=%s",
        len(layers), split, len(layers) - split, first, second,
    )
    return sm._h3_qwen_parallel_report


def _restore_qwen_layer_parallel(sm) -> None:
    """Re-apply a previously installed Qwen layer-MP placement.

    ``ModelPatcher`` uses one load device when restoring a model from CPU.  A
    Qwen MP encoder has two permanent owners, recorded on the language stack,
    so a plain restore would silently gather the tail on GPU0.  This helper is
    intentionally separate from installation: it never replaces the forward
    method or changes the split, it only moves each already-materialised
    module back to its recorded owner.
    """
    transformer = getattr(sm, "transformer", None)
    language = getattr(transformer, "model", None)
    layers = getattr(language, "layers", None)
    first = getattr(language, "_h3_mp_first", None)
    second = getattr(language, "_h3_mp_second", None)
    split = getattr(language, "_h3_mp_split", None)
    if language is None or layers is None or first is None or second is None:
        raise RuntimeError(
            "Qwen layer-MP metadata is missing while restoring the encoder"
        )
    if split is None or not 0 < int(split) < len(layers):
        raise RuntimeError(
            "Qwen layer-MP split is invalid while restoring the encoder: %r"
            % (split,)
        )

    _move_module(language.embed_tokens, first)
    if getattr(transformer, "visual", None) is not None:
        _move_module(transformer.visual, first)
    for index, layer in enumerate(layers):
        _move_module(layer, first if index < int(split) else second)
    if getattr(language, "norm", None) is not None:
        _move_module(language.norm, second)
    if getattr(language, "lm_head", None) is not None:
        _move_module(language.lm_head, second)

    logging.debug(
        "[H3 Qwen MP] restored layer owners after CPU offload: %s/%s split=%d",
        first,
        second,
        int(split),
    )


class _ResidentVAEHandle:
    """Small patcher-shaped object for diagnostics and compatibility only.

    The parallel VAE overrides ``encode``/``decode`` and therefore never enters
    ComfyUI's global DynamicVRAM loader.  Keeping a patcher-shaped handle makes
    existing nodes that inspect ``load_device`` and ``offload_device`` behave
    sensibly without allowing them to move a distributed module tree.
    """

    def __init__(self, model, device, size):
        self.model = model
        self.load_device = device
        self.offload_device = device
        self._size = int(size)
        self._h3_parallel_resident = True

    def is_dynamic(self):
        return False

    def model_size(self):
        return self._size

    def loaded_size(self):
        return self._size

    def current_loaded_device(self):
        return self.load_device


class H3ParallelVAE:
    """ComfyUI VAE facade whose video model is resident on two GPUs."""

    def __init__(self, model: nn.Module, devices, size: int, path: str,
                 layout: "H3VAELayoutManager | None" = None):
        self.first_stage_model = model.eval()
        self.device = devices[0]
        self.parallel_devices = tuple(devices)
        self.layout = layout
        # Do not inherit ComfyUI's mutable intermediate-device policy here.
        # The normal host buffer keeps the 1 MP video out of both resident
        # cards; ``H3_VAE_OUTPUT_DEVICE`` is an explicit opt-in for benchmarks.
        self.output_device = _resolve_vae_output_device(self.parallel_devices)
        self.vae_dtype = torch.float16
        self.latent_channels = 24
        self.latent_dim = 3
        self.output_channels = 3
        self.pad_channel_value = None
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: image
        self.crop_input = True
        self.handles_tiling = True
        self.comfy_has_chunked_io = True
        self.upscale_ratio = (
            lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16
        )
        self.downscale_ratio = (
            lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16
        )
        self.upscale_index_formula = (4, 16, 16)
        self.downscale_index_formula = (4, 16, 16)
        self.path = str(path)
        self.size = int(size)
        self.patcher = _ResidentVAEHandle(self.first_stage_model, self.device, self.size)
        self._h3_parallel_report = getattr(model, "_h3_parallel_report", {})
        self._h3_parallel_report.update({
            "path": self.path,
            "size_bytes": self.size,
            "devices": [str(x) for x in devices],
            "output_device": str(self.output_device),
            "resident": True,
            "host_mmap": False,
        })

    def throw_exception_if_invalid(self):
        if self.first_stage_model is None:
            raise RuntimeError("H3 parallel VAE is invalid")

    def model_size(self):
        return self.size

    def is_dynamic(self):
        return False

    def vae_output_dtype(self):
        import comfy.model_management as mm

        return mm.intermediate_dtype()

    def spacial_compression_decode(self):
        return 16

    def spacial_compression_encode(self):
        return 16

    def temporal_compression_decode(self):
        return 4

    def vae_encode_crop_pixels(self, pixels):
        if self.crop_input:
            dims = pixels.shape[1:-1]
            for index, dim in enumerate(dims):
                target = (dim // 16) * 16
                offset = (dim - target) // 2
                if target != dim:
                    pixels = pixels.narrow(index + 1, offset, target)
        if pixels.shape[-1] > self.output_channels:
            pixels = pixels[..., :self.output_channels]
        return pixels

    @torch.no_grad()
    def encode(self, pixel_samples):
        # Match VAE.encode's IMAGE [B,H,W,C] -> model [B,C,T,H,W] contract.
        pixels = self.vae_encode_crop_pixels(pixel_samples)
        pixels = pixels.movedim(-1, 1)
        if pixels.ndim < 5:
            pixels = pixels.unsqueeze(2)
        pixels = self.process_input(pixels).to(device=self.device, dtype=self.vae_dtype)
        latent = self.first_stage_model.encode(pixels, device=self.device)
        return latent.to(device=self.output_device, dtype=self.vae_output_dtype(), copy=True)

    @torch.no_grad()
    def decode(self, samples_in, **kwargs):
        if samples_in.ndim != 5:
            raise ValueError(f"H3 video VAE expects [B,C,T,H,W], got {samples_in.shape}")
        # Decode is the only stage that wants weight-heavy cuda:0.  Rebalance
        # here rather than at load time so DiT sampling never sees the decode
        # layout.  This runs after the sampler has returned, so the DiT peak and
        # the VAE move never overlap.
        if self.layout is not None:
            self.layout.ensure_decode()
        samples = samples_in.to(device=self.device, dtype=self.vae_dtype)
        shape = self.first_stage_model.decode_output_shape(samples.shape)
        output = torch.empty(shape, device=self.output_device, dtype=torch.float32)
        self.first_stage_model.decode(samples, output_buffer=output, **kwargs)
        # ComfyUI IMAGE convention is [B,T,H,W,C].
        return output.to(device=self.output_device, dtype=self.vae_output_dtype(), copy=False).movedim(1, -1)

    def encode_tiled(self, pixel_samples, **kwargs):
        return self.encode(pixel_samples)

    def decode_tiled(self, samples_in, **kwargs):
        return self.decode(samples_in)


def _load_h3_int8_parallel_model(path: str, state_dict, devices, split: int,
                                 decode_split: int | None = None):
    """Load the INT8-ConvRot H3 VAE without expanding decoder weights.

    Kijai's H3 INT8 checkpoint quantizes the 144 decoder Linear weights, but
    leaves the convolutional encoder and small parameters as F32.  V100 has
    no native INT8 Tensor Core, yet the installed ComfyUI kitchen backend has
    a V100-compatible Triton ``int8_linear`` implementation.  Keep those
    decoder matrices as ``QuantizedTensor`` objects and let that path perform
    the bounded FP16 compute conversion; all other weights are materialised
    as FP16 on their final MP owner.

    This function intentionally does not call ``load_state_dict``.  A raw
    INT8 tensor cannot be wrapped in a normal ``nn.Parameter`` and the usual
    mixed-precision loader expects a fully materialised CPU state dict.  The
    direct per-tensor path preserves the no-host-mmap guarantee.
    """
    import comfy.ops
    import comfy.utils
    from comfy.quant_ops import QuantizedTensor, get_layout_class
    from comfy.ldm.minimax import vae as vae_module

    first, second = devices
    quant_ops = comfy.ops.mixed_precision_ops(
        {"int8_tensorwise": {}}, torch.float16
    )
    # Construct only metadata/shape objects.  In particular, do not allocate
    # the 36 decoder matrices on the host before direct device materialisation.
    with torch.device("meta"):
        model = vae_module.MiniMaxH3VideoVAE(operations=quant_ops).eval()

    buffer_names = {name for name, _ in model.named_buffers()}
    int8_count = 0
    ordinary_count = 0
    logical_bytes = 0
    storage_bytes = 0
    layout = get_layout_class("TensorWiseINT8Layout")

    for key, value in state_dict.items():
        if key.endswith(".comfy_quant") or key.endswith(".weight_scale"):
            continue

        device = _h3_vae_owner(key, devices, split)
        marker_key = f"{key[:-len('.weight')]}.comfy_quant" if key.endswith(".weight") else None
        if marker_key is not None and marker_key in state_dict:
            marker = _direct_tensor_to_device(
                state_dict[marker_key], torch.device("cpu"), torch.uint8
            )
            config = json.loads(marker.detach().cpu().numpy().tobytes())
            if config.get("format") != "int8_tensorwise":
                raise ValueError(
                    f"unsupported H3 VAE quantization for {key}: {config}"
                )

            qdata = _direct_tensor_to_device(value, device, torch.int8)
            scale = _direct_tensor_to_device(
                state_dict[f"{key[:-len('.weight')]}.weight_scale"],
                device,
                torch.float32,
            )
            params = layout.Params(
                scale=scale,
                orig_dtype=torch.float16,
                orig_shape=tuple(value.shape),
                is_weight=True,
                convrot=bool(config.get("convrot", False)),
                convrot_groupsize=int(config.get("convrot_groupsize", 256)),
            )
            comfy.utils.set_attr_param(
                model, key, QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
            )
            parent, _name = comfy.utils.resolve_attr(model, key)
            parent.quant_format = "int8_tensorwise"
            parent.layout_type = "TensorWiseINT8Layout"
            parent._full_precision_mm = False
            int8_count += 1
            logical_bytes += int(value.numel() * 2)
            storage_bytes += int(qdata.numel() * qdata.element_size())
            storage_bytes += int(scale.numel() * scale.element_size())
            continue

        # The decoder's non-quantized parameters and the encoder are computed
        # in FP16 on V100.  Keep the normalization constants in F32; the VAE
        # casts them to the latent dtype at their use sites.
        target_dtype = (
            torch.float32
            if key in {"latents_mean", "latents_std"}
            else torch.float16 if value.dtype.is_floating_point else value.dtype
        )
        loaded = _direct_tensor_to_device(value, device, target_dtype)
        if key in buffer_names:
            comfy.utils.set_attr_buffer(model, key, loaded)
        else:
            comfy.utils.set_attr_param(model, key, loaded)
        ordinary_count += 1
        logical_bytes += int(value.numel() * (2 if value.dtype.is_floating_point else value.element_size()))
        storage_bytes += int(loaded.numel() * loaded.element_size())

    # These buffers are architecture constants rather than checkpoint data.
    comfy.utils.set_attr_buffer(
        model,
        "pixel_mean",
        torch.tensor(vae_module.IMAGENET_MEAN, device=first, dtype=torch.float32).view(
            1, 3, 1, 1, 1
        ),
    )
    comfy.utils.set_attr_buffer(
        model,
        "pixel_std",
        torch.tensor(vae_module.IMAGENET_STD, device=first, dtype=torch.float32).view(
            1, 3, 1, 1, 1
        ),
    )
    # RotaryEmbeddingND's frequency table is deterministic and is not in the
    # checkpoint.  Recreate it directly on the first owner after meta init.
    pos_embed = model.decoder.pos_embed
    dim = int(2 * pos_embed.n_dim * pos_embed.inv_freq.numel())
    inv_freq = 1 / 100.0 ** torch.arange(
        0, 1, 2 * pos_embed.n_dim / dim, device=first, dtype=torch.float32
    )
    comfy.utils.set_attr_buffer(model, "decoder.pos_embed.inv_freq", inv_freq)

    source_decoder = model.decoder
    w8a16_count = 0
    int8_compute_backend = "comfy_kitchen_int8"
    sm70_pair = all(
        torch.cuda.get_device_capability(device) == (7, 0) for device in devices
    )
    if sm70_pair and _enabled("H3_VAE_INT8_SM70_W8A16", True):
        w8a16_count = _install_h3_v100_int8_w8a16(source_decoder)
        if w8a16_count:
            int8_compute_backend = "sm70_convrot_w8a16_fp16_tensorcore"
            logging.info(
                "[H3 VAE MP] installed SM70 W8A16 fallback on %d decoder Linears",
                w8a16_count,
            )
    try:
        int8_tile_batch = max(
            1, int(os.environ.get("H3_VAE_INT8_TILE_BATCH", "2"))
        )
    except ValueError as exc:
        raise ValueError("H3_VAE_INT8_TILE_BATCH must be a positive integer") from exc
    int8_tile_batch = _install_h3_v100_int8_tile_batch(model, int8_tile_batch)
    model.decoder = H3ParallelViTDecoder(source_decoder, first, second, split)
    model.eval()
    layout = H3VAELayoutManager(
        model, devices, split, split if decode_split is None else decode_split
    )
    register_layout_manager(layout)
    install_presample_layout_hook()

    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.synchronize()

    del state_dict
    gc.collect()
    logical_bytes = max(
        logical_bytes,
        sum(p.numel() * 2 for p in model.parameters()),
    )
    # Include the tiny non-parameter buffers in the storage report only via
    # the model-size compatibility field; the measured CUDA allocation is the
    # authoritative peak number in the benchmark.
    model._h3_parallel_report = {
        "split": split,
        "decoder_blocks": len(source_decoder.transformer_blocks),
        "materialized_tensors": ordinary_count + int8_count,
        "quantized_linear_tensors": int8_count,
        "ordinary_tensors": ordinary_count,
        "quant_format": "int8_tensorwise + convrot",
        "compute_dtype": "torch.float16",
        "int8_compute_backend": int8_compute_backend,
        "sm70_w8a16_linears": w8a16_count,
        "int8_spatial_tile_batch": int8_tile_batch,
        "vae_layout": layout.report(),
        "logical_fp16_bytes": logical_bytes,
        "resident_storage_bytes": storage_bytes,
        "decoder_devices": [
            str(_first_parameter_device(source_decoder.transformer_blocks[0])),
            str(_first_parameter_device(source_decoder.transformer_blocks[-1])),
        ],
    }
    return H3ParallelVAE(model, devices, storage_bytes, path, layout=layout)


def _load_h3_parallel_model(path: str, devices, split: int,
                            decode_split: int | None = None):
    import comfy.ops
    import comfy.utils
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    state_dict, _metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    if _h3_int8_vae_state_dict(state_dict):
        return _load_h3_int8_parallel_model(
            path, state_dict, devices, split, decode_split
        )

    model = MiniMaxH3VideoVAE(operations=comfy.ops.disable_weight_init).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        logging.warning("[H3 VAE MP] missing keys: %s", missing[:8])
    if unexpected:
        logging.warning("[H3 VAE MP] unexpected keys: %s", unexpected[:8])

    first, second = devices
    decoder = model.decoder
    materialized = 0
    # Encoder is small (about 0.34 GiB) and stays with the input/latent head.
    materialized += _materialize(model.encoder, first)
    materialized += _materialize(model.quant_conv, first)
    materialized += _materialize(model.post_quant_conv, first)

    materialized += _materialize(decoder.x_embedder, first)
    for block_index, block in enumerate(decoder.transformer_blocks):
        materialized += _materialize(block, first if block_index < split else second)
    materialized += _materialize(decoder.norm_out, second)
    materialized += _materialize(decoder.proj_out, second)
    materialized += _materialize_direct(decoder, "register_tokens", first, buffer=False)
    materialized += _materialize_direct(decoder, "mask_token", first, buffer=True)
    materialized += _materialize_direct(model, "latents_mean", first, buffer=True)
    materialized += _materialize_direct(model, "latents_std", first, buffer=True)

    # These buffers are created by the architecture, not present in the
    # checkpoint.  They are tiny, but placing them explicitly avoids a later
    # root ``.to(cuda:0)`` accidentally dragging decoder tail buffers around.
    _move_module(decoder.pos_embed, first)
    model.pixel_mean = model.pixel_mean.to(first)
    model.pixel_std = model.pixel_std.to(first)

    # Keep the source decoder as the parameter owner, but replace its forward
    # with the explicit two-stage implementation.  Registering it under the
    # wrapper does not duplicate storage: ``source`` is the same module object.
    model.decoder = H3ParallelViTDecoder(decoder, first, second, split)

    del state_dict
    gc.collect()
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.synchronize()

    layout = H3VAELayoutManager(
        model, devices, split, split if decode_split is None else decode_split
    )
    register_layout_manager(layout)
    install_presample_layout_hook()

    size = int(sum(p.numel() * p.element_size() for p in model.parameters()))
    model._h3_parallel_report = {
        "split": split,
        "decoder_blocks": len(decoder.transformer_blocks),
        "materialized_tensors": materialized,
        "weight_bytes": size,
        "vae_layout": layout.report(),
        "decoder_devices": [
            str(_first_parameter_device(decoder.transformer_blocks[0])),
            str(_first_parameter_device(decoder.transformer_blocks[-1])),
        ],
    }
    return H3ParallelVAE(model, devices, size, path, layout=layout)


def load_h3_video_vae_parallel(path: str, devices=None, split=None):
    """Load/reuse a resident two-GPU H3 video VAE.

    The strong process-local cache is intentional: a ComfyUI graph may invoke
    its loader again after node-cache bookkeeping changes, but the user asked
    that a new video request must not fault the 5.2 GB VAE again.
    """
    if not _enabled("H3_VAE_MP", True):
        return None
    pair = _cuda_devices(devices)
    if pair is None:
        return None
    # The VAE is loaded in its *sampling* layout.  Decode moves the boundary
    # blocks to cuda:0 on demand and the pre-sample hook moves them back, so a
    # DiT request never inherits the decode layout.  An explicit ``split``
    # argument still pins both stages to one layout for callers that want the
    # old single-layout behaviour.
    if split is None:
        dit_split, decode_split = _resolve_vae_stage_splits()
    else:
        dit_split = decode_split = _resolve_vae_split_request(split)
    # The current MiniMax H3 decoder has 36 blocks.  Clamp an explicit value
    # early so an accidental ``H3_VAE_SPLIT=36`` cannot silently strand the
    # complete decoder on GPU0 and make the second-stage handoff invalid.
    dit_split = min(dit_split, 35)
    decode_split = min(decode_split, 35)
    key = (
        str(Path(path).resolve()), str(pair[0]), str(pair[1]),
        dit_split, decode_split,
    )
    cached = _VAE_CACHE.get(key)
    if cached is not None:
        logging.info("[H3 VAE MP] reusing resident VAE: %s", path)
        return cached

    logging.info(
        "[H3 VAE MP] loading resident video VAE from disk slices: %s; "
        "dit_split=%d decode_split=%d; devices=%s,%s",
        path, dit_split, decode_split, pair[0], pair[1],
    )
    value = _load_h3_parallel_model(
        str(Path(path).resolve()), pair, dit_split, decode_split
    )
    _VAE_CACHE[key] = value
    logging.info("[H3 VAE MP] resident weights ready; no future VAE DynamicVRAM reload")
    return value


def qwen_mp_enabled(clip_type: str, mode: str) -> bool:
    """Whether the default ClipProj route should use the two-GPU Qwen path."""
    if mode not in {"resident", "resident_tp", "resident_mp"}:
        return False
    if clip_type not in {"krea2", "qwen3vl_4b"}:
        return False
    return _enabled("H3_QWEN_MP", True) and _cuda_devices() is not None


__all__ = [
    "H3ParallelVAE",
    "H3ParallelViTDecoder",
    "H3VAELayoutManager",
    "ensure_sampling_layout",
    "install_presample_layout_hook",
    "install_qwen_layer_parallel",
    "load_h3_video_vae_parallel",
    "qwen_mp_enabled",
    "register_layout_manager",
]
