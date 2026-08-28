"""Disk and NVLink/P2P bridges for MiniMax H3 video+audio latents."""

import os
import threading
import logging

import torch

import comfy.nested_tensor


_PEER_LATENTS = {}
_PEER_LATENTS_LOCK = threading.RLock()
_PEER_CONDITIONINGS = {}
_PEER_CONDITIONINGS_LOCK = threading.RLock()


def _trace_conditioning_stats(conditioning, key):
    """Log compact hidden-state stats without making a host-sized copy."""
    if os.environ.get("H3_CONDITIONING_TRACE", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    dump_path = os.environ.get("H3_CONDITIONING_DUMP", "")
    dump_full = os.environ.get("H3_CONDITIONING_DUMP_FULL", "0").lower() in {
        "1", "true", "yes", "on"
    }
    dumped = False
    for position, tensor in enumerate(_conditioning_tensors(conditioning)):
        if tensor.ndim < 2 or tensor.shape[-1] != 5120 or not tensor.is_floating_point():
            continue
        finite = torch.isfinite(tensor)
        valid = tensor[finite]
        if valid.numel():
            valid = valid.float()
            rms = float(valid.square().mean().sqrt().item())
            max_abs = float(valid.abs().max().item())
        else:
            rms = max_abs = float("nan")
        logging.info(
            "[H3 conditioning] key=%r tensor=%d shape=%s dtype=%s device=%s "
            "finite=%d/%d rms=%.6g max_abs=%.6g",
            key, position, tuple(tensor.shape), tensor.dtype, tensor.device,
            int(finite.sum().item()), tensor.numel(), rms, max_abs,
        )
        if dump_path and not dumped and tensor.shape[0] and tensor.shape[1]:
            value = tensor[0].detach().float().cpu() if dump_full else tensor[0, 0].detach().float().cpu()
            torch.save(value, dump_path)
            what = "full conditioning" if dump_full else "token0"
            logging.info("[H3 conditioning] dumped %s to %s", what, dump_path)
            dumped = True


def _validate_cuda_device(device):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the H3 peer latent bridge")
    if not device.startswith("cuda:"):
        raise ValueError(f"Expected a CUDA device such as cuda:0, got {device!r}")
    index = int(device.split(":", 1)[1])
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device!r} is unavailable")
    return index


def _latent_tensors(value):
    if isinstance(value, comfy.nested_tensor.NestedTensor):
        return tuple(value.unbind()), True
    if isinstance(value, torch.Tensor):
        return (value,), False
    raise TypeError(f"Unsupported H3 latent type: {type(value)!r}")


def _pack_latent(tensors, is_nested):
    value = comfy.nested_tensor.NestedTensor(tuple(tensors)) if is_nested else tensors[0]
    return {"samples": value}


def _conditioning_tensors(value):
    """Yield tensor leaves from ComfyUI's list/dict conditioning structure."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, comfy.nested_tensor.NestedTensor):
        yield from value.unbind()
    elif isinstance(value, dict):
        for item in value.values():
            yield from _conditioning_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _conditioning_tensors(item)


def _conditioning_to_device(value, device):
    """Copy conditioning recursively without mutating ComfyUI's cached value."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device, non_blocking=True)
    if isinstance(value, comfy.nested_tensor.NestedTensor):
        return comfy.nested_tensor.NestedTensor(
            tuple(tensor.detach().to(device, non_blocking=True) for tensor in value.unbind())
        )
    if isinstance(value, dict):
        return {key: _conditioning_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_conditioning_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_conditioning_to_device(item, device) for item in value)
    return value


def _validate_conditioning_finite(conditioning, key):
    found = False
    for position, tensor in enumerate(_conditioning_tensors(conditioning)):
        # ClipProj carries token-position tags in the conditioning metadata.
        # They are intentionally 1-D and are not H3 hidden states; validate
        # only the actual embedding leaves here.
        if tensor.ndim < 2 or tensor.shape[-1] != 5120:
            continue
        found = True
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(
                f"H3 peer conditioning {key!r} tensor {position} contains Inf or NaN"
            )
    if not found:
        raise RuntimeError(f"H3 peer conditioning {key!r} contains no tensors")


class SaveMiniMaxH3Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename": (
                    "STRING",
                    {"default": "./output/h3_latent.pt"},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_latent"
    OUTPUT_NODE = True
    CATEGORY = "dual_v100/H3"

    def save_latent(self, samples, filename):
        path = os.path.abspath(filename)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        value = samples["samples"]
        if isinstance(value, comfy.nested_tensor.NestedTensor):
            tensors = [tensor.detach().to("cpu") for tensor in value.unbind()]
            payload = {"kind": "h3_av_nested", "tensors": tensors}
        else:
            payload = {"kind": "latent", "samples": value.detach().to("cpu")}

        torch.save(payload, path)
        return {"ui": {"latent_path": [path]}}


class LoadMiniMaxH3Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": (
                    "STRING",
                    {"default": "./output/h3_latent.pt"},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_latent"
    CATEGORY = "dual_v100/H3"

    def load_latent(self, filename):
        path = os.path.abspath(filename)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        kind = payload.get("kind")
        if kind == "h3_av_nested":
            value = comfy.nested_tensor.NestedTensor(tuple(payload["tensors"]))
        elif kind == "latent":
            value = payload["samples"]
        else:
            raise ValueError(f"Unsupported H3 latent payload: {kind!r}")
        return ({"samples": value},)


class StoreMiniMaxH3LatentPeer:
    """Keep a completed H3 AV latent on GPU memory for a direct P2P handoff."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "key": ("STRING", {"default": "h3_av_latest"}),
                "device": (["cuda:0", "cuda:1"], {"default": "cuda:0"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "store_latent"
    OUTPUT_NODE = True
    CATEGORY = "dual_v100/H3"

    def store_latent(self, samples, key, device="cuda:0"):
        index = _validate_cuda_device(device)
        tensors, is_nested = _latent_tensors(samples["samples"])
        with torch.cuda.device(index):
            stored = tuple(tensor.detach().to(device, non_blocking=True) for tensor in tensors)
            # The sampler normally finished on this stream already. Synchronizing
            # here makes the cache safe to consume in a later Comfy prompt.
            torch.cuda.synchronize(index)
            for position, tensor in enumerate(stored):
                if not bool(torch.isfinite(tensor).all().item()):
                    raise RuntimeError(f"H3 peer latent {key!r} tensor {position} contains Inf or NaN")
        with _PEER_LATENTS_LOCK:
            _PEER_LATENTS[key] = (stored, is_nested, device)
        sizes = [tensor.numel() * tensor.element_size() for tensor in stored]
        return {
            "ui": {
                "peer_latent_key": [key],
                "peer_latent_device": [device],
                "peer_latent_bytes": [sum(sizes)],
            }
        }


class LoadMiniMaxH3LatentPeer:
    """Transfer an H3 latent directly between peer-accessible CUDA devices."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "h3_av_latest"}),
                "device": (["cuda:0", "cuda:1"], {"default": "cuda:1"}),
                "consume": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_latent"
    CATEGORY = "dual_v100/H3"

    def load_latent(self, key, device="cuda:1", consume=True):
        destination_index = _validate_cuda_device(device)
        with _PEER_LATENTS_LOCK:
            try:
                stored, is_nested, source_device = _PEER_LATENTS[key]
            except KeyError as error:
                raise KeyError(
                    f"No H3 peer latent named {key!r}. Submit the matching Store MiniMax H3 Latent (peer bridge) workflow first."
                ) from error

        source_indices = {tensor.device.index for tensor in stored if tensor.is_cuda}
        if len(source_indices) != 1:
            raise RuntimeError(f"Peer latent {key!r} is not stored on one CUDA device: {source_indices!r}")
        source_index = source_indices.pop()
        if source_index != destination_index and not torch.cuda.can_device_access_peer(source_index, destination_index):
            raise RuntimeError(
                f"CUDA P2P is unavailable from cuda:{source_index} to cuda:{destination_index}; "
                "run scripts/check_nvlink.sh before using the peer bridge"
            )

        with torch.cuda.device(destination_index):
            # GPU-to-GPU Tensor.to uses CUDA peer copy when P2P is available; it
            # never stages this transfer through CPU or a .pt file.
            transferred = tuple(tensor.to(device, non_blocking=True) for tensor in stored)
            if consume:
                torch.cuda.synchronize(destination_index)

        if consume:
            with _PEER_LATENTS_LOCK:
                _PEER_LATENTS.pop(key, None)
        return (_pack_latent(transferred, is_nested),)


class ClearMiniMaxH3LatentPeer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"key": ("STRING", {"default": "h3_av_latest"})}}

    RETURN_TYPES = ()
    FUNCTION = "clear_latent"
    OUTPUT_NODE = True
    CATEGORY = "dual_v100/H3"

    def clear_latent(self, key):
        with _PEER_LATENTS_LOCK:
            existed = _PEER_LATENTS.pop(key, None) is not None
        return {"ui": {"cleared_peer_latent_key": [key], "existed": [existed]}}


class StoreMiniMaxH3ConditioningPeer:
    """Persist H3 conditioning across a Qwen -> DiT staged ComfyUI workflow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "key": ("STRING", {"default": "h3_conditioning_latest"}),
                "device": (["cuda:0", "cuda:1"], {"default": "cuda:0"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "store_conditioning"
    OUTPUT_NODE = True
    CATEGORY = "dual_v100/H3"

    def store_conditioning(self, conditioning, key, device="cuda:0"):
        destination_index = _validate_cuda_device(device)
        source_indices = {
            tensor.device.index
            for tensor in _conditioning_tensors(conditioning)
            if tensor.is_cuda
        }
        if len(source_indices) > 1:
            raise RuntimeError(
                f"H3 conditioning {key!r} spans multiple CUDA devices: {source_indices!r}"
            )
        if source_indices:
            source_index = source_indices.pop()
            if source_index != destination_index and not torch.cuda.can_device_access_peer(source_index, destination_index):
                raise RuntimeError(
                    f"CUDA P2P is unavailable from cuda:{source_index} to cuda:{destination_index}; "
                    "run scripts/check_nvlink.sh before staging H3 conditioning"
                )

        with torch.cuda.device(destination_index):
            stored = _conditioning_to_device(conditioning, device)
            torch.cuda.synchronize(destination_index)
            _validate_conditioning_finite(stored, key)
            _trace_conditioning_stats(stored, key)

        tensors = tuple(_conditioning_tensors(stored))
        with _PEER_CONDITIONINGS_LOCK:
            _PEER_CONDITIONINGS[key] = (stored, device)
        return {
            "ui": {
                "peer_conditioning_key": [key],
                "peer_conditioning_device": [device],
                "peer_conditioning_bytes": [
                    sum(tensor.numel() * tensor.element_size() for tensor in tensors)
                ],
            }
        }


class LoadMiniMaxH3ConditioningPeer:
    """Load stored H3 conditioning onto the DiT device without re-running Qwen."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "h3_conditioning_latest"}),
                "device": (["cuda:0", "cuda:1"], {"default": "cuda:0"}),
                "consume": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load_conditioning"
    CATEGORY = "dual_v100/H3"

    def load_conditioning(self, key, device="cuda:0", consume=False):
        destination_index = _validate_cuda_device(device)
        with _PEER_CONDITIONINGS_LOCK:
            try:
                stored, source_device = _PEER_CONDITIONINGS[key]
            except KeyError as error:
                raise KeyError(
                    f"No H3 peer conditioning named {key!r}. Submit the matching Qwen conditioning workflow first."
                ) from error

        source_index = _validate_cuda_device(source_device)
        if source_index != destination_index and not torch.cuda.can_device_access_peer(source_index, destination_index):
            raise RuntimeError(
                f"CUDA P2P is unavailable from cuda:{source_index} to cuda:{destination_index}; "
                "run scripts/check_nvlink.sh before staging H3 conditioning"
            )

        with torch.cuda.device(destination_index):
            transferred = _conditioning_to_device(stored, device)
            torch.cuda.synchronize(destination_index)
            _validate_conditioning_finite(transferred, key)

        if consume:
            with _PEER_CONDITIONINGS_LOCK:
                _PEER_CONDITIONINGS.pop(key, None)
        return (transferred,)


class ClearMiniMaxH3ConditioningPeer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"key": ("STRING", {"default": "h3_conditioning_latest"})}}

    RETURN_TYPES = ()
    FUNCTION = "clear_conditioning"
    CATEGORY = "dual_v100/H3"

    def clear_conditioning(self, key):
        with _PEER_CONDITIONINGS_LOCK:
            existed = _PEER_CONDITIONINGS.pop(key, None) is not None
        return {"ui": {"cleared_peer_conditioning_key": [key], "existed": [existed]}}


NODE_CLASS_MAPPINGS = {
    "SaveMiniMaxH3Latent": SaveMiniMaxH3Latent,
    "LoadMiniMaxH3Latent": LoadMiniMaxH3Latent,
    "StoreMiniMaxH3LatentPeer": StoreMiniMaxH3LatentPeer,
    "LoadMiniMaxH3LatentPeer": LoadMiniMaxH3LatentPeer,
    "ClearMiniMaxH3LatentPeer": ClearMiniMaxH3LatentPeer,
    "StoreMiniMaxH3ConditioningPeer": StoreMiniMaxH3ConditioningPeer,
    "LoadMiniMaxH3ConditioningPeer": LoadMiniMaxH3ConditioningPeer,
    "ClearMiniMaxH3ConditioningPeer": ClearMiniMaxH3ConditioningPeer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveMiniMaxH3Latent": "Save MiniMax H3 Latent (disk bridge)",
    "LoadMiniMaxH3Latent": "Load MiniMax H3 Latent (disk bridge)",
    "StoreMiniMaxH3LatentPeer": "Store MiniMax H3 Latent (peer bridge)",
    "LoadMiniMaxH3LatentPeer": "Load MiniMax H3 Latent (peer bridge)",
    "ClearMiniMaxH3LatentPeer": "Clear MiniMax H3 Latent (peer bridge)",
    "StoreMiniMaxH3ConditioningPeer": "Store MiniMax H3 Conditioning (peer bridge)",
    "LoadMiniMaxH3ConditioningPeer": "Load MiniMax H3 Conditioning (peer bridge)",
    "ClearMiniMaxH3ConditioningPeer": "Clear MiniMax H3 Conditioning (peer bridge)",
}
