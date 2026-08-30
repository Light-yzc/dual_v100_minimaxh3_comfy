"""ComfyUI model node that replaces H3's 50 blocks with persistent NCCL TP."""

from __future__ import annotations

import gc
import os
import threading
from pathlib import Path

import torch
import torch.nn as nn

import folder_paths
import comfy.model_patcher

from .h3_tp_backbone import DEFAULT_EGRID, normalize_weight_format
from .h3_tp_runtime import (
    RuntimeConfig,
    get_runtime,
    install_postsample_release_hook,
)


def _model_path(name: str) -> str:
    for category in ("unet_gguf", "unet", "diffusion_models"):
        path = folder_paths.get_full_path(category, name)
        if path is not None:
            return path
    raise FileNotFoundError(f"H3 TP diffusion model not found: {name}")


def _lora_path(name: str) -> str:
    path = folder_paths.get_full_path("loras", name)
    if path is None:
        raise FileNotFoundError(f"H3 TP LoRA not found: {name}")
    return path


# ``H3_NODE_CACHE=none`` is useful for large IMAGE outputs, but it also makes
# ComfyUI call every loader node again for the next prompt.  The persistent TP
# workers deliberately survive between prompts; rebuilding this small model
# skeleton on cuda:0 would then allocate a second copy of its non-block
# weights before the TP node can replace ``blocks``.  Keep one patcher per
# checkpoint path so node-cache policy cannot duplicate the resident DiT
# wrapper.  The strong reference is intentional: this is the part of DiT that
# is meant to remain resident for the lifetime of the service.
_STATIC_SKELETON_CACHE: dict[str, object] = {}
_STATIC_SKELETON_CACHE_LOCK = threading.RLock()


class PersistentH3ModelPatcher(comfy.model_patcher.ModelPatcher):
    """Patcher facade for a DiT whose 50 blocks live in the TP workers.

    ComfyUI's normal ``ModelPatcher`` sees the original INT8 block geometry and
    tries to page those (already discarded) weights.  With DynamicVRAM enabled
    that creates a host-pinned arena roughly the size of the whole skeleton
    (about 3 GiB on the V100 host).  The persistent TP runtime is the owner of
    every block, so this facade reports the remaining non-block weights as
    already resident and makes the generic load/unload hooks no-ops.  It still
    keeps the normal patcher API so samplers, LoRA wrappers, and ComfyUI's model
    bookkeeping continue to work.
    """

    def _resident_bytes(self) -> int:
        cached = getattr(self.model, "_h3_persistent_resident_bytes", None)
        if cached is not None:
            return int(cached)

        # ``state_dict`` is intentionally tiny after TP conversion: the proxy
        # block has no parameters and only the embedder/refiner/final weights
        # remain.  QuantizedTensor.nbytes counts its compressed storage.
        total = 0
        for value in self.model.state_dict().values():
            if value is None:
                continue
            try:
                total += int(value.nbytes)
            except (AttributeError, TypeError):
                total += int(value.numel()) * int(value.element_size())
        total = max(1, total)
        self.model._h3_persistent_resident_bytes = total
        return total

    def model_size(self) -> int:
        # ``size`` is copied by ModelPatcher.clone(); keep it aligned with the
        # post-TP state dict rather than the pre-TP 3 GiB skeleton estimate.
        size = self._resident_bytes()
        self.size = size
        return size

    def loaded_size(self) -> int:
        # All weights represented by this patcher are materialised by the
        # static loader before TP starts.  The 50 block payloads are external.
        return self.model_size()

    def _mark_loaded(self, device=None) -> None:
        if device is not None:
            self.model.device = device
        self.model.model_loaded_weight_memory = self.model_size()
        self.model.model_offload_buffer_memory = 0
        self.model.model_lowvram = False
        self.model.lowvram_patch_counter = 0
        self.model.current_weight_patches_uuid = self.patches_uuid

    def load(self, device_to=None, *args, **kwargs):
        """Satisfy direct patcher loads without creating a VBAR/host arena."""
        self._mark_loaded(self.load_device if device_to is None else device_to)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        """Keep the resident TP model untouched when ComfyUI asks for VRAM."""
        if force_patch_weights:
            raise RuntimeError(
                "Persistent H3 TP does not support force_patch_weights; "
                "apply LoRA before enabling TP"
            )
        self._mark_loaded(device_to)
        return 0

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        # Unloading would leave the NCCL workers with a model that the generic
        # patcher can no longer reconstruct.  The explicit post-sample hook
        # releases transient activations instead.
        return 0

    def partially_unload_ram(self, ram_to_unload, *args, **kwargs):
        return 0

    def detach(self, unpatch_all=True):
        # ``free_memory`` may detach an old clone when a new prompt arrives.
        # Do not move the shared non-block weights to CPU or eject LoRA
        # injections from the clone that is about to run.
        return self.model


def _unload_before_block_tree_rewrite(model_patcher):
    """Restore DynamicVRAM backups before replacing ``diffusion_model.blocks``.

    The persistent worker deliberately removes the original 50 block modules.
    A previously-used GGUF DynamicVRAM patcher can still hold their loaded
    weights in ``backup``; a later partial load would then try to restore paths
    such as ``blocks.0.attn.qkv_proj.weight`` into the one proxy module.  That
    produces an AttributeError and, worse, leaves stale VRAM accounting behind.
    Restore/unload while the original tree is still present, then perform the
    rewrite with an empty backup set.  The worker owns all 50 blocks thereafter.
    """
    # The INT8 static skeleton is deliberately constructed with its retained
    # (non-block) weights already on cuda:0.  Calling the generic unpatch path
    # here would move that resident tensor set to CPU before the TP proxy is
    # installed, creating a second ~3 GiB anonymous RSS allocation.  The
    # static loader has no DynamicVRAM backup to restore, so there is nothing
    # to unpatch in this case.
    if getattr(getattr(model_patcher, "model", None), "_h3_static_skeleton", False):
        return

    unpatch = getattr(model_patcher, "unpatch_model", None)
    if unpatch is None:
        return
    unpatch(model_patcher.offload_device, unpatch_weights=True)


class PersistentH3TPBlocks(nn.Module):
    """One model-tree block representing the complete external 50-layer chain."""

    def __init__(self, runtime):
        super().__init__()
        # A runtime owns process/pipe/NCCL state, not model parameters.  Keep it
        # out of nn.Module traversal and DynamicVRAM's state-dict accounting.
        object.__setattr__(self, "_runtime", runtime)

    def forward(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options=None,
    ):
        profile = os.environ.get("H3_TP_PROFILE", "0").lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        stage_profile = os.environ.get("H3_TP_STAGE_PROFILE", "0").lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        return self._runtime.forward(
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            te_speed=(transformer_options or {}).get("h3_tp_te_speed"),
            group_cache=(transformer_options or {}).get("h3_tp_group_cache"),
            sigma=(transformer_options or {}).get("sigmas"),
            sample_sigmas=(transformer_options or {}).get("sample_sigmas"),
            cond_or_uncond=(transformer_options or {}).get("cond_or_uncond"),
            profile=profile,
            stage_profile=stage_profile,
        )


class MiniMaxH3TensorParallel:
    @classmethod
    def INPUT_TYPES(cls):
        unets = sorted(
            {
                name
                for category in ("unet_gguf", "unet", "diffusion_models")
                for name in folder_paths.get_filename_list(category)
                if name.lower().endswith((".gguf", ".safetensors"))
            }
        )
        return {
            "required": {
                "model": ("MODEL",),
                "unet_name": (unets,),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "chunk_rows": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 8192, "step": 128},
                ),
            },
            "optional": {
                "weight_format": (
                    ["auto", "q4", "int8_convrot"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto selects Q4 for GGUF and the official INT8-ConvRot "
                            "safetensors backend"
                        ),
                    },
                ),
                "runtime": (
                    "H3_TP_RUNTIME",
                    {
                        "tooltip": (
                            "Shared runtime from MiniMaxH3DualRuntimeLoader. "
                            "Required when Qwen32 TP conditioning and H3 TP "
                            "must use the same NCCL worker."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "enable_tp"
    CATEGORY = "dual_v100/H3"
    TITLE = "MiniMax H3 Persistent NCCL TP (2x V100)"

    def enable_tp(
        self,
        model,
        unet_name,
        lora_name,
        strength=1.0,
        chunk_rows=2048,
        runtime=None,
        weight_format="auto",
    ):
        new_model = model.clone()
        dm = new_model.model.diffusion_model
        if type(dm).__name__ != "MiniMaxH3Model":
            raise TypeError(
                f"MiniMaxH3TensorParallel requires MiniMaxH3Model, got {type(dm).__name__}"
            )
        # The static INT8 loader is cached across prompts.  Once the first
        # prompt has converted its shared model to the persistent proxy, the
        # cached patcher may still be the original compatibility facade.  Do
        # not run its normal unpatch path (which would move resident weights to
        # CPU); simply hand the clone back as a persistent patcher too.
        if getattr(dm, "h3_tp_enabled", False):
            if not isinstance(new_model, PersistentH3ModelPatcher):
                new_model.__class__ = PersistentH3ModelPatcher
            new_model._mark_loaded(new_model.load_device)
            return (new_model,)

        _unload_before_block_tree_rewrite(new_model)
        if not getattr(dm, "use_adaln_curves", False):
            raise ValueError("the persistent TP path currently requires the pruned curve H3 base")
        if len(dm.blocks) != 50:
            raise ValueError(f"expected 50 H3 blocks before TP conversion, got {len(dm.blocks)}")
        if new_model.get_injections("bypass_lora"):
            raise RuntimeError(
                "Apply the H3 TP node before MiniMaxH3TurboLoRA.  The Turbo node "
                "will then patch only token_refiner/final_layer; the persistent "
                "workers own all 50 backbone LoRA shards."
            )

        resolved_model_path = str(Path(_model_path(unet_name)).resolve())
        selected_format = normalize_weight_format(weight_format, resolved_model_path)
        config = RuntimeConfig(
            model_path=resolved_model_path,
            lora_path=str(Path(_lora_path(lora_name)).resolve()),
            egrid_path=str(Path(DEFAULT_EGRID).resolve()),
            dit_format=selected_format,
            lora_strength=float(strength),
            staging_mib=int(os.environ.get("H3_TP_STAGING_MIB", "4")),
            chunk_rows=int(chunk_rows),
            timeout_seconds=int(os.environ.get("H3_TP_TIMEOUT", "900")),
            results_dir=os.environ.get(
                "H3_TP_RESULTS_DIR",
                "/home/regen/code/minimax_v100/results/h3_tp_e2e",
            ),
        )
        if runtime is None:
            runtime_object = get_runtime(config)
            runtime_source = "H3 node"
        else:
            runtime_object = getattr(runtime, "runtime", runtime)
            active_config = getattr(runtime_object, "config", None)
            if active_config is None:
                raise TypeError(
                    "runtime must be the H3_TP_RUNTIME output from "
                    "MiniMaxH3DualRuntimeLoader"
                )
            compared_fields = (
                "model_path",
                "dit_format",
                "lora_path",
                "egrid_path",
                "lora_strength",
                "staging_mib",
                "chunk_rows",
                "timeout_seconds",
                "results_dir",
            )
            mismatches = {
                name: (getattr(active_config, name), getattr(config, name))
                for name in compared_fields
                if getattr(active_config, name) != getattr(config, name)
            }
            if mismatches:
                raise RuntimeError(
                    "shared H3/Qwen runtime does not match the H3 TP node: "
                    f"{mismatches}"
                )
            runtime_source = "shared Qwen32/H3 handle"

        # The DynamicVRAM loader has only created disk-backed meta parameters at
        # this point.  Dropping the original block tree therefore releases file
        # descriptors/metadata, not a host or GPU copy.  The proxy has no model
        # parameters; both ranks load only their persistent compressed shards.
        original_blocks = dm.blocks
        dm.blocks = nn.ModuleList([PersistentH3TPBlocks(runtime_object)])
        dm.h3_tp_enabled = True
        dm.h3_tp_original_layers = 50
        dm.h3_tp_runtime = runtime_object
        del original_blocks
        # From this point on the generic ComfyUI loader must not try to page
        # the discarded block tree.  Convert the facade before any downstream
        # node clones it (Turbo LoRA in particular calls ``model.clone()``).
        new_model.__class__ = PersistentH3ModelPatcher
        new_model._mark_loaded(new_model.load_device)
        gc.collect()

        # Keep the post-TP, no-paging facade in the skeleton cache.  The cache
        # intentionally stores the clean TP model (before Turbo LoRA), so a
        # later prompt can clone it without rebuilding or reloading DiT.
        with _STATIC_SKELETON_CACHE_LOCK:
            for cache_key, cache_value in tuple(_STATIC_SKELETON_CACHE.items()):
                if getattr(cache_value, "model", None) is new_model.model:
                    _STATIC_SKELETON_CACHE[cache_key] = new_model

        # Install once the runtime is known to be the TP one.  Doing it at
        # import time would wrap KSAMPLER.sample for graphs that never touch
        # H3 TP at all.
        install_postsample_release_hook()

        print(
            "[H3TP] model converted to persistent 2-way NCCL TP; workers start "
            f"lazily after conditioning is ready ({runtime_source}); "
            f"weights={selected_format}. Apply "
            "Turbo LoRA next for token_refiner/final-layer adapters.",
            flush=True,
        )
        return (new_model,)


class MiniMaxH3TensorParallelInt8(MiniMaxH3TensorParallel):
    """Explicit INT8-ConvRot alias for workflows and hardware experiments.

    The base node remains format-auto and continues to accept the historical
    Q4 GGUF route.  This alias narrows the model picker and fails closed if a
    Q4 file is selected, making an INT8 A/B workflow unambiguous.
    """

    @classmethod
    def INPUT_TYPES(cls):
        values = super().INPUT_TYPES()
        allowed = [
            name
            for name in values["required"]["unet_name"][0]
            if name.lower().endswith((".safetensors", ".sft"))
        ]
        values["required"]["unet_name"] = (sorted(allowed),)
        values.setdefault("optional", {})["weight_format"] = (
            ["int8_convrot"],
            {"default": "int8_convrot", "tooltip": "Official H3 INT8 ConvRot only"},
        )
        return values

    TITLE = "MiniMax H3 Persistent INT8 ConvRot TP (2x V100)"

    def enable_tp(self, *args, **kwargs):
        kwargs["weight_format"] = "int8_convrot"
        return super().enable_tp(*args, **kwargs)


class MiniMaxH3Int8StaticLoader:
    """Build an INT8 H3 model without DynamicVRAM's whole-model staging.

    The following TP node replaces ``diffusion_model.blocks`` immediately, so
    materialising the 50 block weights through the native ``UNETLoader`` only
    creates a large AIMDO host staging arena that is never used.  This loader
    keeps the small embedder/refiner/final-layer weights, leaves block entries
    disk-backed until they are discarded by ``MiniMaxH3TensorParallelInt8``,
    and returns a classic ModelPatcher with no multi-gigabyte VBAR.
    """

    @classmethod
    def INPUT_TYPES(cls):
        unets = sorted(
            {
                name
                for category in ("unet", "diffusion_models")
                for name in folder_paths.get_filename_list(category)
                if name.lower().endswith((".safetensors", ".sft"))
            }
        )
        return {"required": {"unet_name": (unets,)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "dual_v100/H3"
    TITLE = "MiniMax H3 INT8 Static Skeleton Loader"

    def load_unet(self, unet_name):
        import comfy.model_detection
        import comfy.model_management
        import comfy.model_patcher
        import comfy.sd
        import comfy.utils
        from custom_nodes.NoHostMMap.safetensors import _file_slice_tensor_to_device

        path = str(Path(_model_path(unet_name)).resolve())

        # A new prompt may be evaluated with the node cache disabled.  Release
        # the previous async VAE decoder before any new CUDA allocation; this
        # is deliberately independent of the loader cache below and is a
        # no-op on the first prompt.
        try:
            from .h3_async_vae_bridge import prepare_active_vae_for_qwen

            prepare_active_vae_for_qwen()
        except ImportError:
            # Standalone/unit imports may not have the lifecycle bridge.  In
            # the production ComfyUI package the bridge is present; a runtime
            # lifecycle error must propagate instead of hiding a stale VAE
            # payload and risking a second allocation.
            pass

        with _STATIC_SKELETON_CACHE_LOCK:
            cached = _STATIC_SKELETON_CACHE.get(path)
        if cached is not None:
            print(
                "[H3 INT8 loader] reusing cached static skeleton; "
                "persistent DiT TP remains resident",
                flush=True,
            )
            return (cached,)

        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
        prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        if prefix:
            candidate = comfy.utils.state_dict_prefix_replace(
                sd, {prefix: ""}, filter_keys=True
            )
            if candidate:
                sd = candidate
                sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        model_config = comfy.model_detection.model_config_from_unet(
            sd, "", metadata=metadata
        )
        if model_config is None:
            raise TypeError(
                "MiniMaxH3Int8StaticLoader accepts only a MiniMax H3 diffusion checkpoint"
            )

        parameters = comfy.utils.calculate_parameters(sd)
        weight_dtype = comfy.utils.weight_dtype(sd)
        load_device = torch.device("cuda:0")
        if model_config.quant_config is not None:
            weight_dtype = None
        manual_cast = comfy.model_management.unet_manual_cast(
            None if model_config.quant_config is not None else None,
            load_device,
            model_config.supported_inference_dtypes,
        )
        model_config.set_inference_dtype(
            comfy.model_management.unet_dtype(
                model_params=parameters,
                supported_dtypes=list(model_config.supported_inference_dtypes),
                weight_dtype=weight_dtype,
            ),
            manual_cast,
            device=load_device,
        )

        # Construct directly on the DiT device.  Only the non-block subset
        # below is materialised; the TP node owns every ``blocks.*`` tensor
        # from the checkpoint.  Keeping these small-but-dense refiner/final
        # weights on CUDA avoids retaining a multi-gigabyte CPU copy of the
        # safetensors payload while the persistent shards are loading.
        # Build the architecture on meta so the 50 block modules do not
        # allocate their (unused) INT8 parameter storage on cuda:0.  The TP
        # node immediately replaces this tree with a proxy and loads every
        # block directly into the persistent NCCL workers.  Only the compact
        # embedder/refiner/final subset below is materialised on cuda:0.
        model = model_config.get_model(sd, "", device=torch.device("meta"))
        # mixed_precision_ops keeps the destination and storage dtype in
        # ``factory_kwargs`` and uses them while decoding weights.  The
        # architecture was built on meta to avoid allocating the discarded
        # block tree.  Keep the compact retained layers on CPU in the model's
        # fp16 storage dtype; CastBiasWeightContext streams one layer at a
        # time to the active device during text preprocessing, before the
        # persistent TP shards are admitted.  This avoids both a multi-GiB
        # anonymous host arena and a permanent token-refiner allocation on
        # cuda:0.
        storage_device = torch.device("cpu")
        storage_dtype = getattr(model.diffusion_model, "dtype", None)
        if storage_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            storage_dtype = torch.float16
        for module in model.modules():
            factory_kwargs = getattr(module, "factory_kwargs", None)
            if isinstance(factory_kwargs, dict) and factory_kwargs.get("device") is not None:
                factory_kwargs["device"] = storage_device
                factory_kwargs["dtype"] = storage_dtype

        # The checkpoint payload for these 50 blocks is loaded directly by
        # the two persistent TP workers; it must never pass through this
        # compatibility skeleton.  Drop the parameter-heavy meta module tree
        # before ``load_state_dict`` and retain only its 50-slot structural
        # contract for the following TP node.  Besides releasing thousands of
        # short-lived module/parameter objects, this avoids hundreds of
        # expected "missing weight" warnings on every cold service start.
        source_blocks = model.diffusion_model.blocks
        if len(source_blocks) != 50:
            raise ValueError(
                f"expected 50 H3 blocks in INT8 static skeleton, got {len(source_blocks)}"
            )
        model.diffusion_model.blocks = nn.ModuleList(nn.Identity() for _ in range(50))
        del source_blocks

        kept = {}
        for key, value in sd.items():
            if key.startswith("blocks."):
                continue
            if getattr(value, "is_meta", False):
                storage = getattr(value.untyped_storage(), "_comfy_tensor_file_slice", None)
                if storage is not None:
                    value = _file_slice_tensor_to_device(value, load_device, value.dtype)
            # Leave ordinary retained tensors on CPU.  The mixed-precision
            # loader converts them to the compact model dtype there; forwards
            # use ComfyUI's normal cast path to stage only the active layer on
            # CUDA.  ``.comfy_quant`` metadata must remain CPU for its JSON
            # parser.
            kept[key] = value
        model.load_model_weights(kept, "", assign=True)
        # Marker consumed by _unload_before_block_tree_rewrite().  It is kept
        # on the shared BaseModel (and therefore survives ModelPatcher.clone)
        # so a prompt cannot accidentally page the resident skeleton back to
        # host RAM during TP conversion.
        model._h3_static_skeleton = True
        del kept, sd
        gc.collect()

        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=load_device,
            offload_device=torch.device("cpu"),
        )
        patcher.cached_patcher_init = (self.load_unet, (unet_name,))
        with _STATIC_SKELETON_CACHE_LOCK:
            # ComfyUI executes prompts serially, but retain the first complete
            # object if a future frontend submits concurrent requests.
            existing = _STATIC_SKELETON_CACHE.setdefault(path, patcher)
        if existing is not patcher:
            del patcher
            gc.collect()
            patcher = existing
            print(
                "[H3 INT8 loader] another request completed the skeleton; "
                "reusing that cached instance",
                flush=True,
            )
            return (patcher,)
        print(
            "[H3 INT8 loader] static skeleton ready; block payloads are delegated "
            "to persistent NCCL TP (no DynamicVRAM staging)",
            flush=True,
        )
        return (patcher,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TensorParallel": MiniMaxH3TensorParallel,
    "MiniMaxH3TensorParallelInt8": MiniMaxH3TensorParallelInt8,
    "MiniMaxH3Int8StaticLoader": MiniMaxH3Int8StaticLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TensorParallel": "MiniMax H3 Persistent NCCL TP (2x V100)",
    "MiniMaxH3TensorParallelInt8": "MiniMax H3 Persistent INT8 ConvRot TP (2x V100)",
    "MiniMaxH3Int8StaticLoader": "MiniMax H3 INT8 Static Skeleton Loader",
}


__all__ = [
    "MiniMaxH3TensorParallel",
    "MiniMaxH3TensorParallelInt8",
    "MiniMaxH3Int8StaticLoader",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
