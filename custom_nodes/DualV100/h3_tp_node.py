"""ComfyUI model node that replaces H3's 50 blocks with persistent NCCL TP."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
import torch.nn as nn

import folder_paths

from .h3_tp_backbone import DEFAULT_EGRID
from .h3_tp_runtime import RuntimeConfig, get_runtime


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
                if name.lower().endswith(".gguf")
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
    ):
        new_model = model.clone()
        _unload_before_block_tree_rewrite(new_model)
        dm = new_model.model.diffusion_model
        if type(dm).__name__ != "MiniMaxH3Model":
            raise TypeError(
                f"MiniMaxH3TensorParallel requires MiniMaxH3Model, got {type(dm).__name__}"
            )
        if not getattr(dm, "use_adaln_curves", False):
            raise ValueError("the persistent TP path currently requires the pruned curve H3 base")
        if getattr(dm, "h3_tp_enabled", False):
            return (new_model,)
        if len(dm.blocks) != 50:
            raise ValueError(f"expected 50 H3 blocks before TP conversion, got {len(dm.blocks)}")
        if new_model.get_injections("bypass_lora"):
            raise RuntimeError(
                "Apply the H3 TP node before MiniMaxH3TurboLoRA.  The Turbo node "
                "will then patch only token_refiner/final_layer; the persistent "
                "workers own all 50 backbone LoRA shards."
            )

        config = RuntimeConfig(
            model_path=str(Path(_model_path(unet_name)).resolve()),
            lora_path=str(Path(_lora_path(lora_name)).resolve()),
            egrid_path=str(Path(DEFAULT_EGRID).resolve()),
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
        new_model.size = 0
        gc.collect()

        print(
            "[H3TP] model converted to persistent 2-way NCCL TP; workers start "
            f"lazily after conditioning is ready ({runtime_source}). Apply "
            "Turbo LoRA next for token_refiner/final-layer adapters.",
            flush=True,
        )
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TensorParallel": MiniMaxH3TensorParallel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TensorParallel": "MiniMax H3 Persistent NCCL TP (2x V100)",
}


__all__ = [
    "MiniMaxH3TensorParallel",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
