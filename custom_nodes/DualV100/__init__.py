"""Device-aware GGUF loaders and H3 latent bridge for two GPUs."""

import copy
import importlib
import os

import torch
import folder_paths


gguf_nodes = importlib.import_module("custom_nodes.ComfyUI-GGUF.nodes")
multigpu = importlib.import_module("custom_nodes.ComfyUI-MultiGPU")
comfy_nodes = importlib.import_module("nodes")
dynamic_guard = importlib.import_module(
    "custom_nodes.ComfyUI-MultiGPU.clip_dynamic_load_list_guard"
)
from . import h3_latent_io as latent_io
from . import h3_v100_attention
from . import h3_v100_fp32_linear
from . import h3_v100_rms_rope
from . import h3_tp_node
from . import h3_ref2v_tp
from . import h3_group_cache_tp
from . import h3_qwen32_tp_node
from .h3_async_vae_bridge import (
    install_turbo_sampler_hook,
    maybe_load_async_vae_facade,
)
from .h3_model_parallel import load_h3_video_vae_parallel

# TE-Speed is intentionally an optional add-on.  The base TP route must still
# load if a deployment omits this experimental file altogether.
try:
    from . import h3_te_speed_tp
except ImportError as error:
    if getattr(error, "name", None) not in {
        f"{__package__}.h3_te_speed_tp",
        "h3_te_speed_tp",
    }:
        raise
    h3_te_speed_tp = None
    print(
        "[DualV100] optional TE-Speed module is not installed; "
        "the full TP route remains available",
        flush=True,
    )

if os.environ.get("H3_NO_HOST_MMAP", "1").lower() not in {"0", "false", "no"}:
    # ComfyUI's core safetensors helper is imported before custom nodes.  Patch
    # its function reference once at node discovery time; load_torch_file then
    # sees the header-only reader for all later H3 VAE/LoRA loads.
    importlib.import_module("custom_nodes.NoHostMMap.safetensors").install()

# Opt-in and H3-local: this never changes attention dispatch for other ComfyUI
# models, and retains PyTorch SDPA unless H3_V100_ATTENTION explicitly selects
# the numerically/performance-qualified SM70 kernel.
h3_v100_fp32_linear.install_from_env()
h3_v100_attention.install_from_env()
h3_v100_rms_rope.install_from_env()


def _host_mmap_disabled():
    """Keep GGUF weights disk-backed unless static loading is explicitly allowed."""
    return os.environ.get("H3_NO_HOST_MMAP", "1").lower() not in {"0", "false", "no"}


class UnetLoaderGGUFDynamicVRAMMultiGPU(gguf_nodes.UnetLoaderGGUFDynamicVRAM):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        inputs.setdefault("optional", {})["device"] = (
            multigpu.get_device_list(),
            {"default": "cuda:0"},
        )
        return inputs

    FUNCTION = "load_unet_on_device"
    CATEGORY = "dual_v100"
    TITLE = "Unet Loader (Dynamic VRAM / Device)"

    def load_unet_on_device(self, unet_name, device="cuda:0"):
        original = multigpu.get_current_device()
        multigpu.set_current_device(device)
        try:
            with multigpu.cuda_device_guard(device, reason="DualV100Dynamic.unet"):
                return super().load_unet(unet_name)
        finally:
            multigpu.set_current_device(original)


class CLIPLoaderGGUFDynamicVRAMMultiGPU(gguf_nodes.CLIPLoaderGGUFDynamicVRAM):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        inputs.setdefault("optional", {})["device"] = (
            multigpu.get_device_list(),
            {"default": "cuda:1"},
        )
        return inputs

    FUNCTION = "load_clip_on_device"
    CATEGORY = "dual_v100"
    TITLE = "CLIP Loader (Dynamic VRAM / Device)"

    def load_clip_on_device(
        self, clip_name, type="stable_diffusion", device="cuda:1"
    ):
        original = multigpu.get_current_text_encoder_device()
        multigpu.set_current_text_encoder_device(device)
        dynamic_guard.register_clip_dynamic_load_list_guard()
        try:
            with multigpu.cuda_device_guard(
                device, reason="DualV100Dynamic.text_encoder"
            ):
                return super().load_clip(clip_name, type)
        finally:
            multigpu.set_current_text_encoder_device(original)


class UnetLoaderGGUFStaticVRAMMultiGPU(UnetLoaderGGUFDynamicVRAMMultiGPU):
    """Compatibility name; safe mode routes this node to disk-backed DynamicVRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        inputs.setdefault("optional", {})["device"] = (
            multigpu.get_device_list(),
            {"default": "cuda:0"},
        )
        return inputs

    FUNCTION = "load_unet_on_device"
    CATEGORY = "dual_v100"
    TITLE = "Unet Loader (Static VRAM / Device)"

    def load_unet_on_device(self, unet_name, device="cuda:0"):
        if _host_mmap_disabled():
            return super().load_unet_on_device(unet_name, device)
        original = multigpu.get_current_device()
        multigpu.set_current_device(device)
        try:
            with multigpu.cuda_device_guard(device, reason="DualV100Static.unet"):
                return gguf_nodes.UnetLoaderGGUF.load_unet(self, unet_name)
        finally:
            multigpu.set_current_device(original)


class CLIPLoaderGGUFStaticVRAMMultiGPU(CLIPLoaderGGUFDynamicVRAMMultiGPU):
    """Compatibility name; safe mode routes this node to disk-backed DynamicVRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        inputs.setdefault("optional", {})["device"] = (
            multigpu.get_device_list(),
            {"default": "cuda:1"},
        )
        return inputs

    FUNCTION = "load_clip_on_device"
    CATEGORY = "dual_v100"
    TITLE = "CLIP Loader (Static VRAM / Device)"

    def load_clip_on_device(
        self, clip_name, type="stable_diffusion", device="cuda:1"
    ):
        if _host_mmap_disabled():
            return super().load_clip_on_device(clip_name, type, device)
        original = multigpu.get_current_text_encoder_device()
        multigpu.set_current_text_encoder_device(device)
        try:
            with multigpu.cuda_device_guard(
                device, reason="DualV100Static.text_encoder"
            ):
                return gguf_nodes.CLIPLoaderGGUF.load_clip(self, clip_name, type)
        finally:
            multigpu.set_current_text_encoder_device(original)


class VAELoaderH3Device(comfy_nodes.VAELoader):
    """Load an H3 VAE with an explicit, reliable compute device.

    ComfyUI-MultiGPU can be imported under two module names during custom-node
    discovery.  Its generic VAE wrapper then updates one module's logical
    device while ``model_management.vae_device`` reads the other module's
    state, silently constructing every VAE for cuda:0.  H3 needs the VAE on
    cuda:1 so the Q4 DiT can remain resident on cuda:0 between generations.

    The ordinary loader has not materialized the large disk-backed weights
    yet, so retargeting the VAE and patcher immediately after construction is
    zero-copy and keeps the no-host-mmap path intact.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(super().INPUT_TYPES())
        inputs.setdefault("required", {})["device"] = (
            multigpu.get_device_list(),
            {"default": "cuda:1"},
        )
        return inputs

    FUNCTION = "load_vae_on_device"
    CATEGORY = "dual_v100"
    TITLE = "VAE Loader (H3 Explicit Device)"

    def load_vae_on_device(self, vae_name, device="cuda:1"):
        # The async facade defers the decoder until the sampler has returned, so
        # only the ~688 MiB encoder coexists with the DiT forward peak.  That is
        # the only route by which the INT8 VAE fits beside the INT8 DiT at 720p:
        # at that size the two together are ~497 MiB over capacity no matter how
        # the decoder blocks are split, because it is a total-capacity problem.
        # The checkpoint's own layout (FP16 vs INT8-ConvRot) is detected from its
        # header inside the loader, not from this name.
        if (
            os.environ.get("H3_ASYNC_VAE_LOAD", "0").lower()
            not in {"0", "false", "no", "off"}
            and "minimax_h3_video_vae" in vae_name.lower()
        ):
            path = folder_paths.get_full_path_or_raise("vae", vae_name)
            facade = maybe_load_async_vae_facade(path)
            if facade is None:
                raise RuntimeError(
                    "H3_ASYNC_VAE_LOAD=1 requires two peer-accessible CUDA devices"
                )
            install_turbo_sampler_hook()
            checkpoint_format = getattr(
                getattr(facade, "async_handle", None), "checkpoint_format", "unknown"
            )
            print(
                f"[DualV100 VAE] {vae_name} loaded as capped asynchronous "
                f"{checkpoint_format} facade; decoder deferred past the DiT peak",
                flush=True,
            )
            return (facade,)

        # H3's video VAE is a resident two-stage pipeline.  Loading it through
        # the ordinary VAE patcher first would put all 5.2 GB on one card and
        # DynamicVRAM would fault it again on every decode.  The specialised
        # loader assigns each disk slice directly to its permanent GPU.
        is_h3_video_vae = "minimax_h3_video_vae" in vae_name.lower()
        vae_mp_requested = os.environ.get("H3_VAE_MP", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        if is_h3_video_vae and vae_mp_requested:
            path = folder_paths.get_full_path_or_raise("vae", vae_name)
            parallel = load_h3_video_vae_parallel(path)
            if parallel is not None:
                print(
                    f"[DualV100 VAE] {vae_name} loaded resident across "
                    f"{parallel.parallel_devices}; no DynamicVRAM reload",
                    flush=True,
                )
                return (parallel,)
        elif is_h3_video_vae:
            # Say so out loud.  A silent fall-through here looks identical to a
            # layer-MP load in the log, and the two have very different memory
            # profiles: this route puts the whole 5.2 GB decoder behind
            # DynamicVRAM on one card and re-faults it on every decode.
            print(
                f"[DualV100 VAE] {vae_name}: H3_VAE_MP=0, layer-MP skipped; "
                f"single-device DynamicVRAM route on {device}",
                flush=True,
            )

        (vae,) = super().load_vae(vae_name)
        target = torch.device(device)
        register = getattr(vae.patcher, "register_load_device", None)
        if register is not None:
            register(target)
        vae.device = target
        vae.patcher.load_device = target
        # ModelPatcher creates this marker before any large weight is loaded.
        # Keep it coherent with the explicit target for code which consults
        # the model directly rather than the patcher.
        if hasattr(vae.patcher.model, "device"):
            vae.patcher.model.device = target
        print(
            f"[DualV100 VAE] {vae_name} explicitly routed to {target}; "
            f"offload={vae.patcher.offload_device}",
            flush=True,
        )
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUFDynamicVRAMMultiGPU": UnetLoaderGGUFDynamicVRAMMultiGPU,
    "CLIPLoaderGGUFDynamicVRAMMultiGPU": CLIPLoaderGGUFDynamicVRAMMultiGPU,
    "UnetLoaderGGUFStaticVRAMMultiGPU": UnetLoaderGGUFStaticVRAMMultiGPU,
    "CLIPLoaderGGUFStaticVRAMMultiGPU": CLIPLoaderGGUFStaticVRAMMultiGPU,
    "VAELoaderH3Device": VAELoaderH3Device,
    **h3_tp_node.NODE_CLASS_MAPPINGS,
    **h3_ref2v_tp.NODE_CLASS_MAPPINGS,
    **h3_group_cache_tp.NODE_CLASS_MAPPINGS,
    **h3_qwen32_tp_node.NODE_CLASS_MAPPINGS,
    **(
        h3_te_speed_tp.NODE_CLASS_MAPPINGS
        if h3_te_speed_tp is not None
        else {}
    ),
    **latent_io.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnetLoaderGGUFDynamicVRAMMultiGPU": "Unet Loader (Dynamic VRAM / Device)",
    "CLIPLoaderGGUFDynamicVRAMMultiGPU": "CLIP Loader (Dynamic VRAM / Device)",
    "UnetLoaderGGUFStaticVRAMMultiGPU": "Unet Loader (Static VRAM / Device)",
    "CLIPLoaderGGUFStaticVRAMMultiGPU": "CLIP Loader (Static VRAM / Device)",
    "VAELoaderH3Device": "VAE Loader (H3 Explicit Device)",
    **h3_tp_node.NODE_DISPLAY_NAME_MAPPINGS,
    **h3_ref2v_tp.NODE_DISPLAY_NAME_MAPPINGS,
    **h3_group_cache_tp.NODE_DISPLAY_NAME_MAPPINGS,
    **h3_qwen32_tp_node.NODE_DISPLAY_NAME_MAPPINGS,
    **(
        h3_te_speed_tp.NODE_DISPLAY_NAME_MAPPINGS
        if h3_te_speed_tp is not None
        else {}
    ),
    **latent_io.NODE_DISPLAY_NAME_MAPPINGS,
}
