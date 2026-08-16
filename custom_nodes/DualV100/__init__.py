"""Device-aware GGUF loaders and H3 latent bridge for two GPUs."""

import copy
import importlib


gguf_nodes = importlib.import_module("custom_nodes.ComfyUI-GGUF.nodes")
multigpu = importlib.import_module("custom_nodes.ComfyUI-MultiGPU")
dynamic_guard = importlib.import_module(
    "custom_nodes.ComfyUI-MultiGPU.clip_dynamic_load_list_guard"
)
from . import h3_latent_io as latent_io


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


class UnetLoaderGGUFStaticVRAMMultiGPU(gguf_nodes.UnetLoaderGGUF):
    """Load compressed GGUF weights once and keep them on one GPU."""

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
        original = multigpu.get_current_device()
        multigpu.set_current_device(device)
        try:
            with multigpu.cuda_device_guard(device, reason="DualV100Static.unet"):
                return super().load_unet(unet_name)
        finally:
            multigpu.set_current_device(original)


class CLIPLoaderGGUFStaticVRAMMultiGPU(gguf_nodes.CLIPLoaderGGUF):
    """Load the GGUF text encoder statically on the selected GPU."""

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
        original = multigpu.get_current_text_encoder_device()
        multigpu.set_current_text_encoder_device(device)
        try:
            with multigpu.cuda_device_guard(
                device, reason="DualV100Static.text_encoder"
            ):
                return super().load_clip(clip_name, type)
        finally:
            multigpu.set_current_text_encoder_device(original)


NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUFDynamicVRAMMultiGPU": UnetLoaderGGUFDynamicVRAMMultiGPU,
    "CLIPLoaderGGUFDynamicVRAMMultiGPU": CLIPLoaderGGUFDynamicVRAMMultiGPU,
    "UnetLoaderGGUFStaticVRAMMultiGPU": UnetLoaderGGUFStaticVRAMMultiGPU,
    "CLIPLoaderGGUFStaticVRAMMultiGPU": CLIPLoaderGGUFStaticVRAMMultiGPU,
    **latent_io.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnetLoaderGGUFDynamicVRAMMultiGPU": "Unet Loader (Dynamic VRAM / Device)",
    "CLIPLoaderGGUFDynamicVRAMMultiGPU": "CLIP Loader (Dynamic VRAM / Device)",
    "UnetLoaderGGUFStaticVRAMMultiGPU": "Unet Loader (Static VRAM / Device)",
    "CLIPLoaderGGUFStaticVRAMMultiGPU": "CLIP Loader (Static VRAM / Device)",
    **latent_io.NODE_DISPLAY_NAME_MAPPINGS,
}
