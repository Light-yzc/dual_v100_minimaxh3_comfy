"""Disk bridge for MiniMax H3's nested video and audio latent."""

import os

import torch

import comfy.nested_tensor


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


NODE_CLASS_MAPPINGS = {
    "SaveMiniMaxH3Latent": SaveMiniMaxH3Latent,
    "LoadMiniMaxH3Latent": LoadMiniMaxH3Latent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveMiniMaxH3Latent": "Save MiniMax H3 Latent (disk bridge)",
    "LoadMiniMaxH3Latent": "Load MiniMax H3 Latent (disk bridge)",
}
