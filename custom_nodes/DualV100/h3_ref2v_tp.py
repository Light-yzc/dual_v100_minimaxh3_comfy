"""Project-specific MiniMax H3 reference-to-video node for the V100 TP path.

The upstream H3 node is a V3 ``Autogrow`` node.  Its graph representation
normally carries metadata which folds ``ref_image_0`` into the ``ref_images``
mapping before execution.  Hand-authored/API workflows do not always carry
that metadata, and the raw dynamic slot then reaches the upstream method as
an unexpected keyword argument.

This node intentionally has one ordinary, fixed ``IMAGE`` input.  It reuses
the upstream ref2va implementation after constructing the same
``ref_images`` mapping that the V3 normalizer would have produced.  The
upstream implementation calls ``clip.tokenize(..., minimax_ref_items=...)``
and stores ``minimax_refs`` in conditioning; the TP-enabled H3 model consumes
that payload through ``PackedLayout(refs=...)``.  No reference-specific NCCL
protocol or alternate model-parallel route is introduced here.
"""

from __future__ import annotations

import nodes

from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ImageToVideo as _UpstreamImageToVideo,
    MiniMaxH3ReferenceToVideo as _UpstreamRef2V,
)


def _unwrap_node_output(result):
    """Return the tuple expected by a classic custom-node mapping.

    The bundled H3 conditioning nodes use the V3 API and return
    ``io.NodeOutput``.  DualV100 nodes are intentionally registered through
    the classic mapping so they remain usable by API JSON workflows; keep the
    compatibility conversion in one place.
    """
    output = result.result if hasattr(result, "result") else result
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError(
            "MiniMax H3 conditioning node returned an unexpected result: "
            f"{type(result).__name__} -> {type(output).__name__}"
        )
    return output


class MiniMaxH3ReferenceToVideoTP:
    """Fixed-input ref2va bridge for the persistent two-way H3 TP graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": (
                            "Use <Picture 1> as the persistent visual identity and "
                            "composition reference, but do not treat it as a fixed "
                            "first or last frame. Preserve the subject's identity, "
                            "colors, materials, and key visual details while "
                            "generating a natural continuous action."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {"default": 832, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 480, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32},
                ),
                "length": (
                    "INT",
                    {
                        "default": 124,
                        "min": 5,
                        "max": 3600,
                        "step": 17,
                        "tooltip": "Frame count at 24 fps; H3 snaps it to the 17k+5 grid.",
                    },
                ),
                "ref_image_size": (
                    ["match", "max"],
                    {
                        "default": "match",
                        "tooltip": (
                            "match keeps the reference near the generation area; "
                            "max uses the larger reference canvas and costs more "
                            "vision/VAE work."
                        ),
                    },
                ),
                "ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "execute"
    CATEGORY = "dual_v100/H3"
    TITLE = "MiniMax H3 Reference to Video (Persistent TP)"

    def execute(
        self,
        clip,
        vae,
        prompt,
        width,
        height,
        length,
        ref_image_size="match",
        ref_image=None,
    ):
        if ref_image is None:
            raise ValueError("MiniMaxH3ReferenceToVideoTP requires one reference IMAGE")

        # ``execute`` on the current upstream V3 node returns io.NodeOutput.
        # Feed it the already-normalized shape and unwrap the result for the
        # classic custom-node API used by this DualV100 package.
        result = _UpstreamRef2V.execute(
            clip=clip,
            vae=vae,
            audio_vae=None,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            ref_image_size=ref_image_size,
            ref_images={"ref_image_0": ref_image},
            ref_videos=None,
            ref_video_audios=None,
            ref_audios=None,
        )
        return _unwrap_node_output(result)


class MiniMaxH3ReferenceKeyframeToVideoTP:
    """Select ref2va or first/last-keyframe conditioning on one TP graph.

    ``reference_image`` is sent through the project's fixed-input ref2va
    bridge.  ``first_frame``/``last_frame`` are sent through the upstream H3
    keyframe implementation, including its shared geometry transform.  The
    mode is deliberately a regular widget instead of two parallel
    conditioning branches: only one encoder path runs and the persistent H3
    TP/Qwen/VAE objects are not loaded or unloaded when the mode changes.
    """

    MODES = ("first_last_frames", "reference_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    list(cls.MODES),
                    {
                        "default": "first_last_frames",
                        "tooltip": (
                            "Choose first/last frame conditioning or a persistent "
                            "reference-image conditioning path."
                        ),
                    },
                ),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": (
                            "Use the selected image inputs as visual guidance. "
                            "Preserve identity and composition while generating "
                            "natural, coherent motion."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {"default": 832, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 480, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32},
                ),
                "length": (
                    "INT",
                    {
                        "default": 124,
                        "min": 5,
                        "max": 3600,
                        "step": 17,
                        "tooltip": "Frame count at 24 fps; H3 snaps it to the 17k+5 grid.",
                    },
                ),
                "ref_image_size": (
                    ["match", "max"],
                    {
                        "default": "match",
                        "tooltip": (
                            "Only used in reference_image mode. 'max' increases "
                            "vision/VAE and sampling memory."
                        ),
                    },
                ),
            },
            "optional": {
                "reference_image": ("IMAGE", {"lazy": True}),
                "first_frame": ("IMAGE", {"lazy": True}),
                "last_frame": ("IMAGE", {"lazy": True}),
            },
        }

    def check_lazy_status(self, mode, **kwargs):
        """Evaluate only the image sockets selected by ``mode``.

        The workflow intentionally keeps all three sockets connected so a user
        can switch modes without rewiring the graph.  Marking them lazy keeps
        the inactive LoadImage branches out of the execution list and, more
        importantly, prevents a future image/video-producing source from
        allocating its payload unnecessarily.
        """
        mode = str(mode).strip().lower()
        if mode == "reference_image":
            selected = ("reference_image",)
        elif mode == "first_last_frames":
            selected = ("first_frame", "last_frame")
        else:
            # Let execute() produce the more useful invalid-mode error.  Do
            # not request arbitrary branches for a malformed widget value.
            return []
        return [name for name in selected if name in kwargs and kwargs[name] is None]

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "execute"
    CATEGORY = "dual_v100/H3"
    TITLE = "MiniMax H3 Reference / First+Last Frame (Persistent TP)"

    def execute(
        self,
        mode,
        clip,
        vae,
        prompt,
        width,
        height,
        length,
        ref_image_size="match",
        reference_image=None,
        first_frame=None,
        last_frame=None,
    ):
        mode = str(mode).strip().lower()
        if mode not in self.MODES:
            raise ValueError(
                f"unsupported MiniMax H3 conditioning mode {mode!r}; "
                f"choose one of {', '.join(self.MODES)}"
            )

        if mode == "reference_image":
            if reference_image is None:
                raise ValueError(
                    "reference_image mode requires the reference_image input"
                )
            result = _UpstreamRef2V.execute(
                clip=clip,
                vae=vae,
                audio_vae=None,
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                ref_image_size=ref_image_size,
                ref_images={"ref_image_0": reference_image},
                ref_videos=None,
                ref_video_audios=None,
                ref_audios=None,
            )
            return _unwrap_node_output(result)

        # Preserve the upstream semantics: first_frame and last_frame are
        # independently optional, so this mode also remains usable for plain
        # text-to-video when both image sockets are left unconnected.
        result = _UpstreamImageToVideo.execute(
            clip=clip,
            vae=vae,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            first_frame=first_frame,
            last_frame=last_frame,
        )
        return _unwrap_node_output(result)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ReferenceToVideoTP": MiniMaxH3ReferenceToVideoTP,
    "MiniMaxH3ReferenceKeyframeToVideoTP": MiniMaxH3ReferenceKeyframeToVideoTP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ReferenceToVideoTP": "MiniMax H3 Reference to Video (Persistent TP)",
    "MiniMaxH3ReferenceKeyframeToVideoTP": (
        "MiniMax H3 Reference / First+Last Frame (Persistent TP)"
    ),
}


__all__ = [
    "MiniMaxH3ReferenceToVideoTP",
    "MiniMaxH3ReferenceKeyframeToVideoTP",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
