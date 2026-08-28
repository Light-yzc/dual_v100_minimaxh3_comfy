"""TP-aware controller for the optional MiniMax H3 block cache.

The upstream TE-Speed node installs a ``block_loop`` patch on the stock
50-block MiniMax model.  This project replaces that block list with one
``PersistentH3TPBlocks`` proxy, so the upstream patch cannot see or skip the
real blocks.  This node only writes a small immutable configuration into
``transformer_options``; the actual cache decision and block-range execution
live in ``h3_tp_runtime``/``h3_tp_backbone`` and are coordinated by rank 0.
"""

from __future__ import annotations

import copy

from . import h3_q4_cache


def _find_h3_model(model):
    current = getattr(model, "model", None)
    for _ in range(12):
        if current is None:
            return None
        if type(current).__name__ == "MiniMaxH3Model":
            return current
        next_model = None
        for name in ("model", "inner_model", "diffusion_model", "unet_model"):
            next_model = getattr(current, name, None)
            if next_model is not None:
                break
        current = next_model
    return None


class TESpeedMiniMaxH3TP:
    """Optional residual block cache for the project's persistent H3 TP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "TE-Speed ON",
                        "label_off": "TE-Speed OFF",
                        "tooltip": (
                            "Experimental TP-aware residual cache. OFF keeps the "
                            "original full 50-layer TP route."
                        ),
                    },
                ),
                "processing_control_value": (
                    "FLOAT",
                    {
                        "default": 0.12,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Normalized sigma-delta threshold. Set 0 to keep "
                            "the exact full TP route."
                        ),
                    },
                ),
                "processing_percent_1": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 0.49,
                        "step": 0.01,
                        "tooltip": "Beginning of the cache window; earlier steps stay full.",
                    },
                ),
                "processing_percent_2": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.51,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "End of the cache window; later steps stay full.",
                    },
                ),
                "mcs": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": (
                            "Maximum consecutive cache steps. Set 0 to disable "
                            "cache approximation."
                        ),
                    },
                ),
                "device": (
                    ["auto", "cpu", "gpu"],
                    {
                        "default": "cpu",
                        "tooltip": (
                            "Where one standard Q4_0 residual is kept. TP auto "
                            "maps to CPU; dequantization/add is row-chunked."
                        ),
                    },
                ),
            },
            "optional": {
                "cache_depth": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 0.95,
                        "step": 0.05,
                        "tooltip": (
                            "Legacy compatibility only: fraction of trailing H3 "
                            "blocks served from cache. Exact tail_blocks takes "
                            "precedence when present."
                        ),
                    },
                ),
                "tail_blocks": (
                    "INT",
                    {
                        "default": 12,
                        "min": 4,
                        "max": 49,
                        "step": 1,
                        "tooltip": (
                            "Exact number of trailing H3 blocks replaced by one "
                            "whole-tail Q4_0 residual. Use 42 to match an "
                            "eight-block warm prefix."
                        ),
                    },
                ),
                "collect_block_stats": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Collect scalar stats",
                        "label_off": "Stats off",
                        "tooltip": (
                            "Record scalar-only block-boundary, packed-segment, "
                            "sigma, timing and memory data. No activation tensor is "
                            "written to disk."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "dual_v100/H3"
    TITLE = "TE-Speed MiniMax H3 (TP-aware, experimental)"

    def patch(
        self,
        model,
        processing_control_value,
        processing_percent_1,
        processing_percent_2,
        mcs,
        device,
        cache_depth=0.75,
        tail_blocks=None,
        collect_block_stats=False,
        enabled=False,
    ):
        h3 = _find_h3_model(model)
        if h3 is None:
            raise ValueError("TESpeedMiniMaxH3TP requires a MiniMaxH3Model")
        if not getattr(h3, "h3_tp_enabled", False) or not hasattr(
            h3, "h3_tp_runtime"
        ):
            raise ValueError(
                "TESpeedMiniMaxH3TP requires MiniMaxH3TensorParallel first; "
                "the stock TE-Speed node is not compatible with this TP route"
            )
        if len(getattr(h3, "blocks", ())) != 1:
            raise ValueError(
                "TESpeedMiniMaxH3TP expected the one-block PersistentH3TPBlocks "
                "proxy; refusing to patch a non-TP H3 model"
            )

        cache_depth = float(cache_depth)
        if not 0.0 <= cache_depth <= 0.95:
            raise ValueError(f"cache_depth must be in [0, 0.95], got {cache_depth}")
        if tail_blocks is not None:
            tail_blocks = int(tail_blocks)
            if not 1 <= tail_blocks <= 49:
                raise ValueError(
                    f"tail_blocks must be in [1, 49], got {tail_blocks}"
                )
        processing_percent_1 = float(processing_percent_1)
        processing_percent_2 = float(processing_percent_2)
        if not 0.0 <= processing_percent_1 <= processing_percent_2 <= 1.0:
            raise ValueError(
                "processing window must satisfy 0 <= percent_1 <= percent_2 <= 1"
            )

        new_model = model.clone()
        transformer_options = copy.deepcopy(
            new_model.model_options.setdefault("transformer_options", {})
        )
        transformer_options["h3_tp_te_speed"] = {
            "enabled": bool(enabled),
            "control_value": float(processing_control_value),
            "start_percent": processing_percent_1,
            "end_percent": processing_percent_2,
            "mcs": int(mcs),
            "device": str(device).lower(),
            "cache_depth": cache_depth,
            # ``None`` is intentionally preserved for API workflows created
            # before the block-level control existed.  The runtime derives the
            # old rounded boundary only for those workflows; every new UI
            # workflow sends an exact integer tail.
            "tail_blocks": tail_blocks,
            "collect_block_stats": bool(collect_block_stats),
            "cache_format": h3_q4_cache.Q4_FORMAT,
        }
        transformer_options.pop("h3_tp_group_cache", None)
        new_model.model_options["transformer_options"] = transformer_options
        print(
            "[H3 TP TE-Speed] configured optional block cache: "
            f"enabled={bool(enabled)}, "
            f"threshold={float(processing_control_value):.3f}, "
            f"window={processing_percent_1:.2f}..{processing_percent_2:.2f}, "
            f"mcs={int(mcs)}, "
            f"tail={tail_blocks if tail_blocks is not None else f'legacy:{cache_depth:.2f}'}, "
            f"stats={bool(collect_block_stats)}, format=Q4_0, device={device}; "
            "cache is TP-runtime aware and is not the upstream block hook",
            flush=True,
        )
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "TESpeedMiniMaxH3TP": TESpeedMiniMaxH3TP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TESpeedMiniMaxH3TP": "TE-Speed MiniMax H3 (TP-aware, experimental)",
}


__all__ = [
    "TESpeedMiniMaxH3TP",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
