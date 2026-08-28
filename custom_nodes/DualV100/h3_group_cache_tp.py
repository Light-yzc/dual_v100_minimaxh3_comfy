"""ComfyUI configuration node for TP-aware Adaptive Group Residual Cache."""

from __future__ import annotations

import copy

from . import h3_q4_cache
from .h3_te_speed_tp import _find_h3_model


class AdaptiveGroupResidualCacheMiniMaxH3TP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Group Cache ON",
                        "label_off": "Group Cache OFF",
                    },
                ),
                "warm_blocks": (
                    "INT",
                    {"default": 8, "min": 0, "max": 49, "step": 1},
                ),
                "num_groups": (
                    "INT",
                    {"default": 4, "min": 1, "max": 16, "step": 1},
                ),
                "metric": (
                    ["relative_l1", "relative_l2", "cosine"],
                    {"default": "relative_l1"},
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": h3_q4_cache.DEFAULT_GROUP_THRESHOLD,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": (
                            "Q4_0-vs-Q4_0 group-input threshold. Start at 0.005; "
                            "the old FP32-vs-Q4 noise floor does not apply."
                        ),
                    },
                ),
                "max_cache": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 20,
                        "step": 1,
                        "tooltip": "0 means unlimited consecutive cache hits.",
                    },
                ),
                "reference_mode": (
                    ["last_full", "previous_step"],
                    {"default": "last_full"},
                ),
            },
            "optional": {
                "calibration_mode": (
                    ["off", "collect"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Collect bounded AdaLN/cache-risk features for offline "
                            "policy fitting. It never changes the cache decision."
                        ),
                    },
                ),
                "condition_metric": (
                    ["none", "gates", "all_adaln"],
                    {
                        "default": "none",
                        "tooltip": (
                            "AdaLN feature family to log during calibration. "
                            "Use all_adaln for research; it is not a production gate."
                        ),
                    },
                ),
                "collect_block_stats": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable scalar benchmark logging. Leave off for "
                            "production timing and lower overhead."
                        ),
                    },
                ),
                "benchmark_ground_truth": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Secretly executes a true FULL group after CACHE for "
                            "local residual-error measurement. Benchmark only."
                        ),
                    },
                ),
                "oracle_max_mib": (
                    "INT",
                    {"default": 256, "min": 16, "max": 4096, "step": 16},
                ),
                "cache_chunk_rows": (
                    "INT",
                    {
                        "default": h3_q4_cache.DEFAULT_CACHE_CHUNK_ROWS,
                        "min": 32,
                        "max": 2048,
                        "step": 32,
                    },
                ),
                "device": (
                    ["cpu", "gpu"],
                    {
                        "default": "cpu",
                        "tooltip": (
                            "Q4_0 cache storage. CPU minimizes persistent VRAM; "
                            "dequantization always uses bounded row chunks."
                        ),
                    },
                ),
                "feature_mode": (
                    ["q4", "signature"],
                    {
                        "default": "q4",
                        "tooltip": (
                            "q4 keeps the historical full-input gate; signature "
                            "keeps only a bounded stratified FP32 sample. The "
                            "latter is an opt-in quality/speed experiment."
                        ),
                    },
                ),
                "signature_max_tokens": (
                    "INT",
                    {
                        "default": h3_q4_cache.DEFAULT_SIGNATURE_MAX_TOKENS,
                        "min": 4,
                        "max": 16384,
                        "step": 4,
                        "tooltip": (
                            "Maximum sampled token rows per group in signature "
                            "mode; kept on CPU and independent of full sequence."
                        ),
                    },
                ),
                "signature_hidden_samples": (
                    "INT",
                    {
                        "default": h3_q4_cache.DEFAULT_SIGNATURE_HIDDEN_SAMPLES,
                        "min": 4,
                        "max": 256,
                        "step": 4,
                    },
                ),
                "signature_aggregation": (
                    ["weighted", "max_segment"],
                    {
                        "default": "weighted",
                        "tooltip": (
                            "weighted matches the compact global metric; "
                            "max_segment is a more conservative modality gate."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "dual_v100/H3"
    TITLE = "Adaptive Group Residual Cache (TP, Q4_0)"

    def patch(
        self,
        model,
        enabled,
        warm_blocks,
        num_groups,
        metric,
        threshold,
        max_cache,
        reference_mode,
        calibration_mode="off",
        condition_metric="none",
        collect_block_stats=False,
        benchmark_ground_truth=False,
        oracle_max_mib=256,
        cache_chunk_rows=h3_q4_cache.DEFAULT_CACHE_CHUNK_ROWS,
        device="cpu",
        feature_mode="q4",
        signature_max_tokens=h3_q4_cache.DEFAULT_SIGNATURE_MAX_TOKENS,
        signature_hidden_samples=h3_q4_cache.DEFAULT_SIGNATURE_HIDDEN_SAMPLES,
        signature_aggregation="weighted",
    ):
        h3 = _find_h3_model(model)
        if h3 is None:
            raise ValueError(
                "AdaptiveGroupResidualCacheMiniMaxH3TP requires MiniMaxH3Model"
            )
        if not getattr(h3, "h3_tp_enabled", False) or not hasattr(
            h3, "h3_tp_runtime"
        ):
            raise ValueError(
                "Apply MiniMaxH3TensorParallel before Adaptive Group Cache"
            )
        warm_blocks = int(warm_blocks)
        num_groups = int(num_groups)
        remaining = 50 - warm_blocks
        if not 0 <= warm_blocks < 50:
            raise ValueError("warm_blocks must be in [0, 49]")
        if not 1 <= num_groups <= remaining:
            raise ValueError(
                f"num_groups must be in [1, {remaining}] for warm_blocks={warm_blocks}"
            )
        if metric not in {"relative_l1", "relative_l2", "cosine"}:
            raise ValueError(f"unsupported metric {metric!r}")
        if float(threshold) < 0.0 or int(max_cache) < 0:
            raise ValueError("threshold/max_cache must be non-negative")
        if reference_mode not in {"last_full", "previous_step"}:
            raise ValueError(f"unsupported reference_mode {reference_mode!r}")
        calibration_mode = str(calibration_mode).lower()
        condition_metric = str(condition_metric).lower()
        if calibration_mode not in {"off", "collect"}:
            raise ValueError(
                f"unsupported calibration_mode {calibration_mode!r}"
            )
        if condition_metric not in {"none", "gates", "all_adaln"}:
            raise ValueError(
                f"unsupported condition_metric {condition_metric!r}"
            )
        if calibration_mode == "off" and condition_metric != "none":
            raise ValueError(
                "condition_metric requires calibration_mode=collect; "
                "the production cache must not pay collection overhead"
            )
        feature_mode = h3_q4_cache.normalize_feature_mode(feature_mode)
        signature_max_tokens = int(signature_max_tokens)
        signature_hidden_samples = int(signature_hidden_samples)
        if signature_max_tokens <= 0 or signature_hidden_samples <= 0:
            raise ValueError(
                "signature_max_tokens/signature_hidden_samples must be positive"
            )
        if signature_aggregation not in {"weighted", "max_segment"}:
            raise ValueError(
                f"unsupported signature_aggregation {signature_aggregation!r}"
            )
        cache_format = h3_q4_cache.normalize_q4_format("ggml_q4_0")

        new_model = model.clone()
        transformer_options = copy.deepcopy(
            new_model.model_options.setdefault("transformer_options", {})
        )
        transformer_options.pop("h3_tp_te_speed", None)
        transformer_options["h3_tp_group_cache"] = {
            "enabled": bool(enabled),
            "warm_blocks": warm_blocks,
            "num_groups": num_groups,
            "metric": str(metric),
            "threshold": float(threshold),
            "max_cache": int(max_cache),
            "reference_mode": str(reference_mode),
            "calibration_mode": calibration_mode,
            "condition_metric": condition_metric,
            "collect_block_stats": bool(collect_block_stats),
            "benchmark_ground_truth": bool(benchmark_ground_truth),
            "oracle_max_mib": int(oracle_max_mib),
            "cache_chunk_rows": int(cache_chunk_rows),
            "device": str(device),
            "cache_format": cache_format,
            "epsilon": 1e-6,
            "feature_mode": feature_mode,
            "signature_max_tokens": signature_max_tokens,
            "signature_hidden_samples": signature_hidden_samples,
            "signature_aggregation": str(signature_aggregation),
        }
        new_model.model_options["transformer_options"] = transformer_options
        print(
            "[H3 TP Group Cache] configured: "
            f"enabled={bool(enabled)}, warm={warm_blocks}, groups={num_groups}, "
            f"metric={metric}, threshold={float(threshold):.6f}, "
            f"max_cache={int(max_cache)}, reference={reference_mode}, "
            f"format=Q4_0, device={device}, oracle={bool(benchmark_ground_truth)}, "
            f"calibration={calibration_mode}/{condition_metric}, "
            f"feature={feature_mode}/{signature_aggregation}",
            flush=True,
        )
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "AdaptiveGroupResidualCacheMiniMaxH3TP": AdaptiveGroupResidualCacheMiniMaxH3TP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveGroupResidualCacheMiniMaxH3TP": (
        "Adaptive Group Residual Cache (TP, Q4_0)"
    ),
}


__all__ = [
    "AdaptiveGroupResidualCacheMiniMaxH3TP",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
