"""ComfyUI-side controller for a persistent two-process H3 NCCL worker."""

from __future__ import annotations

import atexit
import gc
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from . import h3_q4_cache
from . import h3_qwen32_q2_tp as qwen32
from . import h3_tp_backbone as tp


DEFAULT_RESULTS_DIR = Path(
    os.environ.get(
        "H3_TP_RESULTS_DIR",
        "/home/regen/code/minimax_v100/results/h3_tp_e2e",
    )
)


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: str
    lora_path: str
    egrid_path: str
    # ``auto`` selects Q4 for GGUF and the official INT8-ConvRot reader for
    # safetensors.  Keep this explicit in the shared runtime so rank 0/rank 1
    # cannot accidentally load different matrix backends.
    dit_format: str = "auto"
    lora_strength: float = 1.0
    staging_mib: int = 4
    chunk_rows: int = 2048
    timeout_seconds: int = 900
    results_dir: str = str(DEFAULT_RESULTS_DIR)
    qwen_model_path: str = ""
    qwen_staging_mib: int = 4
    qwen_residency: str = "evict"
    qwen_keep_layers: int = 0
    qwen_cache_dequantized: bool = False
    # Qwen32 can use the output-row TP protocol (the historical experimental
    # route) or the decoupled complete-layer MP backend.  Keep TP as the
    # compatibility default; deployment must opt into MP explicitly.
    qwen_mode: str = "tp"


def _normalize_qwen_mode(value: str | None) -> str:
    """Normalize the Qwen backend selector without importing the MP module."""

    requested = "tp" if value is None else str(value)
    normalized = requested.strip().lower().replace("-", "_")
    if normalized in {"tp", "output_row_tp", "outputrow_tp"}:
        return "tp"
    if normalized in {"mp", "layer_mp", "layer_parallel", "layerpipeline"}:
        return "mp"
    raise ValueError(
        "H3_QWEN32_Q2_MODE must be tp or mp, "
        f"got {value!r}"
    )


class _GroupCacheController:
    """Rank-0 schedule/reset state for Adaptive Group Residual Cache.

    The actual Q4 tensors live independently on both ranks.  This controller
    only owns small scalar state and tells both workers when a new generation
    or shape requires invalidation.  Group decisions themselves are made by
    rank 0 inside the backbone protocol and broadcast before any group range.
    """

    def __init__(self, block_count: int = tp.LAYERS) -> None:
        self.block_count = int(block_count)
        self.config_key: tuple[object, ...] | None = None
        self.shape_key: tuple[object, ...] | None = None
        self.schedule_key: tuple[float, ...] | None = None
        self.last_sigma_raw: float | None = None
        self.sigma_delta: float | None = None
        self.step = -1
        self.generation_id = 0

    @staticmethod
    def _scalar(value: Any) -> float | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().reshape(-1)[0].float().item())
        try:
            return float(value[0] if isinstance(value, (list, tuple)) else value)
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _schedule(value: Any) -> tuple[float, ...]:
        if value is None:
            return ()
        if torch.is_tensor(value):
            return tuple(float(item) for item in value.detach().reshape(-1).float().cpu())
        try:
            return tuple(float(item) for item in value)
        except TypeError:
            return (float(value),)

    @staticmethod
    def _config(value: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {
                "enabled": False,
                "warm_blocks": 8,
                "num_groups": 4,
                "metric": "relative_l1",
                "threshold": h3_q4_cache.DEFAULT_GROUP_THRESHOLD,
                "max_cache": 2,
                "reference_mode": "last_full",
                "calibration_mode": "off",
                "condition_metric": "none",
                "cache_policy": "cpu",
                "cache_chunk_rows": h3_q4_cache.DEFAULT_CACHE_CHUNK_ROWS,
                "cache_format": h3_q4_cache.Q4_FORMAT,
                "epsilon": 1e-6,
                "benchmark_ground_truth": False,
                "oracle_max_mib": 256,
                "collect_block_stats": False,
                "feature_mode": "q4",
                "signature_max_tokens": h3_q4_cache.DEFAULT_SIGNATURE_MAX_TOKENS,
                "signature_hidden_samples": h3_q4_cache.DEFAULT_SIGNATURE_HIDDEN_SAMPLES,
                "signature_aggregation": "weighted",
            }
        config = {
            "enabled": bool(value.get("enabled", True)),
            "warm_blocks": int(value.get("warm_blocks", 8)),
            "num_groups": int(value.get("num_groups", 4)),
            "metric": str(value.get("metric", "relative_l1")).lower(),
            "threshold": float(
                value.get("threshold", h3_q4_cache.DEFAULT_GROUP_THRESHOLD)
            ),
            "max_cache": int(value.get("max_cache", 2)),
            "reference_mode": str(value.get("reference_mode", "last_full")).lower(),
            # Calibration is deliberately an explicit collector.  There is
            # no runtime policy-loading path yet, so a malformed/unknown mode
            # fails closed instead of silently changing cache decisions.
            "calibration_mode": str(value.get("calibration_mode", "off")).lower(),
            "condition_metric": str(value.get("condition_metric", "none")).lower(),
            "cache_policy": str(value.get("device", value.get("cache_policy", "cpu"))).lower(),
            "cache_chunk_rows": int(
                value.get(
                    "cache_chunk_rows", h3_q4_cache.DEFAULT_CACHE_CHUNK_ROWS
                )
            ),
            "cache_format": h3_q4_cache.normalize_q4_format(
                value.get("cache_format", h3_q4_cache.Q4_FORMAT)
            ),
            "epsilon": float(value.get("epsilon", 1e-6)),
            "benchmark_ground_truth": bool(value.get("benchmark_ground_truth", False)),
            "oracle_max_mib": int(value.get("oracle_max_mib", 256)),
            "collect_block_stats": bool(value.get("collect_block_stats", False)),
            "feature_mode": h3_q4_cache.normalize_feature_mode(
                value.get("feature_mode", "q4")
            ),
            "signature_max_tokens": int(
                value.get(
                    "signature_max_tokens", h3_q4_cache.DEFAULT_SIGNATURE_MAX_TOKENS
                )
            ),
            "signature_hidden_samples": int(
                value.get(
                    "signature_hidden_samples",
                    h3_q4_cache.DEFAULT_SIGNATURE_HIDDEN_SAMPLES,
                )
            ),
            "signature_aggregation": str(
                value.get("signature_aggregation", "weighted")
            ).lower(),
        }
        if config["cache_policy"] == "auto":
            config["cache_policy"] = "cpu"
        if config["cache_policy"] not in {"cpu", "gpu"}:
            raise ValueError(f"invalid H3 group cache device {config['cache_policy']!r}")
        if config["metric"] not in {"relative_l1", "relative_l2", "cosine"}:
            raise ValueError(f"invalid H3 group cache metric {config['metric']!r}")
        if config["warm_blocks"] < 0 or config["warm_blocks"] >= 50:
            raise ValueError("H3 group warm_blocks must be in [0, 49]")
        if config["num_groups"] < 1 or config["num_groups"] > 50 - config["warm_blocks"]:
            raise ValueError("H3 group num_groups exceeds the remaining block count")
        if config["threshold"] < 0.0 or config["max_cache"] < 0:
            raise ValueError("H3 group threshold/max_cache must be non-negative")
        if config["reference_mode"] not in {"last_full", "previous_step"}:
            raise ValueError(f"invalid H3 group reference mode {config['reference_mode']!r}")
        if config["calibration_mode"] not in {"off", "collect"}:
            raise ValueError(
                f"invalid H3 group calibration mode {config['calibration_mode']!r}"
            )
        if config["condition_metric"] not in {"none", "gates", "all_adaln"}:
            raise ValueError(
                f"invalid H3 group condition metric {config['condition_metric']!r}"
            )
        if config["signature_aggregation"] not in {"weighted", "max_segment"}:
            raise ValueError(
                "invalid H3 group signature aggregation "
                f"{config['signature_aggregation']!r}"
            )
        if (
            config["calibration_mode"] == "off"
            and config["condition_metric"] != "none"
        ):
            raise ValueError(
                "H3 group condition_metric requires calibration_mode=collect"
            )
        if (
            config["cache_chunk_rows"] <= 0
            or config["oracle_max_mib"] <= 0
            or config["signature_max_tokens"] <= 0
            or config["signature_hidden_samples"] <= 0
        ):
            raise ValueError(
                "H3 group cache_chunk_rows/oracle_max_mib/signature sizes "
                "must be positive"
            )
        return config

    def plan(
        self,
        value: dict[str, Any] | None,
        sigma_value: Any,
        sample_sigmas: Any,
        shape_key: tuple[object, ...],
    ) -> dict[str, Any]:
        config = self._config(value)
        schedule = self._schedule(sample_sigmas)
        config_key = tuple(sorted(config.items()))
        schedule_key = tuple(round(item, 7) for item in schedule)
        reset = (
            config_key != self.config_key
            or shape_key != self.shape_key
            or schedule_key != self.schedule_key
        )
        raw_sigma = self._scalar(sigma_value)
        if raw_sigma is None:
            reset = True
        elif self.last_sigma_raw is not None and raw_sigma > self.last_sigma_raw + 1e-6:
            reset = True
        if reset:
            self.config_key = config_key
            self.shape_key = shape_key
            self.schedule_key = schedule_key
            self.last_sigma_raw = raw_sigma
            self.sigma_delta = None
            self.step = 0
            self.generation_id += 1
        else:
            same_sigma = (
                raw_sigma is not None
                and self.last_sigma_raw is not None
                and abs(raw_sigma - self.last_sigma_raw) <= 1e-6
            )
            if not same_sigma:
                previous_sigma = self.last_sigma_raw
                self.step += 1
                self.last_sigma_raw = raw_sigma
                self.sigma_delta = (
                    None
                    if previous_sigma is None or raw_sigma is None
                    else abs(previous_sigma - raw_sigma)
                )
            else:
                self.sigma_delta = 0.0
        config["clear_cache"] = bool(reset)
        config["generation_id"] = self.generation_id
        config["step"] = self.step
        config["sigma_raw"] = raw_sigma
        config["sigma_delta"] = self.sigma_delta
        config["schedule"] = list(schedule)
        if raw_sigma is None:
            config["enabled"] = False
        return config


class _TESpeedController:
    """Small rank-0 state machine for the TP-aware TE-Speed cache.

    The controller owns only scalar schedule state.  The actual residual is
    kept by ``H3TPResidualCache`` on both ranks, so the decision is made once
    and the exact same block range is sent to rank 1.  This is intentionally
    independent from the upstream ``block_loop`` hook, which cannot see the
    real blocks after this project replaces them with an external TP worker.
    """

    def __init__(self, block_count: int = tp.LAYERS) -> None:
        self.block_count = int(block_count)
        self.config_key: tuple[object, ...] | None = None
        self.schedule_key: tuple[object, ...] | None = None
        self.shape_key: tuple[object, ...] | None = None
        self.start_sigma: float | None = None
        self.sigma_scale = 1.0
        self.last_sigma_raw: float | None = None
        self.last_sigma_normalized: float | None = None
        self.current_sigma_raw: float | None = None
        self.current_sigma_normalized: float | None = None
        self.current_sigma_delta: float | None = None
        self.current_position: float | None = None
        self.step = -1
        self.total_steps: int | None = None
        self.generation_id = 0
        self.consecutive_skips = 0
        self.last_mode = "full"
        self.full_steps = 0
        self.cache_hits = 0
        self.total_blocks = 0
        self.skipped_blocks = 0
        self._active_config = self._config(None)

    @staticmethod
    def _scalar(value: Any) -> float | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().reshape(-1)[0].float().item())
        try:
            value = list(value) if not isinstance(value, (str, bytes)) else [value]
            return float(value[0]) if value else None
        except (TypeError, ValueError, IndexError):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

    @staticmethod
    def _schedule(value: Any) -> tuple[float, ...]:
        if value is None:
            return ()
        if torch.is_tensor(value):
            if value.numel() == 0:
                return ()
            values = value.detach().reshape(-1).float().cpu().tolist()
        else:
            try:
                values = list(value)
            except TypeError:
                values = [value]
        return tuple(float(item) for item in values)

    @staticmethod
    def _config(te_speed: dict[str, Any] | None) -> dict[str, Any]:
        if te_speed is None:
            return {
                "control_value": 0.0,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "mcs": 0,
                "cache_depth": 0.0,
                "tail_blocks": None,
                "tail_source": "legacy_cache_depth",
                "collect_block_stats": False,
                "cache_policy": "cpu",
                "cache_format": h3_q4_cache.Q4_FORMAT,
                "enabled": False,
                "environment_enabled": os.environ.get(
                    "H3_TP_TE_SPEED", "1"
                ).strip().lower()
                not in {"0", "false", "off", "no"},
            }
        if not isinstance(te_speed, dict):
            raise TypeError(f"h3_tp_te_speed must be a dict, got {type(te_speed).__name__}")
        control = float(te_speed.get("control_value", 0.0))
        start = float(te_speed.get("start_percent", 0.1))
        end = float(te_speed.get("end_percent", 0.9))
        mcs = int(te_speed.get("mcs", 2))
        depth = float(te_speed.get("cache_depth", 0.75))
        requested_tail = te_speed.get("tail_blocks")
        tail_blocks = None if requested_tail is None else int(requested_tail)
        collect_block_stats = bool(te_speed.get("collect_block_stats", False))
        policy = str(te_speed.get("device", "cpu")).lower()
        cache_format = h3_q4_cache.normalize_q4_format(
            te_speed.get("cache_format", h3_q4_cache.Q4_FORMAT)
        )
        requested_enabled = bool(te_speed.get("enabled", True))
        environment_enabled = os.environ.get("H3_TP_TE_SPEED", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        if policy == "auto":
            policy = "cpu"
        if control < 0.0 or not 0.0 <= start <= end <= 1.0:
            raise ValueError(
                "invalid H3 TP TE-Speed config: control>=0 and "
                "0 <= start_percent <= end_percent <= 1 are required"
            )
        if (
            mcs < 0
            or not 0.0 <= depth <= 0.95
            or (tail_blocks is not None and tail_blocks <= 0)
        ):
            raise ValueError(
                "invalid H3 TP TE-Speed config: mcs>=0 and "
                "0 <= cache_depth <= 0.95 and tail_blocks>0 are required"
            )
        if policy not in {"cpu", "gpu"}:
            raise ValueError(f"invalid H3 TP TE-Speed cache device {policy!r}")
        return {
            "control_value": control,
            "start_percent": start,
            "end_percent": end,
            "mcs": mcs,
            "cache_depth": depth,
            "tail_blocks": tail_blocks,
            "tail_source": (
                "tail_blocks" if tail_blocks is not None else "legacy_cache_depth"
            ),
            "collect_block_stats": collect_block_stats,
            "cache_policy": policy,
            "cache_format": cache_format,
            # The node switch is the normal control.  The environment switch
            # is a deployment-level kill switch for old workflows that have
            # the experimental node enabled; it never turns caching on by
            # itself and defaults to allowing the node setting.
            "enabled": (
                requested_enabled
                and environment_enabled
                and control > 0.0
                and mcs > 0
                and (
                    (tail_blocks is not None and tail_blocks > 0)
                    or (tail_blocks is None and depth > 0.0)
                )
            ),
            "environment_enabled": environment_enabled,
        }

    def _reset_schedule(self) -> None:
        self.start_sigma = None
        self.sigma_scale = 1.0
        self.last_sigma_raw = None
        self.last_sigma_normalized = None
        self.current_sigma_raw = None
        self.current_sigma_normalized = None
        self.current_sigma_delta = None
        self.current_position = None
        self.step = -1
        self.total_steps = None
        self.consecutive_skips = 0
        self.last_mode = "full"
        self.full_steps = 0
        self.cache_hits = 0
        self.total_blocks = 0
        self.skipped_blocks = 0

    def _warm_blocks(self, config: dict[str, Any]) -> int:
        tail_blocks = config.get("tail_blocks")
        if tail_blocks is not None:
            tail_blocks = int(tail_blocks)
            if not 0 < tail_blocks < self.block_count:
                raise ValueError(
                    f"tail_blocks must be in [1, {self.block_count - 1}], "
                    f"got {tail_blocks}"
                )
            return self.block_count - tail_blocks
        depth = float(config["cache_depth"])
        return max(
            0,
            min(
                self.block_count - 1,
                round(self.block_count * (1.0 - depth)),
            ),
        )

    def _position(self, sigma_normalized: float, schedule: tuple[float, ...]) -> float:
        if len(schedule) > 1:
            self.total_steps = len(schedule) - 1
            index = min(
                range(len(schedule)),
                key=lambda item: abs(schedule[item] - sigma_normalized),
            )
            return min(1.0, max(0.0, index / self.total_steps))
        if self.start_sigma is not None and self.start_sigma > 0.0:
            return min(
                1.0,
                max(0.0, (self.start_sigma / self.sigma_scale - sigma_normalized)
                    / (self.start_sigma / self.sigma_scale)),
            )
        return 1.0

    def _summary(
        self,
        mode: str,
        warm_blocks: int,
        clear_cache: bool,
        *,
        reset_block_stats: bool = False,
    ) -> dict[str, Any]:
        capture_cache = mode == "full" and self._active_config["enabled"]
        collect_anchor_stats = bool(
            mode == "full" and self._active_config["collect_block_stats"]
        )
        capture_snapshot = capture_cache or collect_anchor_stats
        return {
            "mode": mode,
            "start_block": 0,
            "end_block": self.block_count if mode == "full" else warm_blocks,
            "snapshot_at": warm_blocks if capture_snapshot else None,
            "capture_cache": capture_cache,
            "collect_anchor_stats": collect_anchor_stats,
            "collect_block_stats": bool(
                self._active_config["collect_block_stats"]
            ),
            "reset_block_stats": bool(reset_block_stats),
            "clear_cache": bool(clear_cache),
            "cache_policy": self._active_config["cache_policy"],
            "cache_format": self._active_config["cache_format"],
            "cache_enabled": bool(self._active_config["enabled"]),
            "environment_enabled": bool(
                self._active_config.get("environment_enabled", True)
            ),
            "warm_blocks": warm_blocks,
            "boundary_block": warm_blocks,
            "tail_blocks": self.block_count - warm_blocks,
            "tail_source": self._active_config["tail_source"],
            "legacy_cache_depth": self._active_config["cache_depth"],
            "total_blocks": self.block_count,
            "full_steps": self.full_steps,
            "cache_hits": self.cache_hits,
            "total_block_visits": self.total_blocks,
            "skipped_blocks": self.skipped_blocks,
            "consecutive_cache_steps": self.consecutive_skips,
            "step": self.step,
            "total_steps": self.total_steps,
            "generation_id": self.generation_id,
            "sigma_raw": self.current_sigma_raw,
            "sigma_normalized": self.current_sigma_normalized,
            "sigma_delta": self.current_sigma_delta,
            "schedule_position": self.current_position,
        }

    @staticmethod
    def _layout_key(
        residual: torch.Tensor,
        segments: list[list[int]],
        rope_freqs: torch.Tensor,
    ) -> tuple[object, ...]:
        """Return only the cache-invariant packed layout signature.

        ``segments[*][2]`` is an AdaLN row index, not a packed-token layout.
        ComfyUI is allowed to renumber those rows between CFG/conditioning
        calls while the token boundaries and residual shape stay identical.
        Treating the row IDs as a shape key silently invalidated the residual
        on every real denoise step.  The cache correction is intentionally the
        same approximation used by upstream TE-Speed: it is valid for a
        fixed packed shape/layout while the timestep/modulation values change.
        The actual dimensions still invalidate the cache, including reference
        image changes that alter the packed sequence length.
        """
        boundary_key = tuple(
            (int(start), int(stop)) for start, stop, _row in segments
        )
        return (
            tuple(int(value) for value in residual.shape),
            boundary_key,
            tuple(int(value) for value in rope_freqs.shape),
        )

    def plan(
        self,
        te_speed: dict[str, Any] | None,
        sigma_value: Any,
        sample_sigmas: Any,
        shape_key: tuple[object, ...],
        cache_ready: bool,
    ) -> dict[str, Any]:
        config = self._config(te_speed)
        config_key = tuple(config[name] for name in (
            "control_value", "start_percent", "end_percent", "mcs",
            "cache_depth", "tail_blocks", "tail_source", "collect_block_stats",
            "cache_policy", "enabled", "environment_enabled",
            "cache_format",
        ))
        schedule = self._schedule(sample_sigmas)
        schedule_key = tuple(round(value, 7) for value in schedule)
        reset_required = (
            config_key != self.config_key
            or schedule_key != self.schedule_key
            or shape_key != self.shape_key
        )
        if reset_required:
            self._reset_schedule()
            self.generation_id += 1
            self.config_key = config_key
            self.schedule_key = schedule_key
            self.shape_key = shape_key

        self._active_config = config
        raw_sigma = self._scalar(sigma_value)
        warm_blocks = self._warm_blocks(config)
        if raw_sigma is None:
            # Without the sampler sigma there is no safe cache decision.  Keep
            # the exact full TP route and invalidate any stale residual.
            self._reset_schedule()
            self.config_key = config_key
            self.schedule_key = schedule_key
            self.shape_key = shape_key
            return self._summary(
                "full", warm_blocks, True, reset_block_stats=True
            )

        if self.last_sigma_raw is not None and raw_sigma > self.last_sigma_raw + 1e-6:
            # A higher sigma means a new sampling run.  This also handles a
            # second generation after the previous one ended at sigma~=0.
            self._reset_schedule()
            self.generation_id += 1
            reset_required = True

        if self.last_sigma_raw is None:
            first_call = True
            self.start_sigma = raw_sigma
            if schedule and abs(schedule[0]) > 1e-9:
                self.sigma_scale = raw_sigma / schedule[0]
                if abs(self.sigma_scale) < 1e-9:
                    self.sigma_scale = 1.0
            sigma_normalized = raw_sigma / self.sigma_scale
            self.last_sigma_raw = raw_sigma
            self.last_sigma_normalized = sigma_normalized
            self.current_sigma_raw = raw_sigma
            self.current_sigma_normalized = sigma_normalized
            self.current_sigma_delta = None
            self.current_position = 0.0
            self.step = 0
            self.last_mode = "full"
            self.full_steps = 1
            self.total_blocks = self.block_count
            self.consecutive_skips = 0
            return self._summary(
                "full", warm_blocks, True, reset_block_stats=True
            )

        sigma_normalized = raw_sigma / self.sigma_scale
        same_sigma = abs(raw_sigma - self.last_sigma_raw) <= 1e-6
        if same_sigma:
            # CFG's paired calls at one sigma must share the mode.  This is
            # also what prevents the second branch from changing the cache
            # decision midway through a TP collective sequence.
            mode = self.last_mode if self.last_mode == "cache" and cache_ready else "full"
            self.current_sigma_raw = raw_sigma
            self.current_sigma_normalized = sigma_normalized
            self.current_sigma_delta = 0.0
            return self._summary(
                mode,
                warm_blocks,
                mode == "full",
            )

        previous_normalized = self.last_sigma_normalized
        if previous_normalized is None:
            previous_normalized = sigma_normalized
        position = self._position(sigma_normalized, schedule)
        sigma_delta = abs(previous_normalized - sigma_normalized)
        can_cache = (
            config["enabled"]
            and cache_ready
            and warm_blocks < self.block_count
            and config["start_percent"] <= position <= config["end_percent"]
            and sigma_delta < config["control_value"]
            and self.consecutive_skips < config["mcs"]
        )
        mode = "cache" if can_cache else "full"
        self.prev_sigma_normalized = previous_normalized
        self.last_sigma_raw = raw_sigma
        self.last_sigma_normalized = sigma_normalized
        self.current_sigma_raw = raw_sigma
        self.current_sigma_normalized = sigma_normalized
        self.current_sigma_delta = sigma_delta
        self.current_position = position
        self.step += 1
        self.last_mode = mode
        if mode == "cache":
            self.cache_hits += 1
            self.consecutive_skips += 1
            self.total_blocks += self.block_count
            self.skipped_blocks += self.block_count - warm_blocks
        else:
            self.full_steps += 1
            self.consecutive_skips = 0
            self.total_blocks += self.block_count
        return self._summary(
            mode,
            warm_blocks,
            mode == "full" or reset_required,
            reset_block_stats=reset_required,
        )


class H3TPRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = replace(
            config,
            dit_format=tp.normalize_weight_format(config.dit_format, config.model_path),
        )
        self.backbone: tp.H3TPBackbone | None = None
        self.child: subprocess.Popen[str] | None = None
        self.child_stderr = None
        self.temp_dir: Path | None = None
        self.process_started = False
        self.started = False
        self.closed = False
        self.lock = threading.RLock()
        self.forward_index = 0
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.startup_report: dict[str, Any] | None = None
        self.last_profile: dict[str, Any] | None = None
        # Rank 0 decides the mode; each rank has a local residual cache so
        # CACHE steps do not require an extra full hidden-state broadcast.
        self.te_speed = _TESpeedController()
        self.te_cache = tp.H3TPResidualCache("cpu")
        self.group_controller = _GroupCacheController()
        self.group_cache = h3_q4_cache.GroupResidualCache("cpu")
        self.qwen_backbone: qwen32.Qwen32Q2TPBackbone | None = None
        # The MP backend is deliberately lazy and independent of the NCCL
        # child.  It owns only Qwen language layers; this runtime still owns
        # the persistent TP process used by the H3 DiT path.
        self.qwen_mp_runtime: Any | None = None
        self.qwen_mode = _normalize_qwen_mode(config.qwen_mode)
        self.qwen_config: dict[str, Any] | None = None
        self.last_qwen_profile: dict[str, Any] | None = None

    def _rank1_command(self, init_method: str) -> list[str]:
        worker = Path(__file__).with_name("h3_tp_rank1_worker.py")
        command = [
            sys.executable,
            "-u",
            str(worker),
            "--init-method",
            init_method,
            "--model",
            self.config.model_path,
            "--weight-format",
            self.config.dit_format,
            "--lora",
            self.config.lora_path,
            "--egrid",
            self.config.egrid_path,
            "--strength",
            str(self.config.lora_strength),
            "--staging-mib",
            str(self.config.staging_mib),
            "--chunk-rows",
            str(self.config.chunk_rows),
            "--timeout-seconds",
            str(self.config.timeout_seconds),
            "--qwen-model",
            self.config.qwen_model_path,
            "--qwen-staging-mib",
            str(self.config.qwen_staging_mib),
            "--qwen-residency",
            self.config.qwen_residency,
            "--qwen-keep-layers",
            str(self.config.qwen_keep_layers),
        ]
        if self.config.qwen_cache_dequantized:
            command.append("--qwen-cache-dequantized")
        return command

    def _read_child(self, expected: str, timeout: float | None = None):
        if self.child is None or self.child.stdout is None:
            raise RuntimeError("H3 TP rank1 process is not available")
        deadline = time.monotonic() + (
            timeout if timeout is not None else self.config.timeout_seconds
        )
        while time.monotonic() < deadline:
            if self.child.poll() is not None:
                raise RuntimeError(
                    f"H3 TP rank1 exited with code {self.child.returncode}; "
                    f"see {self.results_dir / 'rank1.stderr.log'}"
                )
            readable, _, _ = select.select(
                [self.child.stdout], [], [], min(1.0, deadline - time.monotonic())
            )
            if not readable:
                continue
            line = self.child.stdout.readline()
            if not line:
                continue
            if not line.startswith("H3TP:"):
                continue
            message = json.loads(line[len("H3TP:") :])
            kind = message.get("kind")
            if kind == "error":
                payload = message.get("payload", {})
                raise RuntimeError(
                    "H3 TP rank1 failed: "
                    f"{payload.get('type')}: {payload.get('message')}\n"
                    f"{payload.get('traceback', '')}"
                )
            if kind != expected:
                raise RuntimeError(
                    f"H3 TP rank1 protocol mismatch: expected {expected}, got {kind}"
                )
            return message.get("payload")
        raise TimeoutError(f"timed out waiting for H3 TP rank1 {expected}")

    def release_cached_memory(self) -> dict[str, Any]:
        """Return free allocator blocks on both ranks to the driver.

        A long DiT forward leaves several GiB of empty segments in each rank's
        caching allocator.  ``allocated`` drops but ``reserved`` does not, and
        the driver counts reserved memory as used, so the next stage on that
        card can fail to allocate while the shard itself has plenty of room.
        Measured after a 720p forward: rank1 allocated 7580 MiB but reserved
        11488 MiB, leaving 3908 MiB stranded and breaking the following Qwen
        dequantisation on cuda:1.

        Neither shard is unloaded; reloading the DiT shard costs ~34 s.  This
        only drops cached blocks, so it is safe to call between requests.
        """
        with self.lock:
            return self._release_cached_memory_locked()

    def _release_cached_memory_locked(self) -> dict[str, Any]:
        """``release_cached_memory`` body; caller already holds ``self.lock``."""
        report: dict[str, Any] = {"rank0": None, "rank1": None}
        if self.closed:
            return report

        device = torch.device("cuda:0")
        before_reserved = torch.cuda.memory_reserved(device) / tp.MIB
        before_allocated = torch.cuda.memory_allocated(device) / tp.MIB
        # Drop this request's leftover activations before returning blocks:
        # ``empty_cache`` only hands back segments with no live allocation, so a
        # retained FP32 snapshot would otherwise pin its whole segment.
        transient = None
        if self.backbone is not None:
            transient = self.backbone.release_transient_state()
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        report["rank0"] = {
            "reserved_mib_before": before_reserved,
            "reserved_mib_after": torch.cuda.memory_reserved(device) / tp.MIB,
            "allocated_mib": before_allocated,
            "allocated_mib_after": torch.cuda.memory_allocated(device) / tp.MIB,
            "transient": transient,
        }

        # Rank 1 is a separate process, so its allocator has to be told
        # explicitly.  A dead child is not an error here: this is cleanup, and
        # the caller's real work should decide whether a missing peer matters.
        if self.child is not None and self.child.poll() is None:
            try:
                self._send_child({"cmd": "release_cache"})
                report["rank1"] = self._read_child("release_cache", timeout=60)
            except Exception:
                logging.warning(
                    "[H3TP] rank1 cache release failed; continuing", exc_info=True
                )

        r0 = report["rank0"]
        r1 = report["rank1"] or {}
        # Report allocated as well as reserved.  Reserved alone hides whether
        # the drop came from returning empty segments or from actually freeing
        # this request's activations, which is the number that decides if the
        # next stage fits.
        logging.info(
            "[H3TP] released VRAM: rank0 reserved %.0f -> %.0f MiB, "
            "allocated %.0f -> %.0f MiB; rank1 reserved %.0f -> %.0f MiB, "
            "allocated %.0f -> %.0f MiB (shards still resident)",
            r0["reserved_mib_before"], r0["reserved_mib_after"],
            r0["allocated_mib"], r0["allocated_mib_after"],
            r1.get("reserved_mib_before", 0.0), r1.get("reserved_mib_after", 0.0),
            r1.get("allocated_mib", 0.0), r1.get("allocated_mib_after", 0.0),
        )
        return report

    def _require_live_child(self, stage: str) -> None:
        """Fail fast when rank 1 is gone instead of blocking in a collective.

        NCCL has no way to report a dead peer to a rank already inside a
        collective; ``ALLREDUCE`` just waits for the process-group timeout.
        ``poll()`` is a cheap non-blocking waitpid, so this can sit on the
        per-step path.
        """
        if self.child is None:
            raise RuntimeError(f"{stage}: H3 TP rank1 process is not available")
        code = self.child.poll()
        if code is None:
            return
        log = self.results_dir / "rank1.stderr.log"
        detail = ""
        try:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()
            interesting = [
                line for line in tail
                if "Error" in line or "error" in line or "Traceback" in line
            ]
            detail = "\n".join((interesting or tail)[-6:])
        except OSError:
            pass
        raise RuntimeError(
            f"{stage}: H3 TP rank1 exited with code {code} before the "
            f"collective; the sequence is most likely too long for rank 1's "
            f"share of VRAM. See {log}\n{detail}"
        )

    def _send_child(self, payload: dict[str, Any]) -> None:
        if self.child is None or self.child.stdin is None:
            raise RuntimeError("H3 TP rank1 stdin is unavailable")
        self.child.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.child.stdin.flush()

    def _progress(self, stage: str, current: int, total: int) -> None:
        if current == 1 or current == total or current % 10 == 0:
            print(f"[H3TP rank0] loading {stage} {current}/{total}", flush=True)

    def ensure_process_started(self) -> None:
        with self.lock:
            if self.process_started:
                return
            if self.closed:
                raise RuntimeError("H3 TP runtime was closed")
            if dist.is_initialized():
                raise RuntimeError(
                    "the default torch.distributed group is already initialized; "
                    "H3 TP requires exclusive ownership in this ComfyUI process"
                )
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError("H3 TP requires two visible CUDA devices")
            if not torch.cuda.can_device_access_peer(0, 1):
                raise RuntimeError("H3 TP requires CUDA P2P/NVLink between cuda:0 and cuda:1")

            started_at = time.time()
            self.temp_dir = Path(tempfile.mkdtemp(prefix="minimax_h3_tp_"))
            store_path = self.temp_dir / "nccl_store"
            init_method = f"file://{store_path}"
            stderr_path = self.results_dir / "rank1.stderr.log"
            self.child_stderr = stderr_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONUNBUFFERED": "1",
                    "H3_NO_HOST_MMAP": "1",
                    "CUDA_MODULE_LOADING": env.get("CUDA_MODULE_LOADING", "LAZY"),
                    "NCCL_P2P_LEVEL": env.get("NCCL_P2P_LEVEL", "NVL"),
                    "NCCL_DEBUG": env.get("NCCL_DEBUG", "WARN"),
                    "H3_V100_RMS_ROPE_WARPS": env.get(
                        "H3_V100_RMS_ROPE_WARPS", "1"
                    ),
                }
            )
            self.child = subprocess.Popen(
                self._rank1_command(init_method),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.child_stderr,
                text=True,
                bufsize=1,
                env=env,
            )

            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            try:
                dist.init_process_group(
                    "nccl",
                    init_method=init_method,
                    rank=0,
                    world_size=2,
                    timeout=timedelta(seconds=self.config.timeout_seconds),
                    device_id=device,
                )
                rank1_ready = self._read_child("ready")
            except BaseException:
                self._force_close()
                raise

            self.process_started = True
            self.startup_report = {
                "created_unix": time.time(),
                "process_startup_seconds": time.time() - started_at,
                "backend": "persistent parent rank0 + child rank1 / NCCL",
                "world_size": 2,
                "weight_format": self.config.dit_format,
                "payload_mmap": False,
                "rank0_pid": os.getpid(),
                "rank1_pid": self.child.pid,
                "rank1_ready": rank1_ready,
                "h3_ready": False,
            }
            print(
                f"[H3TP] shared two-rank NCCL process ready in "
                f"{self.startup_report['process_startup_seconds']:.2f}s; "
                f"rank1 pid={self.child.pid}",
                flush=True,
            )

    def ensure_started(self) -> None:
        """Start the shared process group, then lazily materialise H3 shards."""

        with self.lock:
            if self.started:
                return
            self.ensure_process_started()
            started_at = time.time()
            self._send_child({"cmd": "h3_init"})
            try:
                self.backbone = tp.H3TPBackbone(
                    rank=0,
                    device=torch.device("cuda:0"),
                    model_path=self.config.model_path,
                    weight_format=self.config.dit_format,
                    lora_path=self.config.lora_path,
                    egrid_path=self.config.egrid_path,
                    lora_strength=self.config.lora_strength,
                    staging_bytes=self.config.staging_mib << 20,
                    chunk_rows=self.config.chunk_rows,
                    progress=self._progress,
                )
                rank1_load = self._read_child("h3_ready")
            except BaseException:
                self._force_close()
                self.backbone = None
                raise

            self.started = True
            process_seconds = 0.0
            if self.startup_report is not None:
                process_seconds = float(self.startup_report.get("process_startup_seconds", 0.0))
            self.startup_report = {
                "created_unix": time.time(),
                "process_startup_seconds": process_seconds,
                "h3_startup_seconds": time.time() - started_at,
                "startup_seconds": process_seconds + time.time() - started_at,
                "backend": "persistent parent rank0 + child rank1 / NCCL",
                "world_size": 2,
                "model": self.config.model_path,
                "weight_format": self.config.dit_format,
                "lora": self.config.lora_path,
                "lora_strength": self.config.lora_strength,
                "egrid": self.config.egrid_path,
                "payload_mmap": False,
                "rank0_pid": os.getpid(),
                "rank1_pid": self.child.pid,
                "rank0_load": self.backbone.load_stats,
                "rank1_load": rank1_load,
                "h3_ready": True,
            }
            startup_path = self.results_dir / "startup.json"
            startup_path.write_text(
                json.dumps(self.startup_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[H3TP] persistent two-rank backbone ready in "
                f"{self.startup_report['startup_seconds']:.2f}s; "
                f"rank1 pid={self.child.pid}",
                flush=True,
            )

    def forward(
        self,
        residual: torch.Tensor,
        t_emb: torch.Tensor,
        segments,
        rope_freqs: torch.Tensor,
        *,
        te_speed: dict[str, Any] | None = None,
        group_cache: dict[str, Any] | None = None,
        sigma: Any = None,
        sample_sigmas: Any = None,
        cond_or_uncond: Any = None,
        profile: bool = False,
        stage_profile: bool = False,
    ) -> torch.Tensor:
        with self.lock:
            self.ensure_started()
            assert self.backbone is not None
            if residual.device != torch.device("cuda:0"):
                raise ValueError(f"H3 TP parent residual must be on cuda:0, got {residual.device}")
            residual = residual.contiguous()
            t_emb = t_emb.to(device="cuda:0", dtype=torch.float32).contiguous()
            rope_freqs = rope_freqs.to(device="cuda:0", dtype=torch.float16).contiguous()
            serial_segments = [
                [int(start), int(stop), int(row)] for start, stop, row in segments
            ]
            stat_ranges = tp.infer_target_stat_ranges(
                int(residual.shape[0]), serial_segments
            )
            shape_key = self.te_speed._layout_key(
                residual,
                serial_segments,
                rope_freqs,
            )
            group_plan = self.group_controller.plan(
                group_cache,
                sigma,
                sample_sigmas,
                shape_key,
            )
            if group_plan["enabled"] and te_speed and bool(te_speed.get("enabled", False)):
                raise ValueError(
                    "H3 whole-tail TE-Speed and Adaptive Group Cache cannot be enabled together"
                )
            if group_plan["enabled"]:
                plan = self.te_speed.plan(
                    None,
                    sigma,
                    sample_sigmas,
                    shape_key,
                    self.te_cache.ready,
                )
                if self.te_cache.ready:
                    self.te_cache.clear()
                hidden_mib = (
                    residual.numel() * residual.element_size() / tp.MIB
                )
                if (
                    group_plan["benchmark_ground_truth"]
                    and hidden_mib > group_plan["oracle_max_mib"]
                ):
                    raise RuntimeError(
                        "H3 group-cache benchmark oracle would clone a hidden tensor of "
                        f"{hidden_mib:.1f} MiB, above oracle_max_mib="
                        f"{group_plan['oracle_max_mib']}. Increase it explicitly only for "
                        "a controlled benchmark."
                    )
                plan.update(
                    {
                        "mode": "group",
                        "start_block": 0,
                        "end_block": tp.LAYERS,
                        "snapshot_at": None,
                        "capture_cache": False,
                        "collect_anchor_stats": False,
                        "collect_block_stats": bool(
                            group_plan["collect_block_stats"]
                            or group_plan["benchmark_ground_truth"]
                        ),
                        "reset_block_stats": bool(group_plan["clear_cache"]),
                        "clear_cache": False,
                        "cache_enabled": True,
                    }
                )
            else:
                plan = self.te_speed.plan(
                    te_speed,
                    sigma,
                    sample_sigmas,
                    shape_key,
                    self.te_cache.ready,
                )
                if self.group_cache.entries:
                    self.group_cache.clear()
            if self.te_cache.set_policy(plan["cache_policy"]):
                plan["clear_cache"] = True
            if plan["clear_cache"]:
                self.te_cache.clear()
            command = {
                "cmd": "forward",
                "residual_shape": list(residual.shape),
                "t_emb_shape": list(t_emb.shape),
                "rope_shape": list(rope_freqs.shape),
                "segments": serial_segments,
                "mode": plan["mode"],
                "start_block": plan["start_block"],
                "end_block": plan["end_block"],
                "snapshot_at": plan["snapshot_at"],
                "capture_cache": plan["capture_cache"],
                "collect_anchor_stats": plan["collect_anchor_stats"],
                "collect_block_stats": plan["collect_block_stats"],
                "reset_block_stats": plan["reset_block_stats"],
                "stat_ranges": [list(item) for item in stat_ranges],
                "cache_enabled": plan["cache_enabled"],
                "cache_device": plan["cache_policy"],
                "cache_format": plan["cache_format"],
                "clear_cache": plan["clear_cache"],
                "generation_id": plan["generation_id"],
                "step": plan["step"],
                "sigma_raw": plan["sigma_raw"],
                "sigma_normalized": plan["sigma_normalized"],
                "sigma_delta": plan["sigma_delta"],
                "boundary_block": plan["boundary_block"],
                "tail_blocks": plan["tail_blocks"],
                "group_cache": group_plan,
                "profile": bool(profile),
                "stage_profile": bool(stage_profile),
            }
            try:
                self._send_child(command)
                # A dead rank 1 is the one failure mode that cannot be detected
                # from inside a collective: rank 0 simply blocks in ALLREDUCE
                # until the 900 s watchdog fires.  Rank 1 dies silently when it
                # cannot allocate the communication buffer, which for long
                # sequences is large (a 362-frame 720p step allreduces 542 M
                # elements, about 2 GiB in FP32).  Check liveness before
                # entering the collective so the request fails in seconds with
                # a usable message instead of hanging for 15 minutes.
                self._require_live_child("H3 forward")
                tp.broadcast_inputs_rank0(residual, t_emb, rope_freqs)
                output, rank0_metrics = self.backbone.forward(
                    residual,
                    t_emb,
                    serial_segments,
                    rope_freqs,
                    profile=profile,
                    stage_profile=stage_profile,
                    start_block=plan["start_block"],
                    end_block=plan["end_block"],
                    snapshot_at=plan["snapshot_at"],
                    collect_block_stats=plan["collect_block_stats"],
                    stat_ranges=stat_ranges,
                    reset_block_stats=plan["reset_block_stats"],
                    group_cache=self.group_cache,
                    group_config=group_plan,
                )
                tail_residual_stats = None
                if plan["mode"] == "group":
                    self.backbone.clear_snapshot()
                elif plan["mode"] == "full":
                    if plan["capture_cache"] or plan["collect_anchor_stats"]:
                        tail_residual_stats = self.te_cache.store(
                            output,
                            self.backbone.take_snapshot(),
                            stat_ranges=stat_ranges,
                            collect_stats=plan["collect_block_stats"],
                            retain=plan["capture_cache"],
                        )
                    else:
                        self.backbone.clear_snapshot()
                else:
                    self.te_cache.add_to(
                        output, measure=plan["collect_block_stats"]
                    )

                rank1_metrics = self._read_child("forward")
                output_report = tp.tensor_scalar_stats(
                    output, stat_ranges if plan["collect_block_stats"] else None
                )
                output_stats = output_report["overall"]
                rank0_metrics["output_rms"] = output_stats["rms"]
                rank0_metrics["output_max_abs"] = output_stats["max_abs"]
                rank0_metrics["finite"] = output_stats["finite"]
                rank0_metrics["process_memory"] = tp.process_memory_stats()
                if plan["collect_block_stats"]:
                    block_stats = rank0_metrics.get("block_stats") or {}
                    if plan["mode"] != "group":
                        block_stats["tail_residual"] = (
                            tail_residual_stats
                            if tail_residual_stats is not None
                            else self.te_cache.anchor_stats
                        )
                        block_stats["cache_operation"] = dict(
                            self.te_cache.last_operation
                        )
                    block_stats["final_output_after_cache_add"] = output_report
                    rank0_metrics["block_stats"] = block_stats
                if not rank0_metrics["finite"]:
                    raise RuntimeError("H3 TP TE-Speed output produced NaN/Inf after cache correction")
                rank0_metrics["allocated_mib"] = torch.cuda.memory_allocated(torch.device("cuda:0")) / tp.MIB
                rank0_metrics["reserved_mib"] = torch.cuda.memory_reserved(torch.device("cuda:0")) / tp.MIB
                rank0_metrics["peak_allocated_mib"] = max(
                    float(rank0_metrics["peak_allocated_mib"]),
                    torch.cuda.max_memory_allocated(torch.device("cuda:0")) / tp.MIB,
                )
            except BaseException:
                # A failed mode/range must not leave rank 1 waiting on a stale
                # protocol command or a cache that belongs to another shape.
                self._force_close()
                self.backbone = None
                self.started = False
                self.te_cache.clear()
                self.group_cache.clear()
                raise
            rank0_metrics["te_speed_mode"] = plan["mode"]
            rank0_metrics["te_speed_block_range"] = [
                plan["start_block"], plan["end_block"]
            ]
            rank0_metrics["te_speed_cache_bytes"] = self.te_cache.bytes
            rank0_metrics["te_speed_cache_device"] = plan["cache_policy"]
            rank0_metrics["te_speed_generation_id"] = plan["generation_id"]
            rank0_metrics["te_speed_step"] = plan["step"]
            rank0_metrics["te_speed_sigma_raw"] = plan["sigma_raw"]
            rank0_metrics["te_speed_sigma_normalized"] = plan["sigma_normalized"]
            rank0_metrics["te_speed_sigma_delta"] = plan["sigma_delta"]
            rank0_metrics["te_speed_boundary_block"] = plan["boundary_block"]
            rank0_metrics["te_speed_tail_blocks"] = plan["tail_blocks"]
            rank0_metrics["group_cache_mode"] = bool(group_plan["enabled"])
            rank0_metrics["group_cache_bytes"] = self.group_cache.bytes
            rank1_metrics["te_speed_mode"] = plan["mode"]
            rank1_metrics["te_speed_block_range"] = [
                plan["start_block"], plan["end_block"]
            ]
            self.forward_index += 1
            report = {
                "created_unix": time.time(),
                "forward_index": self.forward_index,
                "shape": {
                    "residual": list(output.shape),
                    "t_emb": list(t_emb.shape),
                    "rope": list(rope_freqs.shape),
                    "segments": serial_segments,
                },
                "rank0": rank0_metrics,
                "rank1": rank1_metrics,
                "max_rank_total_ms": max(
                    rank0_metrics["total_ms"], rank1_metrics["total_ms"]
                ),
                "max_rank_collective_ms": max(
                    rank0_metrics["collective_ms"], rank1_metrics["collective_ms"]
                ),
                "rank_output_rms_abs_diff": abs(
                    rank0_metrics["output_rms"] - rank1_metrics["output_rms"]
                ),
                "profile": bool(profile),
                "stage_profile": bool(stage_profile),
                "finite": bool(rank0_metrics["finite"] and rank1_metrics["finite"]),
                "te_speed": {
                    **plan,
                    "cache_ready_after": self.te_cache.ready,
                    "cache_bytes_rank0": self.te_cache.bytes,
                    "cfg_branch_count": (
                        len(cond_or_uncond)
                        if isinstance(cond_or_uncond, (list, tuple))
                        else None
                    ),
                },
                "group_cache": {
                    **group_plan,
                    "cache_after": self.group_cache.summary(),
                },
                "models_reloaded": False,
            }
            self.last_profile = report
            stamp = time.strftime("%Y%m%d-%H%M%S")
            profile_path = self.results_dir / (
                f"forward_{self.forward_index:04d}_{output.shape[0]}t_{stamp}.json"
            )
            profile_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[H3TP] forward#{self.forward_index} S={output.shape[0]} "
                f"{report['max_rank_total_ms'] / 1000.0:.3f}s "
                f"NCCL={report['max_rank_collective_ms'] / 1000.0:.3f}s "
                f"peak=({rank0_metrics['peak_allocated_mib']:.0f},"
                f"{rank1_metrics['peak_allocated_mib']:.0f}) MiB",
                flush=True,
            )
            return output

    # ------------------------------------------------------------------
    # Qwen32B Q2 output-row TP protocol
    # ------------------------------------------------------------------
    @staticmethod
    def _tree_spec(value: Any) -> Any:
        """Describe a tensor/list/tuple tree for the rank-1 pipe."""

        if value is None:
            return None
        if torch.is_tensor(value):
            return {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
            }
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [H3TPRuntime._tree_spec(v) for v in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [H3TPRuntime._tree_spec(v) for v in value]}
        raise TypeError(f"unsupported Qwen tensor-tree value: {type(value).__name__}")

    @staticmethod
    def _tree_to_device(value: Any, device: torch.device) -> Any:
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=device).contiguous()
        if isinstance(value, tuple):
            return tuple(H3TPRuntime._tree_to_device(v, device) for v in value)
        if isinstance(value, list):
            return [H3TPRuntime._tree_to_device(v, device) for v in value]
        raise TypeError(f"unsupported Qwen tensor-tree value: {type(value).__name__}")

    @staticmethod
    def _tree_broadcast_rank0(value: Any, spec: Any, device: torch.device) -> Any:
        """Broadcast a nested tensor tree in deterministic depth-first order."""

        if spec is None:
            return None
        kind = spec.get("kind")
        if kind == "tensor":
            if not torch.is_tensor(value):
                raise TypeError("Qwen tree value/spec mismatch on rank0")
            dtype = getattr(torch, spec["dtype"])
            tensor = value.to(device=device, dtype=dtype).contiguous()
            if list(tensor.shape) != list(spec["shape"]):
                raise ValueError("Qwen tensor shape changed while preparing broadcast")
            dist.broadcast(tensor, src=0)
            return tensor
        items = spec.get("items", [])
        values = [H3TPRuntime._tree_broadcast_rank0(v, s, device) for v, s in zip(value, items)]
        return tuple(values) if kind == "tuple" else values

    @staticmethod
    def _tree_broadcast_rank1(spec: Any, device: torch.device) -> Any:
        if spec is None:
            return None
        kind = spec.get("kind")
        if kind == "tensor":
            dtype = getattr(torch, spec["dtype"])
            tensor = torch.empty(tuple(int(v) for v in spec["shape"]), device=device, dtype=dtype)
            dist.broadcast(tensor, src=0)
            return tensor
        values = [H3TPRuntime._tree_broadcast_rank1(s, device) for s in spec.get("items", [])]
        return tuple(values) if kind == "tuple" else values

    def _ensure_qwen_backbone(self) -> qwen32.Qwen32Q2TPBackbone:
        if self.qwen_mode == "mp":
            raise RuntimeError(
                "Qwen32 backend is layer-MP; output-row TP backbone is disabled"
            )
        if not self.config.qwen_model_path:
            raise RuntimeError("Qwen32 Q2 is not configured on this shared runtime")
        if self.qwen_backbone is None:
            self.qwen_backbone = qwen32.Qwen32Q2TPBackbone(
                self.config.qwen_model_path,
                rank=0,
                world_size=2,
                device="cuda:0",
                # MiniMaxH3ClipModel drives the stock Qwen GGUF path with
                # FP32 activations/dequantized weights.  Preserve that
                # arithmetic for the correctness route; FP16 is not an
                # acceptable silent performance fallback here.
                dtype=torch.float32,
                staging_mib=self.config.qwen_staging_mib,
                residency=self.config.qwen_residency,
                keep_layers=self.config.qwen_keep_layers,
                cache_dequantized=self.config.qwen_cache_dequantized,
            )
            self.qwen_config = {
                "path": self.config.qwen_model_path,
                "residency": self.config.qwen_residency,
                "keep_layers": self.config.qwen_keep_layers,
                "rank": 0,
            }
        return self.qwen_backbone

    def _ensure_qwen_mp_runtime(self):
        """Create the standalone layer-MP backend only when Qwen is requested.

        The H3 runtime remains the owner of the DiT TP process.  Keeping this
        backend lazy means a text-only request does not initialize NCCL, and a
        later DiT request can still start the existing persistent worker.
        """

        if self.qwen_mode != "mp":
            raise RuntimeError(
                "Qwen32 backend is output-row TP; layer-MP was not selected"
            )
        if not self.config.qwen_model_path:
            raise RuntimeError("Qwen32 Q2 is not configured on this shared runtime")
        if self.qwen_mp_runtime is None:
            from .h3_qwen32_q2_mp import Qwen32Q2LayerMPRuntime

            raw_devices = os.environ.get("H3_QWEN32_MP_DEVICES", "cuda:0,cuda:1")
            devices = tuple(
                item.strip() for item in raw_devices.split(",") if item.strip()
            )
            if len(devices) != 2:
                raise ValueError(
                    "H3_QWEN32_MP_DEVICES must contain two comma-separated devices"
                )
            split = os.environ.get("H3_QWEN32_MP_SPLIT", "auto")
            output_device = os.environ.get("H3_QWEN32_OUTPUT_DEVICE", "cuda:1")
            self.qwen_mp_runtime = Qwen32Q2LayerMPRuntime(
                self.config.qwen_model_path,
                devices=devices,
                layer_split=split,
                staging_mib=self.config.qwen_staging_mib,
                residency=self.config.qwen_residency,
                keep_layers=self.config.qwen_keep_layers,
                cache_dequantized=self.config.qwen_cache_dequantized,
                dtype=torch.float32,
                output_device=output_device,
                check_peer_access=True,
                enforce_capacity=True,
            )
            self.qwen_config = {
                "path": self.config.qwen_model_path,
                "residency": self.config.qwen_residency,
                "keep_layers": self.config.qwen_keep_layers,
                "mode": "mp",
                "devices": list(devices),
            }
        return self.qwen_mp_runtime

    def qwen_forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: Any = None,
        deepstack_embeds: Sequence[torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one Qwen32 output-row TP encode on both persistent ranks."""

        with self.lock:
            if self.qwen_mode == "mp":
                # Complete-layer MP is process-local and intentionally does
                # not enter the NCCL protocol.  The same H3 runtime remains
                # available for the later DiT forward.
                #
                # Hand back cached allocator blocks first.  On a second request
                # the previous DiT forward has left several GiB of empty
                # segments reserved on both cards, which the driver counts as
                # used; the layer-MP dequantisation on cuda:1 then fails even
                # though rank1's shard has room.  Released without touching
                # either shard.
                self._release_cached_memory_locked()
                backend = self._ensure_qwen_mp_runtime()
                output = backend.qwen_forward(
                    hidden,
                    attention_mask=attention_mask,
                    freqs_cis=freqs_cis,
                    deepstack_embeds=deepstack_embeds,
                    visual_pos_masks=visual_pos_masks,
                )
                self.last_qwen_profile = backend.last_qwen_profile
                return output
            # Qwen conditioning is allowed to run before the H3 DiT model is
            # materialised.  Both paths share this process group and lock, but
            # delaying the Q4 shard load leaves the cards' full headroom for
            # the Qwen layer-local dequantisation step.
            self.ensure_process_started()
            try:
                if self.qwen_backbone is None:
                    self._ensure_qwen_backbone()
                assert self.qwen_backbone is not None
                from . import h3_async_vae_bridge

                h3_async_vae_bridge.prepare_active_vae_for_qwen()
                hidden = hidden.to(device="cuda:0", dtype=torch.float32).contiguous()
                tree = {
                    "attention_mask": self._tree_spec(attention_mask),
                    "freqs_cis": self._tree_spec(freqs_cis),
                    "deepstack_embeds": self._tree_spec(deepstack_embeds),
                    "visual_pos_masks": self._tree_spec(visual_pos_masks),
                }
                command = {
                    "cmd": "qwen_forward",
                    "hidden_shape": list(hidden.shape),
                    "hidden_dtype": "float32",
                    "tree": tree,
                }
                self._send_child(command)
                dist.broadcast(hidden, src=0)
                attn = self._tree_broadcast_rank0(attention_mask, tree["attention_mask"], torch.device("cuda:0"))
                freqs = self._tree_broadcast_rank0(freqs_cis, tree["freqs_cis"], torch.device("cuda:0"))
                deep = self._tree_broadcast_rank0(deepstack_embeds, tree["deepstack_embeds"], torch.device("cuda:0"))
                visual = self._tree_broadcast_rank0(visual_pos_masks, tree["visual_pos_masks"], torch.device("cuda:0"))
                output = self.qwen_backbone.forward_hidden(
                    hidden,
                    attention_mask=attn,
                    freqs_cis=freqs,
                    gather=lambda value, **kw: qwen32.all_gather_output_rows(
                        value, rank=0, world_size=2, label=kw.get("label")
                    ),
                    deepstack_embeds=deep,
                    visual_pos_masks=visual,
                )
                dist.barrier()
                rank1_metrics = self._read_child("qwen_forward")
                report = {
                    "rank0": self.qwen_backbone.stats(),
                    "rank1": rank1_metrics,
                    "shape": list(output.shape),
                    "finite": bool(torch.isfinite(output).all().item()),
                }
                if not report["finite"]:
                    raise RuntimeError("Qwen32 Q2 rank0 output produced NaN/Inf")
                self.last_qwen_profile = report
                return output
            except BaseException:
                # A failed collective/input broadcast leaves rank 1 in an
                # unknown protocol position.  Tear down both sides instead of
                # allowing the next request to silently reuse mismatched NCCL
                # ordering or an H3 backbone tied to a destroyed group.
                self._force_close()
                self.backbone = None
                self.started = False
                self.te_cache.clear()
                self.group_cache.clear()
                raise

    def configure_qwen(
        self,
        model_path: str,
        *,
        staging_mib: int | None = None,
        residency: str | None = None,
        keep_layers: int | None = None,
        cache_dequantized: bool | None = None,
        mode: str | None = None,
    ) -> None:
        """Attach the shared Qwen configuration before either backend starts."""

        with self.lock:
            requested_mode = _normalize_qwen_mode(
                self.qwen_mode if mode is None else mode
            )
            active = bool(
                self.process_started
                or self.qwen_backbone is not None
                or self.qwen_mp_runtime is not None
            )
            if active:
                if str(model_path) != self.config.qwen_model_path:
                    raise RuntimeError(
                        "cannot change Qwen model after the runtime has started"
                    )
                requested = (
                    self.config.qwen_staging_mib if staging_mib is None else int(staging_mib),
                    self.config.qwen_residency if residency is None else str(residency),
                    self.config.qwen_keep_layers if keep_layers is None else int(keep_layers),
                    self.config.qwen_cache_dequantized if cache_dequantized is None else bool(cache_dequantized),
                    requested_mode,
                )
                current = (
                    self.config.qwen_staging_mib,
                    self.config.qwen_residency,
                    self.config.qwen_keep_layers,
                    self.config.qwen_cache_dequantized,
                    self.qwen_mode,
                )
                if requested != current:
                    raise RuntimeError(
                        "cannot change Qwen runtime options after execution started"
                    )
                return
            config = replace(
                self.config,
                qwen_model_path=str(model_path),
                qwen_staging_mib=(self.config.qwen_staging_mib if staging_mib is None else int(staging_mib)),
                qwen_residency=(self.config.qwen_residency if residency is None else str(residency)),
                qwen_keep_layers=(self.config.qwen_keep_layers if keep_layers is None else int(keep_layers)),
                qwen_cache_dequantized=(self.config.qwen_cache_dequantized if cache_dequantized is None else bool(cache_dequantized)),
                qwen_mode=requested_mode,
            )
            if config.qwen_residency not in {"evict", "partial", "full"}:
                raise ValueError("Qwen residency must be evict, partial, or full")
            object.__setattr__(self, "config", config)
            self.qwen_mode = requested_mode

    def qwen_clear(self, *, notify_vae: bool = True) -> dict[str, Any]:
        """Clear Qwen payload on both ranks before optionally opening the VAE gate.

        Failed conditioning calls use ``notify_vae=False``: both ranks are
        still cleared, but no downstream DiT/VAE work may start from an
        incomplete conditioning result.
        """

        with self.lock:
            if self.qwen_mode == "mp":
                try:
                    backend = self.qwen_mp_runtime
                    result = None
                    if backend is not None:
                        result = backend.qwen_clear(notify_vae=notify_vae)
                    self.te_cache.clear()
                    self.group_cache.clear()
                    rank0 = None if result is None else result.get("rank0")
                    return {
                        "mode": "mp",
                        "configured": bool(self.config.qwen_model_path),
                        "rank0": rank0,
                        "rank1": None,
                        "rank1_clear": None,
                        "last_profile": self.last_qwen_profile,
                        "vae_notified": bool(
                            result is not None and result.get("vae_notified", False)
                        ),
                    }
                except BaseException:
                    self._force_close()
                    self.backbone = None
                    self.started = False
                    raise
            self.ensure_process_started()
            try:
                self._send_child({"cmd": "qwen_clear"})
                if self.qwen_backbone is not None:
                    self.qwen_backbone.clear()
                self.te_cache.clear()
                self.group_cache.clear()
                dist.barrier()
                torch.cuda.synchronize(torch.device("cuda:0"))
                torch.cuda.empty_cache()
                rank1 = self._read_child("qwen_clear")
                if notify_vae:
                    # The bridge is notified only after rank1 has completed
                    # its clear, the NCCL barrier has passed, and both
                    # allocators are synchronized.
                    from . import h3_async_vae_bridge

                    h3_async_vae_bridge.notify_qwen_cleared()
                return {
                    "configured": bool(self.config.qwen_model_path),
                    "rank0": None if self.qwen_backbone is None else self.qwen_backbone.stats(),
                    "rank1": rank1.get("stats") if isinstance(rank1, dict) else rank1,
                    "rank1_clear": rank1,
                    "last_profile": self.last_qwen_profile,
                    "vae_notified": bool(notify_vae),
                }
            except BaseException:
                self._force_close()
                self.backbone = None
                self.started = False
                raise

    def qwen_trim(self, keep_layers: int | Sequence[int] = 0) -> dict[str, Any]:
        with self.lock:
            if self.qwen_mode == "mp":
                try:
                    keep = (
                        [int(v) for v in range(int(keep_layers))]
                        if isinstance(keep_layers, int)
                        else [int(v) for v in keep_layers]
                    )
                    backend = self._ensure_qwen_mp_runtime()
                    result = backend.qwen_trim(keep)
                    return {
                        "mode": "mp",
                        "rank0": result.get("rank0"),
                        "rank1": None,
                        "keep_layers": keep,
                    }
                except BaseException:
                    self._force_close()
                    self.backbone = None
                    self.started = False
                    raise
            self.ensure_process_started()
            keep = [int(v) for v in range(int(keep_layers))] if isinstance(keep_layers, int) else [int(v) for v in keep_layers]
            try:
                # Keep lazy construction symmetric.  The child has always
                # called ensure_qwen() for qwen_trim; rank 0 must do the same.
                backbone = self._ensure_qwen_backbone()
                self._send_child({"cmd": "qwen_trim", "keep_layers": keep})
                backbone.trim(keep)
                dist.barrier()
                rank1 = self._read_child("qwen_trim")
                return {
                    "rank0": backbone.stats(),
                    "rank1": rank1,
                    "keep_layers": keep,
                }
            except BaseException:
                self._force_close()
                self.backbone = None
                self.started = False
                raise

    def qwen_stats(self) -> dict[str, Any]:
        with self.lock:
            if self.qwen_mode == "mp":
                backend = self.qwen_mp_runtime
                return {
                    "mode": "mp",
                    "configured": bool(self.config.qwen_model_path),
                    "rank0": None if backend is None else backend.qwen_stats().get("rank0"),
                    "rank1": None,
                    "last_profile": self.last_qwen_profile,
                }
            self.ensure_process_started()
            self._send_child({"cmd": "qwen_stats"})
            rank1 = self._read_child("qwen_stats")
            return {
                "configured": bool(self.config.qwen_model_path),
                "rank0": None if self.qwen_backbone is None else self.qwen_backbone.stats(),
                "rank1": rank1,
                "last_profile": self.last_qwen_profile,
            }

    def _force_close(self) -> None:
        if self.qwen_backbone is not None:
            try:
                self.qwen_backbone.clear()
            except Exception:
                pass
            self.qwen_backbone = None
            self.qwen_config = None
        if self.qwen_mp_runtime is not None:
            try:
                self.qwen_mp_runtime.close()
            except Exception:
                pass
            self.qwen_mp_runtime = None
            self.qwen_config = None
        # Teardown order matters and is the difference between a 5 s failure and
        # a 900 s hang.  ``destroy_process_group`` is itself collective: it
        # joins the NCCL communicator, so it only returns once *both* ranks
        # call it.  When rank 0 aborts mid-forward (an OOM in a block, say),
        # rank 1 is still inside the matching collective for that forward and
        # will never reach its own destroy.  Rank 0 then blocks in
        # ``ncclCommDestroy`` -> ``pthread_join`` while rank 1 spins in the
        # collective until the 900 s watchdog fires.
        #
        # Kill the child *first* so its communicator dies and any collective
        # rank 0 is waiting on fails fast, then abort rather than destroy.
        # ``abort()`` tears the communicator down locally and does not wait for
        # the peer.
        if self.child is not None and self.child.poll() is None:
            try:
                self.child.kill()
                self.child.wait(timeout=10)
            except Exception:
                pass
        if dist.is_initialized():
            aborted = False
            try:
                group = dist.distributed_c10d._get_default_group()
                backend = group._get_backend(torch.device("cuda:0"))
            except Exception:
                backend = None
            if backend is not None and hasattr(backend, "abort"):
                try:
                    backend.abort()
                    aborted = True
                except Exception:
                    pass
            try:
                # With the communicator already aborted this only drops the
                # Python-side group state and cannot block on the peer.
                dist.destroy_process_group()
            except Exception:
                pass
            if not aborted:
                logging.warning(
                    "[H3TP] NCCL communicator could not be aborted; a stale "
                    "collective may still be pending"
                )
        if self.child_stderr is not None:
            self.child_stderr.close()
            self.child_stderr = None
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
        self.te_cache.clear()
        self.group_cache.clear()
        self.process_started = False

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            if self.child is not None and self.child.poll() is None:
                try:
                    self._send_child({"cmd": "shutdown"})
                    self._read_child("shutdown", timeout=5)
                    # Give rank 1 a moment to exit on its own so the shared
                    # teardown below sees an already-dead peer instead of
                    # having to kill it.
                    self.child.wait(timeout=10)
                except Exception:
                    pass
            self._force_close()
            self.backbone = None
            self.started = False
            self.process_started = False
            self.closed = True


_RUNTIME: H3TPRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def get_runtime(config: RuntimeConfig) -> H3TPRuntime:
    global _RUNTIME
    normalized_config = replace(
        config,
        dit_format=tp.normalize_weight_format(config.dit_format, config.model_path),
    )
    with _RUNTIME_LOCK:
        if _RUNTIME is None or _RUNTIME.closed:
            _RUNTIME = H3TPRuntime(normalized_config)
            return _RUNTIME
        if _RUNTIME.config != normalized_config:
            raise RuntimeError(
                "an H3 TP runtime with different model/LoRA settings is already active; "
                "restart the service to change persistent worker configuration"
            )
        return _RUNTIME


def active_runtime() -> H3TPRuntime | None:
    """Return the started runtime, or ``None`` when there is nothing to act on.

    Deliberately does not create one, unlike :func:`get_runtime`: callers are
    cleanup paths that must be inert on graphs which never used H3 TP.  A
    runtime that exists but has not started owns no CUDA payload yet, so it is
    reported as absent too.
    """
    with _RUNTIME_LOCK:
        runtime = _RUNTIME
    if runtime is None or runtime.closed or not runtime.started:
        return None
    return runtime


def close_runtime() -> None:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            _RUNTIME.close()
            _RUNTIME = None


_POSTSAMPLE_HOOK_MARKER = "_h3_tp_postsample_release"
_POSTSAMPLE_HOOK_LOCK = threading.Lock()


def install_postsample_release_hook() -> bool:
    """Free DiT activations on both ranks as soon as the sampler returns.

    Without this the last denoise step's tensors stay reachable until Python
    happens to collect them, and the caching allocator keeps every segment it
    grew during the forward.  At 720p/243f INT8 that leaves ~2.9 GiB reserved
    plus an ~800 MiB dead FP32 residual on each card at the exact moment VAE
    decode wants room.  The driver counts reserved bytes as used, so layer-MP
    decode on cuda:1 can fail while both shards are nominally within budget.

    Placement is the whole point: this has to be after the sampler's last step
    and before the decode node runs.  ``KSAMPLER.sample`` is the single choke
    point every H3 route shares, so wrapping its exit covers video decode,
    audio decode and the CPU-offload VAE path alike -- unlike a hook on
    ``H3ParallelVAE.decode``, which only fires when the layer-MP VAE is
    resident.

    The returned latent is not touched.  Only allocator blocks that nothing
    references any more, plus each shard's per-request snapshot, are dropped;
    weights stay resident so the next request does not pay the ~34 s reload.
    """
    if os.environ.get("H3_TP_POSTSAMPLE_RELEASE", "1").strip().lower() in {
        "0", "false", "no", "off",
    }:
        logging.info(
            "[H3TP] post-sample release disabled by H3_TP_POSTSAMPLE_RELEASE"
        )
        return False
    try:
        samplers = __import__("comfy.samplers", fromlist=["KSAMPLER"])
    except Exception:
        logging.debug("[H3TP] comfy.samplers unavailable; post-sample hook skipped")
        return False

    sampler_class = getattr(samplers, "KSAMPLER", None)
    if sampler_class is None:
        return False

    with _POSTSAMPLE_HOOK_LOCK:
        current = sampler_class.sample
        if getattr(current, _POSTSAMPLE_HOOK_MARKER, False):
            return False

        import functools

        @functools.wraps(current)
        def sample_then_release(self, *args, **kwargs):
            output = current(self, *args, **kwargs)
            runtime = _RUNTIME
            if runtime is None or runtime.closed or not runtime.started:
                return output
            try:
                runtime.release_cached_memory()
            except Exception:
                # Cleanup must never fail a finished sample.  Decode may still
                # OOM afterwards, but the latent the user already paid for
                # stays valid and recoverable from the SaveLatent node.
                logging.warning(
                    "[H3TP] post-sample release failed; continuing", exc_info=True
                )
            return output

        setattr(sample_then_release, _POSTSAMPLE_HOOK_MARKER, True)
        sample_then_release._h3_tp_postsample_original_sample = current
        sampler_class.sample = sample_then_release
        logging.info(
            "[H3TP] post-sample DiT activation release installed on KSAMPLER.sample"
        )
        return True


atexit.register(close_runtime)


__all__ = [
    "H3TPRuntime",
    "RuntimeConfig",
    "active_runtime",
    "close_runtime",
    "get_runtime",
    "install_postsample_release_hook",
]
