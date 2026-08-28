"""Persistent two-way Q4 Tensor Parallel backbone for MiniMax H3.

This module contains no ComfyUI node code, so rank 1 can import it as a small
standalone process.  Both ranks keep only their compressed Q4_0 shards and
Turbo-LoRA shards resident.  Every block computes local heads/MLP channels and
uses two FP32 NCCL all-reduces to restore the replicated residual stream.

The implementation is intentionally specialized to the released H3 geometry
and the two V100-SXM2 cards used by this project.  It never mmaps a model file,
never creates a full CPU weight, and drops sequentially-read payload pages from
the cgroup page cache after each bounded staging transfer.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gguf
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel

try:  # Package import in ComfyUI.
    from . import h3_lora_tp as lora_tp
    from . import h3_group_cache_calibration as group_calibration
    from . import h3_q4_cache as q4_cache
    from . import h3_q4_tp as q4_tp
    from . import h3_v100_fp32_ops as fp32_ops
    from .h3_v100_rms_rope import h3_qk_rms_rope_sm70_
except ImportError:  # Standalone rank-1 script import from this directory.
    import h3_lora_tp as lora_tp
    import h3_group_cache_calibration as group_calibration
    import h3_q4_cache as q4_cache
    import h3_q4_tp as q4_tp
    import h3_v100_fp32_ops as fp32_ops
    from h3_v100_rms_rope import h3_qk_rms_rope_sm70_

from custom_nodes.NoHostMMap.gguf_reader import NoMmapGGUFReader


HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FFN = 14336
LAYERS = 50
TP_SIZE = 2
ADALN_INPUT = 8
ADALN_TIME_DIM = 2688
ADALN_MODALITIES = 3
ADALN_EXPAND = 6
FP16_SCALE_TARGET = 32752.0
MIB = 1 << 20


def process_memory_stats() -> dict[str, float | None]:
    """Return current/peak process RSS without importing psutil.

    Linux reports these values as KiB in ``/proc/self/status``.  Failure to
    read procfs is non-fatal because the same code is also importable by small
    offline tooling on other platforms.
    """

    values: dict[str, float | None] = {"rss_mib": None, "rss_peak_mib": None}
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return values
    keys = {"VmRSS:": "rss_mib", "VmHWM:": "rss_peak_mib"}
    for line in status.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in keys:
            values[keys[fields[0]]] = float(fields[1]) / 1024.0
    return values


def infer_target_stat_ranges(
    sequence: int,
    modulation_segments: Sequence[Sequence[int]],
) -> list[tuple[str, int, int]]:
    """Infer bounded context/audio/video ranges from H3's packed stream.

    MiniMax H3 always appends target audio and target video as the final two
    contiguous layout segments.  Text can be split into several modulation
    runs and reference media can sit between text and the targets, so the
    prefix is deliberately named ``context`` rather than incorrectly calling
    all of it text.
    """

    sequence = int(sequence)
    if sequence < 0:
        raise ValueError(f"sequence must be non-negative, got {sequence}")
    if len(modulation_segments) >= 2:
        audio = tuple(int(value) for value in modulation_segments[-2][:2])
        video = tuple(int(value) for value in modulation_segments[-1][:2])
        if (
            0 <= audio[0] < audio[1]
            and audio[1] == video[0]
            and video[0] < video[1] == sequence
        ):
            ranges: list[tuple[str, int, int]] = []
            if audio[0] > 0:
                ranges.append(("context", 0, audio[0]))
            ranges.extend(
                [
                    ("audio", audio[0], audio[1]),
                    ("video", video[0], video[1]),
                ]
            )
            return ranges
    return [("packed", 0, sequence)]


def _validated_stat_ranges(
    sequence: int,
    ranges: Sequence[Sequence[Any]] | None,
) -> list[tuple[str, int, int]]:
    if ranges is None:
        return []
    validated: list[tuple[str, int, int]] = []
    for item in ranges:
        if len(item) != 3:
            raise ValueError(f"invalid H3 TP stat range {item!r}")
        label, start, stop = str(item[0]), int(item[1]), int(item[2])
        if not label or not 0 <= start < stop <= sequence:
            raise ValueError(f"invalid H3 TP stat range {item!r} for S={sequence}")
        validated.append((label, start, stop))
    return validated


@torch.inference_mode()
def tensor_scalar_stats(
    tensor: torch.Tensor,
    ranges: Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Calculate scalar norms without allocating a tensor-sized square/abs.

    ``torch.linalg.vector_norm`` reduces directly.  That matters at 1 MP,
    where a seemingly harmless ``tensor.square()`` temporary for the FP32 H3
    stream would be roughly 0.76 GiB per rank.
    """

    if tensor.ndim != 2:
        raise ValueError(f"H3 TP scalar stats expect a 2D tensor, got {tensor.shape}")
    selected = [("overall", 0, int(tensor.shape[0]))]
    selected.extend(_validated_stat_ranges(int(tensor.shape[0]), ranges))
    pending: list[tuple[str, int, int, int, torch.Tensor, torch.Tensor]] = []
    for label, start, stop in selected:
        flat = tensor[start:stop].reshape(-1)
        count = int(flat.numel())
        if count == 0:
            continue
        pending.append(
            (
                label,
                start,
                stop,
                count,
                torch.linalg.vector_norm(flat),
                torch.linalg.vector_norm(flat, ord=float("inf")),
            )
        )
    if pending:
        packed = torch.stack(
            [value for item in pending for value in (item[4], item[5])]
        ).float().cpu()
    else:
        packed = torch.empty(0, dtype=torch.float32)
    reports: dict[str, dict[str, Any]] = {}
    for index, (label, start, stop, count, _l2, _linf) in enumerate(pending):
        l2 = float(packed[index * 2].item())
        max_abs = float(packed[index * 2 + 1].item())
        reports[label] = {
            "token_start": start,
            "token_stop": stop,
            "tokens": stop - start,
            "elements": count,
            "l2_norm": l2,
            "rms": l2 / math.sqrt(count),
            "max_abs": max_abs,
            "finite": bool(math.isfinite(l2) and math.isfinite(max_abs)),
        }
    overall = reports.pop("overall")
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "overall": overall,
        "segments": reports,
    }


@torch.inference_mode()
def deterministic_input_sketch(
    tensor: torch.Tensor,
    *,
    max_tokens: int = 2048,
    hidden_samples: int = 32,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Copy a deterministic, bounded sample used for adjacent-input deltas."""

    if tensor.ndim != 2:
        raise ValueError(f"H3 TP input sketch expects 2D, got {tensor.shape}")
    token_stride = max(1, math.ceil(int(tensor.shape[0]) / max_tokens))
    hidden_stride = max(1, math.ceil(int(tensor.shape[1]) / hidden_samples))
    sketch = tensor[::token_stride, ::hidden_stride].detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    return sketch, {
        "sample_elements": int(sketch.numel()),
        "full_elements": int(tensor.numel()),
        "sample_fraction": float(sketch.numel() / max(1, tensor.numel())),
        "token_stride": token_stride,
        "hidden_stride": hidden_stride,
    }


class _CudaStageRecorder:
    """Optional CUDA-event recorder with scalar-only output.

    Fine-grained timings are explicitly opt-in.  The recorder retains only
    CUDA event pairs until the forward synchronizes and then emits scalar
    count/total/mean/min/max values; it never copies or stores activations.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._pending: dict[str, torch.cuda.Event] = {}
        self._samples: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def begin(self, label: str) -> torch.cuda.Event | None:
        if not self.enabled:
            return None
        if label in self._pending:
            raise RuntimeError(f"nested H3 CUDA stage is not supported: {label}")
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._pending[label] = event
        return event

    def end(self, label: str) -> torch.cuda.Event | None:
        if not self.enabled:
            return None
        start = self._pending.pop(label, None)
        if start is None:
            raise RuntimeError(f"H3 CUDA stage ended without begin: {label}")
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._samples.setdefault(label, []).append((start, end))
        return end

    def summary(self) -> dict[str, dict[str, float | int]]:
        if self._pending:
            raise RuntimeError(f"unfinished H3 CUDA stages: {sorted(self._pending)}")
        result: dict[str, dict[str, float | int]] = {}
        for label, samples in self._samples.items():
            values = [start.elapsed_time(end) for start, end in samples]
            total = float(sum(values))
            result[label] = {
                "count": len(values),
                "total_ms": total,
                "mean_ms": total / len(values),
                "min_ms": float(min(values)),
                "max_ms": float(max(values)),
            }
        return result


DEFAULT_MODEL = Path(
    "/mnt/GALAX/minimax-h3/models/diffusion_models/"
    "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf"
)
DEFAULT_LORA = Path(
    "/mnt/GALAX/minimax-h3/models/loras/"
    "minimax_h3_turbo_v4_step600_ema.safetensors"
)
DEFAULT_EGRID = Path(
    "/home/regen/minimax-h3/ComfyUI/custom_nodes/"
    "ComfyUI-MiniMax-H3-Turbo/h3_silu_temb_grid.safetensors"
)


@dataclass(frozen=True)
class DenseGGUFSpec:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    data_offset: int
    n_bytes: int


@dataclass
class TPBlockWeights:
    q4: dict[str, Any]
    lora: dict[str, Any]
    norm1: torch.Tensor
    norm2: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    adaln_weight: torch.Tensor
    adaln_bias: torch.Tensor
    # LightX v1.1 has no block AdaLN adapters.  Keep the slot optional so the
    # persistent worker can still own its 50 core blocks; the regular ComfyUI
    # path remains responsible for any token-refiner adapters in that file.
    adaln_lora_a: torch.Tensor | None
    adaln_lora_b: torch.Tensor | None


class H3TPResidualCache:
    """Bounded Q4_0 residual cache used by the TP-aware TE-Speed path.

    The cache is deliberately local to one TP process.  Both NCCL ranks keep
    their own identical residual so a cache step does not need a third full
    sequence broadcast.  The persistent tensor is standard GGML Q4_0.  It is
    dequantized only in bounded row chunks while adding to the FP32 stream, so
    CACHE never creates a complete dequantized copy.  No cache tensor is
    written to disk.
    """

    def __init__(
        self,
        policy: str = "cpu",
        chunk_rows: int = q4_cache.DEFAULT_CACHE_CHUNK_ROWS,
    ) -> None:
        self.tensor: q4_cache.Q4Tensor | None = None
        self.anchor_stats: dict[str, Any] | None = None
        self.last_operation: dict[str, Any] = {"kind": "idle"}
        self.chunk_rows = int(chunk_rows)
        if self.chunk_rows <= 0:
            raise ValueError("H3 TP Q4 cache chunk_rows must be positive")
        self.policy = "cpu"
        self.set_policy(policy)

    def set_policy(self, policy: str) -> bool:
        normalized = str(policy).lower()
        if normalized == "auto":
            normalized = "cpu"
        if normalized not in {"cpu", "gpu"}:
            raise ValueError(f"H3 TP TE-Speed cache policy must be cpu/gpu, got {policy!r}")
        changed = normalized != self.policy
        if changed:
            self.clear()
            self.policy = normalized
        return changed

    def clear(self) -> None:
        old = self.tensor
        self.tensor = None
        self.anchor_stats = None
        self.last_operation = {"kind": "clear"}
        del old

    @property
    def ready(self) -> bool:
        return self.tensor is not None

    @property
    def bytes(self) -> int:
        return 0 if self.tensor is None else self.tensor.bytes

    @torch.inference_mode()
    def store(
        self,
        output: torch.Tensor,
        snapshot: torch.Tensor | None,
        *,
        stat_ranges: Sequence[Sequence[Any]] | None = None,
        collect_stats: bool = False,
        retain: bool = True,
    ) -> dict[str, Any] | None:
        if snapshot is None:
            raise RuntimeError("H3 TP TE-Speed FULL step did not capture its warm snapshot")
        if snapshot.shape != output.shape or snapshot.dtype != output.dtype:
            raise ValueError(
                "H3 TP TE-Speed snapshot/output mismatch: "
                f"{tuple(snapshot.shape)}/{snapshot.dtype} vs "
                f"{tuple(output.shape)}/{output.dtype}"
            )
        self.clear()
        # Reuse the snapshot allocation.  This is output - h_warm without a
        # third full-sequence GPU tensor at 1 MP.
        build_start = build_end = None
        if collect_stats:
            build_start = torch.cuda.Event(enable_timing=True)
            build_end = torch.cuda.Event(enable_timing=True)
            build_start.record()
        snapshot.neg_().add_(output)
        if build_end is not None:
            build_end.record()
        residual_stats = (
            tensor_scalar_stats(snapshot, stat_ranges) if collect_stats else None
        )
        build_ms = None
        if build_end is not None:
            build_end.synchronize()
            build_ms = float(build_start.elapsed_time(build_end))

        quantize_report = None
        if retain:
            self.tensor = q4_cache.quantize_q4_0(
                snapshot,
                policy=self.policy,
                chunk_rows=self.chunk_rows,
                measure=collect_stats,
            )
            quantize_report = dict(self.tensor.quantize_report)
            del snapshot
        else:
            del snapshot
        self.anchor_stats = residual_stats if retain else None
        self.last_operation = {
            "kind": "anchor_store" if retain else "anchor_analyze",
            "format": q4_cache.Q4_FORMAT if retain else None,
            "measured": bool(collect_stats),
            "residual_build_ms": build_ms,
            "quantize": quantize_report,
            "cache_bytes_after": self.bytes,
            "cache_device": self.policy if retain else None,
            "chunk_rows": self.chunk_rows,
        }
        return residual_stats

    @torch.inference_mode()
    def add_to(self, output: torch.Tensor, *, measure: bool = False) -> torch.Tensor:
        if self.tensor is None:
            raise RuntimeError("H3 TP TE-Speed CACHE step has no residual")
        self.last_operation = q4_cache.add_q4_to_(
            output,
            self.tensor,
            chunk_rows=self.chunk_rows,
            measure=measure,
        )
        return output


def core_names(block: int) -> dict[str, str]:
    prefix = f"blocks.{block}"
    return {
        "qkv": f"{prefix}.attn.qkv_proj.weight",
        "out_proj": f"{prefix}.attn.out_proj.weight",
        "fc1": f"{prefix}.mlp.fc1.weight",
        "fc2": f"{prefix}.mlp.fc2.weight",
    }


def dense_names(block: int) -> dict[str, str]:
    prefix = f"blocks.{block}"
    return {
        "norm1": f"{prefix}.norm1.weight",
        "norm2": f"{prefix}.norm2.weight",
        "q_norm": f"{prefix}.attn.q_norm.weight",
        "k_norm": f"{prefix}.attn.k_norm.weight",
        "adaln_weight": f"{prefix}.adaln_proj.linear.weight",
        "adaln_bias": f"{prefix}.adaln_proj.linear.bias",
    }


def adaln_lora_names(block: int) -> tuple[str, str]:
    base = f"blocks.{block}.adaln_proj.linear"
    return f"{base}.lora_A.weight", f"{base}.lora_B.weight"


def _inspect_dense_specs(path: os.PathLike[str] | str, names: set[str]):
    scalar = {
        gguf.GGMLQuantizationType.F16: torch.float16,
        gguf.GGMLQuantizationType.F32: torch.float32,
    }
    reader = NoMmapGGUFReader(path)
    result: dict[str, DenseGGUFSpec] = {}
    for tensor in reader.tensors:
        if tensor.name not in names:
            continue
        dtype = scalar.get(tensor.tensor_type)
        if dtype is None:
            raise ValueError(
                f"dense TP tensor {tensor.name} uses unsupported {tensor.tensor_type.name}"
            )
        shape = tuple(int(value) for value in reversed(tensor.shape))
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if expected != int(tensor.n_bytes):
            raise ValueError(
                f"dense TP tensor {tensor.name} byte mismatch: {expected} != {tensor.n_bytes}"
            )
        result[tensor.name] = DenseGGUFSpec(
            name=tensor.name,
            dtype=dtype,
            shape=shape,
            data_offset=int(tensor.data_offset),
            n_bytes=int(tensor.n_bytes),
        )
    missing = names.difference(result)
    if missing:
        raise KeyError(f"GGUF is missing dense TP tensors: {sorted(missing)}")
    return result


def _read_dense(
    reader: Any,
    spec: DenseGGUFSpec,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    raw = torch.empty(spec.n_bytes, dtype=torch.uint8, device=reader.device)
    reader._copy_contiguous(raw, 0, spec.data_offset, spec.n_bytes)
    value = raw.view(spec.dtype).reshape(spec.shape)
    if target_dtype is not None and value.dtype != target_dtype:
        value = value.to(target_dtype)
    return value


def _validate_geometry(q4_specs: dict[str, Any]) -> None:
    expected = {
        "qkv": (3 * INNER, HIDDEN),
        "out_proj": (HIDDEN, INNER),
        "fc1": (2 * FFN, HIDDEN),
        "fc2": (HIDDEN, FFN),
    }
    for block in range(LAYERS):
        names = core_names(block)
        actual = {role: q4_specs[name].shape for role, name in names.items()}
        if actual != expected:
            raise ValueError(f"block {block} H3 geometry mismatch: {actual}")


def _load_egrid(path: Path, device: torch.device, staging_bytes: int) -> torch.Tensor:
    name = "silu_t_emb_grid"
    specs, _ = lora_tp.inspect_safetensors(path, {name})
    with lora_tp.SafeTensorDiskReader(path, device, staging_bytes) as reader:
        # ClipProj conditioning is intentionally FP32, which makes H3's curve
        # t_emb and the installed Turbo AdaLN injection FP32 as well.  Preserve
        # that path; only the attention/MLP branch is reduced to FP16 later.
        return reader.read_full(specs[name], torch.float32)


class H3TPBackbone:
    """One rank's persistent weights and local half of the 50-layer backbone."""

    def __init__(
        self,
        *,
        rank: int,
        device: torch.device | str,
        model_path: os.PathLike[str] | str = DEFAULT_MODEL,
        lora_path: os.PathLike[str] | str = DEFAULT_LORA,
        egrid_path: os.PathLike[str] | str = DEFAULT_EGRID,
        lora_strength: float = 1.0,
        staging_bytes: int = 4 << 20,
        chunk_rows: int = 2048,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        if rank not in (0, 1):
            raise ValueError(f"H3 TP rank must be 0/1, got {rank}")
        if staging_bytes <= 0 or chunk_rows <= 0:
            raise ValueError("staging_bytes and chunk_rows must be positive")
        self.rank = rank
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.lora_path = Path(lora_path)
        self.egrid_path = Path(egrid_path)
        self.lora_strength = float(lora_strength)
        self.staging_bytes = staging_bytes
        self.chunk_rows = chunk_rows
        # The residual stream is deliberately FP32 on V100.  Keeping one
        # full-sequence FP32 MLP update is safe at the baseline shape, but a
        # 1MP request with keyframe/reference rows can cross the last few GiB
        # on rank 0.  Reduce and apply the MLP update in bounded groups while
        # retaining the same FP32/NCCL computation.
        self.mlp_reduce_rows = max(
            self.chunk_rows,
            int(os.environ.get("H3_TP_MLP_REDUCE_ROWS", "8192")),
        )
        self.local_heads = HEADS // TP_SIZE
        self.local_inner = INNER // TP_SIZE
        self.local_ffn = FFN // TP_SIZE
        self.progress = progress
        self.fused_fp32_ops = os.environ.get(
            "H3_TP_FUSED_FP32_OPS", "1"
        ).strip().lower() not in {"0", "false", "off", "no", "eager"}
        self.fp32_ops_warps = int(os.environ.get("H3_TP_FP32_OPS_WARPS", "8"))
        # PyTorch's SM70 efficient-SDPA kernel is materially faster when Q/K/V
        # use standard contiguous BHSD storage.  H3's fused projection instead
        # leaves three token-major views with a 3x sequence stride.  An
        # explicit bounded deinterleave costs ~2.5 ms at S=37746 but saves
        # ~80 ms in SDPA.  The production launcher selects Q-only after its
        # exact-output gate; the original strided path remains the fallback.
        compact_qkv = os.environ.get("H3_TP_COMPACT_QKV", "0").strip().lower()
        if compact_qkv in {"0", "false", "off", "no", "none", "strided"}:
            self.compact_qkv_mode = "none"
        elif compact_qkv in {"q", "query"}:
            self.compact_qkv_mode = "q"
        elif compact_qkv in {"1", "true", "on", "yes", "all", "compact", "bhsd"}:
            self.compact_qkv_mode = "all"
        else:
            raise ValueError(
                "H3_TP_COMPACT_QKV must be none/0, q, or all/1; "
                f"got {compact_qkv!r}"
            )
        self.compact_qkv = self.compact_qkv_mode != "none"
        self.compact_qkv_min_sequence = int(
            os.environ.get("H3_TP_COMPACT_QKV_MIN_SEQUENCE", "4096")
        )
        if self.compact_qkv_min_sequence < 0:
            raise ValueError("H3_TP_COMPACT_QKV_MIN_SEQUENCE must be non-negative")
        # The packed sequence/modality layout is stable across the four Turbo
        # denoise steps.  Keep only this tiny int32 row map between forwards;
        # it avoids rebuilding the same CUDA buffer while leaving all model
        # weights and activations on their existing lifetime.
        self._mod_rows_cache_key: tuple[object, ...] | None = None
        self._mod_rows_cache: torch.Tensor | None = None
        # A FULL TE-Speed pass captures one in-flight warm-prefix snapshot.
        # It is consumed by H3TPResidualCache immediately after forward; it is
        # never retained across calls by the backbone itself.
        self._last_snapshot: torch.Tensor | None = None
        # Block-stat collection keeps only a bounded deterministic sample of
        # the previous input.  At 1 MP this is hundreds of KiB, not another
        # 0.76 GiB FP32 hidden stream.
        self._te_input_sketch: torch.Tensor | None = None
        self._te_input_sketch_meta: dict[str, int | float] | None = None

        if self.device.type != "cuda" or self.device.index != rank:
            raise ValueError(f"rank {rank} must own cuda:{rank}, got {self.device}")
        if torch.cuda.get_device_capability(self.device) != (7, 0):
            raise ValueError("persistent H3 TP backbone is specialized for SM70")
        if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
            raise RuntimeError("initialize a two-rank NCCL process group before H3TPBackbone")

        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.blocks, self.adaln_table, self.egrid, bytes_by_kind = self._load()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        self.load_stats = {
            "seconds": time.monotonic() - started,
            "rank": rank,
            "device": str(self.device),
            "compressed_q4_mib": bytes_by_kind["q4"] / MIB,
            "core_lora_mib": bytes_by_kind["core_lora"] / MIB,
            "adaln_base_mib": bytes_by_kind["adaln_base"] / MIB,
            "adaln_lora_mib": bytes_by_kind["adaln_lora"] / MIB,
            "other_mib": bytes_by_kind["other"] / MIB,
            "allocated_mib": torch.cuda.memory_allocated(self.device) / MIB,
            "reserved_mib": torch.cuda.memory_reserved(self.device) / MIB,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / MIB,
            "payload_mmap": False,
            "staging_mib": staging_bytes / MIB,
        }

    def _report(self, stage: str, current: int, total: int) -> None:
        if self.progress is not None:
            self.progress(stage, current, total)

    def take_snapshot(self) -> torch.Tensor | None:
        snapshot = self._last_snapshot
        self._last_snapshot = None
        return snapshot

    def clear_snapshot(self) -> None:
        snapshot = self._last_snapshot
        self._last_snapshot = None
        del snapshot

    def clear_block_stats_state(self) -> None:
        previous = self._te_input_sketch
        self._te_input_sketch = None
        self._te_input_sketch_meta = None
        del previous

    @torch.inference_mode()
    def _sampled_input_change(self, residual: torch.Tensor) -> dict[str, Any]:
        current, metadata = deterministic_input_sketch(residual)
        previous = self._te_input_sketch
        previous_meta = self._te_input_sketch_meta
        report: dict[str, Any] = {
            **metadata,
            "estimator": "deterministic_strided_hidden_sample",
            "available": False,
        }
        if (
            previous is not None
            and previous_meta is not None
            and previous.shape == current.shape
            and previous_meta.get("token_stride") == metadata["token_stride"]
            and previous_meta.get("hidden_stride") == metadata["hidden_stride"]
        ):
            delta = current - previous
            count = max(1, int(delta.numel()))
            delta_l2 = float(torch.linalg.vector_norm(delta).item())
            previous_l2 = float(torch.linalg.vector_norm(previous).item())
            report.update(
                {
                    "available": True,
                    "sampled_l2_norm": delta_l2,
                    "sampled_rms": delta_l2 / math.sqrt(count),
                    "sampled_relative_rms": delta_l2 / max(previous_l2, 1e-30),
                    "sampled_max_abs": float(
                        torch.linalg.vector_norm(delta, ord=float("inf")).item()
                    ),
                }
            )
            del delta
        self._te_input_sketch = current
        self._te_input_sketch_meta = metadata
        del previous
        return report

    def _load(self):
        all_core_names = {
            name for block in range(LAYERS) for name in core_names(block).values()
        }
        q4_specs, _ = q4_tp.inspect_q4_matrices(self.model_path, all_core_names)
        _validate_geometry(q4_specs)
        all_dense_names = {
            name for block in range(LAYERS) for name in dense_names(block).values()
        }
        dense_specs = _inspect_dense_specs(self.model_path, all_dense_names | {"adaln_t_table"})

        all_lora_names = set()
        optional_adaln_names = set()
        for block in range(LAYERS):
            for pair in lora_tp.h3_lora_names(block).values():
                all_lora_names.update(pair)
            adaln_names = adaln_lora_names(block)
            all_lora_names.update(adaln_names)
            optional_adaln_names.update(adaln_names)
        lora_specs, _ = lora_tp.inspect_safetensors(
            self.lora_path,
            all_lora_names,
            optional_names=optional_adaln_names,
        )

        q4_by_block: list[dict[str, Any]] = []
        dense_by_block: list[dict[str, torch.Tensor]] = []
        q4_bytes = 0
        adaln_base_bytes = 0
        other_bytes = 0
        with q4_tp.Q4DiskReader(
            self.model_path, self.device, self.staging_bytes
        ) as reader:
            for block in range(LAYERS):
                q4_values = {}
                for role, name in core_names(block).items():
                    shard = reader.read_tp_shard(
                        q4_specs[name], role, self.rank, TP_SIZE
                    )
                    q4_values[role] = shard
                    q4_bytes += shard.raw.numel()
                q4_by_block.append(q4_values)

                values = {}
                for role, name in dense_names(block).items():
                    target_dtype = (
                        torch.float32
                        if role in {"adaln_weight", "adaln_bias"}
                        else None
                    )
                    values[role] = _read_dense(
                        reader, dense_specs[name], target_dtype
                    )
                dense_by_block.append(values)
                adaln_base_bytes += (
                    values["adaln_weight"].numel()
                    * values["adaln_weight"].element_size()
                    + values["adaln_bias"].numel()
                    * values["adaln_bias"].element_size()
                )
                other_bytes += sum(
                    values[role].numel() * values[role].element_size()
                    for role in ("norm1", "norm2", "q_norm", "k_norm")
                )
                self._report("base", block + 1, LAYERS)
            adaln_table = _read_dense(
                reader, dense_specs["adaln_t_table"], torch.float32
            )
            other_bytes += adaln_table.numel() * adaln_table.element_size()

        core_lora_by_block: list[dict[str, Any]] = []
        adaln_lora_by_block: list[tuple[torch.Tensor, torch.Tensor]] = []
        core_lora_bytes = 0
        adaln_lora_bytes = 0
        with lora_tp.SafeTensorDiskReader(
            self.lora_path, self.device, self.staging_bytes
        ) as reader:
            for block in range(LAYERS):
                core_values = {}
                for role, (a_name, b_name) in lora_tp.h3_lora_names(block).items():
                    a_spec, b_spec = lora_specs[a_name], lora_specs[b_name]
                    target_dtype = (
                        torch.float16 if role in {"qkv", "fc1"} else torch.float32
                    )
                    if role == "qkv":
                        a = reader.read_full(a_spec, target_dtype)
                        b = reader.read_output_segments(
                            b_spec, 3, self.rank, TP_SIZE, target_dtype
                        )
                    elif role == "fc1":
                        a = reader.read_full(a_spec, target_dtype)
                        b = reader.read_output_segments(
                            b_spec, 2, self.rank, TP_SIZE, target_dtype
                        )
                    else:
                        a = reader.read_input_shard(
                            a_spec, self.rank, TP_SIZE, target_dtype
                        )
                        b = reader.read_full(b_spec, target_dtype)
                    value = lora_tp.LoRALinearShard(a=a, b=b, role=role)
                    core_values[role] = value
                    core_lora_bytes += (
                        a.numel() * a.element_size() + b.numel() * b.element_size()
                    )
                core_lora_by_block.append(core_values)

                a_name, b_name = adaln_lora_names(block)
                has_a = a_name in lora_specs
                has_b = b_name in lora_specs
                if has_a != has_b:
                    raise KeyError(
                        f"LoRA has only one AdaLN factor for block {block}: "
                        f"A={has_a}, B={has_b}"
                    )
                if has_a:
                    adaln_a = reader.read_full(lora_specs[a_name], torch.float32)
                    adaln_b = reader.read_full(lora_specs[b_name], torch.float32)
                    adaln_lora_bytes += (
                        adaln_a.numel() * adaln_a.element_size()
                        + adaln_b.numel() * adaln_b.element_size()
                    )
                    adaln_lora_by_block.append((adaln_a, adaln_b))
                else:
                    adaln_lora_by_block.append((None, None))
                self._report("lora", block + 1, LAYERS)

        egrid = _load_egrid(self.egrid_path, self.device, self.staging_bytes)
        other_bytes += egrid.numel() * egrid.element_size()
        blocks = []
        for block in range(LAYERS):
            dense = dense_by_block[block]
            adaln_a, adaln_b = adaln_lora_by_block[block]
            blocks.append(
                TPBlockWeights(
                    q4=q4_by_block[block],
                    lora=core_lora_by_block[block],
                    norm1=dense["norm1"],
                    norm2=dense["norm2"],
                    q_norm=dense["q_norm"],
                    k_norm=dense["k_norm"],
                    adaln_weight=dense["adaln_weight"],
                    adaln_bias=dense["adaln_bias"],
                    adaln_lora_a=adaln_a,
                    adaln_lora_b=adaln_b,
                )
            )
        return blocks, adaln_table, egrid, {
            "q4": q4_bytes,
            "core_lora": core_lora_bytes,
            "adaln_base": adaln_base_bytes,
            "adaln_lora": adaln_lora_bytes,
            "other": other_bytes,
        }

    @staticmethod
    def _power_of_two_scale(x: torch.Tensor) -> torch.Tensor:
        row_max = x.detach().abs().amax(dim=-1, keepdim=True)
        ratio = (row_max / FP16_SCALE_TARGET).clamp_min_(1.0)
        return torch.exp2(torch.ceil(torch.log2(ratio)))

    def _get_modulation_rows(
        self,
        sequence: int,
        segments: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, bool]:
        key = (
            int(sequence),
            tuple(tuple(int(value) for value in segment) for segment in segments),
        )
        if self._mod_rows_cache_key == key and self._mod_rows_cache is not None:
            return self._mod_rows_cache, True
        rows = fp32_ops.make_modulation_rows(sequence, segments, self.device)
        self._mod_rows_cache_key = key
        self._mod_rows_cache = rows
        return rows, False

    def _add_lora_(
        self,
        output: torch.Tensor,
        x: torch.Tensor,
        shard: Any,
        *,
        stages: _CudaStageRecorder | None = None,
        stage_name: str | None = None,
    ) -> torch.Tensor:
        if stages is not None and stage_name is not None:
            stages.begin(stage_name)
        try:
            rows = x.reshape(-1, x.shape[-1])
            out_rows = output.reshape(-1, output.shape[-1])
            for start in range(0, rows.shape[0], self.chunk_rows):
                stop = min(start + self.chunk_rows, rows.shape[0])
                source = rows[start:stop]
                if source.dtype != shard.a.dtype:
                    source = source.to(shard.a.dtype)
                delta = functional.linear(functional.linear(source, shard.a), shard.b)
                if delta.dtype != out_rows.dtype:
                    delta = delta.to(out_rows.dtype)
                out_rows[start:stop].add_(delta, alpha=self.lora_strength)
        finally:
            if stages is not None and stage_name is not None:
                stages.end(stage_name)
        return output

    def _column_linear(
        self,
        x: torch.Tensor,
        matrix: Any,
        lora: Any,
        *,
        stages: _CudaStageRecorder | None = None,
        stage_prefix: str = "column",
    ) -> torch.Tensor:
        if stages is not None:
            stages.begin(f"{stage_prefix}_dequant")
        try:
            weight = q4_tp.dequantize_q4_0(matrix, torch.float16)
        finally:
            if stages is not None:
                stages.end(f"{stage_prefix}_dequant")
        if stages is not None:
            stages.begin(f"{stage_prefix}_gemm")
        try:
            output = functional.linear(x, weight)
        finally:
            if stages is not None:
                stages.end(f"{stage_prefix}_gemm")
        del weight
        return self._add_lora_(
            output,
            x,
            lora,
            stages=stages,
            stage_name=f"{stage_prefix}_lora",
        )

    def _attention_row_linear(
        self,
        x_fp16: torch.Tensor,
        matrix: Any,
        lora: Any,
        *,
        stages: _CudaStageRecorder | None = None,
        stage_prefix: str = "out_proj",
    ) -> torch.Tensor:
        if stages is not None:
            stages.begin(f"{stage_prefix}_dequant")
        try:
            weight = q4_tp.dequantize_q4_0(matrix, torch.float16)
        finally:
            if stages is not None:
                stages.end(f"{stage_prefix}_dequant")
        if stages is not None:
            stages.begin(f"{stage_prefix}_gemm")
        try:
            output = torch.mm(x_fp16, weight.t(), out_dtype=torch.float32)
        finally:
            if stages is not None:
                stages.end(f"{stage_prefix}_gemm")
        del weight
        # Keep the projection input resident as FP16.  _add_lora_ converts only
        # one bounded row chunk to the LoRA's FP32 dtype instead of allocating a
        # full-sequence FP32 attention tensor at 1 MP.
        return self._add_lora_(
            output,
            x_fp16,
            lora,
            stages=stages,
            stage_name=f"{stage_prefix}_lora",
        )

    def _fc2_row_linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        lora: Any,
        *,
        safe: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        stages: _CudaStageRecorder | None = None,
    ) -> torch.Tensor:
        if safe is None or scale is None:
            if stages is not None:
                stages.begin("fc2_input_scale")
            try:
                scale = self._power_of_two_scale(x)
                safe = (x / scale).to(torch.float16)
            finally:
                if stages is not None:
                    stages.end("fc2_input_scale")
        if stages is not None:
            stages.begin("fc2_gemm")
        try:
            output = torch.mm(safe, weight.t(), out_dtype=torch.float32)
            output.mul_(scale)
        finally:
            if stages is not None:
                stages.end("fc2_gemm")
        return self._add_lora_(
            output,
            x,
            lora,
            stages=stages,
            stage_name="fc2_lora",
        )

    @staticmethod
    def _mod_scale_shift(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        segments: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        for start, stop, row in segments:
            x[start:stop].mul_(1.0 + scale[row].to(x.dtype)).add_(
                shift[row].to(x.dtype)
            )
        return x

    @staticmethod
    def _mod_gate(
        residual: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor,
        segments: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        for start, stop, row in segments:
            residual[start:stop].addcmul_(
                update[start:stop].float(), gate[row].float()
            )
        return residual

    @staticmethod
    def _mod_gate_range(
        residual: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor,
        segments: Sequence[Sequence[int]],
        start: int,
    ) -> torch.Tensor:
        """Apply a bounded MLP update whose first row is ``start``."""
        stop = start + update.shape[0]
        for segment_start, segment_stop, row in segments:
            lo = max(start, segment_start)
            hi = min(stop, segment_stop)
            if lo >= hi:
                continue
            update_lo = lo - start
            update_hi = hi - start
            residual[lo:hi].addcmul_(
                update[update_lo:update_hi].float(), gate[row].float()
            )
        return residual

    def _adaln(self, block: TPBlockWeights, t_emb: torch.Tensor, silu_temb: torch.Tensor):
        output = functional.linear(t_emb, block.adaln_weight, block.adaln_bias)
        if block.adaln_lora_a is not None and block.adaln_lora_b is not None:
            delta = functional.linear(
                functional.linear(silu_temb, block.adaln_lora_a),
                block.adaln_lora_b,
            )
            output.add_(delta, alpha=self.lora_strength)
        output = output.view(
            output.shape[0] * ADALN_MODALITIES,
            ADALN_EXPAND * HIDDEN,
        )
        return output.chunk(ADALN_EXPAND, dim=-1)

    @torch.inference_mode()
    def _group_condition_signature(
        self,
        start_block: int,
        end_block: int,
        t_emb: torch.Tensor,
        silu_temb: torch.Tensor,
        hidden_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build a bounded CPU AdaLN signature for one cache group.

        This is intentionally calibration-only.  It reruns the tiny AdaLN
        projections before the group decision, but does not retain an
        activation or alter the regular block computation.  The signature is
        sampled over hidden channels and has no dependency on packed sequence
        length, so it remains safe for the 1 MP route.
        """

        start_block = int(start_block)
        end_block = int(end_block)
        if not 0 <= start_block < end_block <= len(self.blocks):
            raise ValueError(
                f"invalid H3 calibration group [{start_block}, {end_block})"
            )
        signature: torch.Tensor | None = None
        for local_index, block_index in enumerate(range(start_block, end_block)):
            modulation = self._adaln(
                self.blocks[block_index], t_emb, silu_temb
            )
            sampled = group_calibration.sampled_modulation_signature(
                modulation, hidden_indices
            )
            if signature is None:
                signature = torch.empty(
                    (end_block - start_block, *sampled.shape),
                    dtype=torch.float32,
                    device=self.device,
                )
            signature[local_index].copy_(sampled)
            del modulation, sampled
        if signature is None:
            raise RuntimeError("H3 calibration group produced no AdaLN signature")
        cpu_signature = signature.to(device="cpu", dtype=torch.float32)
        report = {
            "estimator": "per_block_per_adaln_row_deterministic_hidden_sample",
            "storage": "cpu",
            "signature_shape": [int(value) for value in cpu_signature.shape],
            "signature_bytes": group_calibration.signature_bytes(cpu_signature),
            "hidden_samples": int(cpu_signature.shape[-1]),
            "adaln_rows": int(cpu_signature.shape[-2]),
            "components": list(group_calibration.CONDITION_COMPONENTS),
            "blocks": int(end_block - start_block),
        }
        del signature
        return cpu_signature, report

    def _forward_block(
        self,
        residual: torch.Tensor,
        block: TPBlockWeights,
        t_emb: torch.Tensor,
        silu_temb: torch.Tensor,
        segments: Sequence[Sequence[int]],
        mod_rows: torch.Tensor | None,
        rope_freqs: torch.Tensor,
        attention_collective_events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
        mlp_collective_events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
        stages: _CudaStageRecorder,
    ) -> torch.Tensor:
        stages.begin("adaln")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._adaln(block, t_emb, silu_temb)
        )
        stages.end("adaln")

        stages.begin("norm1_modulation")
        if mod_rows is None:
            branch = functional.rms_norm(
                residual, (HIDDEN,), weight=block.norm1, eps=1e-5
            ).to(torch.float16)
            self._mod_scale_shift(branch, shift_msa, scale_msa, segments)
        else:
            branch = fp32_ops.h3_fp32_rms_mod_sm70(
                residual,
                block.norm1,
                shift_msa,
                scale_msa,
                mod_rows,
                epsilon=1e-5,
                num_warps=self.fp32_ops_warps,
            )
        stages.end("norm1_modulation")
        packed = self._column_linear(
            branch,
            block.q4["qkv"],
            block.lora["qkv"],
            stages=stages,
            stage_prefix="qkv",
        )
        # QKV has consumed the modulated branch.  Keep only the q/k/v views
        # alive for attention; retaining the full [S, 5376] FP16 branch here
        # costs another ~0.45 GiB at the 1 MP two-keyframe shape and can push
        # rank 0 over 16 GiB before efficient SDPA gets a workspace.
        del branch
        q, k, v = packed.split(self.local_inner, dim=-1)
        sequence = residual.shape[0]
        q = q.view(1, sequence, self.local_heads, HEAD_DIM)
        k = k.view(1, sequence, self.local_heads, HEAD_DIM)
        v = v.view(sequence, self.local_heads, HEAD_DIM)
        stages.begin("qk_norm_rope")
        h3_qk_rms_rope_sm70_(
            q,
            k,
            rope_freqs,
            block.q_norm,
            block.k_norm,
            epsilon=1e-5,
            rot_dim=96,
            stabilize=True,
            num_warps=int(os.environ.get("H3_V100_RMS_ROPE_WARPS", "1")),
        )
        stages.end("qk_norm_rope")
        compact_qkv = (
            self.compact_qkv
            and sequence >= self.compact_qkv_min_sequence
        )
        if compact_qkv:
            stages.begin("attention_qkv_compact")
            qh = q[0].transpose(0, 1).unsqueeze(0).contiguous()
            if self.compact_qkv_mode == "all":
                kh = k[0].transpose(0, 1).unsqueeze(0).contiguous()
                vh = v.transpose(0, 1).unsqueeze(0).contiguous()
            else:
                kh = k[0].transpose(0, 1).unsqueeze(0)
                vh = v.transpose(0, 1).unsqueeze(0)
            stages.end("attention_qkv_compact")
            if self.compact_qkv_mode == "all":
                # All three compact tensors own their storage.  Release the
                # fused QKV allocation before SDPA so its 0.74 GiB can be
                # reused by the attention workspace/output instead of
                # overlapping both layouts.
                del packed, q, k, v
        else:
            qh = q[0].transpose(0, 1).unsqueeze(0)
            kh = k[0].transpose(0, 1).unsqueeze(0)
            vh = v.transpose(0, 1).unsqueeze(0)
        stages.begin("attention_sdpa")
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            attention = functional.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).contiguous()
        stages.end("attention_sdpa")
        attention = attention.reshape(sequence, self.local_inner)
        del qh, kh, vh
        if not compact_qkv or self.compact_qkv_mode != "all":
            del packed, q, k, v

        attention_partial = self._attention_row_linear(
            attention,
            block.q4["out_proj"],
            block.lora["out_proj"],
            stages=stages,
            stage_prefix="out_proj",
        )
        del attention
        collective_start = stages.begin("attention_all_reduce")
        if collective_start is None:
            collective_start = torch.cuda.Event(enable_timing=True)
            collective_start.record()
        dist.all_reduce(attention_partial, op=dist.ReduceOp.SUM) #all_reduce attn result
        collective_end = stages.end("attention_all_reduce")
        if collective_end is None:
            collective_end = torch.cuda.Event(enable_timing=True)
            collective_end.record()
        attention_collective_events.append((collective_start, collective_end))
        stages.begin("attention_residual_gate")
        self._mod_gate(residual, attention_partial, gate_msa, segments)
        stages.end("attention_residual_gate")
        del attention_partial

        stages.begin("norm2_modulation")
        if mod_rows is None:
            branch = functional.rms_norm(
                residual, (HIDDEN,), weight=block.norm2, eps=1e-5
            ).to(torch.float16)
            self._mod_scale_shift(branch, shift_mlp, scale_mlp, segments)
        else:
            branch = fp32_ops.h3_fp32_rms_mod_sm70(
                residual,
                block.norm2,
                shift_mlp,
                scale_mlp,
                mod_rows,
                epsilon=1e-5,
                num_warps=self.fp32_ops_warps,
            )
        stages.end("norm2_modulation")
        stages.begin("fc1_dequant")
        try:
            fc1_weight = q4_tp.dequantize_q4_0(block.q4["fc1"], torch.float16)
        finally:
            stages.end("fc1_dequant")
        stages.begin("fc2_dequant")
        try:
            fc2_weight = q4_tp.dequantize_q4_0(block.q4["fc2"], torch.float16)
        finally:
            stages.end("fc2_dequant")
        # Do not retain a full [sequence, HIDDEN] FP32 update.  The local FC2
        # chunks are packed into a bounded reduce buffer; after NCCL sums that
        # group, apply it immediately to the matching residual rows.  MLP is
        # row-local, so this is algebraically the same as one full all-reduce
        # followed by _mod_gate, while saving roughly 0.7 GiB at 1 MP.
        reduce_rows = min(sequence, self.mlp_reduce_rows)
        stages.begin("mlp_partial_allocation")
        mlp_group = torch.empty(
            (reduce_rows, HIDDEN), dtype=torch.float32, device=residual.device
        )
        stages.end("mlp_partial_allocation")
        for group_start in range(0, sequence, reduce_rows):
            group_stop = min(group_start + reduce_rows, sequence)
            for start in range(group_start, group_stop, self.chunk_rows):
                stop = min(start + self.chunk_rows, group_stop)
                source = branch[start:stop]
                stages.begin("fc1_gemm")
                hidden = functional.linear(source, fc1_weight)
                stages.end("fc1_gemm")
                self._add_lora_(
                    hidden,
                    source,
                    block.lora["fc1"],
                    stages=stages,
                    stage_name="fc1_lora",
                )
                stages.begin("swiglu_scale")
                if self.fused_fp32_ops:
                    swiglu, safe_swiglu, swiglu_scale = fp32_ops.h3_swiglu_scale_sm70(
                        hidden,
                        target=FP16_SCALE_TARGET,
                        num_warps=self.fp32_ops_warps,
                    )
                else:
                    gate, up = hidden.chunk(2, dim=-1)
                    swiglu = functional.silu(gate.float()).mul_(up.float())
                    del gate, up
                    safe_swiglu = None
                    swiglu_scale = None
                stages.end("swiglu_scale")
                partial = self._fc2_row_linear(
                    swiglu,
                    fc2_weight,
                    block.lora["fc2"],
                    safe=safe_swiglu,
                    scale=swiglu_scale,
                    stages=stages,
                )
                mlp_group[start - group_start : stop - group_start].copy_(partial)
                del hidden, swiglu, partial
                if safe_swiglu is not None:
                    del safe_swiglu, swiglu_scale

            stages.begin("mlp_all_reduce")
            collective_start = torch.cuda.Event(enable_timing=True)
            collective_start.record()
            dist.all_reduce(mlp_group[: group_stop - group_start], op=dist.ReduceOp.SUM)
            collective_end = torch.cuda.Event(enable_timing=True)
            collective_end.record()
            stages.end("mlp_all_reduce")
            mlp_collective_events.append((collective_start, collective_end))
            stages.begin("mlp_residual_gate")
            self._mod_gate_range(
                residual,
                mlp_group[: group_stop - group_start],
                gate_mlp,
                segments,
                group_start,
            )
            stages.end("mlp_residual_gate")
        del branch, fc1_weight, fc2_weight, mlp_group
        return residual

    @torch.inference_mode()
    def forward(
        self,
        residual: torch.Tensor,
        t_emb: torch.Tensor,
        segments: Sequence[Sequence[int]],
        rope_freqs: torch.Tensor,
        *,
        profile: bool = False,
        stage_profile: bool = False,
        start_block: int = 0,
        end_block: int | None = None,
        snapshot_at: int | None = None,
        collect_block_stats: bool = False,
        stat_ranges: Sequence[Sequence[Any]] | None = None,
        reset_block_stats: bool = False,
        group_cache: q4_cache.GroupResidualCache | None = None,
        group_config: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.clear_snapshot()
        if reset_block_stats:
            self.clear_block_stats_state()
        if residual.device != self.device or residual.dtype != torch.float32:
            raise ValueError(
                f"rank {self.rank} residual must be FP32 on {self.device}, "
                f"got {residual.dtype}/{residual.device}"
            )
        if residual.ndim != 2 or residual.shape[1] != HIDDEN:
            raise ValueError(f"unexpected H3 residual shape {tuple(residual.shape)}")
        if t_emb.device != self.device or t_emb.dtype != torch.float32:
            raise ValueError("H3 TP t_emb must be FP32 on the rank-local device")
        if rope_freqs.device != self.device or rope_freqs.dtype != torch.float16:
            raise ValueError("H3 TP RoPE table must be FP16 on the rank-local device")
        if rope_freqs.shape[1] != residual.shape[0]:
            raise ValueError("H3 TP RoPE sequence does not match the residual")
        block_count = len(self.blocks)
        start_block = int(start_block)
        end_block = block_count if end_block is None else int(end_block)
        if not 0 <= start_block <= end_block <= block_count:
            raise ValueError(
                f"invalid H3 TP block range [{start_block}, {end_block}) "
                f"for {block_count} blocks"
            )
        if snapshot_at is not None:
            snapshot_at = int(snapshot_at)
            if not start_block <= snapshot_at <= end_block:
                raise ValueError(
                    f"snapshot_at={snapshot_at} is outside block range "
                    f"[{start_block}, {end_block}]"
                )
        group_enabled = bool(group_config and group_config.get("enabled", False))
        if group_enabled:
            if group_cache is None:
                raise ValueError("H3 group cache is enabled without a cache instance")
            if start_block != 0 or end_block != block_count or snapshot_at is not None:
                raise ValueError(
                    "H3 group cache owns the complete block loop and cannot be "
                    "combined with a partial range or whole-tail snapshot"
                )
            if bool(group_config.get("clear_cache", False)):
                group_cache.clear()
            q4_cache.normalize_q4_format(group_config.get("cache_format"))
            group_cache.configure(
                warm_blocks=int(group_config["warm_blocks"]),
                num_groups=int(group_config["num_groups"]),
                block_count=block_count,
                policy=str(group_config.get("cache_policy", "cpu")),
                shape=(int(residual.shape[0]), int(residual.shape[1])),
                feature_mode=group_config.get("feature_mode", "q4"),
                signature_max_tokens=int(
                    group_config.get(
                        "signature_max_tokens",
                        q4_cache.DEFAULT_SIGNATURE_MAX_TOKENS,
                    )
                ),
                signature_hidden_samples=int(
                    group_config.get(
                        "signature_hidden_samples",
                        q4_cache.DEFAULT_SIGNATURE_HIDDEN_SAMPLES,
                    )
                ),
            )

        normalized_stat_ranges = _validated_stat_ranges(
            int(residual.shape[0]), stat_ranges
        )
        block_stats: dict[str, Any] | None = None
        stat_seconds = 0.0
        if collect_block_stats:
            stat_started = time.perf_counter()
            block_stats = {
                "input": tensor_scalar_stats(residual, normalized_stat_ranges),
                "input_change": self._sampled_input_change(residual),
                "ranges": [list(item) for item in normalized_stat_ranges],
            }
            stat_seconds += time.perf_counter() - stat_started

        torch.cuda.reset_peak_memory_stats(self.device)
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        stages = _CudaStageRecorder(stage_profile)
        # Match the installed Turbo node's curve-mode lookup: identify the
        # nearest base curve row from the interpolated 8-D t embedding, then use
        # the corresponding precomputed silu(t_emb) row.
        stages.begin("conditioning_lookup")
        indices = torch.cdist(t_emb.float(), self.adaln_table.float()).argmin(dim=1)
        silu_temb = self.egrid[indices]
        stages.end("conditioning_lookup")
        stages.begin("modulation_row_setup")
        if self.fused_fp32_ops:
            mod_rows, modulation_rows_cached = self._get_modulation_rows(
                residual.shape[0], segments
            )
        else:
            mod_rows = None
            modulation_rows_cached = False
        stages.end("modulation_row_setup")
        attention_collective_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        mlp_collective_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        oracle_attention_collective_events: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        oracle_mlp_collective_events: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        decision_collective_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        block_ms = []
        blocks_executed = 0
        skipped_blocks = 0

        def run_range(
            value: torch.Tensor,
            first: int,
            last: int,
            *,
            oracle: bool = False,
        ) -> torch.Tensor:
            nonlocal blocks_executed
            attention_events = (
                oracle_attention_collective_events
                if oracle
                else attention_collective_events
            )
            mlp_events = (
                oracle_mlp_collective_events if oracle else mlp_collective_events
            )
            for index in range(first, last):
                block = self.blocks[index]
                block_start = block_end = None
                if profile and not oracle:
                    block_start = torch.cuda.Event(enable_timing=True)
                    block_end = torch.cuda.Event(enable_timing=True)
                    block_start.record()
                value = self._forward_block(
                    value,
                    block,
                    t_emb,
                    silu_temb,
                    segments,
                    mod_rows,
                    rope_freqs,
                    attention_events,
                    mlp_events,
                    stages,
                )
                if profile and not oracle:
                    block_end.record()
                    block_ms.append((index, block_start, block_end))
                if not oracle:
                    blocks_executed += 1
            return value

        group_report: dict[str, Any] | None = None
        if group_enabled:
            assert group_cache is not None
            assert group_config is not None
            warm_blocks = int(group_config["warm_blocks"])
            metric = str(group_config["metric"]).lower()
            threshold = float(group_config["threshold"])
            max_cache = int(group_config["max_cache"])
            reference_mode = str(group_config["reference_mode"]).lower()
            calibration_mode = str(
                group_config.get("calibration_mode", "off")
            ).lower()
            condition_metric = str(
                group_config.get("condition_metric", "none")
            ).lower()
            feature_mode = q4_cache.normalize_feature_mode(
                group_config.get("feature_mode", "q4")
            )
            signature_max_tokens = int(
                group_config.get(
                    "signature_max_tokens", q4_cache.DEFAULT_SIGNATURE_MAX_TOKENS
                )
            )
            signature_hidden_samples = int(
                group_config.get(
                    "signature_hidden_samples",
                    q4_cache.DEFAULT_SIGNATURE_HIDDEN_SAMPLES,
                )
            )
            signature_aggregation = str(
                group_config.get("signature_aggregation", "weighted")
            ).lower()
            calibration_collect = calibration_mode == "collect"
            cache_chunk_rows = int(
                group_config.get(
                    "cache_chunk_rows", q4_cache.DEFAULT_CACHE_CHUNK_ROWS
                )
            )
            benchmark_ground_truth = bool(
                group_config.get("benchmark_ground_truth", False)
            )
            collect_group_stats = bool(
                collect_block_stats or benchmark_ground_truth or calibration_collect
            )
            if metric not in {"relative_l1", "relative_l2", "cosine"}:
                raise ValueError(f"unsupported H3 group-cache metric {metric!r}")
            if threshold < 0.0 or max_cache < 0:
                raise ValueError("H3 group-cache threshold/max_cache must be non-negative")
            if reference_mode not in {"last_full", "previous_step"}:
                raise ValueError(
                    f"unsupported H3 group-cache reference mode {reference_mode!r}"
                )
            if calibration_mode not in {"off", "collect"}:
                raise ValueError(
                    f"unsupported H3 group-cache calibration mode {calibration_mode!r}"
                )
            if condition_metric not in {"none", "gates", "all_adaln"}:
                raise ValueError(
                    "unsupported H3 group-cache condition metric "
                    f"{condition_metric!r}"
                )
            if signature_aggregation not in {"weighted", "max_segment"}:
                raise ValueError(
                    "unsupported H3 group-cache signature aggregation "
                    f"{signature_aggregation!r}"
                )
            if signature_max_tokens <= 0 or signature_hidden_samples <= 0:
                raise ValueError("H3 group-cache signature sizes must be positive")
            if calibration_mode == "off" and condition_metric != "none":
                raise ValueError(
                    "H3 group-cache condition_metric requires calibration_mode=collect"
                )
            condition_segments = tuple(
                tuple(int(value) for value in segment) for segment in segments
            )
            condition_hidden_indices = (
                group_calibration.sampled_hidden_indices(
                    HIDDEN, device=self.device
                )
                if calibration_collect and condition_metric != "none"
                else None
            )

            residual = run_range(residual, 0, warm_blocks)
            group_rows: list[dict[str, Any]] = []
            for entry in group_cache.entries:
                current_condition_signature = None
                condition_signature_build = None
                condition_report: dict[str, Any] = {
                    "available": False,
                    "metric": condition_metric,
                    "reference": "last_full",
                }
                condition_error = None
                if condition_hidden_indices is not None:
                    current_condition_signature, condition_signature_build = (
                        self._group_condition_signature(
                            entry.start_block,
                            entry.end_block,
                            t_emb,
                            silu_temb,
                            condition_hidden_indices,
                        )
                    )
                    if (
                        entry.condition_signature is not None
                        and entry.condition_segments is not None
                    ):
                        condition_report = group_calibration.signature_difference(
                            current_condition_signature,
                            entry.condition_signature,
                            current_segments=condition_segments,
                            reference_segments=entry.condition_segments,
                            epsilon=float(group_config.get("epsilon", 1e-6)),
                        )
                        condition_report["reference"] = "last_full"
                        condition_error = group_calibration.selected_condition_error(
                            condition_report, condition_metric
                        )

                current_q4: q4_cache.Q4Tensor | None = None
                current_input_signature: torch.Tensor | None = None
                current_input_signature_metadata: dict[str, Any] | None = None
                if feature_mode == "q4":
                    current_q4 = q4_cache.quantize_q4_0(
                        residual,
                        policy=group_cache.policy,
                        chunk_rows=cache_chunk_rows,
                        measure=collect_group_stats,
                    )
                else:
                    current_input_signature, current_input_signature_metadata = (
                        q4_cache.deterministic_input_signature(
                            residual,
                            max_tokens=signature_max_tokens,
                            hidden_samples=signature_hidden_samples,
                            ranges=segments,
                        )
                    )
                feature_error = None
                feature_report: dict[str, Any] = {
                    "available": False,
                    "metric": metric,
                    "metric_domain": (
                        "q4_dequantized_pair"
                        if feature_mode == "q4"
                        else "fp32_bounded_signature"
                    ),
                }
                if entry.ready:
                    if feature_mode == "q4":
                        assert current_q4 is not None
                        assert entry.previous_input is not None
                        feature_error, measured_feature = q4_cache.relative_difference(
                            current_q4,
                            entry.previous_input,
                            metric=metric,
                            device=self.device,
                            chunk_rows=cache_chunk_rows,
                            epsilon=float(group_config.get("epsilon", 1e-6)),
                            measure=collect_group_stats,
                        )
                    else:
                        assert current_input_signature is not None
                        assert current_input_signature_metadata is not None
                        assert entry.input_signature is not None
                        assert entry.input_signature_metadata is not None
                        feature_error, measured_feature = q4_cache.signature_difference(
                            current_input_signature,
                            entry.input_signature,
                            metric=metric,
                            epsilon=float(group_config.get("epsilon", 1e-6)),
                            current_metadata=current_input_signature_metadata,
                            reference_metadata=entry.input_signature_metadata,
                            aggregation=signature_aggregation,
                        )
                    feature_report = {"available": True, **measured_feature}

                local_can_cache = bool(
                    entry.ready
                    and feature_error is not None
                    and math.isfinite(feature_error)
                    and feature_error < threshold
                    and (max_cache == 0 or entry.cache_count < max_cache)
                )
                packet = torch.tensor(
                    [
                        -1.0 if feature_error is None else float(feature_error),
                        1.0 if local_can_cache else 0.0,
                        float(entry.cache_count),
                        1.0 if entry.ready else 0.0,
                        -1.0 if condition_error is None else float(condition_error),
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                decision_start = torch.cuda.Event(enable_timing=True)
                decision_end = torch.cuda.Event(enable_timing=True)
                decision_start.record()
                dist.broadcast(packet, src=0)
                authoritative_error = float(packet[0].item())
                authoritative_cache = bool(round(float(packet[1].item())))
                authoritative_count = int(round(float(packet[2].item())))
                authoritative_ready = bool(round(float(packet[3].item())))
                authoritative_condition_error = float(packet[4].item())
                error_mismatch = (
                    0.0
                    if feature_error is None and authoritative_error < 0.0
                    else abs(float(feature_error or 0.0) - authoritative_error)
                )
                state_mismatch = bool(
                    local_can_cache != authoritative_cache
                    or entry.cache_count != authoritative_count
                    or entry.ready != authoritative_ready
                    or ((condition_error is None) != (authoritative_condition_error < 0.0))
                )
                validation = torch.tensor(
                    [
                        error_mismatch,
                        0.0
                        if condition_error is None
                        and authoritative_condition_error < 0.0
                        else abs(
                            float(condition_error or 0.0)
                            - authoritative_condition_error
                        ),
                        1.0 if state_mismatch else 0.0,
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                dist.all_reduce(validation, op=dist.ReduceOp.MAX)
                decision_end.record()
                decision_collective_events.append((decision_start, decision_end))
                max_feature_rank_delta = float(validation[0].item())
                max_condition_rank_delta = float(validation[1].item())
                if (
                    float(validation[2].item()) != 0.0
                    or max_feature_rank_delta > 1e-5
                    or max_condition_rank_delta > 1e-5
                ):
                    raise RuntimeError(
                        "H3 group-cache ranks disagreed before a block range: "
                        f"group={entry.group_id}, state_mismatch={validation[2].item()}, "
                        f"feature_delta={max_feature_rank_delta}, "
                        f"condition_delta={max_condition_rank_delta}"
                    )

                decision = "cache" if authoritative_cache else "full"
                row: dict[str, Any] = {
                    "group_id": entry.group_id,
                    "start_block": entry.start_block,
                    "end_block": entry.end_block,
                    "blocks": entry.end_block - entry.start_block,
                    "feature_error": (
                        None if authoritative_error < 0.0 else authoritative_error
                    ),
                    "feature": feature_report,
                    "threshold": threshold,
                    "decision": decision,
                    "cache_count_before": authoritative_count,
                    "max_cache": max_cache,
                    "reference_mode": reference_mode,
                    "feature_mode": feature_mode,
                    "signature_aggregation": signature_aggregation,
                    "rank_feature_error_max_abs_diff": max_feature_rank_delta,
                    "condition_error": (
                        None
                        if authoritative_condition_error < 0.0
                        else authoritative_condition_error
                    ),
                    "rank_condition_error_max_abs_diff": max_condition_rank_delta,
                    "current_input_quantize": (
                        None
                        if current_q4 is None
                        else dict(current_q4.quantize_report)
                    ),
                    "current_input_signature": (
                        None
                        if current_input_signature_metadata is None
                        else dict(current_input_signature_metadata)
                    ),
                }
                if calibration_collect:
                    row["calibration"] = {
                        "mode": calibration_mode,
                        "decision_policy": "feature_only",
                        "condition_metric": condition_metric,
                        "condition": dict(condition_report),
                        "condition_error": condition_error,
                        "condition_signature_build": condition_signature_build,
                        "condition_rank_max_abs_diff": None,
                        "input_feature_error": row["feature_error"],
                        "input_metric": metric,
                        "input_feature_mode": feature_mode,
                        "signature_aggregation": signature_aggregation,
                        "residual_q4_floor_relative_l2": entry.residual_q_floor,
                        "cache_age": authoritative_count,
                        "sigma_delta": group_config.get("sigma_delta"),
                        "target": None,
                    }

                if authoritative_cache:
                    assert entry.residual is not None
                    true_output = None
                    oracle_start = oracle_end = None
                    if benchmark_ground_truth:
                        oracle_start = torch.cuda.Event(enable_timing=True)
                        oracle_end = torch.cuda.Event(enable_timing=True)
                        oracle_start.record()
                        true_output = run_range(
                            residual.clone(),
                            entry.start_block,
                            entry.end_block,
                            oracle=True,
                        )
                        oracle_end.record()
                    row["cache_operation"] = q4_cache.add_q4_to_(
                        residual,
                        entry.residual,
                        chunk_rows=cache_chunk_rows,
                        measure=collect_group_stats,
                    )
                    if true_output is not None:
                        oracle_end.synchronize()
                        row["oracle_full_ms"] = float(
                            oracle_start.elapsed_time(oracle_end)
                        )
                        row["ground_truth"] = (
                            q4_cache.cached_group_ground_truth_error(
                                residual,
                                true_output,
                                entry.residual,
                                chunk_rows=cache_chunk_rows,
                                epsilon=float(group_config.get("epsilon", 1e-6)),
                            )
                        )
                        del true_output
                    if calibration_collect:
                        calibration = row["calibration"]
                        calibration["target"] = (
                            None
                            if "ground_truth" not in row
                            else {
                                "kind": "cache_oracle",
                                "output_relative_l2": row["ground_truth"].get(
                                    "output_relative_l2"
                                ),
                                "residual_relative_l2": row["ground_truth"].get(
                                    "residual_relative_l2"
                                ),
                            }
                        )
                    if reference_mode == "previous_step":
                        if feature_mode == "q4":
                            assert current_q4 is not None
                            previous_input = entry.previous_input
                            entry.previous_input = current_q4
                            del previous_input
                        else:
                            assert current_input_signature is not None
                            previous_signature = entry.input_signature
                            entry.input_signature = current_input_signature
                            entry.input_signature_metadata = (
                                current_input_signature_metadata
                            )
                            del previous_signature
                        current_q4 = None
                        current_input_signature = None
                    else:
                        current_q4 = None
                        current_input_signature = None
                    del current_condition_signature
                    entry.cache_count += 1
                    entry.hit_count += 1
                    skipped_blocks += entry.end_block - entry.start_block
                else:
                    # The decision no longer needs the previous input.  Drop
                    # stale Q4 buffers before the new FULL group is executed,
                    # otherwise a 1 MP refresh temporarily holds old input,
                    # old residual, current input and new residual together.
                    old_input = entry.previous_input
                    old_input_signature = entry.input_signature
                    old_input_signature_metadata = entry.input_signature_metadata
                    old_residual = entry.residual
                    old_condition_signature = entry.condition_signature
                    old_condition_segments = entry.condition_segments
                    old_residual_q_floor = entry.residual_q_floor
                    entry.previous_input = None
                    entry.input_signature = None
                    entry.input_signature_metadata = None
                    entry.residual = None
                    entry.condition_signature = None
                    entry.condition_segments = None
                    entry.residual_q_floor = None
                    del old_input
                    if not (benchmark_ground_truth or calibration_collect):
                        del old_residual
                        old_residual = None
                    del (
                        old_input_signature,
                        old_input_signature_metadata,
                        old_condition_signature,
                        old_condition_segments,
                    )
                    # Keep the live group input only until its residual has
                    # been formed.  In signature mode no tensor-sized feature
                    # reference is retained; in Q4 mode ``current_q4`` is the
                    # historical persistent reference.
                    group_input = residual.clone()
                    if (
                        (benchmark_ground_truth or calibration_collect)
                        and current_q4 is not None
                    ):
                        row["input_quantization_error"] = q4_cache.q4_tensor_error(
                            current_q4,
                            group_input,
                            chunk_rows=cache_chunk_rows,
                        )
                    residual = run_range(
                        residual, entry.start_block, entry.end_block
                    )
                    group_input.neg_().add_(residual)
                    if (
                        (benchmark_ground_truth or calibration_collect)
                        and old_residual is not None
                    ):
                        row["ground_truth"] = (
                            q4_cache.cached_residual_ground_truth_error(
                                old_residual,
                                group_input,
                                residual,
                                chunk_rows=cache_chunk_rows,
                                epsilon=float(group_config.get("epsilon", 1e-6)),
                            )
                        )
                    del old_residual
                    cached_residual = q4_cache.quantize_q4_0(
                        group_input,
                        policy=group_cache.policy,
                        chunk_rows=cache_chunk_rows,
                        measure=collect_group_stats,
                    )
                    residual_q_report = None
                    if benchmark_ground_truth or calibration_collect:
                        residual_q_report = q4_cache.q4_tensor_error(
                            cached_residual,
                            group_input,
                            chunk_rows=cache_chunk_rows,
                        )
                        if benchmark_ground_truth or calibration_collect:
                            row["residual_quantization_error"] = residual_q_report
                        entry.residual_q_floor = float(
                            residual_q_report["relative_l2"]
                        )
                    # All persistent state is standard Q4_0.  The FP32
                    # ``group_input`` allocation is released immediately.
                    if feature_mode == "q4":
                        assert current_q4 is not None
                        entry.previous_input = current_q4
                        entry.input_signature = None
                        entry.input_signature_metadata = None
                    else:
                        assert current_input_signature is not None
                        assert current_input_signature_metadata is not None
                        entry.previous_input = None
                        entry.input_signature = current_input_signature
                        entry.input_signature_metadata = (
                            current_input_signature_metadata
                        )
                    entry.residual = cached_residual
                    if current_condition_signature is not None:
                        entry.condition_signature = current_condition_signature
                        entry.condition_segments = condition_segments
                    else:
                        entry.condition_signature = None
                        entry.condition_segments = None
                    entry.cache_count = 0
                    entry.full_count += 1
                    del group_input
                    row["residual_quantize"] = dict(
                        cached_residual.quantize_report
                    )
                    if calibration_collect:
                        calibration = row["calibration"]
                        calibration["previous_residual_q4_floor_relative_l2"] = (
                            old_residual_q_floor
                        )
                        calibration["next_residual_q4_floor_relative_l2"] = (
                            entry.residual_q_floor
                        )
                        calibration["target"] = (
                            None
                            if "ground_truth" not in row
                            else {
                                "kind": "refresh_reuse_error",
                                "output_relative_l2": row["ground_truth"].get(
                                    "output_relative_l2"
                                ),
                                "residual_relative_l2": row["ground_truth"].get(
                                    "residual_relative_l2"
                                ),
                            }
                        )
                    del current_condition_signature
                row["cache_count_after"] = entry.cache_count
                row["cache_bytes_after"] = entry.bytes
                group_rows.append(row)

            group_report = {
                "enabled": True,
                "format": q4_cache.Q4_FORMAT,
                "warm_blocks": warm_blocks,
                "num_groups": len(group_cache.entries),
                "metric": metric,
                "threshold": threshold,
                "max_cache": max_cache,
                "reference_mode": reference_mode,
                "feature_mode": feature_mode,
                "signature_max_tokens": signature_max_tokens,
                "signature_hidden_samples": signature_hidden_samples,
                "signature_aggregation": signature_aggregation,
                "calibration_mode": calibration_mode,
                "condition_metric": condition_metric,
                "decision_policy": "feature_only",
                "benchmark_ground_truth": benchmark_ground_truth,
                "generation_id": group_config.get("generation_id"),
                "step": group_config.get("step"),
                "sigma_raw": group_config.get("sigma_raw"),
                "sigma_delta": group_config.get("sigma_delta"),
                "groups": group_rows,
                "oracle_full_ms": sum(
                    float(row.get("oracle_full_ms", 0.0)) for row in group_rows
                ),
                "executed_blocks": blocks_executed,
                "skipped_blocks": skipped_blocks,
                "cache": group_cache.summary(),
            }
        else:
            if snapshot_at == start_block:
                self._last_snapshot = residual.clone()
            for index in range(start_block, end_block):
                if snapshot_at == index and self._last_snapshot is None:
                    self._last_snapshot = residual.clone()
                residual = run_range(residual, index, index + 1)
        total_end.record()
        total_end.synchronize()

        stages.begin("finite_check")
        # A full ``torch.isfinite(residual)`` mask costs one byte per hidden
        # element.  At 1 MP that is another ~193 MiB/rank, so use the scalar
        # infinity norm: it is NaN/Inf for a non-finite stream and allocates no
        # tensor-sized temporary.
        finite_probe = float(
            torch.linalg.vector_norm(
                residual.reshape(-1), ord=float("inf")
            ).item()
        )
        finite = bool(math.isfinite(finite_probe))
        stages.end("finite_check")
        if block_stats is not None:
            stat_started = time.perf_counter()
            block_stats["range_output_before_cache_add"] = tensor_scalar_stats(
                residual, normalized_stat_ranges
            )
            stat_seconds += time.perf_counter() - stat_started
            block_stats["collection_ms_excluding_tail_residual"] = (
                stat_seconds * 1000.0
            )
        if stage_profile:
            # The finite reduction's .item() synchronizes its work, but the
            # trailing event itself is still recorded asynchronously.
            torch.cuda.synchronize(self.device)
        metrics = {
            "rank": self.rank,
            "sequence": residual.shape[0],
            "total_ms": total_start.elapsed_time(total_end),
            "collective_ms": sum(
                start.elapsed_time(end)
                for start, end in (*attention_collective_events, *mlp_collective_events)
            ),
            "attention_all_reduce_ms": sum(
                start.elapsed_time(end)
                for start, end in attention_collective_events
            ),
            "mlp_all_reduce_ms": sum(
                start.elapsed_time(end)
                for start, end in mlp_collective_events
            ),
            "group_decision_collective_ms": sum(
                start.elapsed_time(end) for start, end in decision_collective_events
            ),
            "oracle_collective_ms": sum(
                start.elapsed_time(end)
                for start, end in (
                    *oracle_attention_collective_events,
                    *oracle_mlp_collective_events,
                )
            ),
            "mlp_reduce_rows": self.mlp_reduce_rows,
            "profile_enabled": bool(profile),
            "block_ms": [
                {"block": index, "milliseconds": start.elapsed_time(end)}
                for index, start, end in block_ms
            ],
            "block_start": start_block,
            "block_end": end_block,
            "blocks_executed": blocks_executed,
            "blocks_skipped": skipped_blocks,
            "snapshot_captured": self._last_snapshot is not None,
            "snapshot_bytes": (
                0
                if self._last_snapshot is None
                else self._last_snapshot.numel() * self._last_snapshot.element_size()
            ),
            "block_stats_enabled": bool(collect_block_stats),
            "block_stats": block_stats,
            "group_cache": group_report,
            "stage_profile_enabled": bool(stage_profile),
            "stage_ms": stages.summary(),
            "modulation_rows_cached": modulation_rows_cached,
            "finite": finite,
            "fused_fp32_ops": self.fused_fp32_ops,
            "fp32_ops_warps": self.fp32_ops_warps if self.fused_fp32_ops else None,
            "compact_qkv": self.compact_qkv,
            "compact_qkv_mode": self.compact_qkv_mode,
            "compact_qkv_min_sequence": self.compact_qkv_min_sequence,
            "compact_qkv_applied": bool(
                self.compact_qkv
                and residual.shape[0] >= self.compact_qkv_min_sequence
            ),
            "allocated_mib": torch.cuda.memory_allocated(self.device) / MIB,
            "reserved_mib": torch.cuda.memory_reserved(self.device) / MIB,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / MIB,
            "process_memory": process_memory_stats(),
        }
        if group_report is not None:
            metrics["production_estimate_ms"] = max(
                0.0,
                float(metrics["total_ms"]) - float(group_report["oracle_full_ms"]),
            )
        if not finite:
            raise RuntimeError(f"rank {self.rank} H3 TP backbone produced NaN/Inf")
        return residual, metrics


def broadcast_inputs_rank0(
    residual: torch.Tensor,
    t_emb: torch.Tensor,
    rope_freqs: torch.Tensor,
) -> None:
    dist.broadcast(residual, src=0)
    dist.broadcast(t_emb, src=0)
    dist.broadcast(rope_freqs, src=0)


def allocate_and_receive_rank1(
    residual_shape: Sequence[int],
    t_emb_shape: Sequence[int],
    rope_shape: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = torch.empty(tuple(residual_shape), dtype=torch.float32, device=device)
    t_emb = torch.empty(tuple(t_emb_shape), dtype=torch.float32, device=device)
    rope = torch.empty(tuple(rope_shape), dtype=torch.float16, device=device)
    dist.broadcast(residual, src=0)
    dist.broadcast(t_emb, src=0)
    dist.broadcast(rope, src=0)
    return residual, t_emb, rope


__all__ = [
    "DEFAULT_EGRID",
    "DEFAULT_LORA",
    "DEFAULT_MODEL",
    "H3TPResidualCache",
    "H3TPBackbone",
    "HIDDEN",
    "LAYERS",
    "allocate_and_receive_rank1",
    "broadcast_inputs_rank0",
]
