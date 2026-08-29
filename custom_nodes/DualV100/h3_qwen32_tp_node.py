"""Opt-in ComfyUI nodes for the MiniMax H3 Qwen32 Q2 TP route.

The production 4B Q4 + ClipProj path is intentionally untouched by this
module.  A workflow first creates a shared ``H3_TP_RUNTIME`` handle, then
passes it to :class:`Qwen32BQ2TPCLIPLoader`.  The resulting object implements
the small CLIP-like surface consumed by the stock MiniMax H3 conditioning
nodes (``tokenize`` and ``encode_from_tokens_scheduled``).  Token embedding,
vision, MRoPE and DeepStack assembly happen in this process; the 50 language
layers are owned by the shared two-rank runtime.

No CUDA context, process group, GGUF payload, or tokenizer is created at
module import time.  The route is fail-closed unless ``H3_QWEN32_Q2_TP`` is
explicitly enabled.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

try:
    import folder_paths
except ImportError:  # pragma: no cover - allows contract tests without ComfyUI
    folder_paths = None  # type: ignore[assignment]

try:
    from .h3_qwen32_q2_tp import (
        DEFAULT_STAGING_MIB,
        Qwen32Q2DiskReader,
        Qwen32Q2SelectedEmbedding,
        dequantize_ggml,
        inspect_gguf,
    )
    from .h3_tp_backbone import DEFAULT_EGRID, normalize_weight_format
    from .h3_tp_runtime import RuntimeConfig, get_runtime
except ImportError:  # pragma: no cover - standalone source-tree imports
    from h3_qwen32_q2_tp import (  # type: ignore[no-redef]
        DEFAULT_STAGING_MIB,
        Qwen32Q2DiskReader,
        Qwen32Q2SelectedEmbedding,
        dequantize_ggml,
        inspect_gguf,
    )
    from h3_tp_backbone import DEFAULT_EGRID, normalize_weight_format  # type: ignore[no-redef]
    from h3_tp_runtime import RuntimeConfig, get_runtime  # type: ignore[no-redef]


_TRUE = {"1", "true", "yes", "on", "enable", "enabled"}
_QWEN_MODEL_ENV = "H3_QWEN32_Q2_MODEL"
_QWEN_MODE_ENV = "H3_QWEN32_Q2_MODE"
_QWEN_MP_ENABLE_ENV = "H3_QWEN32_Q2_MP"
_DEFAULT_QWEN_MODEL = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
_DEFAULT_RESULTS_DIR = "/home/regen/code/minimax_v100/results/h3_tp_e2e"
_QWEN_HIDDEN = 5120
_PAD_TOKEN = 151643
_VISION_START = 151652
_VISION_END = 151653


def _enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE


def _resolve_qwen_mode(value: str | None = None) -> str:
    requested = (
        os.environ.get(_QWEN_MODE_ENV, "tp") if value is None else str(value)
    )
    normalized = requested.strip().lower().replace("-", "_")
    if normalized in {"tp", "output_row_tp", "outputrow_tp"}:
        return "tp"
    if normalized in {"mp", "layer_mp", "layer_parallel", "layerpipeline"}:
        return "mp"
    raise ValueError(f"{_QWEN_MODE_ENV} must be tp or mp, got {requested!r}")


def _require_opt_in(mode: str | None = None) -> str:
    selected = _resolve_qwen_mode(mode)
    if selected == "mp":
        if not _enabled(_QWEN_MP_ENABLE_ENV):
            raise RuntimeError(
                "Qwen32 Q2 layer-MP is disabled. Set H3_QWEN32_Q2_MODE=mp and "
                "H3_QWEN32_Q2_MP=1 before using the MP loader."
            )
        return selected
    if not _enabled("H3_QWEN32_Q2_TP"):
        raise RuntimeError(
            "Qwen32 Q2 TP is disabled. Set H3_QWEN32_Q2_TP=1 before using "
            "the experimental Qwen32 nodes; the default 4B Q4 route is unchanged."
        )
    return selected


def _filename_list(category: str) -> list[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_filename_list(category))
    except (AttributeError, KeyError):
        return []


def _resolve_path(category: str, name: str, *, direct_ok: bool = True) -> str:
    candidate = Path(os.fspath(name))
    if direct_ok and candidate.is_file():
        return str(candidate.resolve())
    if folder_paths is not None:
        try:
            resolved = folder_paths.get_full_path(category, name)
        except (AttributeError, KeyError):
            resolved = None
        if resolved is not None:
            return str(Path(resolved).resolve())
    raise FileNotFoundError(f"{category} model not found: {name}")


def _resolve_qwen_name(name: str | None) -> str:
    requested = (name or os.environ.get(_QWEN_MODEL_ENV) or _DEFAULT_QWEN_MODEL).strip()
    path = _resolve_path("text_encoders", requested)
    if not path.lower().endswith(".gguf"):
        raise ValueError(f"Qwen32 Q2 loader requires a GGUF file, got {path}")
    return path


def _resolve_unet_name(name: str) -> str:
    for category in ("unet_gguf", "unet", "diffusion_models"):
        try:
            return _resolve_path(category, name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"H3 diffusion model not found: {name}")


def _resolve_lora_name(name: str) -> str:
    return _resolve_path("loras", name)


def _runtime_object(value: Any) -> Any:
    """Unwrap a handle while accepting the runtime itself for compatibility."""

    runtime = getattr(value, "runtime", None)
    return runtime if runtime is not None else value


class H3TPRuntimeHandle:
    """Workflow value carrying one shared :class:`H3TPRuntime` instance.

    The wrapper keeps Qwen model selection and the cached CLIP facade attached
    to the same graph value.  ``__getattr__`` deliberately delegates runtime
    methods so future runtime-aware model nodes can accept either this handle
    or the raw runtime without a second singleton/configuration.
    """

    TYPE = "H3_TP_RUNTIME"

    def __init__(
        self,
        runtime: Any,
        *,
        qwen_model_path: str,
        qwen_staging_mib: int,
    ) -> None:
        self.runtime = runtime
        self.qwen_model_path = str(Path(qwen_model_path).resolve())
        self.qwen_staging_mib = int(qwen_staging_mib)
        self._qwen_clip: _Qwen32TPClip | None = None
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def configure_qwen(
        self,
        *,
        staging_mib: int,
        residency: str = "evict",
        keep_layers: int | Sequence[int] = 0,
        cache_dequantized: bool = False,
        mode: str | None = None,
    ) -> Any:
        method = getattr(self.runtime, "configure_qwen", None)
        if method is None:
            raise RuntimeError(
                "the installed H3TPRuntime does not expose configure_qwen(); "
                "update the shared runtime before enabling the Qwen32 node"
            )
        options: dict[str, Any] = {
            "staging_mib": int(staging_mib),
            "residency": str(residency),
            "keep_layers": keep_layers,
            "cache_dequantized": bool(cache_dequantized),
        }
        if mode is not None:
            options["mode"] = str(mode)
        return method(self.qwen_model_path, **options)

    def qwen_clip(self) -> _Qwen32TPClip | None:
        with self._lock:
            return self._qwen_clip

    def set_qwen_clip(self, clip: _Qwen32TPClip) -> None:
        with self._lock:
            self._qwen_clip = clip


_ABSENT = object()


def _replace_module_value(root: nn.Module, name: str, value: torch.Tensor) -> None:
    """Replace a meta parameter/buffer without invoking a whole state dict."""

    parts = name.split(".")
    parent: Any = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    old = getattr(parent, leaf, _ABSENT)
    if isinstance(old, nn.Parameter):
        setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
    elif torch.is_tensor(old):
        setattr(parent, leaf, value)
    elif old is None and leaf in {"weight", "bias"} and isinstance(parent, nn.Module):
        # With ComfyUI DynamicVRAM (``aimdo``) enabled, ``disable_weight_init``
        # Linear layers skip ``nn.Linear.__init__`` and hold ``weight``/``bias``
        # as unregistered ``None`` attributes until a state dict arrives.
        # Registering the Parameter here is what that lazy loader would do.
        setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
    else:
        raise AttributeError(f"vision module has no parameter/buffer {name!r}")


def _expected_vision_tensors(vision: nn.Module) -> dict[str, tuple[int, ...]]:
    """Map every vision tensor that must be filled to its logical shape.

    ``named_parameters()`` alone is not enough.  Under DynamicVRAM the lazy
    ``disable_weight_init.Linear`` never registers its ``weight``/``bias``, so
    every projection is invisible to ``named_parameters()``; enumerating only
    that would load the norms, leave each Linear at ``None``, and fail inside
    ``F.linear`` on the first vision request instead of at load time.
    """

    expected = {
        name: tuple(parameter.shape) for name, parameter in vision.named_parameters()
    }
    for prefix, module in vision.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        base = f"{prefix}." if prefix else ""
        if getattr(module, "weight", None) is None:
            expected[f"{base}weight"] = (
                int(module.out_features),
                int(module.in_features),
            )
        if getattr(module, "bias", None) is None and getattr(
            module, "comfy_need_lazy_init_bias", False
        ):
            expected[f"{base}bias"] = (int(module.out_features),)
    return expected


def _vision_model_config() -> dict[str, Any]:
    return {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "depth": 27,
        "deepstack_visual_indexes": [8, 16, 24],
        "num_heads": 16,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "spatial_merge_size": 2,
        "num_position_embeddings": 2304,
        "out_hidden_size": _QWEN_HIDDEN,
    }


def _load_vision_frontend(
    layout: Any,
    reader: Qwen32Q2DiskReader,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Construct stock Qwen3-VL vision code and fill it from GGUF slices."""

    try:
        import comfy.ops
        from comfy.text_encoders.qwen3vl import Qwen3VLVisionModel
    except ImportError as exc:  # pragma: no cover - only exercised outside ComfyUI
        raise RuntimeError("Qwen32 vision loading requires ComfyUI text encoders") from exc

    # Meta construction leaves no zero-filled 1.1 GiB placeholder on rank0.
    vision = Qwen3VLVisionModel(
        _vision_model_config(),
        device="meta",
        dtype=dtype,
        ops=comfy.ops.disable_weight_init,
    )
    specs = {
        spec.name.removeprefix("visual."): spec
        for spec in layout.tensors
        if spec.name.startswith("visual.")
    }
    expected = _expected_vision_tensors(vision)
    missing = sorted(set(expected).difference(specs))
    if missing:
        raise ValueError(f"Qwen32 GGUF is missing vision tensors: {missing[:8]}")

    loaded = 0
    for name, module_shape in expected.items():
        spec = specs[name]
        shape = tuple(spec.original_shape or spec.shape)
        if module_shape != shape:
            raise ValueError(
                f"vision tensor {name} shape mismatch: module={module_shape} "
                f"GGUF={shape}"
            )
        raw = reader.read_tensor(spec, device=device)
        try:
            value = dequantize_ggml(raw, spec.qtype, shape, dtype=dtype)
            _replace_module_value(vision, name, value.contiguous())
        finally:
            del raw
        loaded += 1

    # ``inv_freq`` is a non-persistent CPU buffer created by the stock vision
    # rotary helper.  Move it once so vision inputs never trigger an implicit
    # cross-device copy during the first request.
    rotary = getattr(vision, "rotary_pos_emb", None)
    if rotary is not None and hasattr(rotary, "inv_freq"):
        rotary.inv_freq = rotary.inv_freq.to(device=device)
    vision.eval()
    for parameter in vision.parameters():
        parameter.requires_grad_(False)
    logging.info("[H3 Qwen32] loaded %d vision tensors on %s", loaded, device)
    return vision


def _plain_token(value: Any) -> int | None:
    if isinstance(value, int):
        return int(value)
    # bool is Integral but should never be emitted by the tokenizer; keep the
    # explicit branch above narrow so malformed custom tokens fail clearly.
    return None


def _token_parts(item: Any) -> tuple[Any, float]:
    if not isinstance(item, (tuple, list)) or len(item) < 2:
        raise TypeError(f"invalid Qwen token item: {item!r}")
    return item[0], float(item[1])


def _make_attention_mask(
    binary: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return the stock MiniMax Qwen causal mask.

    ``MiniMaxH3ClipModel`` sets ``enable_attention_masks=False``.  Its binary
    pad bookkeeping is retained for token accounting, but is *not* passed to
    Qwen; ``Llama2_.forward`` therefore uses only its causal mask.  Passing a
    custom pad mask here changes the empty-prompt and weighted-token baseline.
    """

    if binary.ndim != 2:
        raise ValueError(f"attention mask must be [B,S], got {tuple(binary.shape)}")
    sequence = int(binary.shape[1])
    if sequence <= 1:
        return None
    return torch.empty(
        sequence,
        sequence,
        dtype=dtype,
        device=binary.device,
    ).fill_(torch.finfo(dtype).min / 4).triu_(1)


def _vision_tags(sequence: int, embeds_info: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    tags = torch.ones(sequence, dtype=torch.long)
    for embed in embeds_info:
        if embed.get("type") != "image":
            continue
        start = int(embed["index"])
        size = int(embed["size"])
        # H3 adaLN tags include vision_start and vision_end around the expanded
        # visual embedding, exactly as stock MiniMaxQwen3VL does.
        tags[max(0, start - 1) : min(sequence, start + size + 1)] = 0
    return tags


def _hash_value(digest: Any, value: Any) -> None:
    """Hash tokenizer payloads without retaining another image-sized object."""

    if torch.is_tensor(value):
        tensor = value.detach().contiguous().to("cpu")
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        # NumPy exposes the tensor storage through the buffer protocol, so the
        # hash is streamed from the existing CPU tensor instead of creating a
        # second ``bytes`` copy of a reference image/video.
        digest.update(memoryview(tensor.numpy()).cast("B"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence\0")
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _hash_value(digest, item)
        return
    digest.update(type(value).__name__.encode("ascii", errors="backslashreplace"))
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8", errors="backslashreplace"))


def _token_digest(qwen_path: str, tokens: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(Path(qwen_path).resolve()).encode("utf-8"))
    _hash_value(digest, tokens)
    return digest.hexdigest()


def _clone_cache_value(value: Any) -> Any:
    """Clone the tiny conditioning tree without retaining Qwen activations."""

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cache_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache_value(item) for item in value]
    return value


def _prepare_async_vae_for_qwen() -> bool:
    try:
        from .h3_async_vae_bridge import prepare_active_vae_for_qwen
    except ImportError:
        return False
    return bool(prepare_active_vae_for_qwen())


class _Qwen32TPClip:
    """CLIP-like facade backed by a shared H3TPRuntime."""

    TYPE = "CLIP"

    def __init__(
        self,
        runtime_handle: H3TPRuntimeHandle,
        *,
        qwen_path: str,
        staging_mib: int,
        residency: str,
        keep_layers: int | Sequence[int],
        cache_dequantized: bool,
    ) -> None:
        self.runtime_handle = runtime_handle
        self.runtime = _runtime_object(runtime_handle)
        self.qwen_path = str(Path(qwen_path).resolve())
        self.staging_mib = int(staging_mib)
        self.residency = str(residency)
        self.keep_layers = keep_layers
        self.cache_dequantized = bool(cache_dequantized)
        self.execution_device = torch.device(
            os.environ.get("H3_QWEN32_DEVICE", "cuda:0")
        )
        # Stock MiniMaxH3ClipModel requests FP32 embedding/language/vision
        # execution even when the GGUF text encoder is loaded with FP16 model
        # metadata.  Q2 decode precision and RMS/attention rounding must
        # match that reference path; using FP16 here causes 50-layer drift.
        self.compute_dtype = torch.float32
        self._closed = False
        self._encode_lock = threading.RLock()
        self._layout = inspect_gguf(self.qwen_path)
        self._reader = Qwen32Q2DiskReader(
            self.qwen_path,
            staging_mib=self.staging_mib,
        )
        self._embedding = Qwen32Q2SelectedEmbedding(
            self._layout,
            device=self.execution_device,
            dtype=self.compute_dtype,
            reader=self._reader,
        )

        try:
            from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
            from comfy.text_encoders.llama import (
                Qwen3VL_32BConfig,
                precompute_freqs_cis,
            )
            import comfy.text_encoders.qwen_vl as qwen_vl
        except ImportError as exc:  # pragma: no cover - ComfyUI is a runtime dep
            self.close()
            raise RuntimeError("Qwen32 CLIP facade requires ComfyUI MiniMax encoders") from exc

        self.tokenizer = MiniMaxH3Tokenizer()
        self._config = Qwen3VL_32BConfig()
        self._precompute_freqs_cis = precompute_freqs_cis
        self._mrope_position_ids = qwen_vl.qwen2vl_mrope_position_ids
        # Text-only prompts never pay the ~1.1 GiB FP16 vision expansion.  An
        # image request loads it just for visual assembly and releases it before
        # rank0 begins loading language-layer shards.
        self._vision: nn.Module | None = None
        self._options: dict[str, Any] = {}
        self.cond_stage_model = self
        self.patcher = self
        self._forward_attempted = False
        self._cache_entries = max(
            0, int(os.environ.get("H3_QWEN32_COND_CACHE_ENTRIES", "4"))
        )
        self._cache: OrderedDict[str, tuple[torch.Tensor, dict[str, Any]]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def __repr__(self) -> str:
        return (
            f"_Qwen32TPClip(path={self.qwen_path!r}, residency={self.residency!r}, "
            f"device={str(self.execution_device)!r})"
        )

    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs: Any) -> dict[str, Any]:
        self._check_open()
        return self.tokenizer.tokenize_with_weights(
            text,
            return_word_ids=return_word_ids,
            **kwargs,
        )

    def _process_one(self, sequence: Sequence[Any]) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Assemble one tokenizer section and invoke the shared Qwen runtime."""

        attention: list[int] = []
        token_ids: list[int] = []
        token_weights: list[float] = []
        other_embeds: list[tuple[int, Any]] = []
        eos = False
        left_pad = False
        end_token = self._options.get("end_token")
        cmp_token = _PAD_TOKEN if end_token is None else int(end_token)
        for index, raw in enumerate(sequence):
            value, weight = _token_parts(raw)
            token = _plain_token(value)
            if token is None:
                other_embeds.append((index, value))
                continue
            if index == 0 and token == _PAD_TOKEN:
                left_pad = True
            if eos or (left_pad and token == _PAD_TOKEN):
                attention.append(0)
            else:
                attention.append(1)
                left_pad = False
            token_ids.append(token)
            token_weights.append(weight)
            # Match SDClipModel.process_tokens: MiniMax H3 has no explicit
            # EOS, so the first non-left-padded PAD marks the end and is
            # masked itself.  Custom callers may still provide an EOS option.
            if not eos and token == cmp_token and not left_pad:
                if end_token is None:
                    attention[-1] = 0
                eos = True

        ids = torch.tensor(
            [token_ids],
            device=self.execution_device,
            dtype=torch.long,
        )
        hidden = self._embedding(ids, out_dtype=torch.float32)
        insertion_index = 0
        pad_extra = 0
        embeds_info: list[dict[str, Any]] = []
        for original_index, raw_embed in other_embeds:
            embed = raw_embed
            if torch.is_tensor(embed):
                embed = {"type": "embedding", "data": embed}
            if not isinstance(embed, Mapping):
                raise TypeError(f"unsupported Qwen multimodal token: {embed!r}")
            embed_type = embed.get("type")
            if embed_type == "embedding":
                visual, extra = embed.get("data"), None
            elif embed_type == "image":
                visual, extra = self._preprocess_image(embed)
            else:
                raise ValueError(f"unsupported Qwen embed type: {embed_type!r}")
            if visual is None:
                insertion_index -= 1
                continue
            visual = visual.view(1, -1, visual.shape[-1]).to(
                device=self.execution_device,
                dtype=torch.float32,
            )
            visual_size = int(visual.shape[1])
            insert_at = insertion_index + int(original_index)
            if visual.shape[-1] != hidden.shape[-1]:
                insertion_index -= 1
                pad_extra += visual_size
                continue
            hidden = torch.cat(
                [hidden[:, :insert_at], visual, hidden[:, insert_at:]],
                dim=1,
            )
            attention = attention[:insert_at] + [1] * visual_size + attention[insert_at:]
            token_weights = (
                token_weights[:insert_at]
                + [1.0] * visual_size
                + token_weights[insert_at:]
            )
            insertion_index += visual_size - 1
            info = {
                "type": "image",
                "index": insert_at,
                "size": visual_size,
                "extra": extra,
            }
            embeds_info.append(info)

        if pad_extra:
            pad_ids = torch.full(
                (1, pad_extra),
                _PAD_TOKEN,
                device=self.execution_device,
                dtype=torch.long,
            )
            pad_hidden = self._embedding(pad_ids, out_dtype=torch.float32)
            hidden = torch.cat([hidden, pad_hidden], dim=1)
            attention.extend([0] * pad_extra)
            token_weights.extend([1.0] * pad_extra)

        binary = torch.tensor(
            [attention],
            device=self.execution_device,
            dtype=torch.long,
        )
        sequence_length = int(hidden.shape[1])
        position_ids = None
        if embeds_info:
            position_ids = self._mrope_position_ids(
                embeds_info,
                sequence_length,
                self.execution_device,
            )
        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=self.execution_device,
                dtype=torch.long,
            ).unsqueeze(0)
        freqs = self._precompute_freqs_cis(
            self._config.head_dim,
            position_ids,
            self._config.rope_theta,
            self._config.rope_scale,
            self._config.rope_dims,
            interleaved_mrope=getattr(self._config, "interleaved_mrope", False),
            device=self.execution_device,
        )
        additive_mask = _make_attention_mask(binary, self.compute_dtype)
        visual_mask = torch.zeros(
            (1, sequence_length),
            device=self.execution_device,
            dtype=torch.bool,
        )
        deepstack: list[torch.Tensor] | None = None
        for info in embeds_info:
            start = int(info["index"])
            stop = start + int(info["size"])
            visual_mask[:, start:stop] = True
            extra = info.get("extra")
            if isinstance(extra, Mapping) and extra.get("deepstack") is not None:
                values = list(extra["deepstack"])
                if deepstack is None:
                    deepstack = values
                else:
                    if len(deepstack) != len(values):
                        raise ValueError("Qwen vision DeepStack layer count mismatch")
                    deepstack = [torch.cat([a, b], dim=0) for a, b in zip(deepstack, values)]

        runtime_forward = getattr(self.runtime, "qwen_forward", None)
        if runtime_forward is None:
            raise RuntimeError(
                "the shared H3TPRuntime does not expose qwen_forward(); "
                "update the runtime worker before enabling Qwen32"
            )
        # Vision owns no persistent bytes when the shared runtime begins its
        # 50-layer collective protocol. DeepStack outputs are small request
        # activations and remain alive independently of the module weights.
        self._release_vision()
        self._forward_attempted = True
        result = runtime_forward(
            hidden,
            attention_mask=additive_mask,
            freqs_cis=freqs,
            deepstack_embeds=deepstack,
            visual_pos_masks=visual_mask if embeds_info else None,
        )
        if isinstance(result, Mapping):
            output = result.get("hidden", result.get("output"))
            if output is None:
                raise TypeError("qwen_forward mapping has no hidden/output tensor")
        elif isinstance(result, (tuple, list)):
            if not result:
                raise TypeError("qwen_forward returned an empty sequence")
            output = result[0]
        else:
            output = result
        if not torch.is_tensor(output):
            raise TypeError(f"qwen_forward returned {type(output).__name__}, expected tensor")
        if output.ndim != 3 or output.shape[-1] != _QWEN_HIDDEN:
            raise ValueError(
                f"Qwen32 runtime output must be [B,S,{_QWEN_HIDDEN}], got {tuple(output.shape)}"
            )
        tags = _vision_tags(sequence_length, embeds_info).to(output.device)
        weights = torch.tensor(
            token_weights,
            device=output.device,
            dtype=output.dtype,
        )
        if int(weights.numel()) != sequence_length:
            raise RuntimeError(
                "Qwen token-weight expansion lost alignment: "
                f"weights={weights.numel()}, sequence={sequence_length}"
            )
        return output.float(), binary, tags, additive_mask, weights

    def _ensure_vision(self) -> nn.Module:
        if self._vision is None:
            self._vision = _load_vision_frontend(
                self._layout,
                self._reader,
                device=self.execution_device,
                dtype=self.compute_dtype,
            )
        return self._vision

    def _release_vision(self) -> None:
        vision = self._vision
        if vision is None:
            return
        self._vision = None
        del vision
        gc.collect()
        if self.execution_device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(self.execution_device):
                torch.cuda.empty_cache()

    def _preprocess_image(self, embed: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        data = embed.get("data")
        if not torch.is_tensor(data):
            raise TypeError("Qwen image embed data must be a torch.Tensor")
        if embed.get("minimax_video_block", False):
            try:
                from comfy.text_encoders.minimax import process_video_block
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("MiniMax video preprocessing is unavailable") from exc
            flatten, grid = process_video_block(data)
        else:
            try:
                from comfy.text_encoders.qwen_vl import process_qwen2vl_images
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Qwen vision preprocessing is unavailable") from exc
            # ``process_qwen2vl_images`` accepts [N,C,H,W] and returns the
            # expanded temporal patch sequence used by Qwen3-VL.
            image = data
            if image.ndim == 3:
                image = image.unsqueeze(0)
            flatten, grid = process_qwen2vl_images(
                image,
                patch_size=16,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        vision = self._ensure_vision()
        merged, deepstack = vision(
            flatten.to(self.execution_device, dtype=self.compute_dtype),
            grid.to(self.execution_device),
        )
        return merged, {"grid": grid, "deepstack": deepstack}

    def encode_from_tokens(
        self,
        tokens: Mapping[str, Any] | Sequence[Any],
        *,
        return_pooled: bool | str = True,
        return_dict: bool = False,
        **_: Any,
    ) -> Any:
        result = self.encode_from_tokens_scheduled(tokens)
        if return_dict:
            cond, extra = result[0]
            return {"cond": cond, **extra}
        cond, extra = result[0]
        pooled = extra.get("pooled_output")
        if return_pooled is False:
            return cond
        return cond, pooled

    def encode_from_tokens_scheduled(
        self,
        tokens: Mapping[str, Any] | Sequence[Any],
        unprojected: bool = False,
        add_dict: Mapping[str, Any] | None = None,
        show_pbar: bool = True,
    ) -> list[list[Any]]:
        del unprojected, show_pbar
        self._check_open()
        cache_key = _token_digest(self.qwen_path, tokens)
        cached = self._cache.pop(cache_key, None)
        if cached is not None:
            self._cache[cache_key] = cached
            self._cache_hits += 1
            cond, cached_extra = _clone_cache_value(cached)
            extra = dict(cached_extra)
            if add_dict:
                extra.update(dict(add_dict))
            logging.info(
                "[H3 Qwen32] conditioning cache hit (%d hit / %d miss)",
                self._cache_hits,
                self._cache_misses,
            )
            return [[cond, extra]]
        self._cache_misses += 1
        if isinstance(tokens, Mapping):
            if "qwen3vl_32b" in tokens:
                sections = tokens["qwen3vl_32b"]
            elif len(tokens) == 1:
                sections = next(iter(tokens.values()))
            else:
                raise ValueError("Qwen32 CLIP accepts only qwen3vl_32b token sections")
        else:
            sections = tokens
        if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
            raise TypeError("Qwen token sections must be a sequence")
        if len(sections) == 0:
            sections = [[(_PAD_TOKEN, 1.0)]]
        has_weights = False
        has_multimodal = False
        max_section_tokens = 1
        for section in sections:
            if not isinstance(section, Sequence) or isinstance(section, (str, bytes)):
                raise TypeError("each Qwen token section must be a sequence")
            max_section_tokens = max(max_section_tokens, len(section))
            for raw in section:
                value, weight = _token_parts(raw)
                has_weights = has_weights or weight != 1.0
                has_multimodal = has_multimodal or _plain_token(value) is None
        if has_weights and has_multimodal:
            # Stock ClipTokenWeightEncoder's empty-baseline interpolation is
            # only shape-safe for text sections.  Silently applying weights
            # to a visual span would shift every subsequent token, so fail
            # closed and require an explicitly prepared embedding instead.
            raise ValueError(
                "Qwen32 token weights cannot be combined with image/video embeds; "
                "encode the multimodal prompt with unit weights"
            )
        with self._encode_lock:
            self._forward_attempted = False
            clear_called = False
            try:
                outputs = [self._process_one(section) for section in sections]
                if has_weights:
                    # Match ClipTokenWeightEncoder: encode a pad-only section
                    # at the longest token length and interpolate each weighted
                    # section around that empty baseline.
                    empty = [(_PAD_TOKEN, 1.0)] * max_section_tokens
                    empty_output = self._process_one(empty)
                    empty_hidden = empty_output[0]
                    weighted_outputs = []
                    for item in outputs:
                        output, binary, tags, mask, weights = item
                        if empty_hidden.shape[1] < output.shape[1]:
                            raise RuntimeError(
                                "Qwen weight baseline is shorter than the encoded section"
                            )
                        baseline = empty_hidden[:, : output.shape[1]]
                        factors = weights.to(
                            device=output.device,
                            dtype=output.dtype,
                        ).reshape(1, -1, 1)
                        weighted_outputs.append(
                            (
                                (output - baseline) * factors + baseline,
                                binary,
                                tags,
                                mask,
                                weights,
                            )
                        )
                    outputs = weighted_outputs
                # The tokenizer's max_length is intentionally very large, but
                # retain stock CLIP's concatenation behavior for hand-built
                # multi-section input.  Transfer the completed conditioning to
                # Comfy's intermediate device before opening the Qwen-clear
                # gate; the DiT must never observe a partial/failed request.
                cond = torch.cat([item[0] for item in outputs], dim=1)
                tags = torch.cat([item[2] for item in outputs], dim=0)
                del outputs
                target = _intermediate_device(cond.device)
                extra: dict[str, Any] = {
                    # MiniMaxH3 has no pooled representation; retaining the
                    # key keeps this facade compatible with ComfyUI's CLIP
                    # conditioning nodes and prompt blending utilities.
                    "pooled_output": None,
                    "minimax_token_tags": tags.to(target),
                }
                cond = cond.to(target)

                if self._forward_attempted:
                    clear = getattr(self.runtime, "qwen_clear", None)
                    if clear is None:
                        raise RuntimeError(
                            "the shared H3TPRuntime does not expose qwen_clear()"
                        )
                    # A successful clear is the one and only point that may
                    # notify the async-VAE gate.  The runtime waits for both
                    # ranks' barrier/synchronization before issuing it.
                    clear_called = True
                    clear(notify_vae=True)
                self._release_vision()
            except BaseException:
                self._release_vision()
                if self._forward_attempted and not clear_called:
                    clear = getattr(self.runtime, "qwen_clear", None)
                    if clear is not None:
                        try:
                            # Keep both ranks in a known unloaded state, but
                            # never let a failed conditioning request unlock
                            # VAE/DiT work.
                            clear(notify_vae=False)
                        except BaseException:
                            logging.exception(
                                "[H3 Qwen32] failed to clear Qwen after a conditioning error"
                            )
                raise

            if self._cache_entries > 0:
                self._cache[cache_key] = _clone_cache_value((cond, extra))
                while len(self._cache) > self._cache_entries:
                    self._cache.popitem(last=False)
            if add_dict:
                extra.update(dict(add_dict))
            return [[cond, extra]]

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Qwen32 CLIP facade is closed")

    # Compatibility methods used by generic ComfyUI CLIP utilities.  They do
    # not load or patch another model and intentionally remain side-effect free.
    def set_clip_options(self, options: Mapping[str, Any]) -> None:
        self._options.update(dict(options))
        if "execution_device" in options and options["execution_device"] is not None:
            self.execution_device = torch.device(options["execution_device"])

    def reset_clip_options(self) -> None:
        self._options.clear()

    def load_model(self, *_: Any, **__: Any) -> None:
        return None

    def get_sd(self) -> dict[str, Any]:
        return {}

    def clone(self) -> _Qwen32TPClip:
        return self

    def add_patches(self, *_: Any, **__: Any) -> list[Any]:
        raise RuntimeError("Qwen32 Q2 TP CLIP has no patchable dense state dict")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cache.clear()
        self._release_vision()
        embedding = getattr(self, "_embedding", None)
        if embedding is not None:
            try:
                embedding.close()
            except Exception:
                logging.exception("[H3 Qwen32] failed to close selected embedding")
        reader = getattr(self, "_reader", None)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                logging.exception("[H3 Qwen32] failed to close GGUF reader")
        gc.collect()


def _intermediate_device(fallback: torch.device) -> torch.device:
    try:
        import comfy.model_management
        return torch.device(comfy.model_management.intermediate_device())
    except (ImportError, AttributeError):
        return fallback


class MiniMaxH3DualRuntimeLoader:
    """Create the one runtime handle shared by H3 DiT and Qwen32 nodes."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        unets = sorted(
            name
            for category in ("unet_gguf", "unet", "diffusion_models")
            for name in _filename_list(category)
            if name.lower().endswith((".gguf", ".safetensors"))
        )
        loras = _filename_list("loras")
        qwen = sorted(
            name
            for name in _filename_list("text_encoders")
            if name.lower().endswith(".gguf") and "qwen" in name.lower()
        )
        if not qwen:
            qwen = [_DEFAULT_QWEN_MODEL]
        return {
            "required": {
                "unet_name": (unets,),
                "lora_name": (loras,),
                "qwen_name": (qwen,),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "staging_mib": ("INT", {"default": DEFAULT_STAGING_MIB, "min": 4, "max": 64, "step": 1}),
                "chunk_rows": ("INT", {"default": 2048, "min": 128, "max": 8192, "step": 128}),
            }
        }

    RETURN_TYPES = ("H3_TP_RUNTIME",)
    RETURN_NAMES = ("runtime",)
    FUNCTION = "load_runtime"
    CATEGORY = "dual_v100/H3 experimental"
    TITLE = "MiniMax H3 Shared TP Runtime (Qwen32 opt-in)"

    def load_runtime(
        self,
        unet_name: str,
        lora_name: str,
        qwen_name: str,
        strength: float = 1.0,
        staging_mib: int = DEFAULT_STAGING_MIB,
        chunk_rows: int = 2048,
    ) -> tuple[H3TPRuntimeHandle]:
        qwen_mode = _require_opt_in()
        model_path = _resolve_unet_name(unet_name)
        lora_path = _resolve_lora_name(lora_name)
        qwen_path = _resolve_qwen_name(qwen_name)
        config = RuntimeConfig(
            model_path=model_path,
            lora_path=lora_path,
            egrid_path=str(Path(DEFAULT_EGRID).resolve()),
            dit_format=normalize_weight_format("auto", model_path),
            lora_strength=float(strength),
            staging_mib=int(staging_mib),
            chunk_rows=int(chunk_rows),
            timeout_seconds=int(os.environ.get("H3_TP_TIMEOUT", "900")),
            results_dir=os.environ.get("H3_TP_RESULTS_DIR", _DEFAULT_RESULTS_DIR),
            qwen_model_path=qwen_path,
            qwen_staging_mib=int(staging_mib),
            qwen_residency="evict",
            qwen_keep_layers=0,
            qwen_cache_dequantized=False,
            qwen_mode=qwen_mode,
        )
        runtime = get_runtime(config)
        handle = H3TPRuntimeHandle(
            runtime,
            qwen_model_path=qwen_path,
            qwen_staging_mib=int(staging_mib),
        )
        return (handle,)


class Qwen32BQ2TPCLIPLoader:
    """Load the opt-in Qwen32 Q2 facade for stock MiniMax H3 conditioning."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "runtime": ("H3_TP_RUNTIME",),
                "residency": (["evict", "partial", "full"], {"default": "evict"}),
                "keep_layers": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "cache_dequantized": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "dual_v100/H3 experimental"
    TITLE = "Qwen32B Q2 Output-Row TP CLIP"

    def load_clip(
        self,
        runtime: H3TPRuntimeHandle,
        residency: str = "evict",
        keep_layers: int = 0,
        cache_dequantized: bool = False,
    ) -> tuple[_Qwen32TPClip]:
        return self._load_clip_impl(
            runtime,
            residency=residency,
            keep_layers=keep_layers,
            cache_dequantized=cache_dequantized,
            mode=None,
        )

    def _load_clip_impl(
        self,
        runtime: H3TPRuntimeHandle,
        *,
        residency: str,
        keep_layers: int,
        cache_dequantized: bool,
        mode: str | None,
    ) -> tuple[_Qwen32TPClip]:
        selected_mode = _require_opt_in(mode)
        if not isinstance(runtime, H3TPRuntimeHandle):
            raise TypeError(
                "Qwen32BQ2TPCLIPLoader requires the H3_TP_RUNTIME output from "
                "MiniMaxH3DualRuntimeLoader"
            )
        if runtime.qwen_clip() is not None:
            clip = runtime.qwen_clip()
            assert clip is not None
            if (
                clip.residency != residency
                or int(clip.keep_layers) != int(keep_layers)
                or clip.cache_dequantized != bool(cache_dequantized)
            ):
                raise RuntimeError(
                    "the shared runtime already has a Qwen32 CLIP with different "
                    "residency settings; use one loader configuration per runtime"
                )
            return (clip,)
        staging_mib = int(
            os.environ.get(
                "H3_QWEN32_STAGING_MIB",
                os.environ.get(
                    "H3_QWEN32_Q2_STAGING_MIB",
                    str(runtime.qwen_staging_mib),
                ),
            )
        )
        runtime.configure_qwen(
            staging_mib=staging_mib,
            residency=residency,
            keep_layers=int(keep_layers),
            cache_dequantized=bool(cache_dequantized),
            mode=selected_mode,
        )
        clip = _Qwen32TPClip(
            runtime,
            qwen_path=runtime.qwen_model_path,
            staging_mib=staging_mib,
            residency=residency,
            keep_layers=int(keep_layers),
            cache_dequantized=bool(cache_dequantized),
        )
        runtime.set_qwen_clip(clip)
        return (clip,)


class Qwen32BQ2MPCLIPLoader(Qwen32BQ2TPCLIPLoader):
    """Load the decoupled complete-layer Qwen32 Q2 MP facade.

    The runtime handle remains the same H3/DiT value; only the Qwen backend
    selector changes.  Keeping a separate node name makes workflow intent
    explicit while the older TP node can still follow the deployment mode.
    """

    TITLE = "Qwen32B Q2 Layer-MP CLIP"

    def load_clip(
        self,
        runtime: H3TPRuntimeHandle,
        residency: str = "evict",
        keep_layers: int = 0,
        cache_dequantized: bool = False,
    ) -> tuple[_Qwen32TPClip]:
        return self._load_clip_impl(
            runtime,
            residency=residency,
            keep_layers=keep_layers,
            cache_dequantized=cache_dequantized,
            mode="mp",
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3DualRuntimeLoader": MiniMaxH3DualRuntimeLoader,
    "Qwen32BQ2TPCLIPLoader": Qwen32BQ2TPCLIPLoader,
    "Qwen32BQ2MPCLIPLoader": Qwen32BQ2MPCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3DualRuntimeLoader": "MiniMax H3 Shared TP Runtime (Qwen32 opt-in)",
    "Qwen32BQ2TPCLIPLoader": "Qwen32B Q2 Output-Row TP CLIP",
    "Qwen32BQ2MPCLIPLoader": "Qwen32B Q2 Layer-MP CLIP",
}

__all__ = [
    "H3TPRuntimeHandle",
    "MiniMaxH3DualRuntimeLoader",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "Qwen32BQ2MPCLIPLoader",
    "Qwen32BQ2TPCLIPLoader",
]
