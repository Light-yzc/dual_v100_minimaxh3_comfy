# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import gguf
import json
import torch
import logging
import argparse
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

class ModelTemplate:
    arch = "invalid"  # string describing architecture
    shape_fix = False # whether to reshape tensors
    keys_detect = []  # list of lists to match in state dict
    keys_banned = []  # list of keys that should mark model as invalid for conversion
    keys_hiprec = []  # list of keys that need to be kept in fp32 for some reason
    keys_noquant = [] # list of keys that must retain their source precision
    keys_ignore = []  # list of strings to ignore keys by when found

    def handle_nd_tensor(self, key, data):
        raise NotImplementedError(f"Tensor detected that exceeds dims supported by C++ code! ({key} @ {data.shape})")

def key_matches(key, patterns):
    """
    Match a tensor name against a list of patterns.

    Plain patterns match anywhere in the key (today's behavior, unchanged --
    e.g. "pos_embedder", "scale_shift_table", ".modulation").

    A pattern prefixed with '^' matches only at the START of the key, e.g.
    "^tmlp." matches "tmlp.0.weight" but NOT "blocks.5.txtmlp.0.weight" --
    use this for short/generic fragments that would otherwise collide with
    an unrelated, similarly-named submodule elsewhere in the tensor name
    (see ModelKrea2: bare "tmlp."/"tproj." also matched every per-block
    "txtmlp."/"txtproj." tensor, silently forcing far more of the model to
    F32 than intended).
    """
    for pattern in patterns:
        if pattern.startswith("^"):
            if key.startswith(pattern[1:]):
                return True
        elif pattern in key:
            return True
    return False

class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]

class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight",]

class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight",]

class ModelHiDream(ModelTemplate):
    arch = "hidream"
    keys_detect = [
        (
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"
        )
    ]
    keys_hiprec = [
        # nn.parameter, can't load from BF16 ver
        ".ff_i.gate.weight",
        "img_emb.emb_pos"
    ]

class CosmosPredict2(ModelTemplate):
    arch = "cosmos"
    keys_detect = [
        (
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
        )
    ]
    keys_hiprec = ["pos_embedder"]
    keys_ignore = ["_extra_state", "accum_"]

class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = [
        (
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )
    ]

    def handle_nd_tensor(self, key, data):
        # hacky but don't have any better ideas
        path = f"./fix_5d_tensors_{self.arch}.safetensors" # TODO: somehow get a path here??
        if os.path.isfile(path):
            raise RuntimeError(f"5D tensor fix file already exists! {path}")
        fsd = {key: torch.from_numpy(data)}
        tqdm.write(f"5D key found in state dict! Manual fix required! - {key} {data.shape}")
        save_file(fsd, path)

class ModelWan(ModelHyVid):
    arch = "wan"
    keys_detect = [
        (
            "blocks.0.self_attn.norm_q.weight",
            "text_embedding.2.weight",
            "head.modulation",
        )
    ]
    keys_hiprec = [
        ".modulation" # nn.parameter, can't load from BF16 ver
    ]

class ModelLTXV(ModelTemplate):
    arch = "ltxv"
    keys_detect = [
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        ),
        # LTX 2.3 audio-video checkpoints replace the video-only caption
        # projection with audio/video connector modules.
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "audio_adaln_single.linear.weight",
        ),
    ]
    keys_hiprec = [
        "scale_shift_table", # nn.Parameter, can't load from BF16 base quant
        "learnable_registers", # Connector nn.Parameter, not a Linear weight
    ]
    # LTX's native INT8 ConvRot checkpoints intentionally leave these
    # projections in BF16. The many 32-output gate logits are particularly
    # small, so ConvRot dispatch overhead outweighs their INT8 benefit.
    keys_noquant = [
        "adaln_single",
        "patchify_proj",
        "proj_out",
        "to_gate_logits",
    ]

class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
        ), # Non-diffusers
        ("label_emb.0.0.weight",),
    ]

class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ), # Non-diffusers
    ]

class ModelLumina2(ModelTemplate):
    arch = "lumina2"
    keys_detect = [
        ("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight")
    ]

class ModelIdeogram(ModelTemplate):
    arch = "ideogram"
    keys_detect = [
        (
            "t_embedding.mlp_in.weight",
            "layers.0.attention.qkv.weight",
            "final_layer.linear.weight",
        )
    ]

class ModelKrea2(ModelTemplate):
    """
    Krea-2 is a novel architecture from krea.ai — NOT Ideogram4.
    Key structure (verified from Krea2_Turbo_fp8mixed.safetensors header):
      blocks.N.attn.{wq,wk,wv,wo,gate}  — separate Q/K/V/O projections + gating
      blocks.N.attn.qknorm.{qnorm,knorm} — Q/K norms
      blocks.N.mlp.{up,gate,down}        — SwiGLU-style MLP
      blocks.N.mod.lin                   — per-block modulation
      blocks.N.{pre,post}norm.scale      — RMSNorm scales
      txtfusion.layerwise_blocks.N.*     — layerwise text-image cross-attention
      txtfusion.refiner_blocks.N.*       — refiner text-image cross-attention
      txtfusion.projector                — text projector
      first.weight / last.*              — input / output projections
      tmlp.N / tproj.N                   — timestep MLP / projection

    NOTE: ComfyUI core must have krea2 diffusion model support
    (i.e. a detection branch for 'blocks.0.attn.wq.weight' in model_detection.py
    and a matching supported_models entry) for the GGUF to load correctly.
    Krea-2 was released 2026-06-22; verify your ComfyUI build is up to date.
    """
    arch = "krea2"
    keys_detect = [
        (
            "blocks.0.attn.wq.weight",
            "txtfusion.projector.weight",
            "first.weight",
        ),
        (
            "blocks.0.attn.wq.weight",
            "txtfusion.layerwise_blocks.0.attn.wq.weight",
            "last.linear.weight",
        ),
    ]
    keys_hiprec = [
        "^first.",
        "^last.",
        "^tproj.",
        "^tmlp.",
        "^txtmlp.",
        "^txtfusion.projector.",
    ]


class ModelMinimaxH3(ModelTemplate):
    arch = "minimax_h3"
    keys_detect = [
        (
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "blocks.0.attn.qkv_proj.weight",
            "final_layer.video_out.weight",
        )
    ]
    # This is a model buffer used to interpolate timestep embeddings, rather
    # than a Linear weight. Keep it in FP32 for the native interpolation path.
    keys_hiprec = ["adaln_t_table"]


arch_list = [ModelFlux, ModelSD3, ModelAura, ModelHiDream, CosmosPredict2,
             ModelLTXV, ModelHyVid, ModelWan, ModelSDXL, ModelSD1, ModelLumina2,
             ModelKrea2, ModelIdeogram, ModelMinimaxH3]

def is_model_arch(model, state_dict):
    # check if model is correct
    matched = False
    invalid = False
    for match_idx, match_list in enumerate(model.keys_detect):
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            if len(model.keys_detect) > 1:
                # Multiple detect variants usually mean multiple known checkpoint
                # exports of the same architecture (e.g. different key subsets
                # across releases). Logging which one matched makes it obvious
                # which variant you're actually converting.
                logging.info(f"* Matched keys_detect variant #{match_idx} for '{model.arch}': {match_list}")
            break
    assert not invalid, "Model architecture not allowed for conversion! (i.e. reference VS diffusers format)"
    return matched

def detect_arch(state_dict):
    model_arch = None
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            model_arch = arch()
            break
    assert model_arch is not None, "Unknown model architecture!"
    return model_arch

def validate_key_patterns(model_arch, state_dict):
    """
    Warn if a configured key pattern (keys_hiprec / keys_noquant / keys_ignore)
    matches no tensor at all in this checkpoint.

    These patterns are substring matches against tensor names (`any(x in key ...)`),
    hand-written against one specific checkpoint export. If the upstream model
    renames a layer in a later release, the pattern silently stops firing --
    no error, no warning, just quietly reduced precision/behavior on tensors
    that were meant to be protected. This check surfaces that case early,
    at conversion time, instead of relying on someone noticing degraded output
    later.

    This is intentionally a warning, not an assert: a pattern legitimately
    matching nothing can happen for known reasons too, e.g. a checkpoint
    variant that simply doesn't include that sub-module (see ModelKrea2's
    two keys_detect alternatives, which exist for exactly this reason).
    """
    for attr in ("keys_hiprec", "keys_noquant", "keys_ignore"):
        for pattern in getattr(model_arch, attr, []):
            if not any(key_matches(key, [pattern]) for key in state_dict.keys()):
                logging.warning(
                    f"[{model_arch.arch}] '{pattern}' in {attr} matched no tensor in this "
                    f"checkpoint -- possible naming drift in the source model, or an "
                    f"intentionally absent sub-module for this checkpoint variant. "
                    f"Verify this is expected before trusting the output precision."
                )

QUANT_TYPE_MAP = {
    "F16":  (gguf.GGMLQuantizationType.F16,  gguf.LlamaFileType.MOSTLY_F16),
    "BF16": (gguf.GGMLQuantizationType.BF16, gguf.LlamaFileType.MOSTLY_BF16),
    "Q8_0": (gguf.GGMLQuantizationType.Q8_0, gguf.LlamaFileType.MOSTLY_Q8_0),
    "Q5_1": (gguf.GGMLQuantizationType.Q5_1, gguf.LlamaFileType.MOSTLY_Q5_1),
    "Q5_0": (gguf.GGMLQuantizationType.Q5_0, gguf.LlamaFileType.MOSTLY_Q5_0),
    "Q4_1": (gguf.GGMLQuantizationType.Q4_1, gguf.LlamaFileType.MOSTLY_Q4_1),
    "Q4_0": (gguf.GGMLQuantizationType.Q4_0, gguf.LlamaFileType.MOSTLY_Q4_0),
    "Q8_CR": (gguf.GGMLQuantizationType.I8, None),  # INT8 ConvRot (ComfyUI native)
    # Q4_PT is retired pending a performant Ampere W4A16 backend.
    # "Q4_PT": (gguf.GGMLQuantizationType.I8, None),
}

TARGET_SIZE_QUANT_TYPE = "TARGET_SIZE"
MEBIBYTE = 1024 * 1024


def _tensor_size_bytes(shape, quant_type):
    """Return the GGUF payload size for a tensor with the given quantization."""
    n_params = 1
    for dim_size in shape:
        n_params *= dim_size

    if quant_type == gguf.GGMLQuantizationType.I8:
        return n_params

    block_size, type_size = gguf.constants.GGML_QUANT_SIZES[quant_type]
    if n_params % block_size:
        raise ValueError(
            f"{quant_type.name} requires a tensor size divisible by {block_size}, "
            f"got shape {tuple(shape)} ({n_params} elements)."
        )
    return n_params // block_size * type_size


def _default_qtype(data_or_dtype):
    return (
        gguf.GGMLQuantizationType.BF16
        if getattr(data_or_dtype, "dtype", data_or_dtype) == torch.bfloat16
        else gguf.GGMLQuantizationType.F16
    )


def _is_target_core_tensor(key, data, model_arch):
    if len(data.shape) != 2:
        return False
    if data.numel() <= QUANTIZATION_THRESHOLD:
        return False
    if key_matches(key, model_arch.keys_hiprec) or key_matches(key, model_arch.keys_noquant):
        return False
    return True


def _can_use_q4_0(data):
    block_size, _ = gguf.constants.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_0]
    return data.shape[-1] % block_size == 0


def plan_target_size_quantization(state_dict, model_arch, max_size_mb):
    """
    Select per-tensor types that fit a maximum serialized payload size.

    Core 2-D tensors start in Q8_CR. The center of their checkpoint order is
    downgraded first, leaving the beginning and end at Q8_CR as long as possible.
    Once every Q4-compatible core tensor is Q4_0, ordinary 1-D tensors may be
    reduced to BF16. Protected tensors always remain F32.
    """
    if max_size_mb <= 0:
        raise ValueError("--max-size-mb must be greater than zero.")

    plan = {}
    core_tensors = []
    one_dimensional_tensors = []

    for key, data in state_dict.items():
        if key_matches(key, model_arch.keys_ignore):
            continue
        if key.endswith(".comfy_quant") or key.endswith("_scale") and len(data.shape) == 0:
            continue
        if len(data.shape) == 0 or len(data.shape) > MAX_TENSOR_DIMS:
            continue

        n_params = data.numel()
        if len(data.shape) == 1 or n_params <= QUANTIZATION_THRESHOLD or key_matches(key, model_arch.keys_hiprec):
            plan[key] = gguf.GGMLQuantizationType.F32
            if len(data.shape) == 1 and not key_matches(key, model_arch.keys_hiprec):
                one_dimensional_tensors.append((key, data))
        elif key_matches(key, model_arch.keys_noquant):
            plan[key] = _default_qtype(data)
        elif len(data.shape) == 4 and "conv" in key.lower():
            plan[key] = gguf.GGMLQuantizationType.F16
        elif _is_target_core_tensor(key, data, model_arch):
            plan[key] = gguf.GGMLQuantizationType.I8
            if _can_use_q4_0(data):
                core_tensors.append((key, data))
        else:
            plan[key] = _default_qtype(data)

    def plan_size():
        total = 0
        for key, data in state_dict.items():
            if key not in plan:
                continue
            qtype = plan[key]
            total += _tensor_size_bytes(data.shape, qtype)
            if qtype == gguf.GGMLQuantizationType.I8:
                # Q8_CR stores a F32 scale for every output row.
                total += data.shape[0] * 4
        return total

    target_size = int(max_size_mb * MEBIBYTE)
    maximum_size = plan_size()
    if maximum_size <= target_size:
        return plan, maximum_size, maximum_size

    center = (len(core_tensors) - 1) / 2
    for index, (key, _) in sorted(
        enumerate(core_tensors),
        key=lambda item: (abs(item[0] - center), item[0]),
    ):
        plan[key] = gguf.GGMLQuantizationType.Q4_0
        current_size = plan_size()
        if current_size <= target_size:
            return plan, maximum_size, current_size

    for key, _ in one_dimensional_tensors:
        plan[key] = gguf.GGMLQuantizationType.BF16
        current_size = plan_size()
        if current_size <= target_size:
            return plan, maximum_size, current_size

    minimum_size = plan_size()
    raise ValueError(
        f"Cannot shrink this model to {max_size_mb:g} MiB. "
        f"The smallest supported TARGET_SIZE output is {minimum_size / MEBIBYTE:.2f} MiB "
        f"(all Q4_0-compatible core matrices at Q4_0 and ordinary 1-D tensors at BF16). "
        "Q3 and lower quantization are not supported."
    )


def quantize_int8_convrot(weight, convrot_groupsize=256):
    """
    Quantize a 2D Linear weight to INT8 with ConvRot grouping.
    Uses per-output-channel scales to match ComfyUI's TensorWiseINT8Layout.
    """
    weight = weight.to(torch.float32)
    orig_shape = tuple(weight.shape)
    groupsize = next(
        (
            size
            for size in (convrot_groupsize, 64, 16, 4)
            if size <= weight.shape[1] and weight.shape[1] % size == 0
        ),
        None,
    )
    if groupsize is not None:
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

        hadamard = _build_hadamard(groupsize, device=weight.device, dtype=weight.dtype)
        weight = _rotate_weight(weight, hadamard, groupsize)

    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 127.0
    qdata = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    quant_conf = {
        "format": "int8_tensorwise",
        "convrot": groupsize is not None,
        "weight_rotated": groupsize is not None,
        "per_row": True,
    }
    if groupsize is not None:
        quant_conf["convrot_groupsize"] = groupsize
    return qdata, scale, quant_conf, orig_shape


def retired_quantize_int4_pytorch(weight, group_size=64):
    """
    Quantize a 2D Linear weight for PyTorch's native INT4 kernel.
    Weights are serialized as packed uint8 [n, k//2]. The runtime converts this
    portable representation to PyTorch's device-specific INT4 layout.
    """
    orig_shape = tuple(weight.shape)
    n, k = orig_shape
    weight = weight.to(torch.float32)

    pad = 0
    if k % group_size != 0:
        pad = group_size - (k % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))
        k = k + pad

    w_grouped = weight.reshape(n, k // group_size, group_size)
    w_min = w_grouped.amin(dim=-1, keepdim=True)
    w_max = w_grouped.amax(dim=-1, keepdim=True)
    scale = (w_max - w_min) / 15.0
    scale = scale.clamp_min(1e-9)
    q = ((w_grouped - w_min) / scale).round().clamp(0, 15).to(torch.uint8)

    q_flat = q.reshape(n, k)
    packed = ((q_flat[:, 0::2] << 4) | q_flat[:, 1::2]).to(torch.uint8)

    qsz = torch.zeros(k // group_size, n, 2, dtype=torch.float32)
    qsz[:, :, 0] = scale.reshape(n, k // group_size).t()
    # _weight_int4pack_mm dequantizes as (q - 8) * scale + offset.
    qsz[:, :, 1] = (w_min + 8 * scale).reshape(n, k // group_size).t()

    quant_conf = {
        "format": "int4_compact_gemm",
        "group_size": group_size,
        "orig_shape": orig_shape,
        "pad": pad,
    }
    return packed, qsz, quant_conf, orig_shape

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert diffusion model safetensors/ckpt to GGUF."
        " By default produces an F16/BF16 GGUF; use --quant-type to quantize."
    )
    parser.add_argument("--src", required=True, help="Source model ckpt/safetensors file.")
    parser.add_argument("--dst", help="Output GGUF file path.")
    parser.add_argument(
        "--quant-type",
        choices=list(QUANT_TYPE_MAP.keys()),
        default=None,
        help="Target quantization type for eligible 2-D+ tensors "
             "(1-D biases/scales stay F32). Defaults to F16/BF16 matching the source dtype.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        help=(
            "Maximum output payload size in MiB. Selects TARGET_SIZE quantization: "
            "Q8_CR core weights are progressively changed to Q4_0 from the model "
            "center outward, then ordinary 1-D tensors to BF16 if necessary."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error("No input provided!")

    return args

def strip_prefix(state_dict):
    # prefix for mixed state dict
    prefix = None
    for pfx in ["model.diffusion_model.", "model."]:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break

    # prefix for uniform state dict
    if prefix is None:
        for pfx in ["net."]:
            if all([x.startswith(pfx) for x in state_dict.keys()]):
                prefix = pfx
                break

    # strip prefix if found
    if prefix is not None:
        logging.info(f"State dict prefix found: '{prefix}'")
        sd = {}
        for k, v in state_dict.items():
            if prefix not in k:
                continue
            k = k.replace(prefix, "")
            sd[k] = v
    else:
        logging.debug("State dict has no prefix")
        sd = state_dict

    return sd

def load_state_dict(path, progress_callback=None):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        with tqdm(total=1, desc="Reading checkpoint", unit="file") as progress:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            progress.update()
        if progress_callback is not None:
            progress_callback("read", 1, 1)
        for subkey in ["model", "module"]:
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
        if len(state_dict) < 20:
            raise RuntimeError(f"pt subkey load failed: {state_dict.keys()}")
    else:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            state_dict = {}
            for index, key in enumerate(tqdm(keys, desc="Reading tensors", unit="tensor"), start=1):
                state_dict[key] = checkpoint.get_tensor(key)
                if progress_callback is not None:
                    progress_callback("read", index, len(keys))

    return strip_prefix(state_dict)

def load_safetensors_metadata(path):
    if not path.endswith(".safetensors"):
        return {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        return checkpoint.metadata() or {}

def handle_tensors(writer, state_dict, model_arch, quant_type=None, quant_type_name=None, quantization_plan=None, progress_callback=None):
    # Pre-collect per-tensor FP8 scales (0-dim float32 tensors named "{key}_scale").
    # These must be applied to their FP8 weight tensors before GGUF quantization.
    # The actual weight value is fp8_value * scale; ignoring scale produces wrong magnitudes.
    fp8_scales = {
        k[:-len("_scale")]: v.item()
        for k, v in state_dict.items()
        if k.endswith("_scale") and len(v.shape) == 0 and v.dtype == torch.float32
    }
    if fp8_scales:
        tqdm.write(f"Found {len(fp8_scales)} FP8 per-tensor scale(s); will apply before quantization.")

    name_lengths = tuple(sorted(
        ((key, len(key)) for key in state_dict.keys()),
        key=lambda item: item[1],
        reverse=True,
    ))
    if not name_lengths:
        return
    max_name_len = name_lengths[0][1]
    if max_name_len > MAX_TENSOR_NAME_LENGTH:
        bad_list = ", ".join(f"{key!r} ({namelen})" for key, namelen in name_lengths if namelen > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f"Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad_list}")
    for tensor_index, (key, data) in enumerate(tqdm(state_dict.items()), start=1):
        old_dtype = data.dtype

        if key_matches(key, model_arch.keys_ignore):
            tqdm.write(f"Filtering ignored key: '{key}'")
            continue

        # comfy_quant tensors are FP8 scale factors specific to ComfyUI's custom FP8 format.
        # weight_scale tensors are 0-dim per-tensor FP8 scales (e.g. from torchao/fp8 fine-tunes).
        # Both are meaningless after GGUF re-quantization and must be dropped so the loader
        # does not try to apply them to already-GGUF-dequantized weights.
        if key.endswith(".comfy_quant") or key.endswith("_scale") and len(data.shape) == 0:
            tqdm.write(f"Dropping FP8 scale tensor: '{key}'")
            continue

        # 0-dim (scalar) tensors cannot be stored in GGUF and have no meaningful weight data.
        if len(data.shape) == 0:
            tqdm.write(f"Skipping 0-dim scalar tensor: '{key}'")
            continue

        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        # this is so we don't break torch 2.0.X
        elif data.dtype in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
            data = data.to(torch.float32)
            if key in fp8_scales:
                data = data * fp8_scales[key]  # apply per-tensor dequantization scale
            data = data.numpy()
        else:
            data = data.numpy()

        n_dims = len(data.shape)
        data_shape = data.shape
        data_qtype = _default_qtype(old_dtype)

        # The max no. of dimensions that can be handled by the quantization code is 4.
        if n_dims > MAX_TENSOR_DIMS:
            model_arch.handle_nd_tensor(key, data)
            continue

        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size

        _FP8_DTYPES = {getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)} - {None}
        apply_quantization_rules = (
            quant_type_name == "Q8_CR"
            or old_dtype in (torch.float32, torch.bfloat16)
            or old_dtype in _FP8_DTYPES
        )
        if quantization_plan is not None and key in quantization_plan:
            data_qtype = quantization_plan[key]
        elif apply_quantization_rules:
            if n_dims == 1:
                # One-dimensional tensors should be kept in F32. This is a
                # universal safety net and must take priority over
                # keys_noquant -- a broad keys_noquant prefix (e.g. Krea2's
                # "^last.") would otherwise also match that submodule's 1D
                # bias/scale tensors and silently downgrade them from the
                # F32 they'd normally always get to whatever the generic
                # default happens to be (F16/BF16).
                data_qtype = gguf.GGMLQuantizationType.F32
            elif n_params <= QUANTIZATION_THRESHOLD:
                data_qtype = gguf.GGMLQuantizationType.F32
            elif key_matches(key, model_arch.keys_hiprec):
                # More specific than keys_noquant by design: keys_hiprec
                # forces F32 even when a broader keys_noquant pattern for
                # the same submodule would also match (e.g. Krea2's
                # "last.modulation.lin" needs full F32, even though the
                # broader "^last." keys_noquant entry also matches it).
                data_qtype = gguf.GGMLQuantizationType.F32
            elif key_matches(key, model_arch.keys_noquant):
                pass
            elif n_dims == 4 and "conv" in key.lower():
                # Native quantized paths are Linear-only.
                data_qtype = gguf.GGMLQuantizationType.F16
            elif quant_type is not None:
                data_qtype = quant_type

        if quant_type_name == "Q8_CR" and n_dims > 1 and n_dims != 2:
            # Custom native layouts only represent Linear matrices.
            data_qtype = gguf.GGMLQuantizationType.F16
        # Q4_PT layout restrictions are retained with
        # retired_quantize_int4_pytorch and are intentionally not selectable.

        # Q8_CR is the supported custom quantization path.
        if data_qtype == gguf.GGMLQuantizationType.I8 and n_dims == 2:
            qdata, scale, quant_conf, orig_shape = quantize_int8_convrot(torch.from_numpy(data))
            writer.add_tensor(key, qdata.numpy(), raw_dtype=gguf.GGMLQuantizationType.I8)
            writer.add_tensor(f"{key}_scale", scale.numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
            writer.add_string(f"comfy.gguf.quant.{key}", json.dumps(quant_conf))
            if progress_callback is not None:
                progress_callback("quantize", tensor_index, len(state_dict))
            continue

            # Q4_PT emission is retired with its runtime backend.

        if (model_arch.shape_fix                        # NEVER reshape for models such as flux
            and n_dims > 1                              # Skip one-dimensional tensors
            and n_params >= REARRANGE_THRESHOLD         # Only rearrange tensors meeting the size requirement
            and (n_params / 256).is_integer()           # Rearranging only makes sense if total elements is divisible by 256
            and not (data.shape[-1] / 256).is_integer() # Only need to rearrange if the last dimension is not divisible by 256
        ):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except (AttributeError, gguf.QuantError) as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        new_name = key # do we need to rename?

        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len + 4}s' % f'{new_name}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        writer.add_tensor(new_name, data, raw_dtype=data_qtype)
        if progress_callback is not None:
            progress_callback("quantize", tensor_index, len(state_dict))

def convert_file(path, dst_path=None, interact=True, overwrite=False, quant_type_name=None, max_size_mb=None, progress_callback=None):
    # load & run model detection logic
    state_dict = load_state_dict(path, progress_callback=progress_callback)
    source_metadata = load_safetensors_metadata(path)
    model_arch = detect_arch(state_dict)
    logging.info(f"* Architecture detected from input: {model_arch.arch}")
    validate_key_patterns(model_arch, state_dict)

    if max_size_mb is not None and quant_type_name not in (None, TARGET_SIZE_QUANT_TYPE):
        raise ValueError("--max-size-mb cannot be combined with --quant-type.")

    quantization_plan = None
    if max_size_mb is not None:
        quant_type_name = TARGET_SIZE_QUANT_TYPE
        quantization_plan, maximum_size, selected_size = plan_target_size_quantization(
            state_dict, model_arch, max_size_mb
        )
        logging.info(
            "TARGET_SIZE selected %.2f MiB from a Q8_CR baseline of %.2f MiB.",
            selected_size / MEBIBYTE,
            maximum_size / MEBIBYTE,
        )

    # resolve quant type from name if provided
    quant_type = None
    if quantization_plan is not None:
        ftype_name = TARGET_SIZE_QUANT_TYPE
        ftype_gguf = None
    elif quant_type_name is not None and quant_type_name in QUANT_TYPE_MAP:
        quant_type, ftype_gguf = QUANT_TYPE_MAP[quant_type_name]
        ftype_name = quant_type_name
    else:
        # detect & set dtype from source file
        dtypes = [x.dtype for x in state_dict.values()]
        dtypes = {x: dtypes.count(x) for x in set(dtypes)}
        main_dtype = max(dtypes, key=dtypes.get)

        if main_dtype == torch.bfloat16:
            ftype_name = "BF16"
            ftype_gguf = gguf.LlamaFileType.MOSTLY_BF16
        # elif main_dtype == torch.float32:
        #     ftype_name = "F32"
        #     ftype_gguf = None
        else:
            ftype_name = "F16"
            ftype_gguf = gguf.LlamaFileType.MOSTLY_F16

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{ftype_name}.gguf"
    elif "{ftype}" in dst_path: # lcpp logic
        dst_path = dst_path.replace("{ftype}", ftype_name)

    if os.path.isfile(dst_path) and not overwrite:
        if interact:
            input("Output exists enter to continue or ctrl+c to abort!")
        else:
            raise OSError("Output exists and overwriting is disabled!")

    # handle actual file
    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if ftype_gguf is not None:
        writer.add_file_type(ftype_gguf)
    if "config" in source_metadata:
        writer.add_string("config", source_metadata["config"])

    handle_tensors(
        writer,
        state_dict,
        model_arch,
        quant_type=quant_type,
        quant_type_name=quant_type_name,
        quantization_plan=quantization_plan,
        progress_callback=progress_callback,
    )
    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    fix = f"./fix_5d_tensors_{model_arch.arch}.safetensors"
    if os.path.isfile(fix):
        logging.warning(f"\n### Warning! Fix file found at '{fix}'")
        logging.warning(" you most likely need to run 'fix_5d_tensors.py' after quantization.")

    return dst_path, model_arch

if __name__ == "__main__":
    args = parse_args()
    convert_file(args.src, args.dst, quant_type_name=args.quant_type, max_size_mb=args.max_size_mb)
