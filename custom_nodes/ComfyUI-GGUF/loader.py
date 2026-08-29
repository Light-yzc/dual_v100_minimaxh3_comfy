# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import warnings
import logging
import torch
import gguf
import json
import re
import os
import threading
import ctypes
import comfy.memory_management
from .ops import GGMLTensor
from .dequant import is_quantized, dequantize_tensor
from .quant_ops import make_quantized
from custom_nodes.NoHostMMap.gguf_reader import (
    NoMmapGGUFReader,
    meta_tensor_for_gguf,
)

IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "hyvid", "wan", "lumina2", "qwen_image", "ideogram", "krea2", "minimax_h3"}
TXT_ARCH_LIST = {"t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3"}
VIS_TYPE_LIST = {"clip-vision", "mmproj"}
DIRECT_STAGING_BYTES = 8 << 20


class _DirectGGUFReader:
    """Read GGUF payloads into a chosen device without mapping the file.

    This is only for the small, resident Qwen Q4 path.  The destination is
    allocated in its final device/dtype and the host side is capped at one
    staging buffer, so a multi-gigabyte CPU state dict is never created.
    """

    def __init__(self, path, device, staging_bytes=DIRECT_STAGING_BYTES):
        self.path = os.fspath(path)
        self.device = torch.device(device)
        self.file = open(self.path, "rb", buffering=0)
        self.staging = torch.empty(staging_bytes, dtype=torch.uint8, device="cpu")
        try:
            os.posix_fadvise(self.file.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass

    def close(self):
        if not self.file.closed:
            self.file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _read(self, offset, size):
        if size > self.staging.numel():
            raise ValueError(
                f"GGUF tensor chunk {size} exceeds direct staging buffer "
                f"{self.staging.numel()}"
            )
        view_type = ctypes.c_ubyte * size
        view = memoryview(view_type.from_address(self.staging.data_ptr()))
        try:
            self.file.seek(offset)
            done = 0
            while done < size:
                count = self.file.readinto(view[done:])
                if count is None or count <= 0:
                    raise OSError(f"short GGUF read at {offset + done}: {self.path}")
                done += count
        finally:
            view.release()
        return self.staging[:size]

    def read_tensor(self, tensor, storage):
        """Return one raw GGUF tensor in final device storage."""
        target = torch.empty(storage.shape, dtype=storage.dtype, device=self.device)
        target_bytes = target.view(torch.uint8).reshape(-1)
        copied = 0
        while copied < int(tensor.n_bytes):
            count = min(self.staging.numel(), int(tensor.n_bytes) - copied)
            source = self._read(int(tensor.data_offset) + copied, count)
            # Blocking copy is deliberate: the bounded staging buffer is reused
            # immediately and this path is load-time only.
            target_bytes[copied:copied + count].copy_(source)
            try:
                os.posix_fadvise(
                    self.file.fileno(),
                    int(tensor.data_offset) + copied,
                    count,
                    os.POSIX_FADV_DONTNEED,
                )
            except (AttributeError, OSError):
                pass
            copied += count
        return target

def device_supports_bf16():
    """
    Return True if the active torch device can run bf16 natively. On devices
    without native bf16 support, computation silently falls back to fp32 which
    is very slow, so callers should load tensors as fp16 instead.
    """
    try:
        import comfy.model_management
        return comfy.model_management.should_use_bf16(comfy.model_management.get_torch_device())
    except Exception:
        # If support can't be determined, keep the previous bf16 behavior.
        return True


def dynamic_gguf_file_slice(path):
    """
    Create file handle metadata used by DynamicVRAM to transfer a GGUF tensor
    directly from disk to GPU memory.  The safe H3 profile deliberately keeps
    an ordinary file descriptor instead of constructing ModelMMAP.
    """
    if not comfy.memory_management.aimdo_enabled:
        return None

    if _no_host_mmap_enabled():
        return open(path, "rb", buffering=0), threading.Lock()

    import comfy_aimdo.model_mmap

    model_mmap = comfy_aimdo.model_mmap.ModelMMAP(path)
    return model_mmap, threading.Lock()


def attach_dynamic_file_slice(torch_tensor, file_ref, file_lock, offset, size):
    storage = torch_tensor.untyped_storage()
    if hasattr(file_ref, "get_file_handle"):
        slice_ref = file_ref.get_file_handle()
        storage._comfy_tensor_mmap_refs = (file_ref,)
    else:
        slice_ref = file_ref
        # Keep the ordinary descriptor open for the lifetime of every meta
        # tensor that points at this model file.
        storage._comfy_tensor_file_refs = (file_ref,)
    storage._comfy_tensor_file_slice = comfy.memory_management.TensorFileSlice(
        slice_ref,
        file_lock,
        offset,
        size,
    )


def attach_dynamic_file_slice_to_value(value, file_ref, file_lock, offset, size):
    """Attach the source range to the final state-dict value.

    A GGUF value is reshaped, and BF16 values may also be converted to FP16,
    after the reader creates the initial meta tensor.  Those operations are
    allowed to allocate a new meta storage, so attaching the descriptor only
    to the reader tensor is not sufficient.  Quantized values keep their raw
    storage in ``_qdata``; ordinary tensors and Parameters are handled
    directly.  The descriptor is metadata only, and the payload remains on
    disk until DynamicVRAM requests this one tensor.
    """
    storage_tensor = getattr(value, "_qdata", value)
    attach_dynamic_file_slice(storage_tensor, file_ref, file_lock, offset, size)


def _no_host_mmap_enabled():
    return os.environ.get("H3_NO_HOST_MMAP", "1").lower() not in {"0", "false", "no"}


def _gguf_reader(path):
    return NoMmapGGUFReader(path) if _no_host_mmap_enabled() else gguf.GGUFReader(path)

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]].item())
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_list_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        return tuple(str(field.parts[part_idx], encoding="utf-8") for part_idx in field.data)
    elif field_type in [int, float, bool]:
        return tuple(field_type(field.parts[part_idx][0]) for part_idx in field.data)
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_gguf_metadata(reader):
    """Extract all simple metadata fields like safetensors"""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:  # Simple scalar fields only
                if field.types[0] == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == gguf.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except:
            continue
    return metadata

def gguf_tensor_count(path):
    return len(_gguf_reader(path).tensors)


def gguf_sd_loader(
    path,
    handle_prefix="model.diffusion_model.",
    is_text_model=False,
    dynamic=False,
    progress_callback=None,
    direct_device=None,
):
    """
    Read state dict as fake tensors
    """
    if _no_host_mmap_enabled() and not dynamic and direct_device is None:
        raise RuntimeError(
            "H3_NO_HOST_MMAP=1 requires DynamicVRAM for GGUF models; "
            "use the resident direct-device loader or the DualV100 dynamic loader."
        )
    reader = _gguf_reader(path)
    dynamic_file_slice = dynamic_gguf_file_slice(path) if dynamic else None
    # ``direct_device`` may be a single device or a selector
    # ``(source_tensor_name, source_path) -> device``.  The selector form lets
    # a resident model-parallel loader put every raw tensor on its final owner
    # while it is read, instead of materialising the whole GGUF on cuda:0 and
    # retaining/copying the tail on cuda:1 afterwards.
    direct_readers = {}

    def direct_reader_for(tensor_name):
        if direct_device is None or dynamic:
            return None
        target = (
            direct_device(tensor_name, os.fspath(path))
            if callable(direct_device)
            else direct_device
        )
        target = torch.device(target)
        key = str(target)
        result = direct_readers.get(key)
        if result is None:
            result = _DirectGGUFReader(path, target)
            direct_readers[key] = result
        return result

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = get_field(reader, "general.architecture", str)
    type_str = get_field(reader, "general.type", str)
    if arch_str in [None, "pig", "cow"]:
        if is_text_model:
            raise ValueError(f"This gguf file is incompatible with llama.cpp!\nConsider using safetensors or a compatible gguf file\n({path})")
        compat = "sd.cpp" if arch_str is None else arch_str
        # import here to avoid changes to convert.py breaking regular models
        from .tools.convert import detect_arch
        try:
            arch_str = detect_arch(set(val[0] for val in tensors)).arch
        except Exception as e:
            raise ValueError(f"This model is not currently supported - ({e})")
    elif arch_str not in TXT_ARCH_LIST and is_text_model:
        if type_str not in VIS_TYPE_LIST:
            raise ValueError(f"Unexpected text model architecture type in GGUF file: {arch_str!r}")
    elif arch_str not in IMG_ARCH_LIST and not is_text_model:
        raise ValueError(f"Unexpected architecture type in GGUF file: {arch_str!r}")

    if compat:
        logging.warning(f"Warning: This gguf model file is loaded in compatibility mode '{compat}' [arch:{arch_str}]")

    # Q8_CR weights must use ComfyUI's native TensorWiseINT8Layout rather than
    # the generic GGML layout so DynamicVRAM retains native INT8 ConvRot kernels.
    custom_quant_configs = {}
    for field_name in reader.fields:
        if field_name.startswith("comfy.gguf.quant."):
            key = field_name[len("comfy.gguf.quant."):]
            field = reader.get_field(field_name)
            custom_quant_configs[key] = json.loads(str(field.parts[field.data[-1]], "utf-8"))

    custom_quant_tensor_names = {
        tensor_name
        for key, quant_conf in custom_quant_configs.items()
        if quant_conf.get("format") == "int8_tensorwise"
        for tensor_name in (key, f"{key}_scale")
    }

    # main loading loop
    # Devices without native bf16 fall back to slow fp32 compute, so load the
    # full-precision BF16 storage tensors as fp16 there instead.
    bf16_storage_dtype = torch.bfloat16 if device_supports_bf16() else torch.float16
    state_dict = {}
    qtype_dict = {}
    for tensor_index, (sd_key, tensor) in enumerate(tensors, start=1):
        tensor_name = tensor.name
        direct_reader = direct_reader_for(tensor_name)
        if direct_reader is not None:
            storage = meta_tensor_for_gguf(tensor, torch)
            torch_tensor = direct_reader.read_tensor(tensor, storage)
        elif _no_host_mmap_enabled():
            torch_tensor = meta_tensor_for_gguf(tensor, torch)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                torch_tensor = torch.from_numpy(tensor.data)
        if dynamic_file_slice is not None:
            file_ref, file_lock = dynamic_file_slice
            attach_dynamic_file_slice(
                torch_tensor,
                file_ref,
                file_lock,
                tensor.data_offset,
                tensor.n_bytes,
            )

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any([tensor_name.endswith(x) for x in (".proj_in.weight", ".proj_out.weight")]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # add to state dict
        if dynamic and sd_key not in custom_quant_tensor_names:
            if tensor.tensor_type in {
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
            }:
                state_dict[sd_key] = torch_tensor.view(*shape)
            elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                state_dict[sd_key] = torch_tensor.view(torch.bfloat16).reshape(shape).to(
                    dtype=torch.float32 if len(shape) <= 1 else bf16_storage_dtype,
                )
            else:
                state_dict[sd_key] = make_quantized(torch_tensor, tensor.tensor_type, shape)
        elif tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
            state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
        else:
            state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        # view/reshape/to and the native quantization conversion above may
        # replace the meta storage.  Re-attach the exact source range to the
        # value that will actually enter the model state dict.
        if dynamic_file_slice is not None:
            file_ref, file_lock = dynamic_file_slice
            attach_dynamic_file_slice_to_value(
                state_dict[sd_key], file_ref, file_lock,
                tensor.data_offset, tensor.n_bytes,
            )

        # BF16 GGUF tensors are full-precision storage, not compressed quants.
        if not dynamic and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            dtype = torch.float32 if len(shape) <= 1 else bf16_storage_dtype
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=dtype)

        # keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
        if progress_callback is not None:
            progress_callback(tensor_index, len(tensors))

    # print loaded tensor type counts
    logging.info("gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))
    for direct_reader in direct_readers.values():
        direct_reader.close()

    # mark largest tensor for vram estimation
    qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # extra info to return
    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader)
    }

    # Detect custom ComfyUI native quantization metadata
    warned_unrotated_convrot = False
    for field_name in reader.fields:
        if not field_name.startswith("comfy.gguf.quant."):
            continue
        key = field_name[len("comfy.gguf.quant."):]
        field = reader.get_field(field_name)
        quant_conf = custom_quant_configs[key]
        fmt = quant_conf.get("format")

        if fmt in {"int4_compact_gemm", "int4_pytorch"}:
            raise ValueError(
                "Q4_PT GGUF files are retired because PyTorch's Ampere INT4 "
                "kernel is not performance-competitive. Reconvert as Q8_CR."
            )

        weight_key = key
        scale_key = f"{key}_scale"
        if weight_key not in state_dict or scale_key not in state_dict:
            logging.warning(f"Missing custom quant tensors for {weight_key}")
            continue

        weight_ggml = state_dict[weight_key]
        scale_ggml = state_dict[scale_key]

        if fmt == "int8_tensorwise":
            if quant_conf.get("convrot") and not quant_conf.get("weight_rotated", False):
                if not warned_unrotated_convrot:
                    logging.warning(
                        "Disabling ConvRot because this GGUF does not mark its weights "
                        "as pre-rotated. Reconvert with the current converter to enable ConvRot."
                    )
                    warned_unrotated_convrot = True
                quant_conf["convrot"] = False
                quant_conf.pop("convrot_groupsize", None)
            elif quant_conf.get("convrot"):
                groupsize = quant_conf.get("convrot_groupsize", 256)
                weight_shape = weight_ggml.shape if dynamic else weight_ggml.tensor_shape
                if weight_shape[-1] % groupsize != 0:
                    logging.warning(
                        "Disabling ConvRot for %s because %d input features are not "
                        "divisible by group size %d.",
                        weight_key,
                        weight_shape[-1],
                        groupsize,
                    )
                    quant_conf["convrot"] = False
                    quant_conf.pop("convrot_groupsize", None)

            # Convert to ComfyUI native state-dict layout
            if dynamic:
                weight = weight_ggml.view(torch.int8).reshape(weight_ggml.shape)
                scale = scale_ggml.view(torch.float32).reshape(scale_ggml.shape)
            else:
                weight = weight_ggml.data.view(torch.int8).reshape(weight_ggml.tensor_shape)
                scale = scale_ggml.data.view(torch.float32).reshape(scale_ggml.tensor_shape)

            state_dict[weight_key] = torch.nn.Parameter(weight, requires_grad=False)
            state_dict[scale_key] = torch.nn.Parameter(scale, requires_grad=False)

            layer_prefix = weight_key[:weight_key.rfind("weight")]
            quant_json = json.dumps(quant_conf)
            state_dict[f"{layer_prefix}comfy_quant"] = torch.nn.Parameter(
                torch.tensor(list(quant_json.encode("utf-8")), dtype=torch.uint8),
                requires_grad=False,
            )
            extra["gguf_quant_mode"] = "int8_convrot"

        # Retained loader adaptation for the retired Q4_PT experiment. The
        # explicit rejection above prevents this branch from being executed.
        elif fmt in {"int4_compact_gemm", "int4_pytorch"}:
            # Keep compact INT4 storage while exposing the original shape to
            # ComfyUI's model detector. It uses first.weight.shape to infer
            # Krea2's latent channel count before custom ops load the tensor.
            orig_shape = torch.Size(tuple(quant_conf["orig_shape"]))
            weight = GGMLTensor(
                weight_ggml.data.view(torch.uint8).reshape(weight_ggml.tensor_shape),
                tensor_type=weight_ggml.tensor_type,
                tensor_shape=orig_shape,
            )
            scale = scale_ggml.data.view(torch.float32).reshape(scale_ggml.tensor_shape)

            state_dict[weight_key] = torch.nn.Parameter(weight, requires_grad=False)
            state_dict[scale_key] = torch.nn.Parameter(scale, requires_grad=False)
            layer_prefix = weight_key[:weight_key.rfind("weight")]
            state_dict[f"{layer_prefix}comfy_quant"] = torch.nn.Parameter(
                torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8),
                requires_grad=False,
            )
            extra["gguf_quant_mode"] = "int4_pytorch"

    if direct_reader is not None:
        direct_reader.close()
    return (state_dict, extra)

# for remapping llama.cpp -> original key names
T5_SD_MAP = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

GEMMA3_SD_MAP = LLAMA_SD_MAP.copy()
GEMMA3_SD_MAP.update({
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "post_attention_norm": "post_attention_layernorm",
})

CLIP_VISION_SD_MAP = {
    "mm.": "visual.merger.mlp.",
    "v.post_ln.": "visual.merger.ln_q.",
    "v.patch_embd": "visual.patch_embed.proj",
    "v.blk.": "visual.blocks.",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k,v in raw_sd.items():
        for s,d in key_map.items():
            k = k.replace(s,d)
        sd[k] = v
    return sd

def llama_permute(raw_sd, n_head, n_head_kv):
    # Reverse version of LlamaModel.permute in llama.cpp convert script
    sd = {}
    permute = lambda x,h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)
    for k,v in raw_sd.items():
        if k.endswith(("q_proj.weight", "q_proj.bias")):
            v.data = permute(v.data, n_head)
        if k.endswith(("k_proj.weight", "k_proj.bias")):
            v.data = permute(v.data, n_head_kv)
        sd[k] = v
    return sd

def gemma3_norm_corrections(sd):
    # Reverse change from Gemma3Model modify_tensors in llama.cpp convert script
    norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "model.norm.weight"
    ]
    corrected = 0
    for key in list(sd.keys()):
        if any(p in key for p in norm_patterns):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=torch.float32) - 1.0
            else:
                sd[key] = sd[key].float() - 1.0
            corrected += 1
    #logging.info(f"Gemma3: Applied -1 norm correction to {corrected} tensors")
    return sd

def strip_quant_suffix(name):
    pattern = r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        name = name[:match.start()]
    return name

def _merge_qwen3_patch_embedding(vsd, path, dynamic):
    """Join llama.cpp's two temporal patch slices without a CPU copy.

    Qwen3-VL mmproj files store ``v.patch_embd.weight`` and
    ``v.patch_embd.weight.1`` as two adjacent [H, W, C, O] GGUF tensors.  The
    ComfyUI Conv3d expects one [O, C, T, H, W] tensor.  ``torch.stack`` is
    harmless for an ordinary CPU state dict but drops the file-slice metadata
    on a meta tensor, which would later turn into a ``Cannot copy out of meta``
    error.  When DynamicVRAM is active, make one meta view and attach a single
    contiguous disk slice covering both source tensors.  This keeps the host
    RAM bound and preserves resident materialisation semantics.
    """
    first_key = "v.patch_embd.weight"
    second_key = "v.patch_embd.weight.1"
    if second_key not in vsd:
        return

    first = vsd.pop(first_key)
    second = vsd.pop(second_key)
    if tuple(first.shape) != tuple(second.shape) or first.ndim != 4:
        raise ValueError(
            "Qwen3-VL mmproj temporal patch slices have incompatible shapes: "
            f"{tuple(first.shape)} and {tuple(second.shape)}"
        )

    shape = (first.shape[0], first.shape[1], 2, first.shape[2], first.shape[3])
    if not dynamic:
        # The no-host-mmap deployment rejects this path before it reaches
        # here, but retain the ordinary loader behaviour for normal GGUF use.
        vsd[first_key] = torch.stack([first, second], dim=2)
        return

    reader = NoMmapGGUFReader(path)
    entries = {tensor.name: tensor for tensor in reader.tensors}
    first_info = entries.get(first_key)
    second_info = entries.get(second_key)
    if first_info is None or second_info is None:
        raise KeyError("Qwen3-VL mmproj patch embedding offsets are missing")
    first_end = first_info.data_offset + first_info.n_bytes
    if first_end != second_info.data_offset:
        raise RuntimeError(
            "Qwen3-VL mmproj temporal patch slices are not contiguous; "
            "refusing to create an unbounded staging copy"
        )

    file_slice = dynamic_gguf_file_slice(path)
    if file_slice is None:
        raise RuntimeError(
            "Qwen3-VL mmproj requires DynamicVRAM for no-host-mmap loading"
        )
    combined = torch.empty(shape, dtype=first.dtype, device="meta")
    attach_dynamic_file_slice(
        combined,
        file_slice[0],
        file_slice[1],
        first_info.data_offset,
        first_info.n_bytes + second_info.n_bytes,
    )
    vsd[first_key] = combined


def _map_qwen3_mmproj(vsd):
    """Map a Qwen3-VL llama.cpp mmproj into ComfyUI's Qwen3-VL names."""
    mapped = {}
    deepstack_indices = {5: 0, 11: 1, 17: 2}
    for key, value in vsd.items():
        if key.startswith("v.blk."):
            key = "model.visual.blocks." + key[len("v.blk."):]
            key = key.replace(".attn_out.", ".attn.proj.")
            key = key.replace(".attn_qkv.", ".attn.qkv.")
            key = key.replace(".ffn_up.", ".mlp.linear_fc1.")
            key = key.replace(".ffn_down.", ".mlp.linear_fc2.")
            key = key.replace(".ln1.", ".norm1.")
            key = key.replace(".ln2.", ".norm2.")
        elif key.startswith("v.deepstack."):
            parts = key.split(".", 4)
            source_index = int(parts[2])
            if source_index not in deepstack_indices:
                raise ValueError(f"Unexpected Qwen3-VL DeepStack index: {source_index}")
            suffix = parts[3] + "." + parts[4]
            suffix = suffix.replace("fc1.", "linear_fc1.")
            suffix = suffix.replace("fc2.", "linear_fc2.")
            key = (
                "model.visual.deepstack_merger_list."
                f"{deepstack_indices[source_index]}.{suffix}"
            )
        elif key.startswith("v.post_ln."):
            key = "model.visual.merger.norm." + key[len("v.post_ln."):]
        elif key.startswith("v.patch_embd."):
            key = "model.visual.patch_embed.proj." + key[len("v.patch_embd."):]
        elif key.startswith("v.position_embd."):
            key = "model.visual.pos_embed." + key[len("v.position_embd."):]
        elif key.startswith("mm.0."):
            key = "model.visual.merger.linear_fc1." + key[len("mm.0."):]
        elif key.startswith("mm.2."):
            key = "model.visual.merger.linear_fc2." + key[len("mm.2."):]
        else:
            raise ValueError(f"Unexpected Qwen3-VL mmproj tensor: {key}")
        mapped[key] = value
    return mapped


def gguf_mmproj_loader(path, dynamic=False, direct_device=None):
    # Reverse version of Qwen2VLVisionModel.modify_tensors
    logging.info("Attenpting to find mmproj file for text encoder...")

    # get name to match w/o quant suffix
    tenc_fname = os.path.basename(path)
    tenc = os.path.splitext(tenc_fname)[0].lower()
    tenc = strip_quant_suffix(tenc)

    # try and find matching mmproj
    target = []
    root = os.path.dirname(path)
    for fname in os.listdir(root):
        name, ext = os.path.splitext(fname)
        if ext.lower() != ".gguf":
            continue
        if "mmproj" not in name.lower():
            continue
        if tenc in name.lower():
            target.append(fname)

    if len(target) == 0:
        logging.error(f"Error: Can't find mmproj file for '{tenc_fname}' (matching:'{tenc}')! Qwen-Image-Edit will be broken!")
        return {}
    if len(target) > 1:
        logging.error(f"Ambiguous mmproj for text encoder '{tenc_fname}', will use first match.")

    logging.info(f"Using mmproj '{target[0]}' for text encoder '{tenc_fname}'.")
    target = os.path.join(root, target[0])
    vsd, _ = gguf_sd_loader(
        target,
        is_text_model=True,
        dynamic=dynamic,
        direct_device=direct_device,
    )

    # Qwen3-VL uses a different vision naming/layout from Qwen2-VL.  In
    # particular it has three DeepStack mergers at layers 5/11/17 and its
    # mmproj stores a combined QKV matrix, so applying CLIP_VISION_SD_MAP here
    # would silently create missing weights and break reference-image input.
    qwen3 = any(key.startswith("v.deepstack.") for key in vsd)
    if qwen3:
        _merge_qwen3_patch_embedding(vsd, target, dynamic)
        return _map_qwen3_mmproj(vsd)

    # concat 4D to 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    # run main replacement
    vsd = sd_map_replace(vsd, CLIP_VISION_SD_MAP)

    # handle split Q/K/V
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        # filter out attentions + group
        for k,v in vsd.items():
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v) else torch.float16)
                )

        # recombine
        for k,v in attns.items():
            suffix = k.split(".")[-1]
            vsd[k] = torch.cat([
                v[f"q.{suffix}"],
                v[f"k.{suffix}"],
                v[f"v.{suffix}"],
            ], dim=0)
        del attns

    return vsd

def gguf_tokenizer_loader(path, temb_shape):
    # convert gguf tokenizer to spiece
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please make sure sentencepiece and protobuf are installed.\npip install sentencepiece protobuf")
    spm = model.ModelProto()

    reader = _gguf_reader(path)

    if get_field(reader, "tokenizer.ggml.model", str) == "t5":
        if temb_shape == (256384, 4096): # probably UMT5
            spm.trainer_spec.model_type == 1 # Unigram (do we have a T5 w/ BPE?)
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    spm.normalizer_spec.add_dummy_prefix = get_field(reader, "tokenizer.ggml.add_space_prefix", bool)
    spm.normalizer_spec.remove_extra_whitespaces = get_field(reader, "tokenizer.ggml.remove_extra_whitespaces", bool)

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    for idx, (token, score, toktype) in enumerate(zip(tokens, scores, toktypes)):
        # # These aren't present in the original?
        # if toktype == 5 and idx >= temb_shape[0]%1000):
        #     continue

        piece = spm.SentencePiece()
        piece.piece = token
        piece.score = score
        piece.type = toktype
        spm.pieces.append(piece)

    # unsure if any of these are correct
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.vocab_size = len(tokens) # split off unused?
    spm.trainer_spec.max_sentence_length = 4096
    spm.trainer_spec.eos_id = get_field(reader, "tokenizer.ggml.eos_token_id", int)
    spm.trainer_spec.pad_id = get_field(reader, "tokenizer.ggml.padding_token_id", int)

    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")
    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def gguf_tekken_tokenizer_loader(path, temb_shape):
    # convert ggml (hf) tokenizer metadata to tekken/comfy data
    logging.info("Attempting to recreate tekken tokenizer from GGUF file metadata...")
    import json
    import base64
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    reader = _gguf_reader(path)

    model_str = get_field(reader, "tokenizer.ggml.model", str)
    if model_str == "gpt2":
        if temb_shape == (131072, 5120): # probably Mistral
            data = {
                "config": {"num_vocab_tokens": 150000, "default_vocab_size": 131072},
                "vocab": [],
                "special_tokens": [],
            }
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    decoder = {v: k for k, v in bytes_to_unicode().items()}
    for idx, (token, toktype) in enumerate(zip(tokens, toktypes)):
        if toktype == 3:
            data["special_tokens"].append(
                {'rank': idx, 'token_str': token, 'is_control': True}
            )
        else:
            tok = bytes([decoder[char] for char in token])
            data["vocab"].append({
                "rank": len(data["vocab"]),
                "token_bytes": base64.b64encode(tok).decode("ascii"),
                "token_str": tok.decode("utf-8", errors="replace") # ?
            })

    logging.info(f"Created tekken tokenizer with vocab size of {len(data['vocab'])} (+{len(data['special_tokens'])})")
    del reader
    return torch.ByteTensor(list(json.dumps(data).encode('utf-8')))

def gguf_gemma3_tokenizer_loader(path):
    #TODO: merge into gguf_tokenizer_loader
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please install sentencepiece and protobuf.\npip install sentencepiece protobuf")
    spm = model.ModelProto()
    reader = _gguf_reader(path)

    spm.normalizer_spec.name = "identity"
    spm.normalizer_spec.add_dummy_prefix = False
    spm.trainer_spec.model_type = 2
    spm.trainer_spec.input_format = "tsv"
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.max_sentence_length = 4192
    spm.trainer_spec.bos_piece = "<bos>"

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktype = get_list_field(reader, "tokenizer.ggml.token_type", int)

    if not tokens or not scores or not toktype:
        raise ValueError("Missing tokenizer metadata")

    for idx in range(len(tokens)):
        piece = spm.SentencePiece()
        piece.piece = tokens[idx]
        if idx == 3:  # UNK position
            piece.type = 2  # UNK Token
            piece.score = 0.0 # UNK Score
        else:
            piece.type = toktype[idx]
            piece.score = scores[idx]
        spm.pieces.append(piece)

    spm.trainer_spec.vocab_size = len(spm.pieces)
    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")

    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def inject_qwen3vl_detection_markers(sd):
    """Add visual sentinels when a llama.cpp Qwen3-VL GGUF excludes its vision tower."""
    ln_key = "model.layers.0.input_layernorm.weight"
    lm_hidden = int(sd[ln_key].shape[0]) if ln_key in sd else 2560
    vis_hidden = 1024 if lm_hidden == 2560 else 1152
    merge_dim = vis_hidden * 4  # spatial_merge_size=2

    if lm_hidden == 5120:
        # MiniMax H3 uses the truncated Qwen3-VL-32B encoder. Its detector
        # deliberately checks this unprefixed visual key plus layer 49.
        marker_key = "visual.deepstack_merger_list.0.norm.weight"
    else:
        marker_key = "model.visual.deepstack_merger_list.0.norm.weight"

    sd[marker_key] = torch.zeros(merge_dim)
    if lm_hidden != 5120:
        sd["model.visual.merger.linear_fc2.weight"] = torch.zeros(lm_hidden, merge_dim)
    logging.info(
        "qwen3vl GGUF: injected visual marker tensor "
        "(lm_hidden=%d, merge_dim=%d)",
        lm_hidden,
        merge_dim,
    )

def gguf_clip_loader(path, dynamic=False, progress_callback=None, direct_device=None):
    sd, extra = gguf_sd_loader(
        path,
        is_text_model=True,
        dynamic=dynamic,
        progress_callback=progress_callback,
        direct_device=direct_device,
    )
    arch = extra.get("arch_str", None)
    if arch in {"t5", "t5encoder"}:
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape == (256384, 4096):
            # non-standard Comfy-Org tokenizer
            sd["spiece_model"] = gguf_tokenizer_loader(path, sd[temb_key].shape)
            # TODO: dequantizing token embed here is janky but otherwise we OOM due to tensor being massive.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in {"llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3"}:
        # TODO: pass model_options["vocab_size"] to loader somehow
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape[0] >= (64 * 1024):
            if arch == "llama" and sd[temb_key].shape == (131072, 5120):
                # non-standard Comfy-Org tokenizer
                sd["tekken_model"] = gguf_tekken_tokenizer_loader(path, sd[temb_key].shape)
            elif arch == "gemma3":
                sd["spiece_model"] = gguf_gemma3_tokenizer_loader(path)
            # See note above for T5.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        if arch == "gemma3":
            sd = sd_map_replace(sd, GEMMA3_SD_MAP)
            sd = gemma3_norm_corrections(sd)
        else:
            sd = sd_map_replace(sd, LLAMA_SD_MAP)
        if arch == "llama":
            sd = llama_permute(sd, 32, 8) # L3 / Mistral
        if arch in {"qwen2vl", "qwen3vl"}:
            vsd = gguf_mmproj_loader(
                path,
                dynamic=dynamic,
                direct_device=direct_device,
            )
            sd.update(vsd)
        if arch == "qwen3vl" and "model.visual.deepstack_merger_list.0.norm.weight" not in sd:
            # Standard llama.cpp Qwen3-VL GGUFs omit the visual tower. Without it,
            # detect_te_model() mis-classifies the state dict as a Qwen3 LM instead
            # of Qwen3-VL. MiniMax H3 additionally uses Qwen3-VL-32B truncated to
            # 50 layers, whose detector relies on an unprefixed visual marker.
            # Inject zero sentinel tensors with shapes that exactly match the model
            # parameters so that load_state_dict(strict=False) doesn't raise a size
            # mismatch error while still satisfying detect_te_model()'s key checks.
            inject_qwen3vl_detection_markers(sd)
    elif arch == "ideogram":
        # Dequantize Ideogram model for inference
        logging.info("Dequantizing Ideogram model for inference...")
        # Use BF16 to save VRAM while maintaining quality, but fall back to FP16
        # on devices that don't support bf16 (avoids slow fp32 compute fallback).
        target_dtype = torch.bfloat16 if device_supports_bf16() else torch.float16
        dequantized_count = 0
        for key in list(sd.keys()):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=target_dtype)
                dequantized_count += 1
        logging.info(f"Dequantized {dequantized_count} tensors for Ideogram model ({target_dtype})")
    else:
        pass
    return sd
