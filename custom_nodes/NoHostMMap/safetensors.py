"""Header-only safetensors loader for ComfyUI's AIMDO path."""

from __future__ import annotations

import json
import os
import struct
import threading
import ctypes

import torch


def _is_file_backed_quantized(value):
    """Return whether *value* is a QuantizedTensor with a disk-backed qdata.

    GGUF Q4/Q5 state-dict values are wrapper tensors: their logical shape and
    dtype describe the dequantized matrix, while the bytes on disk live in
    ``value._qdata``. Treating the wrapper as an ordinary tensor would try to
    allocate the full FP16 matrix and then call ``meta.to(device)``. That is
    both the wrong byte range and the exact ``Cannot copy out of meta`` failure
    seen when moving a resident Qwen layer.
    """
    qdata = getattr(value, "_qdata", None)
    if qdata is None or not isinstance(qdata, torch.Tensor):
        return False
    return getattr(
        qdata.untyped_storage(), "_comfy_tensor_file_slice", None
    ) is not None


# These tensors are consumed directly by ComfyUI's quantized module loader.
# They are all small auxiliary values (the 4B INT8 encoder's complete set is
# only a few MiB), unlike ``weight`` tensors which must remain disk-backed.
_CPU_AUX_SUFFIXES = (
    ".comfy_quant",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".pre_quant_scale",
    ".weight_s_rel",
    ".weight_s_channel",
    ".weight_codebook",
    ".bias",
)


_INTXLNK_MAGIC = b"IntxLNK\x01"
_INTXLNK_MAX_PAYLOAD_BYTES = 4096
_MAX_SAFETENSORS_HEADER_BYTES = 64 << 20


def resolve_no_host_path(path):
    """Resolve ordinary symlinks and bounded ``IntxLNK`` link files.

    Some model stores use a tiny regular file instead of a POSIX symlink.  Its
    first eight bytes are ``IntxLNK\x01`` and the rest is an UTF-16LE target
    path.  Passing that file to safetensors makes the marker bytes look like a
    little-endian header size, which can request an absurd allocation before
    any useful validation happens.  Read only a few KB here and never read a
    model payload.
    """
    current = os.path.realpath(os.fspath(path))
    seen = set()
    for _ in range(8):
        if current in seen:
            raise ValueError(f"cyclic IntxLNK path: {path}")
        seen.add(current)
        if not os.path.isfile(current):
            raise FileNotFoundError(f"model path is not a regular file: {current}")

        with open(current, "rb", buffering=0) as handle:
            if handle.read(len(_INTXLNK_MAGIC)) != _INTXLNK_MAGIC:
                return current
            payload = handle.read(_INTXLNK_MAX_PAYLOAD_BYTES + 1)

        if len(payload) > _INTXLNK_MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"IntxLNK target exceeds {_INTXLNK_MAX_PAYLOAD_BYTES} bytes: {current}"
            )
        if len(payload) == 0 or len(payload) % 2:
            raise ValueError(f"invalid UTF-16LE IntxLNK target: {current}")
        try:
            target = payload.decode("utf-16le").rstrip("\x00")
        except UnicodeDecodeError as error:
            raise ValueError(f"invalid UTF-16LE IntxLNK target: {current}") from error
        if target.startswith("\ufeff"):
            target = target[1:]
        if not target or "\x00" in target:
            raise ValueError(f"empty or malformed IntxLNK target: {current}")
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(current), target)
        current = os.path.realpath(os.path.normpath(target))

    raise ValueError(f"too many nested IntxLNK files: {path}")


def _read_header(ckpt):
    """Read only the safetensors header and return its data base offset."""
    ckpt = resolve_no_host_path(ckpt)
    file_ref = open(ckpt, "rb", buffering=0)
    try:
        file_size = os.fstat(file_ref.fileno()).st_size
        prefix = file_ref.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Incomplete safetensors header: {ckpt}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"Safetensors header is unreasonably large ({header_size} bytes): {ckpt}"
            )
        if header_size > max(0, file_size - 8):
            raise ValueError(
                f"Safetensors header exceeds file size ({header_size} bytes): {ckpt}"
            )
        header_bytes = file_ref.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"Incomplete safetensors header: {ckpt}")
        return file_ref, json.loads(header_bytes.decode("utf-8")), 8 + header_size
    except Exception:
        file_ref.close()
        raise


def _read_into(file_ref, offset, tensor):
    """Fill a CPU tensor from a file descriptor without an intermediate blob."""
    size = tensor.numel() * tensor.element_size()
    if size == 0:
        return
    view_type = ctypes.c_ubyte * size
    view = memoryview(view_type.from_address(tensor.data_ptr()))
    try:
        file_ref.seek(offset)
        done = 0
        while done < size:
            count = file_ref.readinto(view[done:])
            if count is None or count <= 0:
                raise OSError(f"Short read while loading safetensors tensor at {offset}")
            done += count
    finally:
        view.release()


def _file_slice_tensor_to_device(tensor, device, dtype=None):
    """Read one disk-backed tensor to ``device`` without a host-sized copy."""

    info = getattr(tensor.untyped_storage(), "_comfy_tensor_file_slice", None)
    if info is None:
        return None

    # The bytes in a safetensors slice always have ``tensor.dtype``.  Loading
    # them straight into a same-width requested dtype is a bit reinterpretation,
    # not a numeric cast (most importantly BF16 bytes interpreted as FP16).  It
    # stays finite but silently destroys LoRA values.  Materialize the on-disk
    # dtype first, then let torch perform the requested conversion.
    source_dtype = tensor.dtype
    target_dtype = dtype or source_dtype
    source = torch.empty(tensor.shape, dtype=source_dtype, device=device)
    if source.numel() * source.element_size() != info.size:
        raise ValueError(
            "File-backed tensor byte size does not match its source dtype: "
            f"{tensor.shape} {source_dtype} ({info.size} bytes)"
        )

    if device.type == "cpu":
        with info.lock:
            _read_into(info.file_ref, info.offset, source)
        return source if target_dtype == source_dtype else source.to(dtype=target_dtype)

    # AIMDO's no-stream host-buffer call has returned success after a failed
    # device copy on V100.  Use a bounded ordinary-read path here instead: it
    # is slower, but every byte is checked and host RAM is capped at 8 MiB.
    source_bytes = source.view(torch.uint8).reshape(-1)
    chunk_bytes = min(8 * 1024 * 1024, info.size)
    staging = torch.empty((chunk_bytes,), dtype=torch.uint8, device="cpu")
    with info.lock:
        offset = 0
        while offset < info.size:
            count = min(chunk_bytes, info.size - offset)
            _read_into(info.file_ref, info.offset + offset, staging[:count])
            source_bytes[offset:offset + count].copy_(staging[:count])
            offset += count
    return source if target_dtype == source_dtype else source.to(dtype=target_dtype)


def _file_slice_value_to_device(value, device, dtype=None):
    """Materialize an ordinary or GGML-quantized disk-backed tensor.

    Ordinary safetensors values can be read directly using their own shape and
    dtype. A GGUF ``QuantizedTensor`` must instead read only ``_qdata`` and
    rebuild the lightweight wrapper around the device-resident raw bytes. No
    dequantized copy is created here; the existing GGML dispatch/dequant path
    remains responsible for the short-lived compute matrix.
    """
    if not _is_file_backed_quantized(value):
        return _file_slice_tensor_to_device(value, device, dtype)

    qdata = value._qdata
    loaded_qdata = _file_slice_tensor_to_device(qdata, device, qdata.dtype)
    if loaded_qdata is None:
        return None

    # Avoid importing comfy.quant_ops at module import time: the safe reader is
    # also used by small standalone header tests before ComfyUI finishes its
    # custom-node discovery.
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        QuantizedTensor = type(value)
    if not isinstance(value, QuantizedTensor):
        raise TypeError(
            "file-backed _qdata belongs to an unsupported quantized tensor: "
            f"{type(value)!r}"
        )
    return QuantizedTensor(loaded_qdata, value._layout_cls, value._params)


def _wrap_quantized_loader(original):
    """Make ComfyUI's quantized loader understand disk-backed meta weights."""
    import comfy.ops

    def load_quantized_module(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        weight_key = f"{prefix}weight"
        weight = state_dict.get(weight_key)
        quant_conf = state_dict.get(f"{prefix}comfy_quant")
        if weight is not None and getattr(weight, "is_meta", False):
            info = getattr(weight.untyped_storage(), "_comfy_tensor_file_slice", None)
            if info is not None:
                device = module.factory_kwargs["device"]
                try:
                    if device is not None:
                        target_dtype = weight.dtype
                        if quant_conf is not None:
                            config = json.loads(quant_conf.detach().cpu().numpy().tobytes())
                            target_dtype = comfy.ops.QUANT_ALGOS[config["format"]]["storage_t"]
                        state_dict[weight_key] = _file_slice_tensor_to_device(
                            weight, torch.device(device), target_dtype
                        )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    if quant_conf is not None:
                        # Let the original loader produce its normal diagnostic
                        # for malformed/unknown quantization configurations.
                        pass
                    else:
                        raise
        return original(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )

    load_quantized_module._h3_no_host_mmap = True
    load_quantized_module._h3_original = original
    return load_quantized_module


def _materialize_file_backed_model(model, device):
    """Materialize a resident model one file-backed parameter at a time."""
    import comfy.utils

    parameters = list(model.named_parameters())
    materialized = 0
    for name, parameter in parameters:
        if not getattr(parameter, "is_meta", False):
            continue
        if getattr(parameter.untyped_storage(), "_comfy_tensor_file_slice", None) is None:
            continue
        loaded = _file_slice_value_to_device(parameter, device, parameter.dtype)
        comfy.utils.set_attr_param(model, name, loaded)
        materialized += 1

    materialized += _materialize_file_backed_buffers(model, device)
    return materialized


def _materialize_file_backed_buffers(model, device):
    """Materialize only file-backed buffers for any model, one at a time."""
    import comfy.utils

    buffers = list(model.named_buffers())
    materialized = 0
    for name, buffer in buffers:
        if not getattr(buffer, "is_meta", False):
            continue
        if getattr(buffer.untyped_storage(), "_comfy_tensor_file_slice", None) is None:
            continue
        loaded = _file_slice_tensor_to_device(buffer, device, buffer.dtype)
        comfy.utils.set_attr_buffer(model, name, loaded)
        materialized += 1
    return materialized


def _materialize_default_file_backed_params(model, device):
    """Materialize file-backed parameters on modules with nested children.

    DynamicVRAM's ``_load_list`` normally moves these parameters with a plain
    ``param.data.to(device)`` before it builds its per-leaf loading list.  A
    no-host-mmap safetensors reader intentionally represents the parameter as
    a meta tensor plus a disk slice, so that operation has no data to copy.
    Read only these non-leaf parameters directly to the target device.  The
    large leaf weights remain on the normal DynamicVRAM path and the reader's
    8 MiB staging bound still applies.
    """
    import comfy.utils

    target_device = torch.device(device)
    materialized = 0
    for module_name, module in model.named_modules():
        params = dict(module.named_parameters(recurse=False))
        if not params:
            continue

        # Match ModelPatcher._load_list's definition of a non-leaf/default
        # module exactly: it owns parameters and also contains descendants
        # with parameters.  Only this branch performs the unsafe direct .to.
        default = any(
            name not in params
            for name, _ in module.named_parameters(recurse=True)
        )
        if not default:
            continue

        for param_name, parameter in params.items():
            if not getattr(parameter, "is_meta", False):
                continue
            if getattr(parameter.untyped_storage(), "_comfy_tensor_file_slice", None) is None:
                continue

            loaded = _file_slice_value_to_device(parameter, target_device, parameter.dtype)
            model_dtype = getattr(module, param_name + "_comfy_model_dtype", None)
            if model_dtype is not None and loaded.dtype != model_dtype:
                loaded = loaded.to(dtype=model_dtype)
            key = f"{module_name}.{param_name}" if module_name else param_name
            comfy.utils.set_attr_param(model, key, loaded)
            materialized += 1

    return materialized


def _wrap_model_patcher_load_list(original):
    """Make DynamicVRAM's pre-list parameter move disk-slice aware."""

    def load_list(self, for_dynamic=False, default_device=None):
        if default_device is not None and getattr(self, "model", None) is not None:
            count = _materialize_default_file_backed_params(self.model, default_device)
            if count:
                import logging

                logging.info(
                    "[NoHostMMap] materialized %d non-leaf safetensors parameters on %s",
                    count,
                    default_device,
                )
        return original(self, for_dynamic=for_dynamic, default_device=default_device)

    load_list._h3_no_host_mmap = True
    load_list._h3_original = original
    return load_list


def _wrap_model_patcher_load(original):
    """Give resident Clip patchers a disk-slice-aware full-load path."""
    def load(self, *args, **kwargs):
        # ClipProj's resident mode intentionally keeps the text encoder on its
        # selected card.  Pre-fetch only this marked class of model; applying
        # the same policy to a DiT would defeat DynamicVRAM's normal paging.
        if (
            getattr(self, "is_clip", False)
            and self.load_device == self.offload_device
            and getattr(self, "model", None) is not None
        ):
            _materialize_file_backed_model(self.model, self.load_device)
        elif getattr(self, "model", None) is not None:
            # GGUF/other DynamicVRAM models may expose small file-backed
            # buffers (RoPE tables, lookup tables, etc.) through the regular
            # buffer loop.  Materialize only those; their large weights stay
            # on the normal dynamic path.
            _materialize_file_backed_buffers(self.model, self.load_device)
        return original(self, *args, **kwargs)

    load._h3_no_host_mmap = True
    load._h3_original = original
    return load


def _wrap_model_patcher_patch_weight(original):
    """Handle a later patch request for a still disk-backed meta parameter."""
    import comfy.model_patcher
    import comfy.utils

    def patch_weight_to_device(self, key, *args, **kwargs):
        weight, _, _ = comfy.model_patcher.get_key_weight(self.model, key)
        if getattr(weight, "is_meta", False):
            info = getattr(weight.untyped_storage(), "_comfy_tensor_file_slice", None)
            if info is not None and self.offload_device is not None:
                loaded = _file_slice_value_to_device(
                    weight, torch.device(self.offload_device), weight.dtype
                )
                comfy.utils.set_attr_param(self.model, key, loaded)
        return original(self, key, *args, **kwargs)

    patch_weight_to_device._h3_no_host_mmap = True
    patch_weight_to_device._h3_original = original
    return patch_weight_to_device


def load_safetensors_no_mmap(ckpt):
    """Return meta tensors backed by ordinary file-offset descriptors."""
    import comfy.memory_management
    import comfy.utils

    file_ref, header, data_base_offset = _read_header(ckpt)
    try:
        file_lock = threading.Lock()
        state_dict = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            # ComfyUI's quantized module loader consumes these small auxiliary
            # tensors directly (JSON metadata via ``.numpy()``, scales/biases
            # via ``.to(device)``).  They therefore cannot stay meta-backed.
            # Materialize only this bounded set; model weights and ordinary
            # tensors keep the disk-backed TensorFileSlice path.
            if name.endswith(_CPU_AUX_SUFFIXES):
                tensor = torch.empty(
                    info["shape"],
                    dtype=comfy.utils._TYPES[info["dtype"]],
                    device="cpu",
                )
                expected = tensor.numel() * tensor.element_size()
                if expected != end - start:
                    raise ValueError(f"safetensors tensor size mismatch for {name!r}: {ckpt}")
                _read_into(file_ref, data_base_offset + start, tensor)
                state_dict[name] = tensor
                continue
            tensor = torch.empty(
                info["shape"],
                dtype=comfy.utils._TYPES[info["dtype"]],
                device="meta",
            )
            if start == end:
                state_dict[name] = tensor
                continue
            if tensor.numel() * tensor.element_size() != end - start:
                raise ValueError(f"safetensors tensor size mismatch for {name!r}: {ckpt}")
            storage = tensor.untyped_storage()
            storage._comfy_tensor_file_slice = comfy.memory_management.TensorFileSlice(
                file_ref, file_lock, data_base_offset + start, end - start
            )
            # A storage attribute keeps the descriptor open without retaining
            # any file mapping or a copy of the payload.
            storage._comfy_tensor_file_refs = (file_ref,)
            state_dict[name] = tensor
        return state_dict, header.get("__metadata__", {})
    except Exception:
        file_ref.close()
        raise


def load_safetensors_values_no_mmap(ckpt):
    """Read safetensors values with bounded, ordinary file reads.

    This is for small auxiliary files such as ClipProj matrices.  Model
    checkpoints use :func:`load_safetensors_no_mmap` so their payloads remain
    disk-backed until DynamicVRAM transfers one tensor to a device.  Keeping
    this separate makes it explicit that a projection cache may consume its
    own (bounded) CPU size, but never creates a file mapping.
    """
    import comfy.utils

    file_ref, header, data_base_offset = _read_header(ckpt)
    try:
        state_dict = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            tensor = torch.empty(
                info["shape"],
                dtype=comfy.utils._TYPES[info["dtype"]],
                device="cpu",
            )
            expected = tensor.numel() * tensor.element_size()
            if expected != end - start:
                raise ValueError(f"safetensors tensor size mismatch for {name!r}: {ckpt}")
            _read_into(file_ref, data_base_offset + start, tensor)
            state_dict[name] = tensor
        return state_dict, header.get("__metadata__", {})
    except Exception:
        raise
    finally:
        file_ref.close()


def _is_safetensors(path):
    path = os.fspath(path).lower()
    return path.endswith(".safetensors") or path.endswith(".sft")


def _wrap_load_torch_file(original):
    def load_torch_file_no_mmap(
        ckpt, safe_load=False, device=None, return_metadata=False
    ):
        if _is_safetensors(ckpt):
            # The Turbo node's bundled interpolation grid is a small auxiliary
            # tensor, not a model checkpoint.  Read it normally so its later
            # ``.to(cuda)`` does not encounter a meta tensor.
            if os.fspath(ckpt).lower().endswith("h3_silu_temb_grid.safetensors"):
                state_dict, metadata = load_safetensors_values_no_mmap(ckpt)
                return (state_dict, metadata) if return_metadata else state_dict
            state_dict, metadata = load_safetensors_no_mmap(ckpt)
            return (state_dict, metadata) if return_metadata else state_dict
        return original(
            ckpt,
            safe_load=safe_load,
            device=device,
            return_metadata=return_metadata,
        )

    load_torch_file_no_mmap._h3_no_host_mmap = True
    load_torch_file_no_mmap._h3_original = original
    return load_torch_file_no_mmap


def _wrap_bypass_adapter_move(original):
    """Load LoRA bypass tensors in bounded chunks when they are disk-backed."""
    def move_adapter_weights(self, device, dtype=None):
        adapter = self.adapter
        weights = getattr(adapter, "weights", None)
        if isinstance(weights, (list, tuple)):
            changed = False
            materialized = []
            target_device = torch.device(device)
            for weight in weights:
                if (
                    isinstance(weight, torch.Tensor)
                    and getattr(weight, "is_meta", False)
                    and getattr(weight.untyped_storage(), "_comfy_tensor_file_slice", None)
                    is not None
                ):
                    weight = _file_slice_tensor_to_device(
                        weight, target_device, dtype if dtype is not None else weight.dtype
                    )
                    changed = True
                materialized.append(weight)
            if changed:
                adapter.weights = tuple(materialized) if isinstance(weights, tuple) else materialized
        elif isinstance(weights, torch.Tensor):
            if (
                getattr(weights, "is_meta", False)
                and getattr(weights.untyped_storage(), "_comfy_tensor_file_slice", None)
                is not None
            ):
                adapter.weights = _file_slice_tensor_to_device(
                    weights, torch.device(device), dtype if dtype is not None else weights.dtype
                )
        return original(self, device, dtype)

    move_adapter_weights._h3_no_host_mmap = True
    move_adapter_weights._h3_original = original
    return move_adapter_weights


def install():
    """Install the safe reader even when AIMDO is unavailable.

    Current ComfyUI only calls ``load_safetensors`` on the AIMDO branch.  V100
    installations can legitimately run without that optional backend, in
    which case ``load_torch_file`` would otherwise fall back to safetensors'
    file-mapped reader and defeat the low-RAM guarantee.  Patch both entry
    points; the wrapper is idempotent and leaves non-safetensors files alone.
    """
    import comfy.utils

    comfy.utils.load_safetensors = load_safetensors_no_mmap
    original = getattr(comfy.utils, "load_torch_file", None)
    if original is not None and not getattr(original, "_h3_no_host_mmap", False):
        comfy.utils.load_torch_file = _wrap_load_torch_file(original)

    # ComfyUI's quantized Linear loader normally receives ordinary CPU tensors
    # and calls ``weight.to(cuda)`` itself.  Our no-host-mmap reader deliberately
    # supplies meta tensors with file slices, so give that one narrow loader a
    # direct per-layer transfer hook as well.
    import comfy.ops

    quantized_loader = getattr(comfy.ops, "_load_quantized_module", None)
    if quantized_loader is not None and not getattr(
        quantized_loader, "_h3_no_host_mmap", False
    ):
        comfy.ops._load_quantized_module = _wrap_quantized_loader(quantized_loader)

    import comfy.model_patcher

    # AIMDO may select ModelPatcherDynamic for the same CLIP graph.  Patch both
    # implementations; they share the same file-slice contract but have
    # separate ``load`` methods.
    patcher_classes = [comfy.model_patcher.ModelPatcher]
    dynamic_cls = getattr(comfy.model_patcher, "ModelPatcherDynamic", None)
    if dynamic_cls is not None and dynamic_cls not in patcher_classes:
        patcher_classes.append(dynamic_cls)
    for patcher_cls in patcher_classes:
        patcher_load_list = getattr(patcher_cls, "_load_list", None)
        if patcher_load_list is not None and not getattr(
            patcher_load_list, "_h3_no_host_mmap", False
        ):
            patcher_cls._load_list = _wrap_model_patcher_load_list(patcher_load_list)

        patcher_load = getattr(patcher_cls, "load", None)
        if patcher_load is not None and not getattr(
            patcher_load, "_h3_no_host_mmap", False
        ):
            patcher_cls.load = _wrap_model_patcher_load(patcher_load)

        patcher_patch = getattr(patcher_cls, "patch_weight_to_device", None)
        if patcher_patch is not None and not getattr(
            patcher_patch, "_h3_no_host_mmap", False
        ):
            patcher_cls.patch_weight_to_device = _wrap_model_patcher_patch_weight(
                patcher_patch
            )

    import comfy.weight_adapter.bypass

    bypass_hook = getattr(comfy.weight_adapter.bypass, "BypassForwardHook", None)
    bypass_move = getattr(bypass_hook, "_move_adapter_weights_to_device", None)
    if bypass_move is not None and not getattr(bypass_move, "_h3_no_host_mmap", False):
        bypass_hook._move_adapter_weights_to_device = _wrap_bypass_adapter_move(
            bypass_move
        )
