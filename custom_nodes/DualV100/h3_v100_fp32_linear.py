"""Fast FP32-output H3 linear layers for NVIDIA Volta (SM70).

MiniMax H3 needs FP32 residual updates around attention ``out_proj`` and the
MLP ``fc2`` on V100.  Passing an FP32 activation to ComfyUI-GGUF's ordinary
Linear path also asks it to dequantize the whole Q4 weight to FP32 and then
runs a CUDA-core GEMM.  That is numerically safe, but unnecessarily slow.

This module keeps the externally visible result in FP32 while using an FP16
weight/input Tensor Core GEMM with direct FP32 accumulation/output.  The MLP
input is first divided by an exact per-row power of two so values outside the
FP16 range cannot become infinities during the cast; the FP32 output is then
multiplied by the same scale.

The GGUF patch is deliberately narrow: SM70 inference, FP32 input, no merged
weight patches, no bias, and only H3's two full-width row projections.  Turbo
LoRA bypass hooks remain outside the patched base forward, so they still see
the original FP32 input and add their update in FP32 activation space.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import sys
from typing import Any

import torch


FP16_SCALE_TARGET = 32752.0

# (in_features, out_features) -> (role, scale_input)
_H3_WIDE_LINEARS = {
    (7168, 5376): ("attention_out", False),
    (14336, 5376): ("mlp_fc2", True),
}

_ORIGINAL_GGUF_FORWARDS: dict[type, Any] = {}
_ORIGINAL_H3_INITIALIZERS: dict[type, Any] = {}
_INSTALLED = False
_SEEN_ROLES: set[str] = set()
_SEEN_REJECTIONS: set[tuple[object, ...]] = set()
_HIT_COUNTS = {role: 0 for role, _scale in _H3_WIDE_LINEARS.values()}


def power_of_two_fp16_scale(
    x: torch.Tensor,
    target: float = FP16_SCALE_TARGET,
) -> torch.Tensor:
    """Return an exact per-row scale that makes finite FP32 ``x`` fit FP16.

    A power-of-two division/multiplication does not introduce an extra FP32
    rounding step for normal values.  ``target`` is below FP16 max to leave a
    safety margin at the conversion boundary.
    """

    if x.dtype != torch.float32:
        raise ValueError(f"scaled H3 wide linear expects FP32 input, got {x.dtype}")
    if x.ndim < 2:
        raise ValueError(f"scaled H3 wide linear expects at least 2D input, got {x.shape}")
    if not math.isfinite(target) or not 0.0 < target <= torch.finfo(torch.float16).max:
        raise ValueError(f"invalid FP16 scaling target {target}")

    row_max = x.detach().abs().amax(dim=-1, keepdim=True)
    ratio = (row_max / target).clamp_min_(1.0)
    return torch.exp2(torch.ceil(torch.log2(ratio)))


def tensor_core_fp32_output_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    scale_input: bool,
    target: float = FP16_SCALE_TARGET,
) -> torch.Tensor:
    """Run FP16 Tensor Core matmul and materialize its result directly as FP32."""

    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("H3 Tensor Core wide linear requires CUDA tensors")
    if x.device != weight.device:
        raise ValueError(f"input/weight device mismatch: {x.device} != {weight.device}")
    if x.dtype != torch.float32:
        raise ValueError(f"H3 Tensor Core wide input must be FP32, got {x.dtype}")
    if weight.dtype != torch.float16:
        raise ValueError(f"H3 Tensor Core wide weight must be FP16, got {weight.dtype}")
    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError(f"incompatible H3 wide linear shapes: x={x.shape}, weight={weight.shape}")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError(f"incompatible H3 wide bias shape {bias.shape}")

    input_2d = x.reshape(-1, x.shape[-1])
    scale = power_of_two_fp16_scale(input_2d, target) if scale_input else None
    if scale is None:
        tensor_core_input = input_2d.to(torch.float16)
    else:
        tensor_core_input = (input_2d / scale).to(torch.float16)

    output_2d = torch.mm(
        tensor_core_input,
        weight.t(),
        out_dtype=torch.float32,
    )
    if scale is not None:
        output_2d.mul_(scale)
    if bias is not None:
        output_2d.add_(bias.to(device=x.device, dtype=torch.float32))
    return output_2d.reshape(*x.shape[:-1], weight.shape[0])


def _has_merged_patches(layer: Any) -> bool:
    weight = getattr(layer, "weight", None)
    return bool(
        getattr(weight, "patches", ())
        or getattr(layer, "weight_function", ())
    )


def _eligible_role(layer: Any, x: torch.Tensor) -> tuple[str, bool] | None:
    if (
        not torch.is_tensor(x)
        or not x.is_cuda
        or x.dtype != torch.float32
        or x.ndim < 2
        or x.requires_grad
        or getattr(layer, "bias", None) is not None
        or _has_merged_patches(layer)
    ):
        return None
    geometry = (int(getattr(layer, "in_features", -1)), int(getattr(layer, "out_features", -1)))
    role = _H3_WIDE_LINEARS.get(geometry)
    if role is None or x.shape[-1] != geometry[0]:
        return None
    if torch.cuda.get_device_capability(x.device) != (7, 0):
        return None
    return role


def _wide_aware_gguf_forward(layer: Any, x: torch.Tensor) -> torch.Tensor:
    role_spec = _eligible_role(layer, x)
    if role_spec is None:
        geometry = (
            int(getattr(layer, "in_features", -1)),
            int(getattr(layer, "out_features", -1)),
        )
        if geometry in _H3_WIDE_LINEARS:
            rejection = (
                geometry,
                getattr(x, "dtype", None),
                getattr(x, "device", None),
                bool(getattr(x, "requires_grad", False)),
                getattr(layer, "bias", None) is not None,
                _has_merged_patches(layer),
                type(layer).__module__,
            )
            if rejection not in _SEEN_REJECTIONS:
                _SEEN_REJECTIONS.add(rejection)
                logging.info(
                    "H3 SM70 FP32 Tensor Core candidate rejected: geometry=%s "
                    "dtype=%s device=%s grad=%s bias=%s merged_patches=%s class=%s",
                    *rejection,
                )
        original = _ORIGINAL_GGUF_FORWARDS.get(type(layer))
        if original is None:
            raise RuntimeError(f"missing original GGUF forward for {type(layer)!r}")
        return original(layer, x)

    role, scale_input = role_spec
    return _execute_wide_linear(layer, x, role, scale_input)


def _execute_wide_linear(
    layer: Any,
    x: torch.Tensor,
    role: str,
    scale_input: bool,
    *,
    run_every_op: bool = False,
) -> torch.Tensor:
    if run_every_op:
        # A module-instance wrapper becomes BypassForwardHook.original_forward.
        # Calling it skips comfy.ops.Linear.forward, so advance DynamicVRAM's
        # operation dispatcher explicitly before touching the disk/VBAR weight.
        comfy_ops = importlib.import_module("comfy.ops")
        comfy_ops.run_every_op()
    else:
        comfy_ops = importlib.import_module("comfy.ops")

    # Use ComfyUI's public cast context rather than GGUFLayer.cast_bias_weight.
    # The former faults a DynamicVRAM VBAR/meta weight into its bounded CUDA
    # arena and reliably releases/offloads it after the GEMM.
    with comfy_ops.CastBiasWeightContext(
        layer,
        x,
        dtype=torch.float16,
        device=x.device,
        bias_dtype=torch.float32,
        offloadable=True,
        compute_dtype=torch.float16,
        want_requant=False,
    ) as (weight, bias):
        if isinstance(weight, comfy_ops.QuantizedTensor):
            weight = weight.dequantize()
        elif getattr(weight, "tensor_type", None) is not None:
            # Legacy GGMLTensor path outside AIMDO/VBAR.
            weight = layer.get_weight(weight, torch.float16)
        weight = weight.to(device=x.device, dtype=torch.float16)
        if weight.dtype != torch.float16:
            raise RuntimeError(f"H3 {role} dequantized to {weight.dtype}, expected FP16")
        output = tensor_core_fp32_output_linear(
            x,
            weight,
            bias,
            scale_input=scale_input,
        )
    _HIT_COUNTS[role] += 1
    if role not in _SEEN_ROLES:
        _SEEN_ROLES.add(role)
        logging.info(
            "H3 SM70 Tensor Core FP16->FP32 path active for %s (%d->%d, row_scale=%s)",
            role,
            layer.in_features,
            layer.out_features,
            scale_input,
        )
    return output


def _wrap_h3_wide_module(layer: Any, role: str, scale_input: bool) -> None:
    """Wrap a concrete H3 GGUF module before Turbo installs bypass LoRA."""

    if getattr(layer, "_h3_v100_fp32_tc_wrapped", False):
        return
    original_forward = layer.forward

    def forward(x: torch.Tensor, *args, **kwargs):
        geometry = (
            int(getattr(layer, "in_features", -1)),
            int(getattr(layer, "out_features", -1)),
        )
        if (
            not args
            and not kwargs
            and _eligible_role(layer, x) == (role, scale_input)
            and hasattr(layer, "weight")
            and geometry in _H3_WIDE_LINEARS
        ):
            return _execute_wide_linear(
                layer,
                x,
                role,
                scale_input,
                run_every_op=True,
            )
        return original_forward(x, *args, **kwargs)

    layer.forward = forward
    layer._h3_v100_fp32_tc_wrapped = True


def _install_h3_constructor_wrappers() -> int:
    """Mark H3 out_proj/fc2 instances independent of GGUF import aliases."""

    h3_model = importlib.import_module("comfy.ldm.minimax.model")
    patched = 0
    specifications = (
        (h3_model.Attention, "out_proj", "attention_out", False),
        (h3_model.MLP, "fc2", "mlp_fc2", True),
    )
    for module_class, attribute, role, scale_input in specifications:
        current = module_class.__init__
        if getattr(current, "_h3_v100_fp32_tensor_core", False):
            continue

        def wrapped_init(
            instance,
            *args,
            _original=current,
            _attribute=attribute,
            _role=role,
            _scale_input=scale_input,
            **kwargs,
        ):
            _original(instance, *args, **kwargs)
            _wrap_h3_wide_module(
                getattr(instance, _attribute), _role, _scale_input
            )

        wrapped_init._h3_v100_fp32_tensor_core = True
        _ORIGINAL_H3_INITIALIZERS[module_class] = current
        module_class.__init__ = wrapped_init
        patched += 1
    return patched


def runtime_stats() -> dict[str, object]:
    """Return lightweight observability counters for tests and diagnostics."""

    return {
        "installed": _INSTALLED,
        "hit_counts": dict(_HIT_COUNTS),
        "seen_roles": sorted(_SEEN_ROLES),
        "patched_linear_classes": sorted(
            f"{linear.__module__}.{linear.__qualname__}"
            for linear in _ORIGINAL_GGUF_FORWARDS
        ),
        "patched_h3_constructors": sorted(
            f"{module.__module__}.{module.__qualname__}"
            for module in _ORIGINAL_H3_INITIALIZERS
        ),
    }


def install_h3_fp32_tensor_core_linear() -> bool:
    """Install the narrow ComfyUI-GGUF Linear dispatch override once."""

    global _INSTALLED
    if _INSTALLED:
        return True
    if not torch.cuda.is_available():
        logging.warning("H3 FP32 Tensor Core path requested but CUDA is unavailable")
        return False

    gguf_ops = importlib.import_module("custom_nodes.ComfyUI-GGUF.ops")
    candidate_modules = {gguf_ops}
    # ComfyUI's custom-node loader can expose the same plugin under a generated
    # package name as well as ``custom_nodes.<name>``. Patch every already
    # imported ComfyUI-GGUF ops class so loader aliasing cannot bypass us.
    for module_name, module in tuple(sys.modules.items()):
        if "ComfyUI-GGUF" in module_name and hasattr(module, "GGMLOps"):
            candidate_modules.add(module)

    _wide_aware_gguf_forward._h3_v100_fp32_tensor_core = True
    patched = 0
    for module in candidate_modules:
        linear_class = module.GGMLOps.Linear
        original = linear_class.forward_ggml_cast_weights
        if getattr(original, "_h3_v100_fp32_tensor_core", False):
            continue
        _ORIGINAL_GGUF_FORWARDS[linear_class] = original
        linear_class.forward_ggml_cast_weights = _wide_aware_gguf_forward
        patched += 1
    constructor_count = _install_h3_constructor_wrappers()
    _INSTALLED = True
    logging.info(
        "H3-only SM70 FP16 Tensor Core / FP32-output GGUF Linear dispatch "
        "installed for %d class alias(es), %d H3 constructor(s)",
        patched,
        constructor_count,
    )
    return True


def install_from_env() -> bool:
    value = os.environ.get("H3_V100_FP32_TC", "0").strip().lower()
    if value in {"", "0", "off", "false", "pytorch", "fp32"}:
        return False
    if value not in {"1", "on", "true", "tensorcore", "sm70"}:
        raise ValueError(f"unsupported H3_V100_FP32_TC={value!r}")
    return install_h3_fp32_tensor_core_linear()


__all__ = [
    "FP16_SCALE_TARGET",
    "install_from_env",
    "install_h3_fp32_tensor_core_linear",
    "power_of_two_fp16_scale",
    "runtime_stats",
    "tensor_core_fp32_output_linear",
]
