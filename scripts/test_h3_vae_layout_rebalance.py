"""VAE 采样/decode 双切分 rebalance 的验收门禁。

rebalance 只改权重驻留位置，不改 block 顺序、权重、dtype 或 forward 实现，
所以它在数值上必须是惰性的：同一个 latent 在 18/18 和 24/12 两种布局下解码，
结果应当逐元素相同（``max_abs == 0.0``）。任何非零差异都说明搬移动到了数值路径。

同时验证：
- INT8 的 ``_h3_v100_int8_scale_fp16`` 缓存随 block 一起迁移（不迁移会直接跨设备报错）；
- admission check 在 cuda:0 吃紧时能退档，而不是让后续分配在 collective 里炸；
- 整卡显存用 ``mem_get_info`` 实测，两张卡都留够余量；
- NVLink 搬运耗时。

用法：
    /home/regen/minimax-h3/.venv/bin/python scripts/test_h3_vae_layout_rebalance.py \
        --width 832 --height 480 --frames 124 \
        --output results/vae_layout_832x480.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

COMFY_ROOT = Path(os.environ.get("H3_COMFY_ROOT", "/home/regen/minimax-h3/ComfyUI"))
INT8_VAE = Path(
    "/mnt/GALAX/minimax-h3/experimental/vae_int8/"
    "minimax_h3_video_vae_int8_convrot.safetensors"
)
FP16_VAE = Path("/mnt/GALAX/minimax-h3/models/vae/minimax_h3_video_vae_fp16.safetensors")
MIB = 1024**2


def _bootstrap():
    """Import ``h3_model_parallel`` without DualV100's node registration.

    ``DualV100/__init__.py`` imports ComfyUI-MultiGPU, which needs ComfyUI's
    global node registry to already be populated.  This gate only needs the
    layer-MP module, so load it straight from its file.
    """
    if not COMFY_ROOT.is_dir():
        raise SystemExit(f"ComfyUI root not found: {COMFY_ROOT}")
    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(COMFY_ROOT / "custom_nodes"))

    import importlib.util

    path = COMFY_ROOT / "custom_nodes" / "DualV100" / "h3_model_parallel.py"
    if not path.is_file():
        raise SystemExit(f"未找到部署的模块: {path}")
    spec = importlib.util.spec_from_file_location("h3_model_parallel_gate", path)
    module = importlib.util.module_from_spec(spec)
    # NoHostMMap 的 file-slice reader 走 ``custom_nodes.NoHostMMap``，
    # 上面的 sys.path 已经能满足它。
    spec.loader.exec_module(module)
    return module


def _whole_card_mib() -> list[dict]:
    import torch

    out = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free, total = torch.cuda.mem_get_info()
        out.append(
            {
                "device": f"cuda:{index}",
                "used_mib": round((total - free) / MIB, 1),
                "free_mib": round(free / MIB, 1),
            }
        )
    return out


def _block_devices(model) -> list[str]:
    """每个 decoder block 的驻留卡。INT8 权重要看 ``_qdata`` 的真实位置。"""
    out = []
    for block in model.decoder.source.transformer_blocks:
        owner = None
        for tensor in list(block.parameters()) + list(block.buffers()):
            qdata = getattr(tensor, "_qdata", None)
            owner = getattr(qdata, "device", None) or tensor.device
            break
        out.append(str(owner))
    return out


def _scale_cache_mismatches(model) -> list[str]:
    """INT8 W8A16 的 FP16 scale 缓存必须和它所属 Linear 同卡。"""
    bad = []
    for name, sub in model.decoder.source.named_modules():
        cached = getattr(sub, "_h3_v100_int8_scale_fp16", None)
        if cached is None:
            continue
        weight = getattr(sub, "weight", None)
        qdata = getattr(weight, "_qdata", None)
        owner = getattr(qdata, "device", None) or getattr(weight, "device", None)
        if owner is not None and cached.device != owner:
            bad.append(f"{name}: scale={cached.device} weight={owner}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", default=None,
                        help="默认优先 INT8，其次 FP16")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--dit-split", type=int, default=18)
    parser.add_argument("--decode-split", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.vae:
        vae_path = Path(args.vae)
    elif INT8_VAE.is_file():
        vae_path = INT8_VAE
    else:
        vae_path = FP16_VAE
    if not vae_path.is_file():
        raise SystemExit(f"VAE 不存在: {vae_path}")

    # 环境变量必须在模块加载前设好：切分是在 load 时解析的。
    os.environ["H3_VAE_DIT_SPLIT"] = str(args.dit_split)
    os.environ["H3_VAE_DECODE_SPLIT"] = str(args.decode_split)
    os.environ.pop("H3_VAE_SPLIT", None)
    # 本门禁只测 rebalance，把 tile pipeline 排除在变量之外。
    os.environ["H3_VAE_MP_PIPELINE"] = "0"

    mp = _bootstrap()
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("需要两张 CUDA 卡")

    print(f"[gate] 加载 {vae_path.name}", flush=True)
    load_started = time.perf_counter()
    vae = mp.load_h3_video_vae_parallel(str(vae_path))
    if vae is None:
        raise SystemExit("layer-MP VAE 不可用（H3_VAE_MP=0 或无 P2P）")
    model = vae.first_stage_model
    layout = vae.layout
    if layout is None:
        raise SystemExit("VAE 没有 layout manager；rebalance 未接上")
    print(f"[gate] 加载完成 {time.perf_counter() - load_started:.1f}s  "
          f"dit={layout.dit_split} decode={layout.decode_split}", flush=True)

    is_int8 = bool(getattr(model, "_h3_parallel_report", {}).get(
        "quantized_linear_tensors"))
    result: dict = {
        "vae": str(vae_path),
        "quantized": is_int8,
        "resolution": [args.width, args.height],
        "frames": args.frames,
        "dit_split": layout.dit_split,
        "decode_split": layout.decode_split,
        "nvlink": {},
        "checks": {},
    }

    # 采样布局必须是初始状态：VAE 不能一上来就把重的一半压在 cuda:0。
    assert layout.current_split == args.dit_split, layout.current_split
    result["memory_sampling_layout"] = _whole_card_mib()
    result["blocks_sampling_layout"] = _block_devices(model)
    print(f"[gate] 采样布局 split={layout.current_split} "
          f"{result['memory_sampling_layout']}", flush=True)

    latent_h = args.height // model.vae_ratio
    latent_w = args.width // model.vae_ratio
    latent_t = (args.frames - 5) // 17 * 5 + 2 if args.frames > 1 else 1
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.randn(
        (1, 24, latent_t, latent_h, latent_w), generator=generator,
        dtype=torch.float32,
    ).to(device=vae.device, dtype=torch.float16)
    print(f"[gate] latent {tuple(latent.shape)}", flush=True)

    # ---- 1) 采样布局下解码（rebalance 前的基线）----------------------------
    # decode() 自己会切到 decode 布局，所以基线要直接打模型，绕过那次切换。
    def decode_at_current_layout():
        torch.cuda.synchronize(vae.device)
        started = time.perf_counter()
        shape = model.decode_output_shape(latent.shape)
        buffer = torch.empty(shape, device="cpu", dtype=torch.float32)
        model.decode(latent, output_buffer=buffer)
        torch.cuda.synchronize(vae.device)
        return buffer, time.perf_counter() - started

    baseline, baseline_seconds = decode_at_current_layout()
    peak_sampling = _whole_card_mib()
    print(f"[gate] split={layout.current_split} 解码 {baseline_seconds:.2f}s "
          f"{peak_sampling}", flush=True)

    # ---- 2) 切到 decode 布局，再解码同一个 latent -------------------------
    moved_started = time.perf_counter()
    applied = layout.ensure_decode()
    move_seconds = time.perf_counter() - moved_started
    result["nvlink"]["decode_move_seconds"] = round(move_seconds, 4)
    result["nvlink"]["decode_split_applied"] = applied
    result["blocks_decode_layout"] = _block_devices(model)
    result["memory_decode_layout"] = _whole_card_mib()
    print(f"[gate] rebalance -> split={applied} 用时 {move_seconds * 1000:.1f}ms "
          f"{result['memory_decode_layout']}", flush=True)

    mismatches = _scale_cache_mismatches(model)
    result["checks"]["int8_scale_cache_follows_weight"] = not mismatches
    if mismatches:
        print(f"[gate] INT8 scale 缓存未随权重迁移: {mismatches[:4]}", flush=True)

    rebalanced, rebalanced_seconds = decode_at_current_layout()
    peak_decode = _whole_card_mib()
    print(f"[gate] split={layout.current_split} 解码 {rebalanced_seconds:.2f}s "
          f"{peak_decode}", flush=True)

    # ---- 3) 逐元素一致性 --------------------------------------------------
    max_abs = float((baseline.float() - rebalanced.float()).abs().max().item())
    result["max_abs_difference"] = max_abs
    result["checks"]["bitwise_identical"] = max_abs == 0.0
    result["checks"]["finite"] = bool(torch.isfinite(rebalanced).all().item())

    # ---- 4) 切回采样布局 --------------------------------------------------
    back_started = time.perf_counter()
    restored = layout.ensure_sampling()
    back_seconds = time.perf_counter() - back_started
    result["nvlink"]["sampling_move_seconds"] = round(back_seconds, 4)
    result["blocks_restored"] = _block_devices(model)
    result["memory_restored"] = _whole_card_mib()
    result["checks"]["restored_to_dit_split"] = restored == args.dit_split
    result["checks"]["block_layout_restored"] = (
        result["blocks_restored"] == result["blocks_sampling_layout"]
    )
    print(f"[gate] 切回 split={restored} 用时 {back_seconds * 1000:.1f}ms "
          f"{result['memory_restored']}", flush=True)

    # 切回后必须还能正确解码，确认往返没有损坏权重。
    roundtrip, _ = decode_at_current_layout()
    roundtrip_max_abs = float(
        (baseline.float() - roundtrip.float()).abs().max().item()
    )
    result["roundtrip_max_abs_difference"] = roundtrip_max_abs
    result["checks"]["roundtrip_bitwise_identical"] = roundtrip_max_abs == 0.0

    # ---- 5) admission check 退档 ------------------------------------------
    # 把安全余量抬到超过整卡容量，任何搬移都不该被放行，且必须保持当前布局。
    original_safety = layout.safety_bytes
    layout.safety_bytes = 64 << 30
    before = layout.current_split
    degraded = layout.ensure_decode()
    layout.safety_bytes = original_safety
    result["checks"]["admission_blocks_when_tight"] = degraded == before
    result["admission_degraded_to"] = degraded
    print(f"[gate] admission 压力测试: split 保持 {degraded}（期望 {before}）",
          flush=True)

    result["decode_seconds"] = {
        f"split_{args.dit_split}": round(baseline_seconds, 3),
        f"split_{applied}": round(rebalanced_seconds, 3),
    }
    result["whole_card_peak_mib"] = {
        "sampling_layout_decode": peak_sampling,
        "decode_layout_decode": peak_decode,
    }

    # 两张卡都要留 >= 1 GiB 余量。
    worst_free = min(
        entry["free_mib"]
        for snapshot in (peak_sampling, peak_decode)
        for entry in snapshot
    )
    result["worst_free_mib"] = worst_free
    result["checks"]["headroom_ge_1gib"] = worst_free >= 1024

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"[gate] 已写入 {out}", flush=True)

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    failed = [k for k, v in result["checks"].items() if not v]
    if failed:
        print(f"[gate] FAIL: {failed}", flush=True)
        return 1
    print(f"[gate] PASS: rebalance 逐元素一致，最小余量 {worst_free:.0f} MiB",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
