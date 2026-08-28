"""VAE decode 的显存/速度矩阵：tile_batch x 量化格式，MP 布局固定对比。

回答三个具体问题：
1. ``H3_VAE_INT8_TILE_BATCH`` 在 layer-MP 下是否反而抬高峰值显存？
   它用 ``torch.cat`` 把 N 个 tile 拼到 batch 维再走一次 36 层，
   所以激活、attention 临时量和跨卡 handoff 都按 N 倍放大。
2. tile_batch=1（逐 tile）和 tile_batch=2 的真实速度差多少？
3. INT8 与 FP16 在**同一条代码路径**下的峰值与耗时对比
   （此前把 INT8+tile_batch 和 FP16+逐tile 直接对比是无效的）。

显存用 ``mem_get_info`` 读整卡，并在每个 case 前后各测一次，
增量才是该 case 真实引入的量。torch allocator 数字同时记录以便交叉验证。

用法：
    /home/regen/minimax-h3/.venv/bin/python scripts/benchmark_h3_vae_tile_batch.py \
        --width 832 --height 480 --frames 124 \
        --output results/vae_tile_batch_832x480.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
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


def _load_mp():
    if not COMFY_ROOT.is_dir():
        raise SystemExit(f"ComfyUI root not found: {COMFY_ROOT}")
    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(COMFY_ROOT / "custom_nodes"))
    path = COMFY_ROOT / "custom_nodes" / "DualV100" / "h3_model_parallel.py"
    spec = importlib.util.spec_from_file_location("h3_mp_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _whole_card() -> list[float]:
    """整卡已用 MiB，两张卡。"""
    import torch

    out = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free, total = torch.cuda.mem_get_info()
        out.append(round((total - free) / MIB, 1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--dit-split", type=int, default=18)
    parser.add_argument("--decode-split", type=int, default=24)
    parser.add_argument("--tile-batches", default="1,2,4",
                        help="逗号分隔的 tile_batch 取值")
    parser.add_argument("--repeat", type=int, default=2, help="预热后的计时次数")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    parser.add_argument("--case", default=None,
                        help="内部使用：单个 case 的 json 规格")
    args = parser.parse_args()

    # 每个 case 必须在独立进程里跑：一旦 caching allocator 被上一个
    # case 撑大，整卡读数就再也分不清是谁的了。
    if args.case is None:
        return _run_matrix(args)
    return _run_single_case(args)


def _run_matrix(args) -> int:
    tile_batches = [int(x) for x in args.tile_batches.split(",") if x.strip()]
    formats = []
    if INT8_VAE.is_file():
        formats.append(("int8", str(INT8_VAE)))
    if FP16_VAE.is_file():
        formats.append(("fp16", str(FP16_VAE)))
    if not formats:
        raise SystemExit("找不到任何 VAE checkpoint")

    results = []
    for fmt, path in formats:
        for tb in tile_batches:
            spec = {
                "format": fmt, "vae": path, "tile_batch": tb,
                "width": args.width, "height": args.height,
                "frames": args.frames, "dit_split": args.dit_split,
                "decode_split": args.decode_split,
                "repeat": args.repeat, "seed": args.seed,
            }
            print(f"\n=== {fmt}  tile_batch={tb} ===", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--case", json.dumps(spec)],
                capture_output=True, text=True, timeout=3600,
            )
            payload = None
            for line in proc.stdout.splitlines():
                if line.startswith("__CASE__"):
                    payload = json.loads(line[len("__CASE__"):])
                else:
                    print(line, flush=True)
            if payload is None:
                tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
                print(f"[bench] {fmt}/tb={tb} 失败:\n{tail}", flush=True)
                results.append({**spec, "error": tail})
                continue
            results.append(payload)

    print("\n" + "=" * 78, flush=True)
    print(f"{'格式':6s} {'tb':>3s} {'解码s':>9s} {'GPU0增量':>10s} "
          f"{'GPU1增量':>10s} {'GPU0峰值':>10s} {'GPU1峰值':>10s}", flush=True)
    print("-" * 78, flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['format']:6s} {r['tile_batch']:3d}   ERROR", flush=True)
            continue
        d = r["delta_mib"]
        p = r["peak_whole_card_mib"]
        print(f"{r['format']:6s} {r['tile_batch']:3d} {r['decode_seconds']:9.3f} "
              f"{d[0]:10.1f} {d[1]:10.1f} {p[0]:10.1f} {p[1]:10.1f}", flush=True)

    summary = {
        "resolution": [args.width, args.height],
        "frames": args.frames,
        "dit_split": args.dit_split,
        "decode_split": args.decode_split,
        "cases": results,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"\n[bench] 已写入 {out}", flush=True)
    return 0


def _run_single_case(args) -> int:
    spec = json.loads(args.case)
    os.environ["H3_VAE_DIT_SPLIT"] = str(spec["dit_split"])
    os.environ["H3_VAE_DECODE_SPLIT"] = str(spec["decode_split"])
    os.environ.pop("H3_VAE_SPLIT", None)
    os.environ["H3_VAE_MP_PIPELINE"] = "0"
    # 关键：这个开关此前只在 INT8 loader 里被读，本脚本对两种格式都强制生效，
    # 从而让 INT8 与 FP16 落在同一条 tiled_decode 路径上。
    os.environ["H3_VAE_INT8_TILE_BATCH"] = str(spec["tile_batch"])

    mp = _load_mp()
    import torch

    baseline = _whole_card()
    vae = mp.load_h3_video_vae_parallel(spec["vae"])
    if vae is None:
        raise SystemExit("layer-MP VAE 不可用")
    model = vae.first_stage_model
    layout = vae.layout

    # 本基准只测 decode 布局下的解码，rebalance 本身另有门禁。
    if layout is not None:
        layout.ensure_decode()
    after_load = _whole_card()

    # FP16 loader 原本不装 tile_batch 包装；这里显式补装以做公平对比。
    installed = mp._install_h3_v100_int8_tile_batch(model, spec["tile_batch"])
    wrapped = hasattr(model, "_h3_v100_int8_original_tiled_decode")

    latent_h = spec["height"] // model.vae_ratio
    latent_w = spec["width"] // model.vae_ratio
    latent_t = (spec["frames"] - 5) // 17 * 5 + 2 if spec["frames"] > 1 else 1
    generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
    latent = torch.randn(
        (1, 24, latent_t, latent_h, latent_w), generator=generator,
        dtype=torch.float32,
    ).to(device=vae.device, dtype=torch.float16)

    y_idx, _yl, _yo = model.split_tiles(spec["height"])
    x_idx, _xl, _xo = model.split_tiles(spec["width"])

    def one_decode():
        buffer = torch.empty(
            model.decode_output_shape(latent.shape), device="cpu",
            dtype=torch.float32,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.decode(latent, output_buffer=buffer)
        torch.cuda.synchronize()
        return time.perf_counter() - started, buffer

    # 预热一次，付掉首次 workspace / dequant 成本
    warm_seconds, reference = one_decode()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    times = []
    peak_whole = _whole_card()
    for _ in range(max(1, spec["repeat"])):
        seconds, out = one_decode()
        times.append(seconds)
        now = _whole_card()
        peak_whole = [max(a, b) for a, b in zip(peak_whole, now)]
        del out

    allocator_peak = [
        round(torch.cuda.max_memory_allocated(d) / MIB, 1)
        for d in range(torch.cuda.device_count())
    ]
    allocator_reserved = [
        round(torch.cuda.max_memory_reserved(d) / MIB, 1)
        for d in range(torch.cuda.device_count())
    ]

    payload = {
        "format": spec["format"],
        "tile_batch": spec["tile_batch"],
        "tile_batch_installed": installed,
        "tiled_decode_wrapped": wrapped,
        "tiles": {"rows": len(y_idx), "cols": len(x_idx),
                  "total": len(y_idx) * len(x_idx)},
        "latent_shape": list(latent.shape),
        "decode_seconds": round(min(times), 3),
        "decode_seconds_all": [round(t, 3) for t in times],
        "warmup_seconds": round(warm_seconds, 3),
        "baseline_mib": baseline,
        "after_load_mib": after_load,
        "peak_whole_card_mib": peak_whole,
        # 解码真正引入的整卡增量：峰值减去权重加载后的静态占用
        "delta_mib": [round(p - a, 1) for p, a in zip(peak_whole, after_load)],
        "allocator_peak_allocated_mib": allocator_peak,
        "allocator_peak_reserved_mib": allocator_reserved,
        "finite": bool(torch.isfinite(reference).all().item()),
    }
    print(f"__CASE__{json.dumps(payload, ensure_ascii=False)}", flush=True)
    print(f"[case] {spec['format']} tb={spec['tile_batch']} "
          f"tiles={payload['tiles']['total']} "
          f"解码={payload['decode_seconds']}s "
          f"整卡增量={payload['delta_mib']} MiB", flush=True)
    del latent, reference, vae, model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
