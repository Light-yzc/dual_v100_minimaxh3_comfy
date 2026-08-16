# Windows 双 V100 实验记录

## 硬件与拓扑

- 2 × Tesla V100-SXM2-16GB。
- 物理上存在 6 条 NVLink。
- Windows/WSL 测试未获得可用 CUDA peer access/NCCL，所以没有在该环境继续做 TP。
- 系统内存约 16GB，模型文件位于 HDD；动态 GGUF 每层搬运造成明显停顿。

## 模型组合

| 组件 | 文件 | 约占磁盘 |
|---|---|---:|
| DiT | `minimax_h3_fl2va_pruned_fp8_Q4_0.gguf` | 10.6 GiB |
| 文本编码器 | `qwen3vl-32B-MiniMax-H3-Q2_K.gguf` | 7.9 GiB |
| 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | 5.4 GiB |
| Turbo LoRA | `minimax_h3_turbo_v4_step600_ema.safetensors` | 0.73 GiB |

## 已观察结果

- Q2 文本编码器可完整加载到 GPU1，ComfyUI 报告约 9587MB full load。
- 静态 DiT 加载曾到 GPU0 约 8394MiB 时被外部 RAM guard 主动终止；不是 CUDA OOM。
- 动态 FP16 的 832×480、124 帧、Turbo 4-step 可以进入采样，单 step 约 119–123 秒，但第一步输出出现 NaN。
- 更小的 448×256、22 帧 FP16 路径也出现过 NaN，因此问题不是单纯由 480p 显存压力引起。
- 最终未在 Windows 上验证新的 FP16 RMSNorm 缩放补丁，因为随后决定切换 Ubuntu。

## FP16 NaN 补丁原理

V100 FP16 的最大有限值约 65504。RMSNorm 需要平方并求均值，输入绝对值大于约 256 时，直接 FP16 平方可能先溢出。补丁在每行或每个 attention head 上取最大绝对值并先除以该比例，再执行原 RMSNorm。RMSNorm 对正比例缩放近似不变，因此可以避免平方溢出，同时不把主模型改成全局 FP32。

补丁还没有证明 NaN 的唯一来源就是 RMSNorm。Ubuntu 第一轮必须先跑 `static-smoke-448x256-1step.json`，然后用：

```bash
$INSTALL_ROOT/.venv/bin/python scripts/check_latent.py \
  "$INSTALL_ROOT/ComfyUI/output/h3_static_smoke_448x256_latent.pt"
```

如果仍有非有限值，应在每个 DiT block 的 attention、MLP 和 residual 输出后加有限值检查，定位第一个产生 Inf/NaN 的算子，不要直接扩大分辨率。
