# Qwen32B Q2 TP 重测（2026-08-27）

> 该文档记录的是一次 TP 正确性重测，不代表 TP 已成为默认路线；该次测试使用
> 已启动的 TP worker，output-row TP 仅作显式实验。当前启动器默认使用解耦的
> layer-MP 后端，切换说明见 [`QWEN32B_Q2_MP.md`](QWEN32B_Q2_MP.md)。

## 结论

32B 路线这次确认修好了。之前的输出是与 prompt 无关的重复人脸徽章；重测后的
832×480、124 帧视频是雨夜霓虹街道上的摩托车追逐，并且首、中、尾帧的主体和运镜
语义连续。不是单纯“节点返回 success”。

低资源 smoke、完整 4-step、VAE 解码和有限值检查全部通过。详细机器可读汇总见
[qwen32_q2_retest_summary_20260827.json](../results/qwen32_q2_retest_summary_20260827.json)。

## 实测配置

- 双 Tesla V100-SXM2-16GB，PyTorch 2.8.0+cu126，ComfyUI 0.31.0。
- Qwen 文件：`/mnt/GALAX/minimax-h3/models/text_encoders/qwen3vl-32B-MiniMax-H3-Q2_K.gguf`。
- Qwen 为 output-row TP，两 rank；FP32 activation/计算；`residency=evict`、
  `keep_layers=0`、4 MiB bounded staging、不开 dequantized weight cache。
- H3 使用 `minimax_h3_fl2va_pruned_fp8_Q4_0.gguf` +
  `minimax_h3_turbo_v4_step600_ema.safetensors`，LoRA strength `1.0`。
- 模型和 VAE 权重都从 `/mnt/GALAX` 读取；结果写入 `/home/regen/minimax-h3/ComfyUI/output`。

## 数值与性能门禁

| 阶段 | 结果 |
|---|---|
| contract | 5/5 passed |
| 32B conditioning | `[1,133,5120]`，FP32，全 finite，RMS 18.731，max abs 14850.3，39.179 s |
| 448×256 / 1 step | H3 forward 7.252 s，peak GPU0/1 6616/6541 MiB，latent finite |
| 832×480 / 124f / 4 step | DiT forwards 9.746/9.774/9.778/9.807 s，合计 39.105 s，peak 7994/7527 MiB |
| VAE decode | 30.08 s；复用 resident VAE；观察到 GPU0/1 9310/11882 MiB |

服务 cgroup 峰值：conditioning 约 3.02 GiB，完整采样约 3.89 GiB，解码后约
4.03 GiB，低于 7 GiB 上限。没有发生 OOM、NaN 或 Inf。

完整采样 latent：

- video `[1,24,37,30,52]`，finite，RMS 1.060211。
- audio `[1,32,2,207]`，finite，RMS 0.518997。

## mmap / RAM 检查

header/layout audit 的 `payload_mmap_hits` 为空；运行中的 ComfyUI 和 rank1
`/proc/<pid>/maps` 也没有 Qwen GGUF 映射。reader 只保留普通只读文件描述符，权重
通过 4 MiB staging 分块读取到最终 GPU owner，没有把 8.49 GB payload mmap 到 RAM。

Qwen conditioning 清理后才进入 H3；Qwen peer conditioning 只有 2,724,904 bytes
（约 2.60 MiB）。VAE 使用 resident MP，解码阶段日志确认没有 DynamicVRAM 重复
load/unload。

## 为什么之前会错

关键问题在自定义 Qwen attention wrapper 的 layout 契约：只传
`skip_reshape=True` 不足以保证输出仍是 `[B,H,S,D]`。Comfy attention 默认可能返回
`[B,S,H*D]`，随后被当成 head-major 张量继续处理，会把 token/head 特征静默打乱。
当前路径在 [h3_qwen32_q2_tp.py](../custom_nodes/DualV100/h3_qwen32_q2_tp.py) 的
attention 调用中同时保留 `skip_reshape=True` 和 `skip_output_reshape=True`，并保留
FP32 Qwen 算术、正确的 PAD/EOS 和纯 causal-mask 语义。

另外，Qwen 50 层的两 rank collective、失败清理和 `qwen_clear()` barrier 都在
同一 shared runtime 中执行，避免半完成 conditioning 被 DiT 消费。

## 输出与复现

复现顺序：

```bash
/home/regen/minimax-h3/.venv/bin/python scripts/test_h3_qwen32_tp_contract.py
/home/regen/minimax-h3/.venv/bin/python scripts/submit_workflow.py \
  workflows/qwen32-q2-shared-tp-conditioning-complex-retest-20260827.json \
  --wait --timeout 1200 \
  --output results/qwen32_q2_shared_conditioning_complex_retest_trace_20260827.json
/home/regen/minimax-h3/.venv/bin/python scripts/submit_workflow.py \
  workflows/qwen32-q2-tp-full-complex-retest-832x480-124f-4step-20260827.json \
  --wait --timeout 1800 \
  --output results/qwen32_q2_tp_full_complex_retest_832x480_124f_4step_20260827.json
```

主要产物：

- [重测 MP4](/home/regen/minimax-h3/ComfyUI/output/video/qwen32_q2_tp/retest_complex_full_832x480_124f_4step_20260827_00001_.mp4)
- [完整 latent](/home/regen/minimax-h3/ComfyUI/output/benchmarks/h3_qwen32_q2_tp/qwen32_retest_complex_full_832x480_124f_4step_20260827.pt)
- [conditioning workflow](../workflows/qwen32-q2-shared-tp-conditioning-complex-retest-20260827.json)
- [full workflow](../workflows/qwen32-q2-tp-full-complex-retest-832x480-124f-4step-20260827.json)
- [latent 对比](../results/qwen32_old_vs_retest_latent_compare_832x480_20260827.json)

旧/新 latent 的 video cosine 只有 `-0.0677`，对应视觉上从错误徽章切换为正确追逐
场景；这证明本次重测没有复用旧 conditioning。
