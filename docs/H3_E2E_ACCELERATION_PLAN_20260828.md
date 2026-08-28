# H3 端到端加速方案

日期：2026-08-28。目标是缩短双 V100 上 Qwen32 MP + H3 DiT TP 的真实请求时间，
同时保持 no-host-mmap、FP32 数值稳定路径、NCCL collective 顺序和输出质量。

## 当前基线与瓶颈

已完成的 832×480、124 帧、4-step 重测给出：Qwen32 冷 conditioning 约 **39.18 s**；
DiT 四步约 **39.1 s**（短 reference/文本序列）；VAE decode 约 **30.08 s**。当前
参考图长序列运行约 `S=67k`，均衡后每步 DiT 约 10.5 s、NCCL 约 0.56–0.58 s。
因此冷请求在不重叠阶段时约有 108 s 的可见工作量；warm/cache-hit 请求应分别报告，
不能把两种状态混成一个平均数。
历史 50–59 s 的 NCCL 数字是 rank0 等待 rank1 的失衡放大，不是 300 GB/s 链路的真实
传输时间。VAE `24/12` split 已解决 GPU1 尾部 OOM，但只改善容量，不降低卷积计算量。

## 优先级与预期收益

| 优先级 | 项目 | 预计端到端收益 | 默认策略 |
|---|---|---:|---|
| P0 | conditioning cache（prompt、参考图、模型/路由版本做完整 key） | cache hit 时省 39 s；重复请求 TTFT 接近零 Qwen | 直接进入默认，命中/失效必须可观测 |
| P0 | rank 负载/温度监控与 MP weighted split | 消除失衡等待，避免偶发多几十秒 | 默认开启 fail-closed 监控，不改 TP |
| P1 | 长序列 attention 专项 kernel/profile（先 Q-only，后 tile/chunk） | DiT 阶段目标 10–25%；理论上限受 SDPA 约 34 TFLOPS（远低于 GEMM 峰值）限制 | 仅 A/B；同 seed hash/质量通过后再合入 |
| P1 | Qwen layer prefetch（evict + 单槽、256 MiB 上限） | Qwen 阶段实测 6.4%，冷请求约省 1–3 s | 默认关闭；目标 workflow A/B 后按余量启用 |
| P1 | DiT 期间 capped VAE prefetch、末尾 finalize | 只对冷 VAE 有效，预计隐藏 3–10 s | 默认关闭；每卡保留 ≥1 GiB 安全余量 |
| P1 | resident VAE decoder/audio 的 warm compile | decode 预计 5–10%（约 2–4 s），首次编译成本数秒 | 只在固定 shape、第二次请求启用 |
| P2 | TE-Speed / Group Cache | 稳态最多约 1.2–1.3× DiT，但当前 video relative RMS≈0.64，质量不合格 | 实验工作流，禁止默认 |

## 实施顺序

1. 先补齐端到端事件时间线：`qwen_load/forward/clear`、每步 DiT compute/collective、
   VAE/audio、encode/save；同时记录两卡 allocated/reserved peak、RSS、page-cache、
   temperature/clock 和 cache hit。没有这组数据不接受“加速”结论。
2. 完成 conditioning cache 的跨 seed/steps 命中测试；缓存内容必须包含 FP32
   conditioning、token/reference tags、shape 和模型 revision，失配直接 miss。
3. 在服务空闲时按顺序跑 attention kernel、Qwen prefetch、VAE prefetch/compile 的
   单变量 A/B。任何 rank OOM、NaN/Inf、collective timeout 或输出 hash/质量门禁失败，
   自动回退同步路径。
4. 最后才评估 TE-Speed/Group Cache；先用小尺寸 oracle 校准 threshold，再跑长序列
   holdout，不能用 cache hit 率替代视频/音频质量。

首轮可复用的检查命令：

```bash
/home/regen/minimax-h3/.venv/bin/python scripts/test_h3_qwen32_mp_contract.py
/home/regen/minimax-h3/.venv/bin/python scripts/benchmark_h3_qwen32_mp_prefetch.py \
  --prefetch 0 --dump results/e2e_prefetch_off.pt \
  --output results/e2e_prefetch_off.json
/home/regen/minimax-h3/.venv/bin/python scripts/benchmark_h3_qwen32_mp_prefetch.py \
  --prefetch 1 --reference results/e2e_prefetch_off.pt \
  --output results/e2e_prefetch_on.json
```

端到端项目必须通过 `scripts/submit_workflow.py --wait` 使用同一工作流成对提交，
不能用不同 seed 或不同 page-cache 状态的历史结果互相比较。

## 统一验收矩阵

固定模型、prompt、参考图、seed、scheduler，至少覆盖：

```text
448×256 / 1 step（smoke）
832×480 / 124f / 4 step：text、单 ref、首尾双 ref
cold page-cache、warm、conditioning-cache hit 各 3 次
```

每次保存 wall time 分解、双卡峰值/RSS/mmap、latent/video/audio finite，以及与 Full
Compute 的 video/audio relative RMS、cosine、LPIPS/temporal error。进入默认的最低门槛是：
同 seed 输出通过质量门禁，p95 wall time 至少下降 5%，峰值显存不增加超过 256 MiB，
并连续 10 次请求无 OOM、NCCL skew 或 cache 污染。

## 明确不做

Qwen output-row TP 继续作为实验功能；当前数值 gate 未通过，不能用它替换 MP。也不把
完整 Qwen/VAE 常驻、无预算双向预取、或未校准的 residual cache 写入生产默认。
