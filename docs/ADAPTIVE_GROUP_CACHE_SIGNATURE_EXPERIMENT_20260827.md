# H3 Group Cache：bounded input signature 实验路径

日期：2026-08-27

这份记录补充 `ADAPTIVE_GROUP_CACHE_MATH_REVIEW_20260827.md` 的 P3，不修改
`TP_SPEED_FUTHUR.MD` 定义的第一阶段研究问题，也不改变当前 ComfyUI 默认推理。

## 结论先行

当前 Group Cache 的持久 Q4 residual 本身不是主要的无命中开销；问题在于旧判定每个
Group、每个 step 都要执行：

```text
FP32 group input
  -> Q4_0 quantize
  -> 保存在 CPU 的 previous Q4 input 回传
  -> 两份 Q4 分块 dequant 到 GPU
  -> full hidden metric reduction
```

因此在 `threshold=0.005`、没有 cache hit 的小尺寸测量中，Group Cache 比 Full 慢约
6.2%。这条路径还让四组 `previous_input` 在 1 MP 上占用约 456 MiB/rank 的 CPU 内存。

新增的 `feature_mode=signature` 是一个明确 opt-in 的研究路径：

```text
FP32 group input
  -> deterministic stratified FP32 signature (CPU)
  -> small CPU metric
```

它只替换“是否允许复用”的输入 feature，绝不改变实际生成输出所用的 residual：

```text
FULL  : residual = Q4_0(F_g(x) - x)
CACHE : output   = x + dequant(Q4_0 residual)
```

所以它节省的是判定和 `previous_input`，不是通过改变模型算术换取速度。默认仍为
`feature_mode=q4`，且没有更新任何正式工作流。

## 实现

节点 `Adaptive Group Residual Cache (TP, Q4_0)` 新增可选项：

```text
feature_mode:
  q4          # 默认，旧完整 Q4 input metric
  signature   # 新实验路径

signature_max_tokens: 2048
signature_hidden_samples: 32
signature_aggregation:
  weighted    # 默认，compact signature 的整体 metric
  max_segment # 保守；取 packed segment 的最大误差
```

signature 的 token budget 会按 H3 packed segments 的长度比例分配，并且每个 segment 至少
保留一个 token sample。这样 context 前缀不会把 audio/video 的局部变化完全平均掉。hidden
通道取固定、含两端点的等距 32 维；采样布局、shape 或参数变化会清空 cache，而不是静默复用。

`previous_step` 仍只作为 ablation。主 benchmark 应使用 `last_full`，因为 Q4 residual
锚点来自最近一次 FULL；连续 cache 时与上一步输入比较不能单独证明 residual 仍然安全。

## 1 MP 内存几何

假设当前路径的每 rank packed shape 为：

```text
S = 37746, H = 5376
Q4_0 = 18 bytes / 32 values
```

| 持久项 | 旧 `q4` feature | 新 `signature` feature |
| --- | ---: | ---: |
| 4 个 Q4 residual | 435.4 MiB | 435.4 MiB |
| 4 个 Q4 previous input | 435.4 MiB | 0 |
| 4 个 `[2048, 32]` FP32 signature | 0 | 1.0 MiB |
| 合计 | 870.9 MiB | 436.4 MiB |

即每 rank 约少 **434.5 MiB**（按 MiB 计，约 456 MB）。这不会降低 GPU persistent
VRAM，因为二者默认都在 CPU；它降低 host RAM、PCIe/NVLink 前的 CPU↔GPU搬运与判定成本。

构造 signature 时先选择 32 个 hidden channel，再采样 token。因此 1 MP 最大中间量约为：

```text
37746 × 32 × FP32 ≈ 4.61 MiB/rank
```

而不是克隆完整 `37746 × 5376 × FP32 ≈ 774 MiB` hidden stream。CPU signature 每组仅
256 KiB。

## 质量边界

signature 不是精确 full-hidden metric，也不能直接沿用 Q4 threshold。必须重新校准：

1. 使用 `benchmark_ground_truth=true` 做小尺寸 oracle；不要在 1 MP 开 oracle。
2. 同时记录 Q4 feature、signature feature、AdaLN condition signature、真实 group error。
3. 分开扫 `weighted` 与 `max_segment` 的 threshold；后者预计更保守，数值尺度更大。
4. 用 holdout 的 `output_relative_l2`/LPIPS/temporal error 判断，而不是只看 cache hit。
5. 只有在同质量点比 whole-tail TE-Speed 更快，才可以考虑生产 gate。

一个保守首轮配置（仅 benchmark）是：

```text
warm_blocks=8
num_groups=4
metric=relative_l2
feature_mode=signature
signature_max_tokens=2048
signature_hidden_samples=32
signature_aggregation=max_segment
reference_mode=last_full
max_cache=2
calibration_mode=collect
condition_metric=all_adaln
benchmark_ground_truth=true
```

threshold 不在文档中预设：它必须由同一 prompt/seed/latent 的 oracle 数据标定。`q4` 与
`signature` 的数值尺度不同，直接用 `0.005` 没有意义。

## 已验证的非 GPU 契约

```bash
/home/regen/minimax-h3/.venv/bin/python scripts/test_h3_q4_cache.py \
  --output /mnt/GALAX/h3_q4_cache_signature_current.json

/home/regen/minimax-h3/.venv/bin/python scripts/test_h3_group_cache_calibration.py \
  --output /mnt/GALAX/h3_group_cache_calibration_cpu_current.json
```

已覆盖：

- Q4_0 字节仍与 GGUF reference 完全一致；
- Q4 分块 residual add 仍为精确 reference；
- signature 自比较为零；
- 视频段的局部扰动会被 `max_segment` 捕获；
- sampling layout 变化会 fail closed；
- feature mode 切换会清空旧 cache；
- AdaLN signature 对 row-id 重编号不误报，并能检出 gate 扰动。

## 尚未完成

- 未部署到 `/home/regen/minimax-h3/ComfyUI`；源仓库优先，部署应通过
  `SYNC_ONLY=1 INSTALL_ROOT=$HOME/minimax-h3 ./scripts/setup_ubuntu.sh`。
- 未重启服务，未修改 Qwen TE。
- GPU0 当前仅余约 2.2 GiB 空闲，有一条用户 ComfyUI 进程持有 13.8 GiB；本轮不抢卡、不做
  runtime smoke，以免影响现有服务或触发 OOM。
- 未把 signature 或 AdaLN 校准结果接进默认决策，仍需先跑校准集和 holdout。
