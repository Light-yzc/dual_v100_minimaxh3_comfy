# Qwen3-VL-4B Q4/Q5 与 Video VAE INT8 显存压缩调研

更新时间：2026-08-26

状态：**Qwen Q4_K_M + FP16 vision/mmproj + ridge 已进入生产默认，并已完成双 V100
resident direct-owner 4/32 MP 的纯文本、参考图和 1 MP 实测；Video VAE INT8 的
W8A16 分块 decode 结果见 `VAE_INT8_TEST_RESULTS_20260826.md`**。

当前生产使用：

```text
Qwen3-VL-4B Q4_K_M + FP16 vision/mmproj
mmh3-4b-ClipProj-v3.1 ridge
H3_QWEN_SPLIT=4  # direct Q4 GGUF 默认值；可不显式设置
H3_VAE_SPLIT=12
H3_NO_HOST_MMAP=1
```

## 结论

Q4 已经上线，但实现不是简单把 `.safetensors` 文件名替换成 `.gguf`：

1. language Q4 GGUF 必须与匹配的 FP16 vision/mmproj 一起使用，才能保留 vision tower、
   DeepStack 和参考图路径。
2. ComfyUI-GGUF 对标准 GGML Q4_K_M 仍采用运行时反量化；V100 没有本路线可用的
   原生 W4A16 Tensor Core GEMM，不能把文件压缩比直接当成算力加速比。
3. 已新增 header-only、8 MiB staging 的 resident direct-to-final-owner loader；
   payload mmap 为 0，encoder 不会在每个请求 load/unload。Q4 tensor 在读取时按
   4/32 直接落到 GPU0/GPU1，不再先聚集到 GPU0。
4. Qwen 的大 vocabulary embedding 目前会被 loader 强制反量化成 FP16，因此 Q4/Q5
   的文件缩小量不会完整转换成 GPU0 的可用余量。

用户已接受 Q4 相对 INT8 的参考图 conditioning 误差并指定 Q4 为默认。Q5 仍可作为
后续精度/显存折中候选；text-only GGUF 和 Q4_PT/native W4A16 路线继续排除。

## 本地权重与上游支持证据

`/mnt/GALAX/minimax-h3/models` 当前只有：

| 文件 | 大小 | 状态 |
| --- | ---: | --- |
| `text_encoders/qwen3vl_4b_int8_convrot.safetensors` | `4,864,124,848` bytes（约 4.53 GiB） | 历史精度/fallback 基线 |
| `clip_projections/mmh3-4b-ClipProj-v3.1.safetensors` | `26,256,128` bytes（约 25.0 MiB） | 当前生产 ridge projection |
| `clip_projections/mmh3-4b-ClipProj-v3.1-mlp.safetensors` | `503,423,800` bytes（约 0.47 GiB） | 质量/显存对照，不是默认 |
| `text_encoders/qwen3vl-32B-MiniMax-H3-Q2_K.gguf` | `8,487,968,160` bytes（约 7.91 GiB） | 质量参考，不是 4B 候选 |
| `experimental/qwen3vl_q4/Qwen3VL-4B-Instruct-Q4_K_M.gguf` | `2,497,281,664` bytes（约 2.33 GiB） | 当前生产 language encoder |
| `experimental/qwen3vl_q4/mmproj-Qwen3VL-4B-Instruct-F16.gguf` | `836,180,256` bytes（约 797 MiB） | 当前生产 vision/mmproj，loader 自动配对 |
| `experimental/vae_int8/minimax_h3_video_vae_int8_convrot.safetensors` | `3,171,670,912` bytes（约 2.95 GiB） | header 已确认 INT8 ConvRot，未做 decode gate |

Q4 文件与 mmproj 位于 `/mnt/GALAX/minimax-h3/experimental/qwen3vl_q4/`。它们仍是
两个文件，ClipProj loader 会自动配对；不能单独把 language 主文件当完整 VL encoder。
主文件 398 个 tensor、mmproj 316 个 tensor，视觉拼接、DeepStack、3-axis mRoPE、
4/32 MP 和单参考图 conditioning 均已实际跑通。当前仍没有 Q5 文件，完整 same-seed
视频/音频质量 gate 尚未完成。

本地 ComfyUI-GGUF loader 已声明支持 `qwen3vl` 架构，并实现标准 `Q4_0/Q4_1`
和 `Q5_0/Q5_1` GGML 反量化；README 同时说明 `_K` 量化只保证 text encoder
场景，推理速度可能很慢。现有本地 32B Q2 GGUF 的 header 中确实包含 `visual.*`
张量，但这只能证明“完整 VL GGUF 可以存在”，不能证明任意 4B GGUF 都包含视觉塔
或能通过当前 ClipProj 路线。

## 为什么 Q4/Q5 不会等比例释放 GPU0

当前 4B encoder 的 INT8 对照文件约 4.53 GiB；Q4 direct-owner 的 GPU0 放：

```text
embedding + vision tower + Qwen 前 4 层
```

GPU1 放 Qwen 后 32 层、norm/head 和 projection。即使 Q4 文件少约 2 GiB，
也不能把 2 GiB 全部视为 GPU0 的余量；一部分落在 GPU1，一部分被运行时反量化
workspace、projection 和常驻 buffer 抵消。

更重要的是，当前 GGUF loader 对 vocab 大于 64K 的 `token_embd.weight` 会主动
反量化为 FP16。Qwen3-VL-4B 的 embedding shape 是：

```text
[151936, 2560]
151936 × 2560 × 2 bytes ≈ 742 MiB
```

这块 embedding 仍归 GPU0。除非以后实现量化 embedding lookup 或分块 token lookup，
Q4/Q5 的收益会明显小于文件大小差异。

修复前 Q4 + ridge conditioning 后为 GPU0/GPU1 `3685/1431 MiB`（合计
`5116 MiB`），INT8 + ridge 为 `2345/2339 MiB`（合计 `4684 MiB`）。
direct-to-final-owner 修复后 Q4 steady allocated 为约 `1802/1893 MiB`（合计
`3695 MiB`）：相对旧 Q4 少约 1421 MiB，相对 INT8 少约 989 MiB。参考图输出
逐元素一致，文本 relative RMS `2.05e-8`，所以这部分现在可以作为真实显存收益。

另外，Q4/Q5 不会减少 H3 的 packed activation、QKV、SDPA workspace、FP32 residual
或首尾参考图增加的 `S=37746 → S=41798`；它只减少一部分 Qwen 常驻权重和可能的
部分 Qwen 运行时占用。

## Ridge projection 已先切换

当前默认由 `mmh3-4b-ClipProj-v3.1.safetensors` 提供 `[2560] → [5120]` 的线性
映射。文件中的 `W` 是 FP16、原始 tensor 约 25.0 MiB；当前 no-host-mmap loader
为避免 FP16/FP32 matmul 隐式转换，会在 CPU cache 中读成 FP32，并在实际投影设备
保留约 50 MiB 的矩阵/统计量。

旧的 `-mlp` 文件包含 `32768×2560` 和 `5120×32768` 两个 FP16 layer，原始 tensor
约 480.1 MiB，另有运行时 activation/workspace。当前 Qwen MP 的最终 hidden state
经过 GPU1 的最后 32 层和 norm 后才返回，因此 projection/MLP 的 device cache 直接
落在 GPU1；`StoreMiniMaxH3ConditioningPeer` 随后才把 conditioning 复制到 GPU0。
所以换成 ridge 的直接收益主要是 GPU1 约 430 MiB，**不会等量减少 GPU0**；GPU0 的
H3 rank0、vision/embedding、请求 workspace 仍然存在。只有把释放出的 GPU1 余量用于
重新设计 split，才可能间接改变 GPU0 的安全余量。

需要特别区分 metadata：默认 ridge 文件自身的 `cos_test=0.6874`；MLP 文件记录
`cos_test=0.8116` 和 `cos_test_ridge=0.8144`，后者是该 MLP 校准实验中的 ridge
对照，不等于当前默认 ridge 文件的 metadata。最终仍以同 prompt/seed 的
conditioning、latent、视频和音频 gate 为准，不能把单个 cosine 数字当作画质结论。

### 已完成的 ridge/MLP 实测

下面是 direct-owner 修复前 `H3_QWEN_SPLIT=12` 的 projection 对照，保留作历史精度
数据；当前生产 Q4 split 为 4/32：

| 测试 | ridge | MLP | 说明 |
| --- | ---: | ---: | --- |
| 完整 smoke 首次执行 | 96.136 s | 136.747 s | 两次都 success；冷启动/服务状态会影响绝对值 |
| 完整 smoke 第二次 ridge | 2.011 s | — | 命中 ComfyUI cache，模型没有重新 load |
| conditioning-only | 22.056 s | 24.030 s | 两次均 finite；MLP 多一次 residual network |
| 完成后 GPU0 / GPU1 | 3158 / 2806 MiB | 3158 / 2806 MiB | terminal snapshot 中 projection cache 已不再存活 |
| 完整 smoke 完成后 GPU0 / GPU1 | 14104 / 2806 MiB | 14104 / 2806 MiB | 整卡差异由 H3/外围模块主导，不是 projection |

ridge 完整 smoke 连续两次还确认：ComfyUI RSS 约从 `1061.6` 增至 `1903.7 MiB`，
增长 `842.1 MiB`；encoder/projection 文件在 `/proc/<pid>/maps` 中均无命中。对应日志
和 summary 位于 `/home/regen/minimax-h3/clipproj-smoke-ridge-20260825.log`、
`/home/regen/minimax-h3/benchmark-clipproj-mlp-20260825.json` 及
`/home/regen/minimax-h3/trace-clipproj-*-20260825.json`。

## 当前实现状态与剩余缺口

### 1. GGUF loader 与 Qwen MP 已接通

ClipProj resident MP 入口现在会识别 Qwen3-VL GGUF，调用 ComfyUI-GGUF 的 direct
loader，再调用 `install_qwen_layer_parallel` 安装 4/32 layer-MP hook。纯文本与
参考图输出均 finite，ridge 输入/输出宽度分别为 2560/5120。

当前 tensor owner 已按下列规则实现并通过 gate：

- `blk.0..3`、embedding、vision 从磁盘直达 GPU0；
- `blk.4..35` 与 final norm 从磁盘直达 GPU1；
- 全量先落 GPU0 后再由 `_move_module` 复制 tail 的旧路径已删除；
- `token_embd.weight` 后续实现按 token row 解量化，避免固定 742 MiB FP16 展开。

### 2. no-host-mmap 与 resident 已兼容

当前路径使用 header-only reader、普通文件 I/O 和 8 MiB host staging，将 GGUF raw
storage 直接放到 CUDA 后常驻。Qwen 主文件、mmproj 和 ridge projection 在
`/proc/<pid>/maps` 的命中均为 0，连续 conditioning 不重新读模型。

当前 materialize 路径为：

```text
header-only GGUF reader
→ 8 MiB CPU staging
→ 原始 Q4_K_M shard 直接 materialize 到固定 GPU owner
→ 保留 quantized storage
→ 每层 forward 时 bounded dequant
```

embedding、vision Conv3d、projection cache 和模型 cache key 均已通过纯文本/参考图
conditioning；剩余压缩项只有量化 embedding row lookup。

### 3. V100 性能风险

标准 GGML Q4/Q5 在当前 ComfyUI-GGUF 路径中是“量化存储 + forward 前反量化”，而
不是 V100 上的 fused INT4 GEMM。每个 Qwen Linear 都可能产生 FP16 temporary weight；
Qwen 只执行一次 conditioning，所以总 wall time 未必不可接受，但视觉 tower 和
长 prompt 会变慢，不能根据文件大小推断整体加速。

## 剩余验证计划

文件审计、loader smoke、单参考图视觉 smoke、conditioning 对照和 no-mmap 审计已经
完成。剩余工作继续禁止普通 payload mmap：

1. **完整质量 gate**：固定 prompt/seed/workflow，对比 INT8 baseline 的 latent、
   视频画面、首尾几何和音频；conditioning 指标不能冒充成品质量。
2. **Q-only 参考图复测**：direct-owner + 旧 strided SDPA 的 1 MP 首尾参考图已经
   完整通过；下一轮用当前 Q-only 默认重跑，记录两张卡峰值和最终成品。
3. **量化 embedding**：只反量化实际 token row，记录 lookup 延迟、峰值和与当前 FP16
   展开 embedding 的逐元素误差。

## 暂定决策表

| 路线 | 显存预期 | 视觉兼容性 | V100 性能 | 当前决策 |
| --- | --- | --- | --- | --- |
| 4B INT8 ConvRot | 实测 4684 MiB allocated（12/24） | 已验证 | 精度基线 | fallback/对照 |
| 完整 VL Q5 GGUF | 预计低于 INT8，具体以文件为准 | 未验证 | 未测 | 后续精度折中候选 |
| 4B Q4_K_M + FP16 mmproj | direct-owner 实测约 3695 MiB（4/32） | 已跑通；vision token 相对 INT8 RMS 17.94% | 参考图 warm 749 ms | 生产默认（用户接受取舍） |
| text-only Q4/Q5 GGUF | 更低 | 不支持参考图 | 仅能做文本 smoke | 不用于 H3 I2V |
| Q4_PT/native W4A16 | 理论最低 | 当前路线不通 | V100 不适合 | 排除 |

Q4 已替换 INT8 成为默认，direct-owner 4/32、ridge projection 和 no-host-mmap 为当前
固定路线。Q4 direct-owner 的普通/双参考图 1 MP 请求已经成功；仍需公开的是
Q4-vs-INT8 成品质量 gate 尚未完成，以及当前 FP16 embedding 仍占约 742 MiB。
