# H3 在 V100 上的 Attention Kernel 后续路线

这是一份后续研发记录，不改变当前可运行的 V100 配置。目标是在双 Tesla V100（SM70）上，为 MiniMax H3 找到比 PyTorch memory-efficient SDPA 更快、同时保持 H3 全局 attention 语义的实现。

## 当前结论

本机当前运行环境：

- Tesla V100-SXM2 16 GB，计算能力 `7.0`。
- PyTorch `2.8.0+cu126`。
- Comfy-Kitchen `0.2.28`。
- H3 DiT 使用 `--use-pytorch-cross-attention`，实际走 PyTorch SDPA 的 memory-efficient backend。
- SageAttention 未安装；xformers 和 flash-attn 也未安装。

H3 的 attention 不是由模型类固定选择 SageAttention。当前调用链是：

```text
H3 Attention.forward
  ├─ Kitchen API: rms_rope_split_half(_)
  └─ ComfyUI optimized_attention
       └─ 当前启动参数选择 PyTorch SDPA
```

Kitchen 的 API 已经出现在 H3 的 RMSNorm/3-axis RoPE 路径中，但在本机的 `cu126` 环境下，`comfy/quant_ops.py` 会禁用 Kitchen CUDA backend；Triton backend 也默认禁用。因此这些调用会落到 eager 实现，不能把“调用了 Kitchen API”理解成已经使用了 fused CUDA kernel。

## 2026-08-26 已采用：只整理 Q 的 SDPA 输入布局

本轮没有改写 global attention，也没有改变 mask、softmax、head 或 packed sequence
语义。真正的热点是 H3 fused QKV 投影留下的跨步视图：传给 SDPA 的 Q/K/V 均有
3 倍 sequence stride。SM70 的 PyTorch efficient SDPA 对这个布局明显变慢；只把 Q
复制成标准 contiguous BHSD，就能选择更快的现有 kernel，K/V 保持原 view。

裸 `S=37746 / 28 heads / D=128` gate：

| 输入布局 | SDPA 端到端 | 相对原始 | 数值 |
| --- | ---: | ---: | --- |
| fused-QKV strided | 703.092 ms | 1.000× | baseline |
| 只 compact Q | 620.228 ms | 1.134× | `max_abs=0` |
| compact Q/K/V | 619.002 ms | 1.136× | `max_abs=0` |

完整 50 层、双 rank、同一 Q4+Turbo 输入的结果：

| 路线 | forward | rank0 peak allocated | reserved | 输出 SHA256 |
| --- | ---: | ---: | ---: | --- |
| 原始 strided | 47.442 s | 10225.7 MiB | 10336 MiB | `1b278b8e...98e498fa` |
| compact 全 Q/K/V | 43.121 s | 10166.7 MiB | 11372 MiB | 同上 |
| **只 compact Q** | **44.051 s** | **9973.4 MiB** | **10336 MiB** | 同上 |

全 Q/K/V 只比 Q-only 快约 0.93 s，却要多复制 K/V；裸 gate 多约 516 MiB 输入副本，
完整 allocator reserve 比 Q-only 高约 1 GiB。生产因此选择 Q-only：相对旧基线快
`7.15%`，不增加 allocator reserve，输出 hash 完全一致。小序列复制成本不划算，
所以只在 `S >= 4096` 启用。

启动器默认：

```bash
H3_TP_COMPACT_QKV=q
H3_TP_COMPACT_QKV_MIN_SEQUENCE=4096
```

可用 `H3_TP_COMPACT_QKV=0` 原样回退；`all` 只保留离线实验。该优化继续调用
PyTorch efficient SDPA，并保留 FP32 residual、FP32 attention output、FP32 MLP
和 FP32 NCCL，不是低精度近似。

真实常驻服务的无参考图 1 MP 请求中，四个 `S=37746` forward 都 finite、没有模型
重载，依次为 `43.979 / 44.363 / 47.026 / 51.623 s`。前两次吻合离线 gate；后两次
GPU1 升到 82°C，busy SM clock 最低降到 570 MHz，rank0 collective 等待由约
1.83 s 增到 8.55 s，属于热降频造成的 rank skew。该请求按用户要求在第 4 次
forward 后中断，未保存最终 latent，因此不是新的完整画质 gate。证据见
`results/h3_1mp_no_ref_compact_q_20260826_summary.json`。

1 MP 首尾参考图在 direct-owner 旧布局下已经完整成功，整卡峰值约
`15200/15466 MiB`；Q-only 的参考图复测本轮没有继续，列入下一轮 gate，不能仅凭
无参考图结果宣称通过。

### TileLang 调研结论

本轮没有安装 TileLang。PyPI `tilelang 0.1.13` 和源码包含 SM70 声明、
`mma_sm70.h` 与部分 SM70 测试，但现有普通 FlashAttention 示例没有证明会在 V100
上生成有效的 Tensor Core MHA；部分 SM70 FP16 transpose GEMM 测试还明确落到 FMA
fallback。下一轮若继续，只先跑 `S=2048` 官方 MHA、检查生成源码和精度；不能因为
“支持 sm70”就默认会超过当前 PyTorch SDPA。

## Kitchen kernel：值得做，但先确认真实 backend

Kitchen kernel 值得写进路线，原因是它优化的是每张卡上的本地计算，而不是跨卡通信：

- RMSNorm、3-axis RoPE、layout transform 等操作会在每个 H3 block 或每个 rank 重复执行。
- 如果 H3 做 2-way TP，每个 rank 仍然可以独立复用同一套本地 kernel；只有 row-parallel 输出边界需要 NCCL。
- 这类 kernel 不需要改变模型语义，风险通常低于重写 global attention。
- 正确的 fused kernel 可以减少 kernel launch、临时 tensor 和显存读写，收益会直接体现在每个 denoise step。

但当前第一步不是“马上打开 Kitchen”，而是确认 dispatch。当前环境的事实应按下面方式理解：

| 观察到的现象 | 实际含义 |
| --- | --- |
| H3 调用了 `rms_rope_split_half(_)` | 只说明调用了 Kitchen 风格的 API |
| `comfy/quant_ops.py` 禁用 Kitchen CUDA backend | 该调用可能落到 eager 实现 |
| Kitchen Triton backend 默认禁用 | 不代表本机实际编译/执行了 Triton kernel |
| 命令行带 Kitchen/Sage 相关开关 | 仍需看 backend dispatch 和 kernel 日志，不能只看启动参数 |

因此，任何 benchmark 都要记录实际 backend：`Kitchen CUDA`、`Kitchen Triton` 还是 `eager`。如果最后跑的是 eager，不能把它的耗时写成 Kitchen kernel 的性能。

### 值得优先看的本地 kernel

| 热点 | 价值 | 推荐实现方向 | 注意事项 |
| --- | --- | --- | --- |
| RMSNorm + 3-axis RoPE | 中高；每层重复且容易验证 | 先验证 Kitchen 现有 kernel；不行则做 SM70 CUDA/Triton 窄 kernel | 保留 FP32 统计和 H3 的数值稳定性岛 |
| H3 global attention | 高，但实现风险最高 | H3 专用 SM70 CUDA/CUTLASS；Triton 只做原型 | 不能用 `na3d` 的局部可见范围替代 |
| Q4/convrot Linear | 潜在收益高 | 量化布局确认后再做 CUDA/CUTLASS 或已有 Kitchen 路径 | 先解决 Q4 shard，再谈融合 |
| packed layout / transpose | 中等 | Triton 或 CUDA layout kernel | 必须避免隐式 materialize 大 tensor |

Kitchen 的 `na3d` 不属于 H3 global attention 的直接替代方案。它是局部 3D neighborhood attention，H3 的 packed sequence 同时含 text、reference、audio 和 video token；改变可见范围会改变模型计算，而不是单纯换 kernel。

### Kitchen、Triton、CUDA 和 NCCL 的边界

```text
Kitchen：提供已有的算子抽象/dispatch 和可复用的本地 kernel
Triton：快速验证单卡本地 kernel，确认 SM70 能否生成稳定代码
CUDA/CUTLASS：需要 SM70 控制、Q4 layout 或最终性能时使用
NCCL：只做 rank 间 all-reduce/reduce-scatter/all-gather/send/recv
```

Kitchen/Triton kernel 内不要直接调用 NCCL。正确的数据流是：

```text
本地 Kitchen/Triton/CUDA kernel
        ↓
本地 partial output
        ↓
host/C++ 侧 NCCL collective
        ↓
下一个本地 kernel
```

在 TP 中，qkv projection 和本地 attention 的 kernel 只看到本 rank 的 heads；`out_proj` 或 `fc2` 产生 partial hidden 后，才在 kernel 外调用 NCCL `all_reduce`。因此原来的单卡 kernel 可以复用，但不能继续硬编码完整的 56 heads、7168 attention width 或完整 Q4 weight layout。

### Kitchen kernel 的实施顺序

1. 在不加载大模型的情况下确认 backend dispatch，分别测 Kitchen CUDA、Kitchen Triton 和 eager（如果路径可用）。
2. 用随机 FP16 输入对 RMSNorm、RoPE、layout transform 做逐元素/逐 token 正确性对比。
3. 先优化 RMSNorm + RoPE 这类窄 kernel，再决定是否值得做更大的融合。
4. 单独实现 H3 global attention 原型，并与 PyTorch memory-efficient SDPA 对比；不要拿 `na3d` 做等价性测试。
5. 在 FP16 单卡 block 通过后，把相同 kernel 接到 TP 的 local shard，collective 仍由 NCCL 管理。
6. 最后再针对 Q4/convrot 和 Turbo LoRA 做融合，避免把量化排布错误与 kernel 错误混在一起。

历史上 4B ClipProj 路径曾出现 Inf/NaN，最终定位为 BF16 Turbo LoRA 被同宽按 FP16 字节解释，而不是 ClipProj 本身；该问题已经通过 BF16 materialize 后数值转换修复。当前 kernel gate 使用已验证 finite 的 4B INT8 + v3.1 MLP conditioning，并继续要求 kernel 优化不能掩盖任何 conditioning/LoRA 数值问题。

### Kitchen kernel 验收标准

- 日志明确显示实际执行的是目标 Kitchen/CUDA/Triton backend，而不是 eager fallback。
- 与 PyTorch/eager 参考相比，输出无 Inf/NaN，误差在固定 dtype 和长度下可解释且不随 block 数异常累积。
- 至少测试 `S=128`、`2048` 和接近 1 MP H3 的长序列。
- 记录单次耗时、显存峰值、临时 buffer 和 kernel launch 数。
- 在 `448x256` 1-step、`832x480` 和 `1344x768`/124 帧/4-step 上分别验证。
- TP 模式下每个 rank 的本地输入 shape、head 数和 stride 正确，NCCL 前后的 tensor 都 finite。
- 新 kernel 通过 H3 专用开关选择，保留 PyTorch SDPA/eager fallback，不注册成全局 ComfyUI attention。

## 为什么不能直接搬 Sage 或 `na3d`

### SageAttention

现代 SageAttention 的主要 CUDA 路径面向 SM80 及更新架构，常用实现还依赖 Ampere 之后的指令或低精度数据类型。V100 的 SM70 不能只通过修改编译架构参数获得同样的 kernel。

此外，ComfyUI 的 Sage wrapper 在 kernel 失败时可能回退到 PyTorch attention；命令行出现 Sage 参数不等于每一步都真正运行了 Sage kernel。

### Kitchen `na3d`

Kitchen 的 `na3d` 是局部 3D neighborhood attention，接口要求类似：

```text
[B, T, H, W, heads, head_dim]
```

它的 CUDA/Triton 实现明确要求计算能力至少为 SM80。即使使用 eager fallback，也只是用分块/局部窗口模拟，不能作为 H3 的等价替换。

H3 的实际 attention 输入是：

```text
[1, 56, sequence_length, 128]
```

其中 sequence 混合了 text、reference、audio 和 video token，模型训练的是全局 packed attention。把它重排成 3D 窗口并限制可见范围会改变模型本身，而不只是改变实现。

## 推荐移植目标

后续应该实现一个 **H3 专用的 SM70 global attention kernel**，而不是强行移植 `na3d`。

H3 DiT 的固定接口很适合先做一个窄 kernel：

- `q/k/v`: `[1, 56, S, 128]`。
- 无 mask、无 causal、无 GQA。
- Q/K 已经完成 RMSNorm 和 3-axis RoPE。
- 输入和输出保持 FP16。
- softmax 统计和累加使用 FP32，避免 V100 溢出和数值漂移。
- 采用 online softmax，不能 materialize `S x S` attention matrix。
- 编译目标为 `sm_70`，不能使用 FP8、NVFP4 或 Ampere-only 指令。

第一版只替换 `optimized_attention` 这一段，保留现有 QKV projection、RMSNorm、RoPE、out projection 和 H3 的 FP32 stability islands。这样更容易定位收益和数值变化。

可能的实现顺序：

1. CUDA C++ / CUTLASS 老版本的 FP16 tiled online-softmax kernel。
2. 如果 Triton 3.4 在本机能稳定生成 SM70 的 IEEE FP16 dot，再做一个 Triton 原型用于快速迭代。
3. 只有在 attention kernel 已经证明有效后，才考虑把 QKV、RMSNorm、RoPE 或输出 projection 做更大的融合。

不要把新 kernel 注册成全局 ComfyUI attention。应通过 H3 专用开关或 H3 attention wrapper 选择，并始终保留 PyTorch SDPA fallback，避免影响其他模型。

## 验证门槛

### 阶段 0：无模型 kernel 测试

- 用随机 FP16 Q/K/V 与 PyTorch SDPA 对比输出。
- 测试 `S=128`、`2048` 以及接近 1 MP H3 的长序列。
- 检查最大绝对误差、相对误差、cosine similarity、Inf/NaN。
- 记录 kernel 单次耗时、峰值显存和显存释放后的可用空间。
- 不加载 GGUF、VAE、LoRA，也不修改运行中的 ComfyUI 服务。

### 阶段 1：H3 1-step latent

- 使用同一 seed、同一 Q4 DiT、同一 conditioning。
- 对比 PyTorch SDPA 与新 kernel 的 video/audio latent。
- 两条路径都必须有限；若误差随序列长度或 block 层数累积，停止推进。
- 先跑 `448x256`，再跑 `832x480`。

### 阶段 2：真实速度

- 以 `1344x768`、124 帧、Turbo 4-step 为最终参考，而不是只看小张量 benchmark。
- 记录每个 denoise forward、每个 block、GPU0/GPU1 峰值显存和主机 RSS。
- 只有在保持 latent 质量的前提下，attention 部分有稳定收益，才合入默认路径。

## 当前基线

在 GPU1 上对 `[1, 56, 2048, 128]` 的裸 attention 做过小测试：

```text
PyTorch SDPA:    约 3.6 ms，峰值约 140 MiB
split attention: 约 5.8 ms，峰值约 1 GiB
```

这说明当前 PyTorch memory-efficient SDPA 已经是一个较强的 V100 基线。新 kernel 必须在真实长序列上超过它，不能只比 split attention 快。

## 暂不做的事情

- 不在 V100 上直接启用 `--use-sage-attention`。
- 不用 `na3d` 替换 H3 的全局 attention。
- 不为实验升级 PyTorch/CUDA 或替换现有工作环境。
- 不把 Kitchen 的 eager fallback 当作 fused kernel。
- 不为了测试加载官方大模型或恢复 host mmap 路径。

当前生产路径继续使用 PyTorch SDPA，并默认在长序列只 compact Q；自定义 SM70
kernel、TileLang 和 compact 全 Q/K/V 都保留为后续独立实验，不进入当前默认。
