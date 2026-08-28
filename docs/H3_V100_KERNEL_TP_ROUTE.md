# MiniMax H3 双 V100 Kernel 与 Tensor Parallel 技术路线

更新时间：2026-08-26

本文记录这台双 Tesla V100-SXM2-16GB 机器上，MiniMax H3 从单卡量化推理走向双卡 Tensor Parallel（TP）的真实实现路线。它既是学习材料，也是后续改动的 review 基线。

文中严格区分三种状态：

- **已验证**：有保存的代码、数值结果和性能 JSON。
- **原型/淘汰**：组件数学成立或做过性能实验，但没有进入生产路径。
- **后续研究**：不影响当前可用性，只是下一轮可能继续优化的方向。

> 本轮冻结总表见 [`H3_V100_FREEZE_20260826.md`](H3_V100_FREEZE_20260826.md)。当前生产默认（2026-08-26）：Q4 Qwen 使用 direct-owner `4/32` layer-MP，
> Video VAE 使用 `12/24` decoder MP；两者都不是会产生 hidden-state 漂移的严格
> Qwen TP。H3 长序列使用 PyTorch efficient SDPA，并只 compact Q
>（`H3_TP_COMPACT_QKV=q`，`S>=4096`）。18/18、12/24 Qwen 数据在下文保留为
> 历史基线。direct-owner 的 1 MP 首尾同图参考请求已经完整通过；Q-only 下的同规格
> 复测留到下一轮。

## 1. 目标与不可破坏的约束

最终验收目标：

```text
1344 × 768
124 frames
Turbo 4 steps
Qwen3-VL-4B Q4_K_M + FP16 mmproj + mmh3-4b-ClipProj-v3.1 ridge
H3 Q4_0 + Turbo LoRA
两张 V100 同时计算每个 DiT block
模型跨请求常驻，不反复 load/unload
保存 latent、MP4、逐阶段耗时、GPU/RSS 峰值
```

硬约束：

1. 模型默认从 `/mnt/GALAX/minimax-h3/models` 读取。
2. 不允许 GGUF/safetensors payload 做整文件 host mmap。
3. 不允许先在 CPU 或单卡完整反量化再切 shard。
4. 两卡 TP 必须是两个长期 `torch.distributed` rank + NCCL；benchmark 可用 `torchrun`，Comfy 生产可用 parent-rank0 + child-rank1。`DataParallel`、组件分卡或普通 `device_map` 不算 TP。
5. H3 的 packed global attention 语义不能改成局部窗口。
6. V100 是 SM70，不依赖 FP8、NVFP4、BF16 tensor core 或 Ampere-only 指令。
7. FP32 数值稳定岛必须保留，不能为了 benchmark 好看退回会溢出的纯 FP16 路径。

当前常驻服务还有 cgroup 保护：

```text
MemoryHigh=6500M
MemoryMax=7G
MemorySwapMax=256M
H3_NO_HOST_MMAP=1
```

## 2. 当前完成状态

| 模块 | 状态 | 结论 |
| --- | --- | --- |
| no-host-mmap loader + RAM cgroup | 已验证 | 4–8 MiB bounded staging；两 rank 的模型 payload map 均为 0；服务受 7G hard cap 保护 |
| 4B Encoder + ClipProj v3.1 ridge | 已验证并设为默认 | `Qwen3VL-4B-Instruct-Q4_K_M` + FP16 mmproj + 26 MB ridge 以 direct-owner 4/32 layer MP 常驻；steady allocated 约 1802/1893 MiB，没有每请求重载。相对 INT8 的参考图全局/vision-token RMS 为 8.47%/17.94%，属于已接受取舍，不是数值等价 |
| Q4 + Turbo LoRA 乱码修复 | 已验证 | 根因是 BF16 LoRA 字节被直接按 FP16 解释，不是 4B encoder |
| PyTorch efficient SDPA + Q-only layout | 已验证并启用 | 只整理 Q 的 BHSD 布局，不改全局 attention 算术；`S=37746` 完整 50 层 47.442 → 44.051 s，输出 hash 完全一致 |
| SM70 Q4_0 直解量化 | 已验证、默认 opt-in | 真实 Q4 shard 与 50 层 TP 数值通过；`S=2048` 快约 13.1%，`S=37746` 快约 0.9%，生产默认仍用 eager |
| SM70 fused RMSNorm + partial RoPE | 已验证并启用 | `S=2048` 比 eager 快约 16.35×；same-seed latent/MP4 bitwise identical |
| FP16 Tensor Core / FP32-output wide Linear | 已验证并启用 | 保留 FP32 residual update；真实 `S=2048` TP block 从 35.866 ms 降至 21.165 ms |
| fused FP32 RMSNorm/AdaLN + SwiGLU/scale | 已验证并启用 | 保留稳定岛，完整 50 层 `S=2048` 从 1033.06 ms 降至 944.79 ms，耗时降低约 8.5% |
| 真实 Q4_0 + Turbo LoRA 双卡 TP | 已验证并启用 | QKV/FC1 column parallel，out/FC2 row parallel，FP32 NCCL；两 rank 输出 bitwise identical |
| Qwen3-VL-4B 严格 TP（INT8 基线） | 已测、暂不启用 | 真实 36 层 TP 比 18/18 MP 快约 19.7%（S=256）/36.8%（S=512），但 full-vs-TP hidden 漂移 2.12%/2.56%，未过精度门槛；Q4 生产仍用数值等价 layer-MP，direct-owner 默认 4/32 |
| 完整 50 层 DiT 常驻服务 | 已验证 | Comfy rank0 + 长期 rank1 子进程；十次 forward 均 `models_reloaded=false` |
| 832×480、124 帧、4-step | 已验收 | 采样 42.03 s；清晰连续 H.264 + AAC；音视频 finite |
| 1344×768、124 帧、4-step | 已验收 | 采样 196.46 s；解码 60.15 s；两卡整卡峰值 15.87/14.50 GiB；音视频 finite |
| 视频 VAE 常驻 / 音频 VAE 阶段加载 | 已验证 | 视频 VAE 以 12/24 decoder MP 常驻，音频 VAE 仅在 decode 阶段从 `/mnt/GALAX` 读取；TP/4B encoder 不卸载 |

### 2026-08-25 当前工作区复测

这次复测使用当前工作区代码，并先停掉空闲的 ComfyUI 进程再运行独立
`torchrun`，避免旧模型占用显存污染结果。完整 50 层 H3 backbone 的结果如下：

| packed sequence | 对应尺寸 | warm forward | NCCL all-reduce | 两 rank 峰值显存 | 数值结果 |
| ---: | --- | ---: | ---: | ---: | --- |
| `2048` | kernel 基准 | `945.6 ms` | `65.0 ms` | `6.74 / 6.74 GiB` | rank bitwise identical、finite |
| `14880` | `832×480 / 124f` | `9.695 s` | `498 ms` | `7.91 / 7.90 GiB` | `max_abs=0`、finite |
| `37746` | `1344×768 / 124f` | `47.506 s` | `1.380 s` | `9.99 / 9.96 GiB` | `max_abs=0`、finite |

ComfyUI 端到端 448×256/22f/1-step 的第二次请求命中常驻路径，TP forward
为 `538.2 ms`，没有重新加载模型；输出为正常的 22 帧 H.264 + 32 kHz
双声道 AAC。所有 benchmark 的 `payload_mmap=false`，rank RSS 峰值约
`1.3 GiB`。完整汇总和原始报告见
`results/h3_tp_full_current_summary_20260825.json`。

这里的“全 TP”应限定为 compute-heavy 的 50 层 H3 backbone：QKV/FC1 是
column parallel，out_proj/FC2 是 row parallel，每层两次 NCCL all-reduce。
condition/input projection、两层 token refiner、final heads 仍是小型外围
模块；Qwen 当前是 layer/pipeline split，VAE 是 decoder model parallel。把
这些边界强行改成 TP 会增加通信，当前没有性能收益依据，也不应为了名义上的
“全模块 TP”牺牲数值稳定性。

### Qwen 4B 的真实权重 MP/TP gate（2026-08-25）

为避免只看合成权重，使用 `/mnt/GALAX` 上的
`qwen3vl_4b_int8_convrot.safetensors` 逐层读取 36 个 language block；每次只
保留一个 CPU layer buffer，读完立即放到 GPU 并丢弃，未建立完整 CPU state
dict，也未做 payload mmap。Q/K RMSNorm 也纳入了比较。严格 TP 是诊断路径，
不是当前 ComfyUI loader。

| 序列 | 18/18 layer-MP | 严格 TP | TP 加速 | MP/TP 峰值显存 | full-vs-TP |
| ---: | ---: | ---: | ---: | --- | --- |
| 256 | 134.99 ms | 112.73 ms | 19.7% | MP 1823/1822 MiB；TP 1799/1800 MiB | relative RMS 2.12%，cosine 0.999777，未通过 |
| 512 | 174.76 ms | 127.69 ms | 36.8% | MP 1885/1888 MiB；TP 1836/1837 MiB | relative RMS 2.56%，cosine 0.999678，未通过 |

两种 TP rank 的输出本身均 `max_abs=0` 且 finite；问题是将每层的 head/FFN
partial reduction 传播 36 层后，与单卡完整 attention 的 hidden state 累积漂移
超过当前 `relative RMS <= 0.003`、`cosine >= 0.9999` 门槛。因而“快”不能单独
决定路线：严格 TP 保持禁用；当前 Q4 encoder 使用 direct-owner 4/32 layer-MP，不改变每层
数学语义。这里的 18/18 INT8 数据只作为严格 TP 的历史数值基线，不是当前默认。
conditioning 的数值路径。保存的原始报告为：

- `results/h3_qwen_int8_full_s256_20260825.json`
- `results/h3_qwen_int8_full_s512_20260825.json`
- `results/h3_qwen_int8_tp_layer0_qknorm_20260825.json`

严格 Qwen TP 只有在后续能把 36 层漂移压回门槛后，才值得做成 opt-in；当前不
为了名义上的“全 TP”替换生产 MP。

## 3. H3 block 的真实尺寸

当前 H3 DiT block：

```text
hidden_size = 5376
heads       = 56
head_dim    = 128
inner       = 56 × 128 = 7168
ffn         = 14336
layers      = 50
```

四个主要矩阵采用 PyTorch 的 `W[out, in]` 语义：

| 矩阵 | 完整 shape | H3 输出顺序 |
| --- | ---: | --- |
| `qkv_proj` | `[21504, 5376]` | `[Q(7168), K(7168), V(7168)]` |
| `out_proj` | `[5376, 7168]` | 普通输出投影 |
| `fc1` | `[28672, 5376]` | `[gate(14336), up(14336)]` |
| `fc2` | `[5376, 14336]` | SwiGLU down projection |

TP=2 后每 rank：

```text
local_heads = 28
local_inner = 3584
local_ffn   = 7168
```

## 4. 为什么 QKV/FC1 不能直接把输出维一刀两半

### 4.1 QKV 的正确切法

错误切法：

```text
rank0 = full_qkv[0:10752]
rank1 = full_qkv[10752:21504]
```

这样 rank0 会得到完整 Q 和半个 K，rank1 会得到半个 K 和完整 V，无法做 28 个本地 attention heads。

正确切法是每个 rank 分别取 Q、K、V 中相同的 head 范围：

```text
rank0 rows:
  Q [    0,  3584)
  K [ 7168, 10752)
  V [14336, 17920)

rank1 rows:
  Q [ 3584,  7168)
  K [10752, 14336)
  V [17920, 21504)
```

本地重新拼成：

```text
[local_Q, local_K, local_V] -> [10752, 5376]
```

### 4.2 FC1 的正确切法

FC1 的输出是 `[gate, up]`，不能让 rank0 只拿 gate、rank1 只拿 up。

```text
rank0 rows:
  gate [    0,  7168)
  up   [14336, 21504)

rank1 rows:
  gate [ 7168, 14336)
  up   [21504, 28672)
```

每 rank 得到 `[local_gate, local_up]`，本地执行：

```text
SiLU(local_gate) * local_up -> [S, 7168]
```

## 5. 两种 Parallel Linear 的数学边界

### 5.1 Column Parallel：QKV 与 FC1

权重沿输出行切分：

```text
Y_r = X @ W_r^T
```

每 rank 直接保留自己的输出，不需要 collective：

```text
QKV_r -> 28 local heads -> local attention
FC1_r -> local SwiGLU
```

### 5.2 Row Parallel：out_proj 与 FC2

输入和权重输入列一起切分：

```text
Y_r = X_r @ W_r^T
Y   = sum_r(Y_r)
```

因此每个 block 有两次 NCCL all-reduce：

```text
local attention -> local out_proj partial --+
                                             +-> all_reduce -> full hidden
local SwiGLU   -> local fc2 partial --------+
                                             +-> all_reduce -> full hidden
```

所有 rank 在 block 边界重新持有相同的 `[S, 5376]` residual，下一层可以继续采用相同分片。

## 6. 真实 GGUF Q4_0 字节布局

模型：

```text
/mnt/GALAX/minimax-h3/models/diffusion_models/
minimax_h3_fl2va_pruned_fp8_Q4_0.gguf
```

NoMmap reader 只读取 1 MiB header prefix；模型文件为 `11,377,542,880` bytes，payload 没有映射。

标准 GGML `Q4_0`：

```text
32 values / block
18 bytes / block = FP16 scale(2 bytes) + packed nibbles(16 bytes)
```

block 0 实际数据：

| 矩阵 | shape | offset | 总字节 | 每行字节 |
| --- | ---: | ---: | ---: | ---: |
| out_proj | `[5376,7168]` | 1,927,328 | 21,676,032 | 4,032 |
| qkv_proj | `[21504,5376]` | 23,603,872 | 65,028,096 | 3,024 |
| fc1 | `[28672,5376]` | 88,631,968 | 86,704,128 | 3,024 |
| fc2 | `[5376,14336]` | 175,336,096 | 43,352,064 | 8,064 |

### 6.1 输出行切分

QKV/FC1 的每一行在文件中连续，按上面的分段 row ranges 做普通 `pread/readinto` 即可。每 rank 读取多个连续范围，再在本地按 H3 期望顺序拼接。

### 6.2 输入列切分

out_proj/FC2 的输入列在每一行内部：

```text
out_proj local input = 3584 = 112 Q4 blocks = 2016 bytes/row
fc2 local input      = 7168 = 224 Q4 blocks = 4032 bytes/row
```

虽然两者都刚好落在 Q4 block 边界，但不能把整个 tensor 文件范围一刀两半。必须对每个输出行选择本 rank 的 byte window：

```text
out_proj rank0: each row bytes [0, 2016)
out_proj rank1: each row bytes [2016, 4032)

fc2 rank0: each row bytes [0, 4032)
fc2 rank1: each row bytes [4032, 8064)
```

当前 reader 以 4 MiB 普通 CPU staging 分批读完整行，再只复制本 rank 的列窗口到 CUDA Q4 storage。CPU 同时存在的数据有严格上限，不会出现完整模型副本。

完整布局报告：`results/h3_q4_tp_layout.json`。

## 7. Turbo LoRA 的分片

LoRA 文件：

```text
/mnt/GALAX/minimax-h3/models/loras/
minimax_h3_turbo_v4_step600_ema.safetensors
```

矩阵语义：

```text
W_eff = W + B @ A
A = [rank, in]
B = [out, rank]
rank = 64
```

block 0 实际 shape：

| 模块 | A | B |
| --- | ---: | ---: |
| qkv_proj | `[64,5376]` | `[21504,64]` |
| out_proj | `[64,7168]` | `[5376,64]` |
| fc1 | `[64,5376]` | `[28672,64]` |
| fc2 | `[64,14336]` | `[5376,64]` |

分片规则来自 base weight 的同一维度：

| Parallel 类型 | Base shard | LoRA shard |
| --- | --- | --- |
| QKV/FC1 column parallel | 切 W 输出行 | A 复制；B 按 QKV 或 gate/up 输出行分段 |
| out_proj/FC2 row parallel | 切 W 输入列 | A 切输入列；B 复制 |

行并行 LoRA 的等价性：

```text
delta_r = B @ (A_r @ X_r)
delta   = sum_r(delta_r)
```

所以 base partial 和 LoRA partial 在同一个 FP32 all-reduce 中相加，不增加 collective 次数。

当前验证：

- QKV/FC1 的 B shard 重组后与完整 BF16 B 逐 bit 相同。
- out_proj/FC2 的 A shard 重组后与完整 BF16 A 逐 bit 相同。
- 复制的 A/B 逐 bit 相同。
- 所有 LoRA tensor finite。
- source BF16 先按 BF16 materialize，再数值转换为 FP16/FP32；禁止同宽位 reinterpret。

AdaLN 是另一类：pruned H3 把完整时间投影折叠成 curve table，AdaLN LoRA 需要复制小 factor 并按现有 E-grid runtime injection 处理，不能套上述四个大矩阵的 TP Linear。

## 8. V100 上必须保留的数值稳定岛

当前生产服务启用：

```text
H3_FP32_RESIDUAL=1
H3_FP32_ATTN_OUT=1
H3_FP32_MLP=1
H3_FP32_MLP_CHUNK_ROWS=2048
H3_V100_FP32_TC=1
```

TP block 必须匹配：

```text
replicated residual                 FP32
norm -> QKV -> Q/K norm -> SDPA     FP16
attention out_proj partial          FP32
attention all-reduce                FP32
norm -> FC1                         FP16
SwiGLU                              FP32
FC2 partial                         FP32
MLP all-reduce                      FP32
```

### 8.1 为什么“FP32 稳定”原来会很慢

原生产实现把 attention 输出和 SwiGLU 输出升成 FP32 后直接调用 GGUF Linear：

```text
x_FP32 -> Q4 weight dequantize to FP32 -> FP32 CUDA-core GEMM -> y_FP32
```

这样能避免 FP16 输出超过 `65504`，但也把 `out_proj` 和 `fc2` 的整张权重临时展开成 FP32。V100 没有 TF32，FP32 GEMM 不走 FP16 Tensor Core，因此这两个投影成为长序列热点。

新路径只保留真正需要的 FP32 边界：

```text
x_FP32 -> safe FP16 input
Q4     -> FP16 weight
FP16 x FP16 Tensor Core, FP32 accumulation/output
       -> y_FP32
```

PyTorch 2.8 的 CUDA `torch.mm(..., out_dtype=torch.float32)` 允许输入和权重为 FP16，同时直接生成 FP32 输出；中间不会先物化一个可能溢出的 FP16 输出。

### 8.2 FC2 的逐 token、二次幂缩放

SwiGLU 必须留在 FP32，因为一对有限的 FP16 gate/up 在相乘时仍可能溢出。SwiGLU 输出本身也可能超过 FP16 范围，所以不能直接 `.half()`。

对每个 token row `i`：

```text
m_i = max_j(abs(x_ij))
s_i = 2 ^ ceil(log2(max(1, m_i / 32752)))
x'_i = FP16(x_i / s_i)
y_i  = s_i * MM_FP16xFP16_to_FP32(x'_i, W_FP16)
```

`s_i` 是精确的二次幂；对正常 FP32 数做除法和恢复时不会额外损失有效位。`32752` 留出 FP16 转换余量。attention out 原本就是 SDPA 产生的 FP16，只是在投影前升到 FP32，因此它不需要 row scaling。

溢出 gate 使用 `1e6..1e8` 的有限 FP32 输入：未经缩放转 FP16 会出现 Inf；缩放后最大绝对值 `32714.84`，全部 finite，输出相对 FP32 GEMM 的 relative RMS 为 `1.47e-4`。

### 8.3 DynamicVRAM/VBAR 的正确接入边界

生产服务的 GGUF 权重不是普通常驻 Tensor；未 fault 的层是 meta/VBAR 占位。不能直接调用第三方 `GGMLOps.cast_bias_weight`，否则会绕开 materialize 生命周期并报：

```text
Cannot copy out of meta tensor; no data!
```

正式实现通过 `comfy.ops.CastBiasWeightContext(..., offloadable=True)`：

1. 从 `/mnt/GALAX` 的 file slice 或 VBAR fault 当前层。
2. 只把当前权重 materialize/dequantize 为 FP16。
3. 在 context 内完成 Tensor Core GEMM。
4. context 退出时按 ComfyUI 的正式生命周期释放临时 cast。

这次失败的 bring-up 日志保留在 `results/h3_fp32_tc_e2e/cold_seed2011_v3/`，不能删除后假装第一次就成功。

### 8.4 Turbo LoRA 与保守回退

wide wrapper 在 Turbo 注入之前成为 Linear 的 base forward；之后 `BypassForwardHook` 仍按原语义执行：

```text
base_FP32(x) + B_FP32 @ (A_FP32 @ x_FP32)
```

row scaling 只发生在 base GEMM 内部，LoRA 看见的仍是原始 FP32 activation。当前生产日志确认 `qkv_proj.forward_owner=BypassForwardHook`，不是把 LoRA 静默绕掉。

以下情况自动回到原 FP32 路径：

- 非 SM70；
- 输入不是 FP32 CUDA tensor；
- 不是 H3 `7168->5376` / `14336->5376` 投影；
- bias 或 merge-mode weight patch；
- training / requires-grad。

首个实验把 out_proj/FC2 和 residual 都放在 FP16，虽然 cosine 很高，绝对误差和溢出风险都不符合真实 H3 路径，因此该实验只保留为失败记录，不能作为验收数据。

标准 TP 会改变 reduction order，因此不要求与单卡 dense GEMM bitwise identical。当前 block gate 使用：

```text
rank0 output == rank1 output bitwise
all values finite
cosine vs dense >= 0.999999
relative RMS error vs dense <= 0.002
```

最终仍必须以 50 层 same-seed latent 和视频/音频结果为更强验收。

### 8.5 保留 FP32 边界、减少中间张量与 kernel launch

完整 TP 接入后，`S=2048` profiler 显示原有防 NaN 路径还有两类纯框架开销：

1. 每个 block 的两次 FP32 RMSNorm 会先物化 `[S,5376]` FP32 tensor，再 cast FP16，并按 text/audio/video segment 分多次做 AdaLN scale/shift。
2. FC1 后由多个 eager op 完成 FP32 `SiLU(gate) * up`、逐 token `amax/log2/ceil/exp2`、缩放和 safe FP16 cast。

`h3_v100_fp32_ops.py` 为 SM70 增加两个生产 kernel：

```text
FP32 residual
  -> FP32 RMS statistics
  -> RMSNorm
  -> 与 eager 相同的 FP16 rounding boundary
  -> token->AdaLN-row 查表
  -> FP16 scale/shift
  -> FP16 attention/MLP branch

FP16 [gate,up]
  -> FP32 SwiGLU
  -> per-token power-of-two scale
  -> FP32 SwiGLU（给 FC2 LoRA）
  +  safe FP16 branch（给 base FC2 Tensor Core GEMM）
  +  FP32 row scale（恢复 base FC2 输出）
```

这不是把 FP32 稳定路径改回 FP16。生产中仍然保持：

- replicated residual 为 FP32；
- attention out_proj 和 FC2 直接输出 FP32；
- Turbo LoRA 的 row-parallel partial 为 FP32；
- 每层两次 NCCL all-reduce 为 FP32；
- residual gate/update 为 FP32。

独立 `S=2048` gate：

| 操作 | eager | fused | speedup | 数值结论 |
| --- | ---: | ---: | ---: | --- |
| RMSNorm + FP16 materialize + AdaLN | 0.6555 ms | 0.09295 ms | 7.05× | finite，relative RMS `2.97e-4`，cosine `1.0` |
| FP32 SwiGLU + scale + safe cast | 1.1078 ms | 0.18893 ms | 5.86× | finite；SwiGLU relative RMS `4.84e-10` |

完整 50 层 Q4+LoRA TP、`S=2048`：

| 路径 | max-rank forward | 相对原路径 |
| --- | ---: | ---: |
| eager FP32 路径 | 1033.06 ms | 1.000× |
| fused RMSNorm/AdaLN | 985.61 ms | 1.048× |
| 再 fused SwiGLU/scale | 944.79 ms | 1.093× |

最终路径约减少 `8.5%` wall time。两 rank 始终逐 bit 相同并且全部 finite；相对 eager FP32 50 层输出的 relative RMS 为 `0.0049505`、cosine 为 `0.99998784`。这里的差异主要来自 Triton reduction/rounding 顺序，已再通过真实 4-step latent、视频和音频验收。

模块里也保留了 fused FP32 gated residual 实验，但该操作自身仅约 `1.02×`，完整路径收益不足以抵消新增生产分支和 review 成本，因此没有启用。

结果：

- `results/h3_fp32_ops_sm70_s2048_w8_swiglu.json`
- `results/h3_tp_backbone_50l_s2048_warm.json`
- `results/h3_tp_backbone_50l_s2048_fp32_fused.json`
- `results/h3_tp_backbone_50l_s2048_fp32_fused_swiglu.json`

## 9. Kernel 路线与取舍

### 9.1 Global attention Triton 原型：正确但淘汰

SM70 online-softmax Triton kernel 不物化 `S×S`，数值正确，但 `S=2048`：

```text
PyTorch efficient SDPA ≈ 3.57 ms
Triton prototype       ≈ 36.0 ms
```

PTX 生成了 Volta MMA，但达到约 255 registers/thread 并 spill。生产路径继续用 PyTorch efficient SDPA。

结果：

- `results/h3_attention_sm70_compile_gate.json`
- `results/h3_attention_sm70_s2048_sweep.json`

### 9.2 Fused RMSNorm + partial RoPE：采用

融合内容：

```text
FP16 overflow stabilizer
FP32 RMS statistics
RMSNorm
96-dim split-half RoPE
32-dim passthrough
Q/K in-place writeback
```

最佳 `num_warps=1`：

| S | eager | fused SM70 | speedup |
| ---: | ---: | ---: | ---: |
| 128 | 0.288 ms | 0.0356 ms | 8.1× |
| 2048 | 3.185 ms | 0.1949 ms | 16.35× |
| 8192 | 11.917 ms | 0.8662 ms | 13.76× |

`448×256` same-seed 1-step 的 video latent、audio latent 和 MP4 均 bitwise identical。

### 9.3 FP32-output wide Linear：采用

独立真实矩阵尺寸、GEMM-only gate（`S=64`）：

| 投影 | FP32 GEMM | TC FP16→FP32 | speedup | relative RMS |
| --- | ---: | ---: | ---: | ---: |
| attention out `7168->5376` | 0.5571 ms | 0.2406 ms | 2.315× | 0.0002079 |
| MLP FC2 `14336->5376` | 1.0593 ms | 0.4910 ms | 2.157× | 0.0002077 |

真实 Q4_0 + Turbo LoRA + 双卡 NCCL、`S=2048`：

```text
原 FP32 TP block       35.866 ms
TC/FP32-output block   21.165 ms
加速                    1.695×
relative RMS            0.0004522
cosine                   0.99999988
rank0 == rank1           bitwise
```

结果：

- `results/h3_v100_fp32_linear_gate.json`
- `results/h3_q4_lora_tp_block0_s128_fp32_tc_module.json`
- `results/h3_q4_lora_tp_block0_s2048_fp32_tc_module.json`

### 9.4 Q4 反量化与 global attention：按序列长度区分

真实 Q4+LoRA TP block、`S=128` 的中位 component 时间：

```text
QKV dequant  1.414 ms
out dequant  0.511 ms
FC1 dequant  1.822 ms
FC2 dequant  0.941 ms
合计          4.688 ms
```

旧 FP32-wide block 为 `7.587 ms`，短序列下反量化约占 61.8%。启用新 wide path 后，真实 `S=128` block 为 `6.901 ms`，反量化仍占主导；`S=2048` 时四张矩阵反量化约 4.79 ms，整个 block 为 21.165 ms，约占 22.6%。

完整目标 shape 改变了优先级。`1344×768 / 124f` packed sequence 为 `37746`，旧
strided SDPA 的 50 层 forward 约 `47.4–48.0 s`，其中 FP32 NCCL 通常只有
`1.2–1.8 s`。Q4 反量化成本大致不随 token 数线性增长，而 global attention/GEMM
会快速增长，因此目标尺寸的主要热点是 attention 和矩阵计算，不是磁盘读取或 NCCL。
后文 9.4.2 的 Q-only layout 已把冷态基线降到约 44.05 s。

#### 9.4.1 SM70 Triton 直解量化 gate

`h3_v100_q4_ops.py` 的 Q4_0 kernel 直接从常驻 `uint8` shard 解出 FP16 矩阵，
不物化 eager 路径的 nibble/scales 中间 tensor。它只覆盖标准 GGML Q4_0，严格检查
SM70、32-value block 对齐和 raw byte geometry；非严格模式失败会回退当前 eager
实现，严格模式用于部署前 gate。

真实 block 0 shard 的结果为：

| 矩阵 | eager | Triton | 加速 | eager/triton 额外显存 |
| --- | ---: | ---: | ---: | ---: |
| QKV `[10752,5376]` | 1.36 ms | 0.74 ms | 1.83× | 169.7 / 110.3 MiB |
| out_proj `[5376,3584]` | 0.48 ms | 0.27 ms | 1.79× | 56.4 / 36.8 MiB |
| FC1 `[14336,5376]` | 1.78 ms | 0.93 ms | 1.92× | 226.6 / 148.0 MiB |
| FC2 `[5376,7168]` | 0.91 ms | 0.50 ms | 1.81× | 113.0 / 74.0 MiB |

四个矩阵的 `max_abs=0`、`finite=true`、cosine 约为 1。完整 50 层 TP 的同 seed
结果保持 rank bitwise identical：

| packed sequence | eager | Triton Q4 | 变化 | peak allocated |
| ---: | ---: | ---: | ---: | ---: |
| `2048` | 944.79 ms | 820.74 ms | 13.1% faster | 6898.7 / 6899.2 MiB |
| `14880` | 9694.94 ms | 9500.45 ms | 2.0% faster | 8097.1 / 7959.1 MiB |
| `37746` | 47.506 s | 47.077 s | 0.9% faster | 10225.7 / 9973.9 MiB |

这项算子优化了短/中序列的 kernel overhead，但在目标 `S=37746` 上 global SDPA
仍约为 `35.17 s`，所以不默认切换。离线验证方式：

```bash
H3_TP_Q4_DEQUANT=triton H3_TP_Q4_DEQUANT_STRICT=1 \
  torchrun --standalone --nproc_per_node=2 \
  scripts/benchmark_h3_tp_backbone.py --sequence 37746 --warmup 1 \
  --output results/h3_tp_backbone_50l_s37746_q4triton_20260825.json
```

生产仍使用 `H3_TP_Q4_DEQUANT=eager`；后续若做 Q4 fused GEMM，必须复用同一
raw shard/32-value block 约束，并重新通过 `S=14880/37746` wall-time gate。

后续 kernel 优先级：

1. 生产路径保持 PyTorch efficient SDPA + Q-only layout；现有 Triton global-attention 原型慢约 10×，不接入。
2. 针对 SM70 的 SDPA 调度或 Q4 decode+GEMM 只能在 `S=14880/37746` 上证明 wall-time 收益后采用。
3. 保留 Q4 compressed shard 常驻；任何 fused Q4 kernel 都不得重新引入完整 FP16/FP32 权重常驻或 host mmap。
4. NCCL 仅占目标 forward 的约 2.5–3.2%，不以牺牲 FP32 all-reduce 稳定性换取小幅通信优化。
5. 不根据 `S=128` microbenchmark 提前替换已经通过 1 MP 验收的生产路径。

#### 9.4.2 Q-only SDPA layout gate（已采用）

H3 fused QKV 的原始 Q/K/V 都是 sequence stride 为 3 倍的跨步 view。SM70 efficient
SDPA 对 Q 的布局最敏感：只把 Q 复制成 contiguous BHSD，就能得到几乎与全 Q/K/V
compact 相同的裸 kernel 收益，同时避免 K/V 的约 516 MiB 额外输入副本。

| `S=37746` 完整 50 层 | forward | rank0 peak allocated | reserved | SHA256 |
| --- | ---: | ---: | ---: | --- |
| 原始 strided | 47.442 s | 10225.7 MiB | 10336 MiB | `1b278b8e...98e498fa` |
| compact 全 Q/K/V | 43.121 s | 10166.7 MiB | 11372 MiB | 同上 |
| **只 compact Q** | **44.051 s** | **9973.4 MiB** | **10336 MiB** | 同上 |

因此生产选择 Q-only，阈值为 `S>=4096`；小序列保持 strided。回退开关是
`H3_TP_COMPACT_QKV=0`，`all` 仅保留实验。该变化没有触碰 softmax、global visibility、
FP32 residual/MLP/output 或 NCCL reduction order。

常驻服务 1 MP 的前两个冷态 forward 为 `43.979/44.363 s`。连续满载后 GPU1 达
82°C、SM clock 最低降至 570 MHz，后两次变为 `47.026/51.623 s`，rank0 collective
等待同步增大；这部分是硬件热降频，不应算成 Q-only 算法回退。请求随后按用户要求
中断，未保存 latent；完整记录见
`results/h3_1mp_no_ref_compact_q_20260826_summary.json`。

### 9.5 `torch.compile` 的价值边界

当前 PyTorch 是 `2.8.0+cu126`，ComfyUI-GGUF 在 2.8 之后允许 full compile；这只说明版本层面不再主动禁用 compile，不代表 H3 的 Q4/NCCL 图可以被完整捕获。生产 TP 的真实 forward 同时包含：

- `Q4 raw -> dequantize_q4_0 -> FP16/FP32 GEMM`；
- 50 层 Python block loop 和按 `sequence/chunk_rows` 展开的 MLP loop；
- 每层两次原生 `dist.all_reduce`；
- 动态 packed sequence、segment/modulation row map 和 profile event；
- 已经独立编译的 Triton RMS/RoPE/FP32 stability kernel。

这些边界使 `torch.compile` 最多优化局部普通 elementwise，不能替代 Q4 fused GEMM、NCCL 或现有 SDPA backend。对不固定 shape 开 `dynamic=True` 又会牺牲专门化和 kernel 质量；对固定 shape 则会为 `S=868/14880/37746` 分别编译 graph，并产生冷启动和 code-cache 成本。

低内存、无模型加载的 SM70 compile gate：

| 测试 | eager | compiled | 结论 |
| --- | ---: | ---: | --- |
| FP32 stability-like elementwise，`S=128,H=5376` | 0.0701 ms | 0.0303 ms | 2.32×；但生产已有手写 Triton |
| efficient SDPA，`S=128,H=28,D=128` | 0.0265 ms | 0.0322 ms | 慢 17.7% |
| efficient SDPA，`S=2048,H=28,D=128` | 1.7917 ms | 1.7865 ms | 仅快 0.29%，等于没有收益 |

因此当前不对完整 H3 TP 开 `torch.compile`。它可以作为后续对 token-refiner、condition projection 或 final head 的可选实验，但这些外围模块不是 `1344×768` 生产路径的主要耗时；不能用它替换当前“手写稳定 kernel + PyTorch efficient SDPA + NCCL TP”的方案。

完整结果：`results/h3_compile_smoke_sm70.json`。

## 10. 已保存的 TP 实测

### 10.1 合成 FP16 数学 gate

缩小 profile：

```text
rank consistency max_abs = 0
dense max_abs            = 0.00390625
dense cosine             = 1.0
```

H3 真尺寸：

| S | TP block time | 两 rank 一致性 |
| ---: | ---: | --- |
| 128 | 1.806 ms | bitwise |
| 2048 | 13.470 ms | bitwise |

结果：

- `results/h3_tp_block_small_correctness.json`
- `results/h3_tp_block_h3_s128.json`
- `results/h3_tp_block_h3_s2048.json`

### 10.2 真实 Q4_0 + Turbo LoRA

`S=128`：

```text
TP                         7.587 ms/block
dense Q4+LoRA             14.508 ms/block
speedup                    1.912×
rank consistency           bitwise
relative RMS vs dense      0.0010619
cosine vs dense            0.99999952
local compressed Q4/rank   103.36 MiB
local LoRA/rank              9.625 MiB
```

`S=2048`：

```text
TP                         35.866 ms/block
rank consistency           bitwise
NCCL all-reduce total       1.152 ms/block
peak allocated/rank       ~824 MiB
```

启用 FP16 Tensor Core / FP32-output wide Linear 后：

| S | 原 FP32-wide TP | 新 TC/FP32-output TP | 新路径数值结论 |
| ---: | ---: | ---: | --- |
| 128 | 7.587 ms | 6.901 ms | vs dense relative RMS `0.0010912`，cosine `0.99999952` |
| 2048 | 35.866 ms | 21.165 ms | vs 原 TP relative RMS `0.0004522`，cosine `0.99999988` |

结果：

- `results/h3_q4_lora_tp_block0_s128_v2.json`
- `results/h3_q4_lora_tp_block0_s2048.json`
- `results/h3_q4_lora_tp_block0_s128_fp32_tc_module.json`
- `results/h3_q4_lora_tp_block0_s2048_fp32_tc_module.json`

### 10.3 完整 50 层生产路径 gate

`448×256`、22 帧、1-step、seed 2011，旧 FP32 路径与新路径使用相同 4B MLP ClipProj conditioning：

| 输出 | finite | relative RMS | cosine |
| --- | --- | ---: | ---: |
| video latent | 是 | 0.0015504 | 0.99999899 |
| audio latent | 是 | 0.0126518 | 0.99992144 |

解码结果：

- 画面是 prompt 指定的蓝色杯子和窗边场景，不是乱码。
- 与旧 FP32 baseline 的 22 帧平均 PSNR 为 `38.623 dB`。
- AAC 解码后音频 finite，NaN/Inf 都为 0；相对旧 baseline cosine 为 `0.99707`。
- cold 请求峰值约：service RSS `2479.8 MiB`，cgroup `4856.6 MiB`，没有 host mmap/RAM 爆发。
- 同服务、不同 seed 的第二次 warm 请求：finite trace 开启时 `1.75 s`，生产关闭 trace 后 `1.46 s`；没有 `Requested to load`，4B encoder 没有重载。

完整产物和逐次日志：

- `results/h3_fp32_tc_e2e/latent_comparison.json`
- `results/h3_fp32_tc_e2e/optimized_448x256_seed2011.mp4`
- `results/h3_fp32_tc_e2e/baseline_vs_optimized_contact.jpg`
- `results/h3_fp32_tc_e2e/cold_seed2011_v4/`
- `results/h3_fp32_tc_e2e/warm_seed2012/`
- `results/h3_fp32_tc_e2e/production_cold_seed2015/`
- `results/h3_fp32_tc_e2e/production_warm_seed2016_service.log`
- `results/h3_fp32_tc_e2e/summary.json`

### 10.4 常驻双 rank：448×256 smoke

同一受 cgroup 保护的 ComfyUI 服务中：

```text
rank0 / Comfy PID  681302
rank1 worker PID   684809
TP startup          34.62 s
首次完整请求         74.25 s（含 4B encoder 与 TP cold load/compile）
第二次 TP forward     0.549 s
models_reloaded       false
rank output diff      0
```

第二次请求之后两个 PID 均未改变。每 rank 常驻约 `6.1 GiB` TP payload，GGUF/safetensors payload map 为 0。TP latent 相对已验证的单卡 optimized latent：video cosine `0.9999957`、audio cosine `0.9999552`，全部 finite。

解码的 22 帧 MP4 是蓝色杯子和窗边场景，H.264 + 32 kHz stereo AAC，不是乱码。乱码根因由 BF16 LoRA 数值转换修复后，4B MLP ClipProj 无需更换。

结果：

- `results/h3_tp_e2e/startup.json`
- `results/h3_tp_e2e/forward_0001_868t_20260824-224610.json`
- `results/h3_tp_e2e/forward_0002_868t_20260824-224828.json`
- `results/h3_tp_e2e/latent_vs_single_card_optimized.json`
- `results/h3_tp_e2e/tp_resident_448x256_seed2012.mp4`

### 10.5 832×480、124 帧、Turbo 4-step

```text
packed sequence     14880
采样总计             42.034 s
四次 TP forward       9.805 / 9.830 / 9.831 / 9.848 s
整卡显存峰值          GPU0 14270 MiB / GPU1 13256 MiB
models_reloaded       false（四次均为 false）
rank output diff      0（四次均为 0）
```

MP4 为 832×480、124 帧、5.167 s，H.264 + stereo AAC。抽帧是清晰连续的绿色玻璃球、咖啡豆和咖啡馆场景；解码音频 165888 frames，NaN/Inf 为 0。

```text
latent SHA256  1bb0b24de1cb0d23cb24850ba9b41edc7ae1f637d863d3a822f5de8be9c82f92
MP4 SHA256     3735a39859668aeac15ed26c82ddeeb873020a76e2a333b0129494e1a95e955e
```

结果：

- `results/h3_tp_e2e/tp_832x480_124f_4step_submit.json`
- `results/h3_tp_e2e/tp_832x480_124f_4step_gpu.csv`
- `results/h3_tp_e2e/forward_0003_14880t_20260824-225154.json` 至 `forward_0006...json`
- `results/h3_tp_e2e/tp_fused_832x480_124f_4step_seed2009_latent.pt`
- `results/h3_tp_e2e/tp_fused_832x480_124f_4step_seed2009.mp4`
- `results/h3_tp_e2e/tp_fused_832x480_124f_4step_seed2009_contact.jpg`

### 10.6 1344×768、124 帧、Turbo 4-step 最终验收

```text
packed sequence       37746
采样总计               196.464 s
四次 TP forward         47.698 / 47.594 / 47.758 / 48.048 s
NCCL / forward          1.211 / 1.500 / 1.507 / 1.482 s
TP allocated 峰值       rank0 9576 MiB / rank1 8535 MiB
整卡监控峰值            GPU0 15870 MiB / GPU1 14504 MiB
解码                    60.15 s
models_reloaded         false（四次均为 false）
rank output diff        0（四次均为 0）
```

latent 全部 finite。MP4 为 1344×768、124 帧、5.167 s，H.264 + 32 kHz stereo AAC；画面清晰、运动连续，不是乱码。AAC 解码得到 165888 个 stereo frame，NaN/Inf 为 0。

```text
latent SHA256  66cc6543904cd77387ade6905379748037049467901c7aff3d8dee42893cb701
MP4 SHA256     1dfcecacc1d93dc9838019985b82ad603aed9e7dfe8a2d31f527b9a1a332e0f4
```

结果：

- `results/h3_tp_e2e/tp_1344x768_124f_4step_submit.json`
- `results/h3_tp_e2e/tp_1344x768_124f_4step_gpu.csv`
- `results/h3_tp_e2e/forward_0007_37746t_20260824-225844.json` 至 `forward_0010...json`
- `results/h3_tp_e2e/tp_fused_1344x768_124f_4step_seed2009_latent.pt`
- `results/h3_tp_e2e/tp_fused_1344x768_124f_4step_seed2009.mp4`
- `results/h3_tp_e2e/tp_fused_1344x768_124f_4step_seed2009_contact.jpg`
- `results/h3_tp_e2e/tp_fused_1344x768_124f_4step_seed2009_audio_stats.json`
- `results/h3_tp_e2e/audit_20260824.json`

### 10.7 1 MP 下的 18/18 layer-MP 显存审计

这里记录的是切换 Q4 默认之前，Qwen3-VL-4B INT8 + ridge 的历史 18/18
language-layer model-parallel 审计；H3 的 50 层 compute-heavy backbone 仍然是双卡
NCCL TP。不能把这组历史层数均分误写成“当前整机显存完全均分”。

真实组合在 `1344×768 / 124 帧 / 4-step` 目标尺寸下连续完成了 4 次
denoise forward，模型没有重载，rank 输出 `max_abs=0` 且全部 finite：

| 项目 | GPU0 | GPU1 | 结论 |
| --- | ---: | ---: | --- |
| 整卡峰值 | `15870 MiB` | `14504 MiB` | 两卡均未 OOM |
| 16 GiB 卡剩余 | `514 MiB` | `1880 MiB` | GPU0 余量很紧 |
| 占用率 | `96.86%` | `88.53%` | 峰值相差 `1366 MiB` |
| H3 TP rank allocated 峰值 | `9576 MiB` | `8535 MiB` | rank0 还承担外围模块 |

因此结论分两层：

1. **稳定性：通过。** 在当前固定工作流、batch=1、`ref_img_size=match`、不让视频
   VAE 与采样并存的条件下，1 MP 已实测跑通；4 次目标尺寸 forward、latent、音频
   都 finite，cgroup `oom=0`、`oom_kill=0`，两个 rank 的模型 payload mmap 仍为 0。
2. **均衡性：Qwen language block 的 18/18 分配是均衡的，但整卡不是完全均衡。** GPU0
   同时承担 H3 rank0、Qwen 前半段及 Comfy 外围/采样 owner，GPU1 承担 H3 rank1、
   Qwen 后半段和 ClipProj，因此最终由 GPU0 的 `15870 MiB` 决定安全上限。

`1366 MiB` 的整卡差距不是 Qwen 18/18 再拆一层就能自动消除的；强行改成严格
Qwen TP 又会引入已记录的 2.12%/2.56% hidden-state 漂移。该 INT8 18/18 配置后来
先被 12/24 取代，Q4 direct-owner 又进一步改为 4/32；1 MP 仍是当前默认上限，不应叠加
batch、`ref_img_size=max`、常驻视频 VAE 或额外 compile workspace。

### 10.8 参考图场景与“真正均衡”的分卡规则

上面的 1 MP 基线是普通 conditioning 的显存审计，不能直接代表首尾参考图的
峰值。`MiniMaxH3ImageToVideo` 的每一张首/尾参考图都会在 H3 packed sequence
中增加一整块视觉 condition rows；即使首尾连接的是同一张图片，也不会因为文件
相同而合并这两块 token。1 MP、124 帧、首尾双参考图时，QKV、attention workspace
和 FP32 residual 的峰值会一起上升，GPU0 可能在第一层 QKV/SDPA 前就只剩很少
余量。因此“无参考图 1 MP 已通过”不等于“首尾参考图 1 MP 也必然通过”。

以后调整 TP/MP 时，不能只看 layer 数是否 50/50，也不能只看 Qwen 是否 18/18。
应按下面的预算决定切分：

```text
peak[i] = H3_TP_shard[i]
        + Qwen/VAE MP resident[i]
        + rank-owner[i]
        + request activation peak[i]
        + kernel/NCCL workspace[i]
```

其中 `rank-owner[GPU0]` 包括 Comfy 主进程、H3 rank0、输入/conditioning 组装和
部分外围模块；所以即使两张卡型号相同，整卡最佳方案也可能不是每个 MP 模块
严格均分。当前正确的调优顺序是：

1. H3 compute-heavy backbone 继续保持几何正确的 2-way TP。QKV/FC1 的 head/FFN
   shard、out/FC2 的 row-parallel partial 和 NCCL all-reduce 不能为了“看起来均衡”
   随意改成按层数切半以外的布局。
2. 先用 MP split 吸收外围差异：把更多 Qwen language tail 和 VAE decoder block
   放到 GPU1，给 GPU0 留出 rank0 owner、参考图 rows 和 attention workspace。层数
   可以是 12/24、14/22 等，最终以真实 `nvidia-smi` 峰值和 finite/quality gate
   决定，不以 18/18 作为教条。
3. 目标不是运行时占用完全相等，而是两张卡都满足
   `peak <= 16384 MiB - safety_margin`。1 MP 参考图建议至少保留约 1 GiB 的
   峰值余量；如果 GPU0 只剩几百 MiB，应视为未通过，即使某次请求侥幸成功。

如果以后确实要做“非均等 H3 TP”，有两条路线，不能混为普通 MP：

- **weighted tensor parallel**：让 GPU1 持有更多 attention heads/FFN channels，
  但必须同步重做 Q4 shard map、LoRA A/B 切分、local head shape、row-parallel
  输出和 NCCL buffer；两 rank 的 collective 顺序和最终 `[S,5376]` partial shape
  仍必须一致。不能只改 `HEADS // 2` 或把 GGUF 文件字节切成不等大小。
- **pipeline/layer parallel**：让 GPU1 承担更多完整 H3 blocks，通过 block 边界
  handoff activation。它可能改善权重分布，但会引入 pipeline bubble、边界同步和
  不同的数值/性能基线；只有在 weighted TP 仍无法满足峰值预算时才评估。

因此本项目的默认原则是：**H3 TP 为计算正确性保持对称，Qwen/VAE MP 为整卡显存
允许有意倾斜；只有经过 shape、collective、finite、same-seed quality 和端到端
峰值审计后，才允许非均等 H3 TP 进入生产。**

## 11. 已实现的常驻进程架构

普通 ComfyUI 单进程不能透明地把 50 层 Python DiT forward 变成 NCCL TP。生产实现因此把 Comfy 主进程本身变成 rank0，并只额外启动一个长期 rank1 子进程：

```text
Comfy 主进程 / PID 681302
  GPU0: rank0 Q4 shard + Turbo LoRA shard
        Qwen Q4 embedding/vision + 前 4 层 + H3 外围小模块
        sampler、API/result owner；decode 阶段临时加载 VAE
  GPU1: rank1 Q4 shard + Turbo LoRA shard
        Qwen Q4 后 32 层 + ClipProj v3.1 ridge（resident）

长期子进程 / PID 684809
  GPU1: rank1 Q4 shard + Turbo LoRA shard

rank0/GPU0 <========== NCCL over NVLink ==========> rank1/GPU1
```

同一个 Comfy 主进程会同时持有 GPU0 的 rank0/Qwen 前半段 context 和 GPU1 的 Qwen 后半段 context；`nvidia-smi` 看 GPU1 时要把主进程的 Qwen tail、ClipProj 与子进程 rank1 的占用相加。生产不是每个请求重新 `torchrun`：`h3_tp_runtime.py` 用 file-store 初始化一次两 rank `torch.distributed/NCCL` process group，然后通过 pipe 给长期子进程发送 forward/shutdown 命令。

### 11.1 一次性加载

1. ClipProj 节点读取 4B Q4_K_M encoder 与 FP16 vision/mmproj 时就按 4/32 language-layer MP 直达 GPU0/GPU1 最终 owner，v3.1 ridge projection 与 encoder 一起 `mode=resident`。
2. `MiniMaxH3TensorParallel` 把原模型树中的 50 个 block 替换成一个无参数 proxy；外围 condition/patch/token-refiner/final-layer 仍由 Comfy 主进程执行。
3. 第一次真正进入 backbone 时，rank0 启动 rank1 并初始化 NCCL。
4. 两 rank 都只读 GGUF/safetensors header；H3 Q4/LoRA shard 使用 4 MiB staging，Qwen GGUF 使用 8 MiB staging，并对已读文件页调用 `DONTNEED`。
5. 每 rank 常驻：Q4 `5167.97 MiB`、core LoRA `481.25 MiB`、AdaLN base `166.11 MiB`、AdaLN LoRA `303.52 MiB` 及约 `12.64 MiB` 小权重。
6. singleton runtime 拒绝在同一服务内静默切换 model/LoRA 配置；需要换权重时显式重启服务，避免旧、新 shard 混用。

首次 TP startup 为 `34.62 s`。之后十次保存的 forward 均为 `models_reloaded=false`，rank1 PID 未变，两个进程的 `/proc/<pid>/maps` 中模型 payload 映射数均为 0。

### 11.2 每个 denoise forward

1. 主进程外围模块在 GPU0 生成 FP32 packed residual、time embedding、segment layout 和 RoPE frequencies。
2. rank0 通过 NCCL broadcast 把这些输入送给 rank1；不传模型权重。
3. 两 rank 以完全相同的 collective 顺序执行 50 层：local QKV、28-head global attention、FP32 out all-reduce、local SwiGLU、FP32 FC2 all-reduce。
4. 每层边界两 rank 都持有逐 bit 相同的 FP32 residual；任一 rank 发现 non-finite 会 fail-fast。
5. rank1 仅通过 pipe 返回 profile/finite 元数据；rank0 的 residual 交回 Comfy 外围 final layer 和 sampler。
6. forward 结束只释放 activation/dequant 临时量，Q4 shard、Turbo LoRA、4B encoder 和 projection 继续常驻。

### 11.3 为什么 VAE 是唯一例外

视频 VAE 文件约 `5.21 GB`。目标采样时整卡峰值已经是 GPU0 `15870 MiB`、GPU1 `14504 MiB`，任何一张 16 GiB V100 都没有再永久放入约 5 GiB VAE 的空间；这不是 offload 策略能消除的容量问题。

当前 decode-only 工作流会保持 ClipProj/conditioning 和 TP runtime 可达，只在 decode 阶段从 `/mnt/GALAX` fault VAE（权重加载约 2 s，主要时间仍是 VAE 计算）。它不会卸载或重载 TP shard、Turbo LoRA、4B encoder/ClipProj。这样满足“每次生成不反复加载 compute-heavy 模型”，同时避免把 VAE 退回有限的 host RAM 或恢复大文件 mmap。

## 12. 实现审计与下一轮优化

### 12.1 原接入阶段均已完成

| 阶段 | 当前状态 | 保存的验收证据 |
| --- | --- | --- |
| A：单 block 分片与 kernel | 完成 | dense/Q4/LoRA gate、rank bitwise consistency、FP32 wide 输出 |
| B：50 层 backbone | 完成 | `h3_tp_backbone.py` 与 `h3_tp_backbone_50l_*.json` |
| C：完整 H3 外围图 | 完成 | Comfy proxy 保留 condition/refiner/final/audio-video packed 语义 |
| D：常驻双进程服务 | 完成 | 固定 PID、十次 forward、`models_reloaded=false`、payload map=0 |
| E：最终大分辨率 | 完成 | 448、832、1344 三档 latent/MP4/GPU CSV/SHA256 |
| F：Q-only SDPA layout | 完成 | 50 层快 7.15%、hash 相同、无 reserve 增长；长序列默认启用 |

### 12.2 做过但没有进入生产的优化

| 实验 | 结果 | 决策 |
| --- | --- | --- |
| row-parallel LoRA 改 FP16 Tensor Core | out_proj 略快，FC2 因 scale/cast 反而更慢 | 保留 FP32 LoRA |
| 50 层 AdaLN batching | FP32 仅 `1.009×`，Tensor Core 为 `0.993×` | 保留逐层 AdaLN |
| fused FP32 gate residual | 单操作约 `1.02×` | 不增加生产分支 |
| 自研 Triton global attention | `S=2048` 约比 efficient SDPA 慢 10×，register spill | 继续 PyTorch SDPA |
| SM70 Triton Q4 直解量化 | block microbenchmark 1.79–1.92×；完整 TP `S=2048` 快 13.1%，`S=37746` 快 0.9% | 保留 opt-in；生产默认 eager，继续研究 fused GEMM |

失败结果没有删除：

- `results/h3_lora_fp32_tc_s2048.json`
- `results/h3_adaln_batch_sm70_m2_tc.json`
- `results/h3_attention_sm70_s2048_sweep.json`
- `results/h3_fp32_tc_e2e/cold_seed2011_v3/`

### 12.3 后续性能研究门槛

当前 Q-only 1 MP 冷态单次 forward 约 44 s；连续满载会受 GPU1 热降频影响。下一轮
先完成 Q-only 双参考图成品 gate，再决定是否继续 TileLang/SM70 global attention 或
Q4 GEMM；不能继续削弱 FP32 稳定岛。

任何下一版候选必须同时满足：

1. 在 `S=14880` 和 `S=37746` 的完整 50 层 wall time 有稳定收益，不能只报小 shape microbenchmark。
2. 两 rank 输出一致、全程 finite，并与当前 same-seed latent 做 relative RMS/cosine 对比。
3. 不增加 host mmap、完整 CPU 权重副本或目标尺寸 GPU 峰值。
4. 继续保存 latent、MP4、音频 finite、GPU CSV 和 profile JSON，能够一键回退到当前基线。
5. 生产默认仍用 PyTorch efficient SDPA + Q-only layout；新 kernel 必须超过这个新基线。

## 13. Review 清单

后续每次改 TP 代码时逐项检查：

- [ ] QKV shard 是否分别来自 Q、K、V 相同 head 范围？
- [ ] FC1 shard 是否同时包含匹配的 gate/up 范围？
- [ ] out_proj/FC2 是否按每行输入列切，而不是文件字节整体切半？
- [ ] 所有 Q4 切点是否落在 32-value block 边界？
- [ ] LoRA 的 A/B 是否沿 base weight 的相同维度切分？
- [ ] row-parallel base 与 LoRA partial 是否只做一次 FP32 all-reduce？
- [ ] rank0/rank1 是否以相同顺序进入每个 collective？
- [ ] Q/K RMSNorm + RoPE 是否仍使用 H3 的 96+32 partial 语义？
- [ ] attention 是否仍为 global、non-causal、无 mask？
- [ ] compact layout 是否只改变 stride/storage，没有改变 Q/K/V 元素或 token/head 顺序？
- [ ] FP32 residual/out_proj/FC2 是否被保留？
- [ ] fused RMS/AdaLN 是否只把 branch materialize 为 FP16，没有把 replicated residual 降精度？
- [ ] fused SwiGLU 是否同时保留 FP32 LoRA 输入、safe FP16 base 输入和二次幂 scale？
- [ ] FC2 转 FP16 前是否使用逐 token 二次幂 scaling，输出是否直接 materialize 为 FP32？
- [ ] DynamicVRAM 层是否通过 `CastBiasWeightContext` fault/uncast，而不是直接 `.to()` meta weight？
- [ ] Tensor Core base wrapper 是否仍位于 `BypassForwardHook` 内部，没有绕开 Turbo LoRA？
- [ ] loader 是否只读 header + bounded staging，且 `/proc/<pid>/maps` 没有模型 payload？
- [ ] compressed shard 是否跨请求常驻，而不是每个 forward 从磁盘读？
- [ ] 连续请求的 rank1 PID 是否不变，profile 是否仍为 `models_reloaded=false`？
- [ ] decode 工作流是否保持 TP/ClipProj 可达，只阶段加载 VAE？
- [ ] 测试是否保存 JSON、latent/MP4 和峰值，而不是只看终端输出？
- [ ] 性能结论是否来自目标 shape，而不是只看 `S=128`？
- [ ] 失败/回退是否会让两个 rank 一起退出，不留下 NCCL hang？

## 14. 复现命令

布局审计：

```bash
/home/regen/minimax-h3/.venv/bin/python \
  scripts/inspect_h3_q4_tp_layout.py \
  --output results/h3_q4_tp_layout.json
```

FP16 数学 gate：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  /home/regen/minimax-h3/.venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  scripts/benchmark_h3_tp_block.py \
  --profile small \
  --output results/h3_tp_block_small_correctness.json
```

真实 Q4+LoRA block：

```bash
CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY NCCL_DEBUG=WARN \
  /home/regen/minimax-h3/.venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  scripts/benchmark_h3_q4_tp_block.py \
  --sequence 128 --warmup 2 --repetitions 10 \
  --output results/h3_q4_lora_tp_block0_s128_v2.json
```

长序列 shard 路径：

```bash
CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY NCCL_DEBUG=WARN \
  /home/regen/minimax-h3/.venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  scripts/benchmark_h3_q4_tp_block.py \
  --sequence 2048 --warmup 2 --repetitions 5 \
  --skip-dense-reference \
  --output results/h3_q4_lora_tp_block0_s2048.json
```

FP32-output Tensor Core 独立 gate：

```bash
/home/regen/minimax-h3/.venv/bin/python \
  scripts/benchmark_h3_fp32_linear.py \
  --device cuda:0 --rows 64 --warmup 2 --repetitions 8 \
  --output results/h3_v100_fp32_linear_gate.json
```

真实 Q4+LoRA、长序列新 wide path：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  /home/regen/minimax-h3/.venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  scripts/benchmark_h3_q4_tp_block.py \
  --sequence 2048 --warmup 2 --repetitions 8 \
  --wide-linear fp16-fp32 --skip-dense-reference \
  --output results/h3_q4_lora_tp_block0_s2048_fp32_tc_module.json
```

fused FP32 操作 gate（只占用一张卡，约百 MiB 级测试张量）：

```bash
/home/regen/minimax-h3/.venv/bin/python \
  scripts/benchmark_h3_fp32_ops.py \
  --sequence 2048 --warps 8 --warmup 10 --repetitions 50 \
  --output results/h3_fp32_ops_sm70_s2048_w8_swiglu.json
```

完整 50 层 fused/eager 对比会加载两张卡的 TP shard，不能与生产服务同时运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
H3_NO_HOST_MMAP=1 \
H3_TP_FUSED_FP32_OPS=1 \
H3_TP_FP32_OPS_WARPS=8 \
  /home/regen/minimax-h3/.venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  scripts/benchmark_h3_tp_backbone.py \
  --sequence 2048 --warmup 1 --compare-eager-fp32 \
  --output results/h3_tp_backbone_50l_s2048_fp32_fused_swiglu.json
```

生产启用：

```bash
H3_NO_HOST_MMAP=1 \
H3_V100_FP32_TC=1 \
H3_V100_ATTENTION=pytorch \
H3_V100_RMS_ROPE=sm70 \
H3_V100_RMS_ROPE_WARPS=1 \
H3_TP_FUSED_FP32_OPS=1 \
H3_TP_FP32_OPS_WARPS=8 \
H3_TP_COMPACT_QKV=q \
H3_TP_COMPACT_QKV_MIN_SEQUENCE=4096 \
H3_TP_RESULTS_DIR=/home/regen/code/minimax_v100/results/h3_tp_e2e \
./scripts/start_comfyui_isolated.sh
```

低风险 smoke 和最终工作流：

```bash
/home/regen/minimax-h3/.venv/bin/python scripts/submit_workflow.py \
  workflows/clipproj-4b-q4-tp-turbo-smoke-448x256-1step.json --wait
```

下面是约 1 MP 的重测试；确认服务空闲且两卡有余量后再运行：

```bash
/home/regen/minimax-h3/.venv/bin/python scripts/submit_workflow.py \
  workflows/clipproj-4b-q4-tp-turbo-1344x768-124f-4step.json --wait
```

最终一次验收的集中审计：`results/h3_tp_e2e/audit_20260824.json`。

## 15. 当前结论

截至本文版本，可以确认：

1. Qwen3-VL-4B Q4_K_M + FP16 mmproj + `mmh3-4b-ClipProj-v3.1` ridge 已设为默认；乱码根因是 BF16 LoRA 数值转换，早期 INT8/MLP 结果只保留为精度和历史对照。
2. 完整 50 层 H3 Q4_0 + Turbo LoRA 已由两个长期 NCCL rank 共同计算，每层 QKV/FC1 column parallel、out/FC2 row parallel。
3. Q4/LoRA shard、4B Qwen encoder 和 ClipProj 跨请求常驻；模型 payload mmap 为 0。Qwen Q4 direct-owner 4/32 已消除 GPU0 tail backing，steady allocated 约 1802/1893 MiB；仍有约 742 MiB FP16 embedding 可继续压缩。
4. FP32 防 NaN 边界完整保留。wide Tensor Core 路径把单 block `35.866` 降至 `21.165 ms`；fused RMS/AdaLN + SwiGLU 又把完整 50 层 `S=2048` 耗时降低约 8.5%。
5. 最终 `1344×768 / 124f / 4-step` 已跑通并保存：采样 `196.464 s`、解码 `60.15 s`、latent/音频全 finite、MP4 清晰连续。
6. VAE 无法在目标采样峰值下永久常驻是双 16 GiB 的物理边界；当前只阶段加载 VAE，不触碰常驻 TP/encoder。
7. 长序列 Q-only layout 已作为默认：完整 50 层 47.442 → 44.051 s，输出 hash 完全一致。SM70 Q4 直解量化仍是 opt-in；自定义 attention/TileLang 暂停到下一轮，NCCL 在未热降频时不是主瓶颈。
