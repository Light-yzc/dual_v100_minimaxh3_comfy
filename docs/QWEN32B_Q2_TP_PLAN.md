# Qwen3-VL-32B Q2 双 V100 精度优先 TP 实施方案

日期：2026-08-27

状态：**output-row TP 保持显式实验功能；layer-MP 已成为 Qwen32 启动器默认。**
本文保留 TP 的设计与实验记录；共享 runtime/conditioning 的 TP gate 仍需独立维护。
Qwen32B 的解耦 layer-MP 实现在 [`QWEN32B_Q2_MP.md`](QWEN32B_Q2_MP.md)，已完成
双卡线上 smoke。H3 DiT 的既有 TP worker 不变；未使用 32B 节点时仍走 4B 路线。

TP 只有在显式设置 `H3_QWEN32_Q2_MODE=tp H3_QWEN32_Q2_TP=1` 时才允许使用；
MP 已通过 runtime 工厂接入并由启动器默认选择，不能在服务运行期间热替换。

本文供后续实现者直接接手。目标是在两张 Tesla V100-SXM2-16GB 上重新启用
`qwen3vl-32B-MiniMax-H3-Q2_K.gguf`，优先保留 32B 的语义容量，同时满足：

- Qwen language stack 由两张卡同时计算，而不是只做顺序 layer-MP；
- TP 本身不再引入已知的 hidden-state 累积漂移；
- 不把 7.9 GiB GGUF 放进系统 RAM，不 mmap 完整 payload；
- 1 MP H3 DiT 和参考图路径不能因 Qwen 常驻而 OOM；
- 相同 prompt/reference 的多 seed 生成不重复执行 Qwen；
- Qwen 释放出的显存允许 VAE 在 DiT 期间受控异步预取，隐藏后续 decode 的冷加载；
- 任何新路径必须独立开关；MP 目前仅完成低资源线上 gate，目标尺寸/多步 gate
  仍需单独记录，不能因此放开 full residency 或替换 4B 工作流。

相关现状和基线见：

- [`H3_V100_FREEZE_20260826.md`](H3_V100_FREEZE_20260826.md)
- [`H3_V100_KERNEL_TP_ROUTE.md`](H3_V100_KERNEL_TP_ROUTE.md)
- [`QWEN_Q4_TEST_RESULTS_20260825.md`](QWEN_Q4_TEST_RESULTS_20260825.md)
- [`NVLINK_AND_TP.md`](NVLINK_AND_TP.md)

## 0. 当前实现状态

- `scripts/audit_qwen32_q2_tp_layout.py` 已在真实 7.9 GiB GGUF 上通过 P0：902 tensors、
  50 层、350 个 language matrices，所有 output-row/file ranges 闭合，payload mmap 为 0，
  header parse RSS 增量约 3 MiB；报告写入 `results/qwen32_q2_tp_layout.json`。
- `h3_qwen32_q2_tp.py` 已实现 header-only layout、direct shard reader、Q2_K/Q3_K
  dequant、lazy linear/backbone、DeepStack 注入和 clear/trim/stats。真实 Q2_K/Q3_K row
  与 ComfyUI-GGUF 的 CPU/V100 FP16/FP32 对照为 bitwise equal，但这不替代完整 P2/P3 gate。
- `h3_async_vae.py` 和 `h3_async_vae_bridge.py` 已实现 header-only handle、每卡预算 ledger、
  bounded staging、direct-owner prefetch 和 sampler 生命周期桥接；共享 Qwen clear/runtime
  接线与真实双卡 overlap gate 仍未完成。
- 已有 `workflows/qwen32-q2-mp-full-smoke-448x256-1step.json` 并完成双卡线上
  smoke；P1–P5 的目标尺寸质量/容量门禁仍属于后续工作。

## 1. 结论先行

最终推荐路线是：

```text
Qwen32B Q2 GGUF
  -> rank0 完成 tokenizer / vision / multimodal 组装
  -> 50 层 language stack 使用两卡 output-row + all-gather TP
  -> 直接输出 5120 维 H3 conditioning，不使用 ClipProj
  -> conditioning 按 prompt/reference hash 缓存
  -> 两 rank 完整卸载 Qwen CUDA payload，只保留 header/offset descriptor
  -> H3 继续使用当前两 rank Q4_0 + LoRA TP
  -> DiT 计算时按每卡预算异步预取 FP16 VAE
  -> 最后一个真实 DiT forward 返回并释放临时显存
  -> 补齐 VAE 尾部，READY event 完成后立即 decode
```

本方案的首版决策已经固定：

- Qwen32B 采用 `evict`，conditioning 完成后 Qwen CUDA owner bytes 必须归零；
- FP16 VAE 是首选 decode 路线，INT8 ConvRot 只保留为显式回退和 A/B 对照；
- 双参考图 1 MP 使用 `1024 MiB` 每卡安全余量的 capped prefetch，禁止全量抢占；
- partial Qwen residency、VAE decode 期间反向预取下一请求 Qwen 都属于后续优化，不进入
  首版正确性路径。

25/25 layer-MP correctness oracle 已完成并作为当前默认安全路径；它负责证明 tokenizer、
vision、mRoPE、DeepStack、50 层输出和 H3 接口正确。output-row TP 仍是独立性能实验，
不是 MP 的隐式回退。

真正 TP 不采用旧的 column-parallel + row-parallel partial-sum 方案。旧 4B INT8
实验在 36 层后产生过约 `2.12%/2.56%` hidden-state 漂移，不能拿来作为精度优先的
32B 路线。新方案把每个 Linear 都按 **output rows** 切分；每个输出元素仍执行完整
dot-product，collective 只拼接结果，不参与数值求和。

## 2. 已核实的本地 Q2 文件几何

模型文件：

```text
/mnt/GALAX/minimax-h3/models/text_encoders/
  qwen3vl-32B-MiniMax-H3-Q2_K.gguf
```

通过 `custom_nodes/NoHostMMap/gguf_reader.py` 的 header-only reader 核实：

| 项目 | 数值 |
| --- | ---: |
| 文件大小 | `8,487,968,160` bytes，约 `7.905 GiB` |
| GGUF header/data offset | `75,040` bytes |
| tensor 数 | `902` |
| language layers | `50`，编号 `0..49` |
| language payload | `8,028,211,200` bytes，约 `7.477 GiB` |
| 每个 language layer | `160,564,224` bytes，约 `153.13 MiB` |
| token embedding | `255,252,480` bytes，约 `243.43 MiB` |
| vision/DeepStack | `204,429,440` bytes，约 `194.96 MiB` |

该文件已经是 H3 专用的 50 层版本，不包含普通 Qwen3-VL-32B 第 50 层之后的未用
language blocks，因此不存在再裁掉 14 层的收益。

language geometry：

```text
hidden                 5120
Q projection output    8192 = 64 heads * 128
K/V projection output  1024 =  8 heads * 128
GQA ratio              8 Q heads / KV head
MLP intermediate       25600
```

每层有 7 个大矩阵和 4 个 norm tensor。文件名虽为 Q2_K，但不能把所有 tensor
硬编码成同一量化类型：例如 `v_proj` 实际为 `Q3_K`，norm 为 `F32`。loader 必须逐
tensor 保留 GGUF header 中的真实 qtype、shape、row stride 和 payload offset。

## 3. 三种分卡方式的选择

| 方式 | 两卡是否同时计算 | 数值风险 | 通信 | 定位 |
| --- | --- | --- | --- | --- |
| 25/25 layer-MP | 否，单请求顺序执行 | 最低 | layer 25 一次 handoff | correctness oracle / fallback |
| 传统 row-parallel TP | 是 | partial sum 和量化列切分会改变累加 | 每层约 2 次 all-reduce | 不采用 |
| output-row all-gather TP | 是 | 每个输出 row 保持完整 dot-product | 每层 4 次 all-gather | 推荐生产候选 |

MP 和推荐 TP 的压缩权重总量相同，理想 language shard 均为：

```text
每 rank：7.477 GiB / 2 = 3.739 GiB，约 3828 MiB
```

embedding 和 vision 合计约 `438 MiB`，只在 rank0 的编码前处理阶段需要，可以在
conditioning 生成后释放，不应作为 DiT 阶段永久负载。

## 4. 推荐 TP 的精确数学拆分

### 4.1 共同约束

- language hidden `X` 在两个 rank 上保持相同副本，shape `[B,S,5120]`；
- RMSNorm、Q/K norm、attention mask、mRoPE/position 信息在两 rank 一致；
- rank0/rank1 始终执行相同的 50 层和完全相同的 collective 顺序；
- shard 顺序固定为 rank0 对应原矩阵前半 output rows，rank1 对应后半；
- all-gather 结果按 rank 顺序拼接，恢复原始 output-row 顺序；
- 不允许某个 rank 因为 text-only、reference 数量或 cache 命中而跳过 collective。

### 4.2 Attention

对每层输入 `X`：

```text
rank0: Q0=[...,4096], K0/V0=[...,512]  -> 32 Q heads + 4 KV heads
rank1: Q1=[...,4096], K1/V1=[...,512]  -> 32 Q heads + 4 KV heads
```

每个 rank 的 GQA ratio 仍为 `32/4=8`，与完整模型一致。各 rank 独立执行本地 head
attention，得到：

```text
A0, A1: [B,S,4096]
A = all_gather(A0,A1): [B,S,8192]
```

`o_proj.weight` 不按输入列切，而按输出 rows 切：

```text
W_o0, W_o1: [2560,8192]
O0 = linear(A,W_o0): [B,S,2560]
O1 = linear(A,W_o1): [B,S,2560]
O  = all_gather(O0,O1): [B,S,5120]
X  = X + O
```

这样每个 `O` 元素仍由完整 8192 维输入计算，不存在两个 FP16 partial output 相加。

### 4.3 MLP

`gate_proj/up_proj` 按输出 rows 切：

```text
G0/U0, G1/U1: [B,S,12800]
Z0 = silu(G0) * U0
Z1 = silu(G1) * U1
Z  = all_gather(Z0,Z1): [B,S,25600]
```

`down_proj.weight` 同样按输出 rows 切：

```text
W_d0, W_d1: [2560,25600]
D0/D1: [B,S,2560]
D = all_gather(D0,D1): [B,S,5120]
X = X + D
```

### 4.4 为什么该拆法更适合 Q2_K

所有大矩阵都沿 output rows 切分：

- 不切开单个 dot-product；
- Q2_K/Q3_K 的输入维量化 block 保持完整；
- 当前矩阵的输入宽度 `5120/8192/25600` 均能按 GGUF quant block 对齐；
- 每个 shard 对应连续 output rows，payload 读取比 input-column strided shard 简单；
- 可先用完整 tensor dequant 后按 rows 切作为 oracle，再验证 direct shard dequant 完全一致。

### 4.5 dtype 和算术顺序

TP 的首要目标是复现普通 Q2 forward，而不是另造一套“看起来更高精度”的算术：

1. 使用与普通 ComfyUI-GGUF Q2 路径相同的 tensor dequant dtype；
2. 每个 output-row shard 使用相同 `F.linear`/kernel、contiguous layout 和输出 dtype；
3. all-gather 只搬运，不做 SUM；
4. gather 完整 `O/D` 后再按普通模型顺序执行 residual add；
5. 不在没有逐层对照的情况下随意把 norm、attention 或 residual 改成另一 dtype；
6. rank 输出必须一致，任一 rank NaN/Inf 都让整个请求失败。

V100 没有本路线可直接使用的原生 W2A16 Tensor Core GEMM。第一版应先对 rank-local
compressed shard 做有界 FP16 dequant，再走 FP16 Tensor Core `F.linear`。自定义 Q2
kernel 是后续性能项，不能与首版正确性实现绑在一起。

## 5. Collective 成本和 buffer

每层固定 4 次 `dist.all_gather_into_tensor`：

1. local attention heads `[B,S,4096]`；
2. local O rows `[B,S,2560]`；
3. local SwiGLU rows `[B,S,12800]`；
4. local Down rows `[B,S,2560]`。

以单参考图历史 shape `S=620`、FP16 为例，每 rank 发送的 local halves 合计约
`27 MiB/layer`，50 层约 `1.3–1.4 GiB/rank`。本机 NCCL all-reduce 已测约
`96–101 GiB/s`，通信量相对于 32B matrix compute 可控，但实际速度必须上机记录；
不能用理论带宽替代 benchmark。

每次 encode 预分配并跨 50 层复用以下 gather buffer，禁止每层反复制造 allocator
碎片：

```text
attention gather  [B,S,8192]
hidden gather     [B,S,5120]   # O/Down 可复用
MLP gather        [B,S,25600]
```

buffer shape 改变时重新分配；同 shape 的连续请求复用。请求结束后根据 residency
policy 释放临时 activation，不能把 MLP gather buffer带进 1 MP DiT。

## 6. GGUF direct-shard loader

### 6.1 硬性安全要求

- 只能用 `NoMmapGGUFReader` 或同等 header-only reader；
- 禁止调用会创建完整 `np.memmap` 的普通 `gguf.GGUFReader`；
- 禁止完整 7.9 GiB CPU state dict、`torch.load` 或长期 CPU compressed copy；
- 模型从 `/mnt/GALAX` 读取，报告/cache 写 `/home/regen`；
- 每 rank 使用 4–8 MiB bounded staging；
- 每层/每 tensor 读取后对已消费范围调用 `POSIX_FADV_DONTNEED`；
- `/proc/<pid>/maps` 中 Q2 payload 命中必须为 0；
- service 继续受 `MemoryHigh=6500M / MemoryMax=7G / SwapMax=256M` 保护。

### 6.2 shard descriptor

建议为每个大 matrix 建立只含标量的描述对象：

```text
path, tensor_name, qtype, original_shape
data_offset, row_bytes, first_output_row, output_row_count
```

不要假设所有 GGUF qtype 的 row storage 相同。通过 GGUF 库的 quant geometry 计算
`row_bytes`，并逐 tensor 验证：

```text
rank0_bytes + rank1_bytes == original_tensor.n_bytes
rank0_rows + rank1_rows   == original_output_rows
每个 shard 起止均满足该 qtype 的 block/alignment 约束
```

第一版 correctness loader 可以“读取一个完整 matrix 的压缩 payload到小型 CPU/GPU
staging，再 dequant 后按 output rows 比较”，但不能把 50 层全部留在 CPU。通过后再
切换为 direct output-row payload range。

### 6.3 embedding 和 vision

- token embedding shape 为 `[151936,5120]`；生产版应只反量化实际 token IDs 对应
  的 rows，避免把完整 embedding 变成约 1.45 GiB FP16 tensor；
- reference image 的 vision tower、merger 和 DeepStack 放 rank0；
- rank0 完成视觉 token 替换和 DeepStack 特征生成后，把 language input、mask、
  position/mRoPE 和每个注入点需要的 DeepStack tensor广播给 rank1；
- text-only 请求不得触发 vision payload materialize；
- vision/embedding CUDA payload 在 conditioning 完成后优先回收。

为避免重写容易出错的 Qwen-VL 多模态语义，建议保留 stock Qwen3-VL 外层模型和
tokenizer/vision forward，只把 50 个 language blocks 替换成一个 TP proxy。这与
当前 H3 用 `PersistentH3TPBlocks` 替换 50 个 DiT blocks 的做法一致。

32B 原生 hidden 宽度已经是 H3 需要的 `5120`，此路线不经过 4B ClipProj，也不加载
`mmh3-4b-ClipProj-v3.1` 或 residual MLP projection。

## 7. 必须复用当前 H3 NCCL runtime

当前 `H3TPRuntime` 由 Comfy 主进程作为 rank0，并启动一个长期 rank1 子进程；默认
process group 被它独占。**不能再为 Qwen 启动第二套独立双卡 NCCL worker。** 那会
造成默认 group 冲突、collective 顺序失配和重复 CUDA context/权重占用。

推荐将 Qwen 加进同一个 runtime：

```text
Comfy / rank0                       persistent child / rank1
----------------                   ------------------------
Qwen outer/tokenizer/vision        Qwen language shard 1
Qwen language shard 0              H3 DiT shard 1
H3 DiT shard 0                     command loop
shared NCCL process group <------> shared NCCL process group
```

`H3TPRuntime.lock` 必须继续串行化 Qwen encode 与 H3 forward，任何时刻只允许一种
collective protocol 在 group 上运行。

rank1 command loop建议新增：

| command | 作用 |
| --- | --- |
| `qwen_forward` | 接收 shape/metadata，broadcast 输入并执行 50 层固定 TP |
| `qwen_trim` | 两 rank 按同一 layer ID 集合释放 compressed CUDA shard |
| `qwen_clear` | 回收全部 Qwen payload和临时 buffer，保留 header descriptors |
| `qwen_stats` | 返回 resident bytes、读盘 bytes、dequant/collective/forward 时间 |

`qwen_forward` 的 tensor 不通过 stdin/JSON 传输。JSON 只发送 shape、dtype、layer
policy 等小型 metadata；实际 hidden/mask/position/DeepStack tensor 使用 NCCL
broadcast/all-gather。

任一 rank 报错后沿用当前 fail-closed 语义：销毁 process group、终止 rank1、清理
Qwen/H3 cache，下次请求重新建立完整 runtime，不能让另一 rank 卡在旧 collective。

## 8. ComfyUI 节点和工作流接口

为避免破坏当前 4B 工作流，建议新增独立实验节点，而不是直接改现有 ClipProj 节点：

```text
MiniMaxH3DualRuntimeLoader
  inputs: DiT GGUF, Turbo LoRA, Qwen32B Q2 GGUF, strength, staging
  output: H3_DUAL_RUNTIME

Qwen32BQ2TPCLIPLoader
  inputs: H3_DUAL_RUNTIME, residency policy
  output: patched CLIP

MiniMaxH3TensorParallelWithRuntime
  inputs: MODEL, H3_DUAL_RUNTIME
  output: MODEL
```

也可以给现有 `MiniMaxH3TensorParallel` 增加 optional runtime handle，但必须保证旧
JSON 工作流的第一个 MODEL 输出和原有 required inputs 不变。

推荐让 `Qwen32BQ2TPCLIPLoader` 返回 stock `MiniMaxH3ImageToVideo` 能直接消费的
CLIP-like 对象；这样 prompt、单参考图、首尾帧和 ref2va 继续复用已有节点语义，
只替换 language block compute。工作流必须显式绕过 ClipProj。

第一版 encode 结束前应自动执行 safe `qwen_clear`，不能等到 H3 第一层才回收：H3
外层在进入 `PersistentH3TPBlocks.forward` 前已经创建 packed residual 和外围张量，
如果到那时才 trim，可能已经先 OOM。

## 9. 显存预算与 residency 状态机

### 9.1 已有基线

当前 Q4 direct-owner steady allocated 约：

```text
GPU0/GPU1 = 1802 / 1893 MiB
```

当前 Q4 路线 1 MP denoise 整卡峰值：

```text
无参考图：       12604 / 11336 MiB
首尾双参考图：   15200 / 15466 MiB
```

Q2 language 完美二分后约 `3828 MiB/rank`。若 conditioning 后已经释放
embedding/vision，用 Q2 language 替换当前 Q4 的粗略静态估算为：

```text
无参考图：
  GPU0 = 12604 - 1802 + 3828 = 14630 MiB
  GPU1 = 11336 - 1893 + 3828 = 13271 MiB

首尾双参考图：
  GPU0 = 15200 - 1802 + 3828 = 17226 MiB
  GPU1 = 15466 - 1893 + 3828 = 17401 MiB
```

所以：

- 1 MP 无参考图有机会让完整 Q2 language 与 DiT denoise 共存，但必须实测；
- 1 MP 双参考图确定不能完整常驻；
- 进入 VAE 异步预取前必须先让两个 Qwen rank完成 trim/clear；
- Q2 与 VAE 是否能在 decode 阶段重新叠加取决于 VAE 类型和输出策略，不能沿用
  denoise 峰值直接判断；
- 不能把文件大小直接当成运行峰值，standard loader 历史上还观测过约
  `8374 MiB` staged / `9587 MB` full load。

### 9.2 状态机

```text
META_ONLY
  header/descriptors 常驻，CUDA payload=0
      |
      v
ENCODING
  materialize resident/missing shards，执行 Qwen TP
      |
      v
DIT_READY
  conditioning 已缓存，释放临时 buffer和指定 layers
      |
      v
DENOISING
  H3 TP；Qwen 不执行；VAE 在显存上限内后台预取
      |
      v
VAE_FINALIZE
  最后一个 DiT forward结束，补齐未预取的 VAE tensors
      |
      v
DECODING
  VAE 已 ready；按现有 tiled/disk-backed 路线解码
```

第一版只实现最安全的 `META_ONLY -> ENCODING -> META_ONLY`，即每个未命中
conditioning cache 的新输入按层读取 Q2，编码完成后清掉全部 Qwen CUDA payload。
这会读约 7.9 GiB，但不会反复构建 Python 模型，也不会搬动 DiT。

第二版再做完整 layer 为单位的 GPU LRU：

- 两 rank 必须保留/回收同一组 global layer IDs；
- 每层每 rank 约 `76.56 MiB` compressed shard；
- 双参考图绝对装入至少要回收约 14 层，但几乎没有安全余量；
- 预留约 512 MiB 时估算需回收约 20 层，最多保留约 30/50；
- 预留约 1 GiB 时应只保留约 23/50；
- 上述值只是规划估算，最终 profile 必须来自真实 allocated/reserved/nvidia-smi 峰值。

建议初始环境开关：

```text
H3_QWEN32_Q2_TP=0             # 总开关，默认关闭
H3_QWEN32_RESIDENCY=evict     # evict | partial | full
H3_QWEN32_KEEP_LAYERS=0       # partial 模式显式值
H3_QWEN32_STAGING_MIB=4
H3_QWEN32_CACHE_MAX_MIB=256
```

`scripts/start_comfyui.sh` 会校验这些值；总开关保持 `0` 时现有 4B loader和工作流不变。

不要一开始做自动显存预测。先用固定 `keep_layers=0/16/24/30/50` 统计真实数据，再
决定是否按 resolution/reference profile 自动选值。

### 9.3 VAE 实测数据与 FP16 价值

VAE 不能只按 checkpoint 文件大小估算。当前同一台双 V100 上已有以下真实数据：

| 项目 | FP16 VAE | INT8 ConvRot VAE |
| --- | ---: | ---: |
| checkpoint 文件 | `5,207,808,496` bytes | `3,171,670,912` bytes |
| materialized/resident storage | `5,207,737,968` bytes | `2,795,657,008` bytes |
| 12/24 load allocated GPU0/GPU1 | `1885.89 / 3085.61 MiB` | `1119.73 / 1553.26 MiB` |
| 本次记录的 cold load | `9.84 s` | `1.94–8.52 s`，受 page cache影响 |
| 832×480 decode | `11.55 s` | `13.36 s` |
| 1 MP decode | `25.62 s`，GPU output 对照 | `28.64 s`，disk-backed output |
| 1 MP 精度 | reference | relative RMSE `0.0938%` |

1 MP 两条 decode 的输出路径不完全相同，不能把 `3.02 s` 差值当成严格公平的速度门；
但 832×480 同路线中 FP16 已明确快约 `1.81 s`，同时没有 INT8 weight error。因此，
如果约 9.84 秒 FP16 冷加载能被 4-step DiT 遮住，FP16 VAE 很可能是更好的端到端
质量/速度选择，不能因为 INT8 更小就把 INT8 写死为唯一方案。

### 9.4 三模型共存的阶段预算

以下是用现有 measured allocated/peak做的规划估算，不是新的上机 gate。Q2 采用本文
direct-shard TP：language `3828 MiB/rank`，embedding+vision 约 `438 MiB` 放 GPU0。

未执行 DiT、只有权重和 Qwen 编码前的基础状态：

```text
H3 + Q2 + INT8 VAE： 11611 / 11513 MiB
H3 + Q2 + FP16 VAE： 12377 / 13045 MiB
```

所以新的双卡 Q2 direct-shard loader完成后，三模型在非 DiT 阶段理论上均可共存；
FP16 路线 GPU1 仍约有 3.3 GiB，预期足够 Qwen 的单层 dequant、`S≈620` gather
buffer 和 vision activation。该结论不适用于现有 single-owner Q2 loader：后者曾在
一张卡占约 `8.4–9.6 GiB`，与 H3 shard叠加会直接超限。

Qwen 清空后，双参考图 DiT 的非 Qwen 基线约为：

```text
GPU0 = 15200 - 1802 = 13398 MiB
GPU1 = 15466 - 1893 = 13573 MiB
```

叠加完整 VAE resident weights：

```text
INT8 12/24：14518 / 15126 MiB   # 可以，最小余量约 1258 MiB
FP16 12/24：15284 / 16659 MiB   # GPU1 会超过 16 GiB，不能完整提前加载
```

无参考图时，Qwen 清空后的基线仅约 `10802/9443 MiB`；完整 FP16 12/24 VAE 后约
`12688/12529 MiB`，可以在 DiT 期间全部加载。单参考图/ref2va 必须实测后建立独立
profile，不能用无参考图或双参考图数字线性插值。

### 9.5 推荐的异步交换流水线

推荐默认生命周期改为：

```text
Qwen encode
  -> 保存小型 conditioning
  -> qwen_clear 两 rank barrier + 各 rank empty_cache

DiT step 0..N-1
  -> H3 正常 TP
  -> 背景线程从 /mnt/GALAX 读取 VAE
  -> direct-to-final-owner H2D，仅加载到 per-rank prefetch budget

最后一个 DiT forward完成
  -> H3 临时 activation已释放
  -> 同步补齐 VAE 剩余 tensors
  -> CUDA event确认 VAE READY

VAE decode
  -> 不再等待完整冷加载
  -> 若队列中存在下一条 conditioning cache miss，可尝试反向预取 Qwen
```

“offload Qwen”在本机不能表示 GPU→CPU。正确语义是删除 compressed CUDA shard，
保留 GGUF header/offset descriptor，使 payload重新成为 disk-backed；否则 7.9 GiB
CPU copy会突破 RAM/cgroup。rank1 子进程也必须执行 `empty_cache()` 并回报 driver 可见
释放完成，否则主进程在 GPU1 上加载 VAE 时仍可能因为另一个进程的 reserved blocks
失败。

#### FP16 双参考图的 capped prefetch

当前 FP16 12/24 owner分布为 `1886/3086 MiB`，双参考图 DiT 期间不能全量加载到
GPU1。按显存安全目标计算：

| 每卡目标余量 | DiT 中允许的 FP16 VAE GPU0/GPU1 | 最后需补齐的 GPU1 tail |
| --- | ---: | ---: |
| 约 512 MiB | `<=2474 / <=2299 MiB` | 约 `787 MiB` |
| 约 1 GiB | `<=1962 / <=1787 MiB` | 约 `1299 MiB` |

因此异步 loader 应支持“已读取但暂不 materialize”与“CUDA resident”分离，并按每卡
ledger 停在上限。默认先使用 1 GiB safety profile：DiT 期间可隐藏约 3.67 GiB 的
FP16 VAE，最后只同步补约 1.30 GiB；完成真实峰值测试后再考虑 512 MiB profile。

不要为了把全部 FP16 VAE提早塞入而直接改 VAE split。`18/18`、`20/16` 可能改善
权重平衡，但会改变 decode 的输出 owner/canvas 和临时峰值；它们只能作为独立 MP
placement benchmark，且必须证明逐元素输出不变、decode 峰值更好后再采用。

#### 异步 loader 实现约束

- safetensors 继续使用 header-only/no-host-mmap reader；
- 每设备一个低优先级 CUDA copy stream，CPU I/O 使用有界 worker；
- staging ring 默认 4–8 MiB，不预读完整 5.2 GB 到 page cache/RAM；
- tensor 直接落最终 VAE owner，不允许先集中到 GPU0 再移动；
- 用 CUDA event标记 tensor/module ready，不在每个 tensor 后全局 synchronize；
- H3 forward期间只做 H2D/分配，不执行 VAE dequant或 decode kernel；
- allocator ledger 同时看 PyTorch allocated、reserved 和 driver free，任一低于 safety
  立即暂停预取；
- decode 调用 `await_ready()`：先等后台已提交 copy，再在 DiT barrier后补齐 tail；
- 后台 OOM/读取错误必须丢弃半成品 handle，回退为 DiT 完成后的同步 no-mmap load；
- FP16 和 INT8 使用同一 async handle接口，但各自保存独立 profile和 owner bytes。

首版不建议在一个正在执行的 DiT block 中无上限加载完整 VAE。即使文件读取能与
GPU compute 重叠，后续 block仍会再次达到 QKV/SDPA 峰值；没有 resident cap 就会在
第二、第三个 denoise step才随机 OOM，难以复现。

#### 反向掩盖下一次 Qwen 加载

832×480 FP16 decode 的实测 peak 为 `2006/3395 MiB`。按 H3+Q2 compressed shards
粗算，decode 阶段仍可能有约 3 GiB/卡余量，因此可以把“当前 DiT 中加载 VAE”反向
配对成“当前 VAE decode 中加载下一请求 Qwen”。不过 1 MP FP16 disk-backed decode
还没有对应完整峰值，首版只在请求队列明确存在且显存 gate通过时启用：

1. 下一请求 conditioning cache命中：不加载 Qwen；
2. 下一请求 cache miss：低优先级预取 Qwen compressed layer shards；
3. 没有排队请求：不做无意义 SSD read；
4. Qwen prefetch不得增加当前视频 decode wall time超过预设门限。

建议新增开关：

```text
H3_ASYNC_VAE_LOAD=0                 # 完整 gate 前默认关闭
H3_ASYNC_VAE_SAFETY_MIB=1024
H3_ASYNC_VAE_STAGING_MIB=4
H3_ASYNC_VAE_PREFETCH_MIB=1962,1787 # GPU0,GPU1 capped resident MiB
```

当前 bridge 固定在 Qwen clear 后开始 prefetch，并在完整 sampler 调用返回后 finalize；
`START_STEP`、`FINALIZE` 和 VAE decode 期间反向 Qwen prefetch 尚未实现，因此启动器不
导出容易造成误解的空开关。

## 10. Conditioning cache

Qwen 只在 prompt/reference 改变时需要运行。历史保存结果的量级：

```text
纯文本 conditioning [1,36,5120] FP32   约 0.7 MiB
单参考图 [1,620,5120] FP32             约 13 MiB
```

因此优先缓存 conditioning，比强行常驻 7.9 GiB Q2 更划算。cache key至少包含：

- Q2 文件 fingerprint/revision；
- tokenizer 和 encoder options；
- 完整 prompt/token IDs；
- reference 图片内容 hash、顺序、resize/crop 参数；
- 模式（text、单图、首尾图、ref2va）；
- 所有会改变 token tags、mask、mRoPE 或 DeepStack 的选项。

缓存必须保存 FP32 conditioning 和完整 metadata/token tags，不能只保存主 tensor。
默认使用有上限的 CPU LRU；需要落盘时写 `/home/regen` 机械盘并原子替换，不写
`/mnt/GALAX`。相同 conditioning 换 seed/步数时直接命中，不启动 Qwen collective。

## 11. 分阶段实施

### P0：header 和 shard audit（已完成）

不使用 GPU，输出 JSON：

- 902 tensor 的 name/qtype/shape/offset/bytes；
- 50 层逐层 bytes；
- 每个 matrix 的 rank0/rank1 output-row descriptor；
- 两 shard bytes 求和与原 tensor完全相等；
- header-only RSS、payload mmap 数为 0。

2026-08-27 的真实模型报告中上述检查全部通过；qtype 为 `Q2_K 417 / Q3_K 50 /
F32 433 / F16 2`，language payload 为 `8,028,211,200` bytes。

### P1：MP correctness oracle（待完成）

实现 25/25 layer-MP，并跑：

- text-only；
- 单参考图；
- 首尾参考图；
- ref2va；
- output shape、token tags、finite、mRoPE/DeepStack 注入点检查。

MP 输出与普通单卡 Q2 workflow 对比。该阶段不讨论速度，只证明外层语义正确。

### P2：单层 output-row TP gate（待完成）

从真实 Q2 layer 0 开始，测试 `S=36/128/620`：

- full layer oracle；
- output-row TP；
- Q/K/V、attention gather、O、SwiGLU、Down、最终 hidden逐阶段误差；
- rank0/rank1 最终 hidden 一致；
- direct compressed shard dequant 与 full-dequant row slice 一致；
- forward、dequant、4 个 collective 的 CUDA event 时间和显存峰值。

目标是 bitwise identical。若因 kernel/layout 只能达到近似，至少要求：

```text
finite=true
rank0_vs_rank1 max_abs=0
TP_vs_full relative RMS <= 1e-6
TP_vs_full cosine >= 0.999999
```

超过门限不能以“Q2 本身已经有误差”为理由放行。

### P3：完整 50 层 conditioning gate（待完成）

同 prompt/reference 比较 Q2 MP 与 Q2 TP：

- `[1,36,5120]` text；
- `[1,620,5120]` 单参考图；
- text token、vision token、attention sink token分别统计；
- max abs、RMSE、relative RMS、cosine、finite、token tags；
- 每层输出可选保存 scalar 统计，不保存 50 份完整 hidden。

同时比较 TP 与 MP速度。预计 32B 的 compute/communication ratio 比 4B 更适合 TP，
但不预设加速比例；cold SSD read 和 warm resident 必须分开报告。

### P4：接入共享 H3 runtime（待完成）

异步 VAE handle/bridge 已存在，但以下 runtime 协议、rank1 命令和 Qwen 节点接线仍未完成：

- 扩展 rank1 protocol；
- Qwen/H3 共用同一 process group和 runtime lock；
- encode 结束执行 safe trim；
- 注册 header-only VAE async handle，并在 Qwen clear barrier后才允许开始预取；
- H3 forward期间按每卡 budget低优先级加载，最后一个 forward后 finalize tail；
- conditioning cache命中时不执行任何 Qwen collective；
- 异常路径两 rank 同时退出；
- 连续请求不重载 DiT、不创建第二个 rank1 进程。

### P5：端到端质量和显存 gate（待完成）

固定模型、LoRA、prompt、reference、seed、sampler、4 steps，依次测试：

1. `448x256` smoke；
2. `832x480 / 124f` text；
3. `832x480 / 124f` reference；
4. `1344x768 / 124f` text；
5. `1344x768 / 124f` 单参考图/ref2va；
6. `768x1344 / 124f` 首尾双参考图；
7. VAE decode 和音频。

每项保存 MP4、video/audio latent、双卡 allocated/reserved/nvidia-smi peak、RSS、
cgroup events、swap、读盘 bytes、Qwen/DiT/VAE阶段耗时、异步加载覆盖率、DiT slowdown、
finalize tail时间和 payload mmap audit。

TP placement gate 应以 Q2 MP 为参考；“32B Q2 是否比 4B 好”是另一组 same-seed
质量 A/B，不能把两者混成一个指标。

### P6：partial residency（待完成）

只有 `evict` 路线完整通过后才测试 `keep_layers=16/24/30/50`。分别统计：

- 新 prompt cold/warm Qwen 时间；
- 每次从 SSD 读取的 payload bytes；
- DiT 1 MP 峰值和安全余量；
- 两次相同 prompt不同 seed是否完全跳过 Qwen；
- reference 数量变化是否正确失效 cache并 trim；
- 连续 10 次请求 allocator reserve是否增长。

之后再做 VAE/Qwen 双向 overlap matrix：

- FP16/INT8 VAE同步加载 vs DiT 异步 capped load；
- no-ref/single-ref/two-ref各自 safety profile；
- VAE decode期间 Qwen prefetch开关对 decode wall time和下一请求 TTFT 的影响；
- queued cache-hit/cache-miss/无队列三种调度；
- 冷 page cache与暖 page cache分开报告。

## 12. 报告格式

每个 benchmark JSON 至少包含：

```text
model path/bytes/fingerprint
qtype counts and layer geometry
mode: MP | output_row_tp
prompt/reference hash（不要保存隐私图片本体）
sequence/token-tag counts
resident layer IDs per rank
payload bytes read per rank
host mmap count
RSS/cgroup/swap peak
GPU allocated/reserved/nvidia-smi peak per stage/rank
load/dequant/GEMM/attention/collective/total milliseconds
rank consistency
conditioning/latent/video/audio error metrics
finite/OOM/oom_kill
cache hit/miss and cache bytes
async VAE requested/resident/deferred bytes per rank
async VAE overlap seconds / finalize seconds / DiT slowdown
```

不要只报 microbenchmark，也不要只用 synthetic hidden 代替真实 conditioning 和最终
音视频 latent。

## 13. 实现文件和剩余边界

实现者可按以下边界组织，文件名允许调整，但不要把所有逻辑堆进节点注册文件：

| 文件 | 状态 | 职责 |
| --- | --- | --- |
| `custom_nodes/DualV100/h3_qwen32_q2_mp.py` | 已实现并通过线上 smoke | 完整 layer-MP backbone/runtime |
| `custom_nodes/DualV100/h3_qwen32_q2_tp.py` | 已实现、显式实验 | geometry、descriptor、direct shard loader、TP block/backbone |
| `custom_nodes/DualV100/h3_qwen32_tp_node.py` | 已实现 | runtime/CLIP proxy/Comfy 节点接口 |
| `custom_nodes/DualV100/h3_async_vae.py` | 已实现 | header-only VAE handle、budget ledger、后台 direct-owner loader |
| `custom_nodes/DualV100/h3_async_vae_bridge.py` | 已实现，待 runtime 接线 | Qwen clear、DiT sampler、finalize 生命周期桥接 |
| `custom_nodes/DualV100/h3_tp_runtime.py` | 已扩展 | shared runtime config、MP/TP Qwen selector、cache/residency 状态 |
| `custom_nodes/DualV100/h3_tp_rank1_worker.py` | 已实现 | H3 DiT rank1 command loop（MP 不新增 Qwen worker） |
| `custom_nodes/DualV100/__init__.py` | 已扩展 | 节点注册和默认开关保持 fail-closed |
| `scripts/audit_qwen32_q2_tp_layout.py` | 已实现并通过 | P0 header/shard audit |
| `scripts/test_h3_async_vae.py` | 已实现 | CPU-only async loader lifecycle和 bounded-I/O tests |
| `scripts/test_h3_async_vae_bridge.py` | 已实现 | CPU-only sampler bridge lifecycle tests |
| `scripts/benchmark_h3_qwen32_q2_tp.py` | 待实现 | P1–P3 MP/TP correctness与性能 |
| `scripts/compare_h3_qwen_conditioning.py` | 待扩展 | segment/token-tag 统计 |
| `scripts/setup_ubuntu.sh` | 已扩展 | 安装复制和 `py_compile` 新文件 |
| `workflows/qwen32-q2-tp-*` | 暂不创建 | P1–P4 完成后再增加 smoke、1MP、reference 工作流 |

仓库 `/home/regen/code/minimax_v100` 应作为实现源；稳定后由部署脚本同步到
`/home/regen/minimax-h3/ComfyUI/custom_nodes/DualV100`。不要只改已安装目录，也不要
只改 `.patch` 而让当前源代码与部署状态分叉。

## 14. 回退和默认策略

- Qwen32 layer-MP 默认 `H3_QWEN32_Q2_MODE=mp H3_QWEN32_Q2_MP=1`；
- output-row TP 默认 `H3_QWEN32_Q2_TP=0`，仅在显式 `mode=tp` 时启用；
- 当前 4B Q4 + ridge 工作流保持可用；
- Q2 output-row TP 任一精度 gate 失败，回退 Q2 layer-MP；
- 任一 1 MP/reference OOM 或安全余量不足，强制 `residency=evict`；
- 不通过 same-seed Q2-vs-4B 实际视频/音频评估，不宣称 Q2 一定改善成品；
- 不因 encoder TP成功而改变当前 H3 Q4/LoRA TP、FP32稳定岛、attention backend或
  VAE 路线。

## 15. 明确排除

本轮不做：

- 把 7.9 GiB Q2 放 CPU RAM 常驻；本机 RAM/cgroup 不允许；
- host payload mmap、完整 CPU state dict或依赖 swap保存模型；
- 第二套独立 NCCL rank1 worker；
- 直接复用旧 4B INT8 strict-TP input-column shard；
- 双参考图 1 MP 下无预算地提前 materialize完整 FP16 VAE；
- 为腾显存把 H3 DiT直接降到 Q3/Q2；质量风险与本任务无关；
- 在正确性完成前写自定义 Q2 Triton/CUDA GEMM；
- 将 TP 的轻微误差用“Q2 本身有量化误差”掩盖；
- 默认让 Qwen、1 MP 双参考图 DiT和 VAE 同时全常驻。

交接实现时应先完成 P0/P1/P2，并把报告交给用户确认，再进入共享 runtime 和端到端
工作流。这样能把“Qwen 语义质量”“TP 数值正确性”“DiT 显存容量”三个问题分开，
避免一次改动后无法判断质量下降来自哪一层。
