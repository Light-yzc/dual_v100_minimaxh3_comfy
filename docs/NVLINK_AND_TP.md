# 双 V100 上的模型常驻、NVLink/P2P 与 H3 Tensor Parallel

本文记录当前机器上的实测结果、已经确认的边界，以及后续把 MiniMax H3 DiT 做成双卡 Tensor Parallel（TP）的实施路线。

本文的目标机器是两张 Tesla V100-SXM2 16 GB（SM70）。模型默认放在 `/mnt/GALAX` 固态，当前默认模型根目录为：

```text
/mnt/GALAX/minimax-h3/models
```

## 结论先行

> 2026-08-25 状态更新：本文中后部保留了 TP 接入前的诊断和规划，供理解设计过程；其中“TP 尚未实现”或“4B 仍有 NaN”的句子属于历史状态。当前实现、FP32 优化与最新 TP 复测以 [H3_V100_KERNEL_TP_ROUTE.md](H3_V100_KERNEL_TP_ROUTE.md) 为准。

当前生产配置是：

```text
GPU0 / rank0：H3 DiT TP shard 0 + Qwen 前 12 层/vision + H3 外围模块
GPU1 / rank1：H3 DiT TP shard 1 + Qwen 后 24 层 + ClipProj v3.1
两张卡：每个 denoise step 共同执行 DiT，Row-Parallel 输出使用 NCCL all-reduce
```

### 显存不均衡的根因：为什么 MP 不是简单 18/18

`18/18` 只均分了 Qwen language block 的数量，不代表两张卡的整卡显存会相等。
当前 H3 TP 是“主进程 rank0 + GPU1 长期 rank1 子进程”，GPU0 同时是 ComfyUI
请求 owner，因此有一组固定的额外负载：

| GPU0 的固定/额外负载 | GPU1 对应负载 |
| --- | --- |
| H3 TP rank0 shard，以及 ComfyUI 主进程、sampler、result/final owner | H3 TP rank1 shard，主要在独立长期 worker 中运行 |
| 输入和 conditioning 组装、FP32 packed residual、time embedding、segments、RoPE 的初始生成 | 通过 NCCL broadcast 接收相同的 forward 输入 |
| Qwen embedding、vision tower、前半 language layers | Qwen language tail、norm/head 和 ClipProj projection |
| 视频 VAE encoder、`x_embedder`、register/mask buffer、前半 decoder blocks | VAE decoder tail、`norm_out` 和 `proj_out` |

H3 TP 并没有把 activation 按 layer 分到不同卡：每个 rank 都要保留完整的 packed
sequence residual，并执行自己的 QKV/attention/MLP 临时 buffer。GPU0 已经多了
请求 owner 和外围模块，所以同样的长序列 workspace 会先把 GPU0 推到上限。

实测普通 1 MP（`1344×768 / 124f / 4-step`）的整卡峰值为：

```text
GPU0 15870 MiB / 16384 MiB
GPU1 14504 MiB / 16384 MiB
差距 1366 MiB；GPU0 只剩 514 MiB
```

首尾参考图会把 packed sequence 从 `S=37746` 增加到 `S=41798`。首图和尾图
即使使用同一个文件，也属于两个 reference slot，不会合并；额外 rows 会同时增加
QKV、SDPA workspace 和 FP32 residual 的峰值。旧的 `18/18 Qwen + 18/18 VAE`
路径在该场景中曾达到 GPU0 约 `15358 MiB`，第一层 QKV 临时 buffer 申请失败，
而 GPU1 约 `12742 MiB`，仍有明显余量。这不是 host mmap 导致的 VRAM 不均衡。

所以当前把可安全做 layer-MP 的外围模块向 GPU1 倾斜：

```text
Qwen language layers：18/18 → 12/24
VAE decoder blocks：  18/18 → 12/24
```

这只是给 GPU0 留出 request activation 和 attention workspace，不改变 H3 50 层
compute-heavy TP 的 head/FFN shard、Q4/LoRA layout、NCCL collective 或 FP32
数值路径。以后调分卡的预算应使用：

```text
peak[i] = H3_TP_shard[i]
        + Qwen/VAE resident[i]
        + rank-owner[i]
        + request activation[i]
        + kernel/NCCL workspace[i]
```

目标是两张卡都保留安全余量，而不是让 `nvidia-smi` 的瞬时数字完全相等。未经
weighted-shard、collective、finite 和 same-seed quality 验证，不要为了均衡直接
改成非对称 H3 TP。

完整 50 层 Q4_0 + Turbo LoRA TP 已接入 ComfyUI：主进程为 rank0，长期子进程为 rank1。两 rank 各自常驻约 6.1 GiB shard，QKV/FC1 做 column parallel，out_proj/FC2 做 row parallel，并保留 FP32 residual、FP32 projection output 和 FP32 NCCL。

模型加载使用 header-only reader、4 MiB bounded staging 和 `DONTNEED`，两进程的模型 payload mmap 数为 0。十次保存的 forward 均为 `models_reloaded=false`；`1344×768 / 124f / 4-step` 已完成，采样 `196.464 s`，音视频 finite。

4B 路径早期“乱码/NaN”的根因不是 ClipProj，而是 BF16 Turbo LoRA tensor 被同宽按 FP16 字节解释。修复为 BF16 materialize 后再做数值转换，448、832、1344 三档输出均已正常。当前 projection 已统一为 26 MB 的 `mmh3-4b-ClipProj-v3.1` ridge；residual MLP 只保留历史精度数据，不再进入默认工作流。32B Q2 仅保留为质量参考。

视频 VAE 采用 12/24 decoder MP 常驻；音频 VAE 仍只在 decode 阶段从 `/mnt/GALAX` fault，避免把约 0.6 GB 的 FP32 音频模块与 1 MP 采样峰值叠加。TP/LoRA/4B encoder/ClipProj 不卸载。

## 历史状态：TP 接入前的组件级分卡

当前代码是组件级模型并行，不是真正的 TP：

```text
GPU0：H3 DiT Q4 + Turbo LoRA
GPU1：Qwen3-VL 文本编码器
VAE：目前按阶段放到可用的 GPU 上执行
```

GPU1 生成的 conditioning 可以通过 CUDA P2P/NVLink 传给 GPU0，避免绕主机内存。这个方案能让两个大组件同时驻留，但 GPU0 上的 DiT 仍由一张卡独立计算。

已经有的通信和分卡能力包括：

- CUDA peer access 检查。
- GPU 到 GPU 的 P2P 传输。
- NCCL 两进程 all-reduce 测试。
- conditioning/latent 的跨进程传递和 finite 检查。
- 组件级常驻 smoke 测试。

这些能力证明了 NVLink/P2P 路径可用，但不能等价为“DiT 已经 TP”。

## 显存实测

以下数字来自本机实际加载/运行记录，不是模型文件大小的简单换算。不同分配器、workspace、显存碎片和工作流尺寸会使最终 `nvidia-smi` 数字上浮。

| 组件 | 加载方式 | 模型/运行时占用 | 观测值与说明 |
| --- | --- | ---: | --- |
| Qwen3-VL-4B + ClipProj | INT8 Encoder，ClipProj resident | Encoder 约 `4.53 GiB`，projection/runtime 约 `0.47 GiB`，合计约 `5.0 GiB` | GPU1 曾观测约 `5078 MiB` |
| Qwen3-VL-32B | `Q2_K` GGUF | 文件 `8,487,968,160` bytes，约 `7.91 GiB`；staged 约 `8101 MiB` | GPU1 曾观测约 `8374 MiB` |
| H3 DiT | Q4 GGUF | staged 约 `10932 MB` | 实际 GPU0 曾达到约 `13.2 GiB`，还要留 activation/workspace |

这里的 `4B projection/runtime` 不是说 learned projection 矩阵本身有 0.47 GiB；小 projection 权重本身很小，表中的数字反映本次运行的 projection、临时张量和分配器保留。

显存预算不能只把文件大小相加：

- DiT 推理需要长序列 activation、attention workspace、量化反量化临时 buffer 和 LoRA 工作区。
- TP 后每张卡都会有本地 activation 和 NCCL buffer。
- VAE 解码的峰值与帧数、分辨率和是否同时保留 DiT activation 有关。
- `nvidia-smi` 的 reserved/used 与 PyTorch allocated 并不相同。

### 方案取舍

| 方案 | 优点 | 主要问题 | 当前定位 |
| --- | --- | --- | --- |
| 32B Q2_K | 保留 32B 架构和容量，适合做质量参照 | 仍然要跑 32B 层数；GPU1 再放 DiT shard 和 workspace 会很紧 | 质量基线/fallback |
| 4B INT8 + ClipProj | 约 5 GiB，计算量和显存访问都显著下降，适合与 DiT shard 共存 | 当前 conditioning 在 H3 路径出现 Inf/NaN，需修数值尺度 | 最终部署候选 |
| 32B INT8/更低比特 | 可能比 Q2 稳定 | 仍是 32B 计算，且 V100 不适合 NVFP4/FP8 路线 | 不作为首选 |

### 4B 与 32B Q2 的质量判断

两者不是同一种压缩：

```text
32B Q2_K：32B 网络 ──低比特量化──> 仍然是 32B 的 hidden states
4B ClipProj：4B 网络 ──学习到的 projection──> H3 需要的 5120 维 conditioning
```

32B Q2 的优势是模型容量还在，通常更适合复杂语义、长提示词和专有名词；风险是 Q2 量化误差会直接污染 text encoder 的 hidden state，而且每次仍要承担 32B 的 prefill/内存访问成本。4B ClipProj 的优势是速度和显存；风险是 4B 的语义容量更低，projection 只是对齐 hidden state，并不能恢复 32B 原本没有生成的语义信息。

目前没有完成足以给出百分比结论的同 prompt、同 seed、同 DiT、同采样步数视频评测，所以不能仅凭文件大小或 hidden-state 的 RMS 断言“32B Q2 一定比 4B 好”或反过来。当前工程状态还多一个更直接的事实：4B 路径尚未数值稳定。

已有诊断数据：

```text
4B ClipProj conditioning: rms=583.447,  max_abs=218240
32B Q2 conditioning:      rms=40.0841, max_abs=14850.8
```

这只是尺度/数值诊断，不是画质指标。4B conditioning 曾被 H3 前处理强制 cast 到 FP16，`218240` 超过 FP16 最大有限值 `65504`，产生 Inf；临时保留 FP32 后可以继续进入 DiT，但在第 43 个 block 的 `attn.qkv_proj` 出现 NaN。因此当前结论是：

- 32B Q2 当前可作为参考输出和 fallback。
- 4B ClipProj 的方向正确，但要先处理 conditioning 的 scale、projection 输出 dtype、H3 `condition_proj` 前后的稳定性岛和量化 Linear 的临时转换。
- 修复后必须用同一 prompt/seed 对比 4B、32B Q2 和可用的 32B 高精度参考，检查 latent 的有限性、余弦相似度和实际视频/音频质量。

## 不 mmap 到 RAM：加载和常驻约束

模型必须从 `/mnt/GALAX` 读取，且大文件加载不能走会把整个文件映射进主机地址空间的路径。当前工作的安全约束是：

- GGUF、safetensors 和 projection 都使用 no-host-mmap 或有上限的 staged/chunked loader。
- 不把 `/mnt/GALAX` 模型复制到 `/home` 或 tmpfs。
- 模型加载期间同时监控进程 RSS、PyTorch CPU allocated 和两张卡的 `nvidia-smi` used/reserved。
- 在单个大权重成功进入 GPU 后及时释放 CPU staging buffer；不能让 CPU 端和 GPU 端长期各保留一份完整权重。
- 任何新 loader 先用小模型和低分辨率 smoke 测试，不直接加载官方 32B 全精度文件。
- 发现 RSS 在加载瞬间跳升时，先停止该路径，不用“多开一个进程”掩盖问题。

“不重复 load/unload”与“不 mmap”是两个独立问题：前者要求模型对象常驻，后者要求首次加载时使用受控的 CPU staging。目标架构应当是长生命周期 worker：启动时加载一次，后续请求只复用同一组模型引用。

## 常驻模型架构

### 第一阶段：当前组件级分卡

这是最容易继续使用、风险最低的路径：

```text
主服务/ComfyUI
  ├─ GPU1 常驻 4B Encoder + ClipProj
  ├─ GPU0 常驻完整 DiT
  ├─ conditioning 通过 P2P/NVLink 传给 GPU0
  └─ VAE 常驻在有预算的一侧，或由独立 VAE worker 持有
```

如果选择 32B Q2，GPU1 约有 8.2 GiB 左右的 Encoder 运行时占用，留给 VAE 或 DiT shard 的空间很少；所以 32B 更适合作为单独的质量基线进程，而不是和 TP shard、VAE 强行堆在同一张 16 GB 卡上。

### 第二阶段：目标 2-way TP 常驻服务

```text
torchrun --standalone --nproc_per_node=2 ...

rank0 / GPU0                         rank1 / GPU1
----------------                    ----------------
DiT shard 0                         DiT shard 1
VAE resident                        4B INT8 Encoder resident
API/结果收集                         ClipProj resident
        \                              /
         \---- NCCL process group ----/
```

两张卡都必须持有一部分 DiT；VAE/Encoder 只是与各自的 DiT shard 共存，不是把一张卡永久“专门给 VAE”、另一张卡永久“专门给 DiT”后还能让 DiT 透明使用双卡。

每次请求的生命周期应是：

1. worker 启动时初始化 CUDA/NCCL，并从 `/mnt/GALAX` 受控加载常驻组件。
2. Encoder 只在首次启动或显式切换模型时加载。
3. DiT shard、LoRA shard、VAE 都保留 Python/C++ 引用，不在一次生成结束后调用全局 unload/offload。
4. 每个请求只创建 prompt、latent 和有界的临时 buffer。
5. 请求结束后释放临时 activation，但保留模型权重和 NCCL communicator。
6. 模型切换作为独立的维护操作执行，不与普通视频生成混在同一条路径里。

ComfyUI 可以继续作为前端，但 TP worker 应该是明确的双进程服务或自定义执行节点。不能让 ComfyUI 的通用 low-VRAM/offload 策略在每个节点执行时重新把模型搬来搬去。

## H3 TP 的真实分片方式

当前 H3 DiT 的关键尺寸为：

```text
hidden_size          = 5376
num_layers           = 50
num_attention_heads  = 56
attention_head_dim   = 128
attention inner      = 56 * 128 = 7168
ffn_hidden_size      = 14336
```

这里 `hidden_size` 不等于 attention inner width；不能把 QKV 当成 `3 * 5376`。2-way TP 的自然切法是：

| 部分 | 未切分形状/语义 | 每卡本地部分 | 跨卡操作 |
| --- | --- | --- | --- |
| 输入 hidden | `[S, 5376]` | 两卡各保留一份 | 通常复制或 broadcast |
| `qkv_proj` | 输出为 `3 * 7168` | 28 heads，即每个 Q/K/V 各 `3584`；融合输出 `3 * 3584` | 无，先做本地 attention |
| Q/K/V attention | 56 heads | 每卡 28 heads，head dim 128 | 无，attention 在本卡完成 |
| `out_proj` | `7168 -> 5376` | 输入列按 3584 切；每卡输出一份 `[S, 5376]` partial | `all_reduce(sum)` |
| `fc1`/`gate_up` | `5376 -> 2 * 14336` | 输出通道切半；融合 SwiGLU 时保持 gate/up 配对，每卡本地 ffn 宽度 7168 | 无，激活和门控本地完成 |
| `fc2`/`down_proj` | `14336 -> 5376` | 输入列每卡 7168 | `all_reduce(sum)` |
| RMSNorm/AdaLN/RoPE/time | 小的逐 token 或逐 channel 运算 | 两卡复制 | 无 |
| text/reference/audio context | packed context | 初版两卡保留相同 layout | 进入 block 前 broadcast 一次或由 rank0/rank1 同步 |

典型 block 的数据流是：

```text
x replicated on both ranks
  ├─ local qkv_proj (head/column parallel)
  ├─ local 28-head attention
  ├─ local out_proj (row parallel)
  ├─ NCCL all_reduce(sum)
  ├─ residual/AdaLN
  ├─ local gate_up/fc1 (output parallel)
  ├─ local SwiGLU
  ├─ local fc2/down_proj (input parallel)
  ├─ NCCL all_reduce(sum)
  └─ next block
```

所以 TP 不是“只把输入 shape 改成一半”。必须同时做到：

- 权重按数学维度切分，而不是按 GGUF 文件字节截半。
- 每卡的 activation layout 与本地 kernel 的输入约定一致。
- Row-Parallel 输出做 sum；不做归并会得到两份错误的 partial output。
- LoRA 的 A/B 矩阵按对应 base Linear 的列并行/行并行规则切分。
- `token_refiner` 的 attention/MLP 也要纳入同样的规则；只有真正小的前后处理模块才适合复制。

原来的单卡 attention、SwiGLU、RMSNorm、RoPE 等 kernel 在满足输入 layout 不变的前提下可以继续复用。新增的主要是分片 Linear wrapper、collective 边界和 loader；不是所有算子都要重写。但如果原 kernel 内部硬编码了完整的 head 数、hidden width 或 contiguous stride，就需要先把这些参数化。

## 通信技术栈怎么分工

| 技术 | 应负责的工作 | 不应负责的工作 |
| --- | --- | --- |
| NCCL / ProcessGroupNCCL | `all_reduce`、`reduce_scatter`、`all_gather`、`send/recv`，利用 NVLink 做 GPU 间通信 | 不负责 Linear、attention 或 Q4 解码数学 |
| Triton | 单卡本地 elementwise、layout transform、局部 attention/量化计算原型 | 不在 Triton kernel 内直接调用 NCCL |
| CuTe/CUTLASS | tiled GEMM、量化 GEMM、attention tile 和复杂 tensor layout | 不是通信库 |
| CUDA C++ | SM70 专用 kernel、融合通信/计算、`cudaMemcpyPeerAsync` 包装、性能最终版 | 不需要一开始就替换所有 PyTorch/Triton 算子 |
| PyTorch distributed | 进程组初始化、collective 调度和原型验证 | 不能替代正确的权重分片 |

推荐的第一版是：

```text
torchrun：一张卡一个 rank
Triton/CUDA：每个 rank 做本地计算
torch.distributed + NCCL：在 Python/C++ host 侧做 collective
```

通信不是在 Triton kernel 中“聚合”的。正确顺序是：本地 kernel 写完 partial tensor，host 侧在合适的 CUDA stream 上发起 NCCL collective，再让下一个本地 kernel 消费归并结果。后续如果 profiling 证明 launch/同步开销明显，再考虑 NCCL device API 或 CUDA C++ 融合；第一版不需要这个复杂度。

## 最小 Triton + NCCL + P2P 示例

下面的例子展示职责边界，不代表完整 H3 TP。把片段另存为临时脚本后，可用下面方式运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 scripts/tp_triton_nccl_example.py
```

一个 rank 内的 Triton 本地算子可以这样写：

```python
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def local_scale_add_kernel(x_ptr, y_ptr, n_elements, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, x * alpha + 1.0, mask=mask)


def local_kernel(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    local_scale_add_kernel[(triton.cdiv(n, 256),)](
        x, y, n, 0.5, BLOCK=256
    )
    return y


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    # 模拟 Row-Parallel Linear 的本地 partial output。
    x = torch.ones(1024 * 1024, device="cuda", dtype=torch.float16)
    partial = local_kernel(x)

    # NCCL 在 Triton kernel 返回之后执行；这是聚合，不是 Triton 算子的一部分。
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    expected = torch.full_like(partial, 3.0)  # 两个 rank 各贡献 1.5
    torch.testing.assert_close(partial, expected)

    # 只有在明确需要“所有权转移”时才用 P2P send/recv。
    if rank == 0:
        request = dist.isend(partial, dst=1, tag=17)
    else:
        received = torch.empty_like(partial)
        request = dist.irecv(received, src=0, tag=17)
    request.wait()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

在真正的 H3 TP wrapper 中，`local_kernel(x)` 会替换为本地 `qkv_proj`、attention、`out_proj` 或 MLP shard。使用原则是：

- `all_reduce`：两卡都有 partial output，结果两卡都要继续使用；这是 row-parallel `out_proj/fc2` 的默认选择。
- `reduce_scatter`：后续计算只需要各卡的一部分 hidden 时，可以把求和和切分合并，减少中间 buffer。
- `all_gather`：后续算子需要完整 hidden 时再聚合；不要无条件在每个小算子后 all-gather。
- `send/recv`：只有一个 rank 产生、另一个 rank 消费时使用，例如组件级 conditioning 或最终 latent 的明确所有权转移。
- `cudaMemcpyPeerAsync`：适合显式 CUDA P2P copy，但不能代替 TP 中需要数学求和的 all-reduce。

NCCL collective 需要所有 rank 以相同顺序进入。某个 rank 因为 Triton kernel、异常或 shape 分支提前退出，另一个 rank 往往会卡在 NCCL；因此每个 collective 前后都要有明确的 shape、dtype、device 和 finite 检查。

## NCCL 通信量和收益预估

已有通信 benchmark 使用接近 `1344x768 / 124 帧 / 4 step` 的 packed shape，默认 H3 参数为 50 个 DiT block、每个 block 两次归并：

```text
每次 all-reduce payload：约 389 MiB（FP16 [S, 5376]）
collective 次数：50 × 2 × 4 = 400
实测总通信：约 1.512 s
实测 bus bandwidth：约 100.6 GiB/s
```

这只是通信可行性测试，不是端到端 TP 加速结果。它还没有计入本地 GEMM、attention、kernel launch 和 rank 间同步，而且采用每次两卡都保留完整 hidden 的 all-reduce 估算。真实实现可以在合适的 block 边界尝试 `reduce_scatter/all_gather`，但要先证明 layout 和数值完全正确。

TP 是否比单卡快，必须用完整 denoise forward 测量：

- 单个 block 的本地计算时间。
- NCCL collective 时间和等待时间。
- 50 层、1/4 step 的总时间。
- GPU0/GPU1 峰值显存和主机 RSS。
- same seed 下 latent 的数值差异以及最终视频/音频质量。

NVLink 不会合并两张卡的显存。任何未切分的 tensor 仍必须适合单卡 16 GB；TP 只是把可切分的权重和计算分布到两张卡。

## Q4 GGUF 和 Turbo LoRA 的主要难点

FP16 或普通 INT8 Linear 的 TP 主要是分片和 collective；当前 H3 Q4 GGUF 更难，原因是量化存储布局不是普通连续 FP16 矩阵：

- QKV、`fc1/gate_up` 按输出通道切，通常比较直接，但必须按量化 group 边界检查。
- `out_proj`、`fc2/down_proj` 按输入通道切时，容易切断 Q4 的 scale/zero/group/pack 边界。
- GGUF 文件不能简单按字节截半，也不能只把一个完整量化 tensor 的 storage view 分给两个 rank。
- `convrot`/量化 matmul 的 layout 需要支持本地 shard；必要时要在加载阶段重新打包成每卡格式。
- Turbo LoRA 的 qkv、`fc1`、`fc2` 必须与 base Linear 使用同一切分；AdaLN/小向量类 LoRA 可以复制。

推荐先实现一个与文件格式无关的 `TensorParallelLinear` 接口：

```text
完整 FP16 权重 -> 正确的 2-way shard -> 单 block 对比
                         |
                         +-> 再接 Q4 shard-aware loader
                         +-> 再接 Turbo LoRA shard
```

这样可以把“TP 数学错误”和“Q4 解码/排布错误”分开定位。

## 分阶段实施路线与验收标准

### 阶段 0：内存安全和基线固定

- 默认模型根目录固定为 `/mnt/GALAX/minimax-h3/models`。
- 所有大模型路径走 no-host-mmap 或 bounded staging。
- 固定当前单卡/组件级工作流、seed、分辨率和采样步数。
- 每次测试前后记录 RSS、GPU used/reserved、PyTorch allocated/peak allocated。
- 确认同一 worker 连续生成两次时没有重新加载权重，也没有 RSS 无界增长。

验收：加载过程不出现 RAM 瞬时打满；重复请求只增长有界的临时 buffer；模型常驻在预期 GPU。

### 阶段 1：通信与 P2P 基线

- 运行 `scripts/check_nvlink.sh`，确认拓扑为 NVLink、双向 peer access 为 true。
- 运行 NCCL all-reduce、P2P send/recv 和 soak 测试。
- 固定 rank 到 GPU，禁止两个进程争用同一张卡。

验收：NCCL 不退回 socket/host staging；collective 没有 hang；带宽和延迟记录在案。

### 阶段 2：FP16 单个 H3 block TP

- 先不接 GGUF、不接 VAE、不接 LoRA。
- 用完整 FP16 权重实现 qkv head split、attention、row-parallel out projection、SwiGLU 和 row-parallel down projection。
- 每个 collective 后检查 shape、dtype、device 和 finite。
- 与单卡 block 对比最大绝对误差、相对误差、cosine similarity。

验收：随机输入和真实 H3 packed 输入都无 Inf/NaN；误差符合 FP16/NCCL 预期；两卡结果一致。

### 阶段 3：扩展到完整 DiT 和 LoRA

- 纳入 50 个 DiT block、2 层 token refiner、final layer 和必要的小模块。
- 接入 Turbo LoRA 的对应 shard。
- 先跑 448x256 低分辨率 1-step，再跑 832x480，最后跑 1344x768/4-step。

验收：same seed 下 TP latent 与单卡参考接近；所有中间和最终 latent finite；峰值显存低于 16 GB 并留出安全余量。

### 阶段 4：Q4 shard-aware loader

- 解析每个 Q4 tensor 的量化 group、scale、pack 和 convrot 排布。
- 为输出切分和输入切分分别实现 shard 规则。
- 加载时直接生成每卡所需的本地格式，避免先在 CPU 构造完整 FP16 权重。

验收：Q4 单 Linear、单 block、完整 DiT 逐层对比均通过；加载期间 RSS 受控；没有按字节截半的隐式错误。

### 阶段 5：常驻双进程服务

- `torchrun` 启动两个长期 worker，各自只绑定一个 GPU。
- 启动阶段一次性加载 DiT shard、4B Encoder、ClipProj、LoRA 和 VAE。
- 请求队列只传 prompt、seed、latent 配置和必要的小 tensor。
- 禁止普通请求触发全局 unload/offload；模型切换单独处理。
- ComfyUI 负责前端/工作流，TP worker 负责实际双卡 denoise。

验收：连续多次生成不重复 load/unload；RSS、显存和耗时稳定；异常请求能让两个 rank 一起退出或恢复，不留下僵死 NCCL 进程。

## 当前推荐的验证矩阵

```text
质量参考：32B Q2_K + 单卡/组件级路径
部署候选：4B INT8 + ClipProj v3.1 + 2-way H3 TP
数学参考：完整 FP16 DiT block
性能参考：当前单卡 Q4 DiT + PyTorch SDPA
通信参考：scripts/benchmark_h3_tp_comm.py
```

每次比较都固定 prompt、seed、帧数、分辨率、采样器和步数。不要用不同 quant、不同 seed 或不同 attention backend 的结果直接判断 4B/32B 的精度损失。

## 暂不做的事情

- 不在 V100 上直接把 NVFP4/FP8 路线当成主方案；V100 是 SM70，目标应是 FP16/普通 INT8/Q4 可用路径。
- 不把 Triton 当通信库，也不在第一版 Triton kernel 中嵌入 NCCL。
- 不把 Q4 GGUF 文件直接按字节一分为二。
- 不用 `DataParallel` 或普通 `device_map` 冒充 H3 TP。
- 不把 H3 的全局 attention 换成会改变可见范围的局部 `na3d`；SM70 attention kernel 另见 [H3_V100_ATTENTION_KERNEL.md](H3_V100_ATTENTION_KERNEL.md)。
- 不为测试恢复会导致 RAM 峰值失控的 host mmap 路径。

相关通信脚本：

- `scripts/check_nvlink.sh`
- `scripts/test_cuda_peer.py`
- `scripts/test_nccl.py`
- `scripts/benchmark_h3_tp_comm.py`
