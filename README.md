# MiniMax H3 on dual Tesla V100 16GB

在两张 **Tesla V100-SXM2 16GB（SM70, NVLink NV6）** 上跑 MiniMax H3 视频生成的完整部署。

V100 没有 BF16、没有 FP8、没有 INT8 Tensor Core、没有 Flash Attention。H3 是 BF16
训练的模型，单卡 16 GB 装不下 DiT + 文本编码器 + VAE。这个仓库解决的就是这两件事：
把 50 层 DiT 切成两个常驻 NCCL rank 做张量并行，同时保留几个必须用 FP32 的数值岛，
让整条链路在 SM70 上既跑得动、又不出黑图。

**本仓库不含模型权重、虚拟环境、缓存或生成结果。** 上游 ComfyUI 及各 custom node
由 `scripts/setup_ubuntu.sh` 按固定 commit 拉取并打补丁，不在本仓库内分发。

## 目录

- [它做了什么](#它做了什么)
- [硬件与软件要求](#硬件与软件要求)
- [安装](#安装)
- [模型下载](#模型下载)
- [启动与停止](#启动与停止)
- [工作流](#工作流)
- [实测性能](#实测性能)
- [架构说明](#架构说明)
- [环境变量](#环境变量)
- [自定义节点清单](#自定义节点清单)
- [验证与门禁](#验证与门禁)
- [故障排查](#故障排查)
- [已被实测排除的路线](#已被实测排除的路线)
- [许可](#许可)

## 它做了什么

**双卡张量并行的 DiT。** 50 层 backbone 的 attention head 和 MLP 通道被显式切分到
两张卡，每层做两次 FP32 NCCL all-reduce。ComfyUI 主进程是 rank0，一个长期存活的
子进程是 rank1。权重按 Q4_0 常驻，用时逐 Linear 解量化成有界临时量。

**SM70 专用 kernel。** 融合的 RMSNorm + AdaLN modulation、QK-norm + RoPE、
Q4_0 直解量化、FP32 输出的 Tensor Core GEMM。都是 opt-in 或经过数值门禁的。

**FP32 数值岛。** V100 无法在原生 FP16 里安全累加 BF16 训练出来的 residual/MLP
动态范围。residual 流、attention out 投影和 MLP FC2 的输出保持 FP32，其余
（GGUF 权重、attention、多数 activation）仍是 FP16。这不是保守，是 832×480 上
不这么做就出黑图。

**INT8 视频 VAE + 阶段布局 rebalance。** 视频 VAE decoder 的 36 个 block 按
layer-MP 分到两卡。采样期和解码期对显存的需求方向相反，所以布局会在两个阶段
之间经 NVLink 切换（见[架构说明](#vae-阶段布局-rebalance)）。

**有界内存加载。** GGUF 和 safetensors 都走 header-only 读取 + 4–8 MiB staging，
直接落到目标 GPU。两个进程的模型 payload mmap 数均为 0；隔离服务默认
`MemoryMax=7G`。这台机器只有 14 GiB RAM，全量 mmap 会让 systemd-oomd 杀掉编辑器。

## 硬件与软件要求

已验证的环境：

| 项 | 值 |
|---|---|
| GPU | 2 × Tesla V100-SXM2-16GB (SM70) |
| 互联 | NVLink NV6，实测 P2P 131.8 GiB/s |
| 驱动 | 580.173.02 (CUDA 13.0) |
| PyTorch | 2.8.0+cu126 |
| Python | 3.13.15 |
| 主机内存 | 14 GiB（这是个约束，不是余量） |
| 模型盘 | SSD，约 60 GB 可用 |

**硬性要求**：两张卡之间必须有可用的 CUDA P2P。张量并行每层都要 all-reduce，
走 PCIe 会让通信成为瓶颈。用 `scripts/check_nvlink.sh` 确认。

**功耗说明**：V100-SXM2 的 `power.max_limit` 是 300 W 且不可调。长序列满载时
`SW Power Capping` 持续生效，稳态 SM 时钟约 1335 MHz（上限 1530）。这是物理限制，
不是配置问题——详见[实测性能](#长序列-dit-性能剖析)。

## 安装

```bash
git clone <this-repo> minimax_v100
cd minimax_v100

# 拉取上游、打补丁、装依赖、同步自定义节点
INSTALL_ROOT=$HOME/minimax-h3 ./scripts/setup_ubuntu.sh
```

脚本会做这些事：

1. 按固定 commit 克隆 ComfyUI 与 4 个 custom node（见下）
2. 应用 `patches/` 里的 8 个补丁（幂等，重复运行会先检测是否已打）
3. 创建 venv 并安装 pinned 依赖
4. 把本仓库的 `custom_nodes/DualV100` 与 `NoHostMMap` 同步进部署树
5. 对所有 Python 文件做语法检查

### 固定的上游版本

| 组件 | commit |
|---|---|
| ComfyUI | `2a68ce33b4c9ea6ee4283e618a74560cefb32694` |
| ComfyUI-GGUF | `72c8990f22b86b06a4c9f4cad628d18825160f79` |
| ComfyUI-MultiGPU | `b51c99a525e9607e43545ee2a8b7694c74a4775a` |
| ComfyUI-MiniMax-H3-Turbo | `4274783a23afcfdbea3b4876cb79effd6c510785` |
| ComfyUI-ClipProj | `c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c` |

这些 commit 是测过的组合。上游更新后 `patches/` 可能失效，不要随意升级。

### 只同步代码改动

改了 `custom_nodes/` 或 `workflows/` 之后不需要重装：

```bash
SYNC_ONLY=1 INSTALL_ROOT=$HOME/minimax-h3 ./scripts/setup_ubuntu.sh
```

这个模式只复制文件并做语法检查，不拉取上游、不重打补丁、不装依赖。

## 模型下载

```bash
./scripts/download_h3_v100_models.sh      # H3 DiT + Turbo LoRA + VAE + 32B 编码器
./scripts/download_h3_clipproj_models.sh   # 4B ClipProj 路线（可选）
```

生产路线需要的文件：

| 类别 | 文件 |
|---|---|
| DiT | `minimax_h3_fl2va_pruned_fp8_Q4_0.gguf` |
| LoRA | `minimax_h3_turbo_v4_step600_ema.safetensors` |
| 文本编码器 | `qwen3vl-32B-MiniMax-H3-Q2_K.gguf` |
| 视频 VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` |

音频 VAE **必须**是 FP32，不要换成 INT8 或 FP16。

模型默认放 `/mnt/GALAX/minimax-h3/models`，用 `H3_MODEL_DIR` 覆盖。TorchInductor
缓存和输出默认写另一块盘（`H3_MECHANICAL_ROOT`），减少 SSD 写入。

## 启动与停止

```bash
./scripts/start_comfyui.sh start     # 启动受保护的 user service
./scripts/start_comfyui.sh logs      # 跟随日志（Ctrl-C 只退出日志）
./scripts/start_comfyui.sh status    # 状态 + 最近日志
./scripts/start_comfyui.sh stop
./scripts/start_comfyui.sh restart
```

界面在 <http://127.0.0.1:8188>。

服务跑在一个内存受限的 systemd user unit 里（`MemoryHigh=6500M`、`MemoryMax=7G`、
`MemorySwapMax=256M`，可用 `H3_MEMORY_HIGH` / `H3_MEMORY_MAX` / `H3_MEMORY_SWAP_MAX`
覆盖）。这是刻意的：从 VS Code 终端直接启动会让大文件 page cache 记在编辑器的
cgroup 上，systemd-oomd 会杀掉编辑器而不是推理进程。

`start_comfyui.sh` 是**所有默认值的唯一来源**。`start_comfyui_isolated.sh` 只做
环境透传，不定义默认值。

命令行提交：

```bash
.venv/bin/python scripts/submit_workflow.py \
    workflows/H3-V100-09-int8-ref2v-832x480-124f-4step.json \
    --wait --timeout 2400 --output results/run.json
```

## 工作流

`workflows/H3-V100-*` 是入口预设，带 `-ui` 后缀的是网页版（含节点布局和说明便签），
不带的是 API 版。

| 预设 | 模式 | 尺寸 |
|---|---|---|
| `H3-V100-09-int8-ref2v-832x480-124f-4step` | 参考图 | 832×480 124帧 4步 |
| `H3-V100-10-int8-fl2v-832x480-124f-4step` | 首尾帧 | 832×480 124帧 4步 |
| `H3-V100-11-int8-ref2v-smoke-448x256-1step` | 参考图 | 448×256 22帧 1步 |
| `H3-V100-11-int8-fl2v-smoke-448x256-1step` | 首尾帧 | 448×256 22帧 1步 |
| `H3-V100-12-int8-ref2v-720p-243f-4step` | 参考图 | 1280×720 243帧 4步 |

先跑 `H3-V100-11-*` 冒烟（约 25 秒），确认链路通了再上大尺寸。

**帧数必须落在 17k+5 网格上**：5, 22, 39, ..., 124, ..., 243, ..., 362。
不在网格上的值会被 H3 静默对齐，你拿到的帧数和请求的不一致。

**切换 ref / 首尾帧不用重连线。** 条件节点的 `mode` 是普通 widget，三个图片输入是
lazy 求值：`reference_image` 模式只求值参考图输入，`first_last_frames` 只求值首尾帧
输入。未选中的输入不会被求值，也不会触发模型重载。

`h3-v100-multimode-*` 保留 FP16 视频 VAE 作为对照。另有 61 个历史基准工作流仍指向
FP16，用于复现过去记录的数字，不要改它们。

## 实测性能

### 端到端

| 尺寸 | 帧 | 步 | 序列 S | 单步 | 端到端 |
|---|---|---|---|---|---|
| 448×256 | 22 | 1 | 1159 | 0.65 s | 25 s |
| 832×480 | 124 | 4 | 15703 | 10.5 s | 82 s |
| 1280×720 | 243 | 1 | 68261 | 136 s | 291 s |

1280×720 243 帧时两卡整卡峰值 14816 / 14760 MiB（余量 1568 / 1624 MiB），
配置为 `H3_VAE_DIT_SPLIT=2`。

### VAE 阶段切分扫描（720p 243帧 1步）

| `H3_VAE_DIT_SPLIT` | GPU0 峰值 | GPU0 余量 | GPU1 峰值 | GPU1 余量 | NCCL 等待 |
|---|---|---|---|---|---|
| **2** | 14816 | **1568** | 14760 | 1624 | 5.8 s |
| 8 | 15200 | 1184 | 14376 | 2008 | 4.4 s |
| 18 | 15872 | **512** | 14218 | 2166 | **14.9 s** |

`dit_split=2` 是唯一两卡余量都超过 1.5 GiB 的配置。`dit_split=18` 下 GPU0 只剩
512 MiB，NCCL 等待涨到 14.9 s——那是显存压力导致的 rank 间倾斜。

搬移 22 个 block 经 NVLink 只需 32 ms。

### INT8 vs FP16 视频 VAE

同一条 `tiled_decode` 路径，448×256：

| 格式 | 解码额外显存 | 说明 |
|---|---|---|
| INT8-ConvRot | 约 306 MiB | W8A16：逐 Linear 反量化成有界临时量，用完即弃 |
| FP16 | > 11 GiB | 走 ComfyUI 普通 Linear，每层 activation 全程驻留 |

差 30 倍以上。在双卡有 DiT 常驻的情况下这个差距是决定性的。INT8 checkpoint 只
量化 decoder 的 144 个 Linear，encoder 的 116 个张量保持原精度，参考图编码质量不变。

### `H3_VAE_INT8_TILE_BATCH` 默认改为 1

批 tile 会在 batch 轴拼接，activation、attention 临时量与跨卡 handoff 都随之放大：

| tile_batch | 解码 | GPU0 增量 | GPU1 增量 |
|---|---|---|---|
| **1** | 0.362 s | 306 MiB | 262 MiB |
| 2 | 0.394 s | 438 MiB | 414 MiB |

在 layer-MP 下是慢 8.8% 又多吃 130 MiB 的净亏。单卡解码时权重反量化的摊销才划算。

### 长序列 DiT 性能剖析

S=68261（720p 243帧），stage profile 实测：

| 阶段 | 占比 | 效率 |
|---|---|---|
| attention SDPA | **79.6%** | 32.3 TFLOPS |
| MLP fc1 + fc2 | 7.0% | 81–84 TFLOPS |
| QKV + out 投影 | 4.5% | 85–88 TFLOPS |
| NCCL all-reduce | 1.2% | 稳态 13 ms/block |
| 融合 RMS+modulation | 0.2% | 830 GB/s（接近 HBM 上限） |
| Q4_0 解量化 | 0.2% | Triton 比 eager 快 2× |

**代码层面已无优化空间。** attention 占近八成且 SDPA 已达其实际上限，四个 GEMM
都在峰值 70%（Volta 上已经很好），融合 kernel 贴着 HBM 带宽。

限制器是 300 W 功耗墙：`power.max_limit` 不可调，`SW Power Capping` 累计 787 秒，
而 thermal slowdown 全为 0（温度 50–69°C 健康）。稳态时钟 1335 MHz 就是 300 W
能支撑的频率。**这也否掉了把 application clock 抬到 1530 的设想**——实测时钟已能
冲到 1470–1530，稳态 1335 是功耗决定的，不是时钟设置决定的。

## 架构说明

### 为什么必须是张量并行

NVLink 不会把两张 16 GB 卡变成一张 32 GB 卡。当前实现显式切分 attention head 和
MLP 通道，每层做两次 FP32 all-reduce。**不能**用普通 `device_map`、组件分卡或
`DataParallel` 替代——那些方案要么装不下，要么通信量和数值行为不对。

### VAE 阶段布局 rebalance

采样期和解码期对显存的需求方向相反：

- **采样期**：DiT 需要 cuda:0 尽量空。一个解码最优的 24/12 布局会把约 3.4 GiB
  （FP16）闲置的解码权重压在 cuda:0，这是 1280×736 在 QKV 投影处 OOM 的直接原因。
- **解码期**：layer-MP 解码是串行的，希望较重的一半在 cuda:0。

所以 VAE 按采样布局加载，首次解码前把边界 block 经 NVLink 搬到 cuda:0，下次采样
入口前搬回：

```
H3_VAE_DIT_SPLIT=2      采样期：2 个 block 在 cuda:0，34 个在 cuda:1
H3_VAE_DECODE_SPLIT=24  解码期：24 个在 cuda:0，12 个在 cuda:1
```

搬移只改权重驻留位置，不动 block 顺序、权重、dtype 或 forward 实现，因此数值上是
惰性的——门禁要求 `max_abs == 0.0` 而非 cosine 近似。

实现要点：

- 逐 block 搬移，每块之间同步并释放源副本，瞬时代价是一个 block 而非整段
- 整卡 admission check（用 `mem_get_info`，不是 torch allocator）不通过时自动退档
  到采样布局，不让后续分配在 collective 里失败
- INT8 的 FP16 scale 缓存随权重一起迁移，否则 rebalance 后第一个 Linear 会拿到
  跨设备操作数
- 挂在 `KSAMPLER.sample` 而非 TP forward：后者 residual 已驻留 cuda:0，搬权重会和
  activation 峰值重叠

设 `H3_VAE_SPLIT` 会钉死单一布局并关闭搬移。

### 显存为什么不平衡

采样期典型分布（832×480）：

```
GPU0: 主进程 15702 MiB  = DiT rank0 分片+激活 + VAE 采样布局权重 + CUDA context
GPU1: 主进程  1646 MiB  + rank1 独立进程 11186 MiB = 12832 MiB
```

GPU0 是瓶颈，因为 rank1 是**独立进程**，它的显存不计入主进程；而主进程要在 GPU0
上和自己的 DiT rank0 挤在一起。这就是采样期要把 VAE 尽量推到 cuda:1 的原因。

判断"某张卡还剩多少"只能看整卡。`forward_*.json` 里的 `allocated_mib` 只是该进程
torch allocator 内的量，看不到另一个 rank、NCCL buffer、CUDA context 和 cuBLAS
workspace。用 `scripts/sample_gpu_during_run.sh`。

### 数值精度取舍

Q4 相对 INT8 + ridge 的实测偏差必须知道：纯文本 relative RMS `1.67%`，参考图全局
`8.47%`，其中 vision token `17.94%`。这是为了显存和速度接受的默认，**不代表**与
INT8 或原版 32B 数值等价。

## 环境变量

`start_comfyui.sh` 导出 41 个 `H3_*` 变量并做校验。常用的：

### VAE

| 变量 | 默认 | 说明 |
|---|---|---|
| `H3_VAE_DIT_SPLIT` | `18` | 采样期 cuda:0 上的 decoder block 数 |
| `H3_VAE_DECODE_SPLIT` | `24` | 解码期 cuda:0 上的 block 数 |
| `H3_VAE_SPLIT` | 未设 | 设了就钉死单一布局，关闭 rebalance |
| `H3_VAE_REBALANCE_SAFETY_MIB` | `1024` | admission check 要保留的整卡余量 |
| `H3_VAE_INT8_TILE_BATCH` | `1` | 见上文实测，layer-MP 下不要设 2 |
| `H3_VAE_INT8_SM70_W8A16` | `1` | SM70 的 W8A16 回退路径 |
| `H3_VAE_OUTPUT_DEVICE` | `cpu` | 长视频输出缓冲位置 |
| `H3_VAE_MP_PIPELINE` | `0` | tile 跨 stage 流水（实验） |

720p 建议 `H3_VAE_DIT_SPLIT=2`。

### DiT 与数值

| 变量 | 默认 | 说明 |
|---|---|---|
| `H3_FP32_RESIDUAL` | `1` | FP32 residual 流，**不要关** |
| `H3_FP32_MLP` | `1` | FP32 MLP FC2 输出 |
| `H3_FP32_ATTN_OUT` | `1` | FP32 attention out 投影 |
| `H3_V100_FP32_TC` | `1` | FP32 输出的 Tensor Core GEMM |
| `H3_V100_ATTENTION` | `pytorch` | 保持 SDPA；Triton 版在 Volta 上更慢 |
| `H3_V100_RMS_ROPE` | `pytorch` | 融合 RMS-RoPE 单独 gate |
| `H3_TP_Q4_DEQUANT` | `eager` | Triton 版需显式 opt-in |
| `H3_TP_COMPACT_QKV` | `q` | 只把 Q 整理成 contiguous |

前四个是数值稳定性开关，关掉会在 832×480 出黑图。

### 内存与诊断

| 变量 | 默认 | 说明 |
|---|---|---|
| `H3_NO_HOST_MMAP` | `1` | header-only 加载，**不要设 0** |
| `H3_VRAM_MODE` | `safe` | `safe` / `resident` / `legacy-static` |
| `H3_DISABLE_PINNED_MEMORY` | `1` | 14 GiB 主机内存下的必要设置 |
| `H3_TP_PROFILE` | `0` | 每步 forward 计时 |
| `H3_TP_STAGE_PROFILE` | `0` | 分阶段 CUDA event 计时 |
| `H3_FINITE_TRACE` | `0` | 逐张量 finite 检查（会同步，很慢） |

覆盖方式：

```bash
H3_VAE_DIT_SPLIT=2 H3_TP_PROFILE=1 ./scripts/start_comfyui.sh restart
```

## 自定义节点清单

`custom_nodes/DualV100` 注册 21 个节点：

**运行时与模型加载**
`MiniMaxH3DualRuntimeLoader`（DiT + Qwen 共享的双卡运行时）、
`MiniMaxH3TensorParallel`（把 50 层换成持久 NCCL TP）、
`UnetLoaderGGUFDynamicVRAMMultiGPU`、`UnetLoaderGGUFStaticVRAMMultiGPU`、
`CLIPLoaderGGUFDynamicVRAMMultiGPU`、`CLIPLoaderGGUFStaticVRAMMultiGPU`、
`VAELoaderH3Device`（显式指定 VAE 驻留卡）

**文本编码器**
`Qwen32BQ2MPCLIPLoader`（layer-MP，默认）、`Qwen32BQ2TPCLIPLoader`（output-row TP，实验）

**条件构建**
`MiniMaxH3ReferenceKeyframeToVideoTP`（ref / 首尾帧双模式）、
`MiniMaxH3ReferenceToVideoTP`

**latent / conditioning 中转**
`SaveMiniMaxH3Latent`、`LoadMiniMaxH3Latent`、
`StoreMiniMaxH3LatentPeer`、`LoadMiniMaxH3LatentPeer`、`ClearMiniMaxH3LatentPeer`、
`StoreMiniMaxH3ConditioningPeer`、`LoadMiniMaxH3ConditioningPeer`、
`ClearMiniMaxH3ConditioningPeer`

**实验性加速**
`TESpeedMiniMaxH3TP`、`AdaptiveGroupResidualCacheMiniMaxH3TP`
（两者都未通过质量门禁，见[已被实测排除的路线](#已被实测排除的路线)）

`custom_nodes/NoHostMMap` 是 header-only 的 GGUF / safetensors 读取器，
保持有界内存行为，不注册节点。

## 验证与门禁

每次改 Python 都要跑语法检查：

```bash
$HOME/minimax-h3/.venv/bin/python -m py_compile \
    custom_nodes/DualV100/*.py custom_nodes/NoHostMMap/*.py
```

改了 NCCL / 运行时用通信门禁：

```bash
INSTALL_ROOT=$HOME/minimax-h3 ./scripts/benchmark_h3_tp_comm.sh
```

改了 VAE 布局用逐元素一致性门禁（要求 `max_abs == 0.0`）：

```bash
.venv/bin/python scripts/test_h3_vae_layout_rebalance.py \
    --width 832 --height 480 --frames 124 \
    --output results/vae_layout.json
```

整卡显存采样（rank telemetry 看不到全貌）：

```bash
./scripts/sample_gpu_during_run.sh mycase -- \
    .venv/bin/python scripts/submit_workflow.py workflows/xxx.json --wait
```

其他：`scripts/smoke_clipproj_v100.sh`、`scripts/scan_h3_vae_splits.py`、
`scripts/test_h3_vae_mp_pipeline.py`、`scripts/check_nvlink.sh`。

没有统一的单元测试套件——验证是硬件和工作流驱动的。记录时请写清确切命令、模型
revision、输出指标、GPU allocated/reserved 峰值、主机 RSS 和 mmap 状态。

## 故障排查

**rank0 OOM 后整机卡 900 秒**
已修复。`destroy_process_group` 本身是 collective，rank0 异常时 rank1 还在对应
collective 里，rank0 会阻塞在 `ncclCommDestroy → pthread_join` 直到看门狗超时。
现在改为先 kill 子进程再 `abort()`（只在本地拆通信器，不等对端）。

**长序列请求挂住不动**
rank1 可能已静默死亡（无法分配通信 buffer——720p 362 帧单步 all-reduce 是
5.42 亿元素约 2 GiB）。NCCL 无法把"对端已死"告知已进入 collective 的 rank。
现在进入 collective 前会用非阻塞 `poll()` 检查，秒级失败并给出可读信息。
720p 15s（362 帧）目前跑不通，就是这个原因。

**832×480 出黑图**
检查 FP32 岛是否被关掉：`H3_FP32_RESIDUAL`、`H3_FP32_MLP`、`H3_FP32_ATTN_OUT`
必须为 1。历史上另一个原因是 BF16 Turbo LoRA 被按 FP16 字节解释，已修复。

**QKV 投影处 OOM**
GPU0 余量不足。降 `H3_VAE_DIT_SPLIT`（720p 用 2），并确认没有设 `H3_VAE_SPLIT`
把布局钉死。用 `sample_gpu_during_run.sh` 看整卡峰值，不要只看 allocator 数字。

**systemd-oomd 杀掉编辑器**
不要从 VS Code 终端直接跑 `main.py`。用 `./scripts/start_comfyui.sh start`，
它会把进程放进受限的 user unit。也不要设 `H3_NO_HOST_MMAP=0`。

**帧数和请求的不一致**
帧数必须落在 17k+5 网格上，否则被静默对齐。

## 已被实测排除的路线

这些不是"没调好"，是已经量到上限或质量不合格。重做等于浪费机时。

| 路线 | 证据 | 结论 |
|---|---|---|
| 自定义 SM70 attention kernel | `S=37746` 纯 cuBLAS bmm 只有 32.3 TFLOPS，低于 efficient SDPA 的 34.0 | 除非改算法，否则方向不成立 |
| query-row 分块 | 最好 1.018×，端到端 1.013×，多 286 MiB 峰值 | 收益不抵风险 |
| sequence parallel | cosine 0.4838，2.01× 是漏算一半 head | 已作废 |
| TE-Speed tail42 | video relative RMS 0.730、cosine 0.721 | 质量不合格 |
| Group Cache `t=0.30` | video relative RMS 0.543、cosine 0.851 | 质量不合格 |
| Group Cache `t=0.005` | 质量合格但跳过 0 个 block，比 full 慢 6.2% | 判定开销吞掉收益 |
| `max-autotune` / 全路径 `torch.compile` | SM70 上 cuBLAS 已快于候选 Triton GEMM | 不进生产 |
| 抬 application clock 到 1530 | 限制器是 300 W 功耗墙，thermal slowdown 全 0 | 无收益 |

Group Cache 的结论要读准：不是"阈值再调调就能上"。`t=0.005` 安全但零命中且负收益，
`t=0.30` 有命中但质量崩，中间没有可用区间——因为判定用的是全局阈值和整张 Q4
`previous_input`。要继续做必须先改判定方式，不要再扫阈值。

**不要做的事**：恢复 host mmap 或完整 CPU 权重副本、用 FP8/NVFP4/BF16 TC/
Ampere-only 指令、用 `DataParallel` 或普通 `device_map` 冒充 TP、为测 kernel 并行
起第二个完整 ComfyUI 服务。

## 许可

本仓库代码按 **MIT** 授权（见 `LICENSE`）。

需要注意的上游许可：

- **ComfyUI 是 GPL-3.0**，`ComfyUI-MultiGPU` 也是 GPL。`patches/` 里有三个补丁修改
  ComfyUI 本体（`comfy/ldm/minimax/model.py`、`comfy/model_base.py`、
  `comfy_extras/nodes_minimax_h3.py`）。这些补丁以 diff 形式分发，应用后产生的
  ComfyUI 副本受 GPL-3.0 约束。
- `ComfyUI-GGUF` 与 `ComfyUI-MiniMax-H3-Turbo` 是 Apache-2.0，`ComfyUI-ClipProj` 是 MIT。
- 模型权重有各自的许可，本仓库不分发权重。

如果你打算把本仓库和 ComfyUI 源码打包成单一分发物，整体需按 GPL-3.0 处理。
当前形式（自己的代码 + patch + 按 commit 拉取上游）刻意避开了这个问题。
