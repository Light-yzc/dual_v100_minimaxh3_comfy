# MiniMax H3 Dual V100 — v0.2

这是一个可直接部署的 ComfyUI 分支，用于在两张 **Tesla V100-SXM2 16GB** 上运行
MiniMax H3 图像/首尾帧到音视频生成。仓库已经包含经过验证的 ComfyUI 核心修改、
custom nodes 和工作流；不包含模型、虚拟环境、输入图片或生成结果。

v0.2 的默认路线是：

```text
Qwen3-VL 32B Q2 layer-MP
  -> 官方 INT8-ConvRot DiT，持久 2-way NCCL TP
  -> 采样结束后延迟加载 INT8-ConvRot 视频 VAE decoder
  -> 双卡 VAE layer-MP + FP16 host 输出画布
```

## v0.2 更新

- 新增官方 INT8-ConvRot DiT 的有界 safetensors reader 和 V100 W8A16 路径。
- 新增 `MiniMaxH3Int8StaticLoader`，绕开原生 `UNETLoader` 的整模 DynamicVRAM/host staging。
- DiT shard 与 NCCL worker 跨请求常驻；下一次请求只释放 VAE decoder，不重复加载 DiT。
- INT8 视频 VAE decoder 延迟到 DiT 峰值之后加载，避免 720p 采样期显存冲突。
- 720p host 视频画布默认 FP16，配合 `--cache-none` 避免 RAM/cgroup 压力。
- Qwen32 默认使用数值等价的 layer-MP；output-row TP 仍是实验功能。
- Qwen32 layer-MP 默认预取下一层压缩 payload；实测逐元素一致并降低 TE wall time。
- 修复首尾帧模式的生命周期顺序：先 VAE encode，再 Qwen，避免 `state=dit_ready`。
- 补齐 reference image、first/last frame、832×480 和 720p 的 API/UI 工作流。

## 硬件与软件要求

| 项目 | 已验证配置 |
|---|---|
| GPU | 2 × Tesla V100-SXM2 16GB（SM70） |
| GPU 互联 | NVLink NV6；必须支持双向 CUDA P2P |
| 驱动 | 580.173.02 |
| PyTorch | 2.8.0+cu126 |
| Python | 3.13.15 |
| 主机内存 | 14 GiB；建议至少 16 GiB |
| 模型盘 | SSD；完整模型约需 55–60 GB |

V100 没有 BF16、FP8 或新架构的 INT8 Tensor Core。本项目保留 FP32 residual、
attention out 和 MLP FC2 数值岛，并用 SM70 FP16 Tensor Core 执行 W8A16 GEMM。

先检查两卡拓扑：

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status
```

GPU0/GPU1 之间应显示 `NV#`，不能是 `SYS`。六条 NVLink 应全部 active。

## 安装

```bash
git clone https://github.com/Light-yzc/minimax_v100.git minimax-h3-v100
cd minimax-h3-v100

python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r custom_nodes/ComfyUI-GGUF/requirements.txt
```

如果系统 CUDA/PyTorch 组合不同，请先按 PyTorch 官方方式安装支持 V100 的 CUDA wheel，
再安装其余 requirements。不要安装 BF16/FP8-only kernel 替换当前 SM70 路径。

## 模型目录

将完整文件放到以下位置：

```text
models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
models/text_encoders/qwen3vl-32B-MiniMax-H3-Q2_K.gguf
models/vae/minimax_h3_video_vae_int8_convrot.safetensors
models/vae/minimax_h3_audio_vae_fp32.safetensors
models/loras/minimax_h3_turbo_v4_step600_ema.safetensors
```

音频 VAE 必须保持 FP32。不要使用只有 header 或未下载完成的 safetensors；加载前应核对
文件大小和 SHA256。模型不属于本仓库，也不会被 Git 跟踪。

如模型在其他 SSD，可设置：

```bash
export H3_MODEL_DIR=/mnt/ssd/minimax-h3/models
```

## 启动

```bash
./start_dual_v100.sh
```

浏览器打开 <http://127.0.0.1:8188>。脚本默认启用：

- Qwen32 Q2 双卡 layer-MP；
- INT8 ConvRot online 路径和 Q-only compact SDPA；
- 异步 VAE，decoder 在 DiT 后加载；
- VAE `18/18` layer split、FP16 host canvas；
- `--cache-none`、禁用 pinned memory 和异步 offload。

可覆盖端口、模型和输出路径：

```bash
H3_PORT=8288 \
H3_MODEL_DIR=/mnt/ssd/minimax-h3/models \
H3_OUTPUT_DIR=/mnt/ssd/minimax-h3/output \
./start_dual_v100.sh
```

停止前台服务使用 `Ctrl-C`。生产环境可将同一命令放入 systemd user service，并按主机
内存设置 `MemoryHigh`/`MemoryMax`；14–16 GiB RAM 的机器不要从 IDE cgroup 直接启动。

## 工作流

发布工作流位于 `user/default/workflows/`，在 ComfyUI 的 Workflows 菜单中直接打开。

**v0.2 推荐入口：**

```text
H3-V100-15-int8-vae-mp-ref2v-832x480-124f-1step-ui.json
```

第一次先用这个工作流确认模型和双卡链路。它已经接好 static INT8 loader、Qwen32 MP、
持久 DiT TP、异步 VAE 和视频输出。默认是 `reference_image`；需要首尾帧时，把节点
`MiniMaxH3ReferenceKeyframeToVideoTP` 的 mode 改成 `first_last_frames`，接入首帧和
尾帧即可。确认 1 step 成功后，再打开下面的 4-step 或 720p 工作流。

| 工作流 | 用途 |
|---|---|
| `H3-V100-11-int8-*-smoke-448x256-1step` | 首次安装冒烟 |
| `H3-V100-09-int8-ref2v-832x480-124f-4step-ui` | 参考图，832×480 |
| `H3-V100-10-int8-fl2v-832x480-124f-4step-ui` | 首尾帧，832×480 |
| `H3-V100-14-int8-vae-mp-ref2v-720p-243f-{1,4}step-ui` | 720p 异步 VAE |
| `H3-V100-15-int8-vae-mp-ref2v-832x480-124f-1step-ui` | v0.2 快速 UI 入口 |

生产 INT8 工作流必须使用：

```text
MiniMaxH3Int8StaticLoader
  -> MiniMaxH3TensorParallelInt8
```

不要把 loader 换回原生 `UNETLoader`。原生路径会为随后被 TP 替换的 50 层 block 创建
额外 host/DynamicVRAM staging，可能增加数 GiB RSS 并导致重复加载。

`MiniMaxH3ReferenceKeyframeToVideoTP` 的 `mode` 可切换：

- `reference_image`：只使用 reference image；
- `first_last_frames`：使用首帧和尾帧作为时间锚点。

三个图像 socket 是 lazy 的，未选中的分支不会执行。帧数应使用 H3 的 `17k+5` 网格，
例如 22、124、243、362。

## 已验证结果

- 首尾帧，832×480、124 帧、1 step：E2E `175.62 s`（含冷加载），输出 H.264，
  分辨率和帧数经 ffprobe 验证。
- 参考图，1280×720、243 帧、1 step：成功输出 H.264 MP4；decoder 在 DiT 后加载。
- 720p、243 帧、4 step：成功出片；DiT 峰值之后每卡释放约 4.3 GiB reserved VRAM，
  再加载 decoder。
- 连续请求可复用 static skeleton 和持久 DiT TP，不创建第二份 DiT。
- Qwen32 两轮 layer-MP：无预取 `29.292 / 20.937 s`，默认 4 MiB staging 预取
  `20.513 / 19.975 s`；98/98 命中且 `max_abs=0`。16 MiB staging 更慢，未采用。
- 448×256/22 帧 smoke 连续两次成功：冷轮 `116.3 s`，常驻复用轮 `34.0 s`；
  第二轮 DiT forward `0.576 s`，VAE 两轮均为 144 个 INT8 Linear。
- static skeleton 冷启动进一步优化为 `108.1 s`，热轮 `32.3 s`；启动日志不再输出
  50 层被 TP 接管的预期 missing-weight 警告。
- Python 文件通过 `py_compile`；工作流 JSON 均通过解析检查。

这些数字是双 V100 实机记录，不是跨机器保证。长序列下 GPU1 如果热降频，rank0 会在
all-reduce 中等待，日志中的 `NCCL=70s` 可能主要是负载失衡而非 NVLink 传输时间。

## 内存与稳定性约束

- 保持 `H3_NO_HOST_MMAP=1`；不要恢复完整模型 mmap 或 CPU 权重副本。
- 720p 路线保持 `H3_ASYNC_VAE_PREFETCH_MIB=0,0`，避免 decoder 预取占用 DiT 峰值余量。
- 保持 `H3_ASYNC_VAE_OUTPUT_DTYPE=fp16`。720p/243f FP32 canvas 约 2.56 GiB，FP16
  约 1.28 GiB。
- 保持 `--cache-none`，避免长视频 IMAGE 节点被 RAM cache 长期持有。
- DiT 采样后只释放临时 activation；INT8 shard 和 NCCL communicator 继续常驻。
- 音频 VAE 保持 FP32；关闭 FP32 residual/FC2/attention-out 会造成 NaN、黑图或画质变化。

## 常见问题

### `Q4 model -> persistent 2-way TP failed`

确认工作流 loader、TP 节点和 checkpoint 格式一致。INT8 checkpoint 必须走
`MiniMaxH3Int8StaticLoader` + `MiniMaxH3TensorParallelInt8`；Q4 GGUF 使用 Q4 TP 节点。

### `VAE encoder must run before DiT prefetch; state=dit_ready`

v0.2 已修复首尾帧内部顺序。确认运行的是本仓库的 `custom_nodes/DualV100`，重启服务后
再提交，避免旧 Python 模块仍驻留进程。

### 第二次请求重新加载或 OOM

确认日志出现：

```text
[H3 INT8 loader] reusing cached static skeleton; persistent DiT TP remains resident
```

如果工作流仍使用原生 `UNETLoader`，请换回 v0.2 发布工作流。

### DiT 很慢、NCCL 数字异常大

先记录两卡温度和 SM clock。两 rank 的真实 collective 通常只有约 2 秒；某 rank 的
collective 很大而另一 rank 正常，表示前者在等待热降频或负载更慢的卡。

### 720p decode 后 RAM 压力

确认使用 `start_dual_v100.sh`、FP16 output canvas 和 `--cache-none`。不要提高 cgroup
上限来强行运行 362 帧；更长视频需要分块 decode 和增量写盘。

## 开发验证

```bash
.venv/bin/python -m py_compile \
  custom_nodes/DualV100/*.py custom_nodes/NoHostMMap/*.py

find user/default/workflows -name 'H3-V100-*.json' -print0 \
  | xargs -0 -n1 python -m json.tool >/dev/null
```

NCCL/runtime 改动还应运行真实双卡通信门禁和至少一个 448×256 smoke。可见工作流改动
必须保留输出 MP4、分辨率/帧数、GPU 峰值、host RSS 和 finite 检查记录。

## 上游与许可

本分支基于 ComfyUI commit `2a68ce33b4c9ea6ee4283e618a74560cefb32694`，并 vendored：

- ComfyUI-GGUF `72c8990f22b86b06a4c9f4cad628d18825160f79`
- ComfyUI-MultiGPU `b51c99a525e9607e43545ee2a8b7694c74a4775a`
- ComfyUI-MiniMax-H3-Turbo `4274783a23afcfdbea3b4876cb79effd6c510785`
- ComfyUI-ClipProj（随发布树固定）
- DualV100 / NoHostMMap（本项目）

完整分发包含 ComfyUI，因此遵循根目录 `LICENSE` 的 GPL-3.0。各 custom node 和模型还受
各自许可证约束；模型权重不随本仓库分发。
