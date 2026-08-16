# MiniMax H3 on dual Tesla V100 16GB

这是双 V100 16GB 上运行 MiniMax H3 GGUF + Turbo LoRA 的实验部署仓库。目标是让主 DiT 保持 FP16，并把 DiT、文本编码器和 VAE 分配到两张卡，尽量减少 HDD 权重搬运。

本仓库只包含部署代码、补丁和 API 工作流，不包含模型、虚拟环境、缓存或生成结果。

## 当前状态

- Windows 已验证两张 V100 都能被 ComfyUI 使用：DiT 放 `cuda:0`，Qwen 文本编码器放 `cuda:1`。
- Windows/WSL 环境未获得可用 CUDA P2P/NCCL，因此没有实现真正的 Transformer Tensor Parallel。
- 832×480、约 5 秒、Turbo 4-step 的 FP16 推理能够进入采样，但原始路径第一步后出现 NaN。
- `patches/comfyui-minimax-h3-v100-fp16-rmsnorm.patch` 为 FP16 RMSNorm/qk 增加按行缩放保护。它保持主推理路径 FP16，已通过语法检查，但在切换 Ubuntu 前尚未完成整段视频验证，必须先跑小工作流复测。
- 静态加载节点能够把 Q2 文本编码器完整常驻 GPU1；DiT 静态常驻测试曾被 Windows 主机内存监控提前中断，不代表 CUDA OOM。

## 重要限制

NVLink 不会自动把两张 16GB 卡变成一张 32GB 卡。当前代码是“组件分卡”：文本编码器在 GPU1、DiT 在 GPU0，主要解决容量和重复加载问题。采样期间 GPU1 大部分时间会空闲。

要让两张卡同时计算每一个 DiT block，需要真正的 TP：切分 attention heads 和 MLP 通道，并在每层加入 NCCL collective。ComfyUI-GGUF 当前算子不具备这套分布式语义，因此 Ubuntu + NVLink 只是必要条件，不是自动开启 TP 的开关。详见 `docs/NVLINK_AND_TP.md`。

## 固定的已测试代码版本

- ComfyUI: `2a68ce33b4c9ea6ee4283e618a74560cefb32694`
- ComfyUI-GGUF: `72c8990f22b86b06a4c9f4cad628d18825160f79`
- ComfyUI-MultiGPU: `b51c99a525e9607e43545ee2a8b7694c74a4775a`
- ComfyUI-MiniMax-H3-Turbo: `4274783a23afcfdbea3b4876cb79effd6c510785`
- Windows 测试环境: PyTorch `2.8.0+cu126`, ComfyUI `0.31.0`

## Ubuntu 快速部署

建议先装好 NVIDIA 驱动，并确认两张卡和 NVLink 都被系统识别。然后：

```bash
git clone <your-github-url> minimax-h3-dual-v100
cd minimax-h3-dual-v100
INSTALL_ROOT=$HOME/minimax-h3 ./scripts/setup_ubuntu.sh
INSTALL_ROOT=$HOME/minimax-h3 ./scripts/check_nvlink.sh
```

把模型放到以下位置：

```text
$INSTALL_ROOT/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_fp8_Q4_0.gguf
$INSTALL_ROOT/ComfyUI/models/text_encoders/qwen3vl-32B-MiniMax-H3-Q2_K.gguf
$INSTALL_ROOT/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors
$INSTALL_ROOT/ComfyUI/models/loras/minimax_h3_turbo_v4_step600_ema.safetensors
```

启动 FP16 静态常驻配置：

```bash
INSTALL_ROOT=$HOME/minimax-h3 ./scripts/start_comfyui.sh
```

浏览器打开 `http://127.0.0.1:8188`。也可以直接提交 API 工作流：

```bash
$INSTALL_ROOT/.venv/bin/python scripts/submit_workflow.py \
  workflows/static-smoke-448x256-1step.json
```

先确认 1-step latent 不含 NaN，再跑：

```bash
$INSTALL_ROOT/.venv/bin/python scripts/submit_workflow.py \
  workflows/turbo-5s-832x480-stage1.json --wait
$INSTALL_ROOT/.venv/bin/python scripts/submit_workflow.py \
  workflows/turbo-5s-832x480-stage2.json --wait
```

## 工作流选择

- `static-smoke-448x256-1step.json`：最小 FP16 数值与静态常驻测试，不解码视频。
- `turbo-5s-832x480-stage1.json`：动态 GGUF 采样，输出嵌套音视频 latent。
- `turbo-5s-832x480-stage1-static.json`：静态常驻版本，速度优先，但 GPU0 激活空间更紧。
- `turbo-5s-832x480-stage2.json`：单独加载视频 VAE 并输出视频，避免采样和 VAE 同时占显存。

静态 loader 适合模型权重能完整放进对应 GPU 的情况；如果 480p 激活导致 GPU0 OOM，把 stage1 的 `UnetLoaderGGUFStaticVRAMMultiGPU` 改为 `UnetLoaderGGUFDynamicVRAMMultiGPU`，代价是速度明显下降。

## 模型删除前的安全性

根目录 `.gitignore` 已排除 `*.gguf`、`*.safetensors`、`*.pt`、输出和缓存。上传前仍建议运行：

```bash
git status --short
git ls-files | grep -Ei '\.(gguf|safetensors|pt|pth|bin)$' && exit 1 || true
```
