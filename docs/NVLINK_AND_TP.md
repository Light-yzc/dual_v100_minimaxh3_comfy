# NVLink、P2P 与 Tensor Parallel

## 这套代码现在做了什么

当前实现是组件级模型并行：

- GPU0：MiniMax H3 DiT GGUF + Turbo LoRA。
- GPU1：Qwen3-VL GGUF 文本编码器。
- VAE：第二阶段单独加载到 GPU1。

这能让两张 16GB 卡分别保存不同的大模型，减少 HDD 到 GPU 的重复搬运。它不会让两张卡在每个 DiT block 上同时计算。

## Ubuntu + NVLink 能直接改善什么

如果 CUDA P2P 可用，GPU1 产生的 conditioning 可以直接传给 GPU0，不必绕系统内存。NCCL 也能在 NVLink 上建立 collective。先运行：

```bash
INSTALL_ROOT=$HOME/minimax-h3 ./scripts/check_nvlink.sh
```

必须同时看到：

1. `nvidia-smi topo -m` 的 GPU0/GPU1 关系是 `NV#`，不是 `SYS` 或 `PHB`。
2. `torch.cuda.can_device_access_peer(0, 1)` 和反向都是 `True`。
3. CUDA peer copy 带宽明显高于 PCIe。
4. 两进程 NCCL all-reduce 成功，日志没有退回 socket 或共享内存传输。

NVLink 仍然不合并显存地址空间。任意单个未切分 tensor 仍必须放进一张卡的 16GB。

## 真正 TP 需要写什么

要让单个样本的 DiT denoise 同时使用两张卡，需要重写以下路径：

1. 进程模型：用 `torchrun --nproc_per_node=2` 启动，每张卡一个进程，建立 NCCL process group。
2. Attention：按 head 切分 q/k/v 和输出投影；每层 attention 输出后做一次 all-reduce 或 reduce-scatter/all-gather。
3. MLP：第一层按输出通道切分，第二层按输入通道切分；每层 MLP 输出后做一次 all-reduce。
4. AdaLN、RoPE、timestep 和 packed layout：在两卡复制，保证 token 排布完全一致。
5. GGUF：当前量化 Linear 包装器默认认为权重和激活属于单设备。需要让 loader 按 rank 切 GGUF tensor，并让量化 matmul 接受 local shard；简单套 `DataParallel` 或 `device_map` 不够。
6. LoRA：A/B 矩阵必须用与对应 base Linear 相同的列并行/行并行规则切分。
7. ComfyUI：主进程负责 API 和图执行，worker 进程只跑分布式 DiT；需要自定义执行节点或独立推理服务。

H3 每个 block 至少有 attention 和 MLP 两次跨卡归并。50 个 block、4 个采样 step 时会产生大量 collective；6-link NVLink 能降低通信成本，但不能把通信视为零。

## 推荐实施顺序

1. 先用本仓库的组件分卡路径确认 Ubuntu FP16 1-step、4-step 和 VAE 解码都正确。
2. 测量静态单 GPU DiT 的每层耗时、P2P 带宽和 NCCL all-reduce 带宽。
3. 先对一个 `DiTBlock` 做 FP16 非量化 TP 原型，比较输出误差。
4. 再接入 GGUF shard 和 Turbo LoRA shard。
5. 最后接 ComfyUI，而不是一开始就在图执行器里调 NCCL。

如果目标只是“能跑且减少 HDD 卡顿”，组件分卡 + 静态常驻优先级更高；如果目标是“单个视频更快”，才值得继续做 TP。
