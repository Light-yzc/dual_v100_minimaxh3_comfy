# Qwen32B Q2 layer-MP（解耦实现）

状态：**已实现、已完成线上 smoke，并已设为启动器默认**（2026-08-27）。
启动器默认使用 `H3_QWEN32_Q2_MODE=mp`、`H3_QWEN32_Q2_MP=1`；output-row
TP 保留为显式实验，需同时设置 `H3_QWEN32_Q2_MODE=tp` 和
`H3_QWEN32_Q2_TP=1`。MP/TP 均可通过环境变量切换，但不要在运行中的服务内热替换。

MP 已完成独立双 V100 text-only gate，并完成线上 ComfyUI 双卡 full smoke。
这些 gate 证明加载、跨卡执行、清理和节点链路正确；不把“MP 一定更快”或最终
画质当作结论，目标尺寸和多步图像/视频仍需同 seed 对照。

## 线上服务 gate（2026-08-27）

服务以以下配置启动并保持运行：

```bash
H3_QWEN32_Q2_MODE=mp H3_QWEN32_Q2_MP=1 H3_QWEN32_Q2_TP=0 \
H3_ASYNC_VAE_LOAD=0 H3_NO_HOST_MMAP=1 INSTALL_ROOT=/home/regen/minimax-h3 \
./scripts/start_comfyui.sh start
```

`workflows/qwen32-q2-mp-full-smoke-448x256-1step.json` 首次执行成功，耗时
**80.07 s**，H3 DiT forward **7.088 s**，峰值 GPU0/GPU1 **6586/6518 MiB**，
输出节点 14 生成有限值 latent（video `[1,24,7,16,28]`、audio `[1,32,2,37]`）。
随后将文本和 seed 改为 2012 再执行，仍成功（**38.17 s**，输出节点 14）；
ComfyUI 仅缓存静态节点，编码和采样重新运行，未出现 MP/TP 回退或错误。
测试后显存约 GPU0/GPU1 **7484/7752 MiB**，服务状态为 active。

## 上机 gate（2026-08-27）

- 硬件/软件：2× Tesla V100-SXM2-16GB，PyTorch 2.8.0+cu126，FP32 Qwen 算术。
- 命令：`PYTHONPATH=/home/regen/minimax-h3/ComfyUI:/home/regen/code/minimax_v100`
  配合 `Qwen32Q2LayerMPRuntime(..., devices=("cuda:0", "cuda:1"),
  layer_split="auto", residency="evict")`，tokenizer prompt 为 `a cat`。
- 结果：50 层完整真实 GGUF，自动 split **25/25**，耗时 **27.19 s**；输出
  `[1,2,5120]` FP32、finite，conditioning 在清理后回到 CPU。
- 通信/内存：一次 activation handoff **40,960 B**；峰值 allocated
  **1420/1421 MiB**、reserved **1616/1616 MiB**；evict 后 `loaded_layers=[]`、
  `resident_bytes=0`。重复 `qwen_clear()` 通过，最终两卡仅桌面约 7 MiB。
- 安全：reader 4 MiB staging，读取约 7.48 GiB 压缩 language payload，
  `payload_mmap_hits=[]`，无 OOM/NaN/Inf；MP 不创建独立 Qwen rank1/NCCL 通道
  （H3 DiT 的既有 TP worker 仍正常运行）。

## 设计

`custom_nodes/DualV100/h3_qwen32_q2_mp.py` 提供完整 layer/pipeline parallel：每个
language layer 的 7 个压缩矩阵和 4 个 norm 只放在一个 GPU；输入 activation 在
连续 layer boundary 处 handoff 一次，随后由尾卡返回结果。读取仍使用
`NoMmapGGUFReader` + 4 MiB bounded staging，禁止完整 CPU state dict、payload mmap
和隐式复制。`evict`（默认）逐层释放矩阵；`partial`/`full` 可用于驻留实验。

`plan_layer_split()` 只读 header 几何并结合 `torch.cuda.mem_get_info()` 的当前基线，
选择归一化峰值最小的连续边界；也接受显式 `H3_QWEN32_MP_SPLIT=12` 这类覆盖。
每次 stats 都报告两卡的 owner bytes、activation peak、handoff bytes 和 capacity
fit，便于把 GPU0 上的 DiT/vision 基线纳入平衡，而不是盲目固定 25/25。

## CPU 预检

```bash
/home/regen/minimax-h3/.venv/bin/python scripts/test_h3_qwen32_mp_contract.py
/home/regen/minimax-h3/.venv/bin/python scripts/audit_qwen32_q2_mp_plan.py \
  --devices cpu,cpu --split auto \
  --output results/qwen32_q2_mp_plan.json
```

真实文件的 header 审计应保持 902 tensors、50 layers、`payload_mmap_hits=[]`。
本次 CPU 规划结果为 25/25，压缩 owner 约 `3828/3828 MiB`，逐层 evict 估算峰值约
`653/653 MiB`（不含调用方 baseline）。有 DiT 常驻时请把 baseline 传给审计器或使用
显式 12/38；不要把这组空闲卡数字直接当成采样峰值。
CUDA 规划可在服务空闲时使用 `--devices cuda:0,cuda:1`；该命令不分配模型权重。

## 切换与回退

MP runtime 暴露与现有 Qwen subset 相同的 `qwen_forward/qwen_clear/qwen_stats` 接口，
所以切换只需在共享 runtime 工厂处：

```python
from .h3_qwen32_q2_mp import Qwen32Q2LayerMPRuntime
self.qwen_backend = Qwen32Q2LayerMPRuntime(model_path, layer_split="auto")
```

现有 CLIP facade 可通过 `Qwen32Q2MPRuntimeHandle`/`build_qwen32_mp_clip()` 复用；
不要在服务运行期间热替换 TP worker。推荐先用 `evict` 做 text-only、图像、448×256
和目标尺寸 same-seed 对照。启动器现在默认 MP；需要 TP 实验时重启服务并显式设置：

```bash
H3_QWEN32_Q2_MODE=tp H3_QWEN32_Q2_TP=1 H3_QWEN32_Q2_MP=0 \
./scripts/start_comfyui.sh restart
```

当前服务可用 `./scripts/start_comfyui.sh stop` 停止并释放显卡。

## TP 定位

output-row TP 继续保留在 `h3_qwen32_q2_tp.py`，仅作为显式实验路线（当前开关
`H3_QWEN32_Q2_TP=1`）。它不由本模块导入或自动回退；TP 的性能、collective 顺序和
数值 gate 仍按 [`QWEN32B_Q2_TP_PLAN.md`](QWEN32B_Q2_TP_PLAN.md) 单独记录。
TP 仍只作为显式实验路线；MP 的图像/视频质量和目标尺寸吞吐尚未完全 gate，但
已通过低资源线上 smoke，因此启动器默认 MP。目标尺寸/多步质量 gate 未完成前，
保留 `evict` 和自动 split，不要启用 full residency 或运行中热替换。

## 实验性逐层预取（动态 load / offload）

增加了默认关闭的单槽预取器，验证“当前 layer 的 dequant/GEMM 执行时读取下一
layer”能否遮住 Qwen/TE 的 SSD 延迟。worker 仅持有下一层压缩矩阵和小型 norm，默认
上限 256 MiB；消费后照常 `evict`，不保留 dense 权重、不改变 activation handoff
顺序，也不与 TP/NCCL 路径耦合。`partial/full` residency 会自动关闭它以避免重复
显存占用。读取失败会安全回退同步 no-mmap load；容量 gate 会把额外一层压缩 payload
纳入峰值估算。

默认保持关闭。空卡时可用以下方式做独立 A/B；不要在已有请求期间重启服务：

```bash
PYTHONPATH=/home/regen/minimax-h3/ComfyUI:$PWD \
  /home/regen/minimax-h3/.venv/bin/python \
  scripts/benchmark_h3_qwen32_mp_prefetch.py --prefetch 0 \
  --output results/qwen32_mp_prefetch_off.json \
  --dump results/qwen32_mp_prefetch_off.pt
PYTHONPATH=/home/regen/minimax-h3/ComfyUI:$PWD \
  /home/regen/minimax-h3/.venv/bin/python \
  scripts/benchmark_h3_qwen32_mp_prefetch.py --prefetch 1 \
  --output results/qwen32_mp_prefetch_on.json \
  --reference results/qwen32_mp_prefetch_off.pt \
  --dump results/qwen32_mp_prefetch_on.pt
```

比较 `mean_seconds`、`profile.backbone.prefetch`（或报告中的 `rank0.prefetch`）、`matrix_*_seconds`、
`layer_clear_seconds` 与两卡峰值；只有 checksum 一致、wall time 稳定下降且有显存
余量时，再以 `H3_QWEN32_MP_PREFETCH=1` 和
`H3_QWEN32_MP_PREFETCH_MAX_MIB=256` 重启启用。当前服务不会被此实现自动变更。

### 实机 A/B（2026-08-27）

空闲的 2× V100-SXM2-16GB、PyTorch 2.8.0+cu126 上，以真实
`qwen3vl-32B-MiniMax-H3-Q2_K.gguf`、`evict`、25/25 split、FP32、sequence=2
完成交叉复测。剔除首次 CUDA/页缓存初始化后，同步版为 **21.625 s**，单槽预取为
**20.236 s**（**6.4%**）；50 层均完成，49/49 预取命中、无 fallback/error，读取
8,028,211,200 B 且没有 payload mmap。两个输出逐元素相同（`max_abs=0`、finite）。

预取峰值 allocated 为 GPU0/GPU1 **1512/1389 MiB**，同步为 **1358/1358 MiB**；
清理后两卡均回到约 8 MiB allocated。完整的固定 seed 448×256/1-step MP 工作流也在
`H3_QWEN32_MP_PREFETCH=1` 下成功（输出节点 14、61.06 s）；这是功能 smoke，不与不同
冷/热状态的历史工作流时间作吞吐比较。由于收益仍受 SSD 和目标工作流影响，默认保持
`H3_QWEN32_MP_PREFETCH=0`，仅在留有约 200 MiB 以上余量并完成目标 workflow A/B 后启用。
