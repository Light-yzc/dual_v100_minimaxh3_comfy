# MiniMax H3 双 V100 性能优化 TODO

更新时间：2026-08-26

状态：**本轮已冻结**。本文件中的“后续/待做”全部是下一轮候选，不会在当前服务上自动执行；本轮不再启动 CUDA benchmark、不切换默认 kernel、不提高 compile autotune 等级，也不改变 TP/MP 分配。

这份清单只作为下一轮工作台。当前生产路径已经跑通，后续改动必须以保存的 1 MP 结果为基线，不能为了单项 benchmark 破坏 FP32 防 NaN、no-host-mmap 或常驻 TP。

## 2026-08-26 本轮收口

本轮性能优化已冻结：生产保持 PyTorch efficient SDPA + Q-only layout、eager Q4
dequant、FP32 stability islands、resident 2-way TP 和 no-host-mmap。暂不继续调 Triton
attention、TileLang、`max-autotune` 或新的 TP 分配；下一轮从 Q-only 首尾双参考图
1 MP 成品/显存 gate 开始。固定默认值和启动入口集中记录在
[`H3_V100_FREEZE_20260826.md`](H3_V100_FREEZE_20260826.md)。

Qwen 严格 TP 的真实 36 层 gate 已完成：虽然 `S=256/512` 比 18/18 layer-MP
快 `19.7%/36.8%`，但 full-vs-TP relative RMS 为 `2.12%/2.56%`，未过精度
门槛。因此生产仍使用数值等价的 Qwen layer-MP；18/18 和 INT8 12/24 只作为精度
基线，当前 Q4 direct-owner 默认按实际权重采用 4/32。严格 TP 只作为后续数值修复实验，
不进入生产路线。详见 `docs/H3_V100_KERNEL_TP_ROUTE.md`。

当前进度：细粒度 CUDA-event profiler 已接入，并已在 2026-08-25 的双 V100 上完成 P0；生产默认关闭逐 block/stage event。SM70 Q4_0 直解量化算子也已完成真实 shard 与完整 50 层 TP gate，但目标长序列只带来约 0.9% wall-time 收益，因此生产默认保持 `H3_TP_Q4_DEQUANT=eager`。2026-08-26 已完成不改 attention 语义的 Q-only 布局优化，完整 50 层快 7.15%，现已成为启动器默认；自定义 global-attention kernel 暂停到下一轮。

作用域说明：常驻 `H3TPBackbone` 的 Q/K RMSNorm + partial RoPE 是 TP block 内部直接调用
的 SM70 路径；`H3_V100_RMS_ROPE` 只负责普通非 TP H3 fallback 的安装开关。当前服务
环境中的 `H3_V100_RMS_ROPE=pytorch` 不改变 TP backbone 的 fused 实现，也不需要为
本轮收口重启服务。

### 2026-08-25 首尾参考图 1 MP OOM 修复（已完成）

原来的 18/18 Qwen + 18/18 VAE split 在 `768×1344 / 124f / 首尾同图 / 4-step`
下会让 GPU0 在第一层 QKV/SDPA 前达到约 15.3 GiB，无法再申请 packed
activation；这不是 host mmap，也不是参考图 VAE encode 单独泄漏。当前默认改为
`H3_QWEN_SPLIT=12`、`H3_VAE_SPLIT=12`，把 Qwen tail 和 VAE decoder 更多放到
GPU1；H3 compute-heavy 50 层 TP 几何不变。

真实复测结果：`S=41798`，4 次 forward 用时约 `61.45/58.01/58.48/61.64 s`，
rank0/rank1 的 H3 peak allocated 约 `13.9/8.5 GiB`，每次 `finite=true`，视频
和 AAC 解码也通过，最终 `768×1344`、124 帧、24 fps MP4 已生成。证据是：

- `results/forward_0001_41798t_20260825-191946.json` 至 `forward_0004...`
- `results/diagnostic-i2v-first-last-same-768x1344-124f-4step-split12.json`
- `workflows/diagnostic-i2v-first-last-same-768x1344-124f-4step.json`

仍不能叠加 batch 或把 audio VAE 也改成常驻；下一步如需更大尺寸，先重新做整卡
峰值审计。

### Qwen3-VL-4B Q4_K_M 默认路线（direct owner 已完成）

本机已接入 `Qwen3VL-4B-Instruct-Q4_K_M.gguf` + 配套 FP16 mmproj，使用
header-only/8 MiB staging、resident direct-to-final-owner `4/32` MP 和 26 MB ridge projection；纯文本与
单参考图 conditioning 均 finite，payload mmap 为 0。用户已接受相对 INT8 的精度
取舍并指定 Q4 为默认。标准 GGML Q4_K_M 在 V100 上仍是运行时反量化，不等于
native INT4 GEMM；Q4_PT/native W4A16 路径继续排除。

direct-owner 修复后，embedding、vision 与前 4 层直达 GPU0，其余 32 层直达 GPU1；
conditioning 后 steady allocated 从旧 Q4 的 `3685/1431 MiB` 降为约
`1802/1893 MiB`，合计从 5116 降到 3695 MiB。相对旧 Q4 少约 1421 MiB，且比
INT8 12/24 的 4684 MiB 少约 989 MiB。图像 conditioning 与修复前逐元素一致，文本
relative RMS 约 `2.05e-8`，所以放置修复没有改变模型结果。

剩余显存项是 `[151936,2560]` token embedding 仍固定展开为约 `742 MiB` FP16；量化
row lookup 留到以后。direct-owner 旧 SDPA 路径已完整通过普通 1 MP 和首尾双参考图
1 MP，端到端分别 `271.956 s` 和 `325.233 s`，整卡峰值分别约
`12604/11336 MiB` 与 `15200/15466 MiB`。完整精度数据与后续 gate 见：

- [QWEN_VL_QUANTIZATION_RESEARCH.md](QWEN_VL_QUANTIZATION_RESEARCH.md)

### 2026-08-26 Q-only attention layout（当前默认）

H3 fused QKV 原来把 stride 为 3×sequence 的 Q/K/V view 直接交给 SM70 efficient
SDPA。只把 Q 整理成 contiguous BHSD，K/V 保留原 view，即可使用更快的现有 PyTorch
kernel；softmax、全局可见范围、FP32 稳定岛和 NCCL 顺序均未改变。

| 50 层 `S=37746` 路线 | forward | rank0 peak allocated | reserved | 输出 |
| --- | ---: | ---: | ---: | --- |
| 原始 strided | 47.442 s | 10225.7 MiB | 10336 MiB | baseline |
| compact 全 Q/K/V | 43.121 s | 10166.7 MiB | 11372 MiB | hash 相同 |
| **只 compact Q** | **44.051 s** | **9973.4 MiB** | **10336 MiB** | hash 相同 |

三条路线 SHA256 都是
`1b278b8ea38542d14d479a9cf21698618d292e9194ff4518848f14b098e498fa`。
Q-only 相对旧基线快 `7.15%`，不增加 reserve；全 Q/K/V 只再快约 0.93 s，却多复制
K/V 并把 reserve 推高约 1 GiB。因此默认固定为：

```bash
H3_TP_COMPACT_QKV=q
H3_TP_COMPACT_QKV_MIN_SEQUENCE=4096
H3_FP32_MLP_CHUNK_ROWS=2048
```

小序列不 compact；`H3_TP_COMPACT_QKV=0` 可完整回退，`all` 仅用于离线研究。
MLP chunk 改成 4096 时输出 hash 仍相同，局部计算约省 0.31 s，但连续满载引发 GPU1
热降频和 rank skew，整次 forward 反而为 49.645 s，所以继续用 2048。

常驻服务的无参考图 1 MP 四次 forward 为
`43.979/44.363/47.026/51.623 s`，均 finite、`models_reloaded=false`；后两次 GPU1
达到 82°C、busy SM clock 最低 570 MHz，rank0 collective 等待从 1.83 s 增到
8.55 s。该请求按用户要求在第 4 次 forward 后中断，没有保存 latent，不能写成完整
画质 gate。遥测和汇总在
`results/h3_1mp_no_ref_compact_q_20260826_summary.json`。Q-only + 1 MP 双参考图没有在
本轮重跑，是下一轮第一个显存/成品 gate。

## 当前基线

硬件与运行约束：

```text
Tesla V100-SXM2-16GB × 2，SM70，NVLink/NCCL
H3 Q4_0 + Turbo LoRA
Qwen3-VL-4B Q4_K_M + FP16 mmproj + mmh3-4b-ClipProj-v3.1（26 MB ridge）
模型根目录：/mnt/GALAX/minimax-h3/models
4–8 MiB bounded staging；禁止完整模型 host mmap
MemoryHigh=6500M，MemoryMax=7G，SwapMax=256M
```

最终目标结果：

| 项目 | 当前结果 |
| --- | ---: |
| 1344×768、124 帧、Turbo 4-step 采样 | 196.464 s |
| 单次 denoise forward | 47.594–48.048 s |
| 单次 forward NCCL | 1.21–1.51 s |
| decode | 60.15 s |
| GPU 峰值 | GPU0 15870 MiB / GPU1 14504 MiB |
| TP/encoder/LoRA 跨请求重载 | 没有 |
| latent/audio finite | 是 |

完整证据：

- `results/h3_tp_e2e/audit_20260824.json`
- `results/h3_tp_e2e/tp_fused_1344x768_124f_4step_seed2009.mp4`
- `docs/H3_V100_KERNEL_TP_ROUTE.md`
- `docs/H3_V100_ATTENTION_KERNEL.md`

### 历史基线：1 MP 下的 18/18 layer-MP 显存结论（已完成）

真实 `1344×768 / 124 帧 / Turbo 4-step` 组合使用 Qwen 18/18 layer-MP，连续
4 次 denoise forward 均通过：整卡峰值为 GPU0 `15870/16384 MiB`、GPU1
`14504/16384 MiB`，分别剩 `514 MiB` 和 `1880 MiB`；`oom=0`、`oom_kill=0`，
latent/音频 finite，模型没有重复加载。Qwen language block 的 MP 切分本身是均衡的，整卡
差距 `1366 MiB` 来自 GPU0 额外承担 H3 rank0、外围模块和请求 owner。

状态：**这是历史 gate，不是当前 Q4 默认分配。** 它证明 1 MP 是双 16 GiB 卡的
保守上限，但 GPU0 余量偏紧；不能在这个工作流上默认叠加 batch、
`ref_img_size=max`、常驻视频 VAE 或不受控的 compile workspace。当前 Q4
direct-owner 路线已改为 `4/32`，VAE 为 `12/24`；更大尺寸仍需重新记录两卡峰值。
严格 Qwen TP 虽能降一点局部显存/提速，但真实 hidden-state 漂移未过精度 gate，
不替代当前 layer-MP。

补充规则：Qwen 18/18 只说明 language block 的层数均分，不代表整卡显存均衡。
GPU0 还承担 H3 rank0、Comfy/conditioning owner 和部分 VAE/Qwen 前段；首尾参考图
会额外增加两块视觉 condition rows，必须单独做峰值审计。后续优先通过
`H3_QWEN_SPLIT` / `H3_VAE_SPLIT` 向 GPU1 倾斜外围 MP，保持 H3 compute-heavy TP
的几何和 collective 对称；不要未经 weighted shard/quality gate 就把 H3 TP 改成
不等分。目标是每卡留出安全余量，而不是强求 `nvidia-smi` 数字完全相等。

2026-08-25 真实 benchmark（显式使用现有 `/home/regen/minimax-h3/ComfyUI/models`；该目录与空的 `/mnt/GALAX` 同属 `/dev/sdb4`，全程 `payload_mmap=false`）：

| sequence | forward | NCCL | peak allocated/rank | finite / rank consistency |
| ---: | ---: | ---: | ---: | --- |
| 2048 | 941.10 ms | 61.17 ms | 6899.7 MiB | 是 / bitwise |
| 14880 | 9694.94 ms | 497.94 ms | 8097.1 MiB | 是 / bitwise |
| 37746 | 47441.93 ms | 1329.31 ms | 10225.7 MiB | 是 / max_abs=0 |

目标尺寸阶段 profile：global SDPA `35168.24 ms`（约 74%），QKV GEMM `2517.93 ms`，FC1 GEMM `3223.37 ms`，Q4 解码四矩阵合计约 `240.60 ms`。因此 Q4 fused GEMM 仍有价值，但不是当前第一热点。

结果文件：

- `results/h3_tp_backbone_50l_s2048_stage_profile_20260825.json`
- `results/h3_tp_backbone_50l_s37746_stage_profile_20260825.json`

### 2026-08-25 SM70 Q4_0 直解量化算子（已验证，默认 opt-in）

`custom_nodes/DualV100/h3_v100_q4_ops.py` 增加了面向 Tesla V100/SM70 的 Triton
Q4_0 kernel：每个程序直接读取 GGML 的 FP16 scale 与低/高 nibble，并写出 FP16
矩阵，跳过 eager 路径中的中间 `scales`/`quants` tensor。它不改变 Q4 shard
布局、不做 host mmap，也没有把 Q4 GEMM 猜成另一种量化格式；失败时默认回退
eager，`H3_TP_Q4_DEQUANT_STRICT=1` 才会将 kernel 错误暴露出来。

真实 block 0 shard 的 microbenchmark：

| 矩阵 | eager | Triton | 加速 | 额外显存变化 | 数值 |
| --- | ---: | ---: | ---: | ---: | --- |
| QKV `[10752,5376]` | 1.36 ms | 0.74 ms | 1.83× | 169.7 → 110.3 MiB | `max_abs=0` |
| out_proj `[5376,3584]` | 0.48 ms | 0.27 ms | 1.79× | 56.4 → 36.8 MiB | `max_abs=0` |
| FC1 `[14336,5376]` | 1.78 ms | 0.93 ms | 1.92× | 226.6 → 148.0 MiB | `max_abs=0` |
| FC2 `[5376,7168]` | 0.91 ms | 0.50 ms | 1.81× | 113.0 → 74.0 MiB | `max_abs=0` |

完整真实 TP（50 层、Q4+Turbo LoRA、双 NCCL rank）结果：

| packed sequence | eager baseline | Triton Q4 | wall-time 变化 | peak allocated | rank consistency |
| ---: | ---: | ---: | ---: | ---: | --- |
| `2048` | 944.79 ms | 820.74 ms | **13.1% faster** | 6898.7 → 6899.2 MiB | bitwise / finite |
| `14880` | 9694.94 ms | 9500.45 ms | **2.0% faster** | 8097.1 → 7959.1 MiB | bitwise / finite |
| `37746` | 47.506 s | 47.077 s | **0.9% faster** | 10225.7 → 9973.9 MiB | bitwise / finite |

目标尺寸中 global SDPA 约占 35.17 s，Q4 解量化四矩阵合计约 0.24 s/层组，
所以直解量化无法替代 attention kernel。当前启动器保守设置为：

```bash
H3_TP_Q4_DEQUANT=eager                 # 生产默认
# 离线验证：
H3_TP_Q4_DEQUANT=triton H3_TP_Q4_DEQUANT_STRICT=1
```

证据文件：`results/h3_q4_dequant_sm70_20260825.json`、
`results/h3_tp_backbone_50l_s2048_q4triton_20260825.json`、
`results/h3_tp_backbone_50l_s14880_q4triton_20260825.json`、
`results/h3_tp_backbone_50l_s37746_q4triton_20260825.json`。

## 后续开工顺序

### P0：先做目标尺寸细分 profile

状态：已完成；目标尺寸 profile 已在双 V100 采集并保存。后续候选仍需复用同一
profile 口径，不要只凭 microbenchmark 猜 kernel。

在不改变生产计算的前提下，为一个 `1344×768 / 124f / 4-step` forward 记录：

- QKV Q4 dequant、QKV GEMM、QKV LoRA；
- Q/K RMSNorm + RoPE；
- global SDPA；
- out_proj dequant/GEMM/LoRA；
- FC1 dequant/GEMM/LoRA；
- SwiGLU + safe FP16 scaling；
- FC2 dequant/GEMM/LoRA；
- 两次 FP32 NCCL；
- residual gate、allocation 和 finite check。

要求：profile 只增加 CUDA event 记录，不复制大 tensor、不保存完整 activation、不改变 dtype。先输出单次 forward JSON，再决定 P1 方向。

入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  scripts/benchmark_h3_tp_backbone.py \
  --sequence 37746 \
  --warmup 1 \
  --stage-profile \
  --output results/h3_tp_backbone_50l_s37746_stage_profile.json
```

ComfyUI 常驻服务需要临时设置 `H3_TP_STAGE_PROFILE=1`；普通生产请求保持默认关闭。输出的 `rank0_profile.stage_ms` 和 rank1 对应字段按阶段给出 count/total/mean/min/max，LoRA 与 chunk 内 GEMM 只保存标量，不保存 activation。

### P2：Q4 decode + GEMM 融合

状态：直接 Q4 解量化子算子已完成并保持 opt-in；真正的 Q4 decode+GEMM 融合仍是
高收益候选，目标尺寸当前排在 global SDPA 之后。

当前路径在 [h3_tp_backbone.py](../custom_nodes/DualV100/h3_tp_backbone.py) 中先把 Q4 shard 完整反量化成 FP16，再调用 GEMM。目标是 tiled Q4 kernel：

```text
Q4 raw nibble + FP16 scale
        ↓
tile 内解码
        ↓
FP16 Tensor Core GEMM，FP32 accumulate/output
        ↓
FP32 LoRA partial / residual
```

必须分别支持：

- QKV/FC1：output-row column parallel；
- out_proj/FC2：每行 input-column shard；
- Q4 切点严格落在 32-value block 边界；
- FC2 的逐 token 二次幂 scaling；
- Turbo LoRA 的 A/B shard 语义不变；
- row-parallel base 与 LoRA partial 仍只做一次 FP32 all-reduce。

不要把所有 50 层永久反量化成 FP16：单层临时权重已经约数百 MiB，全部缓存会突破显存边界。最终 kernel 必须直接读常驻 compressed shard，并保持 CPU staging 与 payload mmap 为 0。

验收顺序：

1. `S=128` 单矩阵数值/速度 gate；
2. `S=2048` 真实 Q4+LoRA TP block；
3. 448×256 1-step latent；
4. 832×480 4-step；
5. 1344×768 4-step wall time。

### P1：VAE decode 优化

状态：音频 BigVGAN 的 `torch.compile(mode="default", dynamic=False)` 已完成验证，但默认推理恢复 eager；compile 只显式 opt-in。

当前 1 MP decode 约 60.15 s，视频 VAE 文件约 5.21 GB，不能与采样 TP 永久同时常驻。可研究：

- video/audio VAE 分阶段 profile，确认 Conv、upsample、latent 转换的耗时占比；
- 固定目标尺寸的 VAE block `torch.compile`，优先测试 `mode=default`，不直接用 `reduce-overhead` CUDA Graph；
- channels-last、预分配输出和 frame chunk buffer；
- 只在 decode 阶段加载 VAE，不能卸载 TP shard、LoRA、4B encoder/ClipProj；
- 保持音频 NaN/Inf 检查和 MP4 编码结果一致。

2026-08-25 单张 V100、`torch 2.8.0+cu126`、SM70 的 block/full-decoder gate 已保存到
`results/h3_vae_compile_sm70.json`。测试不加载 VAE checkpoint、不做 host mmap，只用真实 ComfyUI
模块和合成有限权重；视频 decoder 使用完整 36 层 ViT3D，输入是一个 temporal/spatial chunk：

| 模块/shape | eager | compiled | speedup | compiled extra VRAM | 首次编译 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 视频 ViT3D 36 层，448×256 chunk | 228.87 ms | 198.93 ms | 1.150× | 224 MiB | 16.16 s |
| 视频 ViT3D 36 层，832×480 chunk | 1163.05 ms | 1068.14 ms | 1.089× | 769 MiB | 17.04 s |
| 视频 ViT3D 36 层，1 MP chunk | 9773.48 ms | 9477.96 ms | 1.031× | 2784 MiB | 25.58 s |
| 音频 BigVGAN，200 latent frames | 70.55 ms | 45.86 ms | 1.538× | 69 MiB | 33.00 s |
| ClipProj-like 2560→4096→5120 MLP | 0.207 ms | 0.237 ms | 0.875× | 2.4 MiB | 1.53 s |

所有通过 compile 的 block case 都 finite，视频 36 层 decoder 与 eager 的 cosine 约 `0.999995`，
但这不是完整 VAE 的端到端 decode：没有加载 5.21 GB 权重，也没有把 temporal chunk loop、tiling、
CPU output buffer、视频编码计入时间。因此不把视频 `torch.compile` 全局接入生产；1 MP 视频只快
约 3.1%，却多占约 2.7 GiB，并且每个固定 shape 都需要冷编译。

音频 BigVGAN 的 compile 已完成安全实现，但不同音频 latent `T` 在 `dynamic=False` 下可能产生
新的专门化 graph；因此默认推理不再自动 compile，避免时长变化时后台冷编译。需要固定时长、
明确接受首次编译成本时，再设置 `H3_AUDIO_VAE_COMPILE=1`。启用后仍只在 VAE 经
`load_models_gpu()` 搬到 CUDA 后 lazy compile，compiled wrapper 缓存到同一个
`MiniMaxH3AudioVAE` 实例；编译失败、probe 非有限或数值 gate 不通过时会永久回退 eager。

音频实现位于 `comfy/ldm/minimax/audio_vae.py`，部署补丁为
`patches/comfyui-minimax-h3-audio-compile.patch`。默认参数是 `mode="default"`、`dynamic=False`、
`fullgraph=False`；`max-autotune` 在 SM70 上更慢且冷启动更长，所以不进入默认路线。可用
`H3_AUDIO_VAE_COMPILE=1` 显式开启，或用 `H3_AUDIO_VAE_COMPILE_MODE=reduce-overhead` 做单独实验。
`TORCHINDUCTOR_COMPILE_THREADS=1` 和 `MAX_JOBS=1` 由安全 launcher 默认设置，编译缓存默认放在
`$INSTALL_ROOT/ComfyUI/.torchinductor-cache`，与 `/mnt/GALAX` 模型目录分开。

仍需继续做的是真实 checkpoint 的端到端 gate：首次编译时间、warm decode、峰值显存、
eager/compiled waveform 的 finite/cosine/max_abs，以及最终 MP4 音轨可播放性；这些通过前不要
宣称完整音频 VAE 按 1.538× 等比例加速。

视频 ResNet block（1.43×）、音频 AMP block（1.27–1.61×）和 AttnProjection（1.16×）只是局部
结果，不能据此宣称完整 encode/decode 同比例加速。VAE compile 必须继续记录首编译时间、warm
decode 时间、显存峰值和输出差异，不要把 VAE 的实验结果外推成 TP backbone 的结果。

补测 `mode="max-autotune"`（结果：`results/h3_vae_compile_sm70_max_autotune.json`）后，结论没有改变：

| 模块 | default compiled | max-autotune compiled | max-autotune 相对 default |
| --- | ---: | ---: | ---: |
| 视频 ViT3D 36 层，448×256 chunk | 198.93 ms | 201.73 ms | 0.986× |
| 音频 BigVGAN，200 latent frames | 45.86 ms | 50.45 ms | 0.909× |

视频 case 的第一次冷调优编译约 262.9 s（命中 autotune/code cache 后仍约 9.9 s），音频约 53.9 s。
调优日志显示 SM70 上 cuBLAS `mm/addmm` 已经快于候选 Triton GEMM；因此不在 VAE 或 H3 TP
生产路径启用 `max-autotune`。`max-autotune` 只作为以后新 kernel 的离线探索开关，不能替代
专用 attention kernel。

### P1：SM70 global attention CUDA kernel

状态：**2026-08-28 关闭该路线**。生产保持 PyTorch efficient SDPA + Q-only layout。
原因不是"还没调好"，而是已测出 efficient SDPA 在目标 shape 上已经达到甚至超过
cuBLAS 能给出的上限，详见下面的 ceiling 实测。下一轮不要再投入自定义 attention
kernel（Triton / CUTLASS / TileLang 都不要），除非改变的是算法而不是 kernel。

#### 2026-08-28 attention ceiling 实测（已完成）

`scripts/benchmark_h3_attention_ceiling.py`，单卡 cuda:0、不加载 checkpoint、峰值 < 1.5 GiB，
结果保存在 `results/h3_attention_ceiling_sm70_20260828.json`。

cuBLAS FP16 方阵 GEMM（实测 tensor core 上限）：

| shape | 时间 | TFLOPS |
| --- | --- | --- |
| `4096³` | 1.541 ms | 89.2 |
| `8192³` | 11.175 ms | 98.4 |

efficient SDPA 在真实 local-head shape `[1, 28, S, 128]` 上的效率曲线：

| S | 时间 | TFLOPS | peak |
| --- | --- | --- | --- |
| 2048 | 1.659 ms | 36.2 | 64 MiB |
| 4096 | 6.155 ms | 39.1 | 120 MiB |
| 8192 | 25.755 ms | 37.4 | 232 MiB |
| 16384 | 107.542 ms | 35.8 | 456 MiB |
| 37746 | 600.227 ms | 34.0 | 1040 MiB |

关键的一条是 QK`^T` 的纯 cuBLAS `bmm`（512 query 行对全部 key，batch 到 28 个 local head）。
它一定走 tensor core，没有 softmax、没有任何 kernel 开销，是"自己写 attention"的乐观下界：

| S | bmm TFLOPS | 外推整个 attention 的纯计算 | 物化 `S×S` scores 的 HBM 下界 |
| --- | --- | --- | --- |
| 8192 | 45.1 | 21 ms | 8 ms |
| 16384 | 38.3 | 100 ms | 33 ms |
| 37746 | **32.3** | **633 ms** | **177 ms** |

结论（这是本轮最重要的结果）：

- `S=37746` 时纯 cuBLAS bmm 只有 **32.3 TFLOPS**，比 efficient SDPA 的 **34.0 TFLOPS 还低**。
  98.4 TFLOPS 那个数只在大方阵上成立；attention 的 per-head GEMM 是瘦长的
  （QK`^T` 的 inner dim 只有 128），本来就拿不到方阵效率。所以"SDPA 只跑到峰值 1/3"
  不是 SDPA 的问题，是这个 shape 在 Volta 上的真实上限。
- 任何基于 bmm 的实现，光计算就要 **633 ms**，再加上 fused FMHA 完全不用付的
  **177 ms** scores 读写，下界约 **810 ms**，而现在的 SDPA 实测 **600 ms**。
  也就是说自己写只会更慢，这跟之前四次失败（Triton `0.112×`、TC-split `0.086×` /
  `0.013×`、chunked BMM 最好 `0.561×`）完全一致——那不是调参没调对，是方向本身不成立。
- 之前"29 TFLOPS 接近 FP16 vector 峰值 31.4，可能没用上 tensor core"的猜测是错的，
  应当作废：走 tensor core 的 cuBLAS 在同 shape 上也只有 32.3。

#### 2026-08-28 query-row 分块（已完成，不采用）

`scripts/benchmark_h3_attention_qchunk.py`，结果保存在
`results/h3_attention_qchunk_sm70_20260828.json`。softmax 按 query 行独立，所以把 query
分块、每块仍看**全部** key/value 是数学恒等且 FLOP 完全相同的，只考察 CUTLASS FMHA 的
调度差异。基线 600.932 ms：

| chunk_rows | 时间 | 加速 | peak | max_abs |
| --- | --- | --- | --- | --- |
| 2048 | 593.447 ms | 1.0126× | 1304 MiB | 0.0 |
| 4096 | 590.294 ms | **1.0180×** | 1318 MiB | 0.0 |
| 8192 | 597.120 ms | 1.0064× | 1346 MiB | 0.0 |
| 16384 | 600.754 ms | 1.0003× | 1402 MiB | 0.0 |

全部逐元素 bitwise 相同（`max_abs = 0.0`）。最好也只有 `1.018×`，按 attention 占
forward 74.3% 折算端到端只有 **`1.013×`**，却要多 286 MiB 峰值——GPU1 当前只剩约
1.8 GiB 余量。收益不值这个风险，**不进生产**。

#### 作废：所谓 "sequence parallel 2.01×"

`results/h3_tp_backbone_50l_s37746_sequence_parallel_20260825.json` 记录了
`23598.67 ms vs 47441.93 ms ≈ 2.01×`，**这个结果是错的，不要再当作候选路线**。
同一轮的 `..._s2048_sequence_parallel_compare_20260825.json` 里
`sequence_parallel_compare` 是 `cosine = 0.4838`、`relative_rms = 1.042`、
`max_abs = 115667.09`——输出和基线基本不相关。它的 `output_sha256`
（`76fc3a01…`）与生产的 `1b278b8e…` 不一致就是这个原因。

机制上的错误：当前 TP 是 head-parallel，每个 rank 只有 28/56 个 head；再按 query 行
切分之后，每个 rank 的那半 query 行只累加了自己这 28 个 head 的贡献，另外 28 个 head
的贡献直接丢了。省下的一半时间是漏算出来的，不是优化出来的。这跟上面 bitwise 正确的
query-row 分块是两件不同的事——那个每块都看全部 local head 和全部 key。

同时要修的坑：`scripts/benchmark_h3_tp_backbone.py:217` 的 `numerically_qualified`
只看 `rank_consistency.max_abs == 0.0` 和可选的 `fused_vs_eager`，**根本没读**
`sequence_parallel_compare`。两个 rank 各自算错但错得一致，`max_abs` 依然是 0.0，
于是 cosine 0.48 的结果被标成 `numerically_qualified = True`。以后新增任何对比字段，
必须同时接进这个 gate，否则等于没有 gate。该实现现在已不在树内，只剩 JSON。

现有 Triton 原型在 `S=2048` 约 `36.0 ms`，PyTorch efficient SDPA 约 `3.57 ms`，不能继续只调 Triton tile 就合入。下一版若继续做，应考虑 SM70 CUDA/CUTLASS online-softmax：

- `[B, local_heads, S, 128]`；
- non-causal、无 mask、global attention；
- FP16 Q/K/V，FP32 softmax state/accumulate；
- 不用 FP8、NVFP4、`cp.async` 或 Ampere-only 指令；
- 不物化 `S×S`；
- 保留 H3 专用开关和 PyTorch fallback。

只有在 `S=2048` 和 `S=37746` 都超过 PyTorch efficient SDPA，并通过 448/832/1344 latent gate 后，才允许接入生产。

TileLang `0.1.13` 虽声明覆盖 SM70，源码也有 `mma_sm70.h`，但目前没有证据证明其
普通 MHA 在 V100 上会生成 Tensor Core kernel；部分 SM70 transpose GEMM 测试会落到
FMA fallback。下一轮如继续，先跑 `S=2048` 官方 MHA、检查生成源码并做逐元素 gate，
不过关就停止，不先安装到生产环境。

### P3：临时 buffer 与 kernel launch

状态：中等收益、低风险；Q4 nibble shift 常量缓存和 SM70 直解量化已完成，
workspace 仍待 profile 后处理。

可做但不要先于 P0：

- 预分配 FC2 partial、NCCL buffer 和 chunk workspace；
- 减少每层 `empty_like`、`del` 后重新申请；
- 把 profile event 设为可关闭，生产默认不记录逐 block event；
- 检查 `_add_lora_` 的 chunk 边界和 dtype conversion 是否能复用 buffer；
- finite check 仍需双 rank 协调，不能为了 benchmark 直接删除。

### P4：NCCL/TP 通信

状态：暂不优先。

当前 NCCL 只占目标 forward 约 2.5–3.2%。可研究 async all-reduce，但 residual 下一层需要完整 hidden，通信与计算可重叠空间有限。任何通信优化都必须保持 FP32 reduction 和 rank 顺序，不得为了小幅收益引入 hang/NaN。

#### 2026-08-28：sequence parallel 路线作废（结论：之前的 2.01× 是错的）

不要再按 2026-08-25 那批 `*sequence_parallel*` 结果去实现。那条路线**算错了**，实现也
已经不在代码树里（`h3_tp_backbone.py` 中没有任何 `sequence_parallel` / `local_sequence`
代码），只剩 `results/` 里的 JSON，容易误导下一轮。

错在哪：当前 TP 是 **head-parallel**，rank0 持有 head 0–27 并计算**全部** S 个 query 行。
那个候选实现保留了每 rank 的 28 个 local head，同时只取 S/2 个 query 行。两个 rank
合起来只覆盖 `head 0–27 × row 0…S/2` 加 `head 28–55 × row S/2…S`；
`head 0–27` 的后半段行和 `head 28–55` 的前半段行**从来没算过**。少算一半自然快一倍，
所以 `2.0034×` 和完整 50 层的 `2.01×` 都只是漏算，不是收益。

证据链（三条 JSON 必须一起看，单看任何一条都会得出错误结论）：

- `results/h3_attention_sequence_parallel_s37746_20260825.json`：`speedup 2.0034`，
  且 `numerical_vs_baseline_local_rows` 的 `max_abs = 0.0`。这个 gate 只比较
  **本 rank 的 local 行 × local head**，那部分按构造必然 bitwise 相同，
  因此**根本无法发现漏算**。这是这次踩坑的直接原因。
- `results/h3_tp_backbone_50l_s37746_sequence_parallel_20260825.json`：`2.01×`，
  但 `output_sha256 = 76fc3a01…` 与生产的 `1b278b8e…` 不一致。
- `results/h3_tp_backbone_50l_s2048_sequence_parallel_compare_20260825.json`：这条才是
  真正的判定——`sequence_parallel_compare` 为 `max_abs 115667.09`、`rms 7316.02`、
  `relative_rms 1.042`、`cosine 0.4838`。cosine 0.48 等于输出与基线基本无关。

同时要修的 gate 缺陷：`scripts/benchmark_h3_tp_backbone.py` 的
`numerically_qualified` 只看 `rank_consistency.max_abs == 0.0` 和可选的
`fused_vs_eager`，**没有读 `sequence_parallel_compare`**，所以上面那个 cosine 0.48 的
结果仍被标成 `numerically_qualified: true`。两个 rank 做同样的错事时
`rank_consistency` 会完美通过——rank 一致性永远不能当作正确性 gate。

如果下一轮真要做 sequence/context parallel，正确形式是：每 rank 取 S/2 个 query 行但必须
覆盖**全部 56 个 head**，K/V 全量 all-gather；这跟现在的 head-parallel 权重切分冲突
（out_proj 的 row-parallel 前提就是 head 切分），代价远超收益。而且按上面 P1 的
ceiling 实测，attention 在 `S=37746` 已经贴住 Volta 的真实上限，把同样的 FLOP 换个
切法并不会变快——省时间的唯一途径是**少算**，也就是算法层（例如 residual/group cache
那条在研路线），不是并行切分层。

### P5：参考图片输入 / ref2va workflow

状态：direct-owner + 旧 strided SDPA 已完成 1 MP 首尾双参考图端到端 gate；Q-only
布局下的同规格显存/成品复测留到下一轮。

当前 ComfyUI 核心已经提供 `MiniMaxH3ReferenceToVideo`，不需要把图片伪装成 `CLIPTextEncode` 文本。该节点会把参考图同时送入两条 H3 所需的路径：

```text
LoadImage
   ├─ Qwen3-VL vision token → 4B ClipProj v3.1 ridge → [seq, 5120] conditioning
   └─ video VAE encode      → DiT reference latent（每个 denoise step 重注入）
```

当前已验证 workflow：

- `768×1344 / 124 frames / 4-step`，首尾使用同一参考图；
- Qwen3-VL-4B Q4_K_M + FP16 mmproj + `mmh3-4b-ClipProj-v3.1` ridge 以 direct-owner `4/32` MP 常驻；
- 保持 Q4 DiT + Turbo LoRA 的长期 2-way NCCL TP，不改 rank/packed layout；
- 视频/音频 VAE 在阶段内加载，最终 MP4 与 latent 均保存；
- 完整执行 `325.233 s`，整卡峰值约 `15200/15466 MiB`，无 OOM。

验收时必须保存：

- reference conditioning 的 shape、token tag 和 finite 统计；
- reference latent、最终 AV latent、MP4 与音频 finite/stats；
- VAE encode/decode 时间、GPU0/GPU1 峰值、主进程 RSS/cgroup/OOM 计数；
- 连续两次提交中 TP rank1 PID、Q4/LoRA shard 和 4B encoder 是否保持常驻；
- `/proc/<pid>/maps` 中 GGUF/safetensors payload 映射仍为 0。

下一轮只需在同一 workflow 打开 Q-only 默认后重跑，确认新增 Q copy 没有吃掉 GPU1
约 918 MiB 的旧余量，并保存成品。参考图的 `ref_image_size` 默认用 `match`；`max` 会
增加 vision token 和 VAE/采样压力。若 GPU 峰值不足，优先让 VAE 阶段 offload，不能
卸载或重载 TP/LoRA/4B ClipProj 来凑空间。

## 明确暂不做

- 不对完整 H3 TP 默认开启 `torch.compile`；`S=2048` SDPA compile 仅快约 0.29%，`S=128` 反而慢约 18%。
- 不恢复 host mmap、完整 CPU 权重副本或把 VAE 强塞进采样显存。
- 不把 `na3d`/局部窗口 attention 当成 H3 global attention 替代。
- 不使用 FP8、NVFP4、BF16 Tensor Core 或 Ampere-only kernel。
- 不用 `DataParallel`、普通 `device_map` 或组件分卡冒充 TP。
- 不为了测 kernel 并行启动第二个完整 ComfyUI/TP 服务。

## 每项优化的统一验收清单

- [ ] no-host-mmap 仍为 true，两个 rank 模型 payload maps 为 0。
- [ ] RSS/cgroup 没有异常峰值，`oom=0`、`oom_kill=0`。
- [ ] 两 rank 输出一致，所有中间与最终 latent finite。
- [ ] FP32 residual、out_proj/FC2 FP32 output、FP32 NCCL 保留。
- [ ] Turbo LoRA 不被 bypass，BF16 只做数值转换，不做同宽字节 reinterpret。
- [ ] 448×256、832×480、1344×768 结果都保存 latent、MP4、audio stats、profile 和 SHA256。
- [ ] 新路径相对当前生产基线有目标尺寸 wall-time 收益，不能只看 microbenchmark。
- [ ] 失败时回退 PyTorch SDPA/当前 Q4 path，不留下 NCCL 孤儿进程。
