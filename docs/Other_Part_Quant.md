有，而且我看下来你这个项目现在已经不是“能不能跑”的阶段了，而是很适合开始做 **第二轮：把常驻显存和每层额外 GEMM 榨掉**。

先给结论：**我不会优先把 DiT 再降 bit。** 你现在的 `minimax_h3_fl2va_pruned_fp8_Q4_0.gguf` 本来就是 **Q4_0 DiT**。你下一步最值得试的顺序，我会排成：

1. **ClipProj `-mlp` → 26 MB ridge 版**
2. **Turbo LoRA 离线 merge 进 DiT，再重新 Q4_0**
3. **Qwen 4B INT8 → Q5/Q4，主要为了省 2GB+ 常驻显存**
4. **试 Kijai 的 INT8 ConvRot Video VAE**
5. 最后才是 **Q4_0 DiT → Q4_K / Q5**
6. 真正想继续提速，还是要狠狠干 **SM70 attention / Q4 GEMM kernel**

### 先说最白捡的：你的 ClipProj 其实可以少掉快 500 MB

你现在：

```text
qwen3vl_4b_int8_convrot      ~4.86 GB
mmh3-4b-ClipProj-v3.1-mlp   ~503 MB
```

但作者现在明确建议先用：

```text
mmh3-4b-ClipProj-v3.1
```

只有 **26.3 MB**。作者自己的 v3.1 benchmark 里，4B vs 8B、ridge vs residual MLP 在他们的测量上都没有拉开，甚至明确写了“start with 4B ridge”。([GitHub][1])

也就是说你可以：

```text
503 MB
↓
26 MB
```

直接释放大约 **477 MB**。

这对于你这种 16 GB ×2 卡已经蹭着峰值跑 1 MP 的环境非常值，而且基本没有工程成本。

---

## Qwen 4B：我确实建议你试 Q4/Q5

你目前 4B INT8 ConvRot 是 **4.86 GB**。([Hugging Face][2])

Qwen 官方现在就有 Qwen3-VL-4B-Instruct 的 GGUF：

```text
F16      8.05 GB
Q8_0     4.28 GB
Q4_K_M   2.50 GB
```

也就是说换 Q4_K_M 可以：

```text
4.86 GB
↓
2.50 GB

省约 2.36 GB
```

([Hugging Face][3])

而 ClipProj 的设计本身并不要求某一个特定量化 checkpoint。作者写得很清楚：只要是对应宽度的 Qwen3-VL-4B，量化/finetune variant 都可以用，4B hidden width 是 2560。([GitHub][4])

甚至已经有另一个 ComfyUI GGUF loader 做了 **GGUF Text Encoder + ClipProj**，说明技术路径完全走得通。([GitHub][5])

但对你这里有一个重要区别：

> **Qwen 换 Q4 的主要价值是显存，不是生成速度。**

因为 Qwen 只负责 prompt → conditioning。你现在又已经常驻，不会每个 denoise step 重载。

真正占你 1 MP 时间的是：

```text
conditioning       一次
       ↓
DiT forward × 4    ← 这里四次 × ~60 s
       ↓
VAE decode
```

所以 Qwen 从 INT8 → Q4，可能让 encoder 本身快/慢一些，但不可能把你的 `196s~240s` denoise 大幅砍掉。

### 而且我更建议先做 Q5，再做 Q4

原因就是你已经很在意 numerical gate。

你之前 strict TP：

```text
S=256   +19.7% speed
但 hidden drift 2.12%

S=512   +36.8% speed
但 hidden drift 2.56%
```

你都因为精度没过 gate 放弃了。

那 Q4 权重量化造成的 conditioning 漂移同样应该测。

所以我会做：

```text
当前 INT8 ConvRot
    ↓ baseline

Q5_K_M
    ↓
relative L2
cosine
per-token max error
conditioning 最终 [seq,5120] error
相同 seed 成片

再测 Q4_K_M
```

Qwen 官方仓库目前直接提供的是 Q4_K_M/Q8/F16，没有现成 Q5；Q5 可以从 F16 GGUF 自己 quantize。([Hugging Face][6])

对你而言，我觉得最终很可能是：

```text
Q5  ≈ 最舒服的 production
Q4  ≈ 极限省显存
INT8 ≈ 最稳 baseline
```

当然最后得看你的 conditioning gate。

---

# DiT：你已经 Q4 了

这里要纠正一下你问题里的“DiT 主干换 Q4”：

你 README 现在就是：

```text
minimax_h3_fl2va_pruned_fp8_Q4_0.gguf
                                ^^^^
```

而且公开文件也是 **11.4 GB 的 Q4_0**。([Hugging Face][7])

所以 DiT 现在真正可以比较的是：

```text
Q4_0      ← 你现在
Q4_K
Q4_K_M
Q5_0
Q6_K
Q8
```

而不是“换 Q4”。

目前社区有例如：

```text
pruned Q4_K     10.64 GiB
pruned Q5_0     12.97 GiB
pruned Q6_K     15.45 GiB
pruned Q8_0     19.97 GiB
```

([Hugging Face][8])

另一个仓库也有 Q4_K_M 的 pruned H3，大约 11.4 GB。([Hugging Face][9])

### 我反而不建议你现在上 DiT Q5

因为你现在：

```text
Q4 DiT shard  ≈ 6.1 GiB / GPU
Qwen resident
ClipProj resident
LoRA resident
attention workspace
latent
NCCL buffers
...
```

已经明显是**显存踩线**。

Q5 总 checkpoint 大约比 Q4 多：

```text
13.9 - 11.4 ≈ 2.5 GB
```

2-way TP 就是每张卡大概再多：

```text
~1.25 GB/GPU
```

这个对你的 1344×768 / 124f 太贵了。

而且 Q5 大概率：

* 权重读取量增加
* 解量化复杂一点
* 显存余量更差

得到的是**质量提升**，不是性能优化。

所以：

> **Q5 DiT 是 quality mode，不是 V100 production fast mode。**

你的 Q4_0 其实挺合理。

---

# 倒是 Q4_0 → Q4_K 值得研究

这个比 Q5 有意思。

因为 K-quant 通常可以在相似空间里给更好的量化质量。H3 社区已经有 pruned Q4_K / Q4_K_M 文件。([Hugging Face][9])

但你的问题不是“文件能不能加载”，而是你的：

```text
DualV100 custom shard loader
+
TP quant Linear
+
LoRA
+
SM70 kernel
```

得支持 K block layout。

而 Q4_K 的 block metadata / scale layout 比简单 Q4_0 麻烦。

所以在 V100 上可能出现：

```text
Q4_K
质量 ↑
显存 ≈
但是 dequant kernel 更复杂
      ↓
速度反而 ↓
```

因此如果你的目标第一是 **速度**：

> **我会继续留 Q4_0。**

如果你目标是“Q4 大小下尽量提画质”，再实现 Q4_K。

---

# 但我认为你最值得干的其实是：把 Turbo LoRA merge 掉

这个甚至比 Qwen Q4 更让我感兴趣。

你现在 Turbo LoRA：

```text
~0.78 GB
```

并且 sampling 时 LoRA 不是纯“占显存”。

典型 linear：

```text
Y = XW + scale * X A B
```

所以除了原本的 Q4 linear，你每层还得额外做：

```text
X @ A
↓
@ B
↓
加回 base
```

H3 50 层，attention/MLP 很多 projection。

**你 Turbo LoRA 又是固定生产配置。**

那最理想的路线其实是：

```text
原始较高精度 DiT
        +
Turbo LoRA
        ↓
offline merge
        ↓
Wmerged = W + ΔW
        ↓
重新 quantize 成 Q4_0
        ↓
新的 Turbo-Q4 checkpoint
```

然后生产时直接：

```text
Q4 linear
```

不再有：

```text
Q4 linear
+
LoRA A/B GEMM
```

这会同时得到：

```text
LoRA ~0.78 GB 常驻        → 没了
LoRA runtime GEMM         → 没了
LoRA shard/load 逻辑       → 简单很多
rank0/rank1 LoRA 通信逻辑  → 简化
```

唯一的问题是：

> **LoRA merge 后再 Q4 会把一部分 LoRA delta 量化掉。**

所以必须做 quality gate。

但你这套是 **Turbo 4-step 固定 LoRA**，我非常认为值得做。

甚至可以做两版：

```text
H3-Q4_0-base + runtime LoRA
             ↓ 对照

H3-TurboMerged-Q4_0
```

同 seed、同 conditioning、同 latent，比：

```text
每 step latent relative error
最终 RGB PSNR / SSIM
audio
视觉主观
```

如果过 gate，这可能是你目前**最高价值的模型结构优化之一**。

## LightX v1.1 runtime LoRA：已下载，先做 A/B 再 merge

2026-08-25 已从
[`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
下载 ComfyUI runtime 文件：

```text
/mnt/GALAX/minimax-h3/experimental/lightx_v1_1/
  minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors
```

文件大小为 `1,956,192,992` bytes，SHA256 为
`449d80f301ac571622c72e28b8fd72a4b3681b7a8df8a92f17c8f6ec43f56558`；生产
`models/loras/` 下只有一个指向实验文件的软链接，未复制或替换现有 v4 LoRA。
文件 header 显示：50 个 H3 block、2 个 token-refiner block、rank 128、BF16
因子，且没有旧 v4 那种 block AdaLN LoRA。

这几个差异明天对照时必须保留：

1. 先在同一 conditioning、seed、latent 下比较现有
   `minimax_h3_turbo_v4_step600_ema.safetensors` 与 LightX v1.1；先看 finite、
   每步 latent、视频/音频，再谈速度和显存。
2. runtime reader 已补上 LightX 兼容：自动去掉
   `diffusion_model.` 前缀、过滤 `.alpha` 伪模块，并把 AdaLN 设为 optional；
   416 个 BF16 A/B 因子仍通过 NoHostMMap 保持 disk-backed，只有 208 个 alpha
   标量有界读到 CPU。普通（非 persistent TP）路径已实测通过；persistent TP
   还要做单独的双卡 worker 对照，不能仅凭普通路径通过就宣布 merge 正确。
3. merge 的首选输入应是较高精度的 `minimax_h3_fl2va_bf16.safetensors`，然后
   `W_merged = W + scale·B@A`，最后再重新量化成独立的实验 Q4。当前只有 Q4_0
   DiT 时，只能逐 tensor 做 bounded dequant → LoRA delta → Q4_0 requant；这会把
   LoRA delta 再量化一次，不能覆盖原始 Q4，也不能在 14 GiB 主机上先把整个模型
   展开到 RAM。
4. 输出必须写到 `/mnt/GALAX/minimax-h3/experimental/merged/`，使用临时文件和
   完成后原子 rename；先做单 block/小 tensor dry-run，再做全模型，保留每个 tensor
   的 dtype、shape、scale 和校验报告。

明天的 merge quality gate：runtime v1.1 作为 reference，比较 merged-BF16、
merged-Q4_0、runtime-Q4_0 三条路线的每步 latent relative error、最终 RGB
PSNR/SSIM、音频 finite/波形误差、首尾参考图几何；任何一项不合格都不替换默认
runtime LoRA。

2026-08-25 已完成一个小规模 runtime smoke：

```text
workflow: workflows/lightx-v11-clipproj-smoke-448x256-1step.json
result:   results/lightx-v11-clipproj-smoke-448x256-1step-20260825.json
time:     94.83 s（ComfyUI wall time）
status:   success；finite；无 OOM
loader:   624 raw keys -> 624 canonical，208 bypass adapters
canary:   qkv_proj.forward_owner=BypassForwardHook（确认 LoRA 实际生效）
output:   /home/regen/minimax-h3/ComfyUI/output/lightx_v1_1_clipproj_smoke_448x256_latent.pt
```

该次进程的 `/proc/<pid>/maps` 没有 v1.1 LoRA 文件映射；GPU 结束时约为
`cuda:0 14264 MiB / cuda:1 2806 MiB`。这个显存数字包含当前 resident
encoder、Q4 DiT 和 runtime 状态，不能直接当作 v1.1 相对 v4 的增量；明天应在
同一进程、同一 seed 和同一 conditioning 下重跑两种 LoRA。

随后又用 `workflows/lightx-v11-clipproj-tp-smoke-448x256-1step.json` 做了
persistent 2-way NCCL TP smoke：84.22 s，总体成功；worker 加载 v1.1 的 50 个
block，单次 `S=870` forward 为 7.005 s，NCCL 为 0.038 s，峰值约
`cuda:0 9564 / cuda:1 7072 MiB`，结束时约 `10460 / 10696 MiB`，显存已明显
均衡。第一次 TP 尝试触发了旧 DynamicVRAM backup 的 stale-path bug，现已在 TP
替换 block tree 前显式 restore/unload；该错误不再复现。TP worker 当前按 resident
设计保持运行，停止服务才会释放它。

---

# VAE：有，而且我刚查到现在已经有很有意思的版本了

这部分比你 README 里写的“只有 FP16 VAE”更新了一步。

官方 Comfy-Org 目前主线确实还是：

```text
video VAE FP16   5.21 GB
audio VAE FP32   605 MB
```

([Hugging Face][10])

但是 Kijai 的 experimental 仓库已经放出了：

```text
minimax_h3_video_vae_int8_convrot.safetensors
3.17 GB
```

([Hugging Face][11])

也就是：

```text
5.21 GB
 ↓
3.17 GB

省约 2.04 GB
```

这对你特别有诱惑力。

更关键的是 Kijai 自己在讨论里说：

> INT8 ConvRot VAE 效果不错，在他的测试中约 **1.5× faster**，但要求较新的 ComfyUI 支持。([Hugging Face][12])

### 但是！

我不会直接把“1.5×”套到你的 V100。

因为 V100 Tensor Core 原生擅长的是：

```text
FP16 × FP16
→ FP32 accumulate
```

Volta 没有后来 Turing/Ampere 那种 INT8 Tensor Core。NVIDIA 自己的架构对比里，V100 的 INT8 Tensor TOPS 是空的，而 A100 等后续架构才有。([NVIDIA Images][13])

所以在你的 SM70：

```text
INT8 ConvRot weight
        ↓
旋转/解量化
        ↓
FP16 compute
```

最终是否比纯 FP16 VAE 快，非常看 kernel。

### 但仅仅为了显存，它都值得试

因为你 VAE 最尴尬的就是：

```text
5.21 GB
```

没法和其它东西舒服地同时放。

换成：

```text
3.17 GB
```

一下少 **2 GB**。

再配合前面的：

```text
ClipProj MLP → ridge     -0.48 GB
Qwen INT8 → Q4          -2.36 GB
VAE FP16 → INT8         -2.04 GB
```

理论总权重常驻空间一下能省：

## **~4.9 GB**

这就已经不是小修小补了。

甚至可能把你的 architecture 从：

```text
denoise
 ↓
fault/load VAE
 ↓
decode
```

推进到：

```text
TP + Qwen + VAE
更接近长期可达/部分常驻
```

当然 decode activation peak 还在，所以不能只按文件体积算“肯定全常驻”。

---

## FP8 VAE 我反而不建议

确实还有社区 `fp8mix` VAE，大约 2.6 GiB。官方 MiniMax integration 列表也收录了。([GitHub][14])

但是 Kijai 对这个很直接：

> FP8 VAE 不是好主意；那个旧 fp8mix 也已经被作者自己称为 outdated。([Hugging Face][12])

而且 V100 又没有 FP8 硬件。

所以：

```text
FP8 VAE
```

对 V100 基本属于：

```text
存储省了
+
执行还得转
+
质量风险更大
```

我会直接跳过。

---

# Audio VAE 倒是还有 300 MB 可以捡

社区已经有：

```text
audio VAE BF16
303 MB
```

对比你现在：

```text
FP32
605 MB
```

([Hugging Face][15])

但 V100 不原生支持 BF16 Tensor Core，所以我不会拿 BF16 算。

可以考虑：

```text
BF16 storage
↓ load/dequant
FP32 compute
```

纯粹省约 300 MB resident/storage。

不过相比 video VAE 的 2 GB，这个优先级低很多。

---

# 所以我会把你的下一版路线改成这样

| 改动                            |            省显存 |    速度潜力 | 质量风险 | 我建议        |
| ----------------------------- | -------------: | ------: | ---: | ---------- |
| ClipProj MLP → ridge          |   **~0.48 GB** |       小 |   很低 | **马上做**    |
| Turbo LoRA merge → requant Q4 |   **~0.78 GB** | **中~大** |    中 | **强烈建议实验** |
| Qwen INT8 → Q5                |        ~1–2 GB |       小 |  低~中 | **建议**     |
| Qwen INT8 → Q4                |   **~2.36 GB** |       小 |    中 | **建议 A/B** |
| Video VAE FP16 → INT8 ConvRot |   **~2.04 GB** |     可能中 |    中 | **很值得测**   |
| Audio VAE FP32 → BF16 storage |       ~0.30 GB |     几乎无 |    低 | 次要         |
| DiT Q4_0 → Q4_K               |             ≈0 |     不确定 |   更低 | 质量向实验      |
| DiT Q4_0 → Q5                 | **反而 +2.5 GB** |    可能更慢 |   更低 | 不适合当前生产    |
| DiT Q4 → Q3/Q2                |            省很多 |     未必快 |    高 | 不推荐        |

---

## 如果是我现在接着搞你这个仓库，我第一轮会直接做这 4 个 gate

```text
A. 当前 baseline
   Q4_0 DiT
   runtime Turbo LoRA
   Qwen4B INT8
   ClipProj MLP
   FP16 VAE

B. ridge
   只换 26MB ClipProj

C. Qwen-Q4
   ridge
   Qwen4B Q4_K_M

D. VAE-INT8
   ridge
   Qwen4B Q4_K_M
   video VAE INT8 ConvRot
```

然后再单独搞：

```text
E. TurboMerged-Q4
```

而 **E 我觉得最有可能真正降低 4-step sampling 时间**。

Qwen/QVAE 这些很多是在帮你把系统“腾空”；**LoRA merge 和 SM70 attention/Q4 fused GEMM 才是在直接砍那四个 60 秒 forward。**

尤其你这个项目现在 NCCL 才 3% 左右，通信已经优化得够好了。下一刀该砍的是：

```text
Q4 dequant
  +
FP16 Tensor Core GEMM
  +
LoRA
```

尽量融合成一个 weight path，以及继续做 **SM70 global attention kernel**。V100 的 Tensor Core 原生甜点就是 FP16 输入 + FP32 accumulate，这与你现在保留 FP32 stability island 的设计其实非常匹配。([NVIDIA Images][13])

**我最看好的最终形态大概是：**

```text
DiT:
Turbo-merged Q4_0
2-way TP
FP16 TC / FP32 accumulate

Encoder:
Qwen3-VL-4B Q4/Q5
+ 26 MB ClipProj ridge
layer MP

VAE:
INT8 ConvRot weight
FP16 compute
12/24 MP

Attention:
SM70 dedicated fused global attention
```

这样才是真正针对 **2×V100 16GB** 定制的 H3，而不是简单把现代卡上的低精度格式硬搬过来。

[1]: https://github.com/nicolab28/ComfyUI-ClipProj "GitHub - nicolab28/ComfyUI-ClipProj: Swap a large text encoder for a small one with a learned linear projection. MiniMax H3 conditioning from 15.7 GB down to 5.2 GB. · GitHub"
[2]: https://huggingface.co/Merserk/qwen3vl-4b-int8-convrot/blob/main/qwen3vl_4b_int8_convrot.safetensors?utm_source=chatgpt.com "qwen3vl_4b_int8_convrot.safetensors · Merserk/qwen3vl-4b-int8-convrot at main"
[3]: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/blob/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf?utm_source=chatgpt.com "Qwen3VL-4B-Instruct-Q4_K_M.gguf · Qwen/Qwen3-VL-4B-Instruct-GGUF at main"
[4]: https://github.com/nicolab28/ComfyUI-ClipProj/blob/main/README.md?utm_source=chatgpt.com "ComfyUI-ClipProj/README.md at main · nicolab28/ComfyUI-ClipProj · GitHub"
[5]: https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader?utm_source=chatgpt.com "GitHub - ChrisColeTech/ComfyUI-GGUF-Loader: GGUF Quantization support for native ComfyUI models · GitHub"
[6]: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/main?utm_source=chatgpt.com "Qwen/Qwen3-VL-4B-Instruct-GGUF at main"
[7]: https://huggingface.co/molbal/MiniMax-H3-GGUF/tree/main?utm_source=chatgpt.com "molbal/MiniMax-H3-GGUF at main"
[8]: https://huggingface.co/unsloth/MiniMax-H3-GGUF?utm_source=chatgpt.com "unsloth/MiniMax-H3-GGUF · Hugging Face"
[9]: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main?utm_source=chatgpt.com "leejet/MiniMax-H3-GGUF at main"
[10]: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae?utm_source=chatgpt.com "Comfy-Org/MiniMax-H3 at main"
[11]: https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main?utm_source=chatgpt.com "Kijai/MiniMax-H3-experimental at main"
[12]: https://huggingface.co/Kijai/MiniMax-H3-experimental/discussions/10?utm_source=chatgpt.com "Kijai/MiniMax-H3-experimental · audio_vae_bf16 + video_vae_fp8"
[13]: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf?trk=public_post_comment-text&utm_source=chatgpt.com "NVIDIA TESLA V100 GPU"
[14]: https://github.com/MiniMax-AI/awesome-minimax-h3-integration?utm_source=chatgpt.com "GitHub - MiniMax-AI/awesome-minimax-h3-integration · GitHub"
[15]: https://huggingface.co/dummy9996/minimax_h3_audio_vae_bf16/tree/main?utm_source=chatgpt.com "dummy9996/minimax_h3_audio_vae_bf16 at main"

---

## 2026-08-26：V100 INT8 VAE 实测更新

INT8 ConvRot 已完成 no-host-mmap、12/24 MP、SM70 W8A16 和 spatial tile
batch=2 实测。1MP/124f 为 `28.64 s`，峰值 allocated 为
`1442/2258 MiB`，相对 FP16 reference 的 relative RMSE 为 `0.0938%`。
完整数据、部署开关和回退方式见
[`VAE_INT8_TEST_RESULTS_20260826.md`](VAE_INT8_TEST_RESULTS_20260826.md)。
