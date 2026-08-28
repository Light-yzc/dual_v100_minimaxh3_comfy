如果你的目标是做一颗 **`LightX2V v1.1 + pruned + merged + Q4_0`**，那 base 我会直接选：

```text
Comfy-Org/MiniMax-H3
diffusion_models/
minimax_h3_fl2va_pruned_bf16.safetensors
```

**40.2 GB，BF16，pruned FL2VA。** 这是目前最干净的源。([Hugging Face][1])

然后搭配：

```text
lightx2v/Minimax-h3-Turbo

minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors
```

这个是 v1.1 的 ComfyUI BF16 LoRA，约 1.96 GB。([Hugging Face][2])

最终：

```text
pruned BF16 base
       +
LightX v1.1 BF16 LoRA
       ↓ FP32 merge
merged pruned BF16
       ↓
     Q4_0
       ↓
Turbo-v1.1-pruned-Q4_0.gguf
```

## 为什么一定优先 BF16 base

因为你最后本来就要做一次：

[
W_{Q4} = Q_4(W_{\text{base}}+\Delta W_{\text{LoRA}})
]

最理想是：

```text
BF16 base
   ↓ 转 FP32
+ BF16 LoRA
   ↓
FP32 merge
   ↓
一次 Q4 quantize
```

而不是：

```text
FP8 base
↓ 解量化
+ LoRA
↓
Q4
```

更不要：

```text
现有 Q4
↓ dequant
+ LoRA
↓
再 Q4
```

后两者都是把**已经损失过一次精度的 base** 再拿来量化。

可以简单排：

| Base                | 用来最终做 Q4 |       推荐度 |
| ------------------- | -------: | --------: |
| **pruned BF16**     |        ✅ | **★★★★★** |
| pruned FP8 scaled   |       可以 |       ★★★ |
| pruned INT8 ConvRot |      不建议 |        ★★ |
| 现有 GGUF Q4_0        |    能做但最差 |         ★ |

Comfy-Org 正好同时提供 `pruned_bf16`、`pruned_fp8_scaled` 和 `pruned_int8_convrot`；BF16 pruned 是 40.2 GB，另外两个约 21 GB。([Hugging Face][3])

---

## 而且别用 66.3 GB 那个 full BF16

Comfy-Org 还有：

```text
minimax_h3_fl2va_bf16.safetensors
66.3 GB
```

这是**未 pruned 的完整 33B 结构**。([Hugging Face][4])

你没必要拿它。

因为你的目标生产模型本来就是 pruned：

```text
20.1B pruned
→ Q4 ≈ 11.4 GB
```

所以直接从：

```text
40.2 GB pruned BF16
```

开始最省事。

否则你走：

```text
66.3 GB full BF16
+ LoRA
↓ merge
再执行 full → pruned 转换
↓
Q4
```

平白多一道很复杂的 AdaLN curve folding。

---

## LightX v1.1 这里比 Larry 还省事一点

这一点挺关键。

社区对 LightX v1.1 ComfyUI 文件检查后发现，它已经是正确的 ComfyUI namespace/QKV layout，而且：

> **不需要额外做 AdaLN pruning conversion。**

它有 208 个 LoRA adapter pair，主要对应 attention、MLP、token-refiner；没有 Larry 那种 full-model AdaLN LoRA → pruned AdaLN 时麻烦的维度转换问题。([Hugging Face][5])

所以你可以基本理解成：

```text
minimax_h3_fl2va_pruned_bf16
            +
LightX-v1.1-comfyui-bf16
            ↓
直接 fuse 对应 Linear
```

这比 Larry v4 那套 merge 省心得多。

---

### 还有一个你千万别搞错

选：

```text
FL2VA
```

别选：

```text
Ref2VA
```

LightX 官方明确说这个 Turbo 是在 **FL2V/T2V base** 上蒸馏的。([Hugging Face][6])

所以你的正确组合就是：

```text
BASE
Comfy-Org/MiniMax-H3
└─ minimax_h3_fl2va_pruned_bf16.safetensors
   40.2 GB

LORA
lightx2v/Minimax-h3-Turbo
└─ minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors
   1.96 GB

MERGE
FP32 accumulation

OUTPUT
pruned merged checkpoint

QUANT
Q4_0 GGUF
```

**如果你只记一句：做最终 Q4，永远尽量从 `pruned BF16` merge，而不是从你现在的 Q4 或 FP8 开始。**

你这机器只有 14GB RAM 也不妨碍做，只是 merge/quantizer 要像你当前 loader 一样做 **逐 tensor streaming**，不能把 40GB BF16 一把塞进内存。

[1]: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors?utm_source=chatgpt.com "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors · Comfy-Org/MiniMax-H3 at main"
[2]: https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main?utm_source=chatgpt.com "lightx2v/Minimax-h3-Turbo at main"
[3]: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models?utm_source=chatgpt.com "Comfy-Org/MiniMax-H3 at main"
[4]: https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors?utm_source=chatgpt.com "diffusion_models/minimax_h3_fl2va_bf16.safetensors · Comfy-Org/MiniMax-H3 at main"
[5]: https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blame/main/README.md?utm_source=chatgpt.com "README.md · drbaph/MiniMax-H3-Turbo-Lora-ComfyUI at main"
[6]: https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/1?utm_source=chatgpt.com "lightx2v/Minimax-h3-Turbo · does this works with pruned int 8 Ref2va model too?"
