# Qwen3-VL-4B Q4 当前测试结论

更新时间：2026-08-26。本文只记录已经实际运行得到的结果；完整 H3 same-seed
视频质量测试尚未完成，不把 conditioning 指标直接冒充最终画质结论。

## Compact checkpoint（当前结论）

Qwen3-VL-4B `Q4_K_M` 已在双 V100 上以 resident direct-owner `4/32` layer-MP
跑通纯文本、参考图 conditioning 和 1 MP H3 请求。Q4 direct loader 不使用 host
payload mmap：模型、独立 FP16 mmproj 和 projection 的 `/proc/<pid>/maps` 命中均为 0。

2026-08-26 用户确认后，生产默认改为：

- Qwen：`Qwen3VL-4B-Instruct-Q4_K_M.gguf`
- vision/mmproj：`mmproj-Qwen3VL-4B-Instruct-F16.gguf`
- projection：26 MB 的 `mmh3-4b-ClipProj-v3.1.safetensors`（全 ridge）
- Qwen split：direct-owner `4/32`（GPU0/GPU1）
- 模型生命周期：resident

2026-08-26 已完成相同 prompt、相同 768×768 参考图、修复前 `12/24` 下的
Q4/INT8 全 ridge 对照。全局 relative RMS 从 MLP 路线的 `42.10%` 降至
`8.47%`，百万级离群值消失，证明之前的巨大误差主要由 residual MLP 对视觉
token 的分布外放大造成；但单独看 578 个视觉 token，Q4 对 INT8 仍有
`17.94%` relative RMS、`0.98380` cosine，尚未通过严格的视觉 conditioning gate。

用户已接受上述参考图 conditioning 误差，因此 Q4 现在替换 INT8 成为默认；这是一项
明确的部署取舍，不表示视觉精度 gate 被追溯判定为通过。

之后已经实现按 tensor 名称从普通文件 I/O 直接 materialize 到最终 CUDA owner。
embedding、vision 和前 4 层归 GPU0，后 32 层归 GPU1；steady allocated 从旧 Q4 的
`3685/1431 MiB` 降到约 `1802/1893 MiB`，合计从 5116 降到 3695 MiB。相对旧 Q4
少约 1421 MiB，相对 INT8 12/24 的 4684 MiB 少约 989 MiB。参考图 conditioning
逐元素完全一致；文本 `max_abs=6.10e-5`、relative RMS `2.05e-8`，放置修复通过数值 gate。

direct-owner 路线已完整跑通无参考图 1 MP（271.956 s）和首尾双参考图 1 MP
（325.233 s），整卡峰值分别约 `12604/11336 MiB` 和 `15200/15466 MiB`。

## 修复前 12/24 精度基线与当前 direct-owner gate

测试模型：

- Qwen：`/mnt/GALAX/minimax-h3/experimental/qwen3vl_q4/Qwen3VL-4B-Instruct-Q4_K_M.gguf`
- vision/mmproj：同目录的 `mmproj-Qwen3VL-4B-Instruct-F16.gguf`
- projection：`/mnt/GALAX/minimax-h3/models/clip_projections/`
- 设备：2× Tesla V100 16 GB
- 历史对照分配：Qwen language layer-MP `12/24`（GPU0/GPU1）
- 当前生产分配：direct-owner `4/32`（GPU0/GPU1）
- 加载：resident、无 host payload mmap、8 MiB bounded staging

旧的单 owner direct-loader audit 已确认：

- 713 个 tensor 全部在 `cuda:0`
- CPU tensor：0；meta tensor：0
- 模型文件 `/proc/<pid>/maps` 命中：0
- 加载约 31.3 s
- RSS：约 569 → 1320 MiB
- CUDA allocated：约 3675 MiB
- finite：通过

该数据用于定位重复 owner，已经被 direct-to-final-owner 实现取代。当前 loader 在读取
时就按 `4/32` 选 owner，不再全量落到 GPU0 后移动 tail；Q4 文本与视觉 conditioning
均已重新跑通。

## 纯文本 Q4 对照

conditioning shape 均为 `[1, 36, 5120]`、FP32、finite，模型/projection payload mmap 均为 0。

| 路线 | Q4 + ridge | Q4 + v3.1 MLP |
|---|---:|---:|
| 模型加载 | 20.07 s | 19.71 s |
| projection 加载 | 0.038 s | 1.70 s |
| cold conditioning | 3523 ms | 4567 ms |
| warm conditioning 平均 | 197 ms | 438 ms |
| conditioning RMS | 38.31 | 35.39 |
| conditioning max abs | 16224 | 14850.8 |
| GPU allocated（conditioning 后，GPU0/GPU1） | 3622 / 1389 MiB | 3622 / 1389 MiB |
| service RAM peak | 约 5.5 GiB | 约 5.7 GiB |
| swap peak | 0 | 0 |

MLP 与 ridge 的同 prompt 输出：

- cosine：`0.99120`
- relative RMS：`0.1491`
- 两者均 finite

这里的 cosine 只是两个 projection 输出之间的差异，不能当作相对 32B 真值的精度指标。MLP 文件虽然约 503 MB，但当前 ComfyUI conditioning 在进入 projection 前已被移到 `intermediate_device()`（本机为 CPU），所以这次报告没有看到额外约 480 MiB 的 GPU allocation；MLP 的额外成本主要体现为 CPU/RSS 和 projection 计算时间。

## 参考图 / vision 路径

### 全 ridge：Q4 对 INT8（2026-08-26）

输入为 `/home/regen/minimax-h3/ComfyUI/input/example.png`（768×768 RGB），输出
均为 FP32 `[1, 620, 5120]`，token tags 完全一致：42 个 text、578 个 vision。
两条路线均 finite，模型/projection payload mmap 均为 0。

| 指标 | Q4 + ridge | INT8 + ridge |
|---|---:|---:|
| conditioning RMS | 10.4441 | 10.4677 |
| conditioning max abs | 16224 | 16224 |
| cold conditioning | 7751 ms | 11096 ms |
| warm conditioning（单次） | 752 ms | 900 ms |
| conditioning 后 GPU0 / GPU1 allocated | 3685 / 1431 MiB | 2345 / 2339 MiB |
| systemd service memory peak | 6.1 GiB | 3.4 GiB |

Q4 相对 INT8 的 FP64 累积误差如下。这里的 relative RMS 是
`RMS(Q4-INT8) / RMS(INT8)`，不是“画质下降百分比”。

| token 范围 | relative RMS | cosine | max abs delta |
|---|---:|---:|---:|
| 全部 620 token | 8.471% | 0.996406 | 808.891 |
| 42 个 text tag | 0.853% | 0.999964 | 28.256 |
| 578 个 vision tag | 17.935% | 0.983803 | 808.891 |

text-tag 汇总包含两边完全相同、norm 极大的 token-0 attention sink，因此会把整体
比例压低。排除 sink 后，图像条件中的其余 text token 为 `8.108%` relative RMS、
`0.996709` cosine；这反映视觉 token 经语言层交互后也会影响后续文本位置，不是
tokenizer 长度不一致。

### 为什么先前 MLP 是 42.1%

相同参考图使用 `mmh3-4b-ClipProj-v3.1-mlp.safetensors` 时，Q4/INT8 全局
relative RMS 为 `42.104%`、cosine 为 `0.999562`，最大单点 delta 达
`563180`；Q4/INT8 conditioning RMS 分别为 `1314.74 / 926.18`。高 cosine
来自少数同方向的百万级离群值主导点积，不代表数值更准。

切换为全 ridge 后，最大 delta 从 `563180` 降至 `808.9`，全局 relative RMS
从 `42.10%` 降至 `8.47%`。所以生产默认使用全 ridge 是正确的，当前不需要实现
“文本 MLP、视觉 ridge”混合路径。剩余的 vision-only `17.94%` 需要继续在
projection 前 hidden、vision merger、DeepStack/3D mRoPE 和完整视频 gate 中定位。

## 修复前 Q4 显存没有下降的原因（已解决）

| 路线 | GPU0 allocated | GPU1 allocated | 合计 |
|---|---:|---:|---:|
| Q4 + ridge | 3685 MiB | 1431 MiB | 5116 MiB |
| INT8 + ridge | 2345 MiB | 2339 MiB | 4684 MiB |

Q4 direct-loader audit 在 MP 前已经显示全部 713 个 tensor 位于 GPU0、allocated
约 3675 MiB；安装 `12/24` MP 后 GPU0 基本没有下降，GPU1 又增加约 1423 MiB。
这强烈指向量化 tail 的原始 CUDA0 backing/state reference 仍存活，而 GPU1 得到
第二份。`memory_allocated` 不包含 allocator 的空闲 reserved cache，因此不能仅用
“缓存没清”解释。

该表是修复前证据。当前已经在读取 GGUF 时按最终 owner materialize：embedding、
vision、0–3 层直达 GPU0，4–35 层和 final norm 直达 GPU1，禁止“全放 GPU0 再
`.to(cuda:1)`”。修复后的 steady allocated 为 `1802/1893 MiB`，合计 3695 MiB。

Q4 仍不会相对 INT8 整体减半：独立 FP16 mmproj 约 797 MiB，
`token_embd.weight` 当前会展开为约 742 MiB FP16，且 Q4_K_M 仍有 scale 和
FP16/F32 tensor。量化 row lookup 仍可继续减少 GPU0 压力，但不是当前阻塞项。

## 期间发现并修正的测试问题

第一次 Q4 + MLP benchmark 失败的原因不是 Q4 权重、不是 NaN，也不是显存 OOM，而是 benchmark 外层使用了 `torch.inference_mode()`。v3.1 residual MLP 的普通 `nn.Linear` 在当前 PyTorch 版本收到 inference tensor 后报：

```text
RuntimeError: Inference tensors do not track version counter.
```

benchmark 已改为 `torch.no_grad()`，与生产 ClipProj 的调用方式一致；随后 Q4 + MLP 纯文本和参考图测试均成功。这个改动只在 benchmark harness 中，不代表修改了 Q4 数值路径。

所有 Q4 测试都在独立的 systemd user service 中运行，并设置 RAM 上限；测试结束后 GPU 已释放，没有再把进程放进 VSCode cgroup，也没有再次触发 VSCode 被 oomd 杀掉。

第一次全 ridge 参考图复测使用 `MemoryHigh=6G` 时，被 systemd-oomd 在受限测试
cgroup 内终止（GPU 已释放，VSCode 未受影响）。保持 `MemoryMax=7G` 不变，仅把
soft high 调至 `6500M` 后复测成功，service memory peak 6.1 GiB、swap 0。Q4
loader 的 host 峰值仍偏高，需要和 GPU backing 重复问题一起优化。

## 还不能下结论的部分

1. Q4 direct-owner 的完整 H3/参考图请求已经成功，但还没有完成相对 INT8 的同 seed
   视频、音频和画质对照。
2. conditioning 对照只使用 INT8 作为历史精度基线，不是相对原版 32B encoder 的真值评估。
3. 需要保存 projection 前 `[1,620,2560]` hidden 并按 text/vision tag 比较，确认剩余 `17.94%` 从 vision/mmproj、Q4 language layer 的哪一阶段开始。
4. Q4 已按用户决定进入生产默认，但 Q4-vs-INT8 视觉成品 gate 仍未完成；不得把
   “默认”写成“与 INT8/32B 数值等价”。

## 结果文件

- [`h3_qwen_q4_direct_loader_audit_20260825.json`](../results/h3_qwen_q4_direct_loader_audit_20260825.json)
- [`h3_qwen_q4_mp12_conditioning_20260825.json`](../results/h3_qwen_q4_mp12_conditioning_20260825.json)
- [`h3_qwen_q4_ridge_mp12_conditioning_20260825.json`](../results/h3_qwen_q4_ridge_mp12_conditioning_20260825.json)
- [`h3_qwen_q4_mlp_mp12_conditioning_20260825.json`](../results/h3_qwen_q4_mlp_mp12_conditioning_20260825.json)
- [`h3_qwen_q4_mlp_mp12_image_conditioning_20260825.json`](../results/h3_qwen_q4_mlp_mp12_image_conditioning_20260825.json)
- [`h3_qwen_q4_ridge_mp12_image_conditioning_20260826.json`](../results/h3_qwen_q4_ridge_mp12_image_conditioning_20260826.json)
- [`h3_qwen_int8_ridge_mp12_image_conditioning_20260826.json`](../results/h3_qwen_int8_ridge_mp12_image_conditioning_20260826.json)
- [`h3_qwen_q4_vs_int8_ridge_image_20260826.json`](../results/h3_qwen_q4_vs_int8_ridge_image_20260826.json)
- [`h3_qwen_q4_ridge_direct_owner_mp4_text_20260826.json`](../results/h3_qwen_q4_ridge_direct_owner_mp4_text_20260826.json)
- [`h3_qwen_q4_ridge_direct_owner_mp4_image_conditioning_20260826.json`](../results/h3_qwen_q4_ridge_direct_owner_mp4_image_conditioning_20260826.json)
- [`q4_direct_owner_1mp_no_ref_submit_20260826.json`](../results/q4_direct_owner_1mp_no_ref_submit_20260826.json)
- [`q4_direct_owner_1mp_two_ref_submit_20260826.json`](../results/q4_direct_owner_1mp_two_ref_submit_20260826.json)
