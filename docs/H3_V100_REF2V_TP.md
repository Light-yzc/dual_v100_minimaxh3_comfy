# H3 双 V100 TP 参考图生视频迁移记录

日期：2026-08-27

## 结论

参考图生视频已经接入本项目的双 V100 持久 NCCL TP 路线。工作流不再使用官方
`MiniMaxH3ReferenceToVideo` 的 V3 Autogrow 动态槽，而使用项目节点
`MiniMaxH3ReferenceToVideoTP` 的一个固定 `IMAGE` 输入。

已通过安全 smoke test：

- 448×256，22 帧，1 step；
- Qwen3-VL-4B INT8 ConvRot + `mmh3-4b-ClipProj-v3.1.safetensors`（无 MLP）；
- H3 Q4 + Turbo LoRA；
- Qwen 36 层 `12/24` MP，H3 50 层双卡 NCCL TP；
- 视频 VAE 仍按原方案跨两卡常驻；
- rank1 worker 正常启动并保持常驻；
- `S=1145`，TP forward `5.638s`，NCCL `0.057s`；
- smoke 总耗时 `88.152s`（包含首次加载）；
- 结果 `/home/regen/minimax-h3/ComfyUI/output/benchmarks/h3_tp_e2e/ref2v_tp_smoke_448x256_seed20260827_latent.pt`；
- AV latent：video `(1,24,7,16,28)`、audio `(1,32,2,37)`，均 finite；
- 运行后显存约 GPU0/GPU1 = `11678/13192 MiB`，未发生 OOM。

## 原错误

官方节点定义的是：

```python
io.Autogrow.Input(... prefix="ref_image_", ...)
execute(..., ref_images=None, ...)
```

正常的 V3 UI 工作流会携带动态输入元数据，把 `ref_image_0` 归一化为
`ref_images={...}`。手工/API JSON 没有这段元数据时，ComfyUI 会直接调用：

```python
MiniMaxH3ReferenceToVideo.execute(ref_image_0=image, ...)
```

于是节点在进入 VAE、Qwen 或 TP 之前就报：

```text
unexpected keyword argument 'ref_image_0'
```

这不是显存问题，也不是 NCCL worker 的问题。

## 项目节点的迁移方式

`custom_nodes/DualV100/h3_ref2v_tp.py` 只暴露普通固定输入：

```text
clip, vae, prompt, width, height, length,
ref_image_size, ref_image
```

执行时将一个图片显式包装成：

```python
ref_images={"ref_image_0": ref_image}
```

然后复用 ComfyUI 当前 H3 ref2va 的处理逻辑。这样仍然保留官方已经验证过的：

1. 参考图按 `match`/`max` 规则缩放；
2. 视频 VAE 编码为 reference latent；
3. `clip.tokenize(..., minimax_ref_items=...)` 插入 `<Picture 1>` 视觉 token；
4. conditioning 写入 `minimax_refs`。

项目自己的 `comfy/model_base.py` 随后把它放进：

```text
minimax_refs
  -> PackedLayout(refs=...)
  -> ref_img segments
  -> H3 model forward
  -> PersistentH3TPBlocks
  -> NCCL rank0/rank1
```

因此参考图只改变 packed layout 和 conditioning，不改变 H3 的 TP 切分、NCCL
通信协议或 worker 生命周期。它也不是首帧/尾帧约束，不会写入
`minimax_keyframes`。

## 工作流

仓库内：

- `workflows/h3-v100-ref2v-adjustable-resident.json`：可调分辨率/时长的 API 工作流；
- `workflows/h3-v100-ref2v-adjustable-resident-ui.json`：可在 Web UI 打开的工作流；
- `workflows/h3-v100-ref2v-tp-smoke-448x256-1step.json`：低资源验证工作流。

实际 ComfyUI 已同步到：

```text
/home/regen/minimax-h3/ComfyUI/custom_nodes/DualV100/h3_ref2v_tp.py
/home/regen/minimax-h3/ComfyUI/user/default/workflows/H3-V100-07-ref2v-adjustable-resident.json
/home/regen/minimax-h3/ComfyUI/user/default/workflows/H3-V100-07-ref2v-adjustable-resident-ui.json
```

参考图工作流不再把 audio VAE 接到 conditioning 节点；audio VAE 仍独立接在
`VAEDecodeAudio`，只负责最终音频解码。这样 image-only ref2va 不会无谓地在
参考图编码阶段触发 audio VAE。

## 内存安全约束

- 模型根目录固定为 `/mnt/GALAX/minimax-h3/models`；
- `H3_NO_HOST_MMAP=1`；
- 默认关闭 pinned CPU memory；
- 默认 `enable-dynamic-vram + fast-disk`；
- H3/Qwen/VAE 的持久 TP/MP 生命周期不因每次生成而重复 load/unload；
- `ref_image_size` 默认 `match`，不要无理由切到 `max`，后者会增加 vision token、
  reference latent 和采样压力；
- 首次实机扩大到 832×480 前，先观察两卡余量；如果需要降峰，优先降低参考图尺寸或
  分块 decode，不要为腾显存而卸载/重载 TP worker 和常驻 Qwen。

## 后续验证边界

本次只验证了 reference conditioning + TP denoise + latent 保存，没有宣称
832×480 完整视频/音频 decode 已在本轮复测。扩大测试时应另记：VAE encode/decode
耗时、两卡 peak、主进程 RSS、输出 MP4/audio finite，以及连续第二次生成是否仍复用
同一个 rank1 PID。
