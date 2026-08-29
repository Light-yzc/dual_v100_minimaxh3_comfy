# H3 双 V100 工作流

## 生产路由（INT8 视频 VAE）

`H3-V100-09/10/11` 是当前生产入口，视频 VAE 使用 INT8-ConvRot。

| Preset | 模式 | 尺寸 | 用途 |
|---|---|---|---|
| `H3-V100-09-int8-ref2v-...-ui` | 参考图 | 832×480 124帧 4步 | 日常出片，参考图保持身份 |
| `H3-V100-10-int8-fl2v-...-ui` | 首尾帧 | 832×480 124帧 4步 | 日常出片，首尾帧做时间锚点 |
| `H3-V100-11-int8-ref2v-smoke-...` | 参考图 | 448×256 22帧 1步 | 快速冒烟，约 25 秒 |
| `H3-V100-11-int8-fl2v-smoke-...` | 首尾帧 | 448×256 22帧 1步 | 快速冒烟 |

带 `-ui` 后缀的是网页版（含节点布局和说明便签），不带的是 API 版，配合
`scripts/submit_workflow.py` 使用。

### 切换模式不用重连线

条件节点的 `mode` 是普通 widget，三个图片输入是 lazy 求值：

- `reference_image`：只求值 `reference_image` 输入
- `first_last_frames`：只求值 `first_frame` / `last_frame`

未选中的输入不会被求值，也不会触发模型重载。所以 09 和 10 其实是同一张图的
两个预设，改 `mode` 即可互相切换。

### 为什么用 INT8 视频 VAE

同一条 `tiled_decode` 路径、448×256 实测：

| 格式 | 解码额外显存 | 说明 |
|---|---|---|
| INT8-ConvRot | 约 306 MiB | W8A16：逐 Linear 反量化成有界临时量，用完即弃 |
| FP16 | > 11 GiB | 走 ComfyUI 普通 Linear，每层激活全程驻留 |

双卡在 DiT 常驻的情况下这个差距是决定性的。INT8 checkpoint 只量化 decoder 的
144 个 Linear，encoder 的 116 个张量保持原精度，因此参考图编码质量不变。

音频 VAE 必须保持 FP32，不要改成 INT8 或 FP16。

### 显存布局会自动切换

采样期 18/18、解码期 24/12，边界的 6 个 block 走 NVLink 搬移，实测 7–15 ms。
采样期让 cuda:0 留出余量给 DiT；解码期把重的一半放 cuda:0，因为 layer-MP
解码是串行的。

由 `H3_VAE_DIT_SPLIT` / `H3_VAE_DECODE_SPLIT` 控制。设 `H3_VAE_SPLIT` 会钉死
单一布局并关闭搬移。搬移前有整卡准入检查，cuda:0 放不下时自动退档，不会让
后续分配在 NCCL collective 里失败。

## FP16 对照组

`h3-v100-multimode-*` 保留 FP16 视频 VAE，用于和历史记录对比。日常使用请用
09/10/11。

## 旧的 4B 路由

`H3-V100-01` 到 `08` 使用 Qwen3-VL-4B Q4_K_M + ClipProj v3.1：

1. `H3-V100-01-smoke-448x256-1step`：只测采样和 latent，不加载 VAE
2. `H3-V100-02-video-448x256-1step`：最小完整视频
3. `H3-V100-03-video-832x480-4step`：约 40 秒采样 + 40 秒解码
4. `H3-V100-04-sample-1MP-1344x768-4step`：只采样 1MP latent，约 196 秒
5. `H3-V100-05-decode-1MP-pinned`：读取同一个 latest latent 并解码
6. `H3-V100-06/07/08`：i2v / ref2v / multimode 可调预设

这批 preset 请保持 `ClipProjLoader` 为 `device=cuda:1, mode=resident`，
H3 loader 为 `cuda:0`。不要和 32B/单卡 workflow 混用。

## 通用约束

- 模型根目录：`/mnt/GALAX/minimax-h3/models`
- 不要设置 `H3_NO_HOST_MMAP=0`
- 换图在 `LoadImage` 节点重新上传即可，默认读 `ComfyUI/input/example.png`

服务管理：

```bash
./scripts/start_comfyui.sh start
./scripts/start_comfyui.sh logs
./scripts/start_comfyui.sh stop
```
