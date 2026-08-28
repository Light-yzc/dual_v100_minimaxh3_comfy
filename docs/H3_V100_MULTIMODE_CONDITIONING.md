# H3 双 V100 多模式图像条件

日期：2026-08-27

## 用法

新增节点 `MiniMaxH3ReferenceKeyframeToVideoTP`，在一条持久 TP 工作流中提供两个
互斥模式：

- `first_last_frames`：使用 `first_frame` 和可选 `last_frame`，写入官方 H3 的
  `minimax_keyframes` 条件；首尾帧使用相同的几何变换。
- `reference_image`：使用 `reference_image`，沿项目的固定输入
  `MiniMaxH3ReferenceToVideoTP` 路径写入 `minimax_refs`，图片是身份/构图参考而不是
  时间锚点。

三个图片插槽可以同时接线，切换 `mode` 后只有选中的插槽会被 lazy evaluation 拉取；
两种模式不会各自创建 CLIP、VAE、H3 TP 或 LoRA 实例。默认模式是
`first_last_frames`，以保持原首帧工作流行为。

## 文件

- 节点源码：`custom_nodes/DualV100/h3_ref2v_tp.py`
- API 工作流：`workflows/h3-v100-multimode-adjustable-resident.json`
- UI 工作流：`workflows/h3-v100-multimode-adjustable-resident-ui.json`
- 低资源 smoke：`workflows/h3-v100-multimode-smoke-448x256-1step.json`

部署后的 UI 文件位于：

`/home/regen/minimax-h3/ComfyUI/user/default/workflows/H3-V100-08-multimode-adjustable-resident-ui.json`

## 低资源验证（2026-08-27）

在双 V100 持久 NCCL TP、H3 Q4、Turbo LoRA、Qwen 4B ClipProj ridge、448×256、22
帧、1 step 下串行验证：

- 首尾帧模式：成功，冷启动总耗时约 92.2 s；H3 TP 首次 forward 日志峰值
  `10868/6578 MiB`（GPU0/GPU1）。
- 参考图模式：成功，复用同一进程中的常驻 worker；第二次 forward 日志峰值
  `10829/6545 MiB`，执行约 1.3 s。
- cgroup 主进程服务峰值约 4.5 GiB，限制为 7 GiB；测试未触发 OOM。
- latent 输出：
  `/home/regen/minimax-h3/ComfyUI/output/benchmarks/h3_tp_e2e/multimode_smoke_448x256_latent.pt`
  和
  `/home/regen/minimax-h3/ComfyUI/output/benchmarks/h3_tp_e2e/multimode_smoke_reference_448x256_latent.pt`。

这只是分派和 TP 兼容性 smoke，不代表 832×480 完整视频 decode 的质量或显存上限；
扩大分辨率时仍应串行执行并观察两卡峰值。
