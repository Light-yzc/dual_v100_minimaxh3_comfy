# Qwen32 Q2 + H3 TP-Speed：复杂运镜 832×480 / 5 秒实测

日期：2026-08-27

## 结论

在双 Tesla V100-SXM2-16GB、4-step、832×480、124 帧（24 fps，5.1667 秒）上，
当前 `TE-Speed tail42 / c0.03 / mcs2` 确实有实测加速，但质量门禁未通过：

```text
Full DiT：43.155 s
TE-Speed：31.215 s
冷启动时间也算入时：1.383×
扣除 Full 首个 kernel 冷启动后的稳态归一化：约 1.265×
```

TE-Speed 只在第 2 个 denoise step 命中一次 cache，跳过 42/200 次 block visit；
复杂运镜下该一次 tail residual 复用已经造成明显 latent 漂移，因此 `c0.03` 只能
作为速度/质量对照样例，暂不应设为质量优先默认值。

## 固定条件

- Qwen：`qwen3vl-32B-MiniMax-H3-Q2_K.gguf`，50 层 output-row TP，`evict`，
  conditioning 完成后清理 CUDA payload。
- H3：`minimax_h3_fl2va_pruned_fp8_Q4_0.gguf` +
  `minimax_h3_turbo_v4_step600_ema.safetensors`，LoRA strength `1.0`，双 rank NCCL TP。
- Cache：warm prefix `0..7`，tail `8..49` 共 42 blocks，CPU `Q4_0`，约
  `45,290,448` bytes（约 43.19 MiB），`mcs=2`，threshold `0.03`。
- VAE：FP16 resident MP，split `12/24`；Full 解码后，TE-Speed 解码复用同一个 VAE，
  没有再次 load/unload。
- seed：`2011`；scheduler：`simple`；steps：`4`；CFG 分支：1；无参考图。
- prompt 使用快速横移、绕拍、whip-pan、升降和推进的雨夜霓虹市场追逐场景，包含
  摩托车、无人机、蒸汽、雨滴和反光，用于暴露时序 cache 误差。

## DiT 性能

| 路线 | step 0 | step 1 | step 2 | step 3 | DiT 合计 | 模式 |
|---|---:|---:|---:|---:|---:|---|
| Full | 13.532 s | 9.871 s | 9.870 s | 9.882 s | 43.155 s | FULL/FULL/FULL/FULL |
| TE-Speed | 9.850 s | 1.574 s | 9.884 s | 9.907 s | 31.215 s | FULL/CACHE/FULL/FULL |

TE-Speed 的 cache step 记录：

```text
tail blocks: 42
executed blocks: 8
skipped blocks: 42
Q4 cache add: 29.19 ms
Q4 cache bytes: 45,290,448
```

Full 的第一个 forward 包含首次 kernel/运行时热身，不能直接用请求总时长和 TE-Speed
请求总时长比较。使用 Full 的后三个稳态 step 平均值（约 9.874 s）归一化，TE-Speed
四步平均约 7.804 s，得到约 `1.265×` 的稳态端到端 DiT 加速；这是本次应采用的性能
结论。Full API 请求总时长 `80.34 s` 还包含约 `36.56 s` 的首次 H3 TP 初始化，
TE-Speed 请求总时长 `32.27 s` 不包含重复初始化。

## 显存、RAM 和 mmap

| 项目 | Full | TE-Speed |
|---|---:|---:|
| torch peak GPU0 | 7,954 MiB | 7,994 MiB |
| torch peak GPU1 | 7,527 MiB | 7,527 MiB |
| runtime RSS（运行后） | 约 2,160 MiB | 约 2,225 MiB |
| service cgroup peak | — | 4,711 MiB |
| cgroup 上限 | — | 7 GiB |

Qwen conditioning 单独耗时 `39.26 s`；condition tensor 保持在 peer cache，约几百 KiB，
Qwen 权重随后按 `evict` 清理。检查 `/proc/466679/maps` 和 rank1 `/proc/467348/maps`
没有发现 Qwen GGUF 路径映射，模型 payload 仍通过 bounded direct reader 读取，没有
完整模型 mmap 到 RAM。

VAE 解码：

```text
Full：30.08 s（包含一次 resident VAE load）
TE-Speed：14.02 s（复用已加载 VAE）
```

两条解码结束时物理显存约 GPU0 `8,862 MiB`、GPU1 `11,946 MiB`；该数值是解码完成
后的 resident 状态，不是 DiT peak。两张卡均未 OOM，且没有重复加载 VAE。

## 精度对比（TE-Speed c0.03 vs Full）

使用保存的 AV latent 直接比较：

| tensor | relative RMS | cosine | max abs | finite |
|---|---:|---:|---:|---|
| video | 0.7303 | 0.7215 | 4.7308 | yes |
| audio | 0.6380 | 0.7835 | 1.8058 | yes |

因此本次结果不是“加速成功即可上线”：c0.03 的速度收益真实存在，但复杂运动下
quality degradation 过大。后续应优先测试更短 tail、FP16 residual 对照或更可靠的
feature/group criterion；在质量对照完成前，不要把该参数写入默认路线。

## 输出文件

采样 latent：

- `/home/regen/minimax-h3/ComfyUI/output/benchmarks/qwen32_q2_tp/qwen32_complex_full_832x480_124f_4step_seed2011.pt`
- `/home/regen/minimax-h3/ComfyUI/output/benchmarks/qwen32_q2_tp/qwen32_complex_te_speed_tail42_c0p03_832x480_124f_4step_seed2011.pt`

视频：

- `/home/regen/minimax-h3/ComfyUI/output/video/qwen32_q2_tp/complex_full_832x480_124f_4step_seed2011_00001_.mp4`
- `/home/regen/minimax-h3/ComfyUI/output/video/qwen32_q2_tp/complex_te_speed_tail42_c0p03_832x480_124f_4step_seed2011_00001_.mp4`

两条视频均为 H.264、`832×480`、`24 fps`、`124` 帧、`5.1667 s`。

对应工作流：

- `workflows/qwen32-q2-tp-full-complex-832x480-124f-4step.json`
- `workflows/qwen32-q2-tp-te-speed-tail42-c0p03-complex-832x480-124f-4step.json`
- `workflows/qwen32-q2-shared-tp-conditioning-complex-832x480.json`
- `workflows/qwen32-q2-decode-complex-full-832x480-124f-4step.json`
- `workflows/qwen32-q2-decode-complex-te-speed-tail42-c0p03-832x480-124f-4step.json`

原始逐 forward JSON、API 提交记录和 latent comparison 位于 `results/`：

```text
results/forward_0001_14977t_20260827-173815.json ...
results/forward_0008_14977t_20260827-173956.json
results/qwen32_complex_te_speed_vs_full_832x480_124f_4step.json
results/qwen32_complex_full_832x480_124f_4step_seed2011.json
results/qwen32_complex_te_speed_tail42_c0p03_832x480_124f_4step_seed2011.json
results/qwen32_complex_decode_full_832x480_124f_4step.json
results/qwen32_complex_decode_te_speed_tail42_c0p03_832x480_124f_4step.json
```
