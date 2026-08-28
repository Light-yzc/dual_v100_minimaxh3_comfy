# H3 INT8 Video VAE：双 V100 实测结论

日期：2026-08-26

## 结论

INT8 ConvRot VAE 已可用于部署试跑。当前路线是：

```text
INT8 权重常驻双卡
  -> 当前 Linear 临时解量化为 FP16
  -> V100 FP16 Tensor Core GEMM
  -> 12/24 model parallel
  -> spatial tile batch=2
```

它不 mmap VAE checkpoint，不保留完整 FP16 权重，也不把 1.43 GiB 的
1MP 输出留在 GPU0。正式工作流仍可选 FP16 VAE，便于一键回退。

## 关键结果

| 测试 | FP16 | 旧 INT8 eager | 当前 INT8 |
|---|---:|---:|---:|
| 832×480 / 124f decode | 11.55 s | 29.86 s | 13.36 s（tile batch=1 公平对照） |
| 1MP / 124f decode | 25.62 s（GPU output） | 67.73 s（GPU output） | 28.64 s（disk-backed output，tile batch=2） |

1MP 当前 INT8 对 FP16 reference：

- RMSE：`0.00073252`
- relative RMSE：`0.0938%`
- MAE：`0.00037025`
- cosine：`0.99999956`
- max abs：`0.051625`
- 输出全部 finite，范围 `[0, 1]`

1MP 当前 INT8 PyTorch allocated：

- 常驻：GPU0 `1119.73 MiB`，GPU1 `1553.26 MiB`
- decode 峰值：GPU0 `1442.38 MiB`，GPU1 `2258.13 MiB`
- 峰值合计：`3700.51 MiB`（约 `3.61 GiB`）
- GPU1 比 GPU0 高 `815.75 MiB`

VAE 单测并非 50/50。GPU1 持有 24 个 decoder block、norm/proj 和输出 canvas；
12/24 split 是为了抵消完整服务中 GPU0 的 H3 rank0、外围模块和采样 owner，
因此不要按 VAE 单测擅自改成 18/18。

## 部署试跑

启动：

```bash
cd /home/regen/code/minimax_v100
./scripts/start_comfyui.sh
```

Video VAE 节点选择：

```text
minimax_h3_video_vae_int8_convrot.safetensors
```

启动器默认参数：

```text
H3_VAE_INT8_SM70_W8A16=1
H3_VAE_INT8_TILE_BATCH=2
H3_VAE_SPLIT=12
H3_NO_HOST_MMAP=1
```

若完整服务显存比预期紧，先用
`H3_VAE_INT8_TILE_BATCH=1 ./scripts/start_comfyui.sh` 回退小批处理；若要完全
回退，工作流重新选择 `minimax_h3_video_vae_fp16.safetensors`。

## 结果文件

- `/home/regen/minimax-h3/vae_bench/int8_832x480_sm70_w8a16_disk_20260826.json`
- `/home/regen/minimax-h3/vae_bench/fp16_832x480_disk_20260826.json`
- `/home/regen/minimax-h3/vae_bench/int8_1mp_sm70_w8a16_tilebatch2_disk_20260826.json`
- 1MP candidate raw buffer：`/home/regen/minimax-h3/vae_bench/int8_1mp_sm70_w8a16_tilebatch2_disk_20260826.bcthw.f32`

1MP 测试 checkpoint 未出现在 `/proc/<pid>/maps`。测试时 RSS 峰值约
`4341 MiB`，主要发生在逐帧比较两个 1.43 GiB file-backed 视频时；没有再构造
完整 candidate/reference/delta 三份匿名内存。
