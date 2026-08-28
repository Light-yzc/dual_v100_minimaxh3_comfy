# Qwen32B Q2 TP + Async VAE Delivery

本文是双 V100 上线 `Qwen32B Q2 output-row TP -> DiT -> FP16 VAE` 实验路线的操作单。
**TP 仍是实验路线，不能在当前服务运行期间切换。** 解耦的 layer-MP 已通过线上
smoke，并已成为 Qwen32 启动器默认；实现和回退说明见
[`QWEN32B_Q2_MP.md`](QWEN32B_Q2_MP.md)。未使用 32B 节点时，服务仍运行现有
4B Q4 + ClipProj 工作流。

## 1. 预检与安装

在仓库源目录执行，模型文件只读自 `/mnt/GALAX`，不要把模型复制到仓库或系统盘：

```bash
cd /home/regen/code/minimax_v100
test -f /mnt/GALAX/minimax-h3/models/text_encoders/qwen3vl-32B-MiniMax-H3-Q2_K.gguf
test -f /mnt/GALAX/minimax-h3/models/vae/minimax_h3_video_vae_fp16.safetensors
INSTALL_ROOT=/home/regen/minimax-h3 ./scripts/setup_ubuntu.sh
/home/regen/minimax-h3/.venv/bin/python -m py_compile \
  custom_nodes/DualV100/*.py custom_nodes/NoHostMMap/*.py
./scripts/audit_qwen32_q2_tp_layout.py \
  --output results/qwen32_q2_tp_layout.json
/home/regen/minimax-h3/.venv/bin/python scripts/test_h3_async_vae.py
/home/regen/minimax-h3/.venv/bin/python scripts/test_h3_async_vae_bridge.py
```

安装门禁必须同时满足：布局报告为 902 tensors/50 layers、所有 output-row ranges
闭合、`payload_mmap_count_zero=true`；两个异步测试全通过；`py_compile` 无错误。
安装失败时不要启动服务，保留报告和终端输出供审计。

## 2. 实验启用

确认服务空闲且 `/queue` 为空后，用受保护的 user service 启动。`evict` 是首版唯一
允许的 Qwen residency；FP16 VAE 预取预算仍由启动器的每卡上限控制：

```bash
curl -fsS http://127.0.0.1:8188/queue
H3_QWEN32_Q2_TP=1 \
H3_QWEN32_Q2_MODE=tp \
H3_QWEN32_Q2_MP=0 \
H3_QWEN32_RESIDENCY=evict \
H3_QWEN32_KEEP_LAYERS=0 \
H3_ASYNC_VAE_LOAD=1 \
H3_NO_HOST_MMAP=1 \
./scripts/start_comfyui.sh restart
./scripts/start_comfyui.sh status
```

随后确认 `/object_info` 中出现工作流实际引用的 Qwen32/异步 VAE 节点，并检查日志
中只有一个 `minimax-h3-comfy.service` 主进程、同一持久 rank1 worker，以及
Qwen clear/barrier 后才开始 VAE prefetch。不要为新路线手工再启动第二个 `torchrun`
或 rank1 worker。

## 3. 上机门禁与 API 验证

按顺序提交最终接口对应的低资源工作流（固定 448x256、22 帧、1 step），同一 seed
连续提交两次：

```bash
/home/regen/minimax-h3/.venv/bin/python scripts/submit_workflow.py \
  workflows/qwen32-q2-tp-async-vae-smoke-448x256-1step.json \
  --server http://127.0.0.1:8188 --wait \
  --output results/qwen32_q2_tp_async_vae_smoke_1.json
```

两次响应都必须 `status.completed=true` 且无 `status_str=error`；输出 latent/video/audio
均 finite。日志或 runtime stats 必须证明第二次 conditioning 命中 cache、没有重新
执行 Qwen collective；两 rank 的 clear 顺序一致，Qwen payload 清零后才出现 VAE
prefetch，采样返回后才 finalize tail。记录服务 PID、`nvidia-smi` 峰值、RSS/cgroup
events、swap、`/proc/<pid>/maps`、Qwen/DiT/VAE 耗时和结果 JSON。出现 NaN/Inf、rank
不一致、Qwen payload mmap、OOM、预算越界或 finalize 失败，门禁即失败。

低资源 smoke 通过后，才允许按计划执行 832x480/124f/4step，再执行 1 MP 和参考图
矩阵；每个尺寸都要单独保存结果和显存余量。TP 的开关不得写入默认环境；MP 虽已是
Qwen32 默认，未完成 P1--P5 全部 correctness、质量和容量门禁前，不得启用 full
residency 或替换 4B 工作流。

## 4. 回滚与服务检查

任何门禁失败先停止提交，保留 journal 和 `results/` 审计文件，然后关闭新路径并
重启同一受保护服务：

```bash
./scripts/start_comfyui.sh stop
H3_QWEN32_Q2_MODE=mp H3_QWEN32_Q2_MP=1 H3_QWEN32_Q2_TP=0 H3_ASYNC_VAE_LOAD=0 \
H3_QWEN32_RESIDENCY=evict H3_QWEN32_KEEP_LAYERS=0 \
H3_NO_HOST_MMAP=1 ./scripts/start_comfyui.sh start
./scripts/start_comfyui.sh status
./scripts/smoke_clipproj_v100.sh
```

`status` 必须显示 service active；`/queue` 能返回 JSON；现有 ClipProj smoke 必须
再次完成。回滚只切换开关，不删除模型、缓存或 worker 日志；需要恢复代码时，应在
新的已审计提交上重新运行安装和全部门禁，禁止直接修改已安装目录。
