# H3 VAE MP 显存均衡记录

日期：2026-08-28

## 结论

当前 Qwen32 MP + DiT TP 服务将视频 VAE 的 36 个 decoder block 固定为
`24/12`（GPU0/GPU1），输出 buffer 显式放在 CPU。这样把 VAE 的较大部分放到
采样结束后相对空闲的 GPU0，避免 GPU1 同时承载 DiT rank1 与 24 个 VAE block。

旧的 `12/24` 在 `832×480 / 124f / 4-step` 请求的 decode 尾部失败：GPU1 的
DiT rank1 约占 12.2 GiB，VAE 约占 3.2 GiB，`_finalize_pixels()` 再申请
183 MiB 时只剩 6.6 MiB，触发 OOM。

## 实测

在双 Tesla V100-SXM2-16GB、Qwen32 Q2 MP、参考图模式下，使用同一工作流将
step 临时降为 1 进行显存 gate：

```bash
H3_VAE_SPLIT=24 ./scripts/start_comfyui.sh restart
/home/regen/minimax-h3/.venv/bin/python scripts/submit_workflow.py \
  /tmp/h3-v100-multimode-qwen32-mp-832x480-124f-1step-split24.json \
  --wait --timeout 2400
```

请求成功（102.2 s），无 OOM；完成后 `nvidia-smi` 为 GPU0 `11392 MiB`、GPU1
`10090 MiB`。日志确认：

```text
[H3 VAE MP] ... split=24; devices=cuda:0,cuda:1
```

VAE-only 小尺寸 decode 也通过，输出 `torch.float32`、全部 finite，host mmap
为 false。随后原始 4-step 工作流复用同一 runtime 成功完成（55.2 s，含缓存后的
conditioning/采样/decode），没有再次出现 OOM。

## 配置与回退

启动器默认 `H3_VAE_SPLIT=24`、`H3_VAE_OUTPUT_DEVICE=cpu`。显存紧张或需要
复现旧基线时可使用 `H3_VAE_SPLIT=12`；`H3_VAE_SPLIT=18`、`20`、`28` 等
整数仍受支持。`H3_VAE_OUTPUT_DEVICE=cuda:N` 仅用于明确的短 decode 实验，生产
不要把长视频输出放回 GPU1。
