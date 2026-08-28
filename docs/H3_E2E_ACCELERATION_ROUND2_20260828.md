# H3 端到端加速 Round 2 工作单

日期：2026-08-28
状态：**只是工作单**。本轮没有跑任何 benchmark，没有改默认值，没有启动服务。
下面每一项都写了入口命令和验收门禁，交由执行者按顺序单变量验证。

前置文档：[`H3_E2E_ACCELERATION_PLAN_20260828.md`](H3_E2E_ACCELERATION_PLAN_20260828.md)、
[`H3_V100_PERFORMANCE_TODO.md`](H3_V100_PERFORMANCE_TODO.md)、
[`H3_V100_FREEZE_20260826.md`](H3_V100_FREEZE_20260826.md)。

## 0. 先修正两处会误导执行者的记录

这两条是本轮查代码/磁盘时发现的，动手前必须先确认，否则会去实现已经存在的东西、
或者按一个跑不起来的基线做对比。

**(a) conditioning cache 已经实现，不是待做项。**
`H3_E2E_ACCELERATION_PLAN_20260828.md` 把它列为 P0 待做。实际实现在
[h3_qwen32_tp_node.py:530](../custom_nodes/DualV100/h3_qwen32_tp_node.py#L530)，
环境变量 `H3_QWEN32_COND_CACHE_ENTRIES` 默认 `4`。
待做的不是"实现"，而是**审计 cache key**：确认 prompt、参考图内容 hash、
`ref_image_size`、mode（`first_last_frames` / `reference_image`）、模型与路由版本
全部进 key，任何一项失配必须 miss。key 不完整比没有 cache 更危险。

**(b) INT8 视频 VAE 的 checkpoint 不在磁盘上。**
`H3_V100_FREEZE_20260826.md` 把 `H3_VAE_INT8_SM70_W8A16=1` 列为固定默认，并记录
INT8 1 MP decode `28.64 s`、cosine `0.99999956`。但
`models/vae/` 下只有 `minimax_h3_video_vae_fp16.safetensors`（5.21 GB）和
`minimax_h3_audio_vae_fp32.safetensors`，**没有 `minimax_h3_video_vae_int8_convrot.safetensors`**；
生产工作流引用的也是 FP16 那个。也就是说 INT8 开关当前是空转的，
`28.64 s` 不是现在跑的 decode 时间。

执行者二选一，不要跳过：重新产出 INT8 checkpoint 并复测，或者把 `28.64 s`
标注为"当前不可复现"，decode 基线全部按 FP16 重新测。**在此之前不要用
`28.64 s` 做任何 decode 加速的分母。**

## 1. 已被实测排除的路线：不要重做

这些不是"没调好"，是已经量到上限或已经不合格。重做等于浪费机时。

| 路线 | 证据 | 结论 |
|---|---|---|
| 自定义 SM70 attention kernel（Triton / CUTLASS / TileLang） | `S=37746` 纯 cuBLAS `bmm` 只有 32.3 TFLOPS，比 efficient SDPA 的 34.0 还低；bmm 下界约 810 ms vs 现状 600 ms | 方向不成立，除非改算法而非 kernel |
| query-row 分块 | 最好 1.018×，折算端到端 1.013×，多 286 MiB 峰值 | 收益不抵 GPU1 余量风险 |
| sequence parallel | cosine 0.4838，2.01× 是漏算一半 head 贡献 | 已作废 |
| TE-Speed tail42 | video relative RMS **0.730**、cosine 0.721 | 质量不合格 |
| Group Cache `t=0.30` | video relative RMS **0.543**、cosine 0.851 | 质量不合格 |
| Group Cache `t=0.005` | 质量合格（rel RMS 0.0575、cosine 0.9984）但跳过 0 个 block，比 full **慢 6.2%** | 当前判定开销吞掉全部收益 |
| `max-autotune` / 全路径 `torch.compile` | SM70 上 cuBLAS 已快于候选 Triton GEMM；冷编译数十至数百秒 | 不进生产 |

Group Cache 的结论要读准：它不是"阈值再调调就能上"。`t=0.005` 安全但零命中且负收益，
`t=0.30` 有命中但质量崩，中间没有可用区间——因为判定用的是全局阈值 + 整张 Q4
`previous_input`。要继续做，必须先按
[`ADAPTIVE_GROUP_CACHE_MATH_REVIEW_20260827.md`](ADAPTIVE_GROUP_CACHE_MATH_REVIEW_20260827.md)
的 P1–P3（group-specific 敏感度、AdaLN `d^c`、signature 替代 previous_input）改判定，
不要再扫阈值。这一项排在最后。

## 2. 本轮候选，按"收益 ÷ 风险"排序

### P0-a：VAE decode 的 tile 跨 stage 流水（本轮最大的未开采项）

视频 VAE 是 36 个 decoder block 按 `24/12` 做 layer-MP。layer-MP 是**串行**的：
tile 走完 GPU0 的 24 个 block 才进 GPU1 的 12 个，任一时刻有一张卡在空转。
而 spatial tiling 已经存在（`H3_VAE_INT8_TILE_BATCH=2`），也就是说已经有多个
独立 tile 可以调度。

把 tile `i` 的 GPU1 阶段与 tile `i+1` 的 GPU0 阶段重叠，是**纯调度改动**：
每个 tile 仍完整经过全部 36 个 block、权重不变、数值不变，输出应当逐元素
bitwise 相同。这是本轮唯一一个"理论上不需要拿质量换速度"的加速项。

上界是 MP 段的 2×；按 `24/12` 的不均衡切分和 tile 数有限，现实预期 decode
`1.3–1.6×`。代价是同时有两个 tile 在飞，峰值显存上升，必须重新做双卡审计。

要求：
- 每张卡一个独立 stream，跨卡用 event 而非 `synchronize()` 串死；
- 输出 buffer 保持 `H3_VAE_OUTPUT_DEVICE=cpu`；
- 新增开关默认关闭（建议 `H3_VAE_MP_PIPELINE`），失败回退现有串行路径；
- 验收必须是 **bitwise 相同**（`max_abs == 0.0`）。不是 bitwise 就说明实现动到了
  数值路径，直接回退，不要用 cosine 0.999 蒙过去。
- 先在 448×256 验证 bitwise + 显存，再上 832×480，最后 1 MP。

这一项建议优先做，因为 DiT 已经贴住 Volta 上限（见第 1 节），decode 是剩下
唯一一个既占大头又还有结构性浪费的阶段。

### P0-b：application clock 与 persistence mode（半小时能量完）

当前两张卡：

```text
persistence_mode = Disabled
clocks.applications.graphics = 1312 MHz
clocks.max.sm = 1530 MHz          # 877,1530 是受支持的出厂组合
```

`scripts/enable_v100_performance.sh` 已经存在（应用 `-pm 1` 和 `-ac 877,1530`），
但**没有应用**。纸面上 SM clock 上限 +16.6%，对当前 compute-bound 的 DiT 是直接收益。

但要诚实说另一面：已有记录显示 GPU1 在连续满载下到 82°C、busy SM clock 掉到
570 MHz，rank0 的 collective 等待从 1.83 s 涨到 8.55 s。也就是说这台机器在持续
负载下的真实限制器可能是**散热**，抬高 app clock 上限有可能只是把节流点前移，
甚至加大两卡 skew。所以这是"便宜的测量"，不是"确定的收益"。

做法：同一工作流连续 4 次 forward，分别在默认 1312 和 1530 下各跑一轮，
全程按第 3 节采样 `clocks.current.sm` / `temperature.gpu` / `power.draw`。
判定看**第 3、4 次** forward（第 1 次含冷启动，第 2 次还没热起来），不看第 1 次。
若 1530 下后段 forward 比 1312 更慢或 skew 更大，就明确记录"该机器散热受限，
不抬 app clock"，并把结论写回 freeze 文档，避免以后反复试。

`persistence_mode=Disabled` 与性能无关，但会带来请求间的驱动初始化抖动，
建议独立开启，不与 clock 实验混在一次。

### P1：三个"已实现但默认关闭"的开关，逐个 A/B

代码都在，默认都是关的。执行者不需要写实现，只需要单变量 A/B + 门禁。

| 开关 | 默认 | 实现位置 | 已有说法 | 待验证 |
|---|---|---|---|---|
| `H3_QWEN32_MP_PREFETCH=1`（配 `..._MAX_MIB=256`） | `0` | `h3_qwen32_q2_mp.py` | Qwen 阶段 6.4% | 冷请求真实省几秒；cache hit 时应当无影响 |
| `H3_ASYNC_VAE_LOAD=1`（`..._PREFETCH_MIB=1962,1787`） | `0` | `h3_async_vae.py` / `h3_async_vae_bridge.py` | 预计隐藏 3–10 s | 只对**冷 VAE**有效；必须每卡留 ≥1 GiB 余量 |
| `H3_AUDIO_VAE_COMPILE=1` | `0` | `comfy/ldm/minimax/audio_vae.py` | block 级 1.538× | 真实 checkpoint 的首编译成本、warm decode、waveform finite/cosine、MP4 音轨可播放 |

注意三者的收益都只在**冷路径**。当前 warm 请求（`execution_cached` 覆盖节点 1–14）
已经绕过 conditioning，所以 Qwen prefetch 对 warm 请求应当是 0 收益——如果 A/B
显示 warm 也变快了，那是噪声或 page cache 差异，不是收益。冷/warm 必须分开报，
不能平均。

`H3_ASYNC_VAE_LOAD` 的风险最高：它在 DiT 采样期间往两张卡预取 VAE 权重，而
`24/12` split 本来就是为了解决 GPU1 的 decode 尾部 OOM（旧 `12/24` 在
`_finalize_pixels()` 申请 183 MiB 时只剩 6.6 MiB）。预取上限必须按第 3 节的
整卡采样验证，不能只看 torch allocator 数字。

### P2：vision tower 修复后的显存复测（新增，必须做）

本轮修掉了 `_load_vision_frontend()` 的加载 bug：在 `--enable-dynamic-vram` 下
`disable_weight_init.Linear` 把 `weight`/`bias` 存成未注册的 `None`，
`named_parameters()` 看不到，导致 232 个投影权重从未加载，
日志是 `loaded 119 vision tensors`，直到 vision 前向才炸 `F.linear`。

修复后应当是 `loaded 351 vision tensors on cuda:0`。**低于 351 就是还没修好**，
不要继续往下测。

后果是 vision tower 现在会真的占约 **0.9–1.1 GiB FP16 on cuda:0**——这本来就是
设计预期，之前"看起来便宜"是因为漏加载。在 16 GB 卡上这不是可忽略量，
所以参考图/首尾帧模式的双卡峰值必须重测一遍。`_release_vision()` 已经存在，
确认 conditioning 结束后确实释放（对比释放前后的 `nvidia-smi`，不是 allocator）。

纯文本 prompt 不走 vision，不受影响。

### P3：Group Cache 判定重构

见第 1 节末尾。只有在 P0/P1 做完、并且
[`ADAPTIVE_GROUP_CACHE_MATH_REVIEW_20260827.md`](ADAPTIVE_GROUP_CACHE_MATH_REVIEW_20260827.md)
的 P1–P3 落地后才重启，且第一个门禁是"零命中时不慢于 full"，
而不是"能跳多少 block"。

## 3. 测量协议（所有项共用，不满足就不接受结论）

### 显存必须用 nvidia-smi 采样，不能用 rank telemetry

`forward_*.json` 里的 `allocated_mib` / `peak_allocated_mib` 只是**该进程 torch
allocator 内**的量。它看不到 NCCL 通信 buffer、CUDA context、cuBLAS workspace，
也看不到另一个 rank。两个 rank 的数字不一致往往只是记录口径差异，不是真实分布。
判断"某张卡还剩多少"只能看整卡：

```bash
nvidia-smi --query-gpu=index,timestamp,memory.used,memory.total,utilization.gpu,\
temperature.gpu,clocks.current.sm,power.draw \
  --format=csv,noheader,nounits -l 1 \
  > results/e2e_smi_<case>.csv &
SMI_PID=$!
# ... 提交工作流 ...
kill "$SMI_PID"
```

按 `index` 取 `max(memory.used)` 作为该 case 的整卡峰值。`-l 1` 的 1 s 粒度会漏掉
更短的尖峰，所以**结论要留安全余量，不要卡到 16384 MiB 的最后几十 MiB**。
仓库里目前没有这个采样脚本，建议顺手落成 `scripts/sample_gpu_during_run.sh`。

### 成对提交，不比历史数字

```bash
./scripts/start_comfyui.sh restart          # 需要改环境变量时
/home/regen/minimax-h3/.venv/bin/python scripts/submit_workflow.py \
  workflows/h3-v100-multimode-qwen32-mp-832x480-124f-4step.json \
  --wait --timeout 2400 \
  --output results/<case>.json
```

同一 workflow、同一 seed、同一 scheduler、同一 page-cache 状态，A/B 前后脚跑。
不要拿不同日期的 JSON 互相比。`execution_cached` 的节点列表必须一起记录——
它决定这次是冷还是 warm，两者不可混算。

### 每项都要留下

- 阶段时间线：`qwen_load / forward / clear`、每 step DiT compute 与 collective、
  VAE video/audio decode、encode/save；
- 整卡峰值（上面的 CSV），两张卡都要；
- 主进程 RSS、cgroup、`oom` / `oom_kill` 计数（限制是 `MemoryMax=7G`）；
- latent / audio 的 finite、rank 一致性；
- 输出 SHA256，以及与基线的 relative RMS / cosine；
- `clocks.current.sm` 与 `temperature.gpu` 曲线（用来解释而不是掩盖 skew）。

## 4. 统一验收门禁

任何一项进默认前，全部满足：

- [ ] `payload_mmap = false`，两个 rank 的模型 payload maps 为 0；
- [ ] `oom = 0`、`oom_kill = 0`，RSS 无异常尖峰；
- [ ] 两 rank 输出一致，全部中间与最终 latent finite；
- [ ] FP32 residual、out_proj/FC2 FP32 输出、FP32 NCCL 保留；
- [ ] Turbo LoRA 未被 bypass；
- [ ] 448×256、832×480、1 MP 三档都留了 latent、MP4、audio stats、profile、SHA256；
- [ ] 整卡峰值（nvidia-smi）两张卡都留 ≥1 GiB 余量；
- [ ] 目标尺寸的 **wall-time** 有收益，不接受只有 microbenchmark；
- [ ] 失败可回退到当前路径，不留 NCCL 孤儿进程；
- [ ] 声称"数值不变"的项（P0-a）必须 `max_abs == 0.0`，不接受 cosine 近似。

## 5. 明确不做

- 不为测 kernel 并行起第二个完整 ComfyUI / TP 服务；
- 不恢复 host mmap 或完整 CPU 权重副本；
- 不用 FP8 / NVFP4 / BF16 TC / Ampere-only 指令；
- 不用 `DataParallel`、普通 `device_map` 冒充 TP；
- 不再扫 Group Cache 全局阈值（第 1 节）；
- 不再投入自定义 attention kernel（第 1 节）。
