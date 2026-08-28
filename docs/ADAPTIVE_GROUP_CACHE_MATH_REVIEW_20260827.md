# H3 Adaptive Group Residual Cache：数学审查与下一步候选

日期：2026-08-27

这是一份对当前实现和已有测量的审查记录，不修改
[`TP_SPEED_FUTHUR.MD`](TP_SPEED_FUTHUR.MD) 的第一阶段范围，也没有在本次审查中修改
Qwen text-encoder、服务或默认推理路径。

## 先给结论

Group Cache 的方向是对的：在现有很小的 oracle 样本中，group-input feature
difference 对局部 residual reuse error 的排序能力明显优于单独的 `Δsigma`。但当前
`feature_error < 一个全局 threshold` 还不足以成为质量门禁。

原因不是简单“阈值太小”。对一个 Group，真实函数应写成：

```math
F_g(x, c) = x + r_g(x, c)
```

其中 `x` 是 Group 输入，`c` 是 H3 的 timestep/AdaLN modulation 状态。当前 cache
预测为：

```math
\hat F_g(x_1, c_1) = x_1 + r_g(x_0, c_0)
```

所以局部误差是：

```math
e_g
=
r_g(x_0, c_0) - r_g(x_1, c_1)
\approx
-J^x_{r,g}\Delta x
-J^c_{r,g}\Delta c
```

当前规则只测量了量化后的 `Δx`，没有测量 Group 特有的局部敏感度
`J^x_{r,g}`，也没有覆盖 `Δc`。而 H3 的 `c` 正是每个 Block 的 AdaLN
shift/scale/gate；它随 video/audio timestep 改变，不能由 prompt 是否变化来代替。

因此，优先级不是继续把全局 threshold 从 `0.3` 调得更高，而是：

1. 让判定变成 group-specific、conditioning-aware 的风险估计；
2. 去掉为判定而保存/反量化整张 Q4 `previous_input`；
3. 用总误差预算选择哪些 Group cache，而不是四个 Group 各自拿同一阈值赌运气。

## 已有数据说明了什么

### 小尺寸阈值矩阵

以下为 `448×256 / 4 step / S=868` 的 DiT-only 统计。它是算法 smoke，不是最终视频
质量结论。

| 配置 | DiT 合计 | 相对 Full | 跳过 block visits | video relative RMS | audio relative RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 2123.690 ms | 1.000× | 0 | 0 | 0 |
| Group `t=0.005` | 2255.015 ms | 0.942× | 0 | 0 | 0 |
| Group `t=0.30` | 1834.056 ms | 1.158× | 42 | 0.4538 | 0.8905 |
| Group `t=0.40` | 1384.099 ms | 1.534× | 84 | 0.6364 | 0.8492 |
| Group `t=0.60` | 1364.268 ms | 1.557× | 84 | 0.7795 | 0.9450 |

证据：
[`results/h3_q4_group_matrix_448x256_20260827/matrix_summary.json`](../results/h3_q4_group_matrix_448x256_20260827/matrix_summary.json)

这有两个直接含义：

- `t=0.005` 没有命中 cache 时仍比 Full 慢约 `6.2%`；目前整张输入 Q4
  quantize、CPU/GPU 传输、dequant 和 metric reduction 本身有可见成本。
- 提高 threshold 的确能加速，但目前质量偏差先急剧增大，不是可直接上线的
  quality–speed Pareto。

### oracle 反例：小 feature delta 不等于安全

同一条 `448×256` oracle run 的 step 2，四个 Group 全部因 `threshold=0.3` 被判为
CACHE。`output relative L2` 是在该 Group 当前输入上额外执行真实 FULL 得到的局部误差。

| Group | feature error | output relative L2 | residual relative L2 |
| --- | ---: | ---: | ---: |
| 0 (`[8,18)`) | 0.2608 | 0.0934 | 0.1899 |
| 1 (`[18,28)`) | 0.2374 | 0.2128 | 0.3067 |
| 2 (`[28,38)`) | 0.1660 | 0.3794 | 0.4455 |
| 3 (`[38,50)`) | 0.1280 | 0.3839 | 0.3898 |

后两个 Group 的入口变化反而更小，但 residual/output 误差最大。这正是
`||J^x_{r,g}||` 不同、以及 AdaLN 状态未纳入判定的表现；单一全局 threshold 无法把
Group 0 和 Group 3 排成正确的安全顺序。

证据：
[`results/h3_q4_group_oracle_448x256_20260827/oracle_summary.json`](../results/h3_q4_group_oracle_448x256_20260827/oracle_summary.json)

在这个很小、同一 prompt 的 `n=12` oracle 样本中：

| predictor | Pearson(feature/sigma, local output error) | Spearman |
| --- | ---: | ---: |
| `Δsigma` | 0.479 | 0.325 |
| 当前 feature difference | 0.714 | 0.734 |

这只能说明 feature 有信息量，不能作为跨 prompt 的最终 claim；样本仍需按设计文档扩展到
多 prompt、多 seed、多个 step 数。

### Q4 的误差地板不能忽略

同一 oracle FULL refresh 中，输入 `previous_input` 的 Q4 reconstruction relative L2
约 `0.070–0.095`，residual Q4 reconstruction relative L2 约 `0.043–0.118`，且更深
Group 会出现 restore exponent。也就是说：

```text
feature Q4-vs-Q4 完全相同 = 0
```

并不代表：

```text
把 Q4 residual 加回去的输出误差 = 0
```

后者仍有 residual quantization floor。后续 risk score 要把“残差复用变化造成的误差”和
“Q4 residual 自身误差”分开记录，不能把全部误差归因于 threshold。

## 当前 `previous_step` 需要加一个约束

现有实现在 `reference_mode=previous_step` 命中 cache 后会更新：

```text
previous_input = current_group_input
```

但 `residual` 仍然是最近一次 FULL 的 residual。因此下一步测到的是：

```math
||x_t - x_{t-1}||
```

而真正要判断的 residual 失配仍然相对于：

```math
r(x_{\mathrm{last\ full}}, c_{\mathrm{last\ full}})
```

这使 `previous_step` 成为一种局部变化 heuristic，而不是和 cached residual 同一锚点的
安全判据；连续命中时它可能低估 residual age。

当前建议：

- 生产/主 benchmark 继续把 `last_full` 作为语义正确的 reference；
- `previous_step` 只保留作 ablation；
- 若以后保留它，额外保存一个很小的 `last_full` feature signature，并维护累计风险，至少
  同时约束 local delta、anchor delta 和 cache age。

## 值得做的数学优化，按优先级排序

### P1：Group-specific calibration，而不是全局 threshold

在每次 FULL refresh，旧 residual 仍可按 chunk 读取，新 residual 尚未释放。此时无需保存
FP32 activation，就可以测量：

```math
d^r_g
=
\frac{\|r_g^{new}-r_g^{old}\|_2}
{\|r_g^{old}\|_2+\epsilon}
```

并用稳健上界而不是均值更新每个 Group 的局部敏感度：

```math
\kappa^x_g
\approx
\operatorname{EMA}_{q90}
\left(
\frac{d^r_g}{d^x_g+\epsilon}
\right)
```

第一个可用 score 可以是：

```math
\hat e_g
=
q_g
+ \kappa^x_g d^x_g
+ \kappa^c_g d^c_g
```

其中 `q_g` 是该 Group 最近一次 FULL 时测得的 Q4 residual floor，`d^x_g` 是输入特征变化，
`d^c_g` 是下面的 AdaLN conditioning 变化。使用保守分位数/clip 的目的是避免偶然一次
很稳定的 refresh 把后续 cache 放得过松。

这不是 learned predictor：只是一组在线标量校准，仍可完全 opt-in、fail closed。

### P2：把 H3 的 AdaLN 变化直接纳入判定

对 H3，更正确的函数是 `F_g(x, c)` 而非 `F_g(x)`。`c` 已经作为很小的 `t_emb` 传入，
每个 Block 的 AdaLN 也是小输入维度到 `shift/scale/gate` 的线性/低秩映射。可为每个
Group 计算：

```math
d^c_g
=
\left(
\sum_{\ell\in g}
\sum_{m\in\{\mathrm{context,audio,video}\}}
w_m
\frac{\|A_{\ell,m}(c_t)-A_{\ell,m}(c_{full})\|_2^2}
{\|A_{\ell,m}(c_{full})\|_2^2+\epsilon}
\right)^{1/2}
```

`A` 可以先取各 Block 的 AdaLN gate；后续再加入 shift/scale。其大小只和几十个
`[hidden]` modulation vector 有关，不随 1 MP token 数增长，远小于一次完整 Group
feature dequant。

更 H3-specific 的二阶候选是：在 FULL 时为每个 Block/segment 保存 attention 和 MLP
update 的 channel-RMS。cache 时用：

```math
\|\Delta gate \odot \operatorname{RMS}(u_{old})\|_2
```

估计 `\Delta(gate\odot u)` 的已知一阶项。这比仅看 `Δsigma` 或全局 hidden norm 更贴近
真正的 residual 更新。

### P3：用小 signature 取代整张 Q4 `previous_input`

目前每个 Group 都持有：

```text
Q4 previous_input + Q4 residual
```

在 `S=37746, H=5376` 的 1 MP 几何中，这相当于每个 rank 约：

| 持久项 | CPU cache |
| --- | ---: |
| 4 个 Q4 residual | 456 MiB |
| 4 个 Q4 previous_input | 456 MiB |
| 当前 Group Cache 合计 | 913 MiB |

`previous_input` 只服务于判定，不参与生成。第一版 bounded signature 实验已经落地，
实现/CPU 契约与当前边界见
[`ADAPTIVE_GROUP_CACHE_SIGNATURE_EXPERIMENT_20260827.md`](ADAPTIVE_GROUP_CACHE_SIGNATURE_EXPERIMENT_20260827.md)。
它保留 Q4 residual 输出路径，使用按 packed segment 分层的 CPU FP32 signature 替代
`previous_input`，默认仍关闭。后续仍需验证：

- context/audio/video 各自独立；
- target video token 的 relative-L2 p95/p99，而不是全局 mean；
- deterministic token/hidden samples 或小型 Rademacher sketch；
- 保留 last-full signature，必要时另保留 previous-step signature。

这样可以同时：

- 将持久 cache 从约 `913 MiB/rank` 降至接近 `456 MiB/rank`；
- 消除 `Q4(current) -> CPU -> GPU dequant -> metric` 的主要判定开销；
- 避免 Q4 feature quantization 对判定尺度的干扰；
- 让保守无命中场景不再比 Full 慢约 6%。

最稳妥的切入方式是 two-stage gate：cheap signature 明显不安全就直接 FULL；只有接近边界
的样本才运行更精细的 metric。最终是否完全取消 Q4 input metric，应由 oracle correlation
验证决定。

### P4：用总误差预算做串行决策

Group 不是可并行的独立分支。前面 Group 的误差会被后面真实 Group 的 Jacobian 传播：

```math
E_{final}
\lesssim
\sum_g w_g e_g,
\qquad
w_g \approx \prod_{j>g} L_j
```

`w_g` 不必一开始精确求 Jacobian；可先由 oracle/full 对照离线估计一个保守的 group weight。
运行时目标改为：

```text
在总风险预算 B 内，最大化跳过的 block cost
```

而不是：

```text
每个 Group 都用同一 threshold 独立决定
```

实践上可先使用按 Group 的风险余额：

```text
risk_used += w_g * predicted_error_g
只有 risk_used <= B 时才允许 CACHE
```

它也自然替代固定、过粗的 `max_cache`；`max_cache` 可保留为硬上限。

### P5：静态但不等长的 Group 边界

当前 `[8,18), [18,28), [28,38), [38,50)` 是按 block 数均分，不是按：

```text
预测误差 / 节省时间 / 下游放大
```

均衡 Q4 cache 内存的前提下，可固定仍为 4 Group，但离线搜索不等长连续边界。目标是把
敏感、下游放大大的 Block 放进更小的 Group，把稳定区间合并成较长 Group。这样不需要动态
partition，也不会增加 `2 × group_count` 个完整 cache payload。

### P6：残差的标量校正与 secant（第一阶段通过后再做）

当前等价于固定：

```math
\hat r_t = r_{old}
```

最小增量是先测试一个有界的标量修正：

```math
\hat r_t = \alpha_g(t) r_{old}
```

随后才考虑：

```math
\hat r_t
=
r_{old}
+ \operatorname{clip}(\alpha, -\alpha_{max}, \alpha_{max})
(r_{old}-r_{older})
```

它可能修正 residual 的整体幅值漂移，但不能自动修正方向改变。必须先和 static residual
在相同 quality–speed 点比较；不能默认启用。

一个有利的内存事实是：若 P3 已用 signature 替换 `previous_input`，释放出的一个 Q4 payload
可用于保存 `r_older`，不必增加当前总 cache 内存。

### P7：仅在 Q4 floor 成为瓶颈时重分配精度预算

这不是当前默认方案。若 oracle 证明“即使 `Δx≈0`，Q4 residual floor 仍主导误差”，不要把
所有 cache 又改回 FP32。先考虑：

```text
小 signature + 一份 Q5/Q6 residual
```

而不是：

```text
Q4 previous_input + Q4 residual
```

前者理论上仍可能少于当前两份 Q4 的总 CPU 内存，却把精度优先给真正影响输出的 residual。
这需要独立精度/速度测量，当前仍遵守 Q4_0 默认。

## 不值得作为下一步的方向

- 只继续提高统一 threshold：已有 oracle 已显示它会把低 feature error 的深层 Group 错判为
  安全。
- 并行执行 Group：Group 间有 hidden-state 依赖，会改变网络语义。
- 为了判定保留完整 FP32 input/residual cache：会重新触发 1 MP VRAM/RAM 压力。
- 直接上完整 Jacobian/JVP：对这个长序列 Transformer 的额外计算和内存与“跳过 block”的
  目标相冲突。
- attention KV cache：跨 denoise step 的 Q/K/V 输入仍变化，且显存代价极高；不适合本阶段。

## 建议的受控实验顺序

1. 保持现有 runtime 不变，先扩展 oracle logging：记录 `d_x`、`Δsigma`、`d_c`、Q4
   floor、真实 residual drift 与最终 DiT error。
2. 在原有 prompt/seed 矩阵上比较 `Δsigma`、`d_x`、`d_c` 和组合 score 的
   Spearman、safe-cache ROC-AUC/PR-AUC。
3. 只有组合 score 明显优于当前 `d_x` 后，再实现 opt-in 的 group-specific risk gate。
4. 以现有 opt-in signature 路径重新测试无命中开销、1 MP CPU RSS、peak VRAM 和
   quality–speed Pareto；在 oracle/holdout 证明前不改默认 `q4` feature gate。
5. 最后才做不等长边界与 residual secant；每项单独做 full-reference 视频比较。

这样能回答三个最重要的问题：

```text
1. Feature + AdaLN 是否真的比 Δsigma 更会判断安全 cache？
2. Group-specific 风险能否在相同速度下减小漂移？
3. 去掉整张 previous_input 后，是否既降低 RAM 又让 cache 的净加速更稳定？
```
