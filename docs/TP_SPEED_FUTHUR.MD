# MiniMax H3 Adaptive Group Residual Cache

## 1. 目标

基于现有 TE-Speed MiniMax H3 residual cache 思路，实现一个更细粒度的 **Group Residual Cache**。

现有 TE-Speed 可以抽象为：

```text
Block 0~K      每个 step 重算
Block K+1~N    作为一个整体进行 residual cache
```

一次 FULL step 保存：

```math
R_{tail} = h_N - h_K
```

CACHE step：

```math
\hat h_N = h_K^{new} + R_{tail}^{old}
```

问题是尾部几十个 Transformer Blocks 被视为一个整体：

```text
要么全部重算
要么全部 cache
```

但不同深度的 Block 对 timestep / hidden-state 变化的敏感度很可能不同。

本项目第一阶段验证以下假设：

```math
不同 Block Group 的 residual 稳定性不同
```

因此，将尾部划分为多个 Group，并分别决定是否 cache，有机会在相同质量下跳过更多计算，或者在相同计算量下减小误差。

---

# 2. 第一版范围

第一版只实现：

**Group Residual Cache + Group-level Feature Difference**

暂时不要实现：

* residual extrapolation
* Taylor prediction
* token-wise cache
* spatial cache
* learned predictor
* attention KV cache
* dynamic group partition

目标是先验证：

> “分段以后，根据每一段入口 hidden state 的变化分别决定 cache，是否优于 TE-Speed 的整段 cache。”

---

# 3. Block 分组

假设 MiniMax H3 有 50 个 Transformer Blocks。

默认配置：

```text
Block 0~7    Always Compute

Block 8~17   Group A
Block 18~27  Group B
Block 28~37  Group C
Block 38~49  Group D
```

也可以支持参数：

```text
warm_blocks = 8
num_groups = 4
```

然后自动均分剩余 Blocks。

要求：

* warm prefix 永远完整计算；
* 每个 Group 独立维护 cache；
* 每个 Group 可以独立选择 FULL 或 CACHE；
* 每个 Group 有自己的 residual；
* 每个 Group 有自己的 previous input feature。

---

# 4. Residual 定义

对于 Group \(g\)：

```text
Group 输入：
h_g_in

完整计算 Group：
h_g_out = F_g(h_g_in)
```

定义 residual：

```math
R_g = h_{g,out} - h_{g,in}
```

FULL 时：

```text
h_in
 ↓
完整运行当前 Group
 ↓
h_out

保存：

previous_input[g] = h_in
residual[g]       = h_out - h_in
```

CACHE 时：

```math
\hat h_{g,out}
=
h_{g,in}^{new}
+
R_g^{old}
```

因此不需要保存 Group 内每层 activation。

每个 Group 只需要保存：

```text
previous_input
residual
```

---

# 5. Group Cache 判定

第一版不要使用 cosine similarity。

使用 normalized relative difference。

推荐默认：

```math
E_g =
\frac{
\operatorname{mean}(|h_g^{new}-h_g^{old}|)
}{
\operatorname{mean}(|h_g^{old}|)+\epsilon
}
```

其中：

```text
epsilon = 1e-6
```

也实现 L2 版本供 benchmark：

```math
E_g =
\frac{
\|h_g^{new}-h_g^{old}\|_2
}{
\|h_g^{old}\|_2+\epsilon
}
```

节点参数：

```text
metric:
    relative_l1
    relative_l2

threshold:
    float
```

如果：

```math
E_g < threshold
```

则允许 CACHE。

否则 FULL。

---

# 6. 为什么不用 cosine

不要仅使用：

```math
cos(h_{new},h_{old})
```

因为：

```math
h_new = 1.2 h_old
```

时 cosine 仍然等于 1，但 magnitude 已经变化 20%。

此外，大面积静态 background tokens 可能掩盖少数高速运动 token 的变化。

因此第一版优先使用 relative L1 / L2。

---

# 7. Cache 执行流程

伪代码：

```python
h = input

# Always-compute prefix
h = run_blocks(h, 0, warm_blocks)

for group in groups:

    group_input = h

    # 没有历史 cache 时一定 FULL
    if group.prev_input is None or group.residual is None:
        group_output = run_group(group_input)

        group.prev_input = group_input.detach()
        group.residual = (
            group_output - group_input
        ).detach()

        h = group_output
        continue

    error = relative_difference(
        group_input,
        group.prev_input,
    )

    if error < threshold and group.cache_count < max_cache:
        # CACHE
        h = group_input + group.residual
        group.cache_count += 1

    else:
        # FULL
        group_output = run_group(group_input)

        group.prev_input = group_input.detach()
        group.residual = (
            group_output - group_input
        ).detach()

        group.cache_count = 0

        h = group_output
```

注意：

即使 CACHE，也需要更新：

```text
prev_input
```

还是不更新，需要作为实验变量验证。

第一版建议同时实现两种模式：

```text
reference_mode = last_full
reference_mode = previous_step
```

### last_full

CACHE 时不更新 `prev_input`：

```text
所有 feature difference
都相对于最近一次 FULL 的输入
```

优点：

误差不会因为连续 cache 的近似状态不断漂移。

### previous_step

CACHE 时：

```text
prev_input = current_group_input
```

下一个 step 和最近一步比较。

优点：

更能反映局部 step-to-step 变化。

两者都应该 benchmark。

---

# 8. 保留 max consecutive cache

即使 feature difference 很小，也不要无限 cache。

参数：

```text
max_cache = 1 / 2 / 3 ...
```

例如：

```text
FULL
CACHE
CACHE
FULL
```

用于限制 residual 老化。

第一版默认：

```text
max_cache = 2
```

这样可以直接和 TE-Speed 的 mcs=2 比较。

---

# 9. 与 TE-Speed baseline 的严格对比

至少测试三种模式。

## Baseline A — Full Compute

完全关闭 cache。

```text
所有 Blocks
每个 denoise step 都完整计算
```

这是质量 reference。

---

## Baseline B — TE-Speed-style Whole Tail Cache

保持现有方式：

```text
warm prefix
+
整个 tail residual cache
```

使用相同：

```text
warm block 数
max consecutive cache
采样器
scheduler
seed
prompt
resolution
steps
LoRA
```

---

## Proposed — Group Cache

```text
warm prefix
+
4 Group adaptive residual cache
```

---

# 10. 不能只比较最终视频

必须分三层验证。

---

# 11. 第一层：Block/Group 层面的直接误差

这是最重要的算法验证。

在 benchmark 模式下，即使算法决定 CACHE，也额外偷偷运行一次真实 FULL Group，仅用于测量 ground truth。

正常运行：

```text
Group input
 ↓
CACHE prediction
 ↓
predicted_output
```

Benchmark 模式额外：

```text
同一个 Group input
 ↓
真正 FULL Group
 ↓
true_output
```

计算：

```math
E_{output}
=
\frac{
\|\hat h_{out}-h_{out}^{full}\|_2
}{
\|h_{out}^{full}\|_2+\epsilon
}
```

同时记录：

```math
E_{residual}
=
\frac{
\|R_{cached}-R_{true}\|_2
}{
\|R_{true}\|_2+\epsilon
}
```

这两个指标可以直接回答：

> cached residual 到底近不近？

以及：

> feature difference 能不能预测 residual error？

---

# 12. 验证核心假设：Feature Difference 是否有预测能力

对所有 step、所有 Group 记录：

```text
feature_error
true_residual_error
group_id
step
sigma
cache/full
```

然后画散点：

```text
X = feature difference

Y = true residual approximation error
```

如果我们的 insight 成立，应该看到明显：

```text
feature difference 小
        ↓
residual error 也通常小
```

计算：

```text
Pearson correlation
Spearman rank correlation
```

其中更建议关注：

```text
Spearman
```

因为不要求线性关系。

如果：

```text
Spearman(feature_delta, residual_error)
```

明显为正，并且稳定跨 prompt / seed：

说明：

> front/group-input feature change 确实比随机规则更能预测 cache 是否安全。

---

# 13. 与 sigma-delta 的预测能力直接比较

这一步非常关键。

对于相同的真实 residual error：

记录：

```text
X1 = abs(delta_sigma)
X2 = feature_difference
Y  = true_residual_error
```

分别计算：

```text
corr(delta_sigma, residual_error)

corr(feature_difference, residual_error)
```

如果：

```text
feature_difference
```

对真实 residual error 的相关性明显强于：

```text
delta_sigma
```

这就是第一个很强的证据：

> 我们的 cache criterion 比 TE-Speed 的 sigma heuristic 更能预测 tail residual 是否还能复用。

还可以把问题变成二分类：

定义：

```text
safe cache =
true_residual_error < target_error
```

例如：

```text
target_error = 0.01
```

然后分别用：

```text
delta sigma
```

和：

```text
feature difference
```

预测 safe / unsafe cache。

计算：

```text
ROC-AUC
PR-AUC
```

如果 feature-based classifier 的 AUC 更高，就是非常直接的证明。

---

# 14. 第二层：整个 DiT 输出误差

每个 denoise step 结束后，对比：

```text
Full Compute DiT output

vs

Cached DiT output
```

计算：

```math
E_{DiT}
=
\frac{
\|y_{cache}-y_{full}\|_2
}{
\|y_{full}\|_2+\epsilon
}
```

同时记录：

```text
mean relative L1
relative L2
cosine similarity
95th percentile token error
maximum group error
```

这里特别推荐：

```text
p95 token error
```

而不是只看全局 mean。

因为人物手、脸等少量 token 可能错误非常大，但被大量背景 token 平均掉。

---

# 15. Token-level error

把 hidden tensor 沿 token 维计算：

```math
e_i
=
\frac{
\|h_i^{cache}-h_i^{full}\|_2
}{
\|h_i^{full}\|_2+\epsilon
}
```

统计：

```text
mean
median
p90
p95
p99
max
```

重点看：

```text
p95 / p99
```

如果 Group Cache 的 mean 跟 TE-Speed 类似，但：

```text
p95 / p99
```

显著更低，

说明它对局部高速变化区域更稳。

---

# 16. 第三层：最终视频质量

所有方法必须：

```text
相同 prompt
相同 seed
相同初始 latent
相同 LoRA
相同 scheduler
相同 sampler
相同 resolution
相同 frame count
```

Full Compute 输出作为 reference。

对缓存输出进行 reference-based comparison。

推荐至少：

```text
LPIPS
SSIM
PSNR
```

但 LPIPS 权重最高。

原因：

我们不是在比较“哪个随机生成的视频更漂亮”，而是在比较：

> cache 后是否尽量保持 Full Compute 原来的生成结果。

因此 reference-based metric 很合适。

---

# 17. Temporal Quality

视频 cache 最容易出现：

```text
flicker
运动边缘错误
人物局部漂移
```

所以不能只算逐帧 LPIPS。

至少增加：

```text
Temporal LPIPS
```

简单版本：

对相邻帧：

```math
D_t = LPIPS(frame_t, frame_{t+1})
```

然后比较 Cache 和 Full 的 temporal-change curve：

```math
E_{temporal}
=
mean(
|D_t^{cache}-D_t^{full}|
)
```

更高级版本可以之后加入：

```text
optical-flow warped error
```

但第一版不是必须。

---

# 18. 性能指标

必须记录：

```text
Total wall-clock time

DiT-only wall-clock time

average seconds / denoise step

number of executed blocks

number of skipped blocks

cache hit rate

cache hit rate per Group

peak VRAM

cache tensor memory
```

最重要的是：

```math
Speedup =
\frac{T_{full}}{T_{method}}
```

---

# 19. 不要比较单个参数点

真正证明我们比 TE-Speed 好，必须画：

# Quality–Speed Pareto Curve

对于 TE-Speed：

测试多个 threshold：

```text
0.02
0.04
0.06
0.08
0.10
0.12
...
```

对于 Group Cache：

也扫：

```text
0.005
0.01
0.02
0.03
0.04
...
```

具体范围根据实际 feature-difference 数量级自动调整。

每个点得到：

```text
X = speedup
Y = quality degradation
```

例如：

```text
X = Speedup

Y = final video LPIPS vs Full
```

理想结果：

```text
LPIPS error
↑
│          TE-Speed
│        •
│      •
│    •
│
│        Group Cache
│   •  •   •
│
└────────────────→ Speedup
```

也就是：

> Group Cache 的 Pareto frontier 整体位于 TE-Speed 下方。

这才是最有说服力的结果。

---

# 20. 什么结果才可以说“我们更好”

不要只说：

```text
我们的 LPIPS 更低
```

或者：

```text
我们快 30%
```

正确的 claim 应该是：

## Case A — Same Speed

例如：

```text
TE-Speed:
1.25x speedup
LPIPS = 0.040

Group Cache:
1.25x speedup
LPIPS = 0.021
```

则：

> At matched 1.25× speedup, Group Cache reduces perceptual deviation from the full-compute baseline by 47.5%.

---

## Case B — Same Quality

例如：

```text
TE-Speed:
LPIPS = 0.025
speedup = 1.17x

Group Cache:
LPIPS = 0.025
speedup = 1.31x
```

则：

> At matched perceptual quality, Group Cache provides 1.31× speedup versus 1.17× for TE-Speed.

---

## Case C — Better Cache Decision

例如：

```text
sigma-delta predicting safe cache:
ROC-AUC = 0.62

feature-delta:
ROC-AUC = 0.84
```

则：

> Group-input feature change is a substantially stronger predictor of residual reuse error than sigma distance.

这个结果甚至比最终视频 benchmark 更能证明算法 insight。

---

# 21. Benchmark Dataset

不要只跑一个 prompt。

第一阶段建议至少：

```text
20 prompts
×
3 seeds
=
60 clips
```

场景覆盖：

```text
1. 静态自然风景
2. 缓慢运镜
3. 人物走路
4. 人物高速奔跑
5. 大幅肢体动作
6. 快速镜头移动
7. 粒子 / 水 / 火
8. 复杂多人运动
```

这样可以确认：

```text
Group Cache
```

不是只对静态视频有效。

---

# 22. 4-Step 是主要 Benchmark

重点测试：

```text
4-step Turbo LoRA
```

因为这是目前最值得优化的场景。

另外增加：

```text
8 step
20 step
```

作为辅助实验。

原因：

如果 feature-based cache 在 4-step 这种 sigma 间距非常大的情况下仍然有效，就比 sigma-delta heuristic 更有价值。

---

# 23. Ablation Study

第一阶段至少做：

### A. Group 数

```text
1 group    ≈ TE-Speed tail-style
2 groups
4 groups
8 groups
```

观察：

```text
speed
quality
cache memory
```

---

### B. Warm Prefix

```text
4 blocks
8 blocks
12 blocks
```

---

### C. Metric

```text
relative L1
relative L2
cosine
```

虽然预计 cosine 较差，但最好实验验证。

---

### D. max_cache

```text
1
2
3
unlimited
```

---

### E. reference mode

```text
last_full
previous_step
```

---

# 24. 必须加入 Debug Logging

每个 denoise step 输出类似：

```text
Step 2 sigma=0.42

Prefix: FULL

Group A:
feature_error = 0.011
threshold     = 0.020
decision      = CACHE

Group B:
feature_error = 0.017
decision      = CACHE

Group C:
feature_error = 0.034
decision      = FULL

Group D:
feature_error = 0.051
decision      = FULL
```

Benchmark mode 再输出：

```text
Group A true_residual_error = 0.008

Group B true_residual_error = 0.013
```

最后：

```text
Executed blocks: 132 / 200
Skipped blocks: 68
Theoretical block saving: 34%

Cache hit:
A: 2/3
B: 2/3
C: 1/3
D: 0/3
```

---

# 25. 第一阶段成功标准

满足以下三个条件，就认为值得继续：

### 1.

Feature difference 和真实 residual error 有明显相关性：

```text
Spearman > sigma-delta baseline
```

### 2.

在多个 prompts/seeds 上：

```text
Group Cache
```

的 Quality–Speed Pareto frontier 优于 Whole-tail TE-Speed。

### 3.

4-step 场景仍然能得到有意义的 cache hit，而没有明显增加：

```text
LPIPS
temporal error
p95 token error
```

---

# 26. 第二阶段再考虑

只有第一阶段验证成功后再实现：

```text
Residual linear extrapolation

R_new ≈ R_old + α(R_old - R_older)
```

以及：

```text
Taylor residual prediction
token-aware error detection
adaptive group boundary
group-specific threshold
```

第一阶段不要混入这些变量，否则无法判断到底是哪一个设计带来了提升。

---

# 核心研究问题

最终第一版代码应该回答三个问题：

```text
Q1:
不同深度 Group 的 residual 稳定性是否真的不同？

Q2:
Group 入口 hidden-state 的变化，
是否比 Δsigma 更能预测 residual reuse error？

Q3:
利用这种信息做 Group-level adaptive cache，
是否能得到更好的 Quality–Speed Pareto frontier？
```

如果 Q1/Q2/Q3 都成立，再继续做更高级的 residual prediction。
