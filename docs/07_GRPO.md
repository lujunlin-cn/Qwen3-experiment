# GRPO 指导：组内相对奖励、有效学习信号与失败诊断

版本：v0.1 \| 对应 Notebook：`06_GRPO`

## 1. 目标

复现 GRPO 核心机制，并通过 Nemotron 比赛中的正向与负向案例，教学生判断
GRPO 是否具备有效学习信号。

核心内容：Group Rollout、Verifier Reward、Group Advantage、Policy
Ratio、Clipped Policy Objective、KL 约束和 Zero-Variance Group Audit。

## 2. 成功案例：第 251 名

第 251 名先使用 Solver Trace 做 SFT，再生成多条候选回答，通过 Reward
判断结果，并使用 GRPO 对残余错误做轻量修正。作者报告 v27 SFT
验证准确率约 92.6%，一次 GRPO 后约 93.8%，绝对提升约 1.2
个百分点。作者还观察到约 38% 的 Completion Groups 存在 Reward Variance。

这些数字来自作者特定验证设置，不是 Qwen3-4B 的预期收益。

**可迁移经验：** SFT
已经掌握大部分任务，但仍能采样到不同质量回答时，GRPO
可能用于精修残余错误。

## 3. 失败案例：第八十名

第八十名对 9,500 道题分别采样 8 次，报告约 66.8% 题目为 8/8 全对、约
33.2% 为 0/8 全错，没有 1--7/8 的 Boundary Groups。作者的 GRPO 结果与
SFT 基线持平，约 0.86。

当一个 Group 的奖励全部相同时，组内中心化 Advantage 为零：

``` latex
A_i = (r_i - mean(r)) / (std(r) + epsilon)
```

如果所有奖励相同，则每条回答的中心化优势都为零，该组没有有效的相对策略更新信号。

**可迁移经验：** GRPO 需要当前策略产生不同质量的候选回答，或 Reward
能在组内区分不同轨迹。

## 4. RL Readiness Audit

诊断嵌入 GRPO Notebook，不单独创建 Notebook。对 Equation Numeric 和 Bit
Manipulation：

1.  使用最佳 SFT Checkpoint。
2.  每道题生成多条回答。
3.  使用确定性 Verifier。
4.  统计每题正确次数。
5.  分为 All-correct、Mixed、All-wrong。

  组别          定义              处理方式
  ------------- ----------------- -------------------------------
  All-correct   N/N 正确          主要用于能力保持评测
  Mixed         1 到 N-1 条正确   GRPO 优先训练池
  All-wrong     0/N 正确          检查探索、先验、SFT 和 Reward

建议首轮 N=8。Temperature 和采样数量需在 RTX 5090 上测量。报告
All-correct Ratio、Mixed-group Ratio、All-wrong Ratio、Mean Group Reward
Variance、各任务 Group 分布、Rollout Tokens 和耗时。

## 5. 零方差的理论近似

如果单次回答正确概率为 `p`，每题独立采样 `N` 次，Binary Reward 下：

``` latex
P(Mixed) = 1 - p^N - (1-p)^N
```

该公式用于解释采样数量和成功率的关系。真实模型采样可能相关，最终以实测结果为准。不能通过伪造标签或人为改写
Reward 制造 Mixed Group。

## 6. 没有 Mixed Group 时怎么处理

1.  **检查采样策略**：比较 Temperature 和 Group Size，判断是否探索不足。
2.  **检查基础能力**：检查 SFT 是否覆盖该规则、是否缺少 Drill、Trace
    是否存在不可学习步骤。
3.  **检查 Reward 分辨率**：可加入确定性 Process Reward。
4.  **回到 SFT /
    Drill**：如果模型无法产生正确轨迹，先补充可验证教师轨迹或专项
    Drill，再重新诊断。

Equation Numeric 的 Process Reward
可检查候选规则是否解释示例、全局模式是否一致、Query 计算是否正确。Bit
Manipulation 可检查表达式是否合法、是否解释全部示例、Query
输出是否正确。

Process Reward 必须由程序确定性验证，不能使用模型自评。

## 7. Reward 设计

首版使用 Binary Outcome Reward：

``` text
R_final = 1 if answer is correct else 0
```

可选扩展：

``` text
R = R_final + alpha * R_process
```

其中 `R_process` 必须是可验证的中间步骤奖励，且最终答案正确性保持主导。

第 251 名采用 Exact Match、Answer Similarity、Completion Similarity
和格式奖励的组合。我们不直接照搬所有分量，因为模糊相似度可能让错误答案获得不合理奖励。

## 8. GRPO 核心实现

对同一 Prompt 采样 `G` 条回答：

``` latex
mean_reward = sum(reward_i) / G
std_reward = sqrt(sum((reward_i - mean_reward)^2) / G)
advantage_i = (reward_i - mean_reward) / (std_reward + epsilon)
```

然后计算策略比率并使用 PPO 风格 Clipped Objective。学生手写 Group
Advantage、Ratio、Clipping 和 KL 相关计算，不使用高层 GRPO Trainer。

## 9. 本项目任务安排

  任务               GRPO 定位
  ------------------ -------------------------------------
  Gravity            通常由 SFT 解决，主要做能力保持评测
  Equation Numeric   主要 GRPO 实验
  Bit Manipulation   主要 GRPO 实验

不要强制三个任务都进入 GRPO。根据 Readiness Audit 结果选择训练池。

## 10. 实验设计

-   G0：SFT Rollout 诊断。
-   G1：Binary Outcome Reward。
-   G2：不同 Temperature / Group Size 对照。
-   G3：Deterministic Process Reward（选做）。
-   G4：训练前后能力与 Group 分布对照。

所有 GRPO 实验从同一个最佳 SFT Checkpoint 出发。如果 G1
没有提升，先判断零方差、Reward、采样、模型能力或实现问题，不能直接增加训练步数。

## 11. 训练指标

Pass@1、Mean Reward、Mixed-group Ratio、Zero-Variance Group
Ratio、Reward Variance、Policy Loss、Approx KL、Entropy、Clip
Fraction、Completion Length、峰值显存、Rollout 与 Update 耗时。

## 12. 交付物

RL Readiness Audit、Group Rollout 数据、Reward Verifier、手写 GRPO
Loss、训练日志、SFT vs GRPO 分任务对照、正向与失败案例分析。

## 13. 参考资料

-   [第 251 名：Trace Format Engineering and
    GRPO](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/trace-format-engineering-and-svd-downsampled-lora)
-   [第八十名：Breaking the SFT
    Ceiling](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/breaking-the-sft-ceiling-on-nemotron-3-nano-30b)
-   [The Submitter：GRPO / GSPO
    实验代码](https://github.com/the-submitter/nvidia-nemotron)
-   [Datawhale GRPO
    Notebook](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/18_GRPO_Loss_Tutorial.ipynb)
