# PPO 指导：标准 Actor-Critic 在线策略优化

版本：v0.1 \| 对应 Notebook：`05_PPO`

## 1. 定位

本项目将 PPO 作为标准在线 RL 教学基线，由我们自行设计和实现。

已研究的 Nemotron 头部选手报告中，没有找到一套可直接作为本教程标准 PPO
复现模板的成熟成功方案。The Submitter 的主要有效 RL 结果来自 DPO；第 251
名和第八十名重点是 GRPO 的成功与失败分析。因此，本模块不是复刻某位选手的
PPO，而是结合比赛任务、Verifier 和硬件约束实现标准 PPO。

## 2. 核心组件

-   **Actor**：生成回答的策略模型。
-   **Critic**：估计状态价值。
-   **Old Policy**：生成当前 Rollout 的行为策略。
-   **Reference Policy**：可选的固定 SFT 参考策略，用于 KL 约束。
-   **Reward**：由确定性 Verifier 计算。

为控制显存，优先测试共享 Qwen3-4B Backbone + Scalar Value Head 的
Actor-Critic 结构。是否共享 LoRA 参数，需要通过梯度和稳定性测试确认。

## 3. PPO 核心计算

### 3.1 Probability Ratio

``` latex
r_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
r_t(theta) = exp(log pi_theta(a_t | s_t) - log pi_old(a_t | s_t))
```

### 3.2 PPO-Clip

``` latex
L_CLIP(theta) = E_t[min(r_t(theta) * A_t,
    clip(r_t(theta), 1-epsilon, 1+epsilon) * A_t)]
```

训练代码通常最小化负的 Surrogate Objective。

### 3.3 GAE

``` latex
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * delta_{t+1} + ...
```

必须正确处理 EOS、Padding、Response Mask 和终止状态。

### 3.4 Value Loss

``` latex
L_V = E_t[(V_phi(s_t) - R_hat_t)^2]
```

### 3.5 总目标

``` latex
L = L_policy + c_V * L_V - c_H * H(pi_theta)
```

如果使用 KL-to-Reference，可将 KL 惩罚加入 Reward
或单独加入目标函数。实现时必须说明采用哪一种方式，避免重复惩罚。

## 4. Rollout 与训练流程

采用同步串行执行：

1.  固定当前 Actor Policy。
2.  使用 vLLM 生成回答。
3.  保存 Token IDs、Attention Mask、Response Mask 和行为策略 Log
    Probability。
4.  使用 Verifier 计算 Reward。
5.  使用 Critic 计算 Value。
6.  计算 GAE 和 Return。
7.  执行 PPO 多轮小批量更新。
8.  保存 Adapter。
9.  释放训练进程。
10. 使用更新后的 Adapter 进入下一轮 Rollout。

vLLM 与训练使用同一
GPU，但不同时占用显存。必须确保生成时使用的策略分布与保存的行为 Log
Probability 一致。Temperature、Top-p
等采样变换会影响实际行为分布，不能忽略。

## 5. Reward 设计

首版优先使用确定性 Outcome Reward：

``` text
R_final = 1 if answer is correct else 0
```

Equation Numeric 和 Bit Manipulation 使用任务专用
Verifier。可选地加入确定性 Process
Reward，但每个中间状态都必须有程序验证依据，且最终答案正确性保持主导。

Gravity 通常由 SFT 就能解决，不强制进入 PPO。

## 6. Critic 设计与显存

优先测试共享 Backbone + Scalar Value Head：Backbone 使用 Qwen3-4B；Actor
通过 LoRA / QLoRA 更新；Value Head 输出每个生成位置的标量价值；保存 Old
Policy Log Probability 和 Old Value；训练时计算 Policy Loss 与 Value
Loss。

需要测试 Value Head
是否能稳定学习。若共享参数导致策略和价值目标冲突，再考虑分离 Critic
Adapter。

## 7. PPO 与 GRPO 的区别

PPO 使用 Critic 和 GAE 估计 Advantage。它不要求每个 Prompt 的 Rollout
Group 必须同时出现正确和错误回答。

但如果某类题目的所有轨迹都得到相同终端 Reward，且 Critic 也只能预测相同
Return，该题的策略梯度仍可能缺少有效区分信号。不能将 PPO
的问题简单等同于 GRPO 的组内零方差问题。

## 8. 训练诊断

记录 Mean Reward、Pass@1、Policy Loss、Value Loss、Approx
KL、Entropy、Clip Fraction、Explained Variance、Advantage Mean /
Std、平均 Completion Length、训练耗时和峰值显存。

如果 Value Loss 下降但准确率没有提高，不能据此声称策略能力提升。

## 9. 实验设计

-   PPO-A：Binary Outcome Reward。
-   PPO-B：Outcome + 确定性 Process Reward（选做）。
-   PPO-C：有无 KL-to-Reference 对照（选做）。

所有 PPO 实验从同一个最佳 SFT Checkpoint 出发。首轮只执行少量 Rollout /
Update，先验证数值正确性、梯度方向和显存，再扩大训练。

## 10. 主要风险

-   行为策略 Log Probability 不匹配；
-   Old Policy 没有冻结；
-   Critic Value 与 Response Token 对齐错误；
-   GAE 的终止状态和 Padding Mask 错误；
-   KL 惩罚重复计算；
-   Critic 过拟合；
-   训练更新后 vLLM 没有同步最新 Adapter；
-   将 Reward 上升误判为答案准确率上升。

## 11. 交付物

手写 PPO-Clip Loss、GAE 实现、Actor-Critic 结构、Rollout 数据结构、PPO
训练日志、SFT vs PPO 分任务结果、显存与耗时报告。

## 12. 参考资料

-   [Datawhale PPO
    Notebook](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/16_RLHF_PPO_Memory.ipynb)
-   [Datawhale LeetCode
    仓库](https://github.com/datawhalechina/llm-algo-leetcode)
-   [第 251 名 GRPO
    实验](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/trace-format-engineering-and-svd-downsampled-lora)
-   [第八十名 GRPO
    失败分析](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/breaking-the-sft-ceiling-on-nemotron-3-nano-30b)

说明：比赛报告用于说明任务、Verifier 和 RL 训练的工程约束；本模块的 PPO
算法采用标准 Actor-Critic PPO / GAE 设计，不声称来自某个获奖选手的 PPO
实现。

## 13. 本轮实现核对：Value 与 Action 时间轴

Notebook 05 采用如下明确约定：隐藏状态位置 `t` 是预测 token `t+1` 的动作前状态。于是 `token_logprobs` 的动作轴是 `logits[:, :-1]`，Value 轴是 `values[:, :-1]`，Reward、Done 和 Response Mask 对齐到 token 位置 `[:, 1:]`。GAE、Return、Policy Loss、Value Loss、Entropy 和 Approx KL 均使用这个同一动作轴；Prompt、Padding 和 EOS 后位置不参与计算。

Rollout 的行为 Log Probability 必须与 vLLM 实际采样 Token IDs 一一对应；若缺失或长度不一致则 fail closed。截断回答没有观察到 EOS，Verifier 不给正确奖励。CPU Smoke 增加短序列 Value/Token 对齐断言；真实 Qwen3 PPO、Value Head 训练和显存仍为 `Not Run`。

本 Notebook 当前只记录 Old Policy Logprob 的 Approx KL 诊断，未实现 Reference Policy KL 正则；不能将该诊断解释成 Reference KL。
