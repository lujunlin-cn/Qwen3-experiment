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

Rollout 的行为 Log Probability 必须与 vLLM 实际采样 Token IDs 一一对应；若缺失或长度不一致则 fail closed。截断回答没有观察到 EOS，Verifier 不给正确奖励。CPU Smoke 增加短序列 Value/Token 对齐断言；此前 CPU 阶段的真实 Qwen3 PPO、Value Head 训练和显存标记为 `Not Run`；本轮实测见第 15 节。

本 Notebook 当前只记录 Old Policy Logprob 的 Approx KL 诊断，未实现 Reference Policy KL 正则；不能将该诊断解释成 Reference KL。

## 14. 16K NF4 QLoRA 显存维护实现

PPO **Training 仅接受 NF4 QLoRA**。Notebook 校验父 SFT Manifest 的 `mode=nf4_qlora`、4-bit NF4、Double Quant 和 BF16 Compute；不符合时直接报错。Micro-batch 固定为 1，启用 Gradient Checkpointing，关闭 KV Cache。BF16 模型仅用于只读 Rollout/Logprob 审计。Value Head 保持 FP32，只有 LoRA 参数和 Value Head 参与更新。

概率使用 `selected_logit - logsumexp(logits)`；Entropy 使用 `logZ - sum(exp(z-logZ)*z)`。两者均按序列分块进行 FP32 归约，默认 `PPO_TOKEN_CHUNK_SIZE=128`。训练路径同时 checkpoint LM Head 投影，避免先物化整个 `[1,16384,151936]` logits。冻结的 tied embedding/LM Head 以 BF16 存储，Value Head 与概率归约保持 FP32。重载检查也显式使用 BF16 autocast。

这些是维护者提供的完整辅助函数，不设置学生 TODO。GAE、Advantage 标准化、Ratio、Clip、Value Loss、Entropy 系数、Reward 和 KL 诊断均保留。每个 Micro-batch 按 `有效 Response Token 数 / 累积窗口有效 Token 总数` 加权；Entropy 只用全局有效 Token 总数归一化，不能取 Chunk Mean 的平均。

### 执行入口

先在项目目录设置父 Manifest 和独立输出版本，避免覆盖历史结果：

```bash
export PPO_SFT_MANIFEST=data/processed/checkpoints/vgpu_round1/sft_manifest.json
export PPO_RUN_ID=nf4_memory_v1
export PPO_TOKEN_CHUNK_SIZE=128
export QWEN3_MODEL_PATH=/path/to/Qwen3-4B-Thinking-2507
```

三个入口互斥，均默认关闭。在 Jupyter 中设置环境变量后，从头顺序执行 Notebook：

1. 训练 Profile（已有 bitsandbytes）：只启用 `RUN_PPO_MEMORY_TEST=1`。用真实 4B 模型、16,384 个有效 Attention Token、Micro-batch=1 测量两次前后向和梯度累积。人工拼接的输入仅用于显存机制，不是有效 Rollout；`optimizer_steps=0`。会保存、重载未更新的 Adapter/Value Head 快照，不能将快照作为训练成果。
2. 独立 vLLM 0.11 Profile：只启用 `RUN_PPO_ROLLOUT=1`。总上下文 16,384，`max_num_seqs=2`，显式 KV Cache 预算 4.5 GiB。每题 Completion 上限为 `min(RL_MAX_NEW_TOKENS, 16384 - prompt_tokens)`。Qwen3 的 36 层、8 KV Heads、Head Dim 128、BF16，在两条满长序列下理论 KV 为 `2(K/V)*36*8*128*2(bytes)*16384*2(seqs)=4.5 GiB`，不含模型和执行开销。
3. 退出 vLLM 进程并确认显存释放后，切换训练 Profile，只启用 `RUN_PPO_TRAINING=1`。保存的真实 Train Rollout 先通过来源、Token/Mask 和 Logprob 审计，且至少存在一个未截断、Verifier 正确的回答，才执行一次真实 Optimizer Update。否则明确返回 `Not Run`/阻塞状态，不制造奖励或替换行为 Logprob。

本轮不修改旧 PPO 分支的任务及数据路径。Logprob 审计阈值保留：BF16 Mean/P99 ≤ 0.10/0.60，NF4 ≤ 0.50/3.0；记录 BF16 Rollout 与 NF4 Training 的分布差异，不将二者声称为完全相同策略。

### 检查与测量

CPU 小 Tensor 参考实现覆盖 Logprob、Entropy、两者对 logits 的梯度、完整 PPO Loss 和梯度、投影重算、Padding/Response Mask，以及 Token 加权累积。使用 `torch.testing.assert_close(rtol=3e-5, atol=3e-6)`，不是仅检查 Loss 有限。

GPU 阶段分别保存 PyTorch Peak Allocated/Reserved、`nvidia-smi` 每 0.1 秒采样的 Device Peak、前后向耗时、Chunk Size、Sequence Length 和实际 Optimizer Steps。Device Peak 是采样高水位，不保证捕捉采样间的瞬时峰值。OOM、审计失败、重载失败都保留错误状态，不能把进程正常结束视为 PPO 成功。16K 机制测试与真实 PPO Update 的结果分别报告。


## 15. 本轮真实 vGPU 结果（2026-11-06）

设备上报 RTX 4080 SUPER，PyTorch 可见 31.48 GiB；不等同于物理 5090。未安装新依赖，训练复用现有 Q-QLORA 环境：PyTorch 2.11.0 / CUDA 13.0、Transformers 5.18.0、PEFT 0.21.2、bitsandbytes 0.50.2；Rollout 使用独立 vLLM 0.11.0 环境。

| 验收 | 有效序列长度 | Micro-steps / Optimizer Steps | Peak Allocated | Peak Reserved | Device Peak | 阶段耗时 |
|---|---|---|---|---|---|---|
| 真实 NF4 模型机制测试，人工输入 | 16,384 | 2 / **0** | 13.09 GiB | 15.17 GiB | **15.49 GiB** | 46.57 s |
| 真实 Verifier 合格 Rollout 的 PPO 更新 | 7,009 / 7,185 | 2 / **1** | 9.23 GiB | 11.18 GiB | **11.49 GiB** | 37.32 s |

两者均为 Micro-batch=1、Chunk=128，Device Peak 为 0.1 秒间隔采样高水位。16K 机制测试每次前向约 6.04–6.06 秒、反向约 10.61–10.64 秒；真实 PPO 两次前向为 2.36 / 2.42 秒、反向为 3.88 / 4.00 秒。阶段耗时包含加载、审计和保存重载，不含独立进程的 BF16 Rollout 成本。

4 条真实 Train Rollout 中，2 条 Equation 回答完成且正确；2 条 Bit 回答在总长 16,384 截断，已保留输出并排除。两条合格回答的有效 Response Token 数为 6,932 / 7,082，按 Token 数加权累积后执行一次 AdamW Step。LoRA / Value Head 最大参数变化分别约 `1.00015e-5 / 1.00024e-5`。Adapter 从磁盘重载后 logits 最大差异为 0，FP32 Value Head 重载通过。二进制备份保存在本地 git 忽略目录 `.autodl/checkpoints/nf4_memory_v1/ppo_nf4_qlora`；仓库保存清单、SHA256 和实验报告。

CPU JSON / AST / Smoke 通过，36 项数值与配置断言通过，最大数值差异 `4.76837e-7`。BF16 Logprob 审计 Mean / P99 为 `0.01152 / 0.12065`；NF4 为 `0.13445 / 1.51922`。阈值保持原值，NF4 最大绝对差异仍达 `8.09962`，因此不能将 BF16 Rollout 与 NF4 Training 声称为同一个精确分布。

失败记录同样保留：首次快照重载缺少 BF16 autocast 导致 dtype 冲突；训练审计的 NF4 Backbone 前向缺少 BF16 autocast，FP32 SDPA 尝试分配 32 GiB 后 OOM。单独移除全 1 Mask 的实验未解决 OOM；补齐审计 autocast 后才完成更新。所有真实 NF4 前向必须包含 BF16 autocast，只有 Value Head 和概率归约显式保持 FP32。

**验证边界**：当前尚无未截断且 Verifier 合格的满 16K 真实轨迹；实际更新使用 7,009 / 7,185 Token，没有静默裁短。16K 结果证明真实模型前后向及累积可运行，未将人工输入冒充真实 PPO 更新。未测更新后的准确率或其他 GPU，不能据此声称提分或普遍的显存保证。

原始记录与汇总：

- [完整显存报告](../data/processed/ppo/nf4_memory_v1/ppo_memory_optimization_report.json)
- [16K 机制结果](../data/processed/ppo/nf4_memory_v1/ppo_memory_mechanism_status.json)
- [真实 PPO Update](../data/processed/ppo/nf4_memory_v1/ppo_training_status.json)
- [CPU 等价性](../data/processed/ppo/nf4_memory_v1_cpu/chunk_numerical_equivalence.json)
- [Checkpoint 清单](../data/processed/ppo/nf4_memory_v1/checkpoint_inventory.json)
