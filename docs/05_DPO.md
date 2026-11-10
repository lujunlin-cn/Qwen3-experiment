# DPO 指导：Verifier-guided Offline Preference Optimization

> 第二轮以 [Gravity DPO-B 设计](DPO_GRAVITY_DESIGN.md) 为准：只训练 Gravity 的正确短答/正确长答偏好；不运行 DPO-A。下文原实验路线保留为历史教学与来源说明，不能作为第二轮执行配置。

版本：v0.1 \| 对应 Notebook：`04_DPO`

## 1. 目标

复现并简化 The Submitter 的 DPO 路线：

`SFT Policy → Rollout → Verifier → Chosen / Rejected → DPO`

学生需要理解偏好数据如何产生、Verifier
如何决定标签、偏好对质量如何影响训练、Reference Policy
为什么必须明确，以及 DPO Loss 如何改变策略。

核心 DPO Loss 使用 PyTorch 实现，不调用高层 DPO Trainer 隐藏算法。

## 2. 主要参考：The Submitter

The Submitter 获得 NVIDIA Nemotron 比赛 Open Contribution Award --- Best
RL Method。其有效结果来自 DPO：

  指标           SFT v7   DPO v9
  ------------ -------- --------
  Public LB       0.836    0.844
  Private LB      0.852    0.856

作者从 SFT LoRA 继续训练，使用 vLLM 生成候选回答，通过答案提取器和
Verifier 判断正确性，再构造偏好对。原实验使用 Nemotron 30B-A3B 和 RTX
PRO 6000 96GB，不能把其训练耗时和收益直接迁移到 Qwen3-4B。

## 3. Rollout 与偏好对构造

The Submitter 的数据构造阶段选择 2,169 道题，每题生成 4 条候选，共 8,676
条回答。作者报告训练集得到 105 对、验证集得到 25 对。

选择规则： - Chosen：最短的正确回答。 - Rejected：最短的错误回答。 -
如果没有生成正确回答，但原始数据有可信参考回答，可使用参考回答作为
Chosen。

本项目基础复现： 1. 使用最佳 SFT Checkpoint。 2. 对 Equation Numeric 和
Bit Manipulation 生成多条回答。 3. 使用确定性 Verifier 判断最终答案。 4.
分别保存正确和错误回答。 5. 每组选择最短正确与最短错误回答。 6.
没有完整偏好对的题目不进入标准 DPO 数据集。

Gravity 主要用于能力保持评测，不强制构造 DPO 数据。

## 4. Verifier 设计

优先使用确定性验证：Boxed Answer
Extraction、数值答案比较、字符串规范化、Bit Manipulation
结果比较、Equation Numeric 规则与最终答案验证。

容差必须与任务定义一致。不能将过宽的数值容差用于精确整数任务。

答案正确不代表 CoT
正确。如果中间推理步骤参与偏好标签，必须由独立程序验证。

## 5. 三种偏好构造实验

  ----------------------------------------------------------------------------
  实验              Chosen                 Rejected          目标
  ----------------- ---------------------- ----------------- -----------------
  DPO-A             最短正确               最短错误          复现 The
                                                             Submitter

  DPO-B             规则与答案均验证正确   与 Chosen         研究 Hard
                                           接近但答案错误    Negative

  DPO-C             外部确定性 Solver      SFT Greedy        研究外部 Oracle
                    的正确轨迹             错误回答          偏好
  ----------------------------------------------------------------------------

DPO-A 是主线。DPO-B、DPO-C 是选做对照。

## 6. Binary Failure 与外部 Oracle

第八十名报告中，SFT 模型的 8 次采样大量集中在 8/8 全对或 0/8 全错，缺少
Boundary Group。作者设计了外部 Oracle Chosen + 模型 Greedy Rejected 的
DPO Pipeline，以给全错题提供正向信号。

该作者没有完成实际 DPO
训练。因此这只能作为待验证设计，不能称为已证明有效的方法。

本项目可在 Equation Numeric / Bit Manipulation 上尝试 DPO-C，但必须验证
Solver Trace 的正确性和可学习性。

## 7. DPO 数学目标

定义当前策略 `pi_theta`、固定参考策略 `pi_ref`、Chosen `y_w`、Rejected
`y_l` 和温度参数 `beta`。

``` latex
L_DPO = -log sigmoid( beta * [
    log pi_theta(y_w|x) - log pi_ref(y_w|x)
  - log pi_theta(y_l|x) + log pi_ref(y_l|x)
])
```

对 Completion Token 计算 Log Probability，并正确 Mask Prompt Token 和
Padding Token。

首版使用标准 Sigmoid
DPO。长度归一化等变体作为单独消融，不与标准实现混为一谈。

## 8. Reference Policy 必须明确

本项目固定 Reference Policy 为初始化时的最佳 SFT 策略：

``` text
pi_ref = pi_SFT
```

必须验证 Reference 权重不更新、Reference Log Probability 与初始化 SFT
策略一致、当前 Policy 与 Reference 的差异来自 DPO 更新，并且 PEFT
Adapter 的启用 / 禁用不会意外将 Base Model 当成 Reference。

The Submitter 源码使用 `ref_model=None` 并开启 Reference Log Probability
预计算。其实际行为依赖当时 TRL / PEFT
版本。本项目不照搬未锁定版本的隐式行为。

## 9. 训练与评价

首轮从小规模偏好对开始，先验证 Loss
和梯度方向。记录偏好对数量、各任务产生率、Chosen / Rejected
平均长度、标签质量、DPO Loss、Log Probability、KL to
Reference、训练前后准确率和遗忘情况。

DPO 与 SFT 从同一个最佳 SFT Checkpoint 出发。不能将 DPO
结果与不同数据或不同 Prompt 的 SFT 直接比较。

## 10. 主要风险

偏好对过少；Chosen 轨迹存在错误推理；Rejected 与 Chosen
的差异只是长度或格式；0/N 样本缺少 Chosen；Reference Policy
配置错误；DPO 过拟合小规模偏好集；Guess 任务被错误标记为唯一答案。

## 11. 交付物

Rollout 数据、Verifier 结果、Chosen / Rejected
数据、偏好对质量审计、手写 DPO Loss、DPO Adapter、SFT vs DPO
分任务对照。

## 12. 参考资料

-   [The Submitter
    GitHub](https://github.com/the-submitter/nvidia-nemotron)
-   [The Submitter DPO
    训练代码](https://github.com/the-submitter/nvidia-nemotron/blob/main/src/03_dpo.py)
-   [The Submitter Preference
    Builder](https://github.com/the-submitter/nvidia-nemotron/blob/main/src/02_update_dataset.py)
-   [第八十名方案](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/breaking-the-sft-ceiling-on-nemotron-3-nano-30b)
-   [Datawhale DPO
    Notebook](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/17_DPO_Loss_Tutorial.ipynb)

## 13. 本轮实现核对：Verifier 与 Reference Policy

Notebook 04 的 Rollout 使用与 Notebook 01 相同的 `official_nvidia_metric_v2` 提取和判定协议：先识别 `boxed` / `<final>` / `<answer>`，再按任务回退到 8-bit 或数值格式；数值采用 `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`，截断回答直接拒绝。Chosen 和 Rejected 都必须未截断，并且标签来自独立 Verifier。

DPO Policy 与 Reference 都从同一个 SFT Manifest 的 Base Model 和 Adapter 初始化；Reference 参数冻结，标准序列 Log Probability 仍按 Response Token 求和，长度平均只是另行消融。DPO 输出 Manifest 保存 Parent SFT Adapter、Base Model、Tokenizer 和新的 `policy_version`。当前 GPU Rollout、DPO Update 和 Adapter Reload 仍为 `Not Run`。
