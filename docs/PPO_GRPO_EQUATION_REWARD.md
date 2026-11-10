# Equation Numeric：PPO/GRPO Reward 与消融

状态：设计与 CPU 数值验证；没有新 GPU 奖励分布、数学收益或显存结论。两个算法独立从相同 SFT r1 初始化，只训练 Equation；Roman、Gravity 只做跨任务评测。

## 1. 原始证据与归属纠正

来源正文、取得方式、Hash 和固定上游 commit 见 [research/sources.json](../data/processed/round2/round2_20261110_phaseab_v1/research/sources.json)。Kaggle 直链本次404，通过只读文本代理取得原报告正文；快照保留此取得方式。

| 来源 | 原文能支持的结论 | 不能推定的内容 |
|---|---|---|
| [The Submitter](https://github.com/the-submitter/nvidia-nemotron/tree/c05e11ab0512617a675e9657c84df2841eb32acd) | DPO 有作者提交收益；GRPO 实验 README/源码包含 exact、answer fuzzy、completion fuzzy、boxed、think 奖励 | 这些 GRPO 探索有 state sync/OOM 问题，不能当成获奖 DPO 收益的机制证据 |
| [第251名原文 §6、§8.2](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/trace-format-engineering-and-svd-downsampled-lora) | 可逐行复算的 Solver Trace→SFT→轻量 GRPO；v27 92.6%→93.8%；v18仅38%组有reward variance；弱基线和从无成功的任务未获恢复 | 该原文未披露可核实的多分量奖励权重，也未证明使用了上述 fuzzy/format 组合 |
| [第80名原文 §7.2、§8.2–8.3](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/breaking-the-sft-ceiling-on-nemotron-3-nano-30b) | 9500题、N8/T.7，约6343全对/3157全错、0边界；自写离线clip GRPO与SFT同为.86；讨论采样局限 | 不能把0边界推广成任何温度或任何模型的定理；作者提议的外部Oracle DPO不是已完成DPO收益 |

旧 `docs/07_GRPO.md` 将 Exact/Answer Similarity/Completion Similarity/格式组合归给251名，证据不支持，已更正。它属于 The Submitter 的**实验性 GRPO**实现。本文不采用模糊文本相似度、非空 think 或 boxed 格式作为数学正确性。

可迁移：确定性验证、真实候选方差、短且可复算的 SFT 轨迹、独立采样/训练进程、保留全错失败分析。不可直接迁移：Nemotron 30B/Mamba 架构补丁、排行榜收益、fuzzy similarity 当过程正确、外部教师答充当本轮采样正确答。

## 2. 输入契约与独立裁判

P0–P3 使用同一结构化输出契约，不能只给 Process 组额外格式提示：

```json
{"task_type":"equation_numeric","rules":{"+":"add"},"example_outputs":["17"],"query_output":"23","answer":"23"}
```

这是字段形状示意，不是实验样本。真实题的全部运算符/例子按原顺序填写。不能通过这个例子改变官方题、答案或评分。

Notebook 01 的 `audit_equation_process` 只走独立 Numeric24 Verifier，不调用 Solver，不读隐藏生成规则。所有显式计算按精确字符串匹配：符号前后缀、前导零不能丢失。最终答案仍用官方评分：数值1%容差，非数值形式按原规则。官方真实 Equation 的候选不唯一时，保留歧义标记和官方答案，不删题或强改标签；与要求唯一解的合成 NeMo 接受契约分开。

Thinking 开启时，`audit_closed_completion` 只验证其后的闭合 JSON。它不认证任意自由 Thinking；报告 `proof_scope` 和 `thinking_present_unverified`。NeMo 数据仍不允许未经验证的外围 Trace 进入 Accepted。

## 3. 可验证 Reward

记完整为 stop 且没有 truncation。所有截断样本的以下分量均为0。

- F：完整且官方末答正确时1，否则0。错误答案不获得正确性奖励。
- S：严格对象字段、规则名字/映射、列表长度、字符串类型可解析时1，否则0。它只表示格式。
- E：所有示例中，“按声明规则独立算出的结果=声明输出=原示例输出”的比例，分母固定为完整示例数。漏例、重复字段和 Schema 错误不获得过程分。
- G：整个声明映射符合 Numeric24 family/group 全局约束时1。
- Q：只有规则合法且所有示例精确解释后，Query 的声明结果等于独立计算结果时1。
- C：Q通过，且 answer 与 query_output 精确一致时1。
- P=(E+G+Q+C)/4；格式不合法时P=0。
- L：仅完整、F=1、全部必要闭合证明通过时，`1-min(output_tokens/cap,1)`；其余0。

`exact_trace_pass` 是合法全局映射、全部示例、Query计算、抄录一致性全部通过；最终官方答案是否正确另列。因此对歧义题，一个自洽但与标准答案不同的过程仍可能有过程分，F仍为0。报告歧义子组，防止将它当作已答对。

| 消融 | Combined Reward | 目的 |
|---|---|---|
| P0 | F | Binary 对照 |
| P1 | F+.05S | 仅格式变化 |
| P2 | F+.20P | 可复算过程变化 |
| P3 | F+.20P+.05L | 在必要过程完成后压缩长度 |

组合在 Notebook 06 的 `equation_reward_components` 中直接展示；PPO 从同一教学定义读取。Worker 在 `reward_ablation` 显式启用后重新从原始输出验证、保存逐条分量，再进入目标函数；不把历史 `correct` 布尔值当作过程奖励证据。

单条错误完整回答最高 .25，正确完整回答至少1；不等于任意策略期望 Reward 的提升都保证正确率提升。Format/Process 仍可能被优化偏向，必须以最终准确率门槛和消融控制。不得加入随机噪声、伪造正确标签或用模型自评分来制造梯度。

## 4. 全对/全错/Mixed 组

组类别按 F 划分，不按 Combined Reward；否则给格式分的全错组可能被错误称为 mixed。

| 组 | Binary F | S | P | L |
|---|---|---|---|---|
| 全对 | 相同，GRPO优势0 | 完整格式若相同仍0方差 | 可能有必要证明通过程度差异 | 仅全部必要证明正确的回答可按长度区分 |
| 全错 | 相同，GRPO优势0 | 结构有差异可有低权重信号 | 真正算对部分示例时可区分；无可验证步骤仍0 | 始终0 |
| Mixed | 有真实正确/错误差异 | 辅助，不代替F | 辅助，不可掩盖错误末答 | 只作用于合格正确回答 |

全错并不自动意味着 P>0；全对也不自动意味着一定有长度方差。每组实际保存 F/S/P/L/Combined、population variance/std、mean、advantage 向量、组类别、有效组标记、实际更新数。`group_reward_audit.json` 与 `reward_audit.jsonl` 已接入 Worker。

## 5. GRPO 标准化与目标

G=8；先完整采样一组再算优势，microbatch 不改变组定义。

```text
A_i = (R_i - mean_group(R)) / (std_population(R) + 1e-6)
std<=1e-6 -> A_i=0
ratio_it = exp(logπθ_it - logπbehavior_it)
L = -mean_i mean_response_t min(ratio_it*A_i, clip(ratio_it,.8,1.2)*A_i)
```

第二轮显式 `loss_reduction=sequence_mean`。第一轮 Worker 跨回答按 token mean 归约；它是不同的加权目标，不能把两者写成同一实现。长短回答不等长时要验证区别。policy ratio 分母保留 vLLM 实际 raw logprobs，不用更新后重算值替代。

采样T1、top_p1、top_k−1，避免把截断采样分布概率和原模型概率混用。保留 BF16 inference/training 概率对齐检查；原容差只是数值漂移筛查，不证明两个后端逐 bit 同分布。不得重采样直到凑出 mixed 再隐瞒成本。零方差组记录0更新；整个批次无信号时 `no_group_signal`，不能算训练成功。

## 6. PPO 不能套用组内零方差结论

PPO 每题1条，单独 Value Head；没有按题组内中心化。终端奖励放在最后一个可见 Response action，Value 使用预测该 action 之前的 hidden state。γ=λ=1；长度上限作为有限时域终点，bootstrap=0，不把 length 伪装为观测到 EOS。

```text
δ_t = r_t + γ(1-done_t)V_old(s_(t+1)) - V_old(s_t)
A_t = δ_t + γλ(1-done_t) A_(t+1)
Return_t = A_t + V_old(s_t)
L = L_clip + .5*mean((Vθ-Return)^2) - entropy_coef*Entropy
```

Prompt/Padding 不进入任何均值；rollout batch 在更新前统一算并冻结 old values 和全局 advantage 统计。Policy 用行为 logprob 固定分母，Critic 用 FP32 scalar head。PPO 全错时 F=0，但 Value 尚非0仍可能有 Critic 校正及策略 advantage；Critic 已精确为0时，才可能同时缺信号。CPU 有直接检验，不能将 GRPO 的 std=0判断用于 PPO。

`KL_COEF=0` 意味着**没有 Reference Policy KL 正则**。Old−Current logprob 只是行为策略漂移诊断。当前不实现 Reference KL 路径，非零系数必须报错，不能以它冒充 frozen-SFT KL。Entropy 系数起点0，熵仍记录；要调整需另注册单因素实验。

## 7. 运行与验收

Phase C 先在各自固定 Train 小面板上做 P0–P3，PPO每组8题×1、GRPO每组4题×8；每个算法/消融独立 Run、相同 SFT、相同题序/采样配置/Token cap。可以复用同一冻结 SFT 的首批 Rollout 检查奖励分量，但训练超过首轮后不可把旧策略样本称为 on-policy。

先记录真实更新数、非零梯度、Reward分布、方差、rollout/train时间及设备峰值；再评估三任务同题结果。消融选择仅用 Tune。PPO/GRPO 的最终验收需数学正确率的可信配对证据，截断、长度、证明与跨任务保持共同报告。不因一次小样本上涨宣布“已解决数学”。
