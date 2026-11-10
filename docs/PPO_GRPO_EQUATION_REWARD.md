# Equation Numeric：PPO/GRPO 单一复合 Reward

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

本轮统一使用同一结构化输出契约；原题-only条件另做相同模型配对，不能把格式提示收益冒充训练收益：

```json
{"task_type":"equation_numeric","rules":{"+":"add"},"example_outputs":["17"],"query_output":"23","answer":"23"}
```

这是字段形状示意，不是实验样本。真实题的全部运算符/例子按原顺序填写。不能通过这个例子改变官方题、答案或评分。

Notebook 01 的 `audit_equation_process` 只走独立 Numeric24 Verifier，不调用 Solver，不读隐藏生成规则。所有显式计算按精确字符串匹配：符号前后缀、前导零不能丢失。最终答案仍用官方评分：数值1%容差，非数值形式按原规则。官方真实 Equation 的候选不唯一时，保留歧义标记和官方答案，不删题或强改标签；与要求唯一解的合成 NeMo 接受契约分开。

Thinking 开启时，`audit_closed_completion` 只验证其后的闭合 JSON。它不认证任意自由 Thinking；报告 `proof_scope` 和 `thinking_present_unverified`。NeMo 数据仍不允许未经验证的外围 Trace 进入 Accepted。

## 3. 本轮只实验一个复合奖励 R1

PPO、GRPO统一使用：`R1 = F + 0.20 P + 0.05 S`。不再安排P0–P3消融，也不采用纯二值奖励。长度只统计，暂不加入奖励。

记完整为stop且没有truncation。所有截断回答的奖励分量均为0。

- F：完整且官方末答正确时1，否则0。格式不合法但有合法显式最终答案时，仍独立按官方评分，不暗中把Schema通过作为F条件。
- S：严格对象字段、规则名字/映射、全部示例列表长度、字符串类型合规时1，否则0。它只表示格式。
- E：所有示例中，“按声明规则独立算出的结果=声明输出=原示例输出”的比例，分母固定为完整示例数。
- G：声明映射符合Numeric24 family/group全局约束时1。
- Q：只有规则合法且全部示例精确解释后，Query声明结果等于独立计算结果时1。
- C：Q通过，且answer与query_output精确一致时1。
- P=(E+G+Q+C)/4；格式不合法时P=0，避免漏例/不可解析片段凑过程分。

错误完整回答最高.25，正确完整回答至少1；不能把.25过程分报告为“答对”。这保证单条回报的优先级，不保证任意策略的期望正确率单调上升。过程分只能来自程序复算，不使用模型自评或文本模糊相似度。

例：完整错误回答若Schema合规、全局映射合法、一半示例正确，但Query和抄录未通过，则P=(.5+1+0+0)/4=.375，R=.125，其中F=0。报告它是部分过程信号，不是正确性奖励。

本配方参考251名成功报告中的正确性精修方向，以及The Submitter实验代码中显式分离答案/格式的工程做法；后者没有已证实的GRPO成功结果。精确过程P和权重是本项目设计，**不声称是某位选手已经验证成功的原配方**。

`equation_reward_components`在Notebook06直接展示，PPO读取同一函数；只接受当前R1版本，其他奖励ID报错。Worker在新Run中重新验证原始输出，保存F/S/P/Combined及Token诊断。长度差不会改变R1。

`exact_trace_pass`指闭合JSON中合法映射、全部示例、Query及抄录全部通过；自由Thinking不被这个标志认证。官方Equation候选存在歧义时，一个自洽但不符合标准答案的过程可以有P，F仍为0，必须单列歧义子组。

## 4. 全对/全错/Mixed 组

组类别按 F 划分，不按 Combined Reward；否则给格式分的全错组可能被错误称为 mixed。

| 组 | F | S | P | 长度 |
|---|---|---|---|---|
| 全对 | 相同，GRPO优势0 | 完整格式若相同仍0方差 | 可能有必要证明通过程度差异 | 只统计，不给分 |
| 全错 | 相同，GRPO优势0 | 结构有差异可有低权重信号 | 真正算对部分示例时可区分；无可验证步骤仍0 | 只统计，不给分 |
| Mixed | 有真实正确/错误差异 | 辅助，不代替F | 辅助，不可掩盖错误末答 | 只统计，不给分 |

全错并不自动意味着 P>0；长度即使不同也不影响本轮奖励。每组实际保存 F/S/P/Combined、population variance/std、mean、advantage 向量、组类别、有效组标记、实际更新数。`group_reward_audit.json` 与 `reward_audit.jsonl` 已接入 Worker。

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

Phase C每算法只运行一个R1配方：PPO固定8题×1、GRPO固定4题×8，独立Run、相同SFT、题单、采样设置与cap。先用SFT的4题×8候选检查实际F/S/P分布、每题正确次数、过程分方差及时间成本；不能根据结果临时挑有利题替换冻结面板。

训练分叉后不可复用旧策略样本冒充on-policy。记录真实更新数、非零梯度、组方差、rollout/train时间与峰值；完整组仍可能全同分，必须承认0信号，不加噪声或提高错误答案的正确性标签。

单一配方的结果不能解释“过程分或格式分单独有效”；本轮先验证复合机制和整体效果。若以后需要归因，再另行审核消融。最终看Equation准确率、截断和跨任务保持的逐题配对证据，不因短答或格式率提升宣布数学能力解决。

## 8. Gravity中间过程：公式与代入分别检查

Gravity确实可以用 `g=2d/t²` 审查关键步骤。Notebook01新增 `audit_gravity_formula`，以受限AST与精确有理数检查显式公式是否在代数上等价于 `2*d/t**2`，支持 `2d/(t^2)`、`d/(.5*t*t)`，拒绝 `2*d/t`、`2*d*t²` 和任意代码调用。没有执行eval。

公式出现只证明关系式被写出，不证明变量取值或运算正确。该诊断明确 `substitutions_verified=false`、`whole_trace_verified=false`；DPO-B仍须通过每例g估计、OLS、原Query时间代入与最终位移的数值证明。不能仅搜到这条字符串就给高过程分或通过Accepted。

这用于Gravity DPO-B质量检查/诊断；PPO、GRPO当前仍只训练Equation，不把Gravity公式条件塞进Equation的奖励。若将来增加Gravity RL，需单独注册公式、代入、结果的一致性过程分，不在本轮暗中扩大范围。
