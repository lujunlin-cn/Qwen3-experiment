# Gravity DPO-B 设计

状态：审核稿，CPU 数值检查已执行，真实 GPU 训练未运行。根据最新决定，**本轮只做 DPO-B：正确短答优于正确长答**；不设计或执行正确/错误偏好的 DPO-A。Roman 与 Equation 不进入 DPO 训练。

## 1. The Submitter 方案复核

上游固定版本：`c05e11ab0512617a675e9657c84df2841eb32acd`。原始源码、日志片段、Hash 与 URL 见 [来源记录](../data/processed/round2/round2_20261110_phaseab_v1/research/sources.json)。

| 内容 | 证据 | 复核结论 |
|---|---|---|
| 每题4条 Rollout | `src/02_update_dataset.py:150,1148` | 默认 `TRAJECTORIES=4` |
| 短正确/短错误选择 | 同文件 `select_preference:638–686` | 分别按 `len(output)` 排序；是字符数，不是真实 Token |
| 无正确候选时的行为 | 同函数674–679附近 | 当前代码允许回退到已有 reasoning/response；与“绝不回退”的简述不同 |
| 2,169题/8,676候选 | `logs/02_update_dataset.log:19–34,540–550` | 两轮训练候选1585+529，验证55；合计2169，按每题4条推算8676，不是本项目生成量 |
| 175训练/29验证对 | 同日志704–718 | 明确记录最终完整偏好数量 |
| 162高质量对、使用100对 | `logs/03_dpo.log:28–40` | 过滤175→162，再选[0,100) |
| β=.1、LR=2e-6、2 epoch、有效batch4 | `src/03_dpo.py:223–248` | 当前源码默认；已有某份日志实际75步/3 epochs，不能把默认说成每次实跑配置 |
| Reference | 同文件 DPOConfig 与 Trainer | 预计算 reference logprobs；本项目用冻结初始 SFT adapter 做等价明确定义 |
| Public .836→.844；Private .852→.856 | 上游 README:9–17 | 作者自己的 SFT/DPO 提交结果，不能当作本项目预期收益 |

借鉴的是同题采样、验证与冻结 Reference 的流程。本项目不使用字符数代理 Token 长度，不使用外部教师兜底构造偏好。DPO-B 是当前任务的延伸，**不是 The Submitter 原始偏好方案**。其 GRPO 探索中的相似度/格式奖励也不能归为 DPO 收益原因。

## 2. 数据与质量构造

固定第一轮 SFT r1：权重 SHA256 `d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371`。Policy/Reference 都从它出发，使用同一 BF16 base，不继承任何 PPO/GRPO 或旧 DPO adapter。

只从官方 Gravity Train 采题，每题4条真实 SFT 候选。先保存原始 completion IDs、包含 Thinking 的输出 Token 数、stop/length、采样参数、policy fingerprint；再做独立官方答案验证和闭合证明验证。

筛选：

1. 同题、相同 prompt IDs、相同 SFT policy_version；原题和官方答案不可改变。
2. 两边都必须 `finish_reason=stop` 且未截断；Schema 可解析，最终答案符合官方评分。
3. 两边都要通过必要 Gravity 证明：每例估计、物理关系、Query 时间代入与最终计算一致。
4. Chosen=最短合格回答；Rejected=最长合格回答。
5. `long_tokens-short_tokens >=64`，且 `>=0.20*long_tokens`；差值含 Thinking。
6. 至多每题1对。不足两个正确完整回答、长度差不足、仅有相同 token 序列，均不成对。错误或截断回答只用于诊断覆盖率，不做 Rejected。

不得强制凑对、删除错误样本后将任意剩余项视作 Chosen、跨题配对，或把人工短答替换模型回答。两边的长度差只在质量已经验证后比较。

当前闭合证明检查只认证最终 JSON 字段；自由 Thinking 中的任意文字尚不能全部验证。保留原文供审阅，逐项报告这个范围，不称“全部 Thinking 已验证”。若短答只是删掉必需字段，Verifier 必须拒绝。

预备阶段已将 `gravity_preference_pairs` 写在 Notebook 04，并将它接入 `course_worker.train_dpo` 的显式 `preference_mode=B` 路径。旧通用函数用于第一轮审计，不是第二轮协议入口。

## 3. 数学目标与资源实现

对同题(x,y+,y−)，Response 序列概率包含所有生成动作，不包含输入：

```text
log π(y|x) = Σ_response log π(y_t|x,y_<t)
z = β [(logπθ(y+|x)-logπθ(y−|x))
       -(logπref(y+|x)-logπref(y−|x))]
L = -log sigmoid(z)
```

禁止把 token mean 偷换为标准 DPO 的 sequence sum；不同长度概率差由冻结 SFT Reference 校正。Response mask 覆盖生成的 Thinking、正文和实际 EOS，不覆盖 Prompt/Padding。

共享冻结 BF16 base，训练 default adapter，另载冻结 SFT reference adapter。Reference 始终是初始 SFT，不随 epoch/round 更新。Reference scores 保存 fingerprint 与原始 Token 来源；不能复用另一精度、另一序列或更新后的 policy scores。

顺序反传使用 dL/dlogπ+=−β·sigmoid(−z)，负样本符号相反。两边在同一未更新参数点重算，dropout 关闭，各自反传后才 optimizer.step；每窗口按实际 pair 数平均。CPU 已对 β=.05/.1/.3 与 Dense Loss/梯度逐项比较。

审核起点：BF16 LoRA r8/alpha16/dropout0，microbatch1、累积4、AdamW LR2e-6、β=.1、2 epochs，梯度裁剪1。它是待测起点，不能由选手配置直接推定适合0.6B。先保持 β 不动，只在 Tune 上对 LR 做预注册的单因素比较；数据覆盖不足时先停止，不临时无限加 epochs。

## 4. 采样与训练顺序

Phase C 先固定16题×4候选，cap16384，T1/top_p1/top_k−1。原题后追加同一 Gravity 闭合证明契约，保持 Thinking 开启；不得给候选提供标准 Query 答案。另用原题-only条件做相同模型对照，区分提示词效应和训练效应。

若0个合格 DPO-B 对：记录 `no_preference_signal`、0 steps；本方法短程验证未通过，不伪造训练成功。若只有少量对：可以检查真实梯度和重载，但不能声称质量趋势充分。正式训练面板规模待 Phase C 的正确轨迹覆盖率与长度分布确定后重新冻结，不用 final 选题。

Phase C/D、不同 LR/epoch 的试验各自独立 Run；不把同一 GPU Run 的中途变化混成一个结果。2 epochs 可在同一固定偏好集合训练；Round 2 恢复同一算法的 Adam 状态，不继承别的算法。

## 5. 评测与成功条件

Base、SFT、DPO、PPO、GRPO 在相同 Gravity 验证题上测试；每个 Prompt/契约条件分别比较。固定 cap、greedy、模板、模型/tokenizer revision。Token 不含输入，包含全部 Thinking、错误和截断输出。

必须报告 Accuracy/Wilson95、Truncation、Mean Output Tokens、Correct Output Mean Tokens、两模型均正确题的配对长度差、闭合证明通过率、Format Valid Rate、正确短答比例、逐题偏好数量和有效覆盖率。

“正确短答比例”同时给出分母：全部题中同时正确/必要证明通过，且比同题 SFT 至少短64 Token、短20%的比例；以及 SFT/DPO 均正确题中的比例。不能只保留 DPO 恰好做对的题来比较长度。

观察验收须同时满足：

```text
Accuracy_DPO >= Accuracy_SFT
MeanTokens_DPO < MeanTokens_SFT
Truncation_DPO <= Truncation_SFT
必要证明通过率不下降，且未因省步骤获得短答
```

最终还给出 McNemar 配对准确率/截断检验、配对长度 bootstrap 与随机化检验。观察到无回退不等于统计非劣性；区间宽时如实写“证据不足”。输出变短也不等于推理能力提高。
