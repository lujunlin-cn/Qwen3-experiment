# 第二轮 Phase C 真实 GPU 验证（部分完成）

执行 Suite：`round2_20261114_phasec_r1`；执行源码：`f1502fc`。这是短程工程与学习信号验证，不是收敛实验。正式 Phase D 未启动。

结论：9 条生成输入 Seed 全部通过验证；72 条真实 NeMo 输出仅 1 条 Accepted。PPO 完成两次真实更新，但同题原题评测的 Equation 正确率仍为 0/8。闭合证明对照尚未取回完整结果，服务器随后返回 `Connection refused`，远端最终执行状态未知。不得将本报告解释为全流程验收通过。

用户在执行过程中将后续算法效果评测缩减为 **PPO**，保留同题 Base/SFT 对照；DPO/GRPO 不再做模型效果评测。原注册协议、历史日志及失败结果均保留。NeMo 的合格输出没有并入训练。

## 实验边界

- 模型：Qwen3-0.6B，revision `c1899de289a04d12100db370d81485cdf75e47ca`。
- 设备：RTX 4080 SUPER，32760 MiB。显存阈值为 **8,000,000,000 bytes**，每 0.25 秒采样设备总用量；不能据此保证采样间瞬时上界。
- 所有方法从同一 SFT r1 Adapter 独立出发，其权重 SHA256 为 `d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371`。
- Tune Validation 每类固定 8 题，原题与闭合证明条件分别比较；不触碰锁定 final，不用 Holdout 选择模型。官方划分、答案与评分规则不变。
- 评测 greedy，Thinking 开启。Roman 输出上限 8192，Gravity/Equation 为 16384。所有均长统计仅计生成 Token，包含 Thinking、错误、截断和 EOS。
- 训练使用 LoRA、microbatch 1、梯度累积、decoder checkpointing、关闭训练 KV Cache 和分块词表投影；没有使用 1F1B 流水线。

## NeMo：调用成功，质量问题尚未解决

| 组 | 实际输出 | Accepted | 全阶段秒 | 设备采样峰值 GB |
|---|---:|---:|---:|---:|
| R0 | 18 | 0 | 72.27 | 3.808 |
| A：Generator Prompt | 18 | 0 | 69.92 | 3.808 |
| B：Trace Schema | 18 | 0 | 72.24 | 3.808 |
| C：Generator/Solver 协作 | 18 | 1 | 66.09 | 3.808 |

72 次 HTTP 调用均成功。B 的 9 条 Trace 全部通过 JSON Schema，但数学验证为 0/9。C 唯一 Accepted 为 Roman `100 → C` 的闭合 Trace，已通过三集合泄漏审计；其余 71 条保持 Preview。R0 输出与第一轮对应输出完全相同，不是新增的质量收益证据。每类仅 3 个 Seed，不支持一般接受率改善的结论。

作为输入的 9 条 Seed 均为 `solver_status=solved`、`verification_status=accepted`、`leakage_status=passed`。Seed 合格、HTTP 成功、输出结构合格和最终数据质量合格是不同检查；不能把前面的通过状态当作后面的 Accepted。

分阶段数量、逐条失败、原始 HTTP 和 Accepted Manifest 见 [NeMo 结果](../data/processed/round2/real_gpu_20261114_phasec/NEMO_RESULTS.md)。

## DPO-B、PPO、GRPO 的训练信号

| 阶段 | 候选数 | 实际 optimizer.step | 生成/验证秒 | 更新计算秒 | 阶段设备采样峰值 GB |
|---|---:|---:|---:|---:|---:|
| Equation readiness | 32 | 0 | 898.17 | — | 5.198 |
| Gravity DPO-B rollout | 64 | 0 | 2406.61 | — | 5.198 |
| DPO-B 偏好筛选 | — | 0 | — | 未执行 | 0.271 |
| Equation PPO 第 1 轮 rollout | 8 | 0 | 210.53 | — | 5.198 |
| PPO 第 1 轮更新 | — | 1 | — | 17.87 | 3.360 |
| Equation PPO 第 2 轮 rollout | 8 | 0 | 201.47 | — | 5.198 |
| PPO 第 2 轮更新 | — | 1 | — | 16.87 | 3.177 |
| Equation GRPO rollout | 32 | 0 | 835.64 | — | 5.198 |
| GRPO 无信号检查 | — | 0 | — | 未执行 | 2.036 |

生成、更新计算与包含初始化/验证/保存的整个守卫阶段分别计时，不能混为课程总耗时。DPO 的 0.271 GB 和 GRPO 的 2.036 GB **不是反向训练峰值**。本次没有重训 SFT，不能用历史 SFT 峰值冒充本轮测量。

**DPO-B：**16 题各 4 候选，末答正确 7/64、格式通过 20/64、完整证明通过 0/64，截断 0/64，平均输出 6984.59 Token。双侧正确且完整、满足长度差门槛的偏好对为 0，覆盖率 0/16；状态 `no_preference_signal`。没有训练后 DPO Checkpoint，未验证 DPO 反向训练显存或效果。详见 [DPO-B 记录](../data/processed/round2/real_gpu_20261114_phasec/dpo_gravity_b/RESULTS.md)。

**PPO：**两轮分别对同一批 8 道 Train 题重新采样，累计真实更新 2 次。Adam 参数状态由 step 1 恢复至 step 2；Value Head 恢复，两次独立 Adapter 重载 logprob 最大差均为 0。首轮奖励全 0 时，非零 Critic/GAE 仍产生参数更新；这不代表产生了正确性学习信号。第二轮训练候选末答正确 1/8，格式和过程仍为 0，不能把重新采样的 0/8→1/8 当作配对评测提升。详见 [PPO 记录](../data/processed/round2/real_gpu_20261114_phasec/ppo_equation_r1/RESULTS.md)。

**GRPO：**4 组各 8 候选，末答、格式和过程通过数均为 0；全部组奖励方差和组内优势为 0，有效组比例为 0%。状态 `no_group_signal`，没有 optimizer.step 或训练后策略。无额外噪声、伪造奖励或放宽 Verifier。详见 [GRPO 记录](../data/processed/round2/real_gpu_20261114_phasec/grpo_equation_r1/RESULTS.md)。

PPO/GRPO 仅使用一个奖励：`R1 = F + 0.20P + 0.05S`，分别表示完整末答正确、确定性过程验证、格式通过；长度权重为 0。两者 Reference KL 系数为 0，未实现冻结 Reference KL 路径；old-policy approximate KL 仅用于行为策略漂移诊断。

vLLM 与训练后端的行为概率对齐是带容差检查，不是逐 Token 相等：PPO 两轮平均绝对差约 0.02，第二轮单 Token 最大差 1.163。原始行为 logprob 仍作为分母保留；独立重载的零差值是另一项检查，不能用来掩盖后端数值差异。

## 原题条件：PPO 没有改善数学正确率

| 任务 | 模型 | 正确 /8 | 截断 /8 | 平均输出 Token |
|---|---|---:|---:|---:|
| Roman | Base | 5 | 1 | 2613.12 |
| Roman | SFT | 2 | 2 | 3540.00 |
| Roman | PPO | 3 | 3 | 4518.75 |
| Gravity | Base | 8 | 0 | 6017.25 |
| Gravity | SFT | 7 | 1 | 6059.50 |
| Gravity | PPO | 6 | 1 | 8056.75 |
| Equation | Base | 0 | 6 | 13618.88 |
| Equation | SFT | 0 | 8 | 16384.00 |
| Equation | PPO | 0 | 7 | 15139.12 |

PPO 对照 SFT 的 Equation 正确率均为 0/8；仅少了 1 道截断，平均输出减少 1244.88 Token。该结果不能验收为数学任务被解决或推理能力提升。Roman 增加 1 道正确，但增加 1 道截断；Gravity 净减少 1 道正确、输出增长 1997.25 Token。三任务准确率配对精确 McNemar 的 p 值均为 1。

每任务 n=8、每题占 12.5 个百分点，Wilson 区间很宽；例如 0/8 的 95% 区间上限约 32.4%。本轮不支持稳定提升、非劣效、统计显著或收敛声明。原题输出没有要求闭合 JSON，因此该条件的格式率、完整过程率记未测，不伪报为数学失败 0%。

## 闭合证明条件

对应的 Base/SFT/PPO 同题对照均未取回完整结果。断连前最后只读观察为 Base 12/24 条；最近已同步的本地副本仅保留该任务的 4 条 raw partial。SFT/PPO 闭合条件尚无完成证据，不汇总半份面板为正式指标。连接中断记录见 [interruption](../data/processed/round2/real_gpu_20261114_phasec/connection_interruption.json)。

原题条件的 3 组各 24 条、共 72 条评测已完整取回并独立复核。恢复副本以 `snapshot_incomplete` 导出，源 `state.json` 中的 `running` 保持原样；没有补写远端停止、完成或最终耗时。原监督器仍按最初五模型清单报告缺项，用户取消 DPO/GRPO 评测的指令另存 scope，独立汇总只将 3 个闭合条件任务列为当前所需的未完成评测。

## 已确认的失败边界

Equation readiness 的 32 条中 14 条存在 JSON 解析错误，余下 18 条未通过结构或数值验证；PPO 两轮分别有 2/8、3/8 条解析错误，GRPO 为 12/32。Gravity DPO-B 的 64 条中 44 条解析失败，20 条结构可读但没有完整数值证明。

这些分类描述闭合证明验证入口，不表示已证明所有自然语言 Thinking 都错了。模型原题回答与额外 JSON 契约是不同条件，不能把换提示词后的变化归因于训练。

Gravity 64 条中有 2 条被程序识别出 `g=2d/t²` 的公式关系，但代数关系出现不等于变量代入、所有示例和查询计算都正确；它们没有因此获得完整 Trace 认证或进入 DPO-B。详见 [失败分类](../data/processed/round2/real_gpu_20261114_phasec/closed_trace_failure_categories.json)。

## 归档和验收

远端开跑前 CPU 回归实际通过 11 项课程检查、13 项数值检查、3 项 NeMo 检查和 1 项解释器隔离检查，原始日志与依赖版本见 [环境 Manifest](../data/processed/round2/real_gpu_20261114_phasec/environment/manifest.json)。结果导出工具逐条核对原始回答、Token IDs、官方 Validation 原题与答案，并重新运行 Verifier、过程检查、奖励与统计；篡改答案、回答、Token、评分或截断标记的 5 个反例全部被拒绝。

实际完成的 18 个守卫阶段累计 8331.50 秒（约 2 小时 19 分），不包含安装、阶段间开销与未完成闭合评测，不能称作全流程或收敛时间。所有已完成阶段的设备采样峰值均低于 8 GB，最大 5,197,791,232 bytes；反向训练峰值的有效结论仅覆盖 PPO 本次实际序列，不能扩展为 DPO/GRPO 反向训练已验收。

完整原题指标、正确回答均长、Wilson 区间、逐题配对、长度 bootstrap 与符号翻转检验见 [独立汇总](../data/processed/round2/real_gpu_20261114_phasec/evaluation_snapshot/independent_summary.json) 和 [结果表](../data/processed/round2/real_gpu_20261114_phasec/evaluation_snapshot/RESULTS.md)。已核对 25 个执行源码文件与注册 commit 的 Hash、官方三集合 Hash、已完成阶段采样数量及峰值。

权重、Adam 和 Value Head 二进制保存在本机 `.autodl/round2_20261114/ppo_full/`；Git 仅收录文本证据与 Hash。原注册执行源码和失败日志保留，用户缩减评测范围另存 scope 记录，不改写旧协议。

当前结论：PPO 的真实更新、同算法恢复和已测序列的低显存路径成立；NeMo 质量、DPO-B 数据覆盖、GRPO 有效组信号及数学正确率目标尚未通过。本轮不进入 Phase D。后续应先在 Train 上补齐可独立验证的输出契约覆盖，再重新注册短程实验；不根据本次 Validation 错题反向制造训练样本。
