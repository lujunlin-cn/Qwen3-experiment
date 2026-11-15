# Gravity V2：真实 NVIDIA API 全量测试

**全量测试已完成：P16K为2/3 Accepted；P32K为3/3 Accepted，达到冻结三题的工程门槛。** 两条件各使用同样的3条冻结Gravity Seed，共6个Seed×条件；没有新题扩展、SFT合并或GPU训练。3/3仅代表这批冻结题通过，不代表一般接受率100%。

| 条件 | 模型与采样 | Thinking | 输出上限 |
|---|---|---|---:|
| P16K | Ultra；temperature=1.0，top_p=.95 | enable_thinking=true | 16384 |
| P32K | 与P16K相同 | 同上 | 32768 |

两者使用完全相同的修复后Query合同、原闭合JSON Schema、Solver、独立数学Verifier和数值精度。query_t必须为原始数值字符串，不允许单位或生成后修复。每个条件独立的3/3完整证明通过，才达到冻结三题的工程门槛。

CPU准备见 [V2修复报告](NVIDIA_SYNTHETIC_V2_REPAIR_REPORT.md)，本次批准改变的是是否执行真实API调用。准备文件里的 `live_execution_this_task=false` 属于此前CPU准备阶段的历史记录，保留不覆盖；本次真实调用在独立activation/events/响应记录中体现。

首次执行中，P16K第一条返回HTTP200、finish_reason=length，16384输出Token全部为Thinking，最终JSON为空。随后交互执行会话消失，第二条请求没有观察到最终状态；P32K未启动。具体基础设施原因没有证据，不能把该请求记成成功、失败HTTP或零Token。

独立后台续跑复用第一条已观察的原始HTTP响应，不重复生成它；重新请求结果未知的第二条及余下Seed。导入响应、新请求、原始配置与中断来源分别记录。原P16K第二条可能已在远端执行，其usage未知，因此全流程请求数及总Token只能报告已观察部分与未知部分，不能假装完整计费。

产物：

- [首次调用与中断记录](../data/processed/round3/nvidia_gravity_targeted_v2_20261115_live_r1/)。
- [后台续跑与逐题结果](../data/processed/round3/nvidia_gravity_targeted_v2_20261115_live_r2_continuation/)；按P16K/P32K分目录，禁止混合候选挑选3/3。

Thinking始终作为未验证审计材料，不作为SFT Completion；验收对象仅为原始最终JSON Proof。原R4、V2重审和SFT预览目录均冻结保护。

## P16K 已完成结果

| Seed | finish_reason | 输出Token | 结果 |
|---|---|---:|---|
| gravity-seed1 | length | 16384 | Preview；无最终JSON |
| gravity-seed2 | stop | 16181 | Accepted；全部g_i/OLS/Query时间/距离/末答通过 |
| gravity-seed3 | stop | 9928 | Accepted；全部字段通过，query_t为原数值字符串 |

P16K为2 Accepted、1 Preview，三题均含Thinking，**未达到3/3工程门槛**。已报告输出Token共42493（含Thinking42273）；另有一次已观察到的340秒断连不含usage，以及首次会话中结果未知的第二题请求，不能将42493视为完整计费总量。后台P16K续跑耗时801.607秒，不包含导入的首题211.5秒请求及此前中断等待。

P16K相较旧R4多了两条正确完整证明，但每题单次随机采样、仅3题，不能将差异完全归因于Query提示修复。

## P32K完整结果与逐题对比

| Seed | P16K结果 | P16K输出Token | P32K结果 | P32K输出Token | P32K reasoning Token |
|---|---|---:|---|---:|---:|
| gravity-seed1 | Preview：length | 16384 | Accepted：stop | 9267 | 9161 |
| gravity-seed2 | Accepted：stop | 16181 | Accepted：stop | 16545 | 16441 |
| gravity-seed3 | Accepted：stop | 9928 | Accepted：stop | 16006 | 15890 |

P32K三条均满足：原始HTTP200、逐字响应对应、stop、非空严格JSON、无重复/额外键、完整Schema、每例g_i、OLS、原Query数值字符串、Query未舍入距离、官方末答及三集合Leakage Audit。三条都实际含Thinking，原文和usage保留；Thinking仍为unverified审计材料，不被JSON证明认证，也不进入SFT Completion。

两个条件的三条**实际HTTP请求**已经逐题比较，除`max_tokens=16384/32768`外完全相同。模型、Prompt、temperature、top_p、Thinking、Schema均无漂移；数学Verifier、精度和接受规则未修改。

| 指标 | P16K | P32K |
|---|---:|---:|
| Accepted / Preview | 2 / 1 | 3 / 0 |
| run_status | completed_with_previews | passed |
| engineering_gate_passed | false | true |
| 完成三题 | 是 | 是 |
| 截断率 | 1/3 | 0/3 |
| 三集合完整题目Audit | 3/3通过 | 3/3通过 |
| 已观察HTTP尝试 | 4 | 5 |
| 已观察失败尝试 | 1次断连 | 1次HTTP500、1次断连 |
| 已观察重试 | 1 | 2 |
| 返回候选总输出Token，含Thinking/截断 | 42493 | 41818 |
| 返回候选平均输出Token | 14164.333 | 13939.333 |
| 返回候选reasoning Token | 42273 | 41492 |
| 成功响应请求耗时总和，秒 | 669.163 | 740.790 |
| 含已观察失败的请求耗时总和，秒 | 1009.314 | 1299.003 |
| 条件后台续跑墙钟，秒 | 801.607 | 1301.458 |

P16K续跑墙钟不含原先已完成首题的211.502秒，以及首次会话中断等待；它不是首次至最终的完整P16K成本。P32K墙钟包含其全部观察到的请求/重试。首次授权开始到后台完成共3267.372秒（54.46分钟），含会话中断、恢复和两条件串行执行，不能当作理想API吞吐。

返回候选输出Token共84311，其中Thinking83765；不混入输入。三次已观察失败都未提供usage，另外首次P16K第二题中断请求的远端结果与usage未知，因此84311只是**已报告输出Token下界**，不是完整计费总量。已观察9次HTTP尝试，另有至少1次首次会话的结果未知请求。不能把未知Token填零。

P32K第二题用了16545 Token，说明更高预算在该次样本中提供了实际空间；第一题的成功输出只有9267 Token，所以该题的改进也可能包含随机采样与服务重试影响。每题各单个最终候选、n=3，没有统计显著或一般非劣效结论，也没有混合两条件候选凑3/3。

## 归档、完整性与后续范围

完整逐题Verifier字段见 [per_seed_results.jsonl](../data/processed/round3/nvidia_gravity_targeted_v2_20261115_live_r2_continuation/per_seed_results.jsonl)，汇总、请求同一性与中断限制见 [summary.json](../data/processed/round3/nvidia_gravity_targeted_v2_20261115_live_r2_continuation/summary.json)。条件目录保留 raw HTTP、原始Thinking、finish、usage、latency、Schema、候选、Accepted/Preview、Manifest与Leakage Audit。

当前V2通用筛选器返回的`offline_reaudit_complete=true`仅表示筛选步骤完成，本轮**确实执行了Hosted API**；`continuation_manifest`与`actual_live_execution`明确记录执行方式，不能把该历史字段解读为本轮零调用。原CPU准备文件、原R4、V2重审和6条SFT预览均保持不变。

P32K通过了进入下一阶段讨论的冻结Seed工程门槛。本次没有执行新Seed扩展、Gravity SFT转换或自动合并；`lineage_status=unavailable`仍需保留。Roman/Equation先前数学结果继续以原V2独立报告为准，没有为了这次Gravity测试重跑或覆盖它们。

完成后重新执行V2相关CPU回归：14/14通过，2.617秒；该回归禁止外网、使用Mock响应，不额外消耗Hosted Token。全部真实结果已由独立离线脚本再次核对HTTP原文、数学字段、实际参数和历史Hash；密钥模式扫描及认证Header检查通过。
