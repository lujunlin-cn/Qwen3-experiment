# Gravity V2：真实 NVIDIA API 全量测试

状态：执行中。用户已批准真实 API 测试；两条件各使用同样的3条冻结Gravity Seed，不进行新题扩展、SFT合并或GPU训练。最终质量以逐条V2 Verifier为准，不能把该文档的创建视为实验已通过。

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

P32K已开始，最终结果尚待实际响应和Verifier，不能与P16K候选混合填补失败题。P16K相较旧R4多了两条正确完整证明，但每题单次随机采样、仅3题，不能将差异完全归因于Query提示修复。
