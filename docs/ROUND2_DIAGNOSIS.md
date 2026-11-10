# 第二轮诊断报告

日期：2026-11-10。审查起点：`5059a3e`。状态：Phase A 与 CPU 数值回归；没有启动 GPU、没有新模型质量结果。按最新决定，DPO **只做 Gravity DPO-B**，PPO/GRPO 只训练 Equation Numeric。

## 1. 第一轮证明了什么

冻结 Run 为 `course8h_20261109_r2_cuda_graph`，执行源码为 `98787e2`，模型 Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`。恢复原始证据见 [第一轮总结](GPU_VALIDATION_SUMMARY_20261110.md)。

| 方法 | 真实 Optimizer Steps | 训练采样峰值 GB |
|---|---:|---:|
| SFT | 72 | 1.724 |
| DPO | 4 | 2.416 |
| PPO | 3 | 4.796 |
| GRPO | 2 | 4.412 |

资源结果表明当前 32 GB 设备上真实更新可控制在 8,000,000,000 bytes 的采样阈值以下；不等同于物理 8 GB 显卡验证，也不保证两次采样之间的瞬时峰值。GRPO 第六轮被时间预算中断，不是显存超限。独立 final 评测没有完成，不能用四题 Tune 的选择结果代替 final。

质量证据不足：DPO 只有 9 个偏好对；GRPO 多轮零方差且零更新。三次 PPO 更新与两次 GRPO 更新不足以判定方法能否稳定改善数学正确率。SFT 已改善部分截断与长度，不能预设只有 RL 才能改善。

## 2. NeMo：先独立文本诊断，再用程序复核

原数据未变。逐输出交付在 [per_output_diagnosis.jsonl](../data/processed/round2/round2_20261110_phaseab_v1/nemo_diagnosis/per_output_diagnosis.jsonl)。其中每条包含完整 Seed、重建的 Generator Prompt、原始模型输出、Solver 参考、旧/新 Verifier 结果、Screening 原因、失败位置与修复建议。

独立诊断器为本轮 Codex 的离线文本审查，不是另一次 Qwen GPU 调用，也不是数学裁判。先写入 [model_hypotheses.jsonl](../data/processed/round2/round2_20261110_phaseab_v1/nemo_diagnosis/model_hypotheses.jsonl)，随后程序重新求解和筛查。诊断建议没有参与 Accepted 决策。

历史代理只存了请求 Hash，未存请求正文。因此历史提示词标为“从冻结配置与 Seed 重建”，不能声称恢复了当时的 HTTP 请求正文。新代理已记录实际发送的本地合成请求；真实 SDK + 模拟服务测试检查每个 Seed 的完整题面实际出现在请求中，并拒绝未渲染的 Jinja 字段。

| 阶段 | 分母 | 通过/产出 | 失败或未通过 |
|---|---:|---:|---:|
| Seed 独立完整性/参考审查 | 9 | 9 | 0 |
| Generator HTTP/非空输出 | 18 | 18 | 0 |
| Generator Prompt 原题保护契约 | 9 Prompt | 0 | 9 |
| Trace 顶层结构与 rules 类型初检 | 9 Trace | 5 | 4 |
| Solver：原 Seed 的重新参考求解 | 9 | 9 | 0 |
| Solver：原题保护失败后的 Prompt Variant | 9 | 0 次尝试 | 9 条在保护检查处停止 |
| Verifier：输出契约与数学证明 | 18 | 0 | 18 |
| Screening，含三集合 Audit | 18 | 0 | 18 |
| Accepted | 18 | 0 | 18 Preview |

这些阶段诊断有重叠，不能把同一输出的 Schema、Verifier、Leakage 错误相加当成不同样本。5 条结构初检通过的 Trace 均未通过完整数学证明。Roman 的 3 个 Prompt 另触发 `numeral_exposure_audit_failed`，原因是题面已经无法按契约解析，不是发现了 3 次真实测试集泄漏。

### 逐题故障归因

| Seed | Prompt Variant | Trace Variant |
|---|---|---|
| Roman 1 | 原题被替换，新增错误 `100 -> LXXX` | 缺少必填 `answer` |
| Roman 2 | 添加错误映射与边界标记 | `52+22=74` 但拼接成非规范 `LIIXXII` |
| Roman 3 | 输出 `NEWLINE`、边界标记 | 正确末答中夹入 `0 -> ""`，不满足正值规范分量 |
| Gravity 1 | 输出分隔文本 | 把 g 全写为 2；Query 时间取自示例，末答也错误 |
| Gravity 2 | 输出 `NEWLINE` | g、Query 距离错误，末答抄自示例并含单位 |
| Gravity 3 | 输出 `NEWLINE`、边界标记 | g=2、距离=.5 与数据不符 |
| Equation 1 | 遗漏原题约束句 | rules 是目录列表；示例和 Query 不闭合 |
| Equation 2 | 多行被连成一行并插入 `[` | rules 类型错误，示例只返回 1 项 |
| Equation 3 | 遗漏原题约束句 | 非法 JSON 转义，示例不全、Query 不是计算结果 |

### 八项根因检查结论

1. **Seed 信息**：9/9 原 Seed 参考求解与独立验证通过。Roman 依赖声明的标准 Roman 先验；Gravity 已给关系式；Equation 给出了示例、查询及公开 Numeric24 约束。不能把无限规则空间下的唯一性当作已证明。
2. **Prompt 歧义**：有。引号中的引导句、输入边界与自然语言“newline”指令容易被抄到输出；Equation 大段规则说明容易被复制成目录。
3. **Schema**：有真实违规。4/9 Trace 未通过顶层初检；其余也不保证值正确。
4. **Solver**：本次 9/9 参考一致，未发现错误参考。不能推广为所有样本无误。
5. **Verifier**：本组没有发现应收而未收的样本。拒绝零 Roman 分量是契约要求；Equation 全部显式计算应精确验证，不能使用官方 1% 最终评分容差检查中间计算。
6. **Screening**：重筛仍为 0/18；空 Accepted 状态正确为 `empty_accepted`。不放宽标准来提高通过率。
7. **题型差异**：Roman 主要为复制/规范分解；Gravity 为变量理解/算术/代入；Equation 为规则指派、结构和全例一致性。
8. **NeMo 配置传递**：真实 SDK 模拟测试通过 R0/A/B/C 的 Seed 注入。但发现 `LLMStructuredColumnConfig` 单独使用没有发送 HTTP `response_format`；B 已显式补充 JSON Schema，测试观察到 9/9 Trace 请求携带它。历史请求正文缺失，不能据此反证历史链路绝无字段问题。

更完整的对照设计见 [NeMo 修复方案](NEMO_QUALITY_REPAIR.md)。

## 3. 算法/工程风险

已修复：截断 DPO Rejected；Notebook 与 Runner 的 Dropout、PPO γ/λ 差异；β 不显式传入；加载Notebook API覆盖Worker注册随机种子；PPO standalone 静默缩减生成预算；完成调度却留下 final 成功时间；请求正文不可追溯；第二轮奖励/偏好定义只存在于 Smoke 而未接入 Worker 的风险。详见 [代码审查](ROUND2_CODE_AUDIT.md)。

尚未验证：新 Schema 在 vLLM 0.11.0 上的约束解码；真实 Gravity 正确轨迹的长短分布与覆盖率；新 Reward 的真实分布与反向峰值；greedy 配对结果；完整 final 测试耗时。Draft 配置不可交给旧八小时总调度器直接运行。

## 4. Validation 规模与限制

| 题型 | Train | 官方 Validation | Holdout | 第二轮 Tune | 第二轮锁定 final |
|---|---:|---:|---:|---:|---:|
| Roman | 1260 | 158 | 158 | 48 | 98 |
| Gravity | 1277 | 160 | 160 | 48 | 100 |
| Equation | 586 | 73 | 73 | 24 | 37 |

保守排除第一轮 Tune+final 的每类 12 题，避免继续使用已暴露的面板。剩余 Validation 按固定随机种子划分；官方 split 不动。Phase C 每类 8 题是 Tune 的子集，不是新增 final。Hash 见 [面板 Manifest](../data/processed/round2/round2_20261110_phaseab_v1/panels/manifest.json)。

Equation final 37 题每题仍对应 2.70 个百分点。若只有增加正确而无回退，6 个不一致对的未校正双侧 McNemar p=.03125，已相当于 16.2 个百分点；三主比较的保守首次 Holm 门槛需约 7 个单向不一致对。更小的改进很可能无法检出。未显著不代表等效；DPO 没有观察到下降也不是总体非劣性的统计证明。

交付的扩大面板仅解决“四题跳25个百分点”的粗糙问题，无法保证足够功效。最终以逐题配对表、Wilson 区间、精确 McNemar 和长度配对分析报告，不使用 Holdout 补调参。

当前审核范围已收敛：PPO/GRPO只运行同一复合奖励 R1=F+.20P+.05S，不做多奖励消融，长度只统计。Gravity可用g=2d/t²检查关系式，但公式出现不等于代入正确；DPO-B仍需两边通过独立数值证明。
