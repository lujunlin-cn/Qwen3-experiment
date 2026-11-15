# 三题型合成数据最终 Pilot：来源、成本与 SFT 预览

Run：`nvidia_synthetic_final_20261115_r1`；代码提交：`d4ac993827b556b16b86018d4244a3bb76d34fd6`；配置 SHA256：`7e71f6f2e5ba2fee62b3938806e331b1d90973bd143936f87eb5b6dbebf1b22c`。
全部本轮产物位于 [独立 Run 目录](../data/processed/round3/nvidia_synthetic_final_20261115_r1/)。原 R4、V2 重审和 Gravity P16K/P32K 均保持原 Hash。
本轮没有 GPU 训练、SFT 合并或 DPO/PPO/GRPO 修改。

## 执行状态

实验状态：`completed`。

## A. Train-only 来源证明

Generator/Train Prior/Prompt/Solver/Verifier 位于 Notebook 01；Full Trace Renderer 位于 Notebook 03。Runner 仅调度、归档、计算指标和约束请求。
专用定义加载器不执行 Notebook 的数据配置 Cell，生成进程通过文件读取边界只允许读取注册 Manifest、Train CSV 以及本轮已冻结的中间产物。Train CSV 只解析 `id/source_row/task_type/prompt/split`；不解析 answer。原始文件整体 SHA256 用于完整性校验，不将 answer 内容用作采样依据。
`prepare_read_audit.json` 和 `candidate_pool_freeze.json` 保存实际读取记录。`audit_started.json` 在候选冻结后才创建，然后独立进程加载真实三集合。没有根据 Holdout 重新采样。

| 题型 | Train 行数 | 经验采样特征 | 固定 Task Specification |
|---|---|---|---|
| Roman | 1260 | 公开示例及 Query 数字的出现频率；每题示例数；题内数字不重复 | 规范 Roman 1..3999 |
| Gravity | 1277 | Train OLS g、示例时间、示例数、原始字符串距离精度 | d=0.5gt²、OLS、最终两位小数 |
| Equation | 586 | 联合行布局：运算符/示例数/每运算符示例数/Query 运算符/Deduce-Guess；经验操作数对 | Numeric24、family/group 全局约束及合法规则采样 |

Roman 使用经验出现次数加权抽样；Gravity 每个参数对应一条观测来源。Equation 从经验布局联合抽样，以保留布局和 Query 类型的关联；左右操作数从经验操作数对抽样。运算符频率和左右边缘分布另外审计，不声称是独立采样。Train 中 Equation 的 Deduce/Guess 为 475/111；Guess 子型从公开 Prompt 的独立验证结果标记，不读取官方答案。
旧 Generator 的全局运算符回退已移除；Train 支持为空立即报错。新增 Gravity 显式采样参数入口复用原渲染计算，默认旧采样路径保留。所有原有 Solver 和数学 Verifier 的计算未改写。
[Prior Manifest](../data/processed/round3/nvidia_synthetic_final_20261115_r1/train_prior_manifest.json) 保存数据/源码/规范 Hash、每特征规则、ID 和行号列表 Hash、RNG 20261115 与时间。[完整 Prior](../data/processed/round3/nvidia_synthetic_final_20261115_r1/train_prior.json) 保存每一次经验观测的 Train 来源。每候选的 `prior_feature_lineage` 记录实际抽中的特征、值、来源 ID、source_row 和观测位置。
参数祖先 `prior_source_ids/rows` 与克隆真实样本的 `source_id/source_row` 分开；前者由 `audit_train_prior_lineage` 验证，后者继续由原三集合重复审计检查。结构相似性仅报告。
Roman 采用严格政策：Validation/Holdout 的相同目标映射、或者合成示例暴露这些真实目标映射，均拒绝训练资格；Query 命中真实公开示例单独报告。数学正确、Train 来源正确和无目标映射暴露是三个独立状态。

| 题型 | 候选 | 来源通过 | Solver Proof 通过 | Leakage 通过 | 全部门槛通过 | 选入 Seed |
|---|---|---|---|---|---|---|
| numeral_system | 20 | 20 | 20 | 0 | 0 | 0 |
| gravity | 20 | 20 | 20 | 20 | 20 | 10 |
| equation_numeric | 20 | 20 | 13 | 20 | 13 | 10 |

逐特征、逐样本来源审计见 [train_only_lineage_audit.json](../data/processed/round3/nvidia_synthetic_final_20261115_r1/train_only_lineage_audit.json)。

## B. 固定新 Seed 生成结果

| 题型 | 20 条中选入 | 未选入 | 是否达到 10 条 | 拒绝/未选原因（可重叠） |
|---|---|---|---|---|
| numeral_system | 0 | 20 | False | {"roman_holdout_test_real_queries_in_synthetic_examples": 20, "roman_validation_real_queries_in_synthetic_examples": 20, "roman_validation_same_query_in_real_targets": 16, "roman_holdout_test_same_query_in_real_targets": 13} |
| gravity | 10 | 10 | True | {"passed_gates_not_selected_quota": 10} |
| equation_numeric | 10 | 10 | True | {"independent_solver_proof_failed": 7, "passed_gates_not_selected_quota": 3} |

候选按固定 ID 顺序选取每题型前 10 条门槛通过样本；`passed_gates_not_selected_quota` 是配额外通过样本，不等同于数学或来源失败。没有补采样、放宽精度或改答案。完整候选、拒绝原因和 Hash 均归档。
分布指标见 [train_distribution_audit.json](../data/processed/round3/nvidia_synthetic_final_20261115_r1/train_distribution_audit.json)：数值采用 KS statistic/Wasserstein，离散特征采用 Total Variation；分别报告 20 条原始候选与最终 Seed 面板。样本很小，指标仅用于描述抽样偏移，不构成总体分布一致性的证明。筛选也会带来面板分布偏移。

| 题型 | 特征（全部候选） | KS | Wasserstein | TV |
|---|---|---|---|---|
| numeral_system | query_numbers | 0.136 | 4.943 | Unknown |
| numeral_system | example_numbers | 0.149 | 4.361 | Unknown |
| numeral_system | example_counts | Unknown | Unknown | 0.139 |
| gravity | g_ols | 0.251 | 1.419 | Unknown |
| gravity | times | 0.050 | 0.080 | Unknown |
| gravity | example_counts | Unknown | Unknown | 0.154 |
| gravity | distance_precision | Unknown | Unknown | 0.004 |
| equation_numeric | operators | Unknown | Unknown | 0.250 |
| equation_numeric | operands_left | 0.104 | 3.511 | Unknown |
| equation_numeric | operands_right | 0.078 | 4.024 | Unknown |
| equation_numeric | operand_pairs | Unknown | Unknown | 0.957 |
| equation_numeric | example_counts | Unknown | Unknown | 0.067 |
| equation_numeric | operator_example_counts | Unknown | Unknown | 0.212 |
| equation_numeric | query_operators | Unknown | Unknown | 0.411 |
| equation_numeric | query_types | Unknown | Unknown | 0.011 |

## C. 低推理成本对照

Ultra `nvidia/nemotron-3-ultra-550b-a55b`；SDK 0.10.0 内置 NVIDIA Provider；temperature=1.0、top_p=.95、reasoning_effort=low、并发 1。Roman/Equation 上限 16384，Gravity 32768。A 使用原完整模板与 bounded_precision；B 压缩表达，保留原完整 Numeric24 目录/全局约束和闭合 Schema。Gravity 的原数值 Query 字符串合同两条件相同。
仅发送 `seed_id/task_type/prompt`。隐藏规则、答案、Solver Trace、来源 Manifest 不发送。验收对象始终为原始 HTTP 对应的严格最终 JSON，Thinking 为 unverified 审计材料。未知 Usage 为 Unknown，并按请求 max_tokens 预留预算。
最大 18 次冻结 Seed A/B 加最多 30 次新 Seed，共最多 48 次 HTTP；Completion 总硬预算 500000。发送前持久化记账；底层 RetryTransport 和 RequestExecutor 设置 0 重试，另加实际发送 Guard。失败/中断不重发，同一 Seed×条件不可重复。

| 方案 | 题型 | 数学 Accepted | Completion 合计 | Reasoning 合计 | Completion/Accepted | 正确证明 Median Reasoning | Median latency 秒 |
|---|---|---|---|---|---|---|---|
| 历史 Full Thinking | numeral_system | 3/3 | 4970 | 4806 | 1656.667 | 749 | 14.206 |
| 历史 Full Thinking | gravity | 3/3 | 41818 | 41492 | 13939.333 | 15890 | 266.173 |
| 历史 Full Thinking | equation_numeric | 3/3 | 23191 | 22710 | 7730.333 | 9527 | 170.214 |
| Low Full A | numeral_system | 3/3 | 4543 | 4379 | 1514.333 | 887 | 22.607 |
| Low Full A | gravity | 0/3 | Unknown | Unknown | Unknown | Unknown | 342.511 |
| Low Full A | equation_numeric | 0/3 | Unknown | Unknown | Unknown | Unknown | 1.327 |
| Low Compact B | numeral_system | 0/3 | Unknown | Unknown | Unknown | Unknown | 1.751 |
| Low Compact B | gravity | 0/3 | Unknown | Unknown | Unknown | Unknown | 1.388 |
| Low Compact B | equation_numeric | 0/3 | Unknown | Unknown | Unknown | Unknown | 340.769 |

历史 Roman/Equation 使用 R4 的原始成功响应，Gravity 使用 P32K 原始响应，不重新调用。历史成本表只统计已收到的 Proof 响应；旧实验失败/中断可能 Usage 未知，不能当作完整计费总量，也不能与新单次请求成本无条件等同。历史 R3 Low Thinking 中断不作为对照。

| 题型 | 选择 Prompt | 冻结正确证明 | Reasoning 下降 | 允许新 Seed API |
|---|---|---|---|---|
| numeral_system | A | 3 | -0.184 | False |
| gravity | A | 0 | Unknown | False |
| equation_numeric | A | 0 | Unknown | False |

新 Seed 单次低推理结果：

| 题型 | 输出数 | Accepted | 接受率 | 平均 Completion | 平均 Reasoning | Completion/Accepted | Median latency 秒 |
|---|---|---|---|---|---|---|---|
| numeral_system | 0 | 0 | Unknown | Unknown | Unknown | Unknown | Unknown |
| gravity | 0 | 0 | Unknown | Unknown | Unknown | Unknown | Unknown |
| equation_numeric | 0 | 0 | Unknown | Unknown | Unknown | Unknown | Unknown |

跳过原因：`{"numeral_system": "frozen_quality_or_cost_gate_failed", "gravity": "frozen_quality_or_cost_gate_failed", "equation_numeric": "frozen_quality_or_cost_gate_failed"}`。

实际请求/预算：`{"request_count": 18, "completion_tokens_charged_or_reserved": 348607, "reported_completion_tokens": 4543, "unknown_usage_requests": 15, "keys": ["calibration:A:equation_numeric-seed1", "calibration:A:equation_numeric-seed2", "calibration:A:equation_numeric-seed3", "calibration:A:gravity-seed1", "calibration:A:gravity-seed2", "calibration:A:gravity-seed3", "calibration:A:numeral_system-seed1", "calibration:A:numeral_system-seed2", "calibration:A:numeral_system-seed3", "calibration:B:equation_numeric-seed1", "calibration:B:equation_numeric-seed2", "calibration:B:equation_numeric-seed3", "calibration:B:gravity-seed1", "calibration:B:gravity-seed2", "calibration:B:gravity-seed3", "calibration:B:numeral_system-seed1", "calibration:B:numeral_system-seed2", "calibration:B:numeral_system-seed3"], "blocked_requests": [{"event": "blocked", "key": "calibration:A:gravity-seed1", "reason": "duplicate_seed_condition_request", "at_unix": 1794757655.0979362}, {"event": "blocked", "key": "calibration:A:gravity-seed2", "reason": "duplicate_seed_condition_request", "at_unix": 1794757996.367691}, {"event": "blocked", "key": "calibration:A:gravity-seed3", "reason": "duplicate_seed_condition_request", "at_unix": 1794758339.4773746}, {"event": "blocked", "key": "calibration:A:equation_numeric-seed1", "reason": "duplicate_seed_condition_request", "at_unix": 1794758344.371838}, {"event": "blocked", "key": "calibration:A:equation_numeric-seed2", "reason": "duplicate_seed_condition_request", "at_unix": 1794758345.932532}, {"event": "blocked", "key": "calibration:A:equation_numeric-seed3", "reason": "duplicate_seed_condition_request", "at_unix": 1794758347.439622}, {"event": "blocked", "key": "calibration:B:numeral_system-seed1", "reason": "duplicate_seed_condition_request", "at_unix": 1794758349.333491}, {"event": "blocked", "key": "calibration:B:numeral_system-seed2", "reason": "duplicate_seed_condition_request", "at_unix": 1794758477.3291516}, {"event": "blocked", "key": "calibration:B:numeral_system-seed3", "reason": "duplicate_seed_condition_request", "at_unix": 1794758479.2072897}, {"event": "blocked", "key": "calibration:B:gravity-seed1", "reason": "duplicate_seed_condition_request", "at_unix": 1794758482.0794325}, {"event": "blocked", "key": "calibration:B:gravity-seed2", "reason": "duplicate_seed_condition_request", "at_unix": 1794758483.5534563}, {"event": "blocked", "key": "calibration:B:gravity-seed3", "reason": "duplicate_seed_condition_request", "at_unix": 1794758485.1077118}, {"event": "blocked", "key": "calibration:B:equation_numeric-seed1", "reason": "duplicate_seed_condition_request", "at_unix": 1794758586.4003768}, {"event": "blocked", "key": "calibration:B:equation_numeric-seed2", "reason": "duplicate_seed_condition_request", "at_unix": 1794758927.3286004}, {"event": "blocked", "key": "calibration:B:equation_numeric-seed3", "reason": "duplicate_seed_condition_request", "at_unix": 1794759269.35068}]}`。

## D. Solver-first 与 SFT 产物

所有通过新 Seed Gate 的样本独立重跑原 Python Solver，再以原独立 Verifier 检查闭合 Proof；正式预览使用 Solver Trace，不使用 LLM JSON 或隐藏 Thinking 作为 Completion。
Notebook 03 的新增显式入口复用 `_format_trace()/make_trace_record()/audit_trace_record()`。输出为编号自然语言步骤加 `Final answer: \boxed{...}`；来源明确标记 `solver_generated_rendered`，不声称为 Nemotron 自主 CoT。
本地 Qwen3-0.6B Tokenizer 计算实际 Completion/Chat/Supervised Tokens；空白分词 proxy 只作为原 Audit 的历史字段，不冒充真实 Token 数。无权重加载、无网络、无 CUDA。

| 题型 | Solver Verified | Renderer Passed | Training Eligible | 最终进入 SFT |
|---|---|---|---|---|
| numeral_system | 0 | 0 | 0 | 0 |
| gravity | 10 | 10 | 10 | 0 |
| equation_numeric | 10 | 10 | 10 | 0 |

| Solver-first 题型 | CPU Proof 合计秒 | 平均渲染 Completion Tokens | Hosted Completion Tokens | LLM Reasoning Tokens |
|---|---|---|---|---|
| numeral_system | 0 | Unknown | 0 | 0 |
| gravity | 0.035 | 123.600 | 0 | 0 |
| equation_numeric | 0.005 | 111.300 | 0 | 0 |

Solver-first 的 0 指不发生任何 Hosted LLM 请求；渲染文本自身的 tokenizer 长度另列。
Roman 不属于当前 `gravity/equation_numeric/bit_manipulation` 训练任务池；即使来源和映射检查通过，也只允许预览。本轮未通过严格映射 Gate 的 Roman 不伪装成合格 SFT 预览。
四个状态严格分开：数学 Proof Accepted → SFT Format Audit Passed → SFT Training Eligible（待审核资格）→ 最终进入 SFT。本轮最后一项恒为 0，`merge_into_sft=false`，不修改默认 USE_SYNTHETIC_DATA 或历史 Checkpoint。

渲染样例（首条合格 Seed）：

```text
1. Parse all public time-distance examples.
2. Fit g = 2*sum(t^2*d)/sum(t^4) using only those examples.
3. The fitted g is 13.09133128690559; apply distance = 0.5*g*t^2 to Query t=1.61.
4. The unrounded Query distance is 16.96701991439399; format the final numeric answer.
Final answer: \boxed{16.97}
```

## E. 最终生产决策与限制

1. **三题型均能从可追溯 Train Prior 构造新题**：60/60 来源通过；Roman/Gravity 的数学 Proof 为20/20，Equation 为13/20。严格 Seed Gate 最终选入 Roman 0、Gravity 10、Equation 10。数学可生成和训练资格分开。
2. **低推理的实际结果按题型独立判断**，不以隐藏 Thinking 自评。下表列出预注册 Frozen Gate 与新题表现：

| 题型 | 选择 Prompt | 冻结 Accepted | 正确证明 Median Reasoning 下降 | Frozen 扩展 Gate | 新题 LLM Accepted/尝试 |
|---|---|---|---|---|---|
| numeral_system | A | 3/3 | -18.42% | False | 0/0 |
| gravity | A | 0/3 | Unknown | False | 0/0 |
| equation_numeric | A | 0/3 | Unknown | False | 0/0 |

低推理中实际获得正确完整Proof的组：A/numeral_system:3/3。没有完整返回的题型不能按数学能力公平排名。Roman A的正确证明Median Reasoning为887，历史为749，下降比例为-18.42%（即增加18.42%），未达到20%门槛。不能只选较有利的平均长度来放行。
通过预注册Frozen质量/成本门槛的题型：无；进一步的新Seed质量以实际结果表为准，未调用不能视为通过。
缺失 Usage 为 Unknown。成本下降列只比较**已接受 Proof 的中位数**，不包含传输失败的未知计费；若总 Usage 不完整，不能把这列扩展解释为整条生产链总成本下降。
3. **高推理是否必要**只能基于收到的完整 JSON 判断。传输断连没有返回数学证明，属于未能验证，不能说模型算错，也不能据此证明 low 数学能力不足。历史 Full Thinking 的3/3只说明冻结题曾通过；Hosted 服务没有不可变权重 Revision，跨日比较存在服务与版本不透明限制。
4. **正式教师生产采用 Solver-first + 独立 Verifier + 现有 Renderer**。它不消耗 Hosted Completion/Reasoning Token；独立 Verifier 是必要环节，不能假定旧 Solver 的所有参考证明天然正确。
5. **已有20条待审核 SFT 训练候选**：Gravity 10、Equation 10。Roman 0；最终进入 SFT 0。每条有来源、公开题面重算、完整过程证明、泄漏审计和真实 Tokenizer 长度。
6. **Gravity/Equation 具备下一轮独立登记20–50条 Solver-first试点的工程基础**，但10条不是总体质量保证。本轮不扩大。Roman 的Train支持100个数字，其中94个也是Validation/Holdout目标数字；当前20条均暴露目标映射。不据此调整采样，不补采样，也不更改当前SFT任务池。
7. **无需继续要求 Ultra 550B 承担正式确定性数学教师生产**。保留本轮低推理结果作为独立LLM能力、成本与服务可靠性对照；是否用于辅助教师仅在实际门槛通过的题型范围讨论。

### 同一批冻结题的 Solver-first 对照

| 题型 | Solver Proof Accepted | Renderer Audit | Hosted Completion | Hosted Reasoning | CPU Solver+Verifier median 秒 |
|---|---|---|---|---|---|
| numeral_system | 3/3 | 3 | 0 | 0 | 0.000168 |
| gravity | 3/3 | 3 | 0 | 0 | 0.000202 |
| equation_numeric | 2/3 | 2 | 0 | 0 | 0.001080 |

这是同一批9条旧Seed的CPU数学/格式基准，来源仍未证明，不进入SFT。CPU计时是预热后的Solver+Verifier，不包含Tokenizer启动，也不能与HTTP端到端耗时当作同一种计时。
Equation冻结题有1条旧Solver参考证明被严格过程验证拒绝，新候选中另有3条出现同类参考选择问题：旧Solver按官方1%数值容差收集兼容规则，首个assignment可能只近似解释示例。原独立严格Verifier检查全部中间结果，拒绝这些证明。另4条新Equation候选答案不唯一。所有拒绝记录保留，未修改容差、标签或候选池。

### 逐题型 Token 均值（完整 Usage 才计算）

| 条件 | 题型 | Accepted/尝试 | 平均 Completion | 平均 Reasoning | Input合计 | Completion/Accepted |
|---|---|---|---|---|---|---|
| A | numeral_system | 3/3 | 1514.3333333333333 | 1459.6666666666667 | 1440 | 1514.3333333333333 |
| A | gravity | 0/3 | Unknown | Unknown | Unknown | Unknown |
| A | equation_numeric | 0/3 | Unknown | Unknown | Unknown | Unknown |
| B | numeral_system | 0/3 | Unknown | Unknown | Unknown | Unknown |
| B | gravity | 0/3 | Unknown | Unknown | Unknown | Unknown |
| B | equation_numeric | 0/3 | Unknown | Unknown | Unknown | Unknown |

已报告Completion下界：4543；已报告Reasoning下界：4379。只要有Usage未知，就不能将下界视为完整计费。Roman A平均Completion较历史下降约8.59%，但预注册门槛采用正确Proof中位Reasoning，不能以均值替换门槛。整个Run的实际总计费与整体Tokens/Accepted在Usage不完整时为Unknown。未知请求按各自max_tokens保留预算，不据此推断实际计费。
### 请求与数学质量分开统计

| 阶段/条件 | 题型 | HTTP发送 | HTTP200 | 断连 | HTTP错误状态 | 非空最终输出 | Schema通过 | 数学通过 | Accepted | Usage未知 |
|---|---|---|---|---|---|---|---|---|---|---|
| calibration/A | numeral_system | 3 | 3 | 0 | {} | 3 | 3 | 3 | 3 | 0 |
| calibration/A | gravity | 3 | 0 | 3 | {} | 0 | 0 | 0 | 0 | 3 |
| calibration/A | equation_numeric | 3 | 0 | 0 | {"503": 3} | 0 | 0 | 0 | 0 | 3 |
| calibration/B | numeral_system | 3 | 0 | 0 | {"503": 3} | 0 | 0 | 0 | 0 | 3 |
| calibration/B | gravity | 3 | 0 | 0 | {"503": 3} | 0 | 0 | 0 | 0 | 3 |
| calibration/B | equation_numeric | 3 | 0 | 2 | {"500": 1} | 0 | 0 | 0 | 0 | 3 |

完整错误、Thinking、finish_reason、Usage和响应原文见对应条件/Seed目录。相同Seed×条件无重复HTTP发送；预算账本记录未知请求的max_tokens预留。

### 逐特征来源示例（全部明细保存在候选文件）

| 候选 | 特征 | Train ID | Train source_row | 观测值 |
|---|---|---|---|---|
| numeral_system-final-001 | example_count | 72e3e38e | 4257 | 5 |
| numeral_system-final-001 | example_number_0 | 83258e90 | 4855 | 76 |
| numeral_system-final-001 | example_number_1 | 2345be90 | 1323 | 88 |
| numeral_system-final-001 | example_number_2 | 50126dab | 2970 | 45 |
| numeral_system-final-001 | example_number_3 | d15c3b07 | 7813 | 53 |
| numeral_system-final-001 | example_number_4 | 977d739d | 5612 | 92 |
| numeral_system-final-001 | query_number | cdc86263 | 7667 | 63 |
| gravity-final-001 | example_count | 437acd17 | 2499 | 4 |
| gravity-final-001 | g_source | ba598f58 | 6936 | 13.091104626634596 |
| gravity-final-001 | time_0 | bfd0e232 | 7141 | "3.77" |
| gravity-final-001 | time_1 | 3ea413e6 | 2320 | "2.87" |
| gravity-final-001 | time_2 | 9db488f0 | 5847 | "4.61" |
| gravity-final-001 | time_3 | 8c633efd | 5214 | "3.29" |
| gravity-final-001 | time_4 | 679b5ccf | 3872 | "1.61" |
| gravity-final-001 | precision_0 | 7054453b | 4164 | 2 |
| gravity-final-001 | precision_1 | ea79a0b1 | 8709 | 2 |
| gravity-final-001 | precision_2 | 0fd289d8 | 599 | 2 |
| gravity-final-001 | precision_3 | 4e01d71a | 2891 | 2 |
| equation_numeric-final-001 | layout | c2945f2c | 7240 | {"operator_counts": {"+": 1, "-": 2}, "query_operator": "*", "query_type": "guess", "guess_subtype": "guess_answer_ambiguous", "example_count": 3} |
| equation_numeric-final-001 | operands_+_0 | e5956ffa | 8534 | ["02", "46"] |
| equation_numeric-final-001 | operands_-_0 | 0bcffccd | 442 | ["61", "52"] |
| equation_numeric-final-001 | operands_-_1 | 8bebef83 | 5192 | ["55", "76"] |
| equation_numeric-final-001 | query_operands | 648e2da2 | 3752 | ["67", "35"] |

**三个“通过”与最终训练保持分离**：数学Proof Accepted；SFT格式Audit Passed；SFT Training Eligible（待审核资格）；最终进入SFT。各条状态联合表见 `teacher_status_comparison.jsonl`。本轮最后一项全部为0。

## CPU 回归与工程完整性

相关旧/新CPU回归 **74/74通过**（27.325秒）；独立本地Tokenizer回归 **2/2通过**（3.909秒）；正式登记后的专用回归 **35/35通过**。日志均归档。

本轮专用测试结果：`{"status": "passed", "command": ["/home/hajimi2025/Qwen3-experiment/.venv-data-designer/bin/python", "-m", "unittest", "tests.test_nvidia_synthetic_final", "-v"], "exit_code": 0, "elapsed_seconds": 15.776495239231735, "log_sha256": "87eb6df30417ca791fbda659fca0d15c539260d2e2a2347aa66733ff36afe726", "source_sha256": {"01_Synthetic_Data_Engineering.ipynb": "edef1a2650dabc2d9d54e86eb1b76d7b3a0aee7a91a60462e3f95bab4fd5193c", "03_SFT.ipynb": "07fe5785e6ccb344dae36a29f7a9680e451e8b7415e934ee08e4881a3e52fc72", "utils/run_nvidia_final_synthetic.py": "1280a3594a4d9013108e0f9eeb6eb66618cb12bb0058d059a47196e08d877a17", "configs/nvidia_synthetic_final_20261115.json": "42d1c988f1aac5fd2909c8c92599017bb7a610e30f35961b451c95d5ace2603c", "tests/test_nvidia_synthetic_final.py": "f523c87693c1611f956236db27e94dc9724f77ecdcbbbffe662577e9eaa4f4b9", "tests/test_nvidia_v2_repair.py": "37d862c836583b40a7d83e0e036209284d26a55e2d7e7668cdf8bad3ae9d3e4b", "utils/run_nvidia_v2.py": "491f0f1c746580b181ed3bbfe54d54138497529ce685fed319c97d8d74696bb1", "utils/run_nvidia_synthetic_smoke.py": "1b927b69a1bfdbedb14c067141db1833f40860c217de1cb3cfde701e3b509617", "utils/run_synthetic_smoke.py": "7aa90aa6e05e905e62ff89097057c2233663b10b24ff5c00780f88a7c300793e"}, "live_api_calls": 0, "gpu_used": false}`。
来源、Answer 不变性、读取隔离、全部候选、三题型正反例、唯一性、严格 Roman 政策、A/B 单变量、真实 SDK Mock 请求、无重试硬预算、Unknown Usage、Thinking 隔离、本地 Tokenizer、不可覆盖历史、SFT 默认行为及 DPO/PPO/GRPO 隔离均有 CPU 检查。
相关旧回归的实际命令/日志另存本 Run。历史开发期间失败日志保留在 `development_cpu_attempts/`，不把修复前失败写成通过。
隔离审计：`{"status": "passed", "changed_protected_paths": [], "secret_or_auth_header_files": [], "historical_runs_checked": 5, "gpu_training": false, "merge_into_sft": false, "dpo_ppo_grpo_unchanged": true}`。

## 复现命令

```bash
.venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py prepare
.venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py generate
.venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py audit
.venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py cpu-tests
# API 阶段要求 NVIDIA_API_KEY 已在进程环境中；不把 Key 放到命令或文件中
.venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py calibrate
.venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py run-new-seeds
.venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py render
.venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py report
```

已存在阶段拒绝覆盖；中断后只读取已保存响应，不重新发送已登记请求。新实验应独立登记新配置和 Run ID；本 CLI 固定本轮 Run ID，防止无意扩大。
