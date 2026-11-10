# NeMo 数据质量修复方案

状态：CPU 配置/真实 SDK 模拟传输已验证；R0/A/B/C 的真实 Qwen 生成均 **Not Run**。原始 Generator、Solver、独立 Verifier、Train/Validation/Holdout 三集合 Audit 保留。

## 1. 诊断依据与质量边界

第一轮真实 HTTP 18 次、失败0、输出2743 Token、总阶段71.09秒、设备采样峰值3,808,428,032 bytes；模型 Accepted=0。9 条确定性 Seed 的预审 Accepted 不能当作模型数据的 Accepted。

每条失败证据和阶段计数见 [诊断报告](ROUND2_DIAGNOSIS.md) 与 [逐输出 JSONL](../data/processed/round2/round2_20261110_phaseab_v1/nemo_diagnosis/per_output_diagnosis.jsonl)。例：Roman 2 的 `LIIXXII` 不等于规范74；Gravity 的 `.5` 不是目标时刻的位移；Equation 的规则目录不构成运算符指派。

独立模型诊断仅提出假设。最终筛查必须依赖程序复算；禁止“模型说对了”或只匹配末答就接受 Trace。

## 2. 单因素实验矩阵

所有组从同一批原始 Seed 文件读取。模型 revision、Seed=42、temperature=.3、top_p=.9、关闭 Thinking、Prompt cap1024、Trace cap2048、并发1、无自动纠错，均固定。每组9行、每行两列，18输出；总计72次计划输出。

| 组 | 唯一干预因素 | 其余环节 | 新 Run ID |
|---|---|---|---|
| R0 | 原配置复测，对照模型/服务差异 | 原 Prompt、原自由 JSON、无辅助上下文 | `nemo_round2_20261110_r0_r1` |
| A | 修改 Generator Prompt | 不启用 Schema 约束；不提供 Solver 上下文 | `nemo_round2_20261110_a_r1` |
| B | 修改 Trace Schema 与解码约束 | 原 Prompt；不提供 Solver 上下文 | `nemo_round2_20261110_b_r1` |
| C | Generator/Solver 协作 | 原 Prompt、原自由 JSON | `nemo_round2_20261110_c_r1` |

A：明确输入分隔符不可输出、必须复制全部原题而不是重写/作答；Trace 明确必填 answer、正位值分量、rules 对象与完整 example_outputs。A 只检验 Prompt 包的作用，不区分其中每一句的独立贡献。

B：每类题分成3条 Seed 的同质 shard，避免用一个宽泛 union 混淆题型；保持总请求数18。`LLMStructuredColumnConfig(output_format=...)` 加上显式 HTTP `response_format=json_schema`。Schema 管结构，Verifier 管数学；约束采样产生合法 JSON 不代表会计算。Prompt 列不施加 JSON Schema。

C：辅助上下文仅从公开原题求得。Roman 提供 query 的十进制位值；Gravity 提供原始 t/d、t² 和关系式；Equation 提供解析后的例子、Query、Solver 候选规则。不提供隐藏生成参数，不改原题或标准答案，不提供 Query 答案。记录上下文全文与 Hash，明确这是 Solver 辅助生成实验，不能解释为模型独立推理能力提升。

对照只比较 A−R0、B−R0、C−R0。R0 与旧结果的差别还可能来自代码/环境差异。禁止先把 A+B+C 合并再报告“某一项有效”。每类仅3 Seed，接受率只作描述，不主张统计显著。

## 3. 代码与配置

Notebook 01 直接包含：`smoke_trace_schema`、`smoke_solver_context`、`build_round2_data_designer`、`audit_equation_process`、`audit_closed_completion`。数学验证没有迁入 Runner。

当前有效编译配置为 [nemo_arms_v2](../data/processed/round2/round2_20261110_phaseab_v1/nemo_arms_v2)。先前 `nemo_arms/` 是发现 HTTP Schema 传递问题前的 CPU 快照，保留供审计，**不得用于 GPU**。v2 每组目录含 `design.json`、公开 Seed、逐 shard `compiled_config.json`。首次执行必须另建结果目录，不覆盖这些配置或历史实验。

真实 SDK 模拟测试执行72次模拟 HTTP，检查公开原题注入、输出数、B 的9个 Schema 请求、C 的上下文传递、原样响应筛查与三集合 Audit。模拟响应由测试程序提供，18/18 测试通过不是 Qwen 数据质量结果。

## 4. 接受契约

`Seed → Generator → Solver → Verifier → Screening → Accepted` 各阶段单独记失败原因；模型调用完成与质量验收分开。

- 原题、例子、运算符、数字、单位、查询与标准答案不可改。Prompt Variant 必须是允许引导句加逐字原题，再单独求解和验证。
- Roman：例子映射、每个正规范分量、值之和、拼接和 Query 末答全部检查。
- Gravity：`d=.5*g*t²`；每例 `g_i=2*d_i/t_i²`；给定 OLS 拟合、原 Query 时间、代入计算和末答逐项检查。中间数用比官方末答严格的数值精度；不更改官方末答评分。
- Equation：映射必须覆盖题面运算符；合法24规则和全局约束；所有 example_outputs 的完整顺序及精确字符串；Query 运算及末答内部一致性。中间不能因1%容差获得错误通过。
- 拒绝额外键、重复键、不可解析 JSON、未验证外围文本、缺步骤和未验证 reasoning_content。原始 Trace 禁止修复后冒充模型原答。
- SDK 若返回结构化 dict，只在其等于原始 HTTP JSON 时绑定传输；最终筛查使用原始 HTTP 文本，保留 SDK 值用于审计。不能用 SDK 修复/丢字段后的值掩盖生成违规。
- 三集合 Audit 必须通过；有限 Roman 映射暴露另报，不能声称不存在先验重合。
- `Accepted=0` 合法，状态必须 `empty_accepted`，`all_accepted_verified_and_audited=false`。所有未证实 Trace 只进 Preview。

NeMo 接受的闭合证明没有自由 Thinking。RL 的 `audit_closed_completion` 可以检查 Thinking 后的 JSON，但其 `exact_trace_pass` 范围仅是该证明；未解析的 Thinking 不被认证，不能据此当作 NeMo Accepted。

## 5. 每组必须落盘的产物

`config/seed/source hashes`、模型 revision、实际 HTTP 请求/响应、逐行 Solver/Verifier/Screening、逐阶段失败数、Token usage、阶段耗时、设备采样、`accepted.jsonl`、`preview.jsonl`、`accepted_manifest.json`、三集合 Audit。接受率分母固定为期望18输出；缺请求、截断、重复行另报，不从分母中删掉。

当前可检查的 Manifest 是 [历史输出严格重筛](../data/processed/round2/round2_20261110_phaseab_v1/nemo_diagnosis/strict_rescreen/accepted_manifest.json)：0 Accepted。A/B/C 尚无模型 Accepted Manifest；不预造空文件冒充实验已执行。

GPU 审核后先做 R0，确认环境；再 A/B/C 独立测试。任何组没有提升都保留原始输出和原因，不为演示产量放宽标准，也不自动并入 SFT。
