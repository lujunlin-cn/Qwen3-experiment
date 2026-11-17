# Gravity SFT C2 定向修复与条件 DPO 最终报告

Run：`gravity_sft_c2_dpo_20261116_r1`。最终状态：`no_preference_signal`。

**已完成真实C2增量SFT：20次Optimizer更新，74条Gravity Train教师、2轮；原SFT C保持不变。**
26题诊断：C2最终完整证明认证0/26，官方答案1/26；48题Tune：证明认证0/48，官方答案0/48。
C2共完成312条真实Rollout，得到0/26个严格DPO-B偏好对；门槛为16对。
**本轮未执行DPO训练、DPO Checkpoint重载或Locked Final比较；不得将C2训练结果当作DPO收益。**

## 1. 来源、权重与数据隔离

原SFT C Adapter SHA256：`7033241d41efb086762df20789ce2b70764bb3a5c524e29b8d8f846d3254e3f4`。Base为Qwen/Qwen3-0.6B，Revision `c1899de289a04d12100db370d81485cdf75e47ca`。原C、R1、历史DPO/诊断/增强实验产物逐文件Hash冻结，最终保持不变：`True`，覆盖243个文件。

64条来自历史R1真实训练panel，10条来自C真实使用的Real Control输入及actual inclusion manifest。全部source_id/source_row/Prompt逐条核对实际SFT Train；重新做实际Validation/Holdout Prompt去重，保留原Prompt与答案。不是复用B组的10条合成Gravity，也没有生成新的数学题。

C2教师来源统一标记`solver_generated_rendered`，同时保留`historical_r1`/`added_real_c`原始分组及真实Train行号。Python Solver计算、独立Verifier复核、Python Renderer生成自然语言；不伪称LLM自主CoT。

C2 Adapter：`/home/hajimi2025/Qwen3-experiment/.autodl/gravity_sft_c2_dpo_20261116_r1/C2/adapter`；SHA256：`27bcd5e2082ed628c6be3a1a2819fb81681e12c492668c0ec388540249609b18`。Optimizer SHA256：`518758f50174462dbe2f847053488a89abe63fea414fa67556e05c835d5c0b4a`。未继承其他算法的Optimizer State。

## 2. 教师与最终通道修复

Notebook01新增`gravity_c2_solver_reference()`，复用既有`solve_gravity_prompt()`、公开Prompt解析器和独立`_gravity_verifier_fit()`，仅补充展示所需的公开观测与OLS数值项。原数学Solver、公式、精度、NL Verifier均未改。

Notebook03新增`render_gravity_c2_final_proof()`、`make_gravity_c2_record()`、`encode_gravity_c2_final_example()`和显式训练封装。展示每个公开(t,d)、每项t²d/t⁴、两项数值求和、OLS数值代入、拟合g、Query时间、未舍入距离和两位小数Boxed Answer。无需逐个输出g_i。

完整证明写入assistant.content，assistant消息不提供reasoning_content。固定Qwen3模板保留空`<think>…</think>`，最终通道包含全部证明。Prompt标签为-100，其后全部Token（包括数学和EOS）参与原causal SFT loss。保留默认encode_chat_example和默认SFT数据入口；train_augmentation_branch只增加显式可选encoder参数，默认行为不变。

教师数学/实际序列化后的原NL Verifier：74/74通过；全量数学Token监督74/74。最长完整Chat为1021 Token，上限4096，未截断。每轮实际监督53975 Token。

为避免原有限语法解析器混淆，“全部观测”用Observation编号展示，“数值求和”用Weighted data sum/Fourth-power sum标签；所有数值保留。原Verifier会把“For t=”误当Query时间，部分sum(...)文本也可能被符号正则误读。本轮只采用清晰的教师表达，不修改Verifier、数学容差或历史结果。

教师样例（来自真实Train，最终回答部分）：
```text
1. Read all public time-distance examples:
Observation 1: t = 3.03s, distance = 45.86 m
Observation 2: t = 1.86s, distance = 17.28 m
Observation 3: t = 3.49s, distance = 60.84 m
Observation 4: t = 1.29s, distance = 8.31 m
2. Fit g = 2*sum(t^2*d)/sum(t^4) using all public examples (OLS through the origin).
Compute t^2*d and t^4 for every observation:
3.03^2*45.86 = 421.036074
3.03^4 = 84.28892480999998
1.86^2*17.28 = 59.78188800000001
1.86^4 = 11.96883216
3.49^2*60.84 = 741.0372840000001
3.49^4 = 148.35483601
1.29^2*8.31 = 13.828671
1.29^4 = 2.76922881
Weighted data sum: 421.036074 + 59.78188800000001 + 741.0372840000001 + 13.828671 = 1235.683917
Fourth-power sum: 84.28892480999998 + 11.96883216 + 148.35483601 + 2.76922881 = 247.38182179
Numerical OLS substitution:
2*1235.683917/247.38182179 = 9.990094729344825
3. The fitted g is 9.990094729344825; apply distance = 0.5*g*t^2 to Query t=4.98.
d = 0.5*9.990094729344825*(4.98)^2 = 123.8791726628217
4. The unrounded Query distance is 123.8791726628217; round to two decimal places.
Final answer: \boxed{123.88}
```

## 3. 真实C2训练、梯度及重载

从SFT C初始化，BF16 LoRA r8/alpha16/dropout0；AdamW LR2e-5、weight_decay0.01、eps1e-8、microbatch1、累积8、clip1、2epochs、seed20261116、max_length4096、分块投影64。沿用现有C训练配置，只改变教师目标与其序列化、训练数据集规模。

真实更新20次，监督Token总数107950，改变392个LoRA张量；训练耗时105.30s，含初始化/保存墙钟116.44s。Torch本进程峰值reserved=1522532352 bytes。逐步Loss、gradient_norm、监督Token数和耗时见`sft_train/training_curve.jsonl`。

重载状态：`completed`；最大probe logprob偏差：`0.0`。同一冻结Base/Tokenizer的7项文件逐一核对；没有重新初始化Base冒充C。

## 4. C与C2同条件Greedy评测

原始公开Prompt，Thinking开启，Greedy（do_sample=False，vLLM temperature=0哨兵，非温度1采样）、top_p1/top_k-1、max_tokens16384，EOS与Thinking均计入Completion长度。未发送答案或Solver证明。26题为冻结DPO Train诊断；48题为冻结Validation Tune。Locked Final100也是历史冻结Validation子集，并非官方Holdout、也非首次盲测。

C的26题复用刚完成诊断的真实Greedy输出，逐题校验Prompt Token/模型Hash；C2诊断同用batch1。Tune中C/C2均重新生成，同用batch8和同一后端。没有根据Tune调整训练或解码。官方最终答案与原自然语言Verifier分开统计；OLS为原精度下可认证的最终全样本OLS，并非关键词命中。保守解析器存在表达覆盖限制，未认证不等于全部数学都错误。

### diagnosis

|指标|SFT C|SFT C2|
|---|---:|---:|
|Accuracy|26.92%|3.85%|
|完整最终证明通过率|0.00%|0.00%|
|OLS认证率|0.00%|0.00%|
|Truncation|3.85%|19.23%|
|平均Completion Token|2060.65|3411.12|
|中位Completion Token|1276.00|325.00|
|官方答对平均Token|1478.29|613.00|
|正确且证明认证平均Token|—|—|
|答对数量|7/26|1/26|
|证明认证数量|0/26|0/26|
|格式有效|16/26|3/26|

Accuracy Wilson95%：C [0.13704448322018375, 0.4608304037917617]，C2 [0.006821984993543487, 0.18892778903034757]。McNemar exact p=0.031250；准确率配对Bootstrap95%=[-0.38461538461538464, -0.07692307692307693]（C2减C）。这些小面板/单训练Seed结果不能保证总体泛化或统计非劣。

|ID|C答对|C2答对|C证明认证|C2证明认证|C Token|C2 Token|
|---|---:|---:|---:|---:|---:|---:|
|c04634c0|0|0|0|0|1260|272|
|2ce67ec9|0|0|0|0|1530|220|
|8eecb224|0|0|0|0|492|410|
|6cfb1f29|0|0|0|0|1079|309|
|d54f5b26|1|0|0|0|1315|629|
|eba1891f|0|0|0|0|1292|379|
|bdcccbab|1|0|0|0|1198|272|
|ae5e65cf|1|1|0|0|956|613|
|0ab3d1de|0|0|0|0|7000|16384|
|736cdc47|0|0|0|0|1323|262|
|42fee270|0|0|0|0|1334|16384|
|f19b29c6|1|0|0|0|1443|430|
|76d51969|0|0|0|0|1030|335|
|5fe169fc|1|0|0|0|1630|244|
|16b45644|1|0|0|0|955|16384|
|ff9540e2|0|0|0|0|1141|309|
|gravity-final-001|0|0|0|0|182|155|
|gravity-final-002|0|0|0|0|16384|327|
|gravity-final-003|0|0|0|0|3630|16384|
|gravity-final-004|0|0|0|0|710|277|
|gravity-final-005|0|0|0|0|437|147|
|gravity-final-006|0|0|0|0|316|126|
|gravity-final-007|1|0|0|0|2851|578|
|gravity-final-008|0|0|0|0|1334|323|
|gravity-final-009|0|0|0|0|2198|16384|
|gravity-final-010|0|0|0|0|557|152|

### tune

|指标|SFT C|SFT C2|
|---|---:|---:|
|Accuracy|27.08%|0.00%|
|完整最终证明通过率|0.00%|0.00%|
|OLS认证率|0.00%|0.00%|
|Truncation|0.00%|16.67%|
|平均Completion Token|1154.40|3071.33|
|中位Completion Token|1173.50|365.50|
|官方答对平均Token|1216.31|—|
|正确且证明认证平均Token|—|—|
|答对数量|13/48|0/48|
|证明认证数量|0/48|0/48|
|格式有效|38/48|8/48|

Accuracy Wilson95%：C [0.1656595762745644, 0.40996964982240613]，C2 [6.938893903907228e-18, 0.07410012966611748]。McNemar exact p=0.000244；准确率配对Bootstrap95%=[-0.3958333333333333, -0.14583333333333334]（C2减C）。这些小面板/单训练Seed结果不能保证总体泛化或统计非劣。

|ID|C答对|C2答对|C证明认证|C2证明认证|C Token|C2 Token|
|---|---:|---:|---:|---:|---:|---:|
|4707a5ad|1|0|0|0|1054|348|
|cc716361|1|0|0|0|1121|16384|
|eaff1801|0|0|0|0|1540|338|
|9846834b|1|0|0|0|1351|497|
|ca7325ee|1|0|0|0|1240|411|
|177d60f2|0|0|0|0|1448|293|
|6b01cc50|1|0|0|0|1077|128|
|618c69cb|0|0|0|0|863|16384|
|397d6282|0|0|0|0|1176|196|
|5a4be7fa|1|0|0|0|1268|368|
|7a895e9c|0|0|0|0|994|381|
|c50cd56e|1|0|0|0|1101|275|
|ff039a4e|1|0|0|0|1255|446|
|58e63752|1|0|0|0|1151|340|
|4b9e883a|0|0|0|0|1147|201|
|1d9e08a8|0|0|0|0|990|417|
|b7ebfc5a|0|0|0|0|1399|16384|
|7c30fc77|0|0|0|0|1316|259|
|02beaa05|0|0|0|0|917|225|
|97d0bad8|0|0|0|0|1188|311|
|40ad5039|0|0|0|0|1056|148|
|b037cf33|0|0|0|0|1506|392|
|ce2df062|0|0|0|0|949|393|
|06b3ee9d|1|0|0|0|1729|193|
|634ebedf|1|0|0|0|929|1907|
|90e893ac|0|0|0|0|638|363|
|b303bb5a|0|0|0|0|1228|611|
|2d1d1cea|0|0|0|0|1289|507|
|341252e5|0|0|0|0|1422|1585|
|f61f3d46|0|0|0|0|750|349|
|cd42f555|0|0|0|0|1210|481|
|bfaa2811|0|0|0|0|911|345|
|66df03f5|0|0|0|0|1401|16384|
|e2f56016|0|0|0|0|1145|239|
|f6eef71c|0|0|0|0|661|304|
|9f380129|0|0|0|0|1271|335|
|8ec8d10c|0|0|0|0|1185|303|
|38d47621|0|0|0|0|1316|16384|
|ffe680c2|0|0|0|0|1446|16384|
|c2f263df|1|0|0|0|1109|583|
|3a1ae7b5|0|0|0|0|964|16384|
|8ed96b4b|0|0|0|0|829|202|
|0f17279e|0|0|0|0|923|357|
|e62041e4|0|0|0|0|796|501|
|e560f574|0|0|0|0|1274|232|
|d3f8c793|1|0|0|0|1427|371|
|d29824cf|0|0|0|0|1171|16384|
|d26230fc|0|0|0|0|1280|217|

## 5. 自动DPO Readiness

同一C2、26题，每题先8条（Temperature1、top_p1/top_k-1、Thinking、16384上限）；不足16对时全26题各补4条，最多12条/题。保存原始Completion、Prompt/Response IDs、逐Token Behavior Logprobs、finish_reason、Policy Fingerprint及真实/合成Train Prior来源。没有借用教师答案充当Rollout。

沿用Notebook04原DPO-B构造：同题同Policy/Prompt IDs；两边均官方答对、原NL Verifier认证、stop且未截断；Chosen最短、Rejected最长，差≥64Token且≥长答20%，每题最多一对。错误、截断和unverifiable不能成为Rejected。

实际Rollout 312条；最终答案通过11，完整最终证明认证0，OLS认证0，截断2。有效偏好0/26，覆盖率0.00%；最终状态`no_preference_signal`。

非互斥失败原因：
```json
{
  "missing_or_multiple_numeric_boxed_answer": 223,
  "official_final_answer_incorrect": 295,
  "final_answer_not_rounded_public_ols": 304,
  "full_trace_boxed_format_missing": 272,
  "all_example_ols_fit_not_demonstrated": 306,
  "fitted_g_missing_or_inaccurate": 306,
  "unrounded_query_distance_missing_or_inaccurate": 306,
  "incorrect_numeric_equality_in_prose": 26,
  "missing_verifiable_physical_relation": 263,
  "query_time_missing_or_incorrect": 242,
  "unsupported_arithmetic": 20,
  "not_complete_stop": 2,
  "empty_or_unclosed_final_section": 6,
  "incorrect_reported_g_value": 14,
  "incorrect_intermediate_arithmetic": 23,
  "incorrect_symbolic_arithmetic": 9,
  "unsupported_math_syntax": 4,
  "incorrect_additional_physical_relation": 1,
  "altered_public_observation_in_proof": 4,
  "incorrect_or_unverifiable_optional_indexed_g": 1
}
```

## 6. 条件DPO训练与Tune/Locked Final

DPO未完成；已有曲线记录更新0次。质量门槛未通过，未启动DPO。没有完整DPO训练、重载和Locked Final比较结果。不得以“实现了代码”代替真实训练完成。

## 7. 最终回答与边界

- **修复后的SFT是否真正学会？**教师契约已修复且真实参与训练，但48题Tune最终完整证明认证仍为0，不能声称C2真正稳定学会了在最终回答展示完整OLS。教师74/74通过不能替代模型生成通过。
- **是否产生足够的高质量偏好对？**未达到16对，实际为0对；停止DPO，没有降低标准。
- **下一步依据。**先看独立Verifier的具体失败是否为数学错误、缺步骤、格式或有限解析覆盖问题；不因失败自动加Epoch、换模型、改精度或扩大数据。本轮所有结论绑定固定配置和面板。

CPU实测：Ran 97 tests in 5.996s；通过状态：`True`。日志、阶段完成hook、逐题审计和权重Hash独立保存。本进程显存上限8,000,000,000 bytes，GB10共享UMA，峰值不是全设备显存。没有中止其他GPU进程、Hosted API调用或修改PPO/GRPO。

主要产物：`gravity_c2_teachers_74.jsonl`、`teacher_repair_audit.jsonl`、`sft_train/`、`sft_reload.json`、`comparison/`、`C_vs_C2_*.json`、`rollout8/`、`rollout4/`（若触发）、`readiness*.json`、`preferences_*.jsonl`、`final_status.json`、`artifact_manifest.json`。独立权重位于`.autodl/gravity_sft_c2_dpo_20261116_r1/`，不合并默认SFT入口。
