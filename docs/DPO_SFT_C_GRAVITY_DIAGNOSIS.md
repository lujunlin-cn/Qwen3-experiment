# SFT C Gravity：教师目标、Loss Mask与解码专项诊断

诊断Run：`dpo_sft_c_gravity_diagnosis_20261116_r1`。输入基线Git：`568680995b0d4a24558e33d7fef3b52e6ad075fd`。

**已确认两个阻塞：训练输出通道与DPO验收合同不一致，以及模型未稳定执行正确的全样本OLS。**74条Gravity教师的数值OLS、g、未舍入Query距离和答案均正确，必要步骤也全部参与Loss；但实际序列化把数学步骤放在`<think>…</think>`内，最终回答只监督`Final answer: \boxed{...}`。因此即使完美复现训练目标，现有“只验收Thinking后的完整证明”合同也不会接受。

本次只新增26条真实GPU Greedy回答，复用原312条Sampling。Greedy点估计更高，按题配对Bootstrap区间略排除0，但配对sign-flip检验p=0.082；小样本推断不一致，视为方向性证据，不能宣称稳健显著优于Sampling。 完整证明和OLS认证属于保守有限语法的认证数，不等同于所有未认证过程均错误；不能由0认证直接推出模型从未学会OLS。

## 1. 实际权重、数据及历史来源

Policy仍为SFT C，Adapter SHA256：`7033241d41efb086762df20789ce2b70764bb3a5c524e29b8d8f846d3254e3f4`。Base为Qwen/Qwen3-0.6B，Revision `c1899de289a04d12100db370d81485cdf75e47ca`；7项Base/Tokenizer文件逐一匹配原始指纹。BF16 LoRA，r8、alpha16、dropout0。没有加载其他RL权重，没有训练或保存权重。

历史64条使用真实R1的`recovered_course_r2/panels/sft.jsonl`，核对其原协议Hash，并从Git恢复与真实协议SHA256匹配的Notebook03及Worker，而非用当前格式猜测历史目标。Notebook匹配提交`57e409eb27c68717078fb188ddc10b892d0a2cfc`，Worker匹配提交`98787e25e16e0a6417c937f5398b4b564fb1ee48`。这两个提交分别标识匹配源文件版本，不能当作整轮训练Commit。R1真实Adapter Hash为`d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371`。

新增10条来自C实际使用的`sft_real_control_110.jsonl`，逐条核对C的`sft_actual_inclusion_manifest.jsonl`、2轮使用记录、训练配置、输入Hash和最终Checkpoint Hash；不是B组10条合成Gravity，也不是候选Preview。SFT C继承R1后，新增训练仅使用这10条Gravity及100条Equation，未重放R1的64条Gravity。

逐条原Prompt、source_id/source_row、真实教师文本、独立数值参考、序列化响应及Token监督区间见`teacher_target_audit.jsonl`。实际训练归属有归档Manifest和训练总数的交叉证据；历史没有保存逐token梯度轨迹，不能声称重现了当时的每个梯度。

26道推理题中，1道Prompt与本次核查的74条Gravity教师完全相同，ID为`c04634c0`；其余题不能当作教师记忆复现。已见题c04634c0本次Greedy也给出错误数值。没有R1与C在这26题上的新增同题对照，因此不能判定是R1未学会还是后续增量训练遗忘。

## 2. 74条教师数学审计

复用Notebook01 `solve_gravity_prompt()`，同时用DPO自然语言Verifier独立解析全部公开观测，以`math.fsum`计算`g=2*sum(t^2*d)/sum(t^4)`并重算Query。Solver和独立实现分别计算，未改动数学函数或容差。

|检查|历史R1 64条|C新增真实Train 10条|
|---|---:|---:|
|所有公开观测显式重印在教师回答|0/64|0/10|
|明确使用全部公开示例的OLS|64/64|10/10|
|拟合g正确|64/64|10/10|
|未舍入Query距离正确|64/64|10/10|
|最终答案与独立OLS精确到两位小数相同|64/64|10/10|
|Solver与独立实现一致|64/64|10/10|
|记录元数据中的Full Trace通过原NL Verifier|64/64|10/10|
|实际序列化后的最终通道通过原NL Verifier|0/64|0/10|
|必要数学步骤的所有Token参与Loss|64/64|10/10|

全部公开时间距离完整保留在输入Prompt。教师使用“Parse all public time-distance examples”和全样本OLS公式，**没有把单点估计当作OLS**。但74条都不逐条列观测、不展开数值加权和；它们是正确但简略的教师步骤，并非细致的算术推导示范。不把未打印`g_i`视为缺陷，也未增加`g_i`强制字段。

## 3. 序列化、Tokenization、Loss Mask

Notebook03历史Cell17/当前Cell21的`encode_chat_example()`将`record["reasoning"]`写入assistant的`reasoning_content`，把`record["completion"]`切到`Final answer:`之后作为assistant.content。数据文件看起来是自然语言Full Trace，实际Chat Template目标却是下列结构：

```text
<think>
1. Parse all public time-distance examples.
2. Fit g = 2*sum(t^2*d)/sum(t^4) using only those examples.
3. The fitted g is 17.39206625255608; apply distance = 0.5*g*t^2 to Query t=2.41.
4. The unrounded Query distance is 50.50743000073549; format the final numeric answer.
</think>

Final answer: \boxed{50.51}<|im_end|>
```

这不是Loss Mask漏掉数学步骤。逐条实际Tokenization及offset区间检查显示：Prompt全为-100；OLS公式、拟合g和Query时间、未舍入距离、盒装答案的Token全部是有效causal targets，无Completion截断。`labels[:,1:]`与`logp[t]`预测`input_ids[t+1]`正确对齐。CPU测试将实际Notebook的Loss与Dense Cross Entropy及梯度比对：仅预测被屏蔽Prompt标签的logit位置直接梯度为0，所有Response预测位置有梯度。这不表示Prompt输入的embedding或共享模型参数不会通过Response Loss收到梯度。

对全部192条R1教师重建得到22,082个监督Token；对C全部110条、2轮重建得到77,116个，分别精确匹配真实训练日志。当前Notebook03和匹配历史Notebook均如此。没有发现数学步骤被mask、序列边界切错或误截断。

**训练合同缺陷在于通道选择。**DPO现有Verifier只认`</think>`后的完整证明，而实际SFT目标最终通道不含OLS/g/raw。把数据JSONL里的`completion`直接拿去验收会掩盖这个差异。没有把历史Thinking重新标成已验证证明，也没有降低验收标准。

补充核查：固定Tokenizer在`enable_thinking=True`时，输入末尾实际为`<|im_start|>assistant
`，并未预先打开`<think>`；该标签由模型生成。历史Verifier注释中关于输入已打开Thinking的概括不适用于这个固定模板。诊断仍原样使用现有Verifier，不修改历史逻辑。

## 4. 同条件Greedy与Sampling

26个原始Prompt、输入Token IDs、Adapter、BF16、Thinking、Chat Template、EOS和16,384输出Token上限均一致，逐题逐条校验。Greedy每题1条，`do_sample=False`；vLLM以`temperature=0`表示确定性贪心，Temperature采样参数不适用，绝不称作Temperature=1.0采样。Sampling复用Temperature1.0、top_p1、top_k-1的312条回答，每题12条，没有重生成。运行同一个固定Docker镜像与推理设置，无外网/API。

Accuracy仍使用现有官方最终答案提取和比较；完整证明仍用`gravity-natural-language-v1.2`。OLS使用正确率定义为最终通道具备可解析全观测OLS证据，且拟合g符合原rel_tol1e-4、abs_tol1e-5，没有已识别的矛盾数值过程。它是**认证下界**；只出现OLS关键词不计通过。

|指标|Greedy|Sampling|
|---|---:|---:|
|实际回答|26|312|
|官方答案正确|7/26 (26.92%)|32/312 (10.26%)|
|完整最终证明认证|0/26|0/312|
|最终通道OLS认证|0/26|0/312|
|格式通过|16/26|98/312|
|平均Token（含Thinking及EOS）|2060.65|920.19|
|中位Token|1276.0|471.5|
|官方答对回答平均Token|1478.2857142857142|2734.875|
|截断|1/26|0/312|

Greedy Accuracy Wilson95%：[13.70%, 46.08%]。Sampling的312条共享26题，不能当成312道独立题；按题平均12次结果，按26题做10,000次配对Bootstrap。Greedy减Sampling平均准确率差为16.67个百分点，95%区间[0.64, 34.29]个百分点。Sampling按题聚类区间见`diagnosis_statistics.json`。

Greedy点估计更高，按题配对Bootstrap区间略排除0，但配对sign-flip检验p=0.082；小样本推断不一致，视为方向性证据，不能宣称稳健显著优于Sampling。这些是已用于DPO采样的Train题诊断，包含真实Train与合成Train Prior，不是Validation、Holdout或总体能力保证。只运行一次Greedy，每题12次Sampling；统计结论只针对当前固定协议。

|来源|Greedy Accuracy|Sampling Accuracy|
|---|---:|---:|
|official_real_train|6/16|17/192|
|synthetic_train_prior|1/10|15/120|

### 逐题结果

|ID|来源|Greedy答对|Sampling答对/12|Greedy Token|Sampling平均Token|Greedy证明/OLS认证|
|---|---|---:|---:|---:|---:|---|
|c04634c0|official_real_train|0|2|1260|773.4|0/0|
|2ce67ec9|official_real_train|0|0|1530|703.2|0/0|
|8eecb224|official_real_train|0|1|492|703.2|0/0|
|6cfb1f29|official_real_train|0|0|1079|700.3|0/0|
|d54f5b26|official_real_train|1|1|1315|917.4|0/0|
|eba1891f|official_real_train|0|1|1292|720.6|0/0|
|bdcccbab|official_real_train|1|0|1198|875.8|0/0|
|ae5e65cf|official_real_train|1|5|956|612.8|0/0|
|0ab3d1de|official_real_train|0|1|7000|899.8|0/0|
|736cdc47|official_real_train|0|1|1323|582.4|0/0|
|42fee270|official_real_train|0|0|1334|687.3|0/0|
|f19b29c6|official_real_train|1|1|1443|631.7|0/0|
|76d51969|official_real_train|0|1|1030|588.2|0/0|
|5fe169fc|official_real_train|1|0|1630|448.1|0/0|
|16b45644|official_real_train|1|1|955|974.4|0/0|
|ff9540e2|official_real_train|0|2|1141|1066.8|0/0|
|gravity-final-001|synthetic_train_prior|0|1|182|1211.5|0/0|
|gravity-final-002|synthetic_train_prior|0|0|16384|692.4|0/0|
|gravity-final-003|synthetic_train_prior|0|1|3630|358.6|0/0|
|gravity-final-004|synthetic_train_prior|0|0|710|760.2|0/0|
|gravity-final-005|synthetic_train_prior|0|1|437|991.1|0/0|
|gravity-final-006|synthetic_train_prior|0|1|316|937.3|0/0|
|gravity-final-007|synthetic_train_prior|1|2|2851|2398.6|0/0|
|gravity-final-008|synthetic_train_prior|0|3|1334|1533.5|0/0|
|gravity-final-009|synthetic_train_prior|0|3|2198|1585.5|0/0|
|gravity-final-010|synthetic_train_prior|0|3|557|1570.6|0/0|

逐题JSONL包括期望答案、全部Sampling预测、Greedy预测、失败原因、来源与Token差；原始Greedy文本和Token IDs在`greedy/SFT_C_raw.jsonl`。Sampling原始回答保持在原DPO Run，登记原始Hash，不复制改写原始内容。

## 5. 失败分类及原因边界

下列分类可重叠，不相加为互斥失败总数。不可解析与已证明错误明确区分。

```json
{
  "official_final_incorrect": 19,
  "final_ols_evidence_missing_or_unparseable": 26,
  "final_g_missing_or_inaccurate": 25,
  "final_raw_distance_missing_or_inaccurate": 25,
  "recognized_incorrect_math_claim": 23,
  "unsupported_syntax_not_proven_false": 15,
  "official_correct_but_strict_ols_answer_failed": 6
}
```

完整逐条原因见`greedy_scored.jsonl`、`sampling_diagnostic.jsonl`和`failure_classification.json`。对Sampling重审的官方正确性及完整proof_audit与历史v1.2逐条完全一致，未修改历史验收。

官方最终指标使用原评分容差；数学证明继续使用更严格的原精度。官方答对并不保证精确复现OLS及两位小数答案，更不保证过程正确。`official_correct_but_strict_ols_answer_failed`专门记录这个差异，没有改变官方评分。

Thinking中出现OLS、sum或单点推断用语只作未验证审计信号，见`unverified_thinking_signals`；不将关键词命中率当作OLS正确率，不把隐藏Thinking提交给DPO，也不将其当成已验证完整证明。有限语法Verifier可能漏掉等价LaTeX/自由表述，因此0认证并不能证明所有运算都错。官方最终答案低正确率则是独立于Proof语法的实证问题。

### 人工复核并由独立数值参考支持的失败案例

- `c04634c0`：历史R1已见Prompt；使用前两点并错误地声称g=12.34，独立全样本OLS为10.139445051441283；还将2.06^2写为4.2416（正确为4.2436）。
- `gravity-final-001`：直接使用地球g=9.81，而公开数据的OLS为13.09133128690559，最终12.53也不符16.97。
- `gravity-final-002`：16,384 Token达到上限，未关闭Thinking，没有可验收最终回答；不引用隐藏Thinking为证明。
- `gravity-final-007`：用第一点粗略估g=7.29；最终20.47按官方容差答对，但全样本OLS为7.28142101139225、规范答案20.45，不构成完整OLS证明。
- `gravity-final-008`：漏掉公开的t=4.78,d=188.90，并把其余距离的算术平均115.09当作Query距离；标准OLS答案113.20。

这些具体错误不依赖于全文模板匹配或对LaTeX的解析成败，说明阻塞不只是Verifier覆盖不足。原始最终通道和参考值单独保存为`reviewed_failure_cases.jsonl`；未将隐藏Thinking标为已验证。

## 6. 对五个问题的明确回答

1. **教师目标是否有缺陷？**没有发现74条目标的数值OLS、g、raw或最终答案错误；有确定的输出合同缺陷：正确步骤被序列化进Thinking，最终通道只教授答案。另有教学粒度不足：不展示全部观测的数值累加，简短地直接给出拟合值。
2. **Loss Mask是否覆盖必要步骤？**是，74/74覆盖；全任务监督Token总数与真实R1/C训练记录精确一致，causal mask及梯度CPU检验通过。不能把最终通道没有证明误诊为数学步骤没有参与训练。
3. **Greedy是否明显优于Temperature1.0 Sampling？**Greedy点估计更高，按题配对Bootstrap区间略排除0，但配对sign-flip检验p=0.082；小样本推断不一致，视为方向性证据，不能宣称稳健显著优于Sampling。
4. **当前瓶颈是什么？**首先是教师序列化/最终证明合同错位，而非已确认的Loss Mask或OLS公式错误。最终答案的真实错误另提示模型在这套训练与提示下未稳定掌握数值求解；单一小模型、有限教师和本面板不能区分容量、训练量、教师粒度及其他泛化因素，不能武断归为纯容量不足。采样差异须以第4节真实结果为准，不能单凭Sampling失败就断言原因。
5. **修复SFT还是继续DPO？**先修复SFT教师与最终通道合同，再做隔离SFT回归和Greedy/同温Sampling验证；当前不继续DPO。保持Thinking为未验证内容，把可独立复核的自然语言OLS、拟合g、Query代入和raw步骤明确放在最终回答；提升教师的数值演算示范，保持原数学精度且不强制逐条g_i。只有同题正确完整长短答再次达到原16对门槛才有理由重启DPO-B。此处只是建议，本次没有实施任何SFT修复或训练。

## 7. 执行与隔离证据

本轮Greedy生成耗时1494.61s（不含初始化），本进程Torch峰值reserved=5771362304 bytes，低于8,000,000,000 bytes守卫。GB10为共享UMA，未把该数值当作全设备显存。没有停止其他GPU进程。

完整性检查覆盖212个既有输入/历史/源码文件，全部Hash不变；SFT C及R1 Adapter均保留。未改Notebook01/03/04/05/06及训练Worker；DPO/PPO/GRPO训练步骤0，没有保存新模型。

CPU实际执行记录：

```text
----------------------------------------------------------------------
Ran 55 tests in 5.869s

OK
<sys>:0: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

采用diagnose流程，先冻结5个可证伪假设，再逐条教师数值/序列化/掩码复核及解码对照；按本次边界仅诊断，未执行修复训练。

GPU前54项CPU测试通过。Greedy完成后，离线归档一致性比较因Python tuple与JSON list类型不同而停止；原始回答和失败日志均保留。修复仅使比较落在相同JSON数据域，并新增反例回归，随后55项CPU测试通过，重新完成离线统计。没有改Verifier、数值容差或模型输出，也没有重跑任何GPU生成；源码修正见`offline_analysis_amendment.json`。

复现入口（必须更换独立Run ID，现Run拒绝覆盖）：`python utils/run_dpo_sft_c_gravity_diagnosis.py all`。唯一新增推理为26条Greedy，自动阻塞等待进程完成后再分析，无阶段性确认与训练进度轮询。

产物索引：`teacher_target_audit.jsonl`、`teacher_audit_summary.json`、`greedy/SFT_C_raw.jsonl`、`greedy_scored.jsonl`、`sampling_diagnostic.jsonl`、`per_prompt_comparison.jsonl`、`comparison_summary.json`、`failure_classification.json`、`diagnosis_statistics.json`、`registration.json`、`integrity_check.json`、`cpu_tests.log`、`greedy.log`。

诊断完成不代表解决DPO阻塞；本次没有DPO训练、收益评测或新Checkpoint。
