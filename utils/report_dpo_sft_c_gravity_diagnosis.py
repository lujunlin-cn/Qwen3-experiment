"""Offline publication of the isolated SFT C teacher/decoding diagnosis."""
from __future__ import annotations

from collections import Counter
import json
import random
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.run_dpo_sft_c_gravity_diagnosis import RUN, read, rows, write, write_rows, sha, ADAPTER_HASH


def cluster_interval(values, seed=20261116, draws=10000):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [means[int(.025 * draws)], means[int(.975 * draws)]]


def main():
    q = read(RUN / 'comparison_summary.json')
    integrity = read(RUN / 'integrity_check.json')
    if not integrity['all_unchanged']:
        raise ValueError('Protected artifacts changed')
    g, s = q['greedy'], q['sampling']
    teachers = rows(RUN / 'teacher_target_audit.jsonl')
    per = rows(RUN / 'per_prompt_comparison.jsonl')
    greeds = rows(RUN / 'greedy_scored.jsonl')
    overlap = [r['sample_id'] for r in greeds if r['prompt'] in {x['prompt'] for x in teachers}]
    case_notes = {
        'c04634c0': '历史R1已见Prompt；使用前两点并错误地声称g=12.34，独立全样本OLS为10.139445051441283；还将2.06^2写为4.2416（正确为4.2436）。',
        'gravity-final-001': '直接使用地球g=9.81，而公开数据的OLS为13.09133128690559，最终12.53也不符16.97。',
        'gravity-final-007': '用第一点粗略估g=7.29；最终20.47按官方容差答对，但全样本OLS为7.28142101139225、规范答案20.45，不构成完整OLS证明。',
        'gravity-final-008': '漏掉公开的t=4.78,d=188.90，并把其余距离的算术平均115.09当作Query距离；标准OLS答案113.20。',
        'gravity-final-002': '16,384 Token达到上限，未关闭Thinking，没有可验收最终回答；不引用隐藏Thinking为证明。',
    }
    cases = []
    for r in greeds:
        if r['sample_id'] in case_notes:
            final = r['completion'].split('</think>', 1)[1] if '</think>' in r['completion'] else None
            cases.append({'sample_id': r['sample_id'], 'review': case_notes[r['sample_id']],
                'official_correct': r['correct'], 'predicted': r['predicted'], 'expected_answer': r['expected_answer'],
                'independent_public_reference': r['proof_audit']['reference'], 'finish_reason': r['finish_reason'],
                'final_section_only': final, 'completion_tokens': r['completion_tokens'],
                'thinking_used_as_verified_proof': False})
    write_rows(RUN / 'reviewed_failure_cases.jsonl', cases)
    reg = read(RUN / 'registration.json')
    gpu = read(RUN / 'greedy/result.json')
    cpu = (RUN / 'cpu_final_tests.log').read_text()
    if not cpu.rstrip().endswith('OK') and '\nOK\n' not in cpu:
        raise ValueError('CPU tests did not pass')
    counts = Counter()
    for r in greeds:
        a = r['proof_audit']
        if not a['final_correct']:
            counts['official_final_incorrect'] += 1
        if not a.get('checks', {}).get('all_public_ols'):
            counts['final_ols_evidence_missing_or_unparseable'] += 1
        if 'fitted_g_missing_or_inaccurate' in a['reasons']:
            counts['final_g_missing_or_inaccurate'] += 1
        if 'unrounded_query_distance_missing_or_inaccurate' in a['reasons']:
            counts['final_raw_distance_missing_or_inaccurate'] += 1
        if any(v.startswith('incorrect_') for v in a['reasons']):
            counts['recognized_incorrect_math_claim'] += 1
        if any(v.startswith('unsupported_') for v in a['reasons']):
            counts['unsupported_syntax_not_proven_false'] += 1
        if a['final_correct'] and 'final_answer_not_rounded_public_ols' in a['reasons']:
            counts['official_correct_but_strict_ols_answer_failed'] += 1
    uncertainty = {'sampling_accuracy_prompt_cluster_bootstrap95': cluster_interval([x['sampling']['accuracy'] for x in per]),
                   'greedy_accuracy_prompt_bootstrap95': cluster_interval([x['greedy']['accuracy'] for x in per]),
                   'paired_delta': q['paired_prompt_cluster_accuracy_delta'],
                   'draws': 10000, 'seed': 20261116,
                   'interpretation': '26 fixed Train prompts; exploratory conditional decoding comparison, not generalization or multiple-training-seed inference'}
    write(RUN / 'diagnosis_statistics.json', uncertainty)
    write(RUN / 'failure_classification.json', {'greedy_nonexclusive': dict(counts),
          'sampling_nonexclusive': s['failure_reasons_nonexclusive'],
          'not_all_unverifiable_proofs_are_wrong': True,
          'teacher_contract_failure': {'74_targets_put_ols_inside_thinking': True, '74_final_channels_boxed_only': True}})
    delta = q['paired_prompt_cluster_accuracy_delta']
    ci = delta['bootstrap95']
    if g['accuracy'] > s['accuracy']:
        comparison = ('Greedy点估计更高，按题配对Bootstrap区间略排除0，但配对sign-flip检验p='
                      f'{delta["sign_flip_pvalue"]:.3f}；小样本推断不一致，视为方向性证据，不能宣称稳健显著优于Sampling。'
                      if ci[0] > 0 and delta['sign_flip_pvalue'] >= .05 else
                      'Greedy点估计更高，且按题配对Bootstrap区间排除0；在这26道固定题上有解码差异证据。'
                      if ci[0] > 0 else 'Greedy点估计更高，但按题配对Bootstrap区间包含0，不能认定明显优于Sampling。')
    else:
        comparison = ('Greedy点估计没有优于Sampling；本面板不支持把低正确率主要归因于Temperature=1.0。')
    example = next(x for x in teachers if x['group'] == 'added_real_c')
    lines = [
        '# SFT C Gravity：教师目标、Loss Mask与解码专项诊断', '',
        f'诊断Run：`{RUN.name}`。输入基线Git：`{reg["git_commit"]}`。', '',
        '**已确认两个阻塞：训练输出通道与DPO验收合同不一致，以及模型未稳定执行正确的全样本OLS。**74条Gravity教师的数值OLS、g、未舍入Query距离和答案均正确，必要步骤也全部参与Loss；但实际序列化把数学步骤放在`<think>…</think>`内，最终回答只监督`Final answer: \\boxed{...}`。因此即使完美复现训练目标，现有“只验收Thinking后的完整证明”合同也不会接受。', '',
        f'本次只新增26条真实GPU Greedy回答，复用原312条Sampling。{comparison} 完整证明和OLS认证属于保守有限语法的认证数，不等同于所有未认证过程均错误；不能由0认证直接推出模型从未学会OLS。', '',
        '## 1. 实际权重、数据及历史来源', '',
        f'Policy仍为SFT C，Adapter SHA256：`{ADAPTER_HASH}`。Base为Qwen/Qwen3-0.6B，Revision `c1899de289a04d12100db370d81485cdf75e47ca`；7项Base/Tokenizer文件逐一匹配原始指纹。BF16 LoRA，r8、alpha16、dropout0。没有加载其他RL权重，没有训练或保存权重。', '',
        '历史64条使用真实R1的`recovered_course_r2/panels/sft.jsonl`，核对其原协议Hash，并从Git恢复与真实协议SHA256匹配的Notebook03及Worker，而非用当前格式猜测历史目标。Notebook匹配提交`57e409eb27c68717078fb188ddc10b892d0a2cfc`，Worker匹配提交`98787e25e16e0a6417c937f5398b4b564fb1ee48`。这两个提交分别标识匹配源文件版本，不能当作整轮训练Commit。R1真实Adapter Hash为`d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371`。', '',
        '新增10条来自C实际使用的`sft_real_control_110.jsonl`，逐条核对C的`sft_actual_inclusion_manifest.jsonl`、2轮使用记录、训练配置、输入Hash和最终Checkpoint Hash；不是B组10条合成Gravity，也不是候选Preview。SFT C继承R1后，新增训练仅使用这10条Gravity及100条Equation，未重放R1的64条Gravity。', '',
        '逐条原Prompt、source_id/source_row、真实教师文本、独立数值参考、序列化响应及Token监督区间见`teacher_target_audit.jsonl`。实际训练归属有归档Manifest和训练总数的交叉证据；历史没有保存逐token梯度轨迹，不能声称重现了当时的每个梯度。', '',
        f'26道推理题中，{len(overlap)}道Prompt与本次核查的74条Gravity教师完全相同，ID为`{", ".join(overlap)}`；其余题不能当作教师记忆复现。已见题c04634c0本次Greedy也给出错误数值。没有R1与C在这26题上的新增同题对照，因此不能判定是R1未学会还是后续增量训练遗忘。', '',
        '## 2. 74条教师数学审计', '',
        '复用Notebook01 `solve_gravity_prompt()`，同时用DPO自然语言Verifier独立解析全部公开观测，以`math.fsum`计算`g=2*sum(t^2*d)/sum(t^4)`并重算Query。Solver和独立实现分别计算，未改动数学函数或容差。', '',
        '|检查|历史R1 64条|C新增真实Train 10条|',
        '|---|---:|---:|',
    ]
    group = q['teacher_audit']['groups']
    for title, key in [('所有公开观测显式重印在教师回答', 'all_observations_explicitly_printed_in_target'),
                       ('明确使用全部公开示例的OLS', 'all_observations_referenced_by_ols'),
                       ('拟合g正确', 'fitted_g_correct'),
                       ('未舍入Query距离正确', 'unrounded_query_distance_correct'),
                       ('最终答案与独立OLS精确到两位小数相同', 'final_answer_exact_public_ols'),
                       ('Solver与独立实现一致', 'solver_independent_agreement'),
                       ('记录元数据中的Full Trace通过原NL Verifier', 'metadata_proof_passed'),
                       ('实际序列化后的最终通道通过原NL Verifier', 'actual_final_proof_passed'),
                       ('必要数学步骤的所有Token参与Loss', 'all_required_steps_supervised')]:
        lines.append(f'|{title}|{group["historical_r1"][key]}/64|{group["added_real_c"][key]}/10|')
    lines += [
        '', '全部公开时间距离完整保留在输入Prompt。教师使用“Parse all public time-distance examples”和全样本OLS公式，**没有把单点估计当作OLS**。但74条都不逐条列观测、不展开数值加权和；它们是正确但简略的教师步骤，并非细致的算术推导示范。不把未打印`g_i`视为缺陷，也未增加`g_i`强制字段。', '',
        '## 3. 序列化、Tokenization、Loss Mask', '',
        'Notebook03历史Cell17/当前Cell21的`encode_chat_example()`将`record["reasoning"]`写入assistant的`reasoning_content`，把`record["completion"]`切到`Final answer:`之后作为assistant.content。数据文件看起来是自然语言Full Trace，实际Chat Template目标却是下列结构：', '',
        '```text', example['serialization']['serialized_response'].rstrip(), '```', '',
        '这不是Loss Mask漏掉数学步骤。逐条实际Tokenization及offset区间检查显示：Prompt全为-100；OLS公式、拟合g和Query时间、未舍入距离、盒装答案的Token全部是有效causal targets，无Completion截断。`labels[:,1:]`与`logp[t]`预测`input_ids[t+1]`正确对齐。CPU测试将实际Notebook的Loss与Dense Cross Entropy及梯度比对：仅预测被屏蔽Prompt标签的logit位置直接梯度为0，所有Response预测位置有梯度。这不表示Prompt输入的embedding或共享模型参数不会通过Response Loss收到梯度。', '',
        '对全部192条R1教师重建得到22,082个监督Token；对C全部110条、2轮重建得到77,116个，分别精确匹配真实训练日志。当前Notebook03和匹配历史Notebook均如此。没有发现数学步骤被mask、序列边界切错或误截断。', '',
        '**训练合同缺陷在于通道选择。**DPO现有Verifier只认`</think>`后的完整证明，而实际SFT目标最终通道不含OLS/g/raw。把数据JSONL里的`completion`直接拿去验收会掩盖这个差异。没有把历史Thinking重新标成已验证证明，也没有降低验收标准。', '',
        '补充核查：固定Tokenizer在`enable_thinking=True`时，输入末尾实际为`<|im_start|>assistant\n`，并未预先打开`<think>`；该标签由模型生成。历史Verifier注释中关于输入已打开Thinking的概括不适用于这个固定模板。诊断仍原样使用现有Verifier，不修改历史逻辑。', '',
        '## 4. 同条件Greedy与Sampling', '',
        '26个原始Prompt、输入Token IDs、Adapter、BF16、Thinking、Chat Template、EOS和16,384输出Token上限均一致，逐题逐条校验。Greedy每题1条，`do_sample=False`；vLLM以`temperature=0`表示确定性贪心，Temperature采样参数不适用，绝不称作Temperature=1.0采样。Sampling复用Temperature1.0、top_p1、top_k-1的312条回答，每题12条，没有重生成。运行同一个固定Docker镜像与推理设置，无外网/API。', '',
        'Accuracy仍使用现有官方最终答案提取和比较；完整证明仍用`gravity-natural-language-v1.2`。OLS使用正确率定义为最终通道具备可解析全观测OLS证据，且拟合g符合原rel_tol1e-4、abs_tol1e-5，没有已识别的矛盾数值过程。它是**认证下界**；只出现OLS关键词不计通过。', '',
        '|指标|Greedy|Sampling|', '|---|---:|---:|',
        f'|实际回答|{g["n"]}|{s["n"]}|',
        f'|官方答案正确|{g["correct"]}/{g["n"]} ({100*g["accuracy"]:.2f}%)|{s["correct"]}/{s["n"]} ({100*s["accuracy"]:.2f}%)|',
        f'|完整最终证明认证|{g["proof_pass"]}/{g["n"]}|{s["proof_pass"]}/{s["n"]}|',
        f'|最终通道OLS认证|{g["ols_final_certified"]}/{g["n"]}|{s["ols_final_certified"]}/{s["n"]}|',
        f'|格式通过|{g["format_valid"]}/{g["n"]}|{s["format_valid"]}/{s["n"]}|',
        f'|平均Token（含Thinking及EOS）|{g["mean_tokens"]:.2f}|{s["mean_tokens"]:.2f}|',
        f'|中位Token|{g["median_tokens"]:.1f}|{s["median_tokens"]:.1f}|',
        f'|官方答对回答平均Token|{g["mean_correct_tokens"]}|{s["mean_correct_tokens"]}|',
        f'|截断|{g["truncated"]}/{g["n"]}|{s["truncated"]}/{s["n"]}|',
        '', f'Greedy Accuracy Wilson95%：[{100*g["accuracy_wilson95_descriptive"][0]:.2f}%, {100*g["accuracy_wilson95_descriptive"][1]:.2f}%]。Sampling的312条共享26题，不能当成312道独立题；按题平均12次结果，按26题做10,000次配对Bootstrap。Greedy减Sampling平均准确率差为{100*delta["mean_after_minus_before"]:.2f}个百分点，95%区间[{100*ci[0]:.2f}, {100*ci[1]:.2f}]个百分点。Sampling按题聚类区间见`diagnosis_statistics.json`。', '',
        comparison + '这些是已用于DPO采样的Train题诊断，包含真实Train与合成Train Prior，不是Validation、Holdout或总体能力保证。只运行一次Greedy，每题12次Sampling；统计结论只针对当前固定协议。', '',
        '|来源|Greedy Accuracy|Sampling Accuracy|', '|---|---:|---:|',
        *[f'|{source}|{v["greedy"]["correct"]}/{v["greedy"]["n"]}|{v["sampling"]["correct"]}/{v["sampling"]["n"]}|' for source, v in q['by_source'].items()], '',
        '### 逐题结果', '',
        '|ID|来源|Greedy答对|Sampling答对/12|Greedy Token|Sampling平均Token|Greedy证明/OLS认证|',
        '|---|---|---:|---:|---:|---:|---|',
    ]
    for r in per:
        lines.append(f'|{r["sample_id"]}|{r["source"]}|{r["greedy"]["correct"]}|{r["sampling"]["correct"]}|{r["greedy"]["total_tokens"]}|{r["sampling"]["mean_tokens"]:.1f}|{r["greedy"]["proof_pass"]}/{r["greedy"]["ols_final_certified"]}|')
    lines += ['', '逐题JSONL包括期望答案、全部Sampling预测、Greedy预测、失败原因、来源与Token差；原始Greedy文本和Token IDs在`greedy/SFT_C_raw.jsonl`。Sampling原始回答保持在原DPO Run，登记原始Hash，不复制改写原始内容。', '',
        '## 5. 失败分类及原因边界', '',
        '下列分类可重叠，不相加为互斥失败总数。不可解析与已证明错误明确区分。', '',
        '```json', json.dumps(dict(counts), ensure_ascii=False, indent=2), '```', '',
        '完整逐条原因见`greedy_scored.jsonl`、`sampling_diagnostic.jsonl`和`failure_classification.json`。对Sampling重审的官方正确性及完整proof_audit与历史v1.2逐条完全一致，未修改历史验收。', '',
        '官方最终指标使用原评分容差；数学证明继续使用更严格的原精度。官方答对并不保证精确复现OLS及两位小数答案，更不保证过程正确。`official_correct_but_strict_ols_answer_failed`专门记录这个差异，没有改变官方评分。', '',
        'Thinking中出现OLS、sum或单点推断用语只作未验证审计信号，见`unverified_thinking_signals`；不将关键词命中率当作OLS正确率，不把隐藏Thinking提交给DPO，也不将其当成已验证完整证明。有限语法Verifier可能漏掉等价LaTeX/自由表述，因此0认证并不能证明所有运算都错。官方最终答案低正确率则是独立于Proof语法的实证问题。', '',
        '### 人工复核并由独立数值参考支持的失败案例', '',
        *[f'- `{r["sample_id"]}`：{r["review"]}' for r in cases], '',
        '这些具体错误不依赖于全文模板匹配或对LaTeX的解析成败，说明阻塞不只是Verifier覆盖不足。原始最终通道和参考值单独保存为`reviewed_failure_cases.jsonl`；未将隐藏Thinking标为已验证。', '',
        '## 6. 对五个问题的明确回答', '',
        '1. **教师目标是否有缺陷？**没有发现74条目标的数值OLS、g、raw或最终答案错误；有确定的输出合同缺陷：正确步骤被序列化进Thinking，最终通道只教授答案。另有教学粒度不足：不展示全部观测的数值累加，简短地直接给出拟合值。',
        '2. **Loss Mask是否覆盖必要步骤？**是，74/74覆盖；全任务监督Token总数与真实R1/C训练记录精确一致，causal mask及梯度CPU检验通过。不能把最终通道没有证明误诊为数学步骤没有参与训练。',
        '3. **Greedy是否明显优于Temperature1.0 Sampling？**' + comparison,
        '4. **当前瓶颈是什么？**首先是教师序列化/最终证明合同错位，而非已确认的Loss Mask或OLS公式错误。最终答案的真实错误另提示模型在这套训练与提示下未稳定掌握数值求解；单一小模型、有限教师和本面板不能区分容量、训练量、教师粒度及其他泛化因素，不能武断归为纯容量不足。采样差异须以第4节真实结果为准，不能单凭Sampling失败就断言原因。',
        '5. **修复SFT还是继续DPO？**先修复SFT教师与最终通道合同，再做隔离SFT回归和Greedy/同温Sampling验证；当前不继续DPO。保持Thinking为未验证内容，把可独立复核的自然语言OLS、拟合g、Query代入和raw步骤明确放在最终回答；提升教师的数值演算示范，保持原数学精度且不强制逐条g_i。只有同题正确完整长短答再次达到原16对门槛才有理由重启DPO-B。此处只是建议，本次没有实施任何SFT修复或训练。', '',
        '## 7. 执行与隔离证据', '',
        f'本轮Greedy生成耗时{gpu["generation_seconds"]:.2f}s（不含初始化），本进程Torch峰值reserved={max(x["peak_reserved_bytes"] for x in gpu["memory"])} bytes，低于8,000,000,000 bytes守卫。GB10为共享UMA，未把该数值当作全设备显存。没有停止其他GPU进程。', '',
        f'完整性检查覆盖{integrity["protected_files"]}个既有输入/历史/源码文件，全部Hash不变；SFT C及R1 Adapter均保留。未改Notebook01/03/04/05/06及训练Worker；DPO/PPO/GRPO训练步骤0，没有保存新模型。', '',
        'CPU实际执行记录：', '', '```text', cpu[cpu.rfind("\n----------------------------------------------------------------------"):].strip(), '```', '',
        '采用diagnose流程，先冻结5个可证伪假设，再逐条教师数值/序列化/掩码复核及解码对照；按本次边界仅诊断，未执行修复训练。', '',
        'GPU前54项CPU测试通过。Greedy完成后，离线归档一致性比较因Python tuple与JSON list类型不同而停止；原始回答和失败日志均保留。修复仅使比较落在相同JSON数据域，并新增反例回归，随后55项CPU测试通过，重新完成离线统计。没有改Verifier、数值容差或模型输出，也没有重跑任何GPU生成；源码修正见`offline_analysis_amendment.json`。', '',
        '复现入口（必须更换独立Run ID，现Run拒绝覆盖）：`python utils/run_dpo_sft_c_gravity_diagnosis.py all`。唯一新增推理为26条Greedy，自动阻塞等待进程完成后再分析，无阶段性确认与训练进度轮询。', '',
        '产物索引：`teacher_target_audit.jsonl`、`teacher_audit_summary.json`、`greedy/SFT_C_raw.jsonl`、`greedy_scored.jsonl`、`sampling_diagnostic.jsonl`、`per_prompt_comparison.jsonl`、`comparison_summary.json`、`failure_classification.json`、`diagnosis_statistics.json`、`registration.json`、`integrity_check.json`、`cpu_tests.log`、`greedy.log`。', '',
        '诊断完成不代表解决DPO阻塞；本次没有DPO训练、收益评测或新Checkpoint。',
    ]
    report = ROOT / 'docs/DPO_SFT_C_GRAVITY_DIAGNOSIS.md'
    with report.open('x') as f:
        f.write('\n'.join(lines) + '\n')
    write(RUN / 'artifact_manifest.json', {'status': 'diagnosis_completed_no_training',
        'report': str(report.relative_to(ROOT)), 'report_sha256': sha(report),
        'files': {str(p.relative_to(RUN)): sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()},
        'reporter_sha256': sha(Path(__file__)), 'training_steps': 0, 'new_greedy_responses': 26,
        'sampling_reused': 312, 'history_and_weights_unchanged': True})
    print(json.dumps({'report': str(report), 'greedy': g, 'sampling': s,
                     'paired_accuracy_delta': delta}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
