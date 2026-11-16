"""Offline report for Gravity C2 and the strictly gated DPO-B continuation."""
from __future__ import annotations
import json,re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from utils.run_gravity_sft_c2 import RUN,CONFIG,C2,DPO,read,rows,sha,write


def optional(name):return read(RUN/name) if (RUN/name).exists() else None
def pct(value):return f'{100*value:.2f}%'
def number(value):return '—' if value is None else f'{value:.2f}'


def metric_table(before,after,names):
    lines=['|指标|'+names[0]+'|'+names[1]+'|','|---|---:|---:|']
    for label,key in [('Accuracy','accuracy'),('完整最终证明通过率','proof_rate'),('OLS认证率','ols_rate'),('Truncation','truncation_rate')]:
        lines.append(f'|{label}|{pct(before[key])}|{pct(after[key])}|')
    for label,key in [('平均Completion Token','mean_tokens'),('中位Completion Token','median_tokens'),
                      ('官方答对平均Token','official_correct_mean_tokens'),('正确且证明认证平均Token','correct_verified_mean_tokens')]:
        lines.append(f'|{label}|{number(before[key])}|{number(after[key])}|')
    lines.append(f'|答对数量|{before["correct"]}/{before["n"]}|{after["correct"]}/{after["n"]}|')
    lines.append(f'|证明认证数量|{before["proof_pass"]}/{before["n"]}|{after["proof_pass"]}/{after["n"]}|')
    lines.append(f'|格式有效|{before["format_valid"]}/{before["n"]}|{after["format_valid"]}/{after["n"]}|')
    return lines


def report():
    status=read(RUN/'final_status.json');config=read(CONFIG);reg=optional('registration.json');teacher=optional('teacher_repair_summary.json')
    trained=optional('sft_train/result.json');reload=optional('sft_reload.json');ready=optional('readiness.json')
    dpo=optional('dpo_training_result.json');dpo_reload=optional('dpo_reload.json')
    lines=['# Gravity SFT C2 定向修复与条件 DPO 最终报告','',f'Run：`{RUN.name}`。最终状态：`{status["status"]}`。','']
    if trained:
        diagnosis=optional('C_vs_C2_diagnosis.json');tune=optional('C_vs_C2_tune.json')
        lines += [f'**已完成真实C2增量SFT：{trained["optimizer_steps"]}次Optimizer更新，74条Gravity Train教师、2轮；原SFT C保持不变。**']
        if diagnosis and tune:
            lines += [f'26题诊断：C2最终完整证明认证{diagnosis["after"]["proof_pass"]}/26，官方答案{diagnosis["after"]["correct"]}/26；'
                      f'48题Tune：证明认证{tune["after"]["proof_pass"]}/48，官方答案{tune["after"]["correct"]}/48。']
        if ready:
            lines += [f'C2共完成{ready["rollouts"]}条真实Rollout，得到{ready["pairs"]}/26个严格DPO-B偏好对；门槛为16对。']
    else:lines += ['未完成C2训练；不宣称模型学会或产生新Checkpoint。']
    if dpo:lines += [f'达到门槛后完成真实DPO：{dpo["optimizer_steps"]}次更新，冻结Reference为C2。']
    else:lines += ['**本轮未执行DPO训练、DPO Checkpoint重载或Locked Final比较；不得将C2训练结果当作DPO收益。**']
    lines += ['','## 1. 来源、权重与数据隔离','',f'原SFT C Adapter SHA256：`{status["original_C_sha256"]}`。Base为Qwen/Qwen3-0.6B，Revision `{config["base_revision"]}`。原C、R1、历史DPO/诊断/增强实验产物逐文件Hash冻结，最终保持不变：`{status["all_protected_unchanged"]}`，覆盖{status["protected_files"]}个文件。','',
        '64条来自历史R1真实训练panel，10条来自C真实使用的Real Control输入及actual inclusion manifest。全部source_id/source_row/Prompt逐条核对实际SFT Train；重新做实际Validation/Holdout Prompt去重，保留原Prompt与答案。不是复用B组的10条合成Gravity，也没有生成新的数学题。', '',
        'C2教师来源统一标记`solver_generated_rendered`，同时保留`historical_r1`/`added_real_c`原始分组及真实Train行号。Python Solver计算、独立Verifier复核、Python Renderer生成自然语言；不伪称LLM自主CoT。','']
    if trained:lines += [f'C2 Adapter：`{C2}`；SHA256：`{trained["adapter_sha256"]}`。Optimizer SHA256：`{trained["optimizer_sha256"]}`。未继承其他算法的Optimizer State。','']
    lines += ['## 2. 教师与最终通道修复','',
        'Notebook01新增`gravity_c2_solver_reference()`，复用既有`solve_gravity_prompt()`、公开Prompt解析器和独立`_gravity_verifier_fit()`，仅补充展示所需的公开观测与OLS数值项。原数学Solver、公式、精度、NL Verifier均未改。', '',
        'Notebook03新增`render_gravity_c2_final_proof()`、`make_gravity_c2_record()`、`encode_gravity_c2_final_example()`和显式训练封装。展示每个公开(t,d)、每项t²d/t⁴、两项数值求和、OLS数值代入、拟合g、Query时间、未舍入距离和两位小数Boxed Answer。无需逐个输出g_i。','',
        '完整证明写入assistant.content，assistant消息不提供reasoning_content。固定Qwen3模板保留空`<think>…</think>`，最终通道包含全部证明。Prompt标签为-100，其后全部Token（包括数学和EOS）参与原causal SFT loss。保留默认encode_chat_example和默认SFT数据入口；train_augmentation_branch只增加显式可选encoder参数，默认行为不变。','']
    if teacher:
        lines += [f'教师数学/实际序列化后的原NL Verifier：{teacher["final_proof_passed"]}/74通过；全量数学Token监督{teacher["all_math_tokens_supervised"]}/74。最长完整Chat为{teacher["max_full_chat_tokens"]} Token，上限4096，未截断。每轮实际监督{teacher["supervised_tokens_per_epoch"]} Token。','',
            '为避免原有限语法解析器混淆，“全部观测”用Observation编号展示，“数值求和”用Weighted data sum/Fourth-power sum标签；所有数值保留。原Verifier会把“For t=”误当Query时间，部分sum(...)文本也可能被符号正则误读。本轮只采用清晰的教师表达，不修改Verifier、数学容差或历史结果。', '',
            '教师样例（来自真实Train，最终回答部分）：','```text',rows(RUN/'gravity_c2_teachers_74.jsonl')[0]['completion'],'```','']
    lines += ['## 3. 真实C2训练、梯度及重载','',
        '从SFT C初始化，BF16 LoRA r8/alpha16/dropout0；AdamW LR2e-5、weight_decay0.01、eps1e-8、microbatch1、累积8、clip1、2epochs、seed20261116、max_length4096、分块投影64。沿用现有C训练配置，只改变教师目标与其序列化、训练数据集规模。','']
    if trained:
        lines += [f'真实更新{trained["optimizer_steps"]}次，监督Token总数{trained["supervised_tokens"]}，改变{trained["changed_tensor_count"]}个LoRA张量；训练耗时{trained["training_seconds"]:.2f}s，含初始化/保存墙钟{trained["wall_seconds"]:.2f}s。Torch本进程峰值reserved={trained["memory"]["peak_reserved_bytes"]} bytes。逐步Loss、gradient_norm、监督Token数和耗时见`sft_train/training_curve.jsonl`。', '',
            f'重载状态：`{reload["status"] if reload else "未执行"}`；最大probe logprob偏差：`{reload["max_logprob_delta"] if reload else None}`。同一冻结Base/Tokenizer的7项文件逐一核对；没有重新初始化Base冒充C。','']
    lines += ['## 4. C与C2同条件Greedy评测','',
        '原始公开Prompt，Thinking开启，Greedy（do_sample=False，vLLM temperature=0哨兵，非温度1采样）、top_p1/top_k-1、max_tokens16384，EOS与Thinking均计入Completion长度。未发送答案或Solver证明。26题为冻结DPO Train诊断；48题为冻结Validation Tune。Locked Final100也是历史冻结Validation子集，并非官方Holdout、也非首次盲测。','',
        'C的26题复用刚完成诊断的真实Greedy输出，逐题校验Prompt Token/模型Hash；C2诊断同用batch1。Tune中C/C2均重新生成，同用batch8和同一后端。没有根据Tune调整训练或解码。官方最终答案与原自然语言Verifier分开统计；OLS为原精度下可认证的最终全样本OLS，并非关键词命中。保守解析器存在表达覆盖限制，未认证不等于全部数学都错误。','']
    for panel in ['diagnosis','tune']:
        comp=optional(f'C_vs_C2_{panel}.json')
        if not comp:continue
        lines += [f'### {panel}','']+metric_table(comp['before'],comp['after'],['SFT C','SFT C2'])+['',
            f'Accuracy Wilson95%：C {comp["before"]["wilson95"]}，C2 {comp["after"]["wilson95"]}。McNemar exact p={comp["mcnemar_exact_pvalue"]:.6f}；准确率配对Bootstrap95%={comp["accuracy_delta_bootstrap"]["bootstrap95"]}（C2减C）。这些小面板/单训练Seed结果不能保证总体泛化或统计非劣。','',
            '|ID|C答对|C2答对|C证明认证|C2证明认证|C Token|C2 Token|','|---|---:|---:|---:|---:|---:|---:|']
        lines += [f'|{r["id"]}|{int(r["before_correct"])}|{int(r["after_correct"])}|{int(r["before_verified"])}|{int(r["after_verified"])}|{r["before_tokens"]}|{r["after_tokens"]}|' for r in comp['per_question']]+['']
    lines += ['## 5. 自动DPO Readiness','',
        '同一C2、26题，每题先8条（Temperature1、top_p1/top_k-1、Thinking、16384上限）；不足16对时全26题各补4条，最多12条/题。保存原始Completion、Prompt/Response IDs、逐Token Behavior Logprobs、finish_reason、Policy Fingerprint及真实/合成Train Prior来源。没有借用教师答案充当Rollout。','',
        '沿用Notebook04原DPO-B构造：同题同Policy/Prompt IDs；两边均官方答对、原NL Verifier认证、stop且未截断；Chosen最短、Rejected最长，差≥64Token且≥长答20%，每题最多一对。错误、截断和unverifiable不能成为Rejected。','']
    if ready:
        m=ready['metrics'];lines += [f'实际Rollout {ready["rollouts"]}条；最终答案通过{m["correct"]}，完整最终证明认证{m["proof_pass"]}，OLS认证{m["ols_certified"]}，截断{m["truncated"]}。有效偏好{ready["pairs"]}/26，覆盖率{pct(ready["pair_coverage"])}；最终状态`{ready["status"]}`。','',
            '非互斥失败原因：','```json',json.dumps(m['failure_reasons_nonexclusive'],ensure_ascii=False,indent=2),'```','']
    else:lines += ['Readiness未完成，不填虚构数量。','']
    lines += ['## 6. 条件DPO训练与Tune/Locked Final','']
    if dpo:
        lines += [f'真实DPO更新{dpo["optimizer_steps"]}次，训练{dpo["training_seconds"]:.2f}s，改变{dpo["changed_tensors"]}个张量。标准Sigmoid DPO，beta0.1、LR2e-6、累积4、2轮、clip1；Policy初始化与冻结Reference均C2，Sequence Log Probability按整个Response求和。复用course_worker.train_dpo及原分块数学实现，未改DPO Loss。', '',
            f'Reference未改变：`{dpo["reference_weights_unchanged"]}`。重载：`{dpo_reload}`。DPO Adapter路径`{DPO}`，Hash见Checkpoint Manifest。Tune完成后冻结Checkpoint再执行Locked Final，不根据Final改参数。','']
        for stage in ['tune','final']:
            comp=optional(f'C2_vs_DPO_{stage}.json')
            if comp:lines += [f'### DPO {stage}','']+metric_table(comp['before'],comp['after'],['C2','DPO'])+['',f'DPO-B四项点估计门槛：`{comp["dpo_b_point_gate"]}`；McNemar p={comp["mcnemar_exact_pvalue"]}；配对Bootstrap与逐题结果见`C2_vs_DPO_{stage}.json`。没有预注册非劣界值，不宣称统计非劣。','']
    else:
        partial=rows(DPO.parent/'training_curve.jsonl') if (DPO.parent/'training_curve.jsonl').exists() else []
        lines += [f'DPO未完成；已有曲线记录更新{len(partial)}次。'+('质量门槛未通过，未启动DPO。' if ready and ready['pairs']<16 else '上游或训练失败，完整现场保留。')+
            '没有完整DPO训练、重载和Locked Final比较结果。不得以“实现了代码”代替真实训练完成。','']
    lines += ['## 7. 最终回答与边界','']
    diagnosis=optional('C_vs_C2_diagnosis.json');tune=optional('C_vs_C2_tune.json')
    if diagnosis and tune:
        if tune['after']['proof_pass']:
            learned=f'C2在48题Tune有{tune["after"]["proof_pass"]}条最终完整证明通过，表明在该面板的部分题上实际展示正确OLS；仍需结合通过率和失败分布判断稳定性，不能称总体掌握。'
        else:
            learned='教师契约已修复且真实参与训练，但48题Tune最终完整证明认证仍为0，不能声称C2真正稳定学会了在最终回答展示完整OLS。教师74/74通过不能替代模型生成通过。'
        lines += ['- **修复后的SFT是否真正学会？**'+learned,
            '- **是否产生足够的高质量偏好对？**'+(f'得到{ready["pairs"]}对，达到16对门槛，已按协议进入DPO。' if ready and ready['pairs']>=16 else f'未达到16对，实际为{ready["pairs"] if ready else "Unknown"}对；停止DPO，没有降低标准。'),
            '- **下一步依据。**先看独立Verifier的具体失败是否为数学错误、缺步骤、格式或有限解析覆盖问题；不因失败自动加Epoch、换模型、改精度或扩大数据。本轮所有结论绑定固定配置和面板。','']
    if status['error']:lines += ['停止原因：`'+status['error']+'`。完整现场见failure日志。','']
    cpu=(RUN/'cpu_tests.log').read_text() if (RUN/'cpu_tests.log').exists() else ''
    match=re.search(r'Ran (\d+) tests? in ([0-9.]+)s',cpu)
    lines += [f'CPU实测：{match.group(0) if match else "未完成"}；通过状态：`{bool(match and chr(10)+"OK"+chr(10) in cpu)}`。日志、阶段完成hook、逐题审计和权重Hash独立保存。本进程显存上限8,000,000,000 bytes，GB10共享UMA，峰值不是全设备显存。没有中止其他GPU进程、Hosted API调用或修改PPO/GRPO。', '',
        '主要产物：`gravity_c2_teachers_74.jsonl`、`teacher_repair_audit.jsonl`、`sft_train/`、`sft_reload.json`、`comparison/`、`C_vs_C2_*.json`、`rollout8/`、`rollout4/`（若触发）、`readiness*.json`、`preferences_*.jsonl`、`final_status.json`、`artifact_manifest.json`。独立权重位于`.autodl/gravity_sft_c2_dpo_20261116_r1/`，不合并默认SFT入口。','']
    path=ROOT/'docs/DPO_SFT_C2_FINAL_REPORT.md'
    with path.open('x') as f:f.write('\n'.join(lines))
    manifest={'run_status':status['status'],'report':str(path.relative_to(ROOT)),'report_sha256':sha(path),
        'files':{str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and p.name not in ['driver.log','artifact_manifest.json']},
        'weight_paths':{'C2':str(C2) if trained else None,'DPO':str(DPO) if dpo else None},'hosted_api_calls':0}
    write(RUN/'artifact_manifest.json',manifest)


if __name__=='__main__':report()
