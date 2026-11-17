"""Offline result/report aggregation for the isolated C3 run; no inference."""
from __future__ import annotations
import json,re,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from utils.run_gravity_sft_c3 import RUN,C3,DPO,WEIGHTS,CONFIG,read,rows,sha,write,write_rows


def optional(name):return read(RUN/name) if (RUN/name).exists() else None
def pct(x):return f'{100*x:.2f}%'
def num(x):return 'Unknown' if x is None else f'{x:.2f}'


def comparison_table(comp):
    a,b=comp['before'],comp['after']
    result=['|指标|SFT C|C3|','|---|---:|---:|']
    for name,key in [('官方答案','correct'),('完整双通道证明','proof_pass'),('OLS数值认证','ols_certified'),('截断','truncated')]:
        result.append(f'|{name}|{a[key]}/{a["n"]}|{b[key]}/{b["n"]}|')
    for name,key in [('平均Completion Tokens','mean_tokens'),('中位Completion Tokens','median_tokens'),('官方正确回答平均Tokens','official_correct_mean_tokens'),('正确完整证明平均Tokens','correct_verified_mean_tokens')]:
        result.append(f'|{name}|{num(a[key])}|{num(b[key])}|')
    result += ['',f'Wilson95%：C={a["wilson95"]}；C3={b["wilson95"]}。McNemar exact p={comp["mcnemar_exact_pvalue"]}；C3−C Accuracy配对Bootstrap95%={comp["accuracy_delta_bootstrap"]["bootstrap95"]}。',
        '单训练Seed、固定有限面板；不宣称总体掌握或统计非劣。','']
    return result


def report():
    status=read(RUN/'final_status.json');c=read(CONFIG);teacher=optional('teacher_channel_summary.json');train=optional('sft_train/result.json')
    freeze=optional('c3_checkpoint_freeze.json');ready=optional('readiness.json');reload=optional('sft_reload.json');dpo=optional('dpo_training_result.json')
    tests=(RUN/'cpu_tests.log').read_text() if (RUN/'cpu_tests.log').exists() else '';match=re.search(r'Ran (\d+) tests? in ([0-9.]+)s',tests)
    failures=[]
    for path in sorted(RUN.glob('*_scored.jsonl')):
        for r in rows(path):
            audit=r['proof_audit']
            failures.append({'panel_file':path.name,'sample_id':r['sample_id'],'task_type':r['task_type'],
                'correct':r['correct'],'proof_verified':audit['verified'],'thinking_verified':audit.get('thinking_verified'),
                'legacy_v1_2_verified':r.get('legacy_v1_2_audit',{}).get('verified'),
                'finish_reason':r['finish_reason'],'tokens':len(r['completion_token_ids']),
                'predicted':r.get('predicted'),'expected':r['expected_answer'],'reasons':audit['reasons']})
    for path in [RUN/'rollouts_audited_12.jsonl' if (RUN/'rollouts_audited_12.jsonl').exists() else RUN/'rollouts_audited_8.jsonl']:
        if path.exists():
            for r in rows(path):failures.append({'panel_file':path.name,'sample_id':r['sample_id'],'group_index':r['group_index'],
                'task_type':r['task_type'],'correct':r['correct'],'proof_verified':r['proof_audit']['verified'],
                'thinking_verified':r['proof_audit']['thinking_verified'],'legacy_v1_2_verified':r['legacy_v1_2_audit']['verified'],
                'finish_reason':r['finish_reason'],'tokens':len(r['completion_token_ids']),
                'predicted':r.get('predicted'),'expected':r['expected_answer'],'reasons':r['proof_audit']['reasons']})
    write_rows(RUN/'per_question_failure_analysis.jsonl',failures)
    lines=['# Gravity SFT C3 原生 Thinking 修复与条件 DPO 报告','',f'Run：`{RUN.name}`；最终状态：`{status["status"]}`。','',
        'C3从原SFT C初始化，复用C2的74条完整数值证明及全部训练超参数。唯一训练变量是通道：完整证明进入reasoning_content，assistant.content只保留Final answer与唯一Boxed Answer。训练/推理均enable_thinking=True，无Replay。','',
        f'原C SHA256：`{status["original_C_sha256"]}`。Base `Qwen/Qwen3-0.6B` Revision `{c["base_revision"]}`；BF16 LoRA r8/alpha16/dropout0。',
        f'保护历史/source/权重文件{status["protected_files"]}个，全部未改：`{status["all_protected_unchanged"]}`。C2权重、日志、Manifest和历史v1.2结果只读；默认SFT入口、原Solver、DPO Loss/Reference实现、PPO/GRPO不变。','',
        '## 实现与CPU验收','',
        'Notebook01新增gravity_c3_split_channels、gravity_c3_numeric_certificates、verify_gravity_dual_channel。独立参考复用原public-only OLS校验器，不调用训练模型自评。必须有唯一Thinking关闭边界、非空Thinking和唯一Final数字Box。',
        'Thinking必须展示全部观测、每项t²d/t⁴、两项覆盖全部观测的数值和、正确OLS数值代入、拟合g、Query时间及未舍入距离。封闭AST算术逐式重新计算，结构匹配确保全样本数值和没有被单点估计替代。无强制逐项g_i。Final须同时满足官方答案比较与独立OLS参考，禁止仅靠OLS关键词放行。',
        '数值精度保持rel_tol=1e-4、abs_tol=1e-5。v1.2函数未变，其final-only验收另列保存；新协议不把任意Thinking直接当作正确证明。解析语法有限，无法认证不等于已证明数学错误。',
        'Notebook03复用原render_gravity_c2_final_proof，新增make_gravity_c3_record、encode_gravity_c3_example、train_gravity_c3_branch。原训练循环仅增加默认关闭的on_epoch保存回调，不改变优化器、损失或默认编码器。','',
        f'CPU实测：{match.group(0) if match else "未完成"}；通过：`{bool(match and chr(10)+"OK"+chr(10) in tests)}`。见cpu_tests.log；测试包含74教师、损坏中间项/求和/单点替换/Final冲突、Token Mask、通道、历史隔离及DPO门槛。','']
    if teacher:
        lines += [f'教师：{teacher["dual_verified"]}/74新双通道通过；历史v1.2只检查Final时{teacher["legacy_final_only_verified"]}/74（Final有意只含答案）。74/74数学Token全部监督；每Epoch监督{teacher["supervised_tokens_per_epoch"]} Token，最长Chat {teacher["max_full_chat_tokens"]} Token，上限4096；无截断。',
            '64条historical_r1与10条added_real_c逐条保持source_id/source_row/Prompt/Answer；重新交叉核对实际Train和Validation/Holdout Prompt。来源统一solver_generated_rendered，不伪称模型自主CoT。','']
    lines += ['## 真实训练与Epoch选择','',
        'AdamW LR2e-5、weight_decay0.01、eps1e-8、microbatch1、accumulation8、clip1、2epochs、seed20261116、max_length4096、projection chunk64，与C2相同。每Epoch10次更新，保存权重、Optimizer和重载探针。',
        '选择规则预先固定：Tune48完整双通道证明数、官方正确数、较少截断、较短均长，仍相同则较早Epoch。Roman/Equation回归与Locked Final不参与选择；不根据结果重训或变更超参数。','']
    if train:
        lines += [f'真实Optimizer更新{train["optimizer_steps"]}次，监督Token={train["supervised_tokens"]}；训练{train["training_seconds"]:.2f}s，墙钟{train["wall_seconds"]:.2f}s；本进程Torch峰值reserved={train["memory"]["peak_reserved_bytes"]} bytes。',
            '|Epoch|Updates|Changed tensors|Adapter SHA256|','|---|---:|---:|---|']
        for e in train['epoch_checkpoints']:lines.append(f'|{e["epoch"]}|{e["optimizer_steps"]}|{e["changed_tensor_count"]}|`{e["adapter_sha256"]}`|')
        lines += ['',f'重载：`{json.dumps(reload,ensure_ascii=False)}`。','']
    if freeze:
        lines += [f'选中Epoch {freeze["selected_epoch"]}；C3 Adapter：`{C3}`；SHA256：`{freeze["adapter_sha256"]}`。','',
            '|Epoch|Tune correct|Dual proof|Truncated|Mean Tokens|','|---|---:|---:|---:|---:|']
        for e,m in freeze['tune_metrics_by_epoch'].items():lines.append(f'|{e}|{m["correct"]}/48|{m["proof_pass"]}/48|{m["truncated"]}/48|{m["mean_tokens"]:.2f}|')
        lines += ['']
    lines += ['## 同条件C/C3评测','',
        '原始Prompt、同一固定Tokenizer/Chat Template、Thinking、Greedy（do_sample=False；vLLM temperature=0哨兵）、top_p1/top_k−1、max_tokens16384。真实输出长度包含Thinking、EOS、错误和截断，不含输入。每个比较双方同面板、同Prompt IDs、同Token上限。',
        'Tune48和Locked Final100均为历史Validation面板，不是未经接触的官方Holdout。所有Epoch及可选DPO Checkpoint冻结后才一次执行Locked Final；未据其结果调参。Roman/Equation各50条复用已冻结Validation回归面板，只报告官方最终答案；不虚构其完整自然语言证明认证。','']
    for name in ['tune','diagnosis','numeral_system','equation_numeric','final']:
        comp=optional(f'C_vs_C3_{name}.json')
        if comp:
            lines += [f'### {name}','']+comparison_table(comp)
            if 'subtypes' in comp:lines += ['Equation子类型：','```json',json.dumps(comp['subtypes'],ensure_ascii=False,indent=2),'```','']
    lines += ['## DPO Readiness与条件训练','',
        '选中C3对同26题先8条/题；未达16对则全题补4条，最多12条/题。温度1、top_p1/top_k−1、Thinking开启、max_tokens16384。两侧均需官方正确、新双通道严格通过、stop且未截断；同Prompt IDs/Policy；Chosen最短、Rejected最长，差≥64Token且≥长答20%，每题最多一对。','']
    if ready:
        m=ready['metrics'];lines += [f'真实Rollout {ready["rollouts"]}条；官方正确{m["correct"]}；双通道证明{m["proof_pass"]}；OLS认证{m["ols_certified"]}；截断{m["truncated"]}；有效偏好{ready["pairs"]}/26。',
            '非互斥失败原因：','```json',json.dumps(m['failure_reasons_nonexclusive'],ensure_ascii=False,indent=2),'```','']
    if dpo:
        lines += [f'达到16对门槛后完成真实DPO，{dpo["optimizer_steps"]}次更新；Policy初始与冻结Reference均为选中C3。标准Sigmoid DPO、beta0.1、LR2e-6、accum4、2epochs、clip1；Response Token logprob求和，包含Thinking与EOS。Reference未变：{dpo["reference_weights_unchanged"]}。',
            '仅显式C3作业在既有worker入口注入新双通道auditor，结束即恢复；没有修改损失、Reference冻结或其他实验的默认验收。','']
        for panel in ['tune','final']:
            comp=optional(f'C3_vs_DPO_{panel}.json')
            if comp:lines += [f'DPO {panel}（表中C列指C3 Reference，C3列指DPO）：','']+comparison_table(comp)
    else:lines += ['严格偏好未达16对或上游失败，未启动DPO：0次DPO更新，无DPO Checkpoint或DPO收益声明。C3的SFT更新不得计作DPO更新。','']
    lines += ['## 逐题失败与最终判断','',
        '完整逐题/逐候选记录见per_question_failure_analysis.jsonl；C_vs_C3_*.json包含全部配对正确性、认证和Token长度；*_scored.jsonl保留原始输出、独立参考、数值证书、失败原因和旧v1.2审计。','']
    if (RUN/'C_vs_C3_tune.json').exists():
        tune=read(RUN/'C_vs_C3_tune.json');m=tune['after']
        lines += [f'**是否真正展示完整OLS？** Tune48严格认证{m["proof_pass"]}/48，官方答案{m["correct"]}/48；'+('本面板存在可认证输出，但不能等同于稳定掌握。' if m['proof_pass'] else '不能宣称模型学会了完整OLS证明。'),
            f'**是否具备DPO资格？** '+(f'有效偏好{ready["pairs"]}对，已达到门槛并按协议执行。' if ready and ready['pairs']>=16 else f'有效偏好{ready["pairs"] if ready else "Unknown"}对，不具备≥16对资格；未放宽验证。'),'']
    if status['error']:lines += ['停止原因：`'+status['error']+'`；完整现场保留。','']
    lines += ['没有Replay、Hosted API、默认SFT合并、PPO/GRPO变更或历史权重覆盖；不杀死其他GPU进程。8,000,000,000-byte守卫针对本进程Torch分配器，GB10共享UMA的全设备显存用量不可由该值推断。',
        '权重/Optimizer保存在忽略目录.autodl/gravity_sft_c3_dpo_20261117_r1；Git归档日志、数据、逐题结果、Manifest和Hash。','']
    path=ROOT/'docs/DPO_SFT_C3_FINAL_REPORT.md'
    with path.open('x') as f:f.write('\n'.join(lines))
    write(RUN/'artifact_manifest.json',{'status':status['status'],'report':str(path.relative_to(ROOT)),'report_sha256':sha(path),
        'files':{str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and p.name not in ['driver.log','artifact_manifest.json']},
        'weight_paths':{'C3':str(C3) if C3.exists() else None,'DPO':str(DPO) if DPO.exists() else None}})


if __name__=='__main__':report()
