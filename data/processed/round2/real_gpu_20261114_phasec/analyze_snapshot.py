"""Export and independently audit a finished GPU suite. No generation or training."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from utils.course_protocol import TASKSET,digest,metrics,read,read_rows,write
from utils.round2_evaluation import paired_comparison,paired_mean_analysis,mcnemar_exact


def export_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(source)!=digest(target):raise ValueError(f'Refuse changed evidence: {target}')
    else:shutil.copy2(source,target)


def prompt_comparison(before,after):
    def index(rows):
        result={(r['sample_id'],r['task_type'],r['max_tokens']):r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate prompt comparison key')
        return result
    a,b=index(before),index(after)
    if set(a)!=set(b):raise ValueError('Unmatched prompt comparison questions')
    for key in a:
        for field in ('seed','temperature','top_p','top_k','verifier_version','policy_version'):
            if a[key][field]!=b[key][field]:raise ValueError('Prompt comparison changed policy/decoding')
    results={}
    for task in sorted({key[1] for key in a}):
        keys=sorted(k for k in a if k[1]==task);n=len(keys)
        lost=sum(a[k]['correct'] and not b[k]['correct'] for k in keys)
        gained=sum(not a[k]['correct'] and b[k]['correct'] for k in keys)
        results[task]={'n':n,'accuracy_before':sum(a[k]['correct'] for k in keys)/n,
            'accuracy_after':sum(b[k]['correct'] for k in keys)/n,'lost':lost,'gained':gained,
            'accuracy_mcnemar_pvalue':mcnemar_exact(lost,gained),
            'paired_length':paired_mean_analysis([len(b[k]['completion_token_ids'])-len(a[k]['completion_token_ids']) for k in keys]),
            'conditions':['original','closed_trace_v1'],
            'per_question':[{'sample_id':k[0],'correct_before':a[k]['correct'],'correct_after':b[k]['correct'],
                'prompt_ids_sha_before':hashlib.sha256(json.dumps(a[k]['prompt_token_ids']).encode()).hexdigest(),
                'prompt_ids_sha_after':hashlib.sha256(json.dumps(b[k]['prompt_token_ids']).encode()).hexdigest()} for k in keys]}
    return results


def audit_evaluation(source,rows,result):
    """Re-score unchanged raw generations and check the frozen Validation panel."""
    from utils.course_worker import verify_rows,attach_proof_audits,notebook_api
    panel=read_rows(source.parents[1]/'panels/evaluation.jsonl')
    with (TASKSET/'splits/validation.csv').open() as stream:
        official={r['id']:r for r in csv.DictReader(stream)}
    if len(rows)!=len(panel) or result['records']!=len(panel):
        raise ValueError('Incomplete evaluation rows')
    raw=read_rows(source/'raw_partial_rows.jsonl')
    if len(raw)!=len(panel):raise ValueError('Incomplete raw evaluation evidence')
    contract=read(source/'job.json').get('output_contract')
    for i,(row,seed,original) in enumerate(zip(rows,panel,raw)):
        truth=official[str(seed['id'])]
        if any(seed[field]!=truth[field] for field in ('prompt','task_type','answer')):
            raise ValueError('Registered panel differs from official Validation')
        if (row['sample_id']!=str(seed['id']) or row['task_type']!=seed['task_type']
            or row['expected_answer']!=seed['answer'] or row['source_split']!='validation'
            or row['source']!='vllm_evaluation' or row['fixture']
            or row['output_contract']!=contract):
            raise ValueError('Wrong evaluation question/answer/source/contract')
        if original['panel_index']!=i or len(original['outputs'])!=1:
            raise ValueError('Raw question/output alignment mismatch')
        output=original['outputs'][0]
        for key,raw_key in [('completion','text'),('completion_token_ids','token_ids'),
                            ('finish_reason','finish_reason'),('stop_reason','stop_reason')]:
            if row[key]!=output[raw_key]:raise ValueError('Raw output changed: '+key)
        if row['prompt_token_ids']!=original['prompt_token_ids']:
            raise ValueError('Raw input token IDs changed')
        if (row['truncated']!=(row['finish_reason']=='length')
            or len(row['completion_token_ids'])>row['max_tokens']):
            raise ValueError('Invalid truncation or output token accounting')
        if not contract and row['prompt']!=seed['prompt']:
            raise ValueError('Original prompt changed')
    rescored=verify_rows(copy.deepcopy(rows))
    fields=['correct','predicted','extraction_method','verification_method','verifier_version']
    if contract:
        rescored=attach_proof_audits(rescored)
        fields+=['proof_audit','format_valid','exact_trace_pass','proof_verifier_version','proof_scope']
        reward=notebook_api('grpo')['equation_reward_components']
        for row in rescored:
            if row['task_type']=='equation_numeric':
                row['reward_components']=reward(row,row['proof_audit'],'R1')
    for before,after in zip(rows,rescored):
        for field in fields+(['reward_components'] if 'reward_components' in before else []):
            if before[field]!=after[field]:raise ValueError('Re-scoring mismatch: '+field)
    recomputed=metrics(rescored)
    if recomputed!=result['metrics']:raise ValueError('Reported metrics differ from raw-row metrics')
    return {'status':'passed','records':len(rows),'unchanged_official_answers':True,
            'questions_verified_in_official_validation':True,
            'raw_generation_alignment':True,'output_token_count_recomputed':True,
            'official_verifier_rescored':True,'proof_and_reward_rescored':bool(contract),
            'metrics_recomputed':recomputed,'rows_sha256':digest(source/'rows.jsonl'),
            'raw_generations_sha256':digest(source/'raw_partial_rows.jsonl')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-interrupted-snapshot',action='store_true',
                        help='Export recovered evidence without declaring the running remote suite finished')
    args=parser.parse_args();cache=args.snapshot;out=args.output
    state=read(cache/'state.json');suite=read(cache/'suite.json')
    incomplete=state['status']=='running'
    if incomplete and not args.allow_interrupted_snapshot:
        raise ValueError('Wait for finished scheduling, or explicitly export an incomplete recovered snapshot')
    out.mkdir(parents=True,exist_ok=False)
    remote_suite=Path(next(iter(suite['runs'].values()))).parent
    for name in ('suite.json','state.json','report.json','supervisor.log','readiness_reward_audit.json','user_scope_20261114.json'):
        if (cache/name).exists():export_file(cache/name,out/name)
    sources_verified={};split_hashes={}
    for method,remote in suite['runs'].items():
        run=cache/Path(remote).relative_to(remote_suite)
        protocol=read(run/'protocol.json')
        for split,sha in protocol['split_sha256'].items():
            if digest(TASKSET/f'splits/{split}.csv')!=sha:raise ValueError('Official split changed')
            split_hashes[split]=sha
        for rel,sha in protocol['frozen_files'].items():
            if digest(run/rel)!=sha:raise ValueError('Frozen panel changed')
            export_file(run/rel,out/'registered'/method/rel)
        export_file(run/'protocol.json',out/'registered'/method/'protocol.json')
        for rel,sha in protocol['sources'].items():
            if rel in sources_verified:continue
            original=subprocess.check_output(['git','show',suite['source_commit']+':'+rel],cwd=ROOT)
            if hashlib.sha256(original).hexdigest()!=sha:raise ValueError('Source differs from registered Git commit')
            sources_verified[rel]=sha
    evaluations={};job_summaries=[];raw_manifest=[];repetition=[];evaluation_audits={}
    for key,entry in state['jobs'].items():
        source=cache/Path(entry['output']).relative_to(remote_suite)
        result=read(source/'result.json') if (source/'result.json').exists() else None
        guard=read(source/'guard.json') if (source/'guard.json').exists() else entry['guard']
        if result and digest(source/'result.json')!=entry['result_sha256']:raise ValueError('Result SHA mismatch')
        telemetry=read_rows(source/'device_memory.jsonl') if (source/'device_memory.jsonl').exists() else []
        if telemetry and guard['status']!='running':
            observed=max([guard['device']['used_bytes']]+[r['used_bytes'] for r in telemetry])
            if observed!=guard['device_peak_sampled_bytes'] or len(telemetry)!=guard['samples']:
                raise ValueError('Guard/sample mismatch')
        # NeMo and training evidence already have dedicated published subdirectories.
        if key.startswith('eval_'):
            for file in source.iterdir():
                if file.is_file() and file.suffix not in ('.pt','.bin','.safetensors'):
                    export_file(file,out/'evaluations'/key/file.name)
            if result and guard['status']=='completed' and result['status']=='completed':
                evaluations[key]=read_rows(source/'rows.jsonl')
                evaluation_audits[key]=audit_evaluation(source,evaluations[key],result)
                for row in evaluations[key]:
                    tokens=row['completion_token_ids'];tail=tokens[-2048:]
                    grams=Counter(tuple(tail[i:i+32]) for i in range(max(0,len(tail)-31)))
                    repetition.append({'job':key,'sample_id':row['sample_id'],'task_type':row['task_type'],
                        'output_tokens':len(tokens),'truncated':row['truncated'],
                        'has_thinking_close':'</think>' in row['completion'],
                        'last_2048_max_repeated_32gram_count':max(grams.values(),default=0),
                        'last_2048_distinct_token_ratio':len(set(tail))/len(tail) if tail else None,
                        'raw_completion_tail':row['completion'][-1200:],
                        'scope':'Descriptive repetition diagnostic; no output filtering or scoring changes'})
        if (source/'rows.jsonl').exists():
            raw_manifest.append({'job':key,'relative_source':str((source/'rows.jsonl').relative_to(cache)),
                'sha256':digest(source/'rows.jsonl'),'bytes':(source/'rows.jsonl').stat().st_size,
                'records':len(read_rows(source/'rows.jsonl'))})
        job_summaries.append({'job':key,'kind':entry.get('kind'),'method':entry.get('method'),
            'guard_status':guard['status'],'worker_status':result.get('status') if result else None,
            'wall_seconds':guard.get('wall_seconds'),'peak_sampled_bytes':guard.get('device_peak_sampled_bytes'),
            'memory_check':guard.get('memory_check'),'optimizer_steps':result.get('optimizer_steps',0) if result else 0,
            'training_seconds':result.get('training_seconds') if result else None,
            'generation_seconds':result.get('generation_seconds') if result else None,
            'metrics':result.get('metrics') if result else None})
    comparisons={};prompt_effects={}
    for contract in ('original','closed_trace_v1'):
        for model,parent in [('sft','base'),('ppo','sft')]:
            a,b=f'eval_{parent}_{contract}',f'eval_{model}_{contract}'
            if a in evaluations and b in evaluations:comparisons[b]=paired_comparison(evaluations[a],evaluations[b])
    for model in ('base','sft','ppo'):
        a,b=f'eval_{model}_original',f'eval_{model}_closed_trace_v1'
        if a in evaluations and b in evaluations:prompt_effects[model]=prompt_comparison(evaluations[a],evaluations[b])
    requested_evaluations=[f'eval_{m}_{v}' for m in ('base','sft','ppo') for v in ('original','closed_trace_v1')]
    summary={'status':'snapshot_incomplete' if incomplete else state['status'],
        'source_suite_status':state['status'],'source_commit':suite['source_commit'],'jobs':job_summaries,
        'comparisons':comparisons,'prompt_condition_effects':prompt_effects,
        'independent_evaluation_audits':evaluation_audits,'official_split_sha256_verified':split_hashes,
        'sources_verified_against_registered_git_commit':sources_verified,'raw_rows_manifest':raw_manifest,
        'wall_seconds':state['execution_ended_at_unix']-state['started_at_unix']
            if 'execution_ended_at_unix' in state else None,
        'missing_trained_methods':[m for m in ('dpo','ppo','grpo') if m not in state['checkpoints']],
        'requested_algorithm_evaluation_scope':['PPO'],'comparators':['Base','SFT'],
        'dpo_grpo_evaluations_cancelled_by_user':True,
        'incomplete_recovered_snapshot':incomplete,
        'remote_execution_stopped_confirmed':not incomplete,
        'incomplete_job_keys':[k for k,e in state['jobs'].items() if e['guard']['status']=='running'],
        'requested_evaluations_complete':all(k in evaluations for k in requested_evaluations),
        'missing_requested_evaluations':[k for k in requested_evaluations if k not in evaluations],
        'formal_phase_d_started':False,'locked_final_panel_used':False,'holdout_used_for_selection':False,
        'continuous_memory_bound_verified':False,
        'analysis_script_sha256':digest(Path(__file__)),
        'statistical_family_note':'User reduced algorithm evaluation to PPO; original and closed contracts reported separately.',
        'note':'Phase C screening only, 8 Validation questions per task. No convergence or general accuracy improvement claimed.'}
    write(out/'independent_summary.json',summary)
    write(out/'repetition_diagnostics.json',{'rows':repetition})
    table=['# 第二轮短程真实GPU验证结果','','| 条件 | 模型 | 题型 | n | 正确 | 截断 | 平均输出Token | 正确回答平均Token |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for key,rows in evaluations.items():
        entry=next(j for j in job_summaries if j['job']==key)
        for task,m in entry['metrics'].items():
            correct=m['mean_correct_output_tokens']
            table.append(f'| {key} | {rows[0]["policy_version"]} | {task} | {m["n"]} | {m["correct"]} | {m["truncated"]} | {m["mean_output_tokens"]:.2f} | {correct if correct is not None else "N/A"} |')
    table+=['','输出含Thinking、错误、截断及EOS，不含输入。每类n=8，仅用于Phase C筛查；Wilson与配对检验见JSON。',
            '',f'导出状态：{summary["status"]}；未完成评测：{summary["missing_requested_evaluations"]}。',
            '',f'缺少真实训练后Checkpoint的方法：{summary["missing_trained_methods"]}。不以SFT权重冒充训练结果。',
            '', 'DPO-B无合格偏好，GRPO无组内奖励信号：未满足正式实验门槛，未启动Phase D或锁定final。']
    (out/'RESULTS.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({'status':summary['status'],'jobs':len(job_summaries),'evaluations':len(evaluations),
                      'wall_seconds':summary['wall_seconds'],'output':str(out)}))


if __name__=='__main__':main()
