"""Prepare Phase A evidence/configurations only. This tool has no GPU/run action."""
from __future__ import annotations
import copy
import csv
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import course_protocol as p
from utils import run_synthetic_smoke as smoke

ORIGINAL = p.TASKSET/'synthetic/nemo_qwen3_06b_gpu_smoke_20261109_r1/data_designer'
RECOVERED = ROOT/'data/processed/benchmarks/qwen3-0_6b/real_gpu_20261109/recovered_course_r2'

# This is an independent Codex text diagnosis, written before program re-verification.
# It is a hypothesis log, not a second mathematical verifier or a Qwen GPU call.
MODEL_HYPOTHESES = {
    'numeral_system-seed1': ('题面被替换为错误映射，Trace 缺少 answer。', '逐字复制；显式必填字段。'),
    'numeral_system-seed2': ('题面增加错误映射；52+22 虽为74，Roman 拼接不规范。', '禁止额外示例；按位值分解。'),
    'numeral_system-seed3': ('输出了 NEWLINE/边界标记；Trace 引入零值空 Roman 分量。', '明确边界不输出、分量必须为正。'),
    'gravity-seed1': ('复制了边界文本；g_i 与 Query 时间、答案均不一致。', '隔离复制任务；提供公开输入计算表。'),
    'gravity-seed2': ('NEWLINE 文本污染；常数2被当作加速度，答案抄自示例。', '明确变量含义与代入顺序。'),
    'gravity-seed3': ('NEWLINE/边界文本污染；g=2、距离=.5 与原数据矛盾。', '逐示例计算而非复制公式系数。'),
    'equation_numeric-seed1': ('遗漏原题约束句；rules 是规则目录而非映射，示例与Query不闭合。', '强制对象结构；逐例列全结果。'),
    'equation_numeric-seed2': ('样例行被连接并插入运算符；Trace 缺少示例且规则未指派。', '精确复制换行；Schema 约束映射。'),
    'equation_numeric-seed3': ('遗漏约束句；Trace 疑有非法转义，且输出列表不全。', '严格 JSON；逐运算符候选上下文。'),
}


def diagnose(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    seeds = smoke.rows(ORIGINAL/'private_seeds.jsonl')
    outputs = smoke.rows(ORIGINAL/'raw_model_outputs.jsonl')
    old = {r['candidate_id']: r for r in smoke.rows(ORIGINAL/'preview.jsonl')}
    hypotheses = [{'seed_id': s['seed_id'], 'diagnostic_model': 'Codex; offline independent text review',
                   'hypothesis': MODEL_HYPOTHESES[s['seed_id']][0],
                   'suggested_repair': MODEL_HYPOTHESES[s['seed_id']][1],
                   'program_verified': False, 'qwen_diagnostic_request': False} for s in seeds]
    p.write_rows(destination/'model_hypotheses.jsonl', hypotheses)
    ns = smoke.load_notebook('round2_diagnostic_cpu_only', destination/'scratch')
    config = smoke.read(ORIGINAL/'compiled_config.json')
    from jinja2 import Environment, StrictUndefined
    env = Environment(undefined=StrictUndefined)
    results = []; stages = Counter(); task_failures = {}; seed_results = []
    for seed in seeds:
        candidate = copy.deepcopy(seed); ns['_solve_candidate'](candidate)
        reference = ns['_screen_verification'](candidate)
        reference_ok = bool(reference.get('answer_ok') and reference.get('trace_ok')
                            and ns['numeric_or_text_match'](candidate['answer'], seed['answer']))
        seed_results.append({'seed_id': seed['seed_id'], 'reference_ok': reference_ok,
                             'solver_reference': candidate, 'independent_verification': reference})
        stages['seed_pass'] += reference_ok; stages['solver_pass'] += reference_ok
        output = next(r for r in outputs if r['seed_id'] == seed['seed_id'])
        for i, kind in enumerate(('prompt_variant', 'trace_variant')):
            text = output[kind]; original = old[seed['seed_id']+':'+kind]
            prompt = env.from_string(config['columns'][i]['prompt']).render(**seed)
            # Historical proxy retained request hashes, not the request body.
            # This is explicitly a reconstruction, never claimed as captured wire data.
            verification = ns['verify_smoke_prompt'](seed, text) if kind == 'prompt_variant' else ns['verify_smoke_trace'](seed, text)
            shape = None; json_error = None
            if kind == 'trace_variant':
                try:
                    proof = json.loads(text)
                    keys = set(ns['smoke_trace_schema'](seed['task_type'])['required'])
                    shape = set(proof) == keys and isinstance(proof, dict)
                    if seed['task_type'] == 'equation_numeric': shape = shape and isinstance(proof.get('rules'), dict)
                except (ValueError, TypeError) as error:
                    shape = False; json_error = repr(error)
            stages['generator_outputs'] += bool(text)
            stages['verifier_pass'] += verification['verified']
            stages['verifier_fail'] += not verification['verified']
            stages['screening_rejected'] += bool(original['rejection_reasons'])
            stages['accepted'] += not bool(original['rejection_reasons'])
            task_failures.setdefault(seed['task_type'], Counter()).update(original['rejection_reasons'])
            results.append({'candidate_id': original['candidate_id'], 'seed': seed,
                'generator_prompt': prompt, 'generator_system_prompt': config['columns'][i]['system_prompt'],
                'prompt_evidence': 'reconstructed from frozen compiled config and seed; original request body unavailable',
                'raw_model_output': text, 'solver_reference': candidate,
                'verifier_current': verification, 'verifier_original': original['verification'],
                'schema_shape_valid': shape, 'json_parse_error': json_error,
                'screening_rejection_reasons': original['rejection_reasons'],
                'leakage_audit': original['leakage_audit'],
                'failure_step': ('Generator/protected prompt contract' if kind == 'prompt_variant' else
                                 'Generator/Schema' if not shape else 'Verifier/Trace computations'),
                'model_hypothesis': MODEL_HYPOTHESES[seed['seed_id']][0],
                'suggested_repair': MODEL_HYPOTHESES[seed['seed_id']][1]})
    p.write_rows(destination/'per_output_diagnosis.jsonl', results)
    p.write_rows(destination/'seed_solver_verifier.jsonl', seed_results)
    summary = ns['screen_data_designer_records'](outputs, seeds, destination/'strict_rescreen', require_transport=True)
    payload = {'status': 'diagnosed_cpu_only', 'original_run': 'nemo_qwen3_06b_gpu_smoke_20261109_r1',
        'stages': dict(stages), 'by_task_rejections': {k: dict(v) for k,v in task_failures.items()},
        'schema_shape_pass': sum(r['schema_shape_valid'] is True for r in results),
        'strict_rescreen': summary, 'new_model_calls': 0, 'gpu_experiments': 0,
        'input_hashes': {name: p.digest(ORIGINAL/name) for name in
                         ('private_seeds.jsonl','compiled_config.json','raw_model_outputs.jsonl','preview.jsonl')}}
    p.write(destination/'diagnosis.json', payload)
    return payload


def panels(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    old_ids = {r['id'] for name in ('tune','final') for r in p.read_rows(RECOVERED/f'panels/{name}.jsonl')}
    validation = p.csv_rows('validation'); train = p.csv_rows('train')
    tune = []; final = []; inventory = {}; short = []
    for i, task in enumerate(p.TASKS):
        pool = sorted((r for r in validation if r['task_type'] == task and r['id'] not in old_ids), key=lambda r:r['id'])
        random.Random(20261110+i).shuffle(pool)
        n = 24 if task == 'equation_numeric' else 48
        tune.extend(pool[:n]); final.extend(pool[n:]); short.extend(pool[:8])
        inventory[task] = {'validation_total': sum(r['task_type']==task for r in validation),
            'excluded_previous_panel': 12, 'tune': n, 'locked_final': len(pool)-n,
            'phase_c_smoke_subset_of_tune': 8}
    assert not ({r['id'] for r in tune} & {r['id'] for r in final})
    assert not ({r['id'] for r in train} & {r['id'] for r in tune+final})
    for name, rows in (('tune',tune),('locked_final',final),('phase_c_smoke',short)):
        p.write_rows(destination/f'{name}.jsonl', rows)
    manifest = {'status':'prepared_cpu_only_not_evaluated', 'inventory': inventory,
        'old_panel_ids_excluded': sorted(old_ids), 'holdout_used_for_selection': False,
        'panel_hashes': {f.name:p.digest(f) for f in destination.glob('*.jsonl')},
        'split_hashes': {s:p.digest(p.TASKSET/f'splits/{s}.csv') for s in ('train','validation','holdout_test')},
        'split_policy':'Official split unchanged. Holdout read only for integrity/leakage audit.',
        'final_access_policy':'No final generation until configs, rewards and selected checkpoints frozen.'}
    p.write(destination/'manifest.json', manifest)
    return manifest


def nemo_arms(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    ns=smoke.load_notebook('round2_repair_design_cpu_only',destination/'scratch')
    seeds=smoke.rows(ORIGINAL/'private_seeds.jsonl')
    for arm in ('R0','A','B','C'):
        arm_dir=destination/arm;arm_dir.mkdir()
        public=[{k:s[k] for k in ('seed_id','task_type','prompt')} for s in seeds]
        if arm=='C':
            for row,seed in zip(public,seeds): row['solver_context']=json.dumps(ns['smoke_solver_context'](seed),ensure_ascii=False)
        groups=[(task,[s for s in public if s['task_type']==task]) for task in p.TASKS] if arm=='B' else [(None,public)]
        for task,rows in groups:
            shard=arm_dir/(task or 'all');shard.mkdir()
            p.write_rows(shard/'public_seeds.jsonl',rows)
            builder,_=ns['build_round2_data_designer'](shard/'public_seeds.jsonl','http://127.0.0.1:8000/v1',arm,task)
            p.write(shard/'compiled_config.json',builder.build().model_dump(mode='json'))
        p.write(arm_dir/'design.json',{'synthetic_run_id':f'nemo_round2_20261110_{arm.lower()}_r1',
            'status':'prepared_only_gpu_not_run','arm':arm,'seeds':9,'expected_outputs':18,
            'temperature':.3,'top_p':.9,'seed':42,'prompt_max_tokens':1024,'trace_max_tokens':2048,
            'new_output_directory_required':True,'merge_into_sft':False,'max_correction_steps':0,
            'quality_metrics':['accepted_rate','schema_valid_rate','exact_trace_pass','leakage_rejections',
                               'output_tokens','wall_seconds','device_peak_sampled_bytes'],
            'seed_hash':p.digest(ORIGINAL/'private_seeds.jsonl')})


def train_panels(destination):
    """Freeze Phase C training inputs without looking at validation outcomes."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    train=p.csv_rows('train');selected={}
    for i,(task,count) in enumerate((('gravity',16),('equation_numeric',8))):
        pool=sorted((r for r in train if r['task_type']==task),key=lambda r:r['id'])
        random.Random(20261110+1000+i).shuffle(pool)
        selected[task]=pool[:count]
    panels={'dpo_b':selected['gravity'],'ppo':selected['equation_numeric'],
            'grpo':selected['equation_numeric'][:4]}
    for method,rows in panels.items():
        p.write_rows(destination/f'{method}.jsonl',rows)
    result={'status':'prepared_only_no_model_outputs','source_train_sha256':p.digest(p.TASKSET/'splits/train.csv'),
        'panels':{m:{'n':len(rows),'sha256':p.digest(destination/f'{m}.jsonl')} for m,rows in panels.items()},
        'selection':'Uniform fixed seed; GRPO is a subset of PPO; no validation/holdout selection',
        'reward_ablation_panels_identical':True,'dpo_preference_mode':'B'}
    p.write(destination/'manifest.json',result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('diagnose','panels','nemo-arms','train-panels'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={'diagnose':diagnose,'panels':panels,'nemo-arms':nemo_arms,'train-panels':train_panels}[args.action](args.output)
    print(json.dumps(result or {'status':'prepared_cpu_only'},ensure_ascii=False))


if __name__=='__main__': main()
