"""Offline audit of real HTTP artifacts, using Notebook 01's original math.

No model requests. Never rewrites the frozen inputs or generation results.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from collections import Counter

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[3]
sys.path.insert(0, str(ROOT))
from utils.run_nvidia_synthetic_smoke import load_notebook, rows, read, verify_prepared


def new_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


config, seeds = verify_prepared(RUN)
seed_by_id = {s['seed_id']: s for s in seeds}
result = read(RUN / 'generation/result.json')
requests = rows(RUN / 'generation/http_requests.jsonl')
candidates = rows(RUN / 'generation/candidates.jsonl')
checks = read(RUN / 'seed_checks.json')
expected = {(s['seed_id'], kind) for s in seeds for kind in ('prompt_variant', 'trace_variant')}
observed = {(r['seed_id'], r['variant_kind']) for r in requests}
assert expected == observed and len(requests) == 18
assert result['all_seed_attempts_completed'] and len(candidates) == 18
assert len(rows(RUN / 'generation/accepted.jsonl')) == 9
assert len(rows(RUN / 'generation/preview.jsonl')) == 9
metrics = {}
for task in ('numeral_system', 'gravity', 'equation_numeric'):
    group = [c for c in candidates if c['task_type'] == task]
    prompt = [c for c in group if c['variant_kind'] == 'prompt_variant']
    trace = [c for c in group if c['variant_kind'] == 'trace_variant']
    metrics[task] = {'prompt_accepted': sum(c['verification_status'] == 'accepted' for c in prompt),
        'trace_schema_pass': sum(c['verification'].get('schema_valid', False) for c in trace),
        'closed_proof_all_checks_pass': sum(c['verification']['verified'] for c in trace),
        'full_trace_accepted': sum(c['verification_status'] == 'accepted' for c in trace),
        'trace_truncated': sum(c['transport'].get('finish_reason') == 'length' for c in trace),
        'trace_unverified_thinking_preview': sum('unverified_reasoning_content' in c['rejection_reasons'] for c in trace)}

model_usage = {}
for alias, model in config['models'].items():
    subset = [r for r in requests if r['model'] == model['model']]
    model_usage[alias] = {'model': model['model'], 'requests': len(subset),
        'input_tokens': sum(r.get('usage', {}).get('prompt_tokens', 0) for r in subset),
        'output_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in subset),
        'reported_reasoning_tokens': sum(r.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', 0) for r in subset),
        'usage_missing_requests': sum(not r.get('usage') for r in subset),
        'thinking_present_responses': sum(bool((r.get('response', {}).get('choices') or [{}])[0].get('message', {}).get('reasoning_content')) for r in subset)}
    model_usage[alias]['mean_output_tokens'] = model_usage[alias]['output_tokens'] / len(subset)

diagnoses = []
with tempfile.TemporaryDirectory() as directory:
    ns = load_notebook('nvidia_final_offline_audit', Path(directory) / 'loader')
    for candidate in candidates:
        seed = seed_by_id[candidate['seed_id']]
        index, request = next((i, r) for i, r in enumerate(requests) if
                              (r['seed_id'], r['variant_kind']) == (candidate['seed_id'], candidate['variant_kind']))
        message = request['response']['choices'][0]['message']
        prompt = request['request']['messages'][-1]['content']
        if isinstance(prompt, list):
            prompt = ''.join(p.get('text', '') for p in prompt)
        diagnosis = {'candidate_id': candidate['candidate_id'],
            'public_seed': {k: seed[k] for k in ('seed_id', 'task_type', 'prompt')},
            'private_seed_ref': '../private_seeds.jsonl#' + seed['seed_id'],
            'standard_answer_local_only': seed['answer'], 'solver_reference_local_only': checks[seed['seed_id']]['solver_result'],
            'generator_prompt': prompt, 'raw_content': message.get('content'),
            'raw_http_record_index': index, 'raw_http_ref': 'generation/http_requests.jsonl',
            'thinking_verified': False, 'thinking_present': bool(message.get('reasoning_content')),
            'verifier_result': candidate['verification'], 'screening_reasons': candidate['rejection_reasons'],
            'verification_status': candidate['verification_status'], 'failure_steps': [], 'repair_hypotheses': []}
        if request.get('finish_reason') == 'length':
            diagnosis['failure_steps'].append('Generator: output budget exhausted before closed JSON')
            diagnosis['repair_hypotheses'].append('Separate prompt/budget experiment; retain all mathematical checks and examples')
        if candidate['variant_kind'] == 'trace_variant' and candidate['verification'].get('schema_valid'):
            proof = candidate['verification']['proof']
            if seed['task_type'] == 'gravity':
                examples, query = ns['_gravity_prompt_parts'](seed['prompt'])
                fit = ns['_gravity_verifier_fit'](examples)
                distance = ns['_gravity_verifier_distance'](float(query), fit['g_ols'])
                field_checks = {'g_estimates_count': len(proof['g_estimates']) == len(examples),
                    'g_estimates': [ns['_smoke_close'](a, b) for a, b in zip(proof['g_estimates'], fit['g_estimates'])],
                    'g_ols': ns['_smoke_close'](proof['g_ols'], fit['g_ols']),
                    'query_t_exact': str(proof['query_t']) == query,
                    'query_distance': ns['_smoke_close'](proof['query_distance'], distance),
                    'official_final_answer': ns['official_numeric_match'](proof['answer'], seed['answer'])}
                diagnosis['field_checks'] = field_checks
                diagnosis['reference_fields_local_only'] = {'g_estimates': fit['g_estimates'], 'g_ols': fit['g_ols'],
                    'query_t': query, 'query_distance': distance, 'answer': seed['answer']}
                if not field_checks['query_t_exact']:
                    diagnosis['failure_steps'].append('Verifier: query_t contains a unit suffix; original numeric string required')
                    diagnosis['repair_hypotheses'].append('Generator prompt must explicitly require query_t without units; do not strip or repair model output')
            elif seed['task_type'] == 'equation_numeric':
                diagnosis['independent_process_audit'] = ns['audit_equation_process'](seed, candidate['variant_content'])
            else:
                sample = copy.deepcopy(seed)
                sample.update(answer=proof['answer'], trace={'query_number': proof['query'], 'components': proof['components']})
                diagnosis['independent_process_audit'] = ns['verify_numeral_sample'](sample)
        if 'unverified_reasoning_content' in candidate['rejection_reasons']:
            diagnosis['failure_steps'].append('Screening: free Thinking is not certified by the closed JSON verifier')
            diagnosis['repair_hypotheses'].append('Keep full Trace in Preview; any change to acceptance unit requires an explicit new protocol')
        diagnoses.append(diagnosis)

with (RUN / 'per_output_diagnosis.jsonl').open('x') as f:
    for row in diagnoses:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
total_output = sum(v['output_tokens'] for v in model_usage.values())
report = {'status': 'full_18_output_units_tested_quality_not_fully_passed',
    'all_18_output_units_attempted': True, 'http_responses': len(requests), 'http_failed': result['telemetry']['failed_requests'],
    'original_runtime_status': result['status'], 'original_quality_status': result['quality']['status'],
    'accepted': 9, 'preview': 9, 'metrics': metrics, 'model_usage': model_usage,
    'mean_output_tokens_all_18': total_output / 18, 'output_tokens_all_18': total_output,
    'generation_seconds': result['elapsed_seconds'], 'api_smoke_seconds': read(RUN / 'api_smoke/summary.json')['elapsed_seconds'],
    'free_thinking_certified': False, 'gpu_training_started': False,
    'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
new_json(RUN / 'quality_breakdown.json', report)
lines = ['# NVIDIA Hosted 全量 18 输出单元质量实测', '',
    '**测试已完成；整体数据质量未通过。** 9 Seed × 2 个生成任务均执行，18 次 HTTP 全部返回，无重试。', '',
    'Run：`nvidia_hosted_20261115_ultra_r4_full`；执行源码 `bf024db`。Lightning 生成引导语，Ultra 显式开启 Thinking。', '',
    '| 题型 | Prompt Accepted /3 | Trace Schema /3 | 闭合证明全项通过 /3 | 完整 Trace Accepted | 截断 |',
    '|---|---:|---:|---:|---:|---:|']
for task, m in metrics.items():
    lines.append(f"| {task} | {m['prompt_accepted']} | {m['trace_schema_pass']} | {m['closed_proof_all_checks_pass']} | {m['full_trace_accepted']} | {m['trace_truncated']} |")
lines += ['', '共 9 Accepted（仅 Prompt）、9 Preview（全部 Trace）。9 条 Ultra 响应均含真实 Thinking；JSON 证明检查不能认证任意自由 Thinking，因此完整 Trace 不进入 Accepted。', '',
    'Gravity 前两条达到 16384 输出 Token 上限、没有最终 JSON。第三条全部数值及官方末答通过，但 query_t 返回 `4.62s`，原合同要求 `4.62`，没有自动去单位或放宽规则。', '',
    f"正式生成耗时 {result['elapsed_seconds']:.3f} 秒；仅输出 Token 总计 {total_output}，平均 {total_output / 18:.3f}，包含 Thinking、截断与错误，不含输入。", '',
    '原始 runtime 状态仍为 `failed_runtime`（两次截断触发 SDK 解析错误），quality 为 `completed_with_previews`。Accepted Manifest 的 `passed` 只认证 9 条已接受子集，不表示 18 条或整个实验通过。', '',
    '三集合泄漏审计 18/18 通过；保留原映射暴露政策。每任务仅 3 Seed，结果只作描述，不能证明一般接受率提升。未运行本地 GPU 推理或训练，Accepted/Preview 均未并入 SFT。', '',
    '逐条根因见 `per_output_diagnosis.jsonl`，完整统计见 `quality_breakdown.json`。输入、原始请求/响应/Thinking、Verifier、Manifest 和失败记录均保留。', '']
with (RUN / 'RESULTS.md').open('x') as f:
    f.write('\n'.join(lines))
print(json.dumps(report, ensure_ascii=False, indent=2))
