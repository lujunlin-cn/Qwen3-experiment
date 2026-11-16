"""Read-only teacher/mask audit and one greedy continuation per frozen prompt.

Uses Notebook mathematics and serializers unchanged. No training entrypoint,
optimizer, model save, hosted API or credential lookup is available here.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.dpo_gravity_worker import (BASE, PARENT, RUN as ORIGINAL, LIMIT,
    read, rows, sha, write, write_rows, append, nl_api, audit_nl_rows, mem_report)

RUN = ROOT / 'data/processed/round5/dpo_sft_c_gravity_diagnosis_20261116_r1'
SFT = ROOT / 'data/processed/round4/equation_sft_augmentation_20261116_r2'
R1 = ROOT / 'data/processed/benchmarks/qwen3-0_6b/real_gpu_20261109/recovered_course_r2'
HISTORICAL_NOTEBOOK_COMMIT = '57e409eb27c68717078fb188ddc10b892d0a2cfc'
HISTORICAL_WORKER_COMMIT = '98787e25e16e0a6417c937f5398b4b564fb1ee48'
PYTHON = ROOT / '.venv-low-memory/bin/python'
ADAPTER_HASH = '7033241d41efb086762df20789ce2b70764bb3a5c524e29b8d8f846d3254e3f4'


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def git_file(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def selected_functions(notebook, names, initial=None):
    ns = dict(initial or {})
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        nodes = [node for node in ast.parse(''.join(cell['source'])).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        if nodes:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), f'notebook:cell{i}', 'exec'), ns)
    if not names <= ns.keys():
        raise ValueError('Missing notebook definitions')
    return ns


def sft_apis():
    names = {'_format_trace', 'make_trace_record', 'encode_chat_example', 'course_sft_loss_sum'}
    historical = git_file(HISTORICAL_NOTEBOOK_COMMIT, '03_SFT.ipynb')
    expected = read(R1 / 'protocol.json')['sources']['03_SFT.ipynb']
    if hashlib.sha256(historical).hexdigest() != expected:
        raise ValueError('Historical notebook is not the actual registered R1 source')
    return (selected_functions(json.loads(historical), names, {'max_length': 4096}),
            selected_functions(read(ROOT / '03_SFT.ipynb'), names, {'max_length': 4096}))


def public_solver():
    # Only the existing Gravity parser/Solver definitions, never top-level cells.
    import numpy as np
    return selected_functions(read(ROOT / '01_Synthetic_Data_Engineering.ipynb'),
        {'_gravity_prompt_parts', 'solve_gravity_prompt'}, {'np': np, 're': re})['solve_gravity_prompt']


def identity():
    if sha(PARENT / 'adapter_model.safetensors') != ADAPTER_HASH:
        raise ValueError('SFT C adapter identity mismatch; no fallback')
    checks = read(ORIGINAL / 'identity_audit.json')['base_files']
    if not all(sha(BASE / p) == v['expected'] for p, v in checks.items()):
        raise ValueError('Pinned base/tokenizer changed')
    config = read(PARENT / 'adapter_config.json')
    if (config['r'], config['lora_alpha'], config['lora_dropout']) != (8, 16, 0):
        raise ValueError('LoRA configuration mismatch')
    return {'adapter_sha256': ADAPTER_HASH, 'base_revision': read(ORIGINAL / 'config.json')['base_revision'],
            'base_file_hashes': {p: sha(BASE / p) for p in checks}, 'lora': config}


def protected_hashes():
    result = {}
    for directory in [ORIGINAL, SFT, PARENT, ROOT / '.autodl/round2_20261114/setup/sft_r1_adapter']:
        for p in sorted(directory.rglob('*')):
            if p.is_file():
                result[str(p.relative_to(ROOT))] = sha(p)
    for name in ['01_Synthetic_Data_Engineering.ipynb', '03_SFT.ipynb', '04_DPO.ipynb',
                 '05_PPO.ipynb', '06_GRPO.ipynb', 'utils/course_worker.py',
                 'utils/dpo_gravity_worker.py', 'utils/low_memory_logprobs.py']:
        result[name] = sha(ROOT / name)
    return result


def actual_teachers():
    old, current = sft_apis()
    r1 = rows(R1 / 'panels/sft.jsonl')
    added = rows(SFT / 'sft_real_control_110.jsonl')
    inclusion = [r for r in rows(SFT / 'sft_actual_inclusion_manifest.jsonl') if r['arm'] == 'C']
    actual = {r['source_id']: r for r in inclusion}
    if len(actual) != 110 or len(added) != 110:
        raise ValueError('Actual C inclusion evidence incomplete')
    for r in added:
        x = actual[r['source_id']]
        if not x['finally_entered_isolated_sft'] or x['checkpoint_sha256'] != ADAPTER_HASH or x['epochs_seen'] != 2:
            raise ValueError('C record not linked to actual checkpoint')
    if sha(SFT / 'sft_real_control_110.jsonl') != read(SFT / 'sft_experiment_registration.json')['data_sha256']['sft_real_control_110.jsonl']:
        raise ValueError('C training data differs from registered input')
    expected = read(R1 / 'protocol.json')['frozen_files']['panels/sft.jsonl']
    if sha(R1 / 'panels/sft.jsonl') != expected:
        raise ValueError('R1 actual training panel changed')
    result = []
    for r in r1:
        if r['task_type'] == 'gravity':
            result.append(('historical_r1', old['make_trace_record'](r), r, old))
    for r in added:
        if r['task_type'] == 'gravity':
            result.append(('added_real_c', r, r, current))
    if Counter(x[0] for x in result) != {'historical_r1': 64, 'added_real_c': 10}:
        raise ValueError('Expected exactly 64 historical + 10 actual new real Gravity targets')
    return result, r1, added


def serialized_audit(tok, record, api):
    encoded = api['encode_chat_example'](tok, record, 4096)
    ids, labels, p = encoded['input_ids'], encoded['labels'], encoded['prompt_length']
    # Decode the actual tokenized output, not record['completion'] metadata.
    response = tok.decode(ids[p:], skip_special_tokens=False)
    messages = [{'role': 'user', 'content': record['prompt']},
                {'role': 'assistant', 'content': 'Final answer:' + record['completion'].split('\nFinal answer:', 1)[-1],
                 'reasoning_content': record['reasoning']}]
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
    tokens = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    if tokens.input_ids != ids:
        raise ValueError('Inspected serialization is not the executed notebook encoding')
    offsets = tokens.offset_mapping
    spans = {}
    # Verify all tokens intersecting each necessary step contribute at the causal
    # target position. A Prompt label being masked is expected, not a missing step.
    for name, needle in [('ols_formula', record['reasoning'].splitlines()[1]),
                         ('fitted_g_and_query_time', record['reasoning'].splitlines()[2]),
                         ('raw_query_distance', record['reasoning'].splitlines()[3]),
                         ('boxed_answer', 'Final answer:' + record['completion'].split('\nFinal answer:', 1)[-1])]:
        start = full.index(needle, full.index('<|im_start|>assistant'))
        end = start + len(needle)
        indices = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
        spans[name] = {'char_span': [start, end], 'token_indices': indices, 'tokens': len(indices),
                       'all_supervised': bool(indices) and all(i > 0 and labels[i] == ids[i] for i in indices),
                       'inside_thinking': end <= full.index('</think>', full.index('<|im_start|>assistant'))}
    final = response.split('</think>', 1)[-1].replace('<|im_end|>', '').strip()
    return {'serialized_response': response, 'actual_final_section': final,
            'full_chat_tokens': len(ids), 'prompt_tokens': p,
            'response_supervised_tokens': sum(v != -100 for v in labels[1:]),
            'prompt_fully_masked': all(v == -100 for v in labels[:p]),
            'response_fully_supervised': labels[p:] == ids[p:],
            'no_truncation': len(ids) <= 4096, 'steps': spans,
            'all_required_steps_supervised': all(x['all_supervised'] for x in spans.values()),
            'input_ids_sha256': text_hash(json.dumps(ids)), 'labels_sha256': text_hash(json.dumps(labels))}


def prepare():
    from transformers import AutoTokenizer
    RUN.mkdir(parents=True, exist_ok=False)
    id_audit = identity()
    panel = rows(ORIGINAL / 'train_panel.jsonl')
    sampling = rows(ORIGINAL / 'rollout8/SFT_C_raw.jsonl') + rows(ORIGINAL / 'rollout4/SFT_C_raw.jsonl')
    if len(panel) != 26 or len(sampling) != 312:
        raise ValueError('Frozen panel or sampling responses incomplete')
    hypotheses = [
        {'rank': 1, 'cause': 'SFT channel contract differs from final-proof acceptance',
         'prediction': 'Actual serialized targets place OLS inside think, leaving only boxed answer after close'},
        {'rank': 2, 'cause': 'Sampling temperature amplifies numerical/format errors',
         'prediction': 'Greedy on the identical 26 prompts substantially improves official correctness'},
        {'rank': 3, 'cause': 'Teacher numerical defects or incomplete OLS supervision',
         'prediction': 'Independent public OLS disagrees with teacher g/raw/answer or omitted observations'},
        {'rank': 4, 'cause': 'Loss masking drops mathematical steps',
         'prediction': 'Real encoded OLS/g/raw tokens carry -100 or are lost by causal shifting'},
        {'rank': 5, 'cause': 'Small model has not mastered OLS computation/generalization',
         'prediction': 'Even greedy lacks independently certifiable OLS and gives wrong numerical answers'}]
    write(RUN / 'hypotheses.json', hypotheses)
    reg = {'status': 'registered_before_greedy', 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'identity': id_audit, 'panel_sha256': sha(ORIGINAL / 'train_panel.jsonl'),
           'sampling_files': {str(p.relative_to(ROOT)): sha(p) for p in [ORIGINAL / 'rollout8/SFT_C_raw.jsonl', ORIGINAL / 'rollout4/SFT_C_raw.jsonl']},
           'protected_hashes': protected_hashes(), 'runner_sha256': sha(Path(__file__)),
           'hypotheses_sha256': sha(RUN / 'hypotheses.json'),
           'decode': {'do_sample': False, 'vllm_greedy_temperature_sentinel': 0.0,
                      'temperature_sampling_applicable': False, 'top_p': 1.0, 'top_k': -1,
                      'thinking': True, 'max_tokens': 16384, 'n': 1, 'seed': 20261116},
           'new_requests': 26, 'training_allowed': False, 'hosted_api_allowed': False,
           'ols_metric_definition': 'Final section has verifiable all-public OLS plus fitted g correct at rel1e-4/abs1e-5; parser-limited certification, not keyword count',
           'sampling_aggregation': 'Mean of 12 outcomes per prompt; cluster bootstrap over the 26 prompts, never 312 independent prompts',
           'created_unix': time.time()}
    write(RUN / 'registration.json', reg)
    write_rows(RUN / 'prompt_panel.jsonl', panel)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    teacher_rows, r1, added = actual_teachers()
    old, current = sft_apis()
    solver = public_solver()
    ns = nl_api()
    records = []
    for group, record, source, api in teacher_rows:
        prompt = record['prompt']
        independent = ns['gravity_nl_public_reference'](prompt)
        solved = solver(prompt)
        if solved['status'] != 'solved':
            raise ValueError('Existing solver cannot parse actual teacher')
        target = record['completion']
        verified = ns['verify_gravity_natural_language'](prompt, target, record['final_answer'])
        serialized = serialized_audit(tok, record, api)
        actual_verified = ns['verify_gravity_natural_language'](prompt, serialized['serialized_response'], record['final_answer'], thinking_expected=True)
        trace = source.get('trace', source.get('canonical_solver_trace', {}))
        near = lambda a, b: math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=1e-5)
        g_ok = near(trace['g_ols'], independent['g_ols'])
        raw_ok = near(trace['query_raw'], independent['query_raw'])
        observed = verified.get('observation_claims', [])
        displayed = all(any(near(t, a) and near(d, b) for a, b in observed) for t, d in independent['examples'])
        records.append({'source_id': record['source_id'], 'group': group, 'source_row': source['source_row'],
                        'actual_training_input': str((R1 / 'panels/sft.jsonl' if group == 'historical_r1' else SFT / 'sft_real_control_110.jsonl').relative_to(ROOT)),
                        'prompt': prompt, 'original_answer': source.get('original_answer', record['final_answer']),
                        'teacher_completion_metadata': target, 'reasoning': record['reasoning'],
                        'independent_public_reference': independent, 'existing_solver_result': solved,
                        'public_observation_count': len(independent['examples']),
                        'all_observations_in_original_prompt': True, 'all_observations_explicitly_printed_in_target': displayed,
                        'all_observations_referenced_by_ols': verified['checks'].get('all_public_ols', False),
                        'fitted_g_correct': g_ok, 'unrounded_query_distance_correct': raw_ok,
                        'final_answer_exact_public_ols': record['final_answer'] == independent['answer'],
                        'solver_independent_agreement': near(solved['trace']['g_ols'], independent['g_ols']) and near(solved['trace']['query_raw'], independent['query_raw']) and solved['answer'] == independent['answer'],
                        'metadata_proof_audit': verified, 'actual_serialized_proof_audit': actual_verified,
                        'serialization': serialized,
                        'limitations': ['Target references all observations but does not print them or expand numeric OLS sums',
                                       'Verified metadata Full Trace is not the final channel actually taught']})
    write_rows(RUN / 'teacher_target_audit.jsonl', records)
    # Reproduce exact total supervised counts for ALL actual records, not only
    # the selected 74 Gravity examples. B/C trained C's 110 examples twice.
    r1_count = sum(sum(v != -100 for v in old['encode_chat_example'](tok, old['make_trace_record'](r), 4096)['labels'][1:]) for r in r1)
    c_count = 2 * sum(sum(v != -100 for v in current['encode_chat_example'](tok, r, 4096)['labels'][1:]) for r in added)
    expected_r1 = read(R1 / 'jobs/sft_r1_train_attempt1/result.json')['assistant_training_tokens']
    expected_c = read(SFT / 'sft_C_train/result.json')['supervised_tokens']
    if (r1_count, c_count) != (expected_r1, expected_c):
        raise ValueError('Reconstructed masks do not match actual training token counts')
    hist_worker = git_file(HISTORICAL_WORKER_COMMIT, 'utils/course_worker.py')
    if hashlib.sha256(hist_worker).hexdigest() != read(R1 / 'protocol.json')['sources']['utils/course_worker.py']:
        raise ValueError('Historical worker mismatch')
    write(RUN / 'teacher_audit_summary.json', {
        'groups': {g: {'n': sum(x['group'] == g for x in records), **{k: sum(bool(x[k]) for x in records if x['group'] == g)
            for k in ['all_observations_explicitly_printed_in_target', 'all_observations_referenced_by_ols', 'fitted_g_correct',
                      'unrounded_query_distance_correct', 'final_answer_exact_public_ols', 'solver_independent_agreement']},
            'metadata_proof_passed': sum(x['metadata_proof_audit']['verified'] for x in records if x['group'] == g),
            'actual_final_proof_passed': sum(x['actual_serialized_proof_audit']['verified'] for x in records if x['group'] == g),
            'all_required_steps_supervised': sum(x['serialization']['all_required_steps_supervised'] for x in records if x['group'] == g)}
            for g in ['historical_r1', 'added_real_c']},
        'reconstructed_all_task_supervised_tokens': {'r1_192_once': r1_count, 'c_110_twice': c_count,
            'actual_training_logged_r1': expected_r1, 'actual_training_logged_c': expected_c},
        'historical_notebook_commit': HISTORICAL_NOTEBOOK_COMMIT, 'historical_worker_commit': HISTORICAL_WORKER_COMMIT,
        'source_hash_matched_historical_registration': True, 'weights_modified': False})


def greedy():
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    reg = read(RUN / 'registration.json')
    identity()
    if sha(Path(__file__)) != reg['runner_sha256']:
        raise ValueError('Diagnostic source changed after registration')
    if sha(ORIGINAL / 'train_panel.jsonl') != reg['panel_sha256']:
        raise ValueError('Frozen prompts changed')
    out = RUN / 'greedy'
    out.mkdir(exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(min(.95, LIMIT / torch.cuda.get_device_properties(0).total_memory))
    # Exact backend settings used in the 312 original responses.
    engine = LLM(model=str(BASE), tokenizer=str(BASE), dtype='bfloat16', tensor_parallel_size=1,
        enable_lora=True, max_lora_rank=8, max_loras=1, enable_prefix_caching=False, max_model_len=20480,
        max_num_seqs=8, max_num_batched_tokens=512, kv_cache_memory_bytes=4 * 2**30, gpu_memory_utilization=.10,
        compilation_config={'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [1, 2, 4, 8], 'max_cudagraph_capture_size': 8},
        enforce_eager=False, seed=20261116, generation_config='vllm', logprobs_mode='raw_logprobs')
    tok = engine.get_tokenizer()
    panel = rows(RUN / 'prompt_panel.jsonl')
    sampling = rows(ORIGINAL / 'rollout8/SFT_C_raw.jsonl')
    tokens = {r['sample_id']: r['prompt_token_ids'] for r in sampling}
    request = LoRARequest('gravity_diagnosis_greedy_SFT_C', 1, str(PARENT))
    start = time.perf_counter()
    for index, original in enumerate(panel):
        rendered = tok.apply_chat_template([{'role': 'user', 'content': original['prompt']}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        if tok.encode(rendered, add_special_tokens=False) != tokens[original['id']]:
            raise ValueError('Greedy/Sampling prompt token mismatch')
        params = SamplingParams(n=1, temperature=0., top_p=1., top_k=-1, max_tokens=16384,
                                seed=20261116 + index, skip_special_tokens=False)
        output = engine.generate([rendered], [params], lora_request=request, use_tqdm=False)[0]
        response = output.outputs[0]
        ids = list(response.token_ids)
        if response.finish_reason == 'stop' and ids[-1] not in {151643, 151645}:
            raise ValueError('Stopped response lacks pinned EOS')
        if list(output.prompt_token_ids) != tokens[original['id']]:
            raise ValueError('Engine prompt token mismatch')
        append(out / 'SFT_C_raw.jsonl', {'sample_id': original['id'], 'prompt_id': original['id'],
            'task_type': 'gravity', 'prompt': original['prompt'], 'expected_answer': original['answer'],
            'completion': response.text, 'prompt_token_ids': list(output.prompt_token_ids),
            'completion_token_ids': ids, 'completion_tokens': len(ids), 'finish_reason': response.finish_reason,
            'stop_reason': response.stop_reason, 'truncated': response.finish_reason == 'length',
            'do_sample': False, 'decoding_mode': 'greedy', 'temperature_sampling_applicable': False,
            'vllm_temperature_sentinel': 0., 'top_p': 1., 'top_k': -1, 'thinking_enabled': True,
            'max_tokens': 16384, 'policy_fingerprint': {'adapter_model.safetensors': ADAPTER_HASH,
                'base_revision': reg['identity']['base_revision']}, 'source_split': 'train',
            'provenance': original['provenance'], 'public_model_input_sha256': text_hash(rendered),
            'eos_included': response.finish_reason == 'stop', 'group_index': 0})
        engine.collective_rpc(mem_report)
    write(out / 'result.json', {'status': 'completed', 'records': len(panel),
        'generation_seconds': time.perf_counter() - start, 'memory': engine.collective_rpc(mem_report),
        'training_steps': 0, 'adapter_after_sha256': sha(PARENT / 'adapter_model.safetensors')})


def invoke_greedy():
    c = read(ORIGINAL / 'config.json')
    actual = subprocess.check_output(['docker', 'image', 'inspect', c['backend_image'], '--format', '{{.Id}}'], text=True).strip()
    if actual != c['backend_image_id']:
        raise ValueError('Inference image differs from original sampling backend')
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) * 1024
    if available < c['minimum_host_available_bytes']:
        raise RuntimeError('Insufficient shared memory; other processes left intact')
    write(RUN / 'gpu_inventory_before.json', {'compute_processes': subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True),
        'host_mem_available_bytes': available, 'backend_image_id': actual})
    cmd = ['docker', 'run', '--rm', '--name', 'qwen-gravity-diagnosis-greedy', '--network', 'none', '--gpus', 'all',
           '--ipc', 'host', '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
           '-e', 'VLLM_ENABLE_V1_MULTIPROCESSING=0', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TOKENIZERS_PARALLELISM=false',
           '-v', f'{ROOT}:{ROOT}', '-w', str(ROOT), '--entrypoint', 'python3', c['backend_image'],
           'utils/run_dpo_sft_c_gravity_diagnosis.py', 'greedy']
    with (RUN / 'greedy.log').open('x') as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    write(RUN / 'greedy_completion_hook.json', {'exit_code': proc.returncode, 'time': time.time(),
        'wait_mechanism': 'blocking subprocess completion; no progress polling'})
    if proc.returncode:
        raise RuntimeError('Greedy failed; see preserved log')


def ols_certified(audit):
    reference = audit.get('reference', {})
    values = audit.get('reported_g_values', [])
    bad = {'altered_public_observation_in_proof', 'fitted_g_missing_or_inaccurate', 'incorrect_reported_g_value',
           'incorrect_intermediate_arithmetic', 'incorrect_numeric_equality_in_prose', 'incorrect_symbolic_arithmetic',
           'unsupported_math_syntax', 'unsupported_arithmetic', 'incorrect_or_unverifiable_optional_indexed_g'}
    return bool(audit.get('checks', {}).get('all_public_ols') and values and
        all(math.isclose(v, reference['g_ols'], rel_tol=1e-4, abs_tol=1e-5) for v in values) and not bad.intersection(audit['reasons']))


def same_archived_audit(current, archived):
    # Python parsers return tuples; JSON archives necessarily store lists. Compare
    # the exact JSON-domain values, without numeric tolerances or dropping fields.
    return (current['correct'] == archived['correct'] and
            json.dumps(current['proof_audit'], sort_keys=True, allow_nan=False) ==
            json.dumps(archived['proof_audit'], sort_keys=True, allow_nan=False))


def enriched(records):
    for r in audit_nl_rows(records):
        a = r['proof_audit']
        r['ols_final_certified'] = ols_certified(a)
        # Keyword observations in Thinking are audit signals only, not proofs.
        thinking = r['completion'].split('</think>', 1)[0] if '</think>' in r['completion'] else r['completion']
        r['unverified_thinking_signals'] = {
            'ols_or_least_squares_mention': bool(re.search(r'\bOLS\b|least[ -]squares', thinking, re.I)),
            'sum_mention': bool(re.search(r'\bsum\b|\\sum', thinking)),
            'single_example_language': bool(re.search(r'first\s+(?:example|observation)|one\s+(?:example|observation)|single[ -](?:point|example)', thinking, re.I)),
            'verification_status': 'unverified; signals are not mathematical correctness'}
        yield r


def metrics(rs):
    from utils.round2_evaluation import wilson
    n = len(rs)
    lengths = [len(r['completion_token_ids']) for r in rs]
    correct = sum(r['correct'] for r in rs)
    return {'n': n, 'correct': correct, 'accuracy': correct / n, 'accuracy_wilson95_descriptive': wilson(correct, n),
            'proof_pass': sum(r['proof_audit']['verified'] for r in rs),
            'ols_final_certified': sum(r['ols_final_certified'] for r in rs),
            'format_valid': sum(r['format_valid'] for r in rs), 'truncated': sum(r['truncated'] for r in rs),
            'mean_tokens': statistics.mean(lengths), 'median_tokens': statistics.median(lengths), 'total_tokens': sum(lengths),
            'mean_correct_tokens': statistics.mean(len(r['completion_token_ids']) for r in rs if r['correct']) if correct else None,
            'failure_reasons_nonexclusive': dict(Counter(x for r in rs for x in r['proof_audit']['reasons'])),
            'unverified_thinking_signals': {k: sum(r['unverified_thinking_signals'][k] for r in rs)
                for k in ['ols_or_least_squares_mention', 'sum_mention', 'single_example_language']}}


def analyze():
    from utils.round2_evaluation import paired_mean_analysis
    g = list(enriched(rows(RUN / 'greedy/SFT_C_raw.jsonl')))
    s = list(enriched(rows(ORIGINAL / 'rollout8/SFT_C_raw.jsonl') + rows(ORIGINAL / 'rollout4/SFT_C_raw.jsonl')))
    if len(g) != 26 or len(s) != 312:
        raise ValueError('Incomplete comparison; do not publish as complete')
    before = {(r['sample_id'], r['group_index']): r for r in rows(ORIGINAL / 'final_rollouts_audited_v12.jsonl')}
    if set(before) != {(r['sample_id'], r['group_index']) for r in s}:
        raise ValueError('Archived response identity differs')
    for x in s:
        if not same_archived_audit(x, before[x['sample_id'], x['group_index']]):
            raise ValueError('Original Verifier outcomes changed')
    write_rows(RUN / 'greedy_scored.jsonl', g)
    # Reuse sampling raw files; save only new diagnostic fields, not duplicate IDs/logprobs payloads.
    write_rows(RUN / 'sampling_diagnostic.jsonl', [{k: r[k] for k in ['sample_id', 'group_index', 'correct', 'predicted', 'completion_tokens', 'finish_reason', 'proof_audit', 'ols_final_certified', 'unverified_thinking_signals']} for r in s])
    per_prompt = []
    for greedy_row in g:
        group = [x for x in s if x['sample_id'] == greedy_row['sample_id']]
        if len(group) != 12 or any(x['prompt_token_ids'] != greedy_row['prompt_token_ids'] or x['policy_fingerprint'] != greedy_row['policy_fingerprint'] for x in group):
            raise ValueError('Same-condition identity mismatch')
        gm, sm = metrics([greedy_row]), metrics(group)
        per_prompt.append({'sample_id': greedy_row['sample_id'], 'source': greedy_row['provenance']['source'],
            'prompt': greedy_row['prompt'], 'expected_answer': greedy_row['expected_answer'],
            'greedy': gm, 'sampling': sm, 'greedy_predicted': greedy_row['predicted'],
            'sampling_predictions': [r['predicted'] for r in group],
            'sampling_correct_indices': [r['group_index'] for r in group if r['correct']],
            'accuracy_delta_greedy_minus_sampling_mean': gm['accuracy'] - sm['accuracy'],
            'mean_token_delta_greedy_minus_sampling': gm['mean_tokens'] - sm['mean_tokens'],
            'same_prompt_tokens': True, 'same_policy': True})
    write_rows(RUN / 'per_prompt_comparison.jsonl', per_prompt)
    summary = {'status': 'completed', 'greedy': metrics(g), 'sampling': metrics(s),
        'by_source': {kind: {'greedy': metrics([x for x in g if x['provenance']['source'] == kind]),
                            'sampling': metrics([x for x in s if x['provenance']['source'] == kind])}
                      for kind in ['official_real_train', 'synthetic_train_prior']},
        'paired_prompt_cluster_accuracy_delta': paired_mean_analysis([x['accuracy_delta_greedy_minus_sampling_mean'] for x in per_prompt]),
        'paired_prompt_cluster_mean_token_delta': paired_mean_analysis([x['mean_token_delta_greedy_minus_sampling'] for x in per_prompt]),
        'prompts_with_any_sampling_correct': sum(x['sampling']['correct'] > 0 for x in per_prompt),
        'greedy_correct_sampling_all_wrong': [x['sample_id'] for x in per_prompt if x['greedy']['correct'] and not x['sampling']['correct']],
        'greedy_wrong_sampling_some_correct': [x['sample_id'] for x in per_prompt if not x['greedy']['correct'] and x['sampling']['correct']],
        'greedy_final_answers': [x['predicted'] for x in g],
        'sampling_wilson_limitation': '312 outputs share 26 prompts; Wilson here is descriptive only; inferential comparison resamples 26 prompt clusters',
        'dataset_scope': 'Frozen DPO Train panel, not Validation or Holdout; diagnostic only',
        'teacher_audit': read(RUN / 'teacher_audit_summary.json'),
        'trained_or_saved_weights': False, 'hosted_api_calls': 0}
    write(RUN / 'comparison_summary.json', summary)
    integrity = {p: sha(ROOT / p) == h for p, h in read(RUN / 'registration.json')['protected_hashes'].items()}
    write(RUN / 'integrity_check.json', {'protected_files': len(integrity), 'all_unchanged': all(integrity.values()),
        'mismatches': [p for p, same in integrity.items() if not same], 'adapter_unchanged': sha(PARENT / 'adapter_model.safetensors') == ADAPTER_HASH,
        'new_greedy': len(g), 'sampling_reused_without_regeneration': len(s), 'optimizer_steps': 0})
    if not all(integrity.values()):
        raise ValueError('Protected input/history changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'greedy', 'analyze', 'all'])
    stage = parser.parse_args().stage
    try:
        if stage in ['prepare', 'all']:
            prepare()
        if stage == 'all':
            with (RUN / 'cpu_tests.log').open('x') as f:
                test = subprocess.run([str(PYTHON), '-m', 'unittest', 'tests.test_dpo_sft_c_gravity_diagnosis',
                    'tests.test_dpo_gravity_sft_c', '-v'], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'TOKENIZERS_PARALLELISM': 'false'})
            if test.returncode:
                raise RuntimeError('CPU diagnostic regression failed')
            invoke_greedy()
        if stage == 'greedy':
            greedy()
        if stage in ['analyze', 'all']:
            analyze()
    except BaseException as e:
        if RUN.exists():
            write(RUN / f'failure_{time.time_ns()}.json', {'stage': stage, 'error': repr(e), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
