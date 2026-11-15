"""Durable maintenance continuation, same frozen conditions and Notebook math.

Preserves the first observed P16 response, reissues only the unobserved pending
request and remaining Seeds. No model-output modification or silent token caps.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from utils.run_nvidia_gravity_targeted import validated_config
from utils.run_nvidia_v2 import file_hash, rows, read, write_json_new, write_jsonl_new, load_notebook
from utils.run_nvidia_synthetic_smoke import (require_api_key, capture_requests, sdk_generate, bind_raw_response, redact)

LEDGER = Path(__file__).resolve().parent
PREPARED = ROOT / 'data/processed/round3/nvidia_gravity_targeted_v2_20261115_prepare_r1'
PRIOR = ROOT / 'data/processed/round3/nvidia_gravity_targeted_v2_20261115_live_r1'


def event(value):
    with (LEDGER / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(redact({'at_unix': time.time(), **value}), ensure_ascii=False) + '\n')


def condition_run(condition, suffix):
    parent = PREPARED / f'nvidia_gravity_v2_20261115_{suffix}_r1'
    manifest = read(parent / 'prepared_manifest.json')
    for name, expected in manifest['frozen_files'].items():
        assert file_hash(parent / name) == expected, name
    for name, expected in manifest['source_hashes'].items():
        assert file_hash(ROOT / name) == expected, name
    c = validated_config(parent / 'config.json')
    target = LEDGER / (parent.name + '_continuation1')
    target.mkdir(exist_ok=False)
    for name in ('config.json', 'private_seeds.jsonl', 'public_seeds.jsonl', 'seed_checks.json', 'compiled_config.json'):
        with (target / name).open('xb') as stream: stream.write((parent / name).read_bytes())
    generation = target / 'generation'
    generation.mkdir()
    original_http = parent / 'generation/http_requests.jsonl'
    imported = rows(original_http) if original_http.exists() else []
    assert all(r.get('http_status') == 200 and r.get('status') == 'completed' for r in imported)
    write_jsonl_new(generation / 'imported_http_requests.jsonl', imported)
    write_json_new(target / 'continuation_manifest.json', {'condition': condition, 'parent_prepared_run': parent.name,
        'parent_prepared_manifest_sha256': file_hash(parent / 'prepared_manifest.json'),
        'source_sha256': file_hash(Path(__file__)), 'configuration_unchanged': True,
        'imported_http_source': str(original_http.relative_to(ROOT)) if imported else None,
        'imported_http_sha256': file_hash(original_http) if imported else None,
        'imported_responses': len(imported), 'prior_unobserved_request_usage_unknown': condition == 'P16K',
        'raw_thinking_verified': False, 'merge_into_sft': False})
    seeds = rows(parent / 'private_seeds.jsonl')
    records, errors = [], []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary:
        ns = load_notebook(target.name, Path(temporary) / 'loader')
        with capture_requests(generation / 'new_http_requests.jsonl'):
            for index, seed in enumerate(seeds):
                prior = [r for r in imported if r['seed_id'] == seed['seed_id']]
                if not prior:
                    seed_path = generation / f'public_seed{index}.jsonl'
                    write_jsonl_new(seed_path, [{k: seed[k] for k in ('seed_id', 'task_type', 'prompt')}])
                    builder = ns['build_nvidia_gravity_targeted_v2'](seed_path, c, condition)
                    write_json_new(generation / f'compiled_seed{index}.json', builder.build().model_dump(mode='json'))
                    event({'event': 'request_start', 'condition': condition, 'seed_id': seed['seed_id'],
                           'max_tokens': c['conditions'][condition]['max_tokens']})
                    try:
                        generated = sdk_generate(builder, generation / f'sdk_seed{index}')
                        write_jsonl_new(generation / f'sdk_outputs{index}.jsonl', generated)
                    except Exception as error:
                        errors.append({'seed_id': seed['seed_id'], 'error': redact(str(error))})
                    event({'event': 'request_end', 'condition': condition, 'seed_id': seed['seed_id']})
                combined = imported + rows(generation / 'new_http_requests.jsonl')
                record = bind_raw_response({k: seed[k] for k in ('seed_id', 'task_type', 'prompt')},
                                           combined, seed, ('trace_variant',))
                records.append(record)
                write_json_new(generation / f'raw_seed{index}.json', record)
                last = [r for r in combined if r.get('seed_id') == seed['seed_id']]
                if any(r.get('http_status') in {400, 401, 403, 404, 410, 422} for r in last):
                    break
        combined = imported + rows(generation / 'new_http_requests.jsonl')
        write_jsonl_new(generation / 'http_requests.jsonl', combined)
        write_jsonl_new(generation / 'raw_model_outputs.jsonl', records)
        quality = ns['screen_nvidia_records_v2'](records, seeds, generation, read(parent / 'seed_checks.json'), combined,
            {'condition': condition, 'original_run_id': Path(c['original_run']).name, 'models_called': len(combined),
             'continuation_of': parent.name, 'imported_responses': len(imported)}, variant_kinds=('trace_variant',))
        thinking = sum(bool(r.get('trace_variant__reasoning_content')) for r in records)
        complete = len(records) == 3 and quality['accepted'] == 3 and thinking == 3
        result = {'run_status': quality['run_status'], 'quality': quality, 'condition': condition,
            'errors': errors, 'engineering_gate_passed': complete, 'thinking_present': thinking,
            'thinking_verification_status': 'unverified_audit_only', 'elapsed_seconds': time.perf_counter() - started,
            'elapsed_seconds_scope': 'continuation_wall_time; imported request has separate original latency',
            'actual_http_requests': len(combined), 'new_http_attempts': len(combined) - len(imported),
            'imported_responses': len(imported), 'prior_unobserved_request_usage_unknown': condition == 'P16K',
            'failed_requests': sum(r.get('http_status') != 200 for r in combined),
            'output_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in combined),
            'reasoning_tokens': sum(r.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', 0) for r in combined),
            'all_three_seeds_attempted': len(records) == 3, 'merge_into_sft': False}
        write_json_new(generation / 'result.json', result)
        event({'event': 'condition_complete', 'condition': condition, 'accepted': quality['accepted'],
               'gate_passed': complete})


def main():
    os.environ['NEMO_TELEMETRY_ENABLED'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    require_api_key()
    write_json_new(LEDGER / 'worker_started.json', {'pid': os.getpid(), 'started_at_unix': time.time(),
        'detached_process': True, 'api_key_source': 'environment_only', 'gpu_training': False,
        'authorization': 'User approved full API tests', 'prior_execution': str(PRIOR.relative_to(ROOT))})
    for condition, suffix in [('P16K', 'p16k'), ('P32K', 'p32k')]:
        try: condition_run(condition, suffix)
        except Exception as error:
            event({'event': 'condition_runtime_failure', 'condition': condition, 'error': redact(str(error))})
    write_json_new(LEDGER / 'worker_finished.json', {'finished_at_unix': time.time()})


if __name__ == '__main__':
    main()
