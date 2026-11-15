"""Offline final audit; only read actual API outputs and existing math verifiers."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from utils.run_nvidia_v2 import read, rows, file_hash, tree_hashes, load_notebook

LEDGER = Path(__file__).resolve().parent
PREPARED = ROOT / 'data/processed/round3/nvidia_gravity_targeted_v2_20261115_prepare_r1'


def new_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def analyze():
    conditions = {}
    details = []
    outputs_by_seed = {}
    actual_requests_by_seed = {}
    with tempfile.TemporaryDirectory() as directory:
        ns = load_notebook('gravity_live_final_readonly_audit', Path(directory) / 'loader')
        for condition, suffix in [('P16K', 'p16k'), ('P32K', 'p32k')]:
            target = LEDGER / f'nvidia_gravity_v2_20261115_{suffix}_r1_continuation1'
            generation = target / 'generation'
            if not (generation / 'result.json').exists():
                raise RuntimeError('Condition has not finished: ' + condition)
            result = read(generation / 'result.json')
            candidates = rows(generation / 'candidates.jsonl')
            http = rows(generation / 'http_requests.jsonl')
            by_id = {s['seed_id']: s for s in rows(target / 'private_seeds.jsonl')}
            raw_by_id = {r['seed_id']: r for r in rows(generation / 'raw_model_outputs.jsonl')}
            unique_requests = {(r.get('seed_id'), r.get('variant_kind')) for r in http}
            outputs = []
            reasoning = []
            for candidate in candidates:
                seed = by_id[candidate['seed_id']]
                record = raw_by_id.get(seed['seed_id'], {})
                raw = record.get('trace_variant')
                verification = ns['verify_nvidia_trace_v2'](seed, raw)
                wire = ns['nvidia_wire_check_v2'](seed, record, 'trace_variant', http)
                audit = wire['thinking_audit']
                attempts = [r for r in http if r.get('seed_id') == seed['seed_id']]
                if attempts:
                    actual_requests_by_seed.setdefault(seed['seed_id'], {})[condition] = attempts[-1]['request']
                accepted = (verification['verified'] and wire['verified'] and candidate['leakage_status'] == 'passed')
                assert accepted == (candidate['verification_status'] == 'accepted'), candidate['candidate_id']
                if audit.get('output_tokens') is not None: outputs.append(audit['output_tokens'])
                if audit.get('reasoning_tokens') is not None: reasoning.append(audit['reasoning_tokens'])
                item = {'condition': condition, 'seed_id': seed['seed_id'], 'finish_reason': wire['finish_reason'],
                    'http_status': wire['http_status'], 'schema_valid': verification.get('schema_valid', False),
                    'math_verified': verification['verified'], 'field_audit': verification.get('field_audit', {}),
                    'verification_status': candidate['verification_status'], 'rejection_reasons': candidate['rejection_reasons'],
                    'thinking_present': audit['present'], 'thinking_verification_status': 'unverified_audit_only',
                    'output_tokens': audit.get('output_tokens'), 'reasoning_tokens': audit.get('reasoning_tokens'),
                    'latency_seconds_last_attempt': attempts[-1]['latency_seconds'] if attempts else None,
                    'request_attempt_count': len(attempts),
                    'leakage_status': candidate['leakage_status'], 'source_lineage': candidate['leakage_audit']['lineage_status']}
                details.append(item)
                outputs_by_seed.setdefault(seed['seed_id'], {})[condition] = item
            conditions[condition] = {'run_id': target.name, 'original_result': result,
                'all_three_seeds_attempted': len(raw_by_id) == 3, 'actual_http_attempts': len(http),
                'observed_unique_requests': len(unique_requests), 'retries': len(http) - len(unique_requests),
                'schema_pass': sum(d['condition'] == condition and d['schema_valid'] for d in details),
                'math_pass': sum(d['condition'] == condition and d['math_verified'] for d in details),
                'accepted': sum(d['condition'] == condition and d['verification_status'] == 'accepted' for d in details),
                'preview': sum(d['condition'] == condition and d['verification_status'] == 'preview' for d in details),
                'output_tokens_for_returned_candidates': sum(outputs),
                'mean_output_tokens_returned_candidates': sum(outputs) / len(outputs) if outputs else None,
                'unknown_output_token_count': 3 - len(outputs), 'reasoning_tokens_for_returned_candidates': sum(reasoning),
                'total_reported_output_tokens_all_attempts': sum(r.get('usage', {}).get('completion_tokens', 0) for r in http),
                'http_attempts_without_usage': sum(not r.get('usage') for r in http),
                'finish_reasons': dict(Counter(d['finish_reason'] for d in details if d['condition'] == condition)),
                'prepared_manifest_sha256': file_hash(target / 'prepared_manifest.json'),
                'actual_live_execution': True,
                'historical_offline_reaudit_complete_flag_is_not_execution_mode': True}
    protected = read(LEDGER.parent / 'nvidia_gravity_targeted_v2_20261115_live_r1/protected_trees_before.json')
    for name, expected in protected.items():
        assert tree_hashes(ROOT / 'data/processed/round3' / name) == expected, name
    same_requests = {}
    for seed_id, requests in actual_requests_by_seed.items():
        if set(requests) == {'P16K', 'P32K'}:
            a, b = dict(requests['P16K']), dict(requests['P32K'])
            caps = [a.pop('max_tokens'), b.pop('max_tokens')]
            same_requests[seed_id] = {'same_wire_request_except_token_cap': a == b,
                                     'actual_max_tokens': caps}
            assert a == b and caps == [16384, 32768], seed_id
    complete = all(c['all_three_seeds_attempted'] for c in conditions.values())
    summary = {'status': 'completed_full_two_condition_test' if complete else 'incomplete_provider_error',
        'all_six_seed_conditions_attempted': complete, 'conditions': conditions,
        'sampling_identity_audit': same_requests,
        'per_seed': outputs_by_seed, 'protected_history_unchanged': True,
        'old_prepared_live_execution_flag_is_cpu_preparation_history': True,
        'acceptance_unit': 'raw final JSON proof only', 'thinking': 'retained unverified; excluded from SFT',
        'merge_into_sft': False, 'gpu_training': False,
        'prior_interruption': '../nvidia_gravity_targeted_v2_20261115_live_r1/interruption.json',
        'prior_unobserved_p16_seed2_request_usage_and_status_unknown': True,
        'observed_attempt_count_is_a_lower_bound_for_whole_workflow': True,
        'analysis_script_sha256': file_hash(Path(__file__))}
    new_json(LEDGER / 'summary.json', summary)
    with (LEDGER / 'per_seed_results.jsonl').open('x') as stream:
        for item in details: stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(json.dumps({c: {k: v for k, v in r.items() if k not in ['original_result']} for c, r in conditions.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    analyze()
