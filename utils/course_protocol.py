"""CPU-only experiment protocol, panels, metrics and immutable manifests."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKSET = ROOT / 'data/processed/tasksets/roman_gravity_equation_v1'
TASKS = ('numeral_system', 'gravity', 'equation_numeric')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))

def csv_rows(split):
    with (TASKSET / 'splits' / f'{split}.csv').open() as f:
        return list(csv.DictReader(f))

def priors():
    cells = read(ROOT / '02_Prompt_Engineering.ipynb')['cells']
    ns = {}
    exec(''.join(cells[7]['source']), ns)
    return ns['TASK_PRIOR_TEXT']

def prompt_text(prompt, task, variant, prior=None):
    if variant == 'original':
        return prompt
    if variant != 'prior':
        raise ValueError(variant)
    return prompt + '\n\n' + (prior if prior is not None else priors())[task]

def validate_config(c):
    assert c['schema'] == 'qwen3-course-8h/v1'
    assert sum(c['budgets_seconds'].values()) <= c['total_seconds']
    assert 0 < c['device_limit_bytes'] <= 8_000_000_000
    assert c['max_num_seqs'] == 1 and c['max_model_len'] > max(c['baseline_caps'])
    assert c['ppo']['tasks'] == c['grpo']['tasks'] == ['equation_numeric']
    assert set(c['dpo']['tasks']) == {'gravity', 'equation_numeric'}
    assert set(c['sft']['tasks']) == set(TASKS)
    assert c['grpo']['candidates'] == 8
    assert c['training_prompt'] == 'original'
    assert isinstance(c.get('enforce_eager', True), bool)
    if c.get('compilation_config') is not None:
        assert not c.get('enforce_eager', True)
        assert c['compilation_config'] == {
            'level': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
            'cudagraph_capture_sizes': [1], 'max_capture_size': 1,
        }
    for method in ('sft', 'dpo', 'ppo', 'grpo'):
        assert c[method]['gradient_accumulation'] > 0
        assert c[method]['max_rounds'] >= c['stop']['minimum_rounds']
    return c

def prepare(config, destination):
    c = validate_config(read(config))
    destination = Path(destination).resolve()
    if (destination / 'protocol.json').exists():
        check_protocol(destination, config)
        return read(destination / 'protocol.json')
    train, validation = csv_rows('train'), csv_rows('validation')
    assert not ({r['id'] for r in train} & {r['id'] for r in validation})
    tune, final = [], []
    for i, task in enumerate(TASKS):
        pool = sorted((r for r in validation if r['task_type'] == task), key=lambda r:r['id'])
        random.Random(c['seed'] + i).shuffle(pool)
        a, b = c['panels']['tune_per_task'], c['panels']['final_per_task']
        assert len(pool) >= a+b
        tune += pool[:a]; final += pool[a:a+b]
    assert not ({r['id'] for r in tune} & {r['id'] for r in final})
    write_rows(destination / 'panels/tune.jsonl', tune)
    write_rows(destination / 'panels/final.jsonl', final)
    prior = priors()
    write(destination / 'public_priors.json', prior)
    trace_manifest = read(TASKSET/'real_train_traces.manifest.json')
    if (trace_manifest['status']!='passed' or trace_manifest.get('synthetic_generation_used')
        or trace_manifest['sha256']!=digest(TASKSET/'real_train_traces.jsonl')
        or trace_manifest['source_train_sha256']!=digest(TASKSET/'splits/train.csv')):
        raise ValueError('Verified teacher manifest or Train provenance mismatch')
    traces = read_rows(TASKSET / 'real_train_traces.jsonl')
    assert len(traces)==trace_manifest['records']
    train_by_id = {r['id']: r for r in train}
    chosen = []
    for i, task in enumerate(TASKS):
        pool = sorted((r for r in traces if r['task_type'] == task), key=lambda r:r['source_id'])
        random.Random(c['seed'] + 10+i).shuffle(pool)
        chosen += pool[:c['sft']['per_task']]
    for r in chosen:
        original = train_by_id[r['source_id']]
        assert r['source_split'] == 'train' and r['verification_status'] == 'accepted'
        assert original['prompt'] == r['prompt'] and original['answer'] == r['original_answer']
        assert r['independent_verification_passed']
    write_rows(destination / 'panels/sft.jsonl', chosen)
    # Pre-register Train batches for all rounds: no selection on validation answers.
    for method in ('dpo', 'ppo', 'grpo'):
        settings = c[method]
        pools = {}
        for i, task in enumerate(settings['tasks']):
            pool = sorted((r for r in train if r['task_type'] == task), key=lambda r:r['id'])
            random.Random(c['seed'] + 100+i).shuffle(pool)
            pools[task] = pool
        for k in range(1, settings['max_rounds']+1):
            count = settings['prompts_per_task']
            rows = [r for task in settings['tasks'] for r in pools[task][(k-1)*count:k*count]]
            assert len(rows) == count * len(settings['tasks'])
            write_rows(destination / f'panels/{method}_{k}.jsonl', rows)
    frozen = {str(p.relative_to(destination)):digest(p) for p in sorted((destination/'panels').glob('*'))}
    frozen['public_priors.json'] = digest(destination / 'public_priors.json')
    sources = {str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'utils').glob('course_*.py'))}
    for name in ('low_memory_logprobs.py','benchmark_training.py'):
        sources['utils/'+name] = digest(ROOT/'utils'/name)
    sources.update({p.name:digest(p) for p in ROOT.glob('0[0-6]_*.ipynb')})
    protocol = {'schema':c['schema'], 'config':c, 'config_sha256':digest(config),
        'sources':sources, 'frozen_files':frozen, 'split_sha256':{s:digest(TASKSET/f'splits/{s}.csv') for s in ('train','validation')},
        'traces_sha256':digest(TASKSET/'real_train_traces.jsonl'), 'holdout_used':False,
        'final_panel_used_for_selection':False, 'memory_unit':'bytes; 8 GB = 8000000000 bytes',
        'metrics':'All completions including truncated; output tokens include thinking and EOS, never prompt tokens',
        'sft_showcase':'numeral_system', 'ppo_grpo_gravity_role':'evaluation_only'}
    write(destination / 'protocol.json', protocol)
    return protocol

def check_protocol(run, config=None):
    run = Path(run); p = read(run/'protocol.json')
    if config is not None and digest(config) != p['config_sha256']:
        raise ValueError('Config changed: use a new run directory')
    for rel, sha in p['frozen_files'].items():
        if digest(run/rel) != sha: raise ValueError(f'Frozen panel/prior changed: {rel}')
    for rel, sha in p['sources'].items():
        if digest(ROOT/rel) != sha: raise ValueError(f'Experiment source changed: {rel}; use a new run')
    for split, sha in p['split_sha256'].items():
        if digest(TASKSET/f'splits/{split}.csv') != sha: raise ValueError('Dataset changed')
    if digest(TASKSET/'real_train_traces.jsonl') != p['traces_sha256']: raise ValueError('Teacher data changed')
    return p

def metrics(rows):
    """Output length includes every generated token, also on failed/truncated rows."""
    out = {}
    for task in sorted({r['task_type'] for r in rows}):
        group = [r for r in rows if r['task_type']==task]
        n = len(group); good = sum(bool(r['correct']) for r in group)
        tokens = [len(r['completion_token_ids']) for r in group]
        correct_tokens = [len(r['completion_token_ids']) for r in group if r['correct']]
        # Wilson interval communicates how small the screening panels are.
        z=1.96; frac=good/n; denom=1+z*z/n
        center=(frac+z*z/(2*n))/denom
        radius=z*math.sqrt(frac*(1-frac)/n+z*z/(4*n*n))/denom
        out[task] = {'n':n, 'correct':good, 'accuracy':frac,
            'accuracy_wilson95':[center-radius, center+radius],
            'truncated':sum(bool(r['truncated']) for r in group),
            'truncation_rate':sum(bool(r['truncated']) for r in group)/n,
            'mean_output_tokens':statistics.mean(tokens), 'total_output_tokens':sum(tokens),
            'mean_correct_output_tokens':statistics.mean(correct_tokens) if correct_tokens else None,
            'max_output_tokens':max(tokens)}
        for field, metric in (('exact_trace_pass','exact_trace_pass_rate'), ('format_valid','format_valid_rate')):
            available = [r for r in group if field in r]
            out[task][metric] = sum(bool(r[field]) for r in available)/len(available) if available else None
            out[task][metric+'_verified_n'] = len(available)
        rewarded = [r for r in group if 'reward_components' in r]
        out[task]['reward_mean'] = (statistics.mean(r['reward_components']['combined'] for r in rewarded)
                                     if rewarded else None)
    return out

def validation_score(m, tasks, caps):
    # Macro by task, correctness dominates the modest efficiency term per item.
    return statistics.mean(m[t]['accuracy'] - .15*m[t]['truncation_rate']
                           - .05*m[t]['mean_output_tokens']/caps[t] for t in tasks)

def advance_stop(history, score, stop):
    values = history + [score]
    best = 0; high = values[0]
    for i,v in enumerate(values[1:],1):
        if v > high + stop['min_score_gain']: best=i; high=v
    improved = not history or best == len(values)-1
    plateau = len(values) >= stop['minimum_rounds'] and len(values)-1-best >= stop['patience']
    return improved, plateau
