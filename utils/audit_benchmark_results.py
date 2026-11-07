"""Audit persisted real 0.6B results and adapter tensors without the base model."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from benchmark_training import ROOT, TASKS, TASKSET, verifier


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_path(value):
    path = Path(value)
    if path.is_absolute():
        path = path.relative_to('/root/autodl-tmp/Qwen3-experiment')
    return ROOT / path


def outcome(record, verify):
    if record['truncated']:
        return False
    answer, _ = verify['extract_final_answer'](record['completion'], record['task_type'])
    correct, _ = verify['verify_answer'](record['task_type'], answer, record['expected_answer'])
    return bool(correct)


def adapter_changes(parent_path, child_path):
    import torch
    from safetensors.torch import load_file
    parent = load_file(str(parent_path), device='cpu')
    child = load_file(str(child_path), device='cpu')
    assert parent.keys() == child.keys()
    differences = []
    for name, tensor in child.items():
        assert torch.isfinite(tensor).all()
        differences.append(float((tensor.float() - parent[name].float()).abs().max()))
    assert max(differences) > 0
    return {'method': 'saved_safetensors_compared_on_cpu',
            'parent_weights_sha256': sha(parent_path), 'weights_sha256': sha(child_path),
            'tensor_count': len(differences), 'changed_tensors': sum(d > 0 for d in differences),
            'max_abs_parameter_difference': max(differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="qwen3_06b_baseline_rl_20261107")
    parser.add_argument("--baseline-run-id", default="qwen3-0_6b_round1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/benchmarks/qwen3-0_6b/20261107/summary.json")
    args = parser.parse_args()
    verify = verifier()
    base = TASKSET / "baseline" / args.baseline_run_id
    baseline = read(base / "evaluation_status.json")
    baseline_rows = rows(base / "evaluation_results.jsonl")
    protocol = read(base / "baseline_protocol.json")
    expected = {item for group in protocol["evaluation_ids"].values() for item in group}
    assert baseline["status"] == "completed" and len(baseline_rows) == len(expected) == 150
    assert len({r["id"] for r in baseline_rows}) == 150 and {r["id"] for r in baseline_rows} == expected
    assert sha(base / "evaluation_results.jsonl") == baseline["results_sha256"]
    calibration = rows(base / 'calibration_results.jsonl') + rows(base / 'calibration_extension_results.jsonl')
    calibration_ids = {item for group in protocol['calibration_ids'].values() for item in group}
    assert len(calibration) == 33 and {r['id'] for r in calibration} == calibration_ids
    assert len(calibration_ids) == 9 and not calibration_ids & expected
    with (TASKSET / 'splits/validation.csv').open() as stream:
        validation = {r['id']: r for r in csv.DictReader(stream)}
    trials = collections.defaultdict(list)
    for row in calibration:
        assert row['id'] in protocol['calibration_ids'][row['task_type']]
        trials[(row['task_type'], row['max_new_tokens'])].append(row)
    budgets = read(base / 'final_budget_decision.json')['task_budgets']
    for task, decision in budgets.items():
        for trial in decision['trials']:
            actual = trials[(task, trial['max_new_tokens'])]
            assert len(actual) == trial['records'] == 3
            assert {r['id'] for r in actual} == set(protocol['calibration_ids'][task])
            assert math.isclose(sum(r['correct'] for r in actual) / 3, trial['accuracy'])
            assert math.isclose(sum(r['truncated'] for r in actual) / 3, trial['truncation_rate'])
        valid = [t['max_new_tokens'] for t in decision['trials'] if t['all_complete_and_valid']]
        assert decision['max_new_tokens'] == (min(valid) if valid else max(t['max_new_tokens'] for t in decision['trials']))
    for row in baseline_rows + calibration:
        assert row["source_split"] == "validation" and row["model_id"] == "Qwen/Qwen3-0.6B"
        original = validation[row['id']]
        assert row['prompt'] == original['prompt'] and row['expected_answer'] == original['answer']
        assert len(row["completion_token_ids"]) == row["generation_tokens"] <= row["max_new_tokens"]
        assert len(row["prompt_token_ids"]) == row["prompt_tokens"]
        assert row["prompt_tokens"] + row["generation_tokens"] <= protocol["max_model_len"]
        answer, _ = verify["extract_final_answer"](row["raw_output"], row["task_type"])
        correct, _ = verify["verify_answer"](row["task_type"], answer, row["expected_answer"])
        assert row["truncated"] == (row["finish_reason"] == "length")
        if row["truncated"]:
            answer, correct = None, False
        assert answer == row["extracted_answer"] and bool(correct) == row["correct"]

    processed = ROOT / "data/processed"
    checkpoints = processed / "checkpoints/qwen3-0_6b" / args.run_id
    sft_manifest = read(checkpoints / "sft_manifest.json")
    sft_dir = processed / "sft/qwen3-0_6b" / args.run_id
    sft_status = read(sft_dir / "sft_training_status.json")
    assert sft_status["status"] == sft_manifest["status"] == "completed"
    assert sft_status["adapter_reload"]["status"] == "passed"
    assert sft_status["records"] == 24 and sum(r["optimizer_steps"] for r in sft_status["history"]) > 0
    sft_audit = read(sft_dir / 'sft_data_audit.json')
    assert sft_audit['by_task'] == {task: 8 for task in TASKS}
    assert sft_manifest['training']['data_version'] == sha(TASKSET / 'splits/train.csv')
    assert sft_manifest['training']['real_trace_sha256'] == sha(TASKSET / 'real_train_traces.jsonl')
    assert not sft_manifest['training']['use_synthetic_data']
    adapter_dir = ROOT / sft_manifest["adapter_path"]
    adapter_hashes = {p.name: sha(p) for p in sorted(adapter_dir.glob("adapter_*")) if p.is_file()}
    with (TASKSET / "splits/train.csv").open() as stream:
        train = {r["id"]: r for r in csv.DictReader(stream)}
    traces = rows(TASKSET / 'real_train_traces.jsonl')
    selected_traces = [r for task in TASKS for r in
                       sorted((r for r in traces if r['task_type'] == task), key=lambda r: r['source_id'])[:8]]
    assert [r['source_id'] for r in selected_traces] == sft_audit['source_ids']
    for trace in selected_traces:
        original = train[trace['source_id']]
        assert trace['source_split'] == original['split'] == 'train'
        assert trace['prompt'] == original['prompt'] and trace['original_answer'] == original['answer']
        assert trace['independent_verification_passed'] and trace['verification_status'] == 'accepted'
        assert verify['verify_answer'](trace['task_type'], str(trace['answer']), original['answer'])[0]
    stages = {"sft_training": sft_status}
    rollouts = {}
    weight_updates = {}
    selected_prompts = None
    source_dir = args.output.parent / 'sources'
    source_index = {sha(p): str(p.relative_to(ROOT)) for p in source_dir.iterdir() if p.is_file()}
    provenance = {}
    def audit_protocol(path):
        saved = read(path)
        assert saved['train_sha256'] == sha(TASKSET / 'splits/train.csv')
        provenance[saved['stage']] = {
            'runner': source_index[saved['runner_sha256']],
            'notebook': source_index[saved['notebook_sha256']],
            'protocol_sha256': sha(path),
        }
    audit_protocol(sft_dir / 'sft_protocol.json')
    for method, n in (("dpo", 4), ("ppo", 1), ("grpo", 8)):
        directory = processed / method / "qwen3-0_6b" / args.run_id
        manifest = read(directory / f"{method}_rollout_manifest.json")
        records = rows(directory / f"{method}_smoke_rollouts.jsonl")
        assert manifest["status"] == "completed" and len(records) == manifest["records"] == 6 * n
        assert sha(directory / f"{method}_smoke_rollouts.jsonl") == manifest["rollout_sha256"]
        fingerprint = manifest["sft_fingerprint"]
        assert fingerprint["manifest_sha256"] == sha(checkpoints / "sft_manifest.json")
        assert fingerprint["adapter_files"] == adapter_hashes
        groups, counts = collections.Counter(), collections.Counter()
        indexed, scored = {}, collections.defaultdict(collections.Counter)
        for row in records:
            original = train[row["sample_id"]]
            assert original["split"] == "train" and row["task_type"] == original["task_type"]
            assert row["prompt"] == original["prompt"] and row["expected_answer"] == original["answer"]
            assert not row["fixture"] and row["source"] == "vllm_rollout"
            assert row["policy_version"] == sft_manifest["policy_version"]
            assert row['sft_fingerprint'] == fingerprint and row['source_split'] == 'train'
            assert (row['temperature'], row['top_p'], row['top_k']) == (1.0, 1.0, -1)
            key = (row['sample_id'], row['group_index'])
            assert key not in indexed
            indexed[key] = row
            prompt, completion = row["prompt_token_ids"], row["completion_token_ids"]
            assert row["response_mask"] == [0] * len(prompt) + [1] * len(completion)
            assert 0 < len(completion) <= manifest["max_new_tokens"]
            assert len(prompt) + len(completion) <= manifest["max_model_len"]
            assert len(row["behavior_logprobs"]) == len(completion)
            assert all(math.isfinite(x) and x <= 1e-6 for x in row["behavior_logprobs"])
            assert row["truncated"] == (row["finish_reason"] == "length")
            groups[(row["sample_id"], row["task_type"])] += 1
            counts[row["task_type"]] += 1
            scored[row['task_type']]['correct'] += int(outcome(row, verify))
            scored[row['task_type']]['truncated'] += int(row['truncated'])
            scored[row['task_type']]['total'] += 1
        assert set(groups.values()) == {n} and counts == {task: 2 * n for task in TASKS}
        if selected_prompts is None:
            selected_prompts = set(groups)
        else:
            assert set(groups) == selected_prompts
        assert all({r['group_index'] for r in records if r['sample_id'] == key[0]} == set(range(n)) for key in groups)
        verified = rows(directory / f'{method}_verified_rollouts.jsonl')
        assert len(verified) == len(records)
        verified_keys = set()
        for row in verified:
            key = (row['sample_id'], row['group_index'])
            assert key not in verified_keys
            verified_keys.add(key)
            original = indexed[key]
            assert all(row[k] == original[k] for k in original)
            assert row['correct'] == outcome(row, verify)
            if method == 'grpo':
                assert row['reward'] == float(row['correct'])
        for stage in ("rollout", "training"):
            stages[f"{method}_{stage}"] = read(directory / f"{method}_{stage}_status.json")
            audit_protocol(directory / f'{method}_{stage}_protocol.json')
        training = stages[f"{method}_training"]
        if training["status"] == "completed":
            assert training["optimizer_steps"] > 0 and training["adapter_reload"]["status"] == "passed"
            adapter = result_path(training['adapter_path'])
            assert (adapter / 'adapter_config.json').is_file() and (adapter / 'adapter_model.safetensors').is_file()
            assert sha(adapter / 'adapter_model.safetensors') != adapter_hashes['adapter_model.safetensors']
            weight_updates[method] = adapter_changes(adapter_dir / 'adapter_model.safetensors',
                                                      adapter / 'adapter_model.safetensors')
        if method == 'dpo':
            pairs = rows(directory / 'dpo_smoke_preferences.jsonl')
            assert len(pairs) == training['pairs']
            for pair in pairs:
                group = [r for r in records if r['sample_id'] == pair['sample_id']]
                for label, correct in [('chosen', True), ('rejected', False)]:
                    candidates = [r for r in group if r['completion'] == pair[label]
                                  and r['completion_token_ids'] == pair[f'{label}_token_ids']]
                    assert candidates and any(not r['truncated'] and outcome(r, verify) == correct for r in candidates)
        if method == 'ppo' and training['status'] == 'completed':
            assert training['value_head_reload'] == 'passed' and result_path(training['value_head_path']).is_file()
            assert min(training['max_abs_parameter_update'].values()) > 0
            assert training['usable_records'] == sum(r['training_eligible'] for r in verified)
        if method == 'grpo':
            cases = read(directory / 'cpu_grpo_smoke_report.json')['token_logprob_equivalence']
            assert len(cases) == 12
            assert max(r['max_abs_value_difference'] for r in cases) < 1e-5
            assert max(r['max_abs_gradient_difference'] for r in cases) < 1e-5
            categories = collections.Counter()
            for sample_id, task in groups:
                correct = sum(r['correct'] for r in verified if r['sample_id'] == sample_id)
                categories['all_wrong' if correct == 0 else 'all_correct' if correct == n else 'mixed'] += 1
            readiness = read(directory / 'real_readiness_audit.json')
            assert readiness['groups'] == 6 and readiness['categories'] == dict(categories)
            assert math.isclose(readiness['mixed_ratio'], categories['mixed'] / 6)
            if training['status'] == 'completed':
                assert categories['mixed'] > 0 and training['readiness_audit'] == readiness
        performance = read(directory / 'generation_performance.json')
        assert performance['output_tokens'] == sum(len(r['completion_token_ids']) for r in records)
        assert math.isclose(performance['output_tokens_per_second'],
                            performance['output_tokens'] / performance['generation_seconds'])
        if method == 'grpo' and training['status'] == 'completed':
            logs = training['logs']
            assert logs['response_tokens'] == performance['output_tokens']
            assert logs['micro_batches'] == len(records) and logs['micro_batch_size'] == 1
            assert training['optimizer_steps'] == math.ceil(len(records) / logs['gradient_accumulation_steps'])
        rollouts[method] = {"by_task": dict(counts), "sha256": manifest["rollout_sha256"],
                            "selected_prompt_ids": sorted(key[0] for key in groups),
                            "outcomes_by_task": dict(scored), "performance": performance}
    payload = {"audit_status": "passed", "run_id": args.run_id, "baseline_run_id": args.baseline_run_id,
               "baseline": baseline, "task_budgets": budgets,
               "stages": stages, "rollouts": rollouts,
               "saved_adapter_parameter_updates": weight_updates,
               "source_snapshots": provenance,
               "calibration_results_sha256": sha(base / 'calibration_results.jsonl'),
               "calibration_extension_results_sha256": sha(base / 'calibration_extension_results.jsonl'),
               "all_training_completed": all(stages[f"{name}_training"]["status"] == "completed" for name in ("sft", "dpo", "ppo", "grpo")),
               "parent_sft_manifest_sha256": sha(checkpoints / "sft_manifest.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"audit_status": "passed", "all_training_completed": payload["all_training_completed"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
