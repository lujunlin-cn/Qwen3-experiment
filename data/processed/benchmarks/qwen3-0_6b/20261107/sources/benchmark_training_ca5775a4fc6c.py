"""Measure the notebook algorithms on the current three-task Train split.

Each invocation is a fresh process. Notebook GPU entrypoints stay off while CPU
checks run; explicit bindings below connect their algorithms to the new taskset.
No reward, completion or behavior logprob is manufactured for GPU updates.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("numeral_system", "gravity", "equation_numeric")
TASKSET = ROOT / "data/processed/tasksets/roman_gravity_equation_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


TASKSET_INCOMPATIBLE_CPU_CELLS = {
    ("03_SFT.ipynb", 43): "legacy Bit-specific two-stage coverage assertion; current SFT data is re-audited below",
    ("05_PPO.ipynb", 54): "legacy Equation/Bit selection assertion; current Train-only selection is installed below",
}


def select_task_prompts(train, per_task=2):
    selected = []
    for index, task in enumerate(TASKS):
        pool = train[train.task_type == task].sort_values("id")
        for row in pool.sample(n=per_task, random_state=42 + index).itertuples(index=False):
            selected.append({"id": str(row.id), "task_type": task, "prompt": row.prompt,
                             "answer": row.answer, "expected_answer": row.answer, "source_split": "train"})
    return selected


def load_notebook(name, bind_taskset=False):
    module_name = f"benchmark_notebook_{Path(name).stem}"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    namespace = module.__dict__
    taskset_bound = False
    for index, cell in enumerate(json.loads((ROOT / name).read_text())["cells"]):
        if cell["cell_type"] == "code":
            reason = TASKSET_INCOMPATIBLE_CPU_CELLS.get((name, index)) if bind_taskset else None
            if reason:
                print("CPU_CELL_SKIPPED", name, index, reason, flush=True)
                continue
            print("CPU_CELL", name, index, flush=True)
            exec(compile("".join(cell["source"]), f"{name}:cell{index}", "exec"), namespace)
            if bind_taskset and not taskset_bound and ("train_df" in namespace or "TRAIN_PATH" in namespace):
                import pandas as pd
                train_path = TASKSET / "splits/train.csv"
                namespace["TRAIN_PATH"] = namespace["train_path"] = train_path
                namespace["train_df"] = pd.read_csv(train_path, dtype=str, keep_default_na=False)
                if "real_trace_path" in namespace:
                    namespace["real_trace_path"] = TASKSET / "real_train_traces.jsonl"
                taskset_bound = True
            if bind_taskset and taskset_bound:
                def select(per_task=2):
                    return select_task_prompts(namespace["train_df"], per_task)
                for selector in ("select_smoke_prompts", "select_ppo_smoke_prompts", "select_grpo_smoke_prompts"):
                    if selector in namespace:
                        namespace[selector] = select
    if bind_taskset and not taskset_bound:
        raise RuntimeError(f"{name} did not expose a train_df for taskset binding")
    return namespace


def verifier():
    # Pure definitions from the exact baseline, with no dataset writes or GPU call.
    cells = json.loads((ROOT / "00_Baseline_Diagnosis.ipynb").read_text())["cells"]
    ns = {"re": re, "math": math}
    for index in (7, 18):
        exec(compile("".join(cells[index]["source"]), f"baseline_verifier:{index}", "exec"), ns)
    return ns


def prepare():
    import pandas as pd
    ns = load_notebook("01_Synthetic_Data_Engineering.ipynb")
    train = pd.read_csv(TASKSET / "splits/train.csv", dtype=str, keep_default_na=False)
    rows, audit = [], []
    for row in train.sort_values("id").itertuples(index=False):
        sample = {"prompt": row.prompt, "task_type": row.task_type, "source_id": row.id,
                  "source": "official_real_train", "source_split": "train", "original_answer": row.answer,
                  "source_row": row.source_row, "source_dataset": row.source_dataset, "difficulty": "easy"}
        try:
            if row.task_type == "numeral_system":
                solved = ns["solve_numeral_prompt"](row.prompt)
            elif row.task_type == "gravity":
                solved = ns["solve_gravity_prompt"](row.prompt)
                sample["difficulty"] = "examples_5"
            else:
                solved = ns["solve_equation_numeric_prompt"](row.prompt)
                solved["answer"] = solved.get("query_outputs", [None])[0] if solved.get("answer_unique") else None
                solved["trace"] = dict(solved)
                solved["trace"]["candidate_rules_by_operator"] = solved.get("candidates_by_operator", {})
                sample["difficulty"] = "deduce_numeric"
            sample.update({"answer": solved.get("answer"), "trace": solved.get("trace", {})})
            checked = ns["_screen_verification"](sample)
            matches = ns["numeric_or_text_match"](sample["answer"], row.answer)
            accepted = checked.get("answer_ok") and checked.get("trace_ok") and matches
            audit.append({"id": row.id, "task_type": row.task_type, "accepted": bool(accepted),
                          "solver_status": solved.get("status"), "checks": checked, "official_answer_match": matches})
            if accepted:
                sample.update({"verification_status": "accepted", "independent_verification_passed": True,
                               "verifier_version": "synthetic_independent_verifier_v3"})
                rows.append(sample)
        except Exception as error:
            audit.append({"id": row.id, "task_type": row.task_type, "accepted": False, "error": repr(error)})
    target = TASKSET / "real_train_traces.jsonl"
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if target.exists() and target.read_text() != data:
        raise FileExistsError(target)
    target.write_text(data)
    save(target.with_suffix(".manifest.json"), {"status": "passed", "sha256": digest(target),
        "source_train_sha256": digest(TASKSET / "splits/train.csv"), "records": len(rows),
        "by_task": dict(collections.Counter(row["task_type"] for row in rows)), "synthetic_generation_used": False,
        "source_notebook_sha256": digest(ROOT / "01_Synthetic_Data_Engineering.ipynb"), "audit": audit})
    print("PREPARED", len(rows), collections.Counter(row["task_type"] for row in rows), flush=True)


def generate_rollouts(ns, prompts, source, policy_version, adapter, n):
    import gc
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    engine = LLM(model=source, tokenizer=source, dtype="bfloat16", enable_lora=True, max_lora_rank=8,
                 tensor_parallel_size=1, max_model_len=ns["MAX_MODEL_LEN"], max_num_seqs=ns["BENCHMARK_MAX_NUM_SEQS"], enforce_eager=True,
                 gpu_memory_utilization=0.80, kv_cache_memory_bytes=ns["BENCHMARK_KV_CACHE_BYTES"],
                 generation_config="vllm", logprobs_mode="raw_logprobs", seed=42)
    rendered = [tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in prompts]
    if any(len(tokenizer.encode(text, add_special_tokens=False)) + ns["MAX_NEW_TOKENS"] > ns["MAX_MODEL_LEN"] for text in rendered):
        raise ValueError("Rollout exceeds configured context")
    sampling = SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=ns["MAX_NEW_TOKENS"],
                              logprobs=1, seed=42, skip_special_tokens=False)
    started = time.perf_counter()
    outputs = engine.generate(rendered, sampling, lora_request=LoRARequest("sft", 1, adapter))
    elapsed = time.perf_counter() - started
    rows = []
    for original, request in zip(prompts, outputs):
        assert len(request.outputs) == n
        prompt_ids = list(request.prompt_token_ids)
        for index, out in enumerate(request.outputs):
            ids = list(out.token_ids)
            rows.append({"sample_id": str(original["id"]), "prompt_id": str(original["id"]),
                "task_type": original["task_type"], "prompt": original["prompt"], "expected_answer": original["answer"],
                "completion": out.text, "prompt_token_ids": prompt_ids, "completion_token_ids": ids,
                "response_mask": [0] * len(prompt_ids) + [1] * len(ids), "behavior_logprobs": ns["_chosen_logprob_values"](out),
                "finish_reason": out.finish_reason, "stop_reason": out.stop_reason, "truncated": out.finish_reason == "length",
                "policy_version": policy_version, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "group_index": index, "source": "vllm_rollout", "fixture": False, "trace_verified": False})
    assert len(rows) == len(prompts) * n
    save(ns["RESULT_DIR"] / "generation_performance.json", {"status": "completed", "records": len(rows),
        "generation_seconds": elapsed, "output_tokens": sum(len(r["completion_token_ids"]) for r in rows),
        "output_tokens_per_second": sum(len(r["completion_token_ids"]) for r in rows) / elapsed,
        "max_tokens": ns["MAX_NEW_TOKENS"], "max_model_len": ns["MAX_MODEL_LEN"],
        "max_num_seqs": ns["BENCHMARK_MAX_NUM_SEQS"], "kv_cache_bytes": ns["BENCHMARK_KV_CACHE_BYTES"],
        "by_task": dict(collections.Counter(r["task_type"] for r in rows))})
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "sft", "dpo_rollout", "dpo_training", "ppo_rollout", "ppo_training", "grpo_rollout", "grpo_training"))
    parser.add_argument("--run-id", default="perf_20261107")
    parser.add_argument("--mode", choices=("bf16_lora", "nf4_qlora"), default="nf4_qlora")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--kv-cache-gib", type=float, default=8.0)
    args = parser.parse_args()
    if min(args.max_tokens, args.max_model_len, args.max_num_seqs, args.kv_cache_gib) <= 0 or args.max_tokens >= args.max_model_len:
        parser.error("Positive budgets and max_tokens < max_model_len are required")
    os.chdir(ROOT)
    for key in list(os.environ):
        if key.startswith("RUN_"):
            os.environ[key] = "0"
    os.environ.update({"BASELINE_PHASE": "off", "GPU_SMOKE_RUN_ID": args.run_id, "RUN_MODE": "smoke", "MPLBACKEND": "Agg"})
    if args.stage == "prepare":
        prepare()
        return
    import pandas as pd
    name = {"sft": "03_SFT.ipynb", "dpo": "04_DPO.ipynb", "ppo": "05_PPO.ipynb", "grpo": "06_GRPO.ipynb"}[args.stage.split("_")[0]]
    ns = load_notebook(name, bind_taskset=True)
    ns["TRAIN_PATH"] = ns["train_path"] = TASKSET / "splits/train.csv"
    train = pd.read_csv(ns["TRAIN_PATH"], dtype=str, keep_default_na=False)
    assert set(train.task_type) == set(TASKS) and set(train.split) == {"train"}
    ns["train_df"] = train
    ns["MAX_NEW_TOKENS"] = args.max_tokens
    ns["TARGET_TASKS"] = ns["GROUP_TASKS"] = TASKS
    result_dir = ns.get("RESULT_DIR", ns.get("result_dir"))
    stage_name = "sft_training" if args.stage == "sft" else args.stage
    assert not (result_dir / f"{stage_name}_status.json").exists(), "Existing stage result; use a new Run ID"
    protocol = {"tasks": list(TASKS), "train_sha256": digest(ns["TRAIN_PATH"]), "notebook": name,
        "notebook_sha256": digest(ROOT / name), "runner_sha256": digest(__file__), "run_id": args.run_id,
        "stage": args.stage, "mode": args.mode, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
        "kv_cache_bytes": int(args.kv_cache_gib * 2**30), "holdout_used": False}
    save(result_dir / f"{args.stage}_protocol.json", protocol)
    if args.stage == "sft":
        samples = [json.loads(line) for line in (TASKSET / "real_train_traces.jsonl").read_text().splitlines()]
        expected = json.loads((TASKSET / "real_train_traces.manifest.json").read_text())
        assert digest(TASKSET / "real_train_traces.jsonl") == expected["sha256"]
        original_format = ns["_format_trace"]
        def format_trace(sample):
            if sample["task_type"] != "numeral_system":
                return original_format(sample)
            trace = sample["trace"]
            return (f"1. Verify the examples with standard Roman numeral rules.\n"
                    f"2. Decompose {trace['query_number']} into {trace['components']}.\n"
                    f"3. Concatenate the Roman components.\nFinal answer: \\boxed{{{sample['answer']}}}")
        ns["_format_trace"] = format_trace
        records = []
        for task in TASKS:
            group = sorted((s for s in samples if s["task_type"] == task), key=lambda r: r["source_id"])
            for sample in group[:8]:
                original = train[train.id == sample["source_id"]].iloc[0]
                assert original.prompt == sample["prompt"] and original.answer == sample["original_answer"]
                records.append(ns["make_trace_record"](sample))
        assert collections.Counter(r["task_type"] for r in records) == {task: 8 for task in TASKS}
        for record in records:
            audit = ns["audit_trace_record"](record)
            # The notebook's legacy task whitelist includes Bit. All other
            # schema, source, formatting and length checks remain enforced.
            audit["checks"]["task_supported"] = record["task_type"] in TASKS
            assert all(audit["checks"].values()), audit
        save(result_dir / "sft_data_audit.json", {"status": "passed", "records": len(records),
            "by_task": dict(collections.Counter(r["task_type"] for r in records)),
            "source_ids": [r["source_id"] for r in records], "trace_sha256": expected["sha256"],
            "synthetic_generation_used": False})
        ns["full_trace_records"] = records
        ns["real_trace_path"] = TASKSET / "real_train_traces.jsonl"
        ns["run_sft"] = True
        ns["_records_for_strategy"] = lambda strategy: records
        status = ns["_gpu_stage"]("sft_training", lambda: ns["run_sft_training"]("direct", args.mode))
    else:
        v = verifier()
        def check(task, output, expected, truncated=False):
            if truncated:
                return False, "truncated_output"
            answer, _ = v["extract_final_answer"](output, task)
            return v["verify_answer"](task, answer, expected)
        ns["verify_completion"] = ns["_verify_rollout"] = check
        def independent(task, output, expected, truncated=False):
            correct, method = check(task, output, expected, truncated)
            return {"correct": correct, "reward": float(correct), "verification_method": method,
                    "verifier_version": "official_nvidia_metric_v3"}
        ns["independent_answer_verifier"] = independent
        ns["MAX_MODEL_LEN"] = args.max_model_len
        ns["BENCHMARK_MAX_NUM_SEQS"] = args.max_num_seqs
        ns["BENCHMARK_KV_CACHE_BYTES"] = int(args.kv_cache_gib * 2**30)
        def select(per_task=2):
            return select_task_prompts(train, per_task)
        ns["select_smoke_prompts"] = ns["select_ppo_smoke_prompts"] = ns["select_grpo_smoke_prompts"] = select
        ns["_generate_rollout_records"] = lambda prompts, source, version, adapter, n: generate_rollouts(ns, prompts, source, version, adapter, n)
        ns["RUN_" + args.stage.upper()] = True
        status = ns["run_" + args.stage + "_phase"](args.mode) if args.stage.endswith("training") else ns["run_" + args.stage + "_phase"]()
    save(result_dir / f"benchmark_{args.stage}.json", {**protocol, "result": status})
    print("BENCHMARK_RESULT", json.dumps(status, ensure_ascii=False), flush=True)
    if status.get("status") not in ("completed", "blocked_no_preference_pairs", "blocked_no_mixed_groups", "not_run_no_verified_rollout"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
