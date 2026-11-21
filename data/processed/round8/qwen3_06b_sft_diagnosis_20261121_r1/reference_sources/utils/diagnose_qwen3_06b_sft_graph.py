"""Isolated AutoDL checkpoint diagnostic using notebook training and frozen Tune data."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.benchmark_training import verifier
from utils.run_qwen3_4b_joint_sft import apis, rows, sha, write, write_rows

HISTORY = ROOT / "data/processed/round8/qwen3_4b_joint_sft_20261121_r2"
OUT = HISTORY.with_name("qwen3_06b_sft_diagnosis_20261121_r1")
WEIGHTS = ROOT / ".autodl" / OUT.name
MODELS = ROOT.parent / "huggingface/models"
BASE = MODELS / "Qwen3-0.6B"
BASE4 = MODELS / "Qwen3-4B-Thinking-2507"
R1 = ROOT.parent / "round2_setup/sft_r1_adapter"
TASKS = ("numeral_system", "gravity", "equation_numeric")
CAPS = dict(zip(TASKS, (4096, 8192, 16384)))
STEPS = (4, 12, 24, 48, 96)


def read(path):
    return json.loads(path.read_text())


def notebook_apis():
    math_api, sft = apis()
    official = verifier()
    import numpy as np
    math_api.update(np=np, re=re, gravity_nl_official_extract=official["extract_final_answer"],
                    gravity_nl_official_verify=official["verify_answer"])
    functions = [
        ("01_Synthetic_Data_Engineering.ipynb", math_api, {
            "gravity_c2_solver_reference", "gravity_nl_public_reference", "gravity_nl_final_section",
            "_gravity_nl_eval", "_gravity_nl_normalize", "verify_gravity_natural_language",
            "gravity_c3_split_channels", "gravity_c3_numeric_certificates",
            "verify_gravity_dual_channel", "_gravity_verifier_fit", "_gravity_prompt_parts", "solve_gravity_prompt"
        }),
        ("03_SFT.ipynb", sft, {"render_gravity_c2_final_proof"}),
    ]
    for notebook, namespace, names in functions:
        loaded = set()
        for cell in read(ROOT / notebook)["cells"]:
            if cell["cell_type"] != "code":
                continue
            nodes = [node for node in ast.parse("".join(cell["source"])).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
            if nodes:
                exec(compile(ast.Module(body=nodes, type_ignores=[]), notebook, "exec"), namespace)
                loaded.update(node.name for node in nodes)
        if loaded != names:
            raise ValueError(f"Missing notebook functions: {names - loaded}")
    return math_api, sft, official


def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def render(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=True)


def score(row, body, finish, math_api, official):
    task = row["task_type"]
    final = body.rsplit("</think>", 1)[-1]
    # Historical official evaluator reads the completion as a whole. Also
    # record final-only accuracy to quantify any answer extraction artifact.
    def grade(text):
        answer, method = official["extract_final_answer"](text, task)
        return bool(answer is not None and official["verify_answer"](task, answer, row["answer"])[0]), answer, method
    correct, predicted, method = grade(body)
    final_correct, _, _ = grade(final)
    correct = correct and finish == "stop"
    final_correct = final_correct and finish == "stop"
    result = {"id": row["id"], "task_type": task, "correct": correct, "final_only_correct": final_correct,
              "expected": row["answer"], "predicted": predicted, "extraction_method": method,
              "truncated": finish == "length", "finish_reason": finish,
              "teacher_opening": body.removeprefix("<think>").lstrip().startswith("1. Parse")}
    if task == "gravity":
        proof = math_api["verify_gravity_dual_channel"](row["prompt"], body, row["answer"], finish)
        result.update(proof=proof["verified"], reasons=proof["reasons"])
    return result


def metrics(values):
    per_task = {}
    for task in TASKS:
        group = [r for r in values if r["task_type"] == task]
        correct = [r for r in group if r["correct"]]
        per_task[task] = {
            "n": len(group), "correct": len(correct), "accuracy": len(correct) / len(group),
            "final_only_correct": sum(r["final_only_correct"] for r in group),
            "teacher_opening": sum(r["teacher_opening"] for r in group),
            "truncated": sum(r["truncated"] for r in group),
            "strict_proof": sum(r.get("proof", False) for r in group) if task == "gravity" else None,
            "mean_tokens": statistics.mean(r["tokens"] for r in group),
            "correct_mean_tokens": statistics.mean(r["tokens"] for r in correct) if correct else None,
        }
    per_task["macro"] = statistics.mean(per_task[t]["accuracy"] for t in TASKS)
    return per_task


def prepare():
    frozen = read(HISTORY / "registration.json")
    hashes = {
        HISTORY / "joint_teachers.jsonl": frozen["teachers_sha256"],
        HISTORY / "tune_panel.jsonl": frozen["panel_sha256"],
        BASE / "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        R1 / "adapter_model.safetensors": "d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371",
    }
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input mismatch: {path}")
    math_api, sft, official = notebook_apis()
    teachers = rows(HISTORY / "joint_teachers.jsonl")
    panel = rows(HISTORY / "tune_panel.jsonl")
    if Counter(r["task_type"] for r in teachers) != dict.fromkeys(TASKS, 256):
        raise ValueError("Unexpected teacher counts")
    if Counter(r["task_type"] for r in panel) != dict.fromkeys(TASKS, 50):
        raise ValueError("Unexpected Tune counts")
    if {r["source_id"] for r in teachers} & {r["id"] for r in panel}:
        raise ValueError("Train and Tune IDs overlap")
    tok06, tok4 = tokenizer(BASE), tokenizer(BASE4)
    rich_teachers, audits = [], []
    for row in teachers:
        rich = dict(row)
        if row["task_type"] == "gravity":
            reference = math_api["gravity_c2_solver_reference"](row["prompt"])
            if not official["verify_answer"]("gravity", reference["answer"], row["final_answer"])[0]:
                raise ValueError(f"Teacher and independent Solver disagree: {row['source_id']}")
            rich["completion"] = sft["render_gravity_c2_final_proof"]({**reference, "answer": row["final_answer"]})
            rich["reasoning"] = rich["completion"].rsplit("\nFinal answer:", 1)[0]
        rich_teachers.append(rich)
        for arm, tok, teacher in (("4b_original", tok4, row), ("06_compact", tok06, row),
                                  ("06_full_gravity", tok06, rich)):
            encoded = sft["encode_chat_example"](tok, teacher, 4096)
            boundary = encoded["prompt_length"]
            target = encoded["input_ids"][boundary:]
            if encoded["labels"] != [-100] * boundary + target:
                raise ValueError("Broken supervision mask")
            response = tok.decode(target, skip_special_tokens=False)
            thinking, _ = response.split("</think>")
            if thinking.removeprefix("<think>").strip() != teacher["reasoning"].strip():
                raise ValueError(f"Wrong serialized Thinking: {arm} {teacher['source_id']}")
            check = score({**teacher, "id": teacher["source_id"], "answer": teacher["final_answer"]},
                          response, "stop", math_api, official)
            if not check["final_only_correct"] or (arm == "06_full_gravity" and
                    row["task_type"] == "gravity" and not check["proof"]):
                raise ValueError(f"Invalid teacher: {arm} {check}")
            audits.append({**check, "arm": arm, "supervised_tokens": len(target)})
    refs = {r["id"]: r for r in panel}
    archived_metrics = {}
    for arm in ("base", "r8", "r32"):
        data = rows(HISTORY / f"evaluation/{arm}/rows.jsonl")
        if len(data) != 150 or {r["id"] for r in data} != refs.keys():
            raise ValueError("Historical rows differ from the frozen panel")
        audited = []
        for record in data:
            ref = refs[record["id"]]
            ids = tok4.encode(render(tok4, ref["prompt"]), add_special_tokens=False)
            if record["prompt_token_ids"] != ids or record["max_tokens"] != CAPS[ref["task_type"]]:
                raise ValueError(f"Archived prompt/cap mismatch: {arm} {ref['id']}")
            check = score(ref, record["completion"], record["finish_reason"], math_api, official)
            check.update(tokens=len(record["completion_token_ids"]), historical_correct=record["correct"])
            audited.append(check)
        archived_metrics[arm] = metrics(audited)
        write_rows(OUT / f"archived_4b_{arm}.jsonl", audited)
    seed_key = lambda r: hashlib.sha256(f"early-checkpoint-v1:{r['id']}".encode()).hexdigest()
    dev = [r for task in TASKS for r in sorted((r for r in panel if r["task_type"] == task), key=seed_key)[:10]]
    write_rows(OUT / "screen30.jsonl", dev)
    write_rows(OUT / "confirm120.jsonl", [r for r in panel if r["id"] not in {d["id"] for d in dev}])
    write_rows(OUT / "full_gravity_teachers.jsonl", rich_teachers)
    write_rows(OUT / "teacher_token_audit.jsonl", audits)
    write(OUT / "archived_4b_metrics.json", archived_metrics)
    write(OUT / "registration.json", {
        "input_hashes": {str(path): digest for path, digest in hashes.items()},
        "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in (
            Path(__file__), ROOT / "01_Synthetic_Data_Engineering.ipynb", ROOT / "03_SFT.ipynb",
            ROOT / "utils/low_memory_logprobs.py")},
        "rich_teachers_sha256": sha(OUT / "full_gravity_teachers.jsonl"),
        "checkpoints": STEPS, "greedy_thinking_caps": CAPS,
        "selection": "screen30 macro accuracy, strict Gravity proofs, fewer truncations, earlier step",
        "original_tune_only": True, "holdout_test_used": False,
        "base_revision": read(BASE / "download_manifest.json")["revision"],
        "reference_4b_revision": frozen["config"]["base_revision"],
    })
    print("Frozen CPU audits completed", flush=True)


def train(arm):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from utils.low_memory_logprobs import causal_policy_logprobs

    cfg = read(HISTORY / "registration.json")["config"]
    settings = {**cfg["training"], "seed": cfg["seed"]}
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.set_num_threads(4)
    limit = 8_000_000_000
    torch.cuda.set_per_process_memory_fraction(limit / torch.cuda.get_device_properties(0).total_memory)
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, dtype=torch.bfloat16,
                                                device_map={"": "cpu"}, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                           target_modules=settings["target_modules"], task_type="CAUSAL_LM")).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    _, sft, _ = notebook_apis()
    tok = tokenizer(BASE)
    teachers = rows(HISTORY / "joint_teachers.jsonl" if arm == "compact" else OUT / "full_gravity_teachers.jsonl")
    out = OUT / arm
    out.mkdir()
    weight_dir = WEIGHTS / arm
    weight_dir.mkdir(parents=True)

    model.eval()
    checks = []
    with torch.no_grad():
        for task in TASKS:
            row = next(r for r in teachers if r["task_type"] == task)
            encoded = sft["encode_chat_example"](tok, row, settings["max_length"])
            ids = torch.tensor([encoded["input_ids"]], device="cuda")
            labels = torch.tensor([encoded["labels"]], device="cuda")
            batch = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
            loss, count = sft["course_sft_loss_sum"](model, batch, settings["projection_chunk_size"])
            dense = model(**batch).loss
            torch.testing.assert_close(loss / count, dense, atol=5e-4, rtol=5e-4)
            checks.append({"task_type": task, "n": count, "chunked": float(loss / count), "dense": float(dense)})
    write(out / "loss_check.json", checks)
    torch.cuda.reset_peak_memory_stats()
    checkpoints = []

    def on_step(event):
        with (out / "curve.jsonl").open("a") as log:
            log.write(json.dumps(event) + "\n")
        step = event["optimizer_step"]
        if not math.isfinite(event["gradient_norm"]) or event["torch_peak_reserved_bytes"] >= limit:
            raise ValueError("Gradient or memory guard failed")
        if step not in STEPS:
            return
        path = weight_dir / f"step{step}"
        model.save_pretrained(path)
        model.eval()
        ids = torch.tensor([[2, 3, 4, 5]], device="cuda")
        with torch.no_grad():
            probe = causal_policy_logprobs(model, ids, torch.ones_like(ids), 64).tolist()
        model.train()
        checkpoints.append({"step": step, "sha256": sha(path / "adapter_model.safetensors"),
                            "path": str(path), "probe": probe})
        print(f"{arm} step {step} checkpoint saved", flush=True)

    result = sft["train_augmentation_branch"](model, tok, teachers, settings, on_step=on_step)
    if result["optimizer_steps"] != 96 or [r["step"] for r in checkpoints] != list(STEPS):
        raise ValueError("Wrong training or checkpoint counts")
    result.pop("optimizer_state")
    result.pop("history")
    write(out / "training.json", {**result, "checkpoints": checkpoints, "rank": 8,
                                  "alpha": 16, "learning_rate": settings["learning_rate"],
                                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()})


def evaluate():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from utils.round2_evaluation import mcnemar_exact, paired_mean_analysis

    math_api, _, official = notebook_apis()
    tok = tokenizer(BASE)
    engine = LLM(model=str(BASE), tokenizer=str(BASE), dtype="bfloat16", enable_lora=True, max_lora_rank=8,
                 max_model_len=20480, max_num_seqs=16, max_num_batched_tokens=2048,
                 kv_cache_memory_bytes=16 * 2**30, enforce_eager=False, gpu_memory_utilization=0.75,
                 generation_config="vllm", enable_prefix_caching=False,
                 compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16]})
    sources = {"base06": None, "historical_r1": R1}
    for arm in ("compact", "full_gravity"):
        for value in read(OUT / arm / "training.json")["checkpoints"]:
            path = Path(value["path"])
            if sha(path / "adapter_model.safetensors") != value["sha256"]:
                raise ValueError("Adapter fingerprint mismatch")
            sources[f"{arm}_{value['step']:03d}"] = path
    out = OUT / "evaluation"
    out.mkdir()

    def infer(name, panel, stage):
        path = sources[name]
        adapter = None if path is None else LoRARequest(name, list(sources).index(name) + 1, str(path))
        result = []
        with (out / f"{stage}_{name}.jsonl").open("x") as evidence:
            for task in TASKS:
                group = [r for r in panel if r["task_type"] == task]
                for start in range(0, len(group), 16):
                    batch = group[start:start+16]
                    prompts = [render(tok, r["prompt"]) for r in batch]
                    params = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=CAPS[task],
                                            skip_special_tokens=False)
                    outputs = engine.generate(prompts, params, lora_request=adapter, use_tqdm=False)
                    for row, prompt, output in zip(batch, prompts, outputs, strict=True):
                        if list(output.prompt_token_ids) != tok.encode(prompt, add_special_tokens=False):
                            raise ValueError("vLLM changed input Token IDs")
                        sample = output.outputs[0]
                        scored = score(row, sample.text, sample.finish_reason, math_api, official)
                        scored.update(tokens=len(sample.token_ids), completion=sample.text,
                                      completion_token_ids=list(sample.token_ids),
                                      prompt_token_ids=list(output.prompt_token_ids))
                        result.append(scored)
                        evidence.write(json.dumps(scored, ensure_ascii=False) + "\n")
                    evidence.flush()
        stats = metrics(result)
        write(out / f"{stage}_{name}.json", stats)
        print(f"{stage} {name} macro={stats['macro']:.3f}", flush=True)
        return result

    screen = rows(OUT / "screen30.jsonl")
    confirm = rows(OUT / "confirm120.jsonl")
    screen_results = {name: metrics(infer(name, screen, "screen")) for name in sources}

    def priority(name):
        value = screen_results[name]
        return (value["macro"], value["gravity"]["strict_proof"],
                -sum(value[t]["truncated"] for t in TASKS), -int(name.rsplit("_", 1)[1]))

    chosen = [max((name for name in sources if name.startswith(arm+"_")), key=priority)
              for arm in ("compact", "full_gravity")]
    write(OUT / "checkpoint_selection.json", {"screen": screen_results, "chosen": chosen,
          "primary_candidate": max(chosen, key=priority)})
    archived = {r["id"]: r for r in rows(OUT / "archived_4b_base.jsonl")}
    reference = [archived[r["id"]] for r in confirm]
    result = {"4b_reference": metrics(reference), "screen": screen_results, "chosen": chosen, "confirm": {}}
    for name in ("base06", "historical_r1", *chosen):
        data = infer(name, confirm, "confirm")
        actual = {r["id"]: r for r in data}
        paired = {}
        for task in TASKS:
            group = [r for r in reference if r["task_type"] == task]
            gains = sum(actual[r["id"]]["correct"] and not r["correct"] for r in group)
            losses = sum(r["correct"] and not actual[r["id"]]["correct"] for r in group)
            delta = [int(actual[r["id"]]["correct"]) - int(r["correct"]) for r in group]
            paired[task] = {"gain": gains, "loss": losses, "p_exact": mcnemar_exact(gains, losses),
                            "delta": paired_mean_analysis(delta)}
        result["confirm"][name] = {"metrics": metrics(data), "paired_vs_4b": paired}
    write(OUT / "evaluation_result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compact", "full_gravity", "evaluate"))
    options = parser.parse_args()
    if options.stage == "prepare":
        OUT.mkdir(parents=True, exist_ok=True)
        prepare()
    elif options.stage == "evaluate":
        evaluate()
    else:
        train(options.stage)
