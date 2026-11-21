"""Isolated Qwen3-4B-Thinking joint SFT and LoRA-rank experiment."""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3_4b_joint_sft_20261121_r2.json"
TASKS = ("numeral_system", "gravity", "equation_numeric")
SOURCES = (
    "01_Synthetic_Data_Engineering.ipynb",
    "03_SFT.ipynb",
    "utils/run_qwen3_4b_joint_sft.py",
    "utils/run_nvidia_final_synthetic.py",
    "utils/run_nvidia_v2.py",
    "utils/low_memory_logprobs.py",
    "utils/benchmark_training.py",
    "configs/qwen3_4b_joint_sft_20261121_r2.json",
)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def paths(config):
    run = ROOT / "data/processed/round8" / config["run_id"]
    return run, ROOT / ".autodl" / config["run_id"]


def model_path():
    value = os.environ.get("QWEN3_4B_MODEL_PATH")
    if not value:
        raise ValueError("QWEN3_4B_MODEL_PATH must point to the pinned local snapshot")
    return Path(value).resolve()


def validate_config(config):
    if config["schema"] != "qwen3-4b-joint-sft/v1":
        raise ValueError("Unexpected schema")
    if tuple(config["tasks"]) != TASKS or config["holdout_used"] or config["rl_autostart"]:
        raise ValueError("Task, Holdout or RL policy changed")
    training = config["training"]
    if training["ranks"] != [8, 32] or training["micro_batch_size"] != 1:
        raise ValueError("Unexpected LoRA or microbatch configuration")
    if training["lora_alpha_multiplier"] != 2 or training["mode"] != "bf16_lora":
        raise ValueError("Unexpected LoRA scaling or precision")
    expected_steps = math.ceil(3 * config["teachers_per_task"] / training["gradient_accumulation"])
    if config["teachers_per_task"] != 256 or expected_steps != 96:
        raise ValueError("Registered task/token or optimizer-step budget changed")
    if config["evaluation"]["arms"] != ["base", "r8", "r32"]:
        raise ValueError("Evaluation arms changed")
    roles = config["method_roles"]
    if roles["sft"]["primary_tasks"] != ["numeral_system", "gravity"]:
        raise ValueError("SFT primary task roles changed")
    if roles["sft"]["equation_role"] != "build_initial_correctness_and_reduce_truncation_for_ppo_grpo_readiness":
        raise ValueError("Equation SFT readiness role changed")
    if roles["dpo"]["task"] != "gravity" or "short_over" not in roles["dpo"]["objective"]:
        raise ValueError("DPO Gravity shortening role changed")
    for method in ("ppo", "grpo"):
        if roles[method]["task"] != "equation_numeric" or roles[method]["initialization"] != "selected_sft_independent_arm":
            raise ValueError(f"{method} Equation role or independent initialization changed")
    return config


def training_settings(config):
    return {**config["training"], "seed": config["seed"]}


def verify_base(config):
    base = model_path()
    manifest_path = base / "download_manifest.json"
    manifest = read(manifest_path)
    if (manifest["status"], manifest["model_id"], manifest["revision"]) != (
        "completed", config["base_model_id"], config["base_revision"]
    ):
        raise ValueError("Base model identity or download status mismatch")
    for name, item in manifest["files"].items():
        path = base / name
        if path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
            raise ValueError(f"Base model file changed: {name}")
    from transformers import AutoConfig, AutoTokenizer
    model_config = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if model_config.model_type != "qwen3":
        raise ValueError("Unexpected base architecture")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    vocab_ids = list(tokenizer.get_vocab().values())
    if not rendered or not vocab_ids or min(vocab_ids) < 0 or max(vocab_ids) >= model_config.vocab_size:
        raise ValueError("Tokenizer/config mismatch")
    return base, manifest, tokenizer, model_config


def apis():
    from utils.run_nvidia_final_synthetic import load_definitions, load_renderer
    proof = load_definitions()
    sft = load_renderer()
    wanted = {"course_sft_loss_sum", "train_augmentation_branch", "collate_causal_examples"}
    for cell in read(ROOT / "03_SFT.ipynb")["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "03_SFT.ipynb", "exec"), sft)
    return proof, sft


def make_teacher(raw, original, config, proof_ns, sft_ns, tokenizer, official):
    seed = {
        "candidate_id": raw["source_id"],
        "task_type": raw["task_type"],
        "prompt": original["prompt"],
        "difficulty": "official_real_train",
    }
    proof = proof_ns["solver_first_structured_proof"](seed)
    if not proof["solver_proof_accepted"]:
        raise ValueError("Solver or independent proof rejected real Train sample")
    clean = proof["sample"]
    clean.update(source="verified_real_train", verification_status="accepted")
    canonical = {"verified": True, "scope": "independent non-equation Solver proof"}
    if raw["task_type"] == "equation_numeric":
        clean, canonical = sft_ns["strict_equation_render_reference"](clean, proof_ns)
    record = sft_ns["make_trace_record"](clean)
    record.update(
        source="verified_real_train",
        source_id=raw["source_id"],
        source_row=original["source_row"],
        source_split="train",
        verification_status="accepted",
        independent_verification_passed=True,
        canonical_solver_trace=clean["trace"],
        canonical_reference_audit=canonical,
        solver_proof_accepted=True,
        model_autonomous_cot_claim=False,
        raw_model_thinking_used=False,
    )
    if record["prompt"] != original["prompt"] or not sft_ns["audit_trace_record"](record)["accepted"]:
        raise ValueError("Teacher prompt or SFT audit mismatch")
    predicted, _ = official["extract_final_answer"](record["completion"], record["task_type"])
    official_ok, _ = official["verify_answer"](record["task_type"], predicted, original["answer"])
    if not official_ok:
        raise ValueError("Teacher final answer differs from official Train label")
    if record["task_type"] in ("numeral_system", "equation_numeric") and record["final_answer"] != original["answer"]:
        raise ValueError("Exact string teacher answer differs from official Train label")
    encoded = sft_ns["encode_chat_example"](tokenizer, record, config["training"]["max_length"])
    prompt_length = encoded["prompt_length"]
    if encoded["labels"][:prompt_length] != [-100] * prompt_length:
        raise ValueError("Prompt loss mask mismatch")
    if encoded["labels"][prompt_length:] != encoded["input_ids"][prompt_length:]:
        raise ValueError("Response loss mask mismatch")
    record["actual_token_counts"] = {
        "full_chat_tokens": len(encoded["input_ids"]),
        "supervised_tokens": sum(value != -100 for value in encoded["labels"][1:]),
        "completion_tokens": len(tokenizer.encode(record["completion"], add_special_tokens=False)),
    }
    return record


def select_smoke(records, per_task):
    selected = []
    for task in TASKS:
        group = [record for record in records if record["task_type"] == task]
        selected.extend(sorted(
            group, key=lambda record: (-record["actual_token_counts"]["full_chat_tokens"], record["source_id"])
        )[:per_task])
    return selected


def prepare():
    config = validate_config(read(CONFIG))
    run, _ = paths(config)
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite {run}")
    base, manifest, tokenizer, model_config = verify_base(config)
    proof_ns, sft_ns = apis()
    from utils.benchmark_training import verifier
    official = verifier()
    raw_records = rows(ROOT / config["teacher_pool"])
    with (ROOT / config["train_split"]).open() as stream:
        train = {record["id"]: record for record in csv.DictReader(stream)}
    selected = []
    rejected = []
    for index, task in enumerate(TASKS):
        pool = sorted((record for record in raw_records if record["task_type"] == task), key=lambda record: record["source_id"])
        random.Random(config["seed"] + index).shuffle(pool)
        accepted = []
        for raw in pool:
            original = train.get(raw["source_id"])
            if original is None or original["prompt"] != raw["prompt"]:
                raise ValueError("Real Train teacher provenance mismatch")
            try:
                accepted.append(make_teacher(raw, original, config, proof_ns, sft_ns, tokenizer, official))
            except (ValueError, KeyError, IndexError, TypeError) as error:
                rejected.append({"source_id": raw["source_id"], "task_type": task, "reason": str(error)})
            if len(accepted) == config["teachers_per_task"]:
                break
        if len(accepted) != config["teachers_per_task"]:
            raise ValueError(f"Insufficient verified {task} teachers: {len(accepted)}")
        selected.extend(accepted)
    if len({record["source_id"] for record in selected}) != len(selected):
        raise ValueError("Duplicate teacher IDs")
    panel = rows(ROOT / config["validation_panel"])
    if Counter(record["task_type"] for record in panel) != Counter({task: 50 for task in TASKS}):
        raise ValueError("Tune panel must contain 50 records per task")
    write_rows(run / "joint_teachers.jsonl", selected)
    write_rows(run / "teacher_rejections.jsonl", rejected)
    write_rows(run / "tune_panel.jsonl", panel)
    summary = {
        "status": "ready",
        "records": len(selected),
        "by_task": Counter(record["task_type"] for record in selected),
        "rejected_candidates": len(rejected),
        "supervised_tokens": sum(record["actual_token_counts"]["supervised_tokens"] for record in selected),
        "max_full_chat_tokens": max(record["actual_token_counts"]["full_chat_tokens"] for record in selected),
        "teacher_contract": config["teacher_contract"],
        "equation_exact_reference_count": sum(record["task_type"] == "equation_numeric" for record in selected),
        "base_model": config["base_model_id"],
        "base_revision": config["base_revision"],
        "model_layers": model_config.num_hidden_layers,
        "model_hidden_size": model_config.hidden_size,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "model_embedding_vocab_size": model_config.vocab_size,
        "tokenizer_max_token_id": max(tokenizer.get_vocab().values()),
        "holdout_used": False,
    }
    write(run / "prepare_summary.json", summary)
    registration = {
        "status": "registered_before_training",
        "created_unix": time.time(),
        "config": config,
        "config_sha256": sha(CONFIG),
        "base_path": str(base),
        "base_manifest_sha256": sha(base / "download_manifest.json"),
        "source_sha256": {source: sha(ROOT / source) for source in SOURCES},
        "teachers_sha256": sha(run / "joint_teachers.jsonl"),
        "panel_sha256": sha(run / "tune_panel.jsonl"),
        "selection_policy": "fixed Train-only RNG before validation outcomes; no Holdout access",
        "rank_comparison": "r8 versus r32; identical data, order, epochs, LR and supervised token budget",
        "rl_autostart": False,
    }
    write(run / "registration.json", registration)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


def verify_registration():
    config = validate_config(read(CONFIG))
    run, _ = paths(config)
    registration = read(run / "registration.json")
    if registration["config_sha256"] != sha(CONFIG) or registration["base_manifest_sha256"] != sha(model_path() / "download_manifest.json"):
        raise ValueError("Registered config or base model changed")
    for source, expected in registration["source_sha256"].items():
        if sha(ROOT / source) != expected:
            raise ValueError(f"Registered source changed: {source}")
    if sha(run / "joint_teachers.jsonl") != registration["teachers_sha256"] or sha(run / "tune_panel.jsonl") != registration["panel_sha256"]:
        raise ValueError("Registered data changed")
    return config, run, registration


class MemoryMonitor:
    def __init__(self, output, interval=0.25):
        self.output = Path(output)
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = []
        self.errors = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
                ], text=True, stderr=subprocess.STDOUT).strip()
                sample = {"at_unix": time.time(), "device_used_bytes": int(result.splitlines()[0]) * 1024**2}
                self.samples.append(sample)
                append(self.output, sample)
            except Exception as error:
                self.errors.append(str(error))
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return {
            "samples": len(self.samples),
            "peak_device_used_bytes": max((sample["device_used_bytes"] for sample in self.samples), default=None),
            "errors": self.errors,
            "scope": "whole-device nvidia-smi sampling; not a continuous hardware upper-bound proof",
        }


def load_training_model(config, rank, adapter=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_properties(0).total_memory < config["training"]["device_limit_bytes"]:
        raise ValueError("GPU is smaller than the registered process budget")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(
        min(0.92, config["training"]["device_limit_bytes"] / torch.cuda.get_device_properties(0).total_memory), 0
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_path(), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": "cpu"}, attn_implementation="sdpa",
    )
    base.config.use_cache = False
    if adapter is None:
        training = config["training"]
        lora = LoraConfig(
            r=rank,
            lora_alpha=rank * training["lora_alpha_multiplier"],
            lora_dropout=training["lora_dropout"],
            bias="none",
            target_modules=training["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)
    else:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False, torch_device="cpu")
    model = model.to("cuda")
    if adapter is None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train(adapter is None)
    tokenizer = AutoTokenizer.from_pretrained(model_path(), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.reset_peak_memory_stats()
    return model, tokenizer


def probe(model):
    import torch
    from utils.low_memory_logprobs import causal_policy_logprobs
    was_training = model.training
    model.eval()
    ids = torch.tensor([[2, 3, 4, 5]], device=model.get_input_embeddings().weight.device)
    with torch.no_grad():
        result = causal_policy_logprobs(model, ids, None, 64).float().cpu()
    model.train(was_training)
    return result


def release(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train_arm(rank, smoke=False):
    config, run, registration = verify_registration()
    if rank not in config["training"]["ranks"] or (smoke and rank != config["smoke"]["rank"]):
        raise ValueError("Unexpected rank")
    records = rows(run / "joint_teachers.jsonl")
    if smoke:
        records = select_smoke(records, config["smoke"]["longest_per_task"])
    name = f"smoke/r{rank}" if smoke else f"training/r{rank}"
    out = run / name
    out.mkdir(parents=True, exist_ok=False)
    weights = paths(config)[1] / f"smoke_r{rank}" if smoke else paths(config)[1] / f"r{rank}"
    weights.mkdir(parents=True, exist_ok=False)
    monitor = MemoryMonitor(out / "device_memory.jsonl").start()
    started = time.time()
    try:
        import torch
        from safetensors.torch import load_file
        model, tokenizer = load_training_model(config, rank)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        history = []

        def on_step(event):
            history.append(event)
            append(out / "training_curve.jsonl", event)
            print(f"r{rank} step {event['optimizer_step']} loss {event['token_mean_loss']:.5f} grad {event['gradient_norm']:.4f}", flush=True)
            if event["torch_peak_reserved_bytes"] >= config["training"]["device_limit_bytes"]:
                raise RuntimeError("Own-process memory guard exceeded")

        result = apis()[1]["train_augmentation_branch"](model, tokenizer, records, training_settings(config), on_step=on_step)
        expected = math.ceil(len(records) / config["training"]["gradient_accumulation"]) * config["training"]["epochs"]
        if result["optimizer_steps"] != expected or not history:
            raise ValueError("Incomplete optimizer updates")
        if not all(math.isfinite(event["token_mean_loss"]) and math.isfinite(event["gradient_norm"]) for event in history):
            raise ValueError("Nonfinite training event")
        saved_probe = probe(model)
        adapter = weights / "adapter"
        model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        state = result.pop("optimizer_state", None)
        if state is None:
            raise ValueError("Missing optimizer state")
        torch.save(state, weights / "optimizer.pt")
        tensors = load_file(str(adapter / "adapter_model.safetensors"))
        nonzero_b = sum(key.endswith("lora_B.weight") and bool(torch.count_nonzero(value)) for key, value in tensors.items())
        if not nonzero_b:
            raise ValueError("No nonzero trained LoRA B tensors")
        memory = monitor.stop()
        result.update(
            rank=rank,
            lora_alpha=rank * config["training"]["lora_alpha_multiplier"],
            smoke=smoke,
            records=len(records),
            trainable_parameters=trainable,
            nonzero_lora_b_tensors=nonzero_b,
            adapter_path=str(adapter),
            adapter_sha256=sha(adapter / "adapter_model.safetensors"),
            optimizer_path=str(weights / "optimizer.pt"),
            optimizer_sha256=sha(weights / "optimizer.pt"),
            probe_logprobs=saved_probe.tolist(),
            own_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            own_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            device_memory=memory,
            versions={"torch": torch.__version__, "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__},
            registration_sha256=sha(run / "registration.json"),
            wall_seconds=time.time() - started,
        )
        write(out / "result.json", result)
        release(model)
        print(json.dumps({key: value for key, value in result.items() if key not in ("history", "probe_logprobs")}, indent=2), flush=True)
        return result
    except BaseException as error:
        memory = monitor.stop()
        write(out / "failure.json", {
            "status": "failed", "error": type(error).__name__ + ": " + str(error),
            "traceback": traceback.format_exc(), "device_memory": memory,
        })
        raise


def reload_arm(rank, smoke=False):
    config, run, _ = verify_registration()
    name = f"smoke/r{rank}" if smoke else f"training/r{rank}"
    out = run / name
    result = read(out / "result.json")
    adapter = Path(result["adapter_path"])
    if sha(adapter / "adapter_model.safetensors") != result["adapter_sha256"]:
        raise ValueError("Adapter changed before reload")
    import torch
    model, _ = load_training_model(config, rank, adapter)
    actual = probe(model)
    expected = torch.tensor(result["probe_logprobs"])
    delta = float((actual - expected).abs().max())
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=1e-3)
    report = {"status": "completed", "rank": rank, "smoke": smoke, "max_logprob_delta": delta, "adapter_sha256": result["adapter_sha256"]}
    write(out / "reload.json", report)
    release(model)
    print(json.dumps(report, indent=2))


def has_repeated_ngram(token_ids, n=32, window=2048):
    values = list(token_ids[-window:])
    seen = set()
    for start in range(max(0, len(values) - n + 1)):
        gram = tuple(values[start:start + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def metrics(values, config=None):
    config = config or read(CONFIG)
    result = {}
    for task in TASKS:
        group = [value for value in values if value["task_type"] == task]
        if not group:
            continue
        lengths = [len(value["completion_token_ids"]) for value in group]
        correct = sum(bool(value["correct"]) for value in group)
        result[task] = {
            "n": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
            "truncated": sum(bool(value["truncated"]) for value in group),
            "truncation_rate": sum(bool(value["truncated"]) for value in group) / len(group),
            "short_wrong": sum(
                not value["correct"] and not value["truncated"] and len(value["completion_token_ids"]) <= config["evaluation"]["short_wrong_max_tokens"]
                for value in group
            ),
            "repeated_ngram": sum(bool(value.get("repeated_ngram")) for value in group),
            "extraction_success": sum(value.get("predicted") is not None for value in group),
            "mean_tokens": statistics.mean(lengths),
            "median_tokens": statistics.median(lengths),
            "mean_correct_tokens": statistics.mean([len(value["completion_token_ids"]) for value in group if value["correct"]]) if correct else None,
        }
    result["macro_accuracy"] = statistics.mean(result[task]["accuracy"] for task in TASKS)
    result["sft_primary_macro_accuracy"] = statistics.mean(result[task]["accuracy"] for task in ("numeral_system", "gravity"))
    return result


def mcnemar_exact(discordant_a, discordant_b):
    n = discordant_a + discordant_b
    if not n:
        return 1.0
    k = min(discordant_a, discordant_b)
    return min(1.0, 2 * sum(math.comb(n, index) for index in range(k + 1)) / 2**n)


def evaluate():
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from utils.benchmark_training import verifier
    config, run, _ = verify_registration()
    out = run / "evaluation"
    out.mkdir(exist_ok=False)
    monitor = MemoryMonitor(out / "device_memory.jsonl").start()
    base = model_path()
    evaluation = config["evaluation"]
    official = verifier()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    panel = rows(run / "tune_panel.jsonl")
    rendered = {}
    for record in panel:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=evaluation["enable_thinking"],
        )
        cap = evaluation["caps"][record["task_type"]]
        if len(tokenizer.encode(text, add_special_tokens=False)) + cap > evaluation["max_model_len"]:
            raise ValueError("Evaluation context overflow")
        rendered[record["id"]] = text
    started = time.time()
    try:
        engine = LLM(
            model=str(base), tokenizer=str(base), dtype="bfloat16", tensor_parallel_size=1,
            enable_lora=True, max_lora_rank=max(config["training"]["ranks"]),
            max_model_len=evaluation["max_model_len"], max_num_seqs=evaluation["max_num_seqs"],
            max_num_batched_tokens=evaluation["max_num_batched_tokens"],
            kv_cache_memory_bytes=int(evaluation["kv_cache_gib"] * 2**30),
            gpu_memory_utilization=evaluation["gpu_memory_utilization"],
            enforce_eager=evaluation["enforce_eager"], enable_chunked_prefill=True,
            generation_config="vllm", seed=config["seed"],
        )
        for arm_index, arm in enumerate(evaluation["arms"], 1):
            arm_out = out / arm
            arm_out.mkdir()
            adapter = None
            adapter_sha256 = None
            if arm != "base":
                rank = int(arm[1:])
                training_result = read(run / f"training/r{rank}/result.json")
                adapter = Path(training_result["adapter_path"])
                adapter_sha256 = training_result["adapter_sha256"]
                if sha(adapter / "adapter_model.safetensors") != adapter_sha256:
                    raise ValueError("Evaluation adapter changed")
            request = LoRARequest(arm, arm_index, str(adapter)) if adapter is not None else None
            write(arm_out / "job.json", {"arm": arm, "adapter_sha256": adapter_sha256, "config": evaluation, "base_revision": config["base_revision"]})
            arm_rows = []
            for task in TASKS:
                task_rows = [record for record in panel if record["task_type"] == task]
                cap = evaluation["caps"][task]
                params = SamplingParams(
                    n=1, temperature=evaluation["temperature"], top_p=evaluation["top_p"],
                    top_k=evaluation["top_k"], max_tokens=cap, seed=config["seed"], skip_special_tokens=False,
                )
                for start in range(0, len(task_rows), evaluation["batch_size"]):
                    chunk = task_rows[start:start + evaluation["batch_size"]]
                    outputs = engine.generate([rendered[record["id"]] for record in chunk], params, lora_request=request)
                    for original, output in zip(chunk, outputs):
                        generated = output.outputs[0]
                        ids = list(generated.token_ids)
                        predicted, extraction = official["extract_final_answer"](generated.text, task)
                        correct, verification = official["verify_answer"](task, predicted, original["answer"])
                        truncated = generated.finish_reason == "length"
                        value = {
                            "id": original["id"], "task_type": task, "prompt": original["prompt"],
                            "expected_answer": original["answer"], "arm": arm,
                            "completion": generated.text, "completion_token_ids": ids,
                            "prompt_token_ids": list(output.prompt_token_ids),
                            "finish_reason": generated.finish_reason, "truncated": truncated,
                            "max_tokens": cap, "predicted": predicted, "extraction_method": extraction,
                            "verification_method": verification, "correct": bool(correct and not truncated),
                            "repeated_ngram": has_repeated_ngram(
                                ids, evaluation["repetition_ngram"], evaluation["repetition_window_tokens"]
                            ),
                            "adapter_sha256": adapter_sha256,
                        }
                        arm_rows.append(value)
                        append(arm_out / "rows.jsonl", value)
                    print(f"{arm} {task} {len(arm_rows)}/{len(panel)}", flush=True)
            write(arm_out / "result.json", {"status": "completed", "metrics": metrics(arm_rows, config), "rows": len(arm_rows)})
        memory = monitor.stop()
        write(out / "result.json", {
            "status": "completed", "arms": evaluation["arms"], "rows_per_arm": len(panel),
            "wall_seconds": time.time() - started, "device_memory": memory,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        memory = monitor.stop()
        write(out / "failure.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc(), "device_memory": memory})
        raise


def selection_readiness(candidate, baseline, config=None):
    config = config or read(CONFIG)
    gate = config["sft_selection"]
    checks = {
        "numeral_accuracy_not_below_base": candidate["numeral_system"]["correct"] >= baseline["numeral_system"]["correct"],
        "gravity_accuracy_not_below_base": candidate["gravity"]["correct"] >= baseline["gravity"]["correct"],
        "sft_primary_macro_improved": candidate["sft_primary_macro_accuracy"] > baseline["sft_primary_macro_accuracy"],
        "equation_minimum_correct": candidate["equation_numeric"]["correct"] >= gate["equation_min_correct_on_tune"],
        "equation_accuracy_not_below_base": candidate["equation_numeric"]["correct"] >= baseline["equation_numeric"]["correct"],
        "equation_truncation_reduced": candidate["equation_numeric"]["truncation_rate"] <= min(
            gate["equation_max_truncation_rate"], baseline["equation_numeric"]["truncation_rate"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "note": "SFT panel gate only; DPO/PPO/GRPO still require real Train rollout signal gates"}


def report():
    config, run, _ = verify_registration()
    evaluation = run / "evaluation"
    values = {arm: rows(evaluation / arm / "rows.jsonl") for arm in config["evaluation"]["arms"]}
    result = {"arms": {arm: metrics(group, config) for arm, group in values.items()}, "comparisons": {}}
    result["sft_selection"] = {
        arm: selection_readiness(result["arms"][arm], result["arms"]["base"], config)
        for arm in ("r8", "r32")
    }
    for left, right in (("base", "r8"), ("base", "r32"), ("r8", "r32")):
        a = {value["id"]: value for value in values[left]}
        b = {value["id"]: value for value in values[right]}
        if a.keys() != b.keys():
            raise ValueError("Incomplete paired evaluation")
        comparison = {}
        for task in TASKS:
            ids = [key for key in a if a[key]["task_type"] == task]
            left_only = sum(a[key]["correct"] and not b[key]["correct"] for key in ids)
            right_only = sum(b[key]["correct"] and not a[key]["correct"] for key in ids)
            comparison[task] = {
                f"{left}_only_correct": left_only,
                f"{right}_only_correct": right_only,
                "accuracy_delta_right_minus_left": (right_only - left_only) / len(ids),
                "mcnemar_exact_p": mcnemar_exact(left_only, right_only),
            }
        result["comparisons"][f"{left}_vs_{right}"] = comparison
    write(evaluation / "comparison.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "train", "reload", "evaluate", "report"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "smoke":
        train_arm(read(CONFIG)["smoke"]["rank"], smoke=True)
    elif args.stage == "train":
        train_arm(args.rank)
    elif args.stage == "reload":
        reload_arm(args.rank, smoke=args.smoke)
    elif args.stage == "evaluate":
        evaluate()
    else:
        report()


if __name__ == "__main__":
    main()
