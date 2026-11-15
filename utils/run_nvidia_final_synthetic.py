"""Final synthetic pilot orchestration. Math, priors, prompts and renderer live in notebooks.

No GPU/model loading, no SFT merge, no training. Live stages use only the built-in
NVIDIA provider of data-designer 0.10.0, after a recorded passing CPU regression.
A durable send-before-write ledger prevents retries even across interruptions.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import copy
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from utils.run_nvidia_v2 import file_hash, tree_hashes, write_json_new, write_jsonl_new, sft_training_tasks
from utils.run_synthetic_smoke import read, rows
from utils.run_nvidia_synthetic_smoke import redact, require_api_key, bind_raw_response, sdk_generate

CONFIG = ROOT / "configs/nvidia_synthetic_final_20261115.json"
NOTEBOOK = ROOT / "01_Synthetic_Data_Engineering.ipynb"
SFT = ROOT / "03_SFT.ipynb"
TASKS = ("numeral_system", "gravity", "equation_numeric")
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
SOURCES = (NOTEBOOK, SFT, Path(__file__), CONFIG, ROOT / "tests/test_nvidia_synthetic_final.py",
           ROOT / "tests/test_nvidia_v2_repair.py", ROOT / "utils/run_nvidia_v2.py",
           ROOT / "utils/run_nvidia_synthetic_smoke.py", ROOT / "utils/run_synthetic_smoke.py")
PROTECTED = ("04_DPO.ipynb", "05_PPO.ipynb", "06_GRPO.ipynb", "utils/course_worker.py", "utils/course_run.py",
             "utils/course_protocol.py", "utils/low_memory_logprobs.py")


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def append_event(path, event):
    """Flush before sending: unknown interrupted requests keep their full reservation."""
    with Path(path).open("a") as f:
        f.write(json.dumps(redact(event), ensure_ascii=False, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def config(path=CONFIG, check_sdk=True):
    c = read(path)
    if (c["schema_version"], c["run_id"], c["rng_seed"]) != ("nvidia-synthetic-final/v1", "nvidia_synthetic_final_20261115_r1", 20261115):
        raise ValueError("Use the registered final pilot configuration")
    if (c["model_id"], c["provider"], c["reasoning_effort"], c["temperature"], c["top_p"]) != (
        "nvidia/nemotron-3-ultra-550b-a55b", "nvidia", "low", 1.0, .95): raise ValueError("Frozen teacher settings changed")
    if (c["max_http_requests"], c["completion_token_budget"], c["http_retries"], c["concurrency"]) != (48, 500000, 0, 1):
        raise ValueError("Frozen request/token/retry/concurrency limits changed")
    if c["max_tokens"] != {"numeral_system": 16384, "gravity": 32768, "equation_numeric": 16384}: raise ValueError("Frozen caps changed")
    if tuple(c["tasks"]) != TASKS or c["candidates_per_task"] != 20 or c["target_seeds_per_task"] != 10:
        raise ValueError("Fixed 60-candidate/30-target pilot required")
    if c["merge_into_sft"] or c["gpu_training"] or c["expand_to_100"]: raise ValueError("Preview-only synthetic scope")
    if c["endpoint"] != ENDPOINT.rsplit("/chat/completions", 1)[0] or c["api_key_env"] != "NVIDIA_API_KEY": raise ValueError("Invalid endpoint/key source")
    if "nvapi-" in json.dumps(c): raise ValueError("Credential material forbidden")
    if check_sdk and importlib.metadata.version("data-designer") != "0.10.0": raise ValueError("Pin data-designer 0.10.0")
    return c


def load_definitions():
    """Definitions/static specs only. Does not execute Notebook config or data IO.

    This deliberately does NOT use the historical loader, which reads all three
    splits. All non-function expressions require a closed constant allowlist.
    """
    import __future__
    import types
    static = {"OFFICIAL_VERIFIER_VERSION", "OFFICIAL_NUMERIC_REL_TOL", "OFFICIAL_NUMERIC_ABS_TOL", "GENERATOR_VERSION",
        "ROMAN_GRAMMAR", "ROMAN_VALUES", "NUMERIC24_RULES", "NUMERIC24_RULE_BY_NAME", "NUMERIC24_FORMULAS",
        "EQUATION_NUMERIC_PRIOR", "GRAVITY_PAIR_RE", "NUMERIC_EXPRESSION_RE", "SYNTHETIC_VERIFIER_VERSION",
        "SYNTHETIC_SCHEMA_FIELDS", "SMOKE_PREFIXES", "SMOKE_TASKS", "SMOKE_VERIFIER_VERSION", "SMOKE_TRACE_INSTRUCTIONS",
        "NVIDIA_INTRODUCTIONS", "NVIDIA_VERIFIER_VERSION", "NVIDIA_ACCEPTANCE_PROTOCOL_V2", "NVIDIA_VERIFIER_V2",
        "TRAIN_ONLY_GENERATOR_VERSION", "TRAIN_ONLY_TASKS", "TRAIN_ONLY_FEATURES", "FINAL_BOUNDED_PRECISION",
        "FINAL_GRAVITY_QUERY_CONTRACT", "COMPACT_TRACE_INSTRUCTIONS", "gravity_prior"}
    module = types.ModuleType("final_synthetic_notebook")
    ns = module.__dict__; sys.modules[module.__name__] = module
    ns.update(MODEL_ID="Qwen/Qwen3-0.6B", TASK_ORDER=TASKS, numeric_operator_pool=())
    flags = __future__.annotations.compiler_flag
    for i, cell in enumerate(read(NOTEBOOK)["cells"]):
        if cell["cell_type"] != "code": continue
        selected = []
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                selected.append(node)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in static:
                selected.append(node)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK) + f":cell{i}", "exec", flags=flags), ns)
    if "build_train_only_prior" not in ns: raise ValueError("Final Notebook definitions missing")
    return ns


def load_renderer():
    from utils.run_nvidia_v2 import load_sft_render_definitions
    ns = load_sft_render_definitions()
    for cell in read(SFT)["cells"]:
        if cell["cell_type"] != "code": continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name == "render_solver_first_preview":
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(SFT), "exec"), ns)
    if "render_solver_first_preview" not in ns: raise ValueError("Solver-first SFT renderer missing")
    return ns


@contextlib.contextmanager
def train_read_boundary(c, events):
    """Deny evaluation or alternate real-data reads while building/generating.

    Installed library resources/code and frozen final-run files are unaffected.
    Both Python Path.open and pandas' regular open are intercepted and logged.
    """
    manifest = (ROOT / c["dataset_manifest"]).resolve()
    train = (manifest.parent / "splits/train.csv").resolve()
    original_builtin, original_io = builtins.open, io.open
    def checked(original):
        def open_file(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and not any(x in mode for x in "wax+"):
                p = Path(os.fsdecode(file)).resolve()
                if ROOT / "data" in p.parents:
                    if p in (manifest, train):
                        events.append({"path": str(p.relative_to(ROOT)), "mode": mode, "purpose": "Train-only prior input"})
                    elif not (ROOT / "data/processed/round3" / c["run_id"]) in p.parents:
                        raise PermissionError("Generator cannot read non-Train data: " + str(p.relative_to(ROOT)))
            return original(file, mode, *args, **kwargs)
        return open_file
    with patch("builtins.open", checked(original_builtin)), patch("io.open", checked(original_io)):
        yield


def read_train_only(c):
    import pandas as pd
    path = ROOT / c["dataset_manifest"]
    manifest = read(path)
    train_path = ROOT / manifest["files"]["train"]
    if train_path != path.parent / "splits/train.csv": raise ValueError("Unexpected Train path")
    if file_hash(train_path) != manifest["artifacts"]["train"]["sha256"]: raise ValueError("Train file hash mismatch")
    if [manifest["artifacts"][s]["rows"] for s in ("train", "validation", "holdout_test")] != [3123, 391, 391]:
        raise ValueError("Official dataset split sizes changed")
    # This list deliberately omits answer and source_dataset.
    train = pd.read_csv(train_path, dtype=str, keep_default_na=False, usecols=["id", "source_row", "task_type", "prompt", "split"])
    return manifest, train, train_path


def historical_trees(c):
    names = [c["high_baseline_r4"], "data/processed/round3/nvidia_hosted_20261115_ultra_r4_reaudit_v2",
        "data/processed/round3/nvidia_hosted_20261115_ultra_r4_sft_preview_v2",
        "data/processed/round3/nvidia_gravity_targeted_v2_20261115_live_r1",
        "data/processed/round3/nvidia_gravity_targeted_v2_20261115_live_r2_continuation"]
    return {name: tree_hashes(ROOT / name) for name in names}


def prepare(target=None):
    c = config(); target = Path(target) if target else ROOT / "data/processed/round3" / c["run_id"]
    if target.name != c["run_id"]: raise ValueError("Use the registered Run directory name")
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); ns = load_definitions(); events = []
    with train_read_boundary(c, events):
        manifest, train, train_path = read_train_only(c)
        prior = ns["build_train_only_prior"](train, manifest)
    write_json_new(target / "config.json", c)
    write_json_new(target / "train_prior.json", prior)
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prior_manifest = ns["build_train_prior_manifest"](prior, file_hash(ROOT / c["dataset_manifest"]), file_hash(train_path),
        git, file_hash(NOTEBOOK), sampled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    prior_manifest["prior_file_sha256"] = file_hash(target / "train_prior.json")
    write_json_new(target / "train_prior_manifest.json", prior_manifest)
    write_json_new(target / "prepare_read_audit.json", {"status": "passed", "reads": events,
        "parsed_columns": ["id", "source_row", "task_type", "prompt", "split"], "answer_used": False,
        "validation_or_holdout_read": False, "elapsed_seconds": time.perf_counter()-started})
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in SOURCES}
    from data_designer.engine.models.clients import retry, model_request_executor
    from data_designer.engine.models.clients.adapters import http_model_client
    write_json_new(target / "sdk_http_client_audit.json", {"sdk_version":"0.10.0",
        "official_provider":"nvidia", "custom_provider":False, "request_api":"HTTPX AsyncClient/Client",
        "retry_transport_max_retries":0, "request_executor_max_retries":0,
        "additional_guard":"durable per Seed x condition send-before-network reservation",
        "source_sha256":{m.__name__:file_hash(m.__file__) for m in (retry,model_request_executor,http_model_client)},
        "sdk_conversation_restarts":0,"sdk_correction_steps":0})
    write_json_new(target / "run_manifest.json", {"run_id": c["run_id"], "git_commit": git, "source_sha256": sources,
        "config_sha256": file_hash(target / "config.json"), "dataset_manifest_sha256": file_hash(ROOT / c["dataset_manifest"]),
        "train_prior_manifest_sha256": file_hash(target / "train_prior_manifest.json"), "history_before": historical_trees(c),
        "protected_sha256": {p: file_hash(ROOT / p) for p in PROTECTED},
        "model_revision": "Hosted API does not expose immutable weights revision; response model archived per request",
        "environment": {name: importlib.metadata.version(name) for name in ("data-designer", "httpx", "numpy", "pandas", "scipy")},
        "api_key_source": "environment variable only", "merge_into_sft": False, "gpu_training": False})
    return target


def verify_run(target, check_sources=True, check_sdk=True):
    target = Path(target); manifest = read(target / "run_manifest.json"); c = config(target / "config.json", check_sdk=check_sdk)
    if file_hash(target / "config.json") != manifest["config_sha256"]: raise ValueError("Run configuration changed")
    if file_hash(ROOT / c["dataset_manifest"]) != manifest["dataset_manifest_sha256"]: raise ValueError("Dataset manifest changed")
    if file_hash(target / "train_prior_manifest.json") != manifest["train_prior_manifest_sha256"]: raise ValueError("Prior manifest changed")
    prior_manifest = read(target / "train_prior_manifest.json")
    if file_hash(target / "train_prior.json") != prior_manifest["prior_file_sha256"]: raise ValueError("Prior changed")
    if check_sources:
        for p, h in manifest["source_sha256"].items():
            if file_hash(ROOT / p) != h: raise ValueError("Source changed after registration; do not mix configurations: " + p)
    return c, manifest


def verify_seed_pool(target):
    target=Path(target); c,run=verify_run(target)
    manifest=read(target/"seed_manifest.json")
    if manifest["prior_manifest_sha256"]!=run["train_prior_manifest_sha256"]:raise ValueError("Seed prior provenance changed")
    for name,h in manifest["artifact_sha256"].items():
        if file_hash(target/name)!=h:raise ValueError("Seed audit artifact changed: "+name)
    frozen={s["candidate_id"]:s for s in rows(target/"candidate_pool_pre_audit.jsonl")}
    seeds=rows(target/"seed_accepted.jsonl")
    for seed in seeds:
        original=frozen.get(seed["candidate_id"])
        if not original or any(seed.get(k)!=v for k,v in original.items() if k not in {"answer","trace"}):
            raise ValueError("Accepted Seed differs from pre-audit frozen candidate")
        if not seed["seed_gate_passed"] or seed["lineage_status"]!="declared" or seed["leakage_audit"]["status"]!="passed":
            raise ValueError("Unqualified Seed in Accepted pool")
    return seeds


def generate(target):
    target = Path(target); c, manifest = verify_run(target)
    if (target / "candidate_pool_pre_audit.jsonl").exists(): raise FileExistsError("Candidate pool already exists; no resampling")
    if (target / "audit_started.json").exists(): raise RuntimeError("Audit already started; cannot regenerate")
    ns = load_definitions(); events = []; started = time.perf_counter()
    with train_read_boundary(c, events):
        prior = read(target / "train_prior.json")
        rng = ns["np"].random.default_rng(c["rng_seed"])
        candidates = [ns["generate_train_only_candidate"](prior, rng, task, f"{task}-final-{i:03d}",
            manifest["train_prior_manifest_sha256"], c["rng_seed"]) for task in TASKS for i in range(1, 21)]
        # No Solver selection, no evaluation reads, and exactly 20 attempts/task.
        write_jsonl_new(target / "candidate_pool_pre_audit.jsonl", candidates)
        distribution = ns["audit_train_sampling_distribution"](prior, candidates)
    freeze = {"status": "frozen_before_evaluation_reads", "candidate_count": len(candidates), "per_task": {t:20 for t in TASKS},
        "candidate_pool_sha256": file_hash(target / "candidate_pool_pre_audit.jsonl"),
        "prior_manifest_sha256": manifest["train_prior_manifest_sha256"], "frozen_at_unix": time.time(),
        "elapsed_seconds": time.perf_counter()-started, "evaluation_split_reads": 0, "reads": events,
        "resampling_after_audit_allowed": False}
    write_json_new(target / "candidate_pool_freeze.json", freeze)
    write_json_new(target / "train_distribution_pre_audit.json", distribution)
    return freeze


def verify_candidate_freeze(target):
    target = Path(target); freeze = read(target / "candidate_pool_freeze.json")
    if file_hash(target / "candidate_pool_pre_audit.jsonl") != freeze["candidate_pool_sha256"]: raise ValueError("Frozen candidate pool changed")
    return freeze


def audit(target):
    import pandas as pd
    target = Path(target); c, manifest = verify_run(target); freeze = verify_candidate_freeze(target)
    if (target / "seed_candidates.jsonl").exists(): raise FileExistsError("Audit already exists; do not overwrite")
    write_json_new(target / "audit_started.json", {"started_at_unix":time.time(), "candidate_freeze_sha256":file_hash(target / "candidate_pool_freeze.json"),
        "first_evaluation_content_reads_follow_this_marker": True})
    ns = load_definitions(); prior = read(target / "train_prior.json"); dataset = read(ROOT / c["dataset_manifest"])
    frames, split_checks = [], {}
    for split in ("train", "validation", "holdout_test"):
        path = ROOT / dataset["files"][split]
        if file_hash(path) != dataset["artifacts"][split]["sha256"]: raise ValueError("Dataset split hash mismatch")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False,
            usecols=["id", "source_row", "task_type", "prompt", "answer", "split"])
        if not frame.split.eq(split).all() or len(frame) != dataset["artifacts"][split]["rows"]: raise ValueError("Split contents changed")
        frames.append(frame); split_checks[split] = {"sha256":file_hash(path), "rows":len(frame), "loaded_after_pool_freeze":True}
    real_index = ns["build_real_index"](pd.concat(frames, ignore_index=True))
    candidates, accepted, rejected, lineages = [], [], [], []
    seen_exact, seen_normalized = set(), set(); selected = {t:0 for t in TASKS}
    for frozen in rows(target / "candidate_pool_pre_audit.jsonl"):
        sample = copy.deepcopy(frozen)
        lineage = ns["audit_train_prior_lineage"](sample, prior, manifest["train_prior_manifest_sha256"])
        lineages.append({"candidate_id":sample["candidate_id"], "task_type":sample["task_type"], **lineage})
        try: solved = ns["solver_first_structured_proof"](sample)
        except Exception as error: solved = {"solver_proof_accepted":False, "error": type(error).__name__ + ": " + str(error)}
        if solved.get("sample"):
            sample.update(answer=solved["sample"]["answer"], trace=solved["sample"]["trace"])
        leakage = ns["audit_final_seed_leakage"](sample, real_index, lineage)
        reasons = list(leakage["rejection_reasons"])
        if sample.get("generation_error"): reasons.append(sample["generation_error"])
        if not solved["solver_proof_accepted"]: reasons.append("independent_solver_proof_failed")
        exact, normalized = ns["prompt_hash"](sample["prompt"]), ns["synthetic_prompt_fingerprint"](sample["prompt"])
        if exact in seen_exact or normalized in seen_normalized: reasons.append("within_pool_prompt_duplicate")
        seen_exact.add(exact); seen_normalized.add(normalized)
        gate_passed = not reasons
        quota = selected[sample["task_type"]] < 10
        admitted = gate_passed and quota
        if gate_passed and not quota: reasons.append("passed_gates_not_selected_quota")
        if admitted: selected[sample["task_type"]] += 1
        sample.update(verification_status="accepted" if admitted else "rejected", seed_gate_passed=gate_passed,
            solver_proof_accepted=solved["solver_proof_accepted"], solver_verification=solved.get("verification"),
            solver_sample_verification=solved.get("sample_verification"), solver_error=solved.get("error"),
            lineage_audit=lineage, lineage_status=lineage["lineage_status"], train_only_provenance=lineage["train_only_provenance"],
            leakage_audit=leakage, direct_mapping_leakage=leakage["direct_mapping_leakage"],
            rejection_reasons=sorted(set(reasons)), merge_into_sft=False)
        candidates.append(sample); (accepted if admitted else rejected).append(sample)
    write_jsonl_new(target / "seed_candidates.jsonl", candidates)
    write_jsonl_new(target / "seed_accepted.jsonl", accepted)
    write_jsonl_new(target / "seed_rejected.jsonl", rejected)
    write_jsonl_new(target / "public_new_seeds.jsonl", [{k:s[k] for k in ("seed_id", "task_type", "prompt")} for s in accepted])
    write_json_new(target / "train_only_lineage_audit.json", {"status":"passed" if all(x["train_only_provenance"] for x in lineages) else "failed",
        "records":lineages, "split_checks":split_checks, "candidate_pool_sha256":freeze["candidate_pool_sha256"],
        "by_task":{t:{"candidates":20, "lineage_pass":sum(x["task_type"]==t and x["train_only_provenance"] for x in lineages),
            "solver_pass":sum(s["task_type"]==t and s["solver_proof_accepted"] for s in candidates),
            "leakage_pass":sum(s["task_type"]==t and s["leakage_audit"]["status"]=="passed" for s in candidates),
            "all_seed_gates_pass":sum(s["task_type"]==t and s["seed_gate_passed"] for s in candidates),
            "selected":selected[t]} for t in TASKS}})
    write_json_new(target / "train_distribution_audit.json", {"all_candidates":read(target / "train_distribution_pre_audit.json"),
        "selected_seed_panel":ns["audit_train_sampling_distribution"](prior, accepted), "holdout_used_to_adjust_distribution":False})
    write_json_new(target / "seed_manifest.json", {"status":"passed" if accepted else "empty_accepted", "run_status":"seed_panel_complete" if all(n==10 for n in selected.values()) else "seed_panel_shortfall",
        "accepted_rows":len(accepted), "by_task":selected, "all_accepted_lineage_declared":all(s["lineage_status"]=="declared" for s in accepted),
        "artifact_sha256":{p:file_hash(target/p) for p in ("seed_candidates.jsonl", "seed_accepted.jsonl", "seed_rejected.jsonl", "public_new_seeds.jsonl")},
        "candidate_pool_sha256":freeze["candidate_pool_sha256"], "split_sha256":{k:v["sha256"] for k,v in split_checks.items()},
        "prior_manifest_sha256":manifest["train_prior_manifest_sha256"], "selection":"fixed candidate order, first 10 gate passes/task, no resampling",
        "merge_into_sft":False})
    return read(target / "train_only_lineage_audit.json")["by_task"]


def cpu_regression(target):
    target = Path(target); verify_run(target)
    if (target / "cpu_regression.json").exists(): raise FileExistsError("CPU test record already exists")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "tests") + os.pathsep + str(ROOT), NEMO_TELEMETRY_ENABLED="false")
    # No real API key in the test subprocess; it uses an explicit dummy secret.
    env.pop("NVIDIA_API_KEY", None)
    command = [sys.executable, "-m", "unittest", "tests.test_nvidia_synthetic_final", "-v"]
    started=time.perf_counter(); result=subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    with (target / "cpu_regression.log").open("x") as f: f.write(redact(result.stdout+result.stderr))
    write_json_new(target / "cpu_regression.json", {"status":"passed" if result.returncode==0 else "failed", "command":command,
        "exit_code":result.returncode, "elapsed_seconds":time.perf_counter()-started,
        "log_sha256":file_hash(target/"cpu_regression.log"), "source_sha256":read(target/"run_manifest.json")["source_sha256"],
        "live_api_calls":0, "gpu_used":False})
    if result.returncode: raise RuntimeError("CPU regression failed; live API remains blocked. See cpu_regression.log")
    return read(target / "cpu_regression.json")


class BudgetStop(RuntimeError): pass
class DuplicateRequestBlocked(RuntimeError): pass


class RequestBudget:
    def __init__(self, target, c):
        self.path=Path(target)/"request_ledger.jsonl"; self.c=c; self.lock=threading.RLock()
    def events(self): return rows(self.path) if self.path.exists() else []
    def state(self):
        reservations={}; settled={}; blocked=[]
        for e in self.events():
            if e["event"]=="reserved_before_send": reservations[e["key"]]=e
            elif e["event"]=="settled": settled[e["key"]]=e
            elif e["event"]=="blocked": blocked.append(e)
        charged={k:settled.get(k,{}).get("charged_completion_tokens", r["max_tokens"]) for k,r in reservations.items()}
        return {"request_count":len(reservations), "completion_tokens_charged_or_reserved":sum(charged.values()),
            "reported_completion_tokens":sum(e["completion_tokens"] for e in settled.values() if e["completion_tokens"] is not None),
            "unknown_usage_requests":sum(k not in settled or settled[k]["completion_tokens"] is None for k in reservations),
            "keys":sorted(reservations), "blocked_requests":blocked}
    def reserve(self, key, cap, request_sha256):
        with self.lock:
            state=self.state()
            reason=None
            if key in state["keys"]: reason="duplicate_seed_condition_request"
            elif state["request_count"]>=self.c["max_http_requests"]: reason="request_count_budget"
            elif state["completion_tokens_charged_or_reserved"]+cap>self.c["completion_token_budget"]: reason="completion_token_budget"
            if reason:
                append_event(self.path,{"event":"blocked", "key":key, "reason":reason, "at_unix":time.time()})
                if reason.startswith("duplicate"): raise DuplicateRequestBlocked(reason)
                raise BudgetStop(reason)
            append_event(self.path,{"event":"reserved_before_send", "key":key, "max_tokens":cap,
                "request_sha256":request_sha256, "at_unix":time.time()})
    def settle(self, key, cap, usage):
        usage=usage if isinstance(usage,dict) else {}
        n=usage.get("completion_tokens")
        if type(n) is not int or n<0: n=None
        charge=cap if n is None else max(n,0)
        append_event(self.path,{"event":"settled", "key":key, "completion_tokens":n,
            "charged_completion_tokens":charge, "usage_status":"unknown_reserved_max_tokens" if n is None else "reported",
            "at_unix":time.time()})
        if charge>cap: raise BudgetStop("Provider reported usage above requested cap; abort further requests")


@contextlib.contextmanager
def live_lock(target):
    with (Path(target)/"live_stage.lock").open("a+") as f:
        try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("Another live stage holds the run lock")
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def single_request_capture(target, c, key, seed, condition, destination):
    """Actual HTTPX send guard, no custom provider. Never archive headers.

    The SDK has both RetryTransport and ModelRequestExecutor retry policies.
    Both receive RetryConfig(max_retries=0); the persisted send guard additionally
    refuses a second actual transport call, including after process restart.
    """
    import httpx
    from data_designer.engine.models.clients.retry import RetryConfig, create_retry_transport
    from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
    budget=RequestBudget(target,c); cap=c["max_tokens"][seed["task_type"]]
    orig_sync=httpx.HTTPTransport.handle_request; orig_async=httpx.AsyncHTTPTransport.handle_async_request
    orig_executor=ModelRequestExecutor.__init__
    def no_retry_transport(config=None, **kwargs):
        return create_retry_transport(RetryConfig(max_retries=0), **kwargs)
    def executor_init(self,*args,**kwargs):
        kwargs["retry_config"]=RetryConfig(max_retries=0)
        orig_executor(self,*args,**kwargs)
    def start(request):
        if str(request.url)!=ENDPOINT: raise RuntimeError("Unexpected endpoint; no fallback")
        payload=json.loads(request.content)
        if payload.get("model")!=c["model_id"] or payload.get("reasoning_effort")!="low" or payload.get("max_tokens")!=cap:
            raise RuntimeError("Actual HTTP request differs from frozen model/reasoning/cap")
        if payload.get("temperature")!=1.0 or payload.get("top_p")!=.95 or payload.get("stream",False):
            raise RuntimeError("Actual HTTP sampling parameters changed")
        if "tools" in payload: raise RuntimeError("Tools are not part of the proof experiment")
        def content_text(content):
            if isinstance(content,str): return content
            if isinstance(content,list): return "\n".join(p.get("text","") for p in content if isinstance(p,dict))
            return ""
        text="\n".join(content_text(m.get("content")) for m in payload.get("messages",[]))
        if seed["prompt"] not in text or "SEED_ID="+seed["seed_id"] not in text: raise RuntimeError("Public prompt projection changed")
        schema=load_definitions()["smoke_trace_schema"](seed["task_type"])
        if payload.get("response_format",{}).get("json_schema",{}).get("schema")!=schema: raise RuntimeError("Actual HTTP JSON Schema changed")
        secret=os.environ.get("NVIDIA_API_KEY", "")
        if secret and secret in json.dumps(payload): raise RuntimeError("Credential included in request body")
        budget.reserve(key,cap,jhash(payload))
        # Durable send record survives killed sessions; the reservation remains.
        append_event(Path(target)/"send_attempts.jsonl", {"key":key,"seed_id":seed["seed_id"],"condition":condition,
            "max_tokens":cap,"request_sha256":jhash(payload),"started_at_unix":time.time()})
        return {"seed_id":seed["seed_id"],"task_type":seed["task_type"],"variant_kind":"trace_variant", "condition":condition,
            "key":key,"request":payload,"model":payload["model"],"request_sha256":jhash(payload),"prompt_sha256":jhash(seed["prompt"]),
            "schema_sha256":jhash(schema),"started_at_unix":time.time(),"automatic_retries":0}
    def finish(entry,started,response=None,error=None):
        entry["latency_seconds"]=time.perf_counter()-started
        if response is not None:
            raw=response.content.decode("utf-8",errors="replace")
            try: body=json.loads(raw)
            except ValueError: body={"unparsed_body":raw}
            choice=(body.get("choices") or [{}])[0] if isinstance(body,dict) else {}
            entry.update(http_status=response.status_code, raw_http_response_body=raw,
                response=body, finish_reason=choice.get("finish_reason"), usage=body.get("usage") if isinstance(body,dict) else None,
                status="completed" if response.status_code==200 else "failed")
        else: entry.update(http_status=None,status="failed",usage=None,error=type(error).__name__+": "+str(error))
        append_event(destination,entry)
        budget.settle(key,cap,entry.get("usage"))
    def sync(transport,request):
        entry=start(request); started=time.perf_counter()
        try: response=orig_sync(transport,request); response.read()
        except BaseException as e: finish(entry,started,error=e); raise
        finish(entry,started,response=response); return response
    async def async_handle(transport,request):
        entry=start(request); started=time.perf_counter()
        try: response=await orig_async(transport,request); await response.aread()
        except BaseException as e: finish(entry,started,error=e); raise
        finish(entry,started,response=response); return response
    with patch("data_designer.engine.models.clients.adapters.http_model_client.create_retry_transport",no_retry_transport), \
         patch.object(ModelRequestExecutor,"__init__",executor_init), \
         patch.object(httpx.HTTPTransport,"handle_request",sync), patch.object(httpx.AsyncHTTPTransport,"handle_async_request",async_handle):
        yield


def usage_fields(entry):
    usage=entry.get("usage") or {}
    return {"input_tokens":usage.get("prompt_tokens"), "completion_tokens":usage.get("completion_tokens"),
        "reasoning_tokens":(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "usage_status":"reported" if type(usage.get("completion_tokens")) is int else "unknown"}


def screen_entry(ns, seed, http):
    entry=http[-1] if http else {}
    # Empty usage is kept Unknown in cost accounting; the V2 wire audit expects
    # a mapping while checking response correspondence, and requires no usage.
    normalized=[{**e,"usage":e.get("usage") or {}} for e in http]
    record=bind_raw_response({k:seed[k] for k in ("seed_id","task_type","prompt")}, normalized, seed, ("trace_variant",))
    wire=ns["nvidia_wire_check_v2"](seed,record,"trace_variant", normalized)
    raw=record.get("trace_variant")
    proof=ns["verify_nvidia_trace_v2"](seed,raw)
    accepted=wire["verified"] and proof["verified"]
    return {"seed_id":seed["seed_id"],"task_type":seed["task_type"],"variant_kind":"trace_variant",
        "prompt":seed["prompt"],"variant_content":raw,"verification_status":"accepted" if accepted else "preview",
        "llm_proof_accepted":bool(accepted),"transport":wire,"verification":proof,
        "acceptance_protocol":ns["NVIDIA_ACCEPTANCE_PROTOCOL_V2"],"verifier_version":ns["NVIDIA_VERIFIER_V2"],
        "finish_reason":entry.get("finish_reason"),"http_status":entry.get("http_status"),
        "latency_seconds":entry.get("latency_seconds"),**usage_fields(entry),
        "thinking_present":bool(record.get("trace_variant__reasoning_content")), "thinking_verification_status":"unverified",
        "lineage_status":seed.get("lineage_status","unavailable"),
        "train_only_provenance_passed":seed.get("train_only_provenance"),
        "seed_gate_passed":seed.get("seed_gate_passed"),
        "sft_training_eligible":False,"merge_into_sft":False}


def high_baselines(c, ns):
    seeds=rows(ROOT/c["frozen_seed_source"])
    result=[]
    for seed in seeds:
        source=ROOT/(c["high_baseline_gravity"] if seed["task_type"]=="gravity" else c["high_baseline_r4"])
        http=[r for r in rows(source/"generation/http_requests.jsonl") if r.get("seed_id")==seed["seed_id"] and
              r.get("variant_kind")=="trace_variant" and r.get("http_status")==200]
        if not http: raise ValueError("Missing complete historical baseline")
        screened=screen_entry(ns,seed,[http[-1]])
        if not screened["llm_proof_accepted"]: raise ValueError("Historical high-thinking proof failed unchanged verifier")
        result.append({**screened,"condition":"historical_full_thinking","source_run":str(source.relative_to(ROOT)),
            "source_http_sha256":file_hash(source/"generation/http_requests.jsonl"),"historical_attempts_with_unknown_usage":True})
    return result


def check_live_gate(target):
    c, manifest=verify_run(target); verify_candidate_freeze(target)
    verify_seed_pool(target)
    test=read(Path(target)/"cpu_regression.json")
    if test["status"]!="passed" or test["source_sha256"]!=manifest["source_sha256"]: raise ValueError("Passing CPU regression required for these exact sources")
    if file_hash(Path(target)/"cpu_regression.log")!=test["log_sha256"]: raise ValueError("CPU regression log changed")
    require_api_key()
    return c


def call_one(target, c, ns, seed, phase, condition):
    key=phase+":"+condition+":"+seed["seed_id"]
    directory=Path(target)/phase/condition/seed["seed_id"]
    result_file=directory/"result.json"
    if result_file.exists(): return read(result_file)
    # An interrupted request is never sent again, even if no response was saved.
    if key in RequestBudget(target,c).state()["keys"]:
        return {"seed_id":seed["seed_id"],"task_type":seed["task_type"],"condition":condition,
            "verification_status":"preview","llm_proof_accepted":False,"interrupted_request_not_retried":True,
            "completion_tokens":None,"reasoning_tokens":None,"input_tokens":None,"usage_status":"unknown"}
    directory.mkdir(parents=True,exist_ok=False)
    public={k:seed[k] for k in ("seed_id","task_type","prompt")}
    write_jsonl_new(directory/"public_seed.jsonl",[public])
    builder=ns["build_final_data_designer"](directory/"public_seed.jsonl",c,seed["task_type"],condition)
    write_json_new(directory/"compiled_config.json",builder.build().model_dump(mode="json"))
    started=time.perf_counter(); error=None
    try:
        with single_request_capture(target,c,key,seed,condition,directory/"http_requests.jsonl"):
            sdk_generate(builder,directory/"sdk")
    except (BudgetStop,DuplicateRequestBlocked): raise
    except Exception as e: error=redact(type(e).__name__+": "+str(e))
    http=rows(directory/"http_requests.jsonl") if (directory/"http_requests.jsonl").exists() else []
    raw=bind_raw_response(public,[{**e,"usage":e.get("usage") or {}} for e in http],seed,("trace_variant",))
    write_jsonl_new(directory/"raw_model_outputs.jsonl",[raw])
    screened=screen_entry(ns,seed,http)
    screened.update(condition=condition,phase=phase,sdk_error=error,actual_http_requests=len(http),
        elapsed_seconds=time.perf_counter()-started)
    write_json_new(result_file,screened)
    print(json.dumps({k:screened.get(k) for k in ("seed_id","task_type","condition","verification_status","finish_reason","completion_tokens","reasoning_tokens","elapsed_seconds")}),flush=True)
    return screened


def med(values):
    return statistics.median(values) if values and all(type(v) in (int,float) for v in values) else None


def summarize_proofs(records):
    out={}
    for task in TASKS:
        group=[r for r in records if r["task_type"]==task]
        good=[r for r in group if r.get("llm_proof_accepted")]
        def total(field):
            vals=[r.get(field) for r in group]
            return sum(vals) if vals and all(type(v) in (int,float) for v in vals) else None
        completion=total("completion_tokens")
        out[task]={"outputs":len(group),"accepted":len(good),"preview":len(group)-len(good),
            "acceptance_rate":len(good)/len(group) if group else None,
            "schema_pass":sum(r.get("verification",{}).get("schema_valid",False) for r in group),
            "math_pass":sum(r.get("verification",{}).get("verified",False) for r in group),
            "completion_tokens":completion,"reasoning_tokens":total("reasoning_tokens"),"input_tokens":total("input_tokens"),
            "mean_completion_tokens":completion/len(group) if completion is not None else None,
            "mean_reasoning_tokens":total("reasoning_tokens")/len(group) if total("reasoning_tokens") is not None else None,
            "median_accepted_reasoning_tokens":med([r.get("reasoning_tokens") for r in good]),
            "median_accepted_completion_tokens":med([r.get("completion_tokens") for r in good]),
            "tokens_per_accepted_proof":completion/len(good) if completion is not None and good else None,
            "median_latency_seconds":med([r.get("latency_seconds") for r in group]),
            "unknown_completion_usage":sum(type(r.get("completion_tokens")) is not int for r in group)}
    return out


def select_conditions(c, baselines, results):
    baseline=summarize_proofs(baselines)
    summaries={arm:summarize_proofs([r for r in results if r["condition"]==arm]) for arm in ("A","B")}
    decisions={}
    for task in TASKS:
        a,b=summaries["A"][task],summaries["B"][task]
        if a["accepted"]!=b["accepted"]:best="A" if a["accepted"]>b["accepted"] else "B"
        elif (a["median_accepted_reasoning_tokens"] is not None and b["median_accepted_reasoning_tokens"] is not None
              and a["median_accepted_reasoning_tokens"]!=b["median_accepted_reasoning_tokens"]):
            best="A" if a["median_accepted_reasoning_tokens"]<b["median_accepted_reasoning_tokens"] else "B"
        else:
            ca=a["median_accepted_completion_tokens"];cb=b["median_accepted_completion_tokens"]
            best="A" if (ca if ca is not None else float("inf"))<=(cb if cb is not None else float("inf")) else "B"
        picked=summaries[best][task]
        high=baseline[task]["median_accepted_reasoning_tokens"]; low=picked["median_accepted_reasoning_tokens"]
        reduction=1-low/high if high and low is not None else None
        gate=picked["outputs"]==3 and picked["accepted"]>=2 and reduction is not None and reduction>=.2
        decisions[task]={"selected_condition":best,"frozen_accepted":picked["accepted"],"baseline_median_reasoning_tokens":high,
            "low_median_accepted_reasoning_tokens":low,"reasoning_reduction":reduction,"new_seed_llm_gate_passed":gate,
            "production_route_if_gate_fails":"solver_first", "verifier_standard_unchanged":True}
    return {"baseline":baseline,"conditions":summaries,"decisions":decisions}


def calibrate(target):
    target=Path(target); c=check_live_gate(target); ns=load_definitions()
    if (target/"calibration_summary.json").exists(): raise FileExistsError("Calibration already complete")
    seeds=rows(ROOT/c["frozen_seed_source"])
    if len(seeds)!=9 or {t:sum(s["task_type"]==t for s in seeds) for t in TASKS}!={t:3 for t in TASKS}: raise ValueError("Frozen nine seeds changed")
    baselines=high_baselines(c,ns)
    print(json.dumps({"estimated_requests":18,"new_seed_maximum_requests":30,"max_tokens":c["max_tokens"],
        "completion_token_budget":500000,"max_http_requests":48,"high_baseline":summarize_proofs(baselines),
        "conditions":{"A":"full prompt + bounded precision + low", "B":"compact prompt + low"}},ensure_ascii=False),flush=True)
    results=[]; status="completed"
    with live_lock(target):
        for arm in ("A","B"):
            for seed in seeds:
                try: results.append(call_one(target,c,ns,seed,"calibration",arm))
                except BudgetStop as e: status="budget_exhausted:"+str(e); break
            if status!="completed":break
    write_jsonl_new(target/"high_baseline_proofs.jsonl",baselines)
    write_jsonl_new(target/"calibration_results.jsonl",results)
    summary=select_conditions(c,baselines,results)
    summary.update(status=status,ledger=RequestBudget(target,c).state(),independent_math_verifier="unchanged V2",merge_into_sft=False)
    write_json_new(target/"calibration_summary.json",summary)
    return summary


def run_new_seeds(target):
    target=Path(target); c=check_live_gate(target); ns=load_definitions()
    if (target/"new_seed_summary.json").exists():raise FileExistsError("New-seed results already complete")
    calibration=read(target/"calibration_summary.json"); manifest=read(target/"seed_manifest.json")
    for p,h in manifest["artifact_sha256"].items():
        if file_hash(target/p)!=h:raise ValueError("Audited Seed file changed")
    seeds=verify_seed_pool(target); results=[];skipped={};status="completed"
    with live_lock(target):
        for task in TASKS:
            decision=calibration["decisions"][task]
            if not decision["new_seed_llm_gate_passed"]:
                skipped[task]="frozen_quality_or_cost_gate_failed";continue
            task_seeds=[s for s in seeds if s["task_type"]==task]
            if not task_seeds:skipped[task]="no_seed_passed_strict_seed_gate"
            for seed in task_seeds:
                try:results.append(call_one(target,c,ns,seed,"new_seeds",decision["selected_condition"]))
                except BudgetStop as e:status="budget_exhausted:"+str(e);break
            if status!="completed":break
    write_jsonl_new(target/"llm_proof_accepted.jsonl",[r for r in results if r.get("llm_proof_accepted")])
    write_jsonl_new(target/"llm_proof_preview.jsonl",[r for r in results if not r.get("llm_proof_accepted")])
    write_json_new(target/"new_seed_summary.json",{"status":status,"by_task":summarize_proofs(results),"skipped":skipped,
        "ledger":RequestBudget(target,c).state(),"merge_into_sft":False})
    write_json_new(target/"llm_proof_manifest.json",{"status":"passed" if any(r.get("llm_proof_accepted") for r in results) else "empty_accepted",
        "run_status":status,"accepted_rows":sum(bool(r.get("llm_proof_accepted")) for r in results),
        "preview_rows":sum(not r.get("llm_proof_accepted") for r in results),"skipped":skipped,
        "artifact_sha256":{p:file_hash(target/p) for p in ("llm_proof_accepted.jsonl","llm_proof_preview.jsonl")},
        "prior_manifest_sha256":manifest["prior_manifest_sha256"], "merge_into_sft":False})
    return read(target/"new_seed_summary.json")


def render(target):
    target=Path(target); c,manifest=verify_run(target); verify_candidate_freeze(target)
    if (target/"sft_render_manifest.json").exists():raise FileExistsError("SFT preview already exists")
    ns=load_definitions(); prior=read(target/"train_prior.json"); verified=[];elapsed={}
    for seed in verify_seed_pool(target):
        started=time.perf_counter(); proof=ns["solver_first_structured_proof"](seed)
        if not proof["solver_proof_accepted"]:raise ValueError("Accepted seed failed fresh Solver proof")
        verified.append({"candidate_id":seed["candidate_id"],"seed_id":seed["seed_id"],"task_type":seed["task_type"],
            "prompt":seed["prompt"],**proof,"elapsed_seconds":time.perf_counter()-started})
    write_jsonl_new(target/"solver_verified.jsonl",verified)
    # A separate CPU interpreter supplies the locally installed tokenizer. It
    # does not load model weights or CUDA, and cannot contact the network.
    command=[str(ROOT/c["tokenizer_python"]),str(Path(__file__)),"_render-tokenizer","--output",str(target)]
    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",CUDA_VISIBLE_DEVICES="",TOKENIZERS_PARALLELISM="false")
    env.pop("NVIDIA_API_KEY",None)
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    with (target/"tokenizer_render.log").open("x") as f:f.write(redact(result.stdout+result.stderr))
    if result.returncode:raise RuntimeError("Local tokenizer render failed: see tokenizer_render.log")
    return read(target/"sft_render_manifest.json")


def render_with_tokenizer(target):
    import socket
    from utils.run_nvidia_v2 import local_tokenizer
    target=Path(target); c,manifest=verify_run(target,check_sources=False,check_sdk=False)
    with patch.object(socket.socket,"connect",side_effect=AssertionError("No network during Solver-first rendering")):
        ns=load_definitions(); sft=load_renderer(); tokenizer,provenance=local_tokenizer()
        if tokenizer is None:raise ValueError("Local Qwen3-0.6B tokenizer missing; no word-count fallback")
        prior=read(target/"train_prior.json"); seeds=rows(target/"seed_accepted.jsonl")
        verified={r["candidate_id"]:r for r in rows(target/"solver_verified.jsonl")}
        llm={}
        for name in ("llm_proof_accepted.jsonl","llm_proof_preview.jsonl"):
            if (target/name).exists():llm.update({r["seed_id"]:r["llm_proof_accepted"] for r in rows(target/name)})
        records=[];audits=[]
        for seed in seeds:
            seed={**seed,"llm_proof_accepted":llm.get(seed["seed_id"])}
            record,audit=sft["render_solver_first_preview"](seed,verified[seed["candidate_id"]],ns,prior,
                manifest["train_prior_manifest_sha256"],sft_training_tasks(),tokenizer)
            records.append(record);audits.append(audit)
        write_jsonl_new(target/"sft_rendered_preview.jsonl",records)
        write_json_new(target/"sft_render_audit.json",{"status":"passed" if records and all(a["sft_render_audit_passed"] for a in audits) else "empty_preview",
            "records":audits,"tokenizer":provenance,"current_training_task_pool":sft_training_tasks(),"merge_into_sft":False})
        write_json_new(target/"sft_render_manifest.json",{"status":"preview_only","source":"solver_generated_rendered",
            "input_seed_manifest_sha256":file_hash(target/"seed_manifest.json"),"prior_manifest_sha256":manifest["train_prior_manifest_sha256"],
            "artifact_sha256":{name:file_hash(target/name) for name in ("solver_verified.jsonl","sft_rendered_preview.jsonl","sft_render_audit.json")},
            "by_task":{t:{"solver_verified":sum(r["task_type"]==t for r in records),
                "renderer_passed":sum(r["task_type"]==t and r["sft_render_audit_passed"] for r in records),
                "training_eligible":sum(r["task_type"]==t and r["sft_training_eligible"] for r in records)} for t in TASKS},
            "tokenizer":provenance,"merge_into_sft":False,"finally_entered_sft":0,"gpu_used":False})


def isolation_audit(target):
    target=Path(target); c,manifest=verify_run(target)
    changed=[]
    for name,before in manifest["history_before"].items():
        if tree_hashes(ROOT/name)!=before:changed.append(name)
    for name,before in manifest["protected_sha256"].items():
        if file_hash(ROOT/name)!=before:changed.append(name)
    forbidden=[]; secret=os.environ.get("NVIDIA_API_KEY","").encode()
    for p in target.rglob("*"):
        if not p.is_file():continue
        body=p.read_bytes()
        if b"nvapi-" in body or (secret and secret in body):forbidden.append(str(p.relative_to(target)))
        if re.search(rb'"(?:authorization|api_key)"\s*:',body,re.I):
            # Compiled built-in registry can name an env var, never a secret.
            try:
                text=body.decode(); value=json.loads(text)
                def walk(x):
                    if isinstance(x,dict):
                        for k,v in x.items():
                            if k.lower()=="authorization" or k.lower()=="api_key" and v!="NVIDIA_API_KEY":return True
                            if walk(v):return True
                    elif isinstance(x,list):return any(walk(v) for v in x)
                    return False
                if walk(value):forbidden.append(str(p.relative_to(target)))
            except (ValueError,UnicodeDecodeError):forbidden.append(str(p.relative_to(target)))
    return {"status":"passed" if not changed and not forbidden else "failed","changed_protected_paths":changed,
        "secret_or_auth_header_files":sorted(set(forbidden)),"historical_runs_checked":len(manifest["history_before"]),
        "gpu_training":False,"merge_into_sft":False,"dpo_ppo_grpo_unchanged":not any(p in changed for p in PROTECTED)}


def report(target):
    target=Path(target); c,manifest=verify_run(target)
    ns=load_definitions(); baseline=summarize_proofs(high_baselines(c,ns))
    optional=lambda name:read(target/name) if (target/name).exists() else None
    summary={"run_id":c["run_id"],"git_commit":manifest["git_commit"],"config_sha256":manifest["config_sha256"],
        "lineage":optional("train_only_lineage_audit.json"),"seed_manifest":optional("seed_manifest.json"),
        "cpu_regression":optional("cpu_regression.json"),"high_baseline":baseline,"calibration":optional("calibration_summary.json"),
        "new_seeds":optional("new_seed_summary.json"),"sft":optional("sft_render_manifest.json"),
        "api_budget":RequestBudget(target,c).state(),"isolation":isolation_audit(target),
        "merge_into_sft":False,"finally_entered_sft":0,"gpu_training":False}
    summary["run_status"]=("completed" if summary["calibration"] and summary["new_seeds"] and summary["sft"] else "incomplete_api_not_run")
    if summary["isolation"]["status"]!="passed":summary["run_status"]="failed_isolation"
    write_json_new(target/"summary.json",summary)
    write_final_report(target,summary,c)
    return summary


def write_final_report(target,s,c):
    """Evidence-backed Markdown; Unknown usage is never printed as zero."""
    from collections import Counter
    target=Path(target); rel=str(target.relative_to(ROOT))
    show=lambda value: "Unknown" if value is None else f"{value:.3f}" if isinstance(value,float) else str(value)
    table=lambda headers,body: "\n".join(["| "+" | ".join(headers)+" |", "|"+"|".join(["---"]*len(headers))+"|"]+
        ["| "+" | ".join(map(show,row))+" |" for row in body])+"\n"
    lines=["# 三题型合成数据最终 Pilot：来源、成本与 SFT 预览", "",
        f"Run：`{c['run_id']}`；代码提交：`{s['git_commit']}`；配置 SHA256：`{s['config_sha256']}`。",
        "全部本轮产物位于 [独立 Run 目录](../"+rel+"/)。原 R4、V2 重审和 Gravity P16K/P32K 均保持原 Hash。",
        "本轮没有 GPU 训练、SFT 合并或 DPO/PPO/GRPO 修改。", "", "## 执行状态", "",
        "实验状态：`"+s["run_status"]+"`。"]
    if not s["calibration"]:
        lines += ["当前执行环境缺少 `NVIDIA_API_KEY`，真实 API 阶段未执行。代码与 Mock 通过不能说明 Ultra 低推理生成质量通过。",
            "低推理 A/B、新 Seed LLM 接受率与成本下降均为 **未测**；没有使用历史中断的 Low Thinking 结果填补。"]
    lines += ["", "## A. Train-only 来源证明", "",
        "Generator/Train Prior/Prompt/Solver/Verifier 位于 Notebook 01；Full Trace Renderer 位于 Notebook 03。Runner 仅调度、归档、计算指标和约束请求。",
        "专用定义加载器不执行 Notebook 的数据配置 Cell，生成进程通过文件读取边界只允许读取注册 Manifest、Train CSV 以及本轮已冻结的中间产物。Train CSV 只解析 `id/source_row/task_type/prompt/split`；不解析 answer。原始文件整体 SHA256 用于完整性校验，不将 answer 内容用作采样依据。",
        "`prepare_read_audit.json` 和 `candidate_pool_freeze.json` 保存实际读取记录。`audit_started.json` 在候选冻结后才创建，然后独立进程加载真实三集合。没有根据 Holdout 重新采样。", "",
        table(["题型","Train 行数","经验采样特征","固定 Task Specification"],[
            ["Roman",1260,"公开示例及 Query 数字的出现频率；每题示例数；题内数字不重复","规范 Roman 1..3999"],
            ["Gravity",1277,"Train OLS g、示例时间、示例数、原始字符串距离精度","d=0.5gt²、OLS、最终两位小数"],
            ["Equation",586,"联合行布局：运算符/示例数/每运算符示例数/Query 运算符/Deduce-Guess；经验操作数对","Numeric24、family/group 全局约束及合法规则采样"]]),
        "Roman 使用经验出现次数加权抽样；Gravity 每个参数对应一条观测来源。Equation 从经验布局联合抽样，以保留布局和 Query 类型的关联；左右操作数从经验操作数对抽样。运算符频率和左右边缘分布另外审计，不声称是独立采样。Train 中 Equation 的 Deduce/Guess 为 475/111；Guess 子型从公开 Prompt 的独立验证结果标记，不读取官方答案。",
        "旧 Generator 的全局运算符回退已移除；Train 支持为空立即报错。新增 Gravity 显式采样参数入口复用原渲染计算，默认旧采样路径保留。所有原有 Solver 和数学 Verifier 的计算未改写。",
        "[Prior Manifest](../"+rel+"/train_prior_manifest.json) 保存数据/源码/规范 Hash、每特征规则、ID 和行号列表 Hash、RNG 20261115 与时间。[完整 Prior](../"+rel+"/train_prior.json) 保存每一次经验观测的 Train 来源。每候选的 `prior_feature_lineage` 记录实际抽中的特征、值、来源 ID、source_row 和观测位置。",
        "参数祖先 `prior_source_ids/rows` 与克隆真实样本的 `source_id/source_row` 分开；前者由 `audit_train_prior_lineage` 验证，后者继续由原三集合重复审计检查。结构相似性仅报告。",
        "Roman 采用严格政策：Validation/Holdout 的相同目标映射、或者合成示例暴露这些真实目标映射，均拒绝训练资格；Query 命中真实公开示例单独报告。数学正确、Train 来源正确和无目标映射暴露是三个独立状态。", ""]
    lineage=s["lineage"] or {}; by=lineage.get("by_task",{})
    lines += [table(["题型","候选","来源通过","Solver Proof 通过","Leakage 通过","全部门槛通过","选入 Seed"],
        [[t]+[by.get(t,{}).get(k) for k in ("candidates","lineage_pass","solver_pass","leakage_pass","all_seed_gates_pass","selected")] for t in TASKS]),
        "逐特征、逐样本来源审计见 [train_only_lineage_audit.json](../"+rel+"/train_only_lineage_audit.json)。", "", "## B. 固定新 Seed 生成结果", ""]
    candidates=rows(target/"seed_candidates.jsonl") if (target/"seed_candidates.jsonl").exists() else []
    lines += [table(["题型","20 条中选入","未选入","是否达到 10 条","拒绝/未选原因（可重叠）"],
        [[t,by.get(t,{}).get("selected"),sum(x["task_type"]==t and x["verification_status"]!="accepted" for x in candidates),
          by.get(t,{}).get("selected")==10,
          json.dumps(dict(Counter(reason for x in candidates if x["task_type"]==t for reason in x["rejection_reasons"])),ensure_ascii=False)] for t in TASKS]),
        "候选按固定 ID 顺序选取每题型前 10 条门槛通过样本；`passed_gates_not_selected_quota` 是配额外通过样本，不等同于数学或来源失败。没有补采样、放宽精度或改答案。完整候选、拒绝原因和 Hash 均归档。",
        "分布指标见 [train_distribution_audit.json](../"+rel+"/train_distribution_audit.json)：数值采用 KS statistic/Wasserstein，离散特征采用 Total Variation；分别报告 20 条原始候选与最终 Seed 面板。样本很小，指标仅用于描述抽样偏移，不构成总体分布一致性的证明。筛选也会带来面板分布偏移。", ""]
    dist=read(target/"train_distribution_audit.json") if (target/"train_distribution_audit.json").exists() else None
    if dist:
        drows=[]
        for t,value in dist["all_candidates"]["by_task"].items():
            for feature,stat in value["features"].items():
                drows.append([t,feature,stat.get("ks_statistic"),stat.get("wasserstein_distance"),stat.get("total_variation")])
        lines += [table(["题型","特征（全部候选）","KS","Wasserstein","TV"],drows)]
    lines += ["## C. 低推理成本对照", "",
        "Ultra `nvidia/nemotron-3-ultra-550b-a55b`；SDK 0.10.0 内置 NVIDIA Provider；temperature=1.0、top_p=.95、reasoning_effort=low、并发 1。Roman/Equation 上限 16384，Gravity 32768。A 使用原完整模板与 bounded_precision；B 压缩表达，保留原完整 Numeric24 目录/全局约束和闭合 Schema。Gravity 的原数值 Query 字符串合同两条件相同。",
        "仅发送 `seed_id/task_type/prompt`。隐藏规则、答案、Solver Trace、来源 Manifest 不发送。验收对象始终为原始 HTTP 对应的严格最终 JSON，Thinking 为 unverified 审计材料。未知 Usage 为 Unknown，并按请求 max_tokens 预留预算。",
        "最大 18 次冻结 Seed A/B 加最多 30 次新 Seed，共最多 48 次 HTTP；Completion 总硬预算 500000。发送前持久化记账；底层 RetryTransport 和 RequestExecutor 设置 0 重试，另加实际发送 Guard。失败/中断不重发，同一 Seed×条件不可重复。", ""]
    costrows=[]
    panels=[("历史 Full Thinking",s["high_baseline"])]
    if s["calibration"]:
        panels += [("Low Full A",s["calibration"]["conditions"]["A"]),("Low Compact B",s["calibration"]["conditions"]["B"])]
    else:
        panels += [("Low Full A",{}),("Low Compact B",{})]
    for label,panel in panels:
        for t in TASKS:
            x=panel.get(t,{})
            costrows.append([label,t,f"{x.get('accepted','未测')}/{x.get('outputs','未测')}",x.get("completion_tokens"),x.get("reasoning_tokens"),
                x.get("tokens_per_accepted_proof"),x.get("median_accepted_reasoning_tokens"),x.get("median_latency_seconds")])
    lines += [table(["方案","题型","数学 Accepted","Completion 合计","Reasoning 合计","Completion/Accepted","正确证明 Median Reasoning","Median latency 秒"],costrows),
        "历史 Roman/Equation 使用 R4 的原始成功响应，Gravity 使用 P32K 原始响应，不重新调用。历史成本表只统计已收到的 Proof 响应；旧实验失败/中断可能 Usage 未知，不能当作完整计费总量，也不能与新单次请求成本无条件等同。历史 R3 Low Thinking 中断不作为对照。", ""]
    if s["calibration"]:
        lines += [table(["题型","选择 Prompt","冻结正确证明","Reasoning 下降","允许新 Seed API"],
            [[t,d["selected_condition"],d["frozen_accepted"],d["reasoning_reduction"],d["new_seed_llm_gate_passed"]] for t,d in s["calibration"]["decisions"].items()])]
    if s["new_seeds"]:
        lines += ["新 Seed 单次低推理结果：", "",table(["题型","输出数","Accepted","接受率","平均 Completion","平均 Reasoning","Completion/Accepted","Median latency 秒"],
            [[t]+[s["new_seeds"]["by_task"][t].get(k) for k in ("outputs","accepted","acceptance_rate","mean_completion_tokens","mean_reasoning_tokens","tokens_per_accepted_proof","median_latency_seconds")] for t in TASKS]),
            "跳过原因：`"+json.dumps(s["new_seeds"]["skipped"],ensure_ascii=False)+"`。"]
    lines += ["", "实际请求/预算：`"+json.dumps(s["api_budget"],ensure_ascii=False)+"`。", "", "## D. Solver-first 与 SFT 产物", "",
        "所有通过新 Seed Gate 的样本独立重跑原 Python Solver，再以原独立 Verifier 检查闭合 Proof；正式预览使用 Solver Trace，不使用 LLM JSON 或隐藏 Thinking 作为 Completion。",
        "Notebook 03 的新增显式入口复用 `_format_trace()/make_trace_record()/audit_trace_record()`。输出为编号自然语言步骤加 `Final answer: \\boxed{...}`；来源明确标记 `solver_generated_rendered`，不声称为 Nemotron 自主 CoT。",
        "本地 Qwen3-0.6B Tokenizer 计算实际 Completion/Chat/Supervised Tokens；空白分词 proxy 只作为原 Audit 的历史字段，不冒充真实 Token 数。无权重加载、无网络、无 CUDA。", ""]
    rend=s["sft"] or {}; rt=rend.get("by_task",{})
    lines += [table(["题型","Solver Verified","Renderer Passed","Training Eligible","最终进入 SFT"],
        [[t,rt.get(t,{}).get("solver_verified"),rt.get(t,{}).get("renderer_passed"),rt.get(t,{}).get("training_eligible"),0] for t in TASKS])]
    preview=rows(target/"sft_rendered_preview.jsonl") if (target/"sft_rendered_preview.jsonl").exists() else []
    verified=rows(target/"solver_verified.jsonl") if (target/"solver_verified.jsonl").exists() else []
    lines += [table(["Solver-first 题型","CPU Proof 合计秒","平均渲染 Completion Tokens","Hosted Completion Tokens","LLM Reasoning Tokens"],
        [[t,sum(x["elapsed_seconds"] for x in verified if x["task_type"]==t),
          statistics.mean([x["actual_token_counts"]["completion_tokens"] for x in preview if x["task_type"]==t]) if any(x["task_type"]==t for x in preview) else None,0,0] for t in TASKS]),
        "Solver-first 的 0 指不发生任何 Hosted LLM 请求；渲染文本自身的 tokenizer 长度另列。",
        "Roman 不属于当前 `gravity/equation_numeric/bit_manipulation` 训练任务池；即使来源和映射检查通过，也只允许预览。本轮未通过严格映射 Gate 的 Roman 不伪装成合格 SFT 预览。",
        "四个状态严格分开：数学 Proof Accepted → SFT Format Audit Passed → SFT Training Eligible（待审核资格）→ 最终进入 SFT。本轮最后一项恒为 0，`merge_into_sft=false`，不修改默认 USE_SYNTHETIC_DATA 或历史 Checkpoint。", ""]
    if preview:
        lines += ["渲染样例（首条合格 Seed）：", "", "```text",preview[0]["completion"],"```",""]
    lines += ["## E. 最终生产决策与限制", "",
        "1. 三题型的 Train-only 生成能力以固定 20 条候选和逐特征来源审计为准；数学可生成不等于通过严格训练资格 Gate。Roman 如果目标映射暴露导致零入选，应保持零，不根据 Holdout 寻找避开的数字。",
        "2. 低推理效果最好的题型，只能从本轮完整冻结 A/B 与新 Seed 的真实结果判断；未调用 API 的题型不能排名。冻结门槛为 ≥2/3 数学接受且正确证明 Median Reasoning 比历史下降 ≥20%。未达门槛直接走 Solver-first，不自动升到 Medium/High。",
        "3. 哪些题型仍需要高推理成本，需要本轮低推理实测支撑。已有证据仅表明历史 Ultra 高推理三题型各 3/3，Gravity P32K 41492 Reasoning Tokens；不能把历史低推理中断当作失败或成功证据。",
        "4. 确定性 Roman/Gravity/Numeric24 的规范数学教师数据均适合 Solver-first；它有独立 Verifier、清晰来源和零 Hosted Reasoning 开销。LLM 路线保留为独立能力/成本对照，而非正确性的唯一来源。",
        "5. 可靠 SFT 候选数量严格以本报告 Renderer/Training Eligible 表为准；合格候选仍未自动合并。",
        "6. Gravity/Equation 仅在来源、数学、Leakage、真实 Token 长度和 SFT Audit 均通过的范围具备下一轮独立登记 20–50 条试点的工程基础。Roman 必须先明确严格映射政策与当前任务池的限制。本次不扩大，不基于当前 Holdout 拒绝原因调参或重采样。",
        "7. 正式数学教师生产无需 Ultra 重复执行已由确定性 Solver 完成的计算。只有真实低推理成本门槛和新 Seed 质量均达标，才有证据保留 Ultra 作为辅助教师；没有该证据时默认 Solver-first。",
        "10 条每题型仅是 Pilot 目标，不能解释为总体接受率、统计显著性或泛化保证。", "", "## CPU 回归与工程完整性", "",
        "本轮专用测试结果：`"+json.dumps(s["cpu_regression"],ensure_ascii=False)+"`。",
        "来源、Answer 不变性、读取隔离、全部候选、三题型正反例、唯一性、严格 Roman 政策、A/B 单变量、真实 SDK Mock 请求、无重试硬预算、Unknown Usage、Thinking 隔离、本地 Tokenizer、不可覆盖历史、SFT 默认行为及 DPO/PPO/GRPO 隔离均有 CPU 检查。",
        "相关旧回归的实际命令/日志另存本 Run。历史开发期间失败日志保留在 `development_cpu_attempts/`，不把修复前失败写成通过。",
        "隔离审计：`"+json.dumps(s["isolation"],ensure_ascii=False)+"`。", "", "## 复现命令", "", "```bash",
        ".venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py prepare",
        ".venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py generate",
        ".venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py audit",
        ".venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py cpu-tests",
        "# API 阶段要求 NVIDIA_API_KEY 已在进程环境中；不把 Key 放到命令或文件中",
        ".venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py calibrate",
        ".venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py run-new-seeds",
        ".venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py render",
        ".venv-data-designer/bin/python utils/run_nvidia_final_synthetic.py report",
        "```", "", "已存在阶段拒绝覆盖；中断后只读取已保存响应，不重新发送已登记请求。新实验应独立登记新配置和 Run ID；本 CLI 固定本轮 Run ID，防止无意扩大。", ""]
    text="\n".join(lines)
    for path in (target/"REPORT.md",ROOT/"docs/NVIDIA_SYNTHETIC_FINAL_20261115_REPORT.md"):
        with path.open("x") as f:f.write(text)


def main():
    os.environ.update(NEMO_TELEMETRY_ENABLED="false",MPLBACKEND="Agg")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","generate","audit","cpu-tests","calibrate","run-new-seeds","render","report","_render-tokenizer"))
    p.add_argument("--output",type=Path,default=ROOT/"data/processed/round3/nvidia_synthetic_final_20261115_r1")
    a=p.parse_args()
    function={"prepare":prepare,"generate":generate,"audit":audit,"cpu-tests":cpu_regression,"calibrate":calibrate,
        "run-new-seeds":run_new_seeds,"render":render,"report":report,"_render-tokenizer":render_with_tokenizer}[a.stage]
    result=function(a.output)
    if result is not None:print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
