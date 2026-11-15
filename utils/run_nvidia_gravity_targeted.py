"""Separate Gravity P16K/P32K preparation and future explicit live execution.

This task invokes prepare and CPU mock tests only. Mathematical/schema/prompt
definitions remain in Notebook 01. No automatic live requests or model fallback.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.run_synthetic_smoke import load_notebook, rows, read
from utils.run_nvidia_v2 import file_hash, write_json_new, write_jsonl_new, verify_historical_archive, check_seed_gates

CONFIG = ROOT / "configs/nvidia_gravity_targeted_v2.json"


def validated_config(path=CONFIG):
    c = read(path)
    original = ROOT / c["original_run"]
    verify_historical_archive(original)
    if c["schema_version"] != "nvidia-gravity-targeted/v2" or c["base_generation_config"] != read(original / "config.json"):
        raise ValueError("Keep the R4 generation configuration except the declared query_t addendum and token cap")
    if file_hash(original / "private_seeds.jsonl") != c["seed_source_sha256"]:
        raise ValueError("Gravity input is not from the frozen R4 seeds")
    if (c["expected_seeds"], c["expected_outputs_per_condition"], c["independent_rel_tol"], c["independent_abs_tol"],
            c["merge_into_sft"], c["gpu_training"], c["live_execution_this_task"]) != (3, 3, 1e-4, 1e-5, False, False, False):
        raise ValueError("Do not change the registered mathematical or execution boundaries")
    if set(c["conditions"]) != {"P16K", "P32K"} or [c["conditions"][k]["max_tokens"] for k in ("P16K", "P32K")] != [16384, 32768]:
        raise ValueError("Exactly two independent token-budget conditions are required")
    ids = [c["conditions"][k]["run_id"] for k in ("P16K", "P32K")]
    import re
    if len(set(ids)) != 2 or not all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
        raise ValueError("Distinct valid condition Run IDs required")
    if "nvapi-" in json.dumps(c):
        raise ValueError("API Key material forbidden in configuration")
    return c


def prepare(config_path=CONFIG, output=None):
    c = validated_config(config_path)
    output = Path(output) if output else ROOT / "data/processed/round3" / c["preparation_id"]
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / c["original_run"]
    seeds = [s for s in rows(source / "private_seeds.jsonl") if s["task_type"] == "gravity"]
    if len(seeds) != 3:
        raise ValueError("Exactly three frozen Gravity Seeds required")
    with tempfile.TemporaryDirectory() as temp:
        ns = load_notebook(c["preparation_id"], Path(temp) / "loader")
        gates = check_seed_gates(ns, seeds)
        for condition in ("P16K", "P32K"):
            target = output / c["conditions"][condition]["run_id"]
            target.mkdir()
            write_jsonl_new(target / "private_seeds.jsonl", seeds)
            write_jsonl_new(target / "public_seeds.jsonl", [{k: s[k] for k in ("seed_id", "task_type", "prompt")} for s in seeds])
            write_json_new(target / "seed_checks.json", gates)
            write_json_new(target / "config.json", c)
            builder = ns["build_nvidia_gravity_targeted_v2"](target / "public_seeds.jsonl", c, condition)
            write_json_new(target / "compiled_config.json", builder.build().model_dump(mode="json"))
            write_json_new(target / "prepared_manifest.json", {"status": "prepared_cpu_only", "condition": condition,
                "run_id": target.name, "models_called": 0, "expected_outputs": 3,
                "original_private_seed_sha256": file_hash(source / "private_seeds.jsonl"),
                "condition_private_seed_sha256": file_hash(target / "private_seeds.jsonl"),
                "frozen_files": {name: file_hash(target / name) for name in
                    ("private_seeds.jsonl", "public_seeds.jsonl", "seed_checks.json", "config.json", "compiled_config.json")},
                "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in
                    (ROOT / "01_Synthetic_Data_Engineering.ipynb", Path(__file__), ROOT / "utils/run_nvidia_v2.py",
                     ROOT / "utils/run_nvidia_synthetic_smoke.py", Path(config_path).resolve())},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "acceptance_protocol": ns["NVIDIA_ACCEPTANCE_PROTOCOL_V2"],
                "gate": "3/3 complete final JSON Proofs pass original independent math and leakage audit",
                "merge_into_sft": False})
    write_json_new(output / "preparation_summary.json", {"status": "prepared_cpu_only", "models_called": 0,
        "conditions": c["conditions"], "real_api_run_status": "not_run", "live_execution_this_task": False})
    return output


def run_prepared(target):
    """Future explicit command only; no live invocation in this repair task."""
    from utils.run_nvidia_synthetic_smoke import (require_api_key, sdk_generate, capture_requests,
                                                bind_raw_response, write_new, write_rows, redact)
    require_api_key()
    target = Path(target)
    manifest = read(target / "prepared_manifest.json")
    for name, expected in manifest["frozen_files"].items():
        if file_hash(target / name) != expected: raise ValueError("Prepared input changed: " + name)
    for name, expected in manifest["source_hashes"].items():
        if file_hash(ROOT / name) != expected: raise ValueError("Source changed; prepare a new experiment: " + name)
    c = validated_config(target / "config.json")
    condition = manifest["condition"]
    seeds = rows(target / "private_seeds.jsonl")
    generation = target / "generation"
    generation.mkdir(exist_ok=False)
    records, errors = [], []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        ns = load_notebook(target.name, Path(temp) / "loader")
        with capture_requests(generation / "http_requests.jsonl"):
            for i, seed in enumerate(seeds):
                seed_path = generation / f"public_seed{i}.jsonl"
                write_rows(seed_path, [{k: seed[k] for k in ("seed_id", "task_type", "prompt")}])
                builder = ns["build_nvidia_gravity_targeted_v2"](seed_path, c, condition)
                write_new(generation / f"compiled_seed{i}.json", builder.build().model_dump(mode="json"))
                fatal = False
                try:
                    outputs = sdk_generate(builder, generation / f"sdk_seed{i}")
                    write_rows(generation / f"sdk_outputs{i}.jsonl", outputs)
                except Exception as error:
                    errors.append({"seed_id": seed["seed_id"], "error": redact(str(error))})
                    fatal = any(r.get("http_status") in {400, 401, 403, 404, 410, 422} for r in
                                rows(generation / "http_requests.jsonl") if r.get("seed_id") == seed["seed_id"])
                finally:
                    records.append(bind_raw_response({k: seed[k] for k in ("seed_id", "task_type", "prompt")},
                        rows(generation / "http_requests.jsonl"), seed, ("trace_variant",)))
                if fatal: break  # Includes unsupported 32768; never change cap and retry silently.
        write_rows(generation / "raw_model_outputs.jsonl", records)
        http = rows(generation / "http_requests.jsonl")
        summary = ns["screen_nvidia_records_v2"](records, seeds, generation, read(target / "seed_checks.json"), http,
            {"condition": condition, "original_run_id": Path(c["original_run"]).name, "models_called": len(http)},
            variant_kinds=("trace_variant",))
        thinking = sum(bool(r.get("trace_variant__reasoning_content")) for r in records)
        complete_proofs = summary["accepted"] == 3 and all(r.get("trace_variant__transport", {}).get("finish_reason") == "stop" for r in records)
        result = {"run_status": summary["run_status"], "quality": summary, "condition": condition, "errors": errors,
            "engineering_gate_passed": complete_proofs and thinking == 3, "thinking_present": thinking,
            "thinking_verification_status": "unverified_audit_only", "elapsed_seconds": time.perf_counter() - started,
            "actual_http_requests": len(http), "failed_requests": sum(r.get("http_status") != 200 for r in http),
            "output_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in http),
            "reasoning_tokens": sum(r.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) for r in http),
            "merge_into_sft": False}
        write_new(generation / "result.json", result)
        return result


def main():
    os.environ["NEMO_TELEMETRY_ENABLED"] = "false"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        print(prepare(args.config, args.output))
    else:
        if args.prepared is None: parser.error("run requires an explicit --prepared directory")
        result = run_prepared(args.prepared)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["engineering_gate_passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
