"""Frozen round-two seeds -> official NeMo NVIDIA provider, CPU/local control only.

All generation contracts and mathematical checks live in Notebook 01. HTTPX
instrumentation observes SDK wire attempts without replacing its provider,
authentication, payload generation, retry policy or response parsing.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.run_synthetic_smoke import NOTEBOOK, digest, load_notebook, read, rows

DEFAULT_CONFIG = ROOT / "configs/nvidia_synthetic_smoke.json"
ENDPOINT = "https://integrate.api.nvidia.com/v1"
TASKS = ("numeral_system", "gravity", "equation_numeric")
SOURCE_FILES = (NOTEBOOK, Path(__file__), ROOT / "utils/run_synthetic_smoke.py",
                ROOT / "configs/nvidia_synthetic_smoke.json", ROOT / "configs/nvidia_synthetic_smoke_ultra.json",
                ROOT / "configs/nvidia_synthetic_smoke_ultra_longthinking.json",
                ROOT / "configs/nvidia_synthetic_smoke_ultra_lowthinking.json",
                ROOT / "configs/nemo_smoke_requirements.lock.txt")


def redact(value):
    """Never archive request headers, literal keys, or keys reflected in errors."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()
                if k.lower() not in {"authorization", "api_key", "x-api-key", "proxy-authorization"}}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        secret = os.environ.get("NVIDIA_API_KEY")
        if secret:
            value = value.replace(secret, "[REDACTED]")
        return re.sub(r"nvapi-[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(redact(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, values):
    with Path(path).open("x") as stream:
        for value in values:
            stream.write(json.dumps(redact(value), ensure_ascii=False, allow_nan=False) + "\n")


def require_api_key():
    if not os.environ.get("NVIDIA_API_KEY", "").strip():
        raise RuntimeError("NVIDIA_API_KEY is missing in this process environment; no API request was made")


def validated_config(path=DEFAULT_CONFIG):
    c = read(path)
    if (c["schema_version"], c["data_designer_version"], c["provider"], c["endpoint"], c["api_key_env"]) != (
            "nvidia-hosted-synthetic/v1", "0.10.0", "nvidia", ENDPOINT, "NVIDIA_API_KEY"):
        raise ValueError("Use the registered official NVIDIA provider and pinned SDK")
    if (c["expected_seed_rows"], c["expected_model_outputs"], c["concurrency"], c["merge_into_sft"]) != (9, 18, 1, False):
        raise ValueError("Expected 9 frozen seeds, 18 outputs, concurrency one, no SFT merge")
    if set(c["models"]) != {"nvidia-text", "nvidia-reasoning"}:
        raise ValueError("Both registered aliases are required")
    from data_designer.config.utils.constants import PREDEFINED_PROVIDERS_MODEL_MAP
    for alias, kind in (("nvidia-text", "text"), ("nvidia-reasoning", "reasoning")):
        official = PREDEFINED_PROVIDERS_MODEL_MAP["nvidia"][kind]
        model = c["models"][alias]
        if set(model) != {"model", "temperature", "top_p", "max_tokens", "extra_body"}:
            raise ValueError("No custom model authentication or provider overrides")
        allowed_models = {official["model"]}
        if alias == "nvidia-reasoning":
            # User-authorized alternative; same inference parameters are an
            # experimental control, not a claim about Ultra's official defaults.
            allowed_models.add("nvidia/nemotron-3-ultra-550b-a55b")
        allowed_inference = [official["inference_parameters"]]
        if model["model"] == "nvidia/nemotron-3-ultra-550b-a55b":
            low = copy.deepcopy(official["inference_parameters"])
            low["extra_body"]["reasoning_effort"] = "low"
            allowed_inference.append(low)
        if model["model"] not in allowed_models or not any(all(model.get(k) == v for k, v in params.items())
                                                           for params in allowed_inference):
            raise ValueError("Use a registered model and the SDK 0.10.0 baseline inference settings")
        if type(model["max_tokens"]) is not int or model["max_tokens"] <= 0:
            raise ValueError("max_tokens must be a positive integer")
    if importlib.metadata.version("data-designer") != "0.10.0":
        raise ValueError("Do not upgrade data-designer; use 0.10.0")
    # Neither key material nor arbitrary provider/auth overrides are configuration.
    allowed = set(read(DEFAULT_CONFIG))
    if set(c) != allowed or "nvapi-" in json.dumps(c):
        raise ValueError("Unexpected configuration fields or credential material")
    return c


def check_seeds(config):
    base = ROOT / "data/processed/round2/real_gpu_20261114_phasec"
    paths = {arm: base / f"nemo_{arm}/private_seeds.jsonl" for arm in ("R0", "A", "B", "C")}
    hashes = {arm: digest(p) for arm, p in paths.items()}
    if len(set(hashes.values())) != 1 or hashes["R0"] != config["seed_sha256"]:
        raise ValueError("R0/A/B/C frozen Seed bytes differ")
    source = ROOT / config["seed_source"]
    if digest(source) != hashes["R0"]:
        raise ValueError("Configured source is not the frozen R0 input")
    seeds = rows(source)
    expected = {(task, f"{task}-seed{i}") for task in TASKS for i in (1, 2, 3)}
    if len(seeds) != 9 or {(s["task_type"], s["seed_id"]) for s in seeds} != expected:
        raise ValueError("Require exactly the original three seeds per task")
    return seeds, hashes


def seed_gates(ns, seeds):
    checks = {}
    for seed in seeds:
        solved = copy.deepcopy(seed)
        solver = ns["_solve_candidate"](solved)
        verification = ns["_screen_verification"](solved)
        leakage = ns["audit_synthetic_leakage"](seed, ns["real_index"])
        solver_passed = bool(ns["numeric_or_text_match"](solved["answer"], seed["answer"]))
        passed = bool(solver_passed and verification.get("answer_ok") and verification.get("trace_ok") and
                      verification.get("identifiability", {}).get("answer_unique") and leakage["status"] == "passed")
        checks[seed["seed_id"]] = {"passed": passed, "solver_passed": solver_passed, "solver_result": solver,
                                  "verification": verification, "leakage_audit": leakage}
    if not all(check["passed"] for check in checks.values()):
        raise ValueError("Frozen seeds did not pass the existing Solver/Verifier/leakage gate")
    return checks


def prepare(config=DEFAULT_CONFIG, run_id=None, target=None):
    c = validated_config(config)
    run_id = run_id or c["synthetic_run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid Run ID")
    target = Path(target) if target else ROOT / "data/processed/round3" / run_id
    target.mkdir(parents=True, exist_ok=False)
    seeds, hashes = check_seeds(c)
    ns = load_notebook(run_id, target / "loader")
    checks = seed_gates(ns, seeds)
    # Preserve the exact private input bytes; public files are projections only.
    with (target / "private_seeds.jsonl").open("xb") as stream:
        stream.write((ROOT / c["seed_source"]).read_bytes())
    public = [{k: s[k] for k in ("seed_id", "task_type", "prompt")} for s in seeds]
    write_rows(target / "public_seeds.jsonl", public)
    write_new(target / "seed_checks.json", checks)
    write_new(target / "config.json", {**c, "synthetic_run_id": run_id})
    for task in TASKS:
        shard = target / "prepared" / task
        shard.mkdir(parents=True)
        write_rows(shard / "public_seeds.jsonl", [s for s in public if s["task_type"] == task])
        builder = ns["build_nvidia_data_designer"](shard / "public_seeds.jsonl", c, task)
        write_new(shard / "compiled_config.json", builder.build().model_dump(mode="json"))
    frozen = {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob("*")) if p.is_file()}
    manifest = {"schema_version": "nvidia-hosted-prepared/v1", "status": "prepared_cpu_only", "run_id": run_id,
        "models_called": 0, "seed_rows": 9, "expected_outputs": 18, "historical_seed_hashes": hashes,
        "private_seed_sha256": digest(target / "private_seeds.jsonl"),
        "public_fields": ["seed_id", "task_type", "prompt"], "frozen_files": frozen,
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in SOURCE_FILES},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_revision": "hosted provider does not expose immutable weight revisions; record response model per request",
        "merge_into_sft": False, "prepared_at_unix": time.time()}
    write_new(target / "seed_manifest.json", manifest)
    return target


def verify_prepared(target):
    target = Path(target)
    manifest = read(target / "seed_manifest.json")
    for name, expected in manifest["frozen_files"].items():
        if digest(target / name) != expected:
            raise ValueError("Frozen artifact changed: " + name)
    for name, expected in manifest["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source changed; register a new Run: " + name)
    c = validated_config(target / "config.json")
    seeds, _ = check_seeds(c)
    public = rows(target / "public_seeds.jsonl")
    if public != [{k: s[k] for k in manifest["public_fields"]} for s in seeds]:
        raise ValueError("Public projection mismatch")
    return c, seeds


@contextlib.contextmanager
def capture_requests(destination):
    """Observe actual HTTPX transport attempts (including SDK retries), no headers.

    This narrow hook is version-tested against the pinned SDK. Original HTTPS
    transports still send every request directly to NVIDIA. No proxy/provider.
    """
    import httpx
    lock = threading.Lock()
    original_sync = httpx.HTTPTransport.handle_request
    original_async = httpx.AsyncHTTPTransport.handle_async_request
    with Path(destination).open("x") as stream:
        def start(request):
            if str(request.url) != ENDPOINT + "/chat/completions":
                raise RuntimeError("Unexpected endpoint; NVIDIA smoke forbids fallback")
            payload = json.loads(request.content)
            text = str(payload.get("messages", []))
            seed = re.search(r"SEED_ID=([^\\\s'\"]+)", text)
            kind = "prompt_variant" if "VARIANT_KIND=prompt_variant" in text else "trace_variant"
            return {"request": payload, "model": payload.get("model"), "seed_id": seed.group(1) if seed else None,
                    "variant_kind": kind, "started_at_unix": time.time()}

        def finish(entry, started, response=None, error=None):
            entry["latency_seconds"] = time.perf_counter() - started
            if response is not None:
                try:
                    body = response.json()
                except ValueError:
                    body = {"unparsed_body": response.text}
                choice = (body.get("choices") or [{}])[0]
                entry.update(http_status=response.status_code, response=body, finish_reason=choice.get("finish_reason"),
                             usage=body.get("usage", {}), status="completed" if response.status_code == 200 else "failed")
            else:
                entry.update(status="failed", http_status=None, error=type(error).__name__ + ": " + str(error))
            with lock:
                stream.write(json.dumps(redact(entry), ensure_ascii=False) + "\n")
                stream.flush()

        def handle_sync(transport, request):
            entry = start(request)
            started = time.perf_counter()
            try:
                response = original_sync(transport, request)
                response.read()
            except BaseException as error:
                finish(entry, started, error=error)
                raise
            finish(entry, started, response)
            return response

        async def handle_async(transport, request):
            entry = start(request)
            started = time.perf_counter()
            try:
                response = await original_async(transport, request)
                await response.aread()
            except BaseException as error:
                finish(entry, started, error=error)
                raise
            finish(entry, started, response)
            return response

        with patch.object(httpx.HTTPTransport, "handle_request", handle_sync), patch.object(
                httpx.AsyncHTTPTransport, "handle_async_request", handle_async):
            yield


def telemetry_summary(entries, config=None):
    by_model = {}
    for alias, model in (config or read(DEFAULT_CONFIG))["models"].items():
        subset = [e for e in entries if e.get("model") == model["model"]]
        groups = {(e.get("seed_id"), e.get("variant_kind")) for e in subset}
        usage = {}
        for entry in subset:
            for name, value in entry.get("usage", {}).items():
                if type(value) in (int, float):
                    usage[name] = usage.get(name, 0) + value
        by_model[alias] = {"model": model["model"], "http_requests": len(subset),
            "failed_requests": sum(e["status"] != "completed" for e in subset),
            "retries": len(subset) - len(groups), "usage": usage,
            "latency_seconds": sum(e["latency_seconds"] for e in subset)}
    return {"http_requests": len(entries), "failed_requests": sum(e["status"] != "completed" for e in entries),
            "by_model": by_model, "output_tokens_include_reasoning": True}


def new_designer(path):
    from data_designer.interface import DataDesigner
    from data_designer.engine.secret_resolver import EnvironmentResolver
    from data_designer.config import RunConfig
    designer = DataDesigner(artifact_path=path, secret_resolver=EnvironmentResolver(), auto_configure_logging=False)
    provider = designer.model_provider_registry.get_provider("nvidia")
    if (provider.endpoint, provider.api_key, provider.provider_type) != (ENDPOINT, "NVIDIA_API_KEY", "openai"):
        raise ValueError("Built-in NVIDIA provider is overridden locally; restore official SDK defaults")
    designer.set_run_config(RunConfig(max_in_flight_tasks=1, max_concurrent_row_groups=1,
        max_conversation_restarts=0, max_conversation_correction_steps=0, async_trace=True,
        otel_metrics_port=None))
    return designer


def sdk_generate(builder, path):
    """Persist sanitized SDK diagnostics even on provider/schema exceptions."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    logger = logging.getLogger("data_designer")
    previous_propagate = logger.propagate
    logger.propagate = False
    logger.addHandler(handler)
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            designer = new_designer(path / "sdk")
            designer.validate(builder)
            try:
                return designer.preview(config_builder=builder, num_records=1).dataset.to_dict(orient="records")
            except Exception as error:
                from data_designer.interface.errors import DataDesignerProfilingError
                if (not isinstance(error, DataDesignerProfilingError) or
                        not isinstance(error.__cause__, TypeError) or "NAType" not in str(error.__cause__)):
                    raise
                # SDK 0.10.0's singleton token std is pandas.NA; generation has
                # already completed. We never recover/rewrite an SDK proof. The
                # caller screens the raw wire content under all original gates.
                write_new(path / "profiling_warning.json", {"status": "sdk_singleton_profiling_failed",
                    "error": str(error), "sdk_dataset_available": False,
                    "wire_output_requires_full_independent_validation": True})
                return []
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous_propagate
        with (path / "sdk.log").open("x") as stream:
            stream.write(redact(buffer.getvalue()))


def bind_raw_response(record, requests, seed, kinds=("prompt_variant", "trace_variant")):
    """Screen raw HTTP content, never the SDK-repaired/structured value."""
    for kind in kinds:
        attempts = [e for e in requests if e.get("seed_id") == seed["seed_id"] and e.get("variant_kind") == kind]
        success = [e for e in attempts if e["status"] == "completed"]
        last = success[-1] if success else {}
        choice = (last.get("response", {}).get("choices") or [{}])[0]
        message = choice.get("message", {})
        raw = message.get("content")
        # Provenance comes from the actual request and original public seed.
        record[kind + "__transport"] = {"verified_response_match": bool(last),
            "request_count": len(attempts), "finish_reason": choice.get("finish_reason"),
            "usage": last.get("usage"), "latency_seconds": last.get("latency_seconds")}
        record[kind + "__reasoning_content"] = message.get("reasoning_content")
        if kind == "prompt_variant":
            record["introduction"] = raw
            record[kind] = raw + "\n" + seed["prompt"] if isinstance(raw, str) else None
        else:
            record[kind] = raw
    return record


def schema_probe_object(task):
    """Transport/shape fixture only; never a formal seed or mathematical sample."""
    obj = {"task_type": task, "answer": ""}
    if task == "numeral_system":
        obj.update(query=1, components=[{"arabic": 1, "roman": "I"}])
    elif task == "gravity":
        obj.update(g_estimates=[1], g_ols=1, query_t="1", query_distance=.5)
    elif task == "equation_numeric":
        obj.update(rules={}, example_outputs=[], query_output="")
    else:
        raise ValueError(task)
    return obj


def api_smoke(target):
    require_api_key()  # Before any SDK construction, HTTP calls, or stage creation.
    c, seeds = verify_prepared(target)
    stage = Path(target) / "api_smoke"
    stage.mkdir(exist_ok=False)
    ns = load_notebook(Path(target).name, stage / "loader")
    results = []
    started = time.perf_counter()
    with capture_requests(stage / "http_requests.jsonl"):
        # One Lightning request; one Super request for each actual task schema.
        probes = [("nvidia-text", TASKS[0], "prompt_variant")] + [
            ("nvidia-reasoning", task, "trace_variant") for task in TASKS]
        for index, (alias, task, kind) in enumerate(probes):
            seed = next(s for s in seeds if s["task_type"] == task)
            seed_file = stage / f"probe{index}_public_seed.jsonl"
            public = {k: seed[k] for k in ("seed_id", "task_type", "prompt")}
            if kind == "trace_variant":
                # A minimal API/shape check should not spend a full arithmetic
                # rollout. This fixture is explicitly outside the frozen 9
                # mathematical Seeds and never enters quality statistics.
                public["seed_id"] = "api-schema-" + task
                public["prompt"] = "API shape probe only; not a mathematical training sample."
            write_rows(seed_file, [public])
            builder = ns["build_nvidia_data_designer"](seed_file, c, task, kind)
            if kind == "trace_variant":
                import data_designer.config as dd
                compiled = builder.build()
                minimal = ("API_SCHEMA_ONLY: briefly check the required JSON fields, then return exactly this "
                    "shape fixture. This is a transport test, not a mathematics sample or proof.\n" +
                    json.dumps(schema_probe_object(task)) + "\nSEED_ID={{ seed_id }}\nVARIANT_KIND=trace_variant")
                builder = dd.DataDesignerConfigBuilder(model_configs=list(compiled.model_configs))
                builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_file)), sampling_strategy=dd.SamplingStrategy.ORDERED)
                builder.add_column(compiled.columns[0].model_copy(update={"prompt": minimal}))
            write_new(stage / f"probe{index}_compiled_config.json", builder.build().model_dump(mode="json"))
            result = {"alias": alias, "task_type": task, "status": "failed", "quality_acceptance": "not_evaluated",
                      "purpose": "minimal_api_shape_probe" if kind == "trace_variant" else "minimal_text_api_probe"}
            try:
                outputs = sdk_generate(builder, stage / f"probe{index}")
                write_rows(stage / f"probe{index}_sdk_outputs.jsonl", outputs)
                record = bind_raw_response({}, rows(stage / "http_requests.jsonl"), public, (kind,))
                if not record[kind + "__transport"]["verified_response_match"]:
                    raise ValueError("No observed NVIDIA HTTP response")
                if record[kind + "__transport"]["finish_reason"] != "stop":
                    raise ValueError("API probe unfinished or truncated")
                if kind == "trace_variant":
                    # Strict raw JSON validation only: no extraction/repair by SDK.
                    verification = ns["verify_nvidia_trace"](seed, record[kind])
                    if not verification.get("schema_valid"):
                        raise ValueError("Super raw output does not satisfy the existing closed schema")
                    result["json_schema_supported"] = True
                    result["thinking_present"] = bool(record.get(kind + "__reasoning_content"))
                    if c["require_reasoning_content"] and not result["thinking_present"]:
                        raise ValueError("This registered Run requires actual thinking in reasoning_content")
                elif not isinstance(record.get("introduction"), str) or not record["introduction"]:
                    raise ValueError("Lightning returned no text")
                result.update(status="passed", transport=record[kind + "__transport"])
            except Exception as error:
                result["error"] = redact(type(error).__name__ + ": " + str(error))
            results.append(result)
            write_new(stage / f"probe{index}_result.json", result)
            if alias == "nvidia-reasoning" and result["status"] != "passed":
                break  # No relaxed format, schema fallback, or further experiment.
    models = {alias: {"status": "passed" if len([r for r in results if r["alias"] == alias and r["status"] == "passed"])
                    == (1 if alias == "nvidia-text" else 3) else "failed",
                     "probes": [r for r in results if r["alias"] == alias]} for alias in c["models"]}
    passed = all(v["status"] == "passed" for v in models.values())
    summary = {"status": "passed" if passed else "failed_api_smoke", "models": models,
        "elapsed_seconds": time.perf_counter() - started, "host": "local controller; not AutoDL network validation",
        "telemetry": telemetry_summary(rows(stage / "http_requests.jsonl"), c),
        "prepared_manifest_sha256": digest(Path(target) / "seed_manifest.json"), "generation_quality_passed": False}
    write_new(stage / "summary.json", summary)
    return summary


def run(target):
    require_api_key()
    c, seeds = verify_prepared(target)
    api = read(Path(target) / "api_smoke/summary.json")
    if api["status"] != "passed" or api["prepared_manifest_sha256"] != digest(Path(target) / "seed_manifest.json"):
        raise RuntimeError("Both NVIDIA models and all three schemas must pass API smoke for this Run")
    stage = Path(target) / "generation"
    stage.mkdir(exist_ok=False)
    ns = load_notebook(Path(target).name, stage / "loader")
    records = []
    errors = []
    start = time.perf_counter()
    with capture_requests(stage / "http_requests.jsonl"):
        for index, seed in enumerate(seeds):
            seed_file = stage / f"seed{index}_public.jsonl"
            write_rows(seed_file, [{k: seed[k] for k in ("seed_id", "task_type", "prompt")}])
            builder = ns["build_nvidia_data_designer"](seed_file, c, seed["task_type"])
            write_new(stage / f"seed{index}_compiled_config.json", builder.build().model_dump(mode="json"))
            try:
                outputs = sdk_generate(builder, stage / f"seed{index}")
                write_rows(stage / f"seed{index}_sdk_outputs.jsonl", outputs)
            except Exception as error:
                errors.append({"seed_id": seed["seed_id"], "error": redact(type(error).__name__ + ": " + str(error))})
                break  # Preserve partial HTTP outputs, never retry under a changed schema.
            finally:
                records.append(bind_raw_response({k: seed[k] for k in ("seed_id", "task_type", "prompt")},
                                                  rows(stage / "http_requests.jsonl"), seed))
    write_rows(stage / "raw_model_outputs.jsonl", records)
    write_new(stage / "runtime_errors.json", errors)
    quality = ns["screen_nvidia_records"](records, seeds, stage, read(Path(target) / "seed_checks.json"),
                                          require_reasoning_content=c["require_reasoning_content"])
    result = {"status": "failed_runtime" if errors else quality["status"], "quality": quality,
        "telemetry": telemetry_summary(rows(stage / "http_requests.jsonl"), c), "elapsed_seconds": time.perf_counter() - start,
        "errors": errors, "gpu_used": False, "merge_into_sft": False}
    write_new(stage / "result.json", result)
    return result


def main():
    os.environ["NEMO_TELEMETRY_ENABLED"] = "false"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "api-smoke", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    c = validated_config(args.config)
    run_id = args.run_id or c["synthetic_run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid Run ID")
    target = ROOT / "data/processed/round3" / run_id
    if args.stage == "prepare":
        result = {"status": "prepared_cpu_only", "path": str(prepare(args.config, run_id))}
    else:
        result = api_smoke(target) if args.stage == "api-smoke" else run(target)
    print(json.dumps(redact(result), ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("passed", "prepared_cpu_only") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(redact(type(error).__name__ + ": " + str(error)), file=sys.stderr)
        raise SystemExit(1)
