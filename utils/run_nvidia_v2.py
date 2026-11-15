"""Offline V2 re-audit and isolated SFT rendering. No API/model/GPU entrypoint."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.run_synthetic_smoke import load_notebook, read, rows

ORIGINAL = ROOT / "data/processed/round3/nvidia_hosted_20261115_ultra_r4_full"
REAUDIT = ROOT / "data/processed/round3/nvidia_hosted_20261115_ultra_r4_reaudit_v2"
SFT_PREVIEW = ROOT / "data/processed/round3/nvidia_hosted_20261115_ultra_r4_sft_preview_v2"
SFT_NOTEBOOK = ROOT / "03_SFT.ipynb"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_jsonl_new(path, values):
    with Path(path).open("x") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def tree_hashes(path):
    path = Path(path)
    return {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()}


def assert_tree_unchanged(path, expected):
    if tree_hashes(path) != expected:
        raise ValueError("Historical Run changed; do not overwrite or reinterpret its artifacts")


def verify_historical_archive(source):
    """Compare saved historical artifacts, not current (intentionally new) code."""
    source = Path(source)
    archive = read(source / "archive_manifest.json")
    for name, expected in archive["artifact_hashes"].items():
        if file_hash(source / name) != expected:
            raise ValueError("Historical artifact hash mismatch: " + name)
    original = read(source / "seed_manifest.json")
    for name, expected in original["frozen_files"].items():
        if file_hash(source / name) != expected:
            raise ValueError("Frozen original input changed: " + name)
    private = rows(source / "private_seeds.jsonl")
    if rows(source / "public_seeds.jsonl") != [{k: s[k] for k in ("seed_id", "task_type", "prompt")} for s in private]:
        raise ValueError("Original public projection is inconsistent")
    return original


def check_seed_gates(ns, seeds):
    import copy
    checks = {}
    for seed in seeds:
        sample = copy.deepcopy(seed)
        solver = ns["_solve_candidate"](sample)
        independent = ns["_screen_verification"](sample)
        leakage = ns["audit_synthetic_leakage"](seed, ns["real_index"])
        passed = bool(ns["numeric_or_text_match"](sample["answer"], seed["answer"]) and independent["answer_ok"] and
                      independent["trace_ok"] and independent["identifiability"]["answer_unique"] and leakage["status"] == "passed")
        checks[seed["seed_id"]] = {"passed": passed, "solver_result": solver, "independent_verification": independent,
                                  "leakage_audit": leakage}
    if not all(v["passed"] for v in checks.values()):
        raise ValueError("Seed gates failed; cannot re-audit against an invalid reference")
    return checks


def reaudit(source=ORIGINAL, destination=REAUDIT):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Re-audit cannot write inside the original Run")
    original = verify_historical_archive(source)
    before = tree_hashes(source)
    destination.mkdir(parents=True, exist_ok=False)
    seeds = rows(source / "private_seeds.jsonl")
    write_json_new(destination / "original_tree_before.json", before)
    for name in ("private_seeds.jsonl", "public_seeds.jsonl"):
        with (destination / name).open("xb") as f:
            f.write((source / name).read_bytes())
    with tempfile.TemporaryDirectory() as temp:
        ns = load_notebook(destination.name, Path(temp) / "loader")
        gates = check_seed_gates(ns, seeds)
        write_json_new(destination / "seed_checks.json", gates)
        provenance = {"original_run_id": source.name, "original_run_path": str(source),
            "original_git_commit": original["git_commit"],
            "original_raw_model_outputs_sha256": file_hash(source / "generation/raw_model_outputs.jsonl"),
            "original_http_requests_sha256": file_hash(source / "generation/http_requests.jsonl"),
            "original_private_seed_sha256": file_hash(source / "private_seeds.jsonl"),
            "original_public_seed_sha256": file_hash(source / "public_seeds.jsonl"),
            "original_config_sha256": file_hash(source / "config.json"),
            "original_tree_sha256": file_hash(destination / "original_tree_before.json"),
            "models_called": 0, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "new_sources_sha256": {p.name: file_hash(p) for p in (ROOT / "01_Synthetic_Data_Engineering.ipynb", Path(__file__))}}
        summary = ns["screen_nvidia_records_v2"](rows(source / "generation/raw_model_outputs.jsonl"), seeds, destination,
            gates, rows(source / "generation/http_requests.jsonl"), provenance)
    assert_tree_unchanged(source, before)
    write_json_new(destination / "immutability_audit.json", {"status": "passed", "original_files_checked": len(before),
        "original_tree_unchanged": True, "new_model_calls": 0, "gpu_used": False})
    return summary


def validate_v2_manifest(directory, proof_ns):
    directory = Path(directory)
    manifest = read(directory / "accepted_manifest.json")
    if (manifest.get("schema_version") != "nvidia-accepted-manifest/v2" or manifest.get("status") != "passed" or
            manifest.get("acceptance_protocol") != proof_ns["NVIDIA_ACCEPTANCE_PROTOCOL_V2"] or
            manifest.get("verifier_version") != proof_ns["NVIDIA_VERIFIER_V2"] or
            not manifest.get("all_accepted_verified_and_audited") or manifest.get("merge_into_sft") is not False):
        raise ValueError("V2 Accepted subset Manifest did not pass")
    for name, expected in manifest["artifact_sha256"].items():
        if file_hash(directory / name) != expected:
            raise ValueError("V2 artifact changed: " + name)
    for split, expected in manifest["split_sha256"].items():
        actual = proof_ns["dataset_manifest"]["artifacts"][split]["sha256"]
        if expected != actual:
            raise ValueError("Split changed since V2 audit: " + split)
    if set(manifest["split_sha256"]) != {"train", "validation", "holdout_test"}:
        raise ValueError("Missing split audits")
    if file_hash(proof_ns["manifest_path"]) != manifest["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    origin = manifest["origin"]
    source = Path(origin["original_run_path"])
    before = read(directory / "original_tree_before.json")
    if file_hash(directory / "original_tree_before.json") != origin["original_tree_sha256"]:
        raise ValueError("Original tree audit changed")
    assert_tree_unchanged(source, before)
    for local, key in (("private_seeds.jsonl", "original_private_seed_sha256"),
                       ("public_seeds.jsonl", "original_public_seed_sha256")):
        if file_hash(directory / local) != origin[key]:
            raise ValueError("V2 frozen seed changed")
    seeds = rows(directory / "private_seeds.jsonl")
    if rows(directory / "public_seeds.jsonl") != [{k: s[k] for k in ("seed_id", "task_type", "prompt")} for s in seeds]:
        raise ValueError("Private/public projection mismatch")
    accepted = rows(directory / "accepted.jsonl")
    if len(accepted) != manifest["accepted_rows"] or len({c["candidate_id"] for c in accepted}) != len(accepted):
        raise ValueError("Accepted count or uniqueness mismatch")
    seed_by_id = {s["seed_id"]: s for s in seeds}
    records_by_id = {r["seed_id"]: r for r in rows(source / "generation/raw_model_outputs.jsonl")}
    http = rows(source / "generation/http_requests.jsonl")
    for candidate in accepted:
        seed = seed_by_id[candidate["seed_id"]]
        kind = candidate["variant_kind"]
        raw = records_by_id[seed["seed_id"]]
        if (candidate.get("verification_status") != "accepted" or candidate.get("leakage_status") != "passed" or
                candidate.get("acceptance_protocol") != manifest["acceptance_protocol"] or
                candidate.get("verifier_version") != manifest["verifier_version"] or
                candidate.get("variant_content") != raw.get(kind) or not candidate.get("verification", {}).get("verified")):
            raise ValueError("Accepted provenance/status mismatch")
        if kind not in {"prompt_variant", "trace_variant"}:
            raise ValueError("Unknown V2 variant")
        wire = proof_ns["nvidia_wire_check_v2"](seed, raw, kind, http)
        if not wire["verified"]:
            raise ValueError("Accepted proof has no valid raw HTTP provenance")
        if candidate["prompt"] != (raw[kind] if kind == "prompt_variant" else seed["prompt"]):
            raise ValueError("Accepted Prompt changed")
        verification = (proof_ns["verify_nvidia_trace_v2"](seed, raw[kind]) if kind == "trace_variant" else
                        proof_ns["verify_nvidia_introduction"](seed, raw["introduction"], raw[kind]))
        if not verification["verified"]:
            raise ValueError("Accepted proof failed fresh independent verification")
    return manifest, accepted, seeds


def sft_training_tasks():
    """Read the actual existing build_full_trace_records task tuple, not a copy."""
    for cell in read(SFT_NOTEBOOK)["cells"]:
        if cell["cell_type"] != "code": continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "build_full_trace_records":
                for statement in node.body:
                    if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name) and statement.target.id == "task":
                        return tuple(ast.literal_eval(statement.iter))
    raise ValueError("Cannot identify current SFT task pool; do not guess")


def load_sft_render_definitions():
    """Execute definitions only: no dataset loading, default building, or torch."""
    wanted = {"_format_trace", "make_trace_record", "audit_trace_record", "encode_chat_example",
              "nvidia_proof_to_solver_sample", "render_nvidia_verified_proof", "render_nvidia_verified_records"}
    namespace = {"Path": Path, "json": json, "hashlib": hashlib}
    from collections import Counter
    namespace["Counter"] = Counter
    notebook = read(SFT_NOTEBOOK)
    found = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code": continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "max_length" for t in node.targets):
                # Read the notebook's registered default without changing its env.
                call = node.value
                namespace["max_length"] = int(ast.literal_eval(call.args[0].args[1]))
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(SFT_NOTEBOOK), "exec"), namespace)
                found.add(node.name)
    if found != wanted:
        raise ValueError("Required SFT definitions missing: " + str(wanted - found))
    return namespace


def local_tokenizer(path=None):
    if path is None:
        snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
        path = snapshot if (snapshot / "tokenizer.json").exists() else None
    if path is None:
        return None, {"status": "unavailable_no_local_tokenizer", "actual_counts_reported": False}
    path = Path(path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    files = {p.name: file_hash(p) for p in path.iterdir() if p.is_file() and
             (p.name.startswith("tokenizer") or p.name in {"vocab.json", "merges.txt", "chat_template.jinja"})}
    return tokenizer, {"status": "local_only", "model": "Qwen/Qwen3-0.6B", "path": str(path),
                       "files_sha256": files, "model_loaded": False, "network_used": False}


def render(reaudit_dir=REAUDIT, destination=SFT_PREVIEW, tokenizer_path=None):
    with tempfile.TemporaryDirectory() as temp:
        proof_ns = load_notebook("nvidia_v2_sft_export", Path(temp) / "loader")
        sft = load_sft_render_definitions()
        tokenizer, metadata = local_tokenizer(tokenizer_path)
        return sft["render_nvidia_verified_records"](reaudit_dir, destination, proof_ns, tokenizer, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reaudit", "render"))
    parser.add_argument("--source", type=Path, default=ORIGINAL)
    parser.add_argument("--reaudit-dir", type=Path, default=REAUDIT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    value = (reaudit(args.source, args.output or args.reaudit_dir) if args.stage == "reaudit" else
             render(args.reaudit_dir, args.output or SFT_PREVIEW, args.tokenizer))
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
