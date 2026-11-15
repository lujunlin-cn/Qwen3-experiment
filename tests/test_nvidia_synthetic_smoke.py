"""CPU mathematics + actual SDK with mocked HTTPS transport, no hosted claims."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils import run_nvidia_synthetic_smoke as nvidia
from test_nemo_smoke import reference_proof


class NvidiaSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.prepared = nvidia.prepare(run_id="nvidia_cpu_test", target=cls.root / "prepared")
        cls.config, cls.seeds = nvidia.verify_prepared(cls.prepared)
        cls.ns = nvidia.load_notebook("nvidia_cpu_test", cls.root / "loader")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_frozen_nine_seeds_and_public_projection(self):
        seeds, hashes = nvidia.check_seeds(self.config)
        self.assertEqual(set(hashes.values()), {self.config["seed_sha256"]})
        self.assertEqual(len(seeds), 9)
        public = nvidia.rows(self.prepared / "public_seeds.jsonl")
        for seed, public_seed in zip(seeds, public):
            self.assertEqual(set(public_seed), {"seed_id", "task_type", "prompt"})
            self.assertEqual(public_seed["prompt"], seed["prompt"])
        checks = nvidia.read(self.prepared / "seed_checks.json")
        self.assertTrue(all(v["passed"] for v in checks.values()))

    def test_explicit_models_match_official_defaults_and_builtin_provider(self):
        models = self.ns["nvidia_model_configs"](self.config, "gravity")
        self.assertEqual([m.alias for m in models], ["nvidia-text", "nvidia-reasoning"])
        self.assertTrue(all(m.provider == "nvidia" and m.inference_parameters.max_parallel_requests == 1 for m in models))
        self.assertEqual(models[0].inference_parameters.extra_body, {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(models[1].inference_parameters.extra_body["reasoning_effort"], "medium")
        self.assertTrue(all(m.inference_parameters.temperature == 1.0 and m.inference_parameters.top_p == .95 for m in models))
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "cpu-fixture-only"}):
            designer = nvidia.new_designer(self.root / "provider_compile_only")
        provider = designer.model_provider_registry.get_provider("nvidia")
        self.assertEqual(provider.endpoint, nvidia.ENDPOINT)
        self.assertEqual(provider.api_key, "NVIDIA_API_KEY")
        from data_designer.engine.secret_resolver import EnvironmentResolver
        self.assertIsInstance(designer._secret_resolver, EnvironmentResolver)

    def test_missing_key_fails_before_network_and_run_directory(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(nvidia, "verify_prepared", side_effect=AssertionError("No work")):
            for stage in (nvidia.api_smoke, nvidia.run):
                with self.assertRaisesRegex(RuntimeError, "NVIDIA_API_KEY is missing"):
                    stage(self.root / "never_created")
        self.assertFalse((self.root / "never_created").exists())

    def test_redaction_never_saves_credentials(self):
        secret = "fixture-private-value-for-redaction"
        with patch.dict(os.environ, {"NVIDIA_API_KEY": secret}):
            value = nvidia.redact({"Authorization": "Bearer " + secret, "request": {"api_key": secret},
                                   "error": secret, "response": "nvapi-test-fixture-only"})
            self.assertNotIn(secret, json.dumps(value))
            self.assertNotIn("nvapi-", json.dumps(value))
            self.assertNotIn("Authorization", value)

    def test_prompt_append_exact_and_conservative_new_contract(self):
        for seed in self.seeds:
            for intro in (self.ns["NVIDIA_INTRODUCTIONS"][0], self.ns["NVIDIA_INTRODUCTIONS"][-1]):
                text = intro + "\n" + seed["prompt"]
                self.assertTrue(self.ns["verify_nvidia_introduction"](seed, intro, text)["verified"])
                self.assertFalse(self.ns["verify_nvidia_introduction"](seed, intro, text + "\n")["verified"])
            for bad in ("Use g=2*d/t^2.", "The answer is correct.", "Output 42.", "Do not solve this problem."):
                self.assertFalse(self.ns["verify_nvidia_introduction"](seed, bad, bad + "\n" + seed["prompt"])["verified"])
            # Historical acceptance still rejects the new sentence.
            intro = self.ns["NVIDIA_INTRODUCTIONS"][-1]
            self.assertFalse(self.ns["verify_smoke_prompt"](seed, intro + "\n" + seed["prompt"])["verified"])

    def test_strict_trace_positive_and_incorrect_intermediate(self):
        for seed in self.seeds:
            proof = reference_proof(self.ns, seed)
            raw = json.dumps(proof)
            self.assertTrue(self.ns["verify_nvidia_trace"](seed, raw)["verified"])
            if seed["task_type"] == "numeral_system":
                proof["components"][0]["arabic"] += 1
            elif seed["task_type"] == "gravity":
                proof["g_estimates"][0] *= 2
            else:
                proof["example_outputs"][0] = "999999"
            bad = self.ns["verify_nvidia_trace"](seed, json.dumps(proof))
            self.assertTrue(bad["schema_valid"])
            self.assertFalse(bad["verified"])
            for malformed in (raw[:-1] + ',"answer":"duplicate"}', raw[:-1] + ',"extra":1}',
                              '```json\n' + raw + '\n```', 'NaN', raw + '\nUnverified narrative'):
                self.assertFalse(self.ns["verify_nvidia_trace"](seed, malformed)["verified"])

    def test_verifier_stays_independent_and_requires_all_examples(self):
        for task, solver in (("numeral_system", "solve_numeral_prompt"), ("gravity", "solve_gravity_prompt"),
                             ("equation_numeric", "solve_equation_numeric_prompt")):
            seed = next(s for s in self.seeds if s["task_type"] == task)
            proof = reference_proof(self.ns, seed)
            with patch.dict(self.ns, {solver: lambda *a, **k: (_ for _ in ()).throw(AssertionError("Solver invoked"))}):
                self.assertTrue(self.ns["verify_nvidia_trace"](seed, json.dumps(proof))["verified"])
            if task == "gravity":
                proof["g_estimates"].pop()
            elif task == "equation_numeric":
                proof["example_outputs"].pop()
            else:
                proof["components"] = []
            self.assertFalse(self.ns["verify_nvidia_trace"](seed, json.dumps(proof))["verified"])

    def test_all_three_splits_leakage_and_empty_accepted(self):
        for split in ("train", "validation", "holdout_test"):
            row = self.ns["split_df"][self.ns["split_df"].split == split].iloc[0]
            audit = self.ns["audit_synthetic_leakage"]({"prompt": row.prompt, "task_type": row.task_type}, self.ns["real_index"])
            self.assertEqual(audit["status"], "rejected")
        dest = self.root / "empty_screen"
        summary = self.ns["screen_nvidia_records"]([], self.seeds, dest, nvidia.read(self.prepared / "seed_checks.json"))
        self.assertEqual((summary["status"], summary["accepted"], summary["preview"]), ("empty_accepted", 0, 18))
        self.assertFalse(nvidia.read(dest / "accepted_manifest.json")["all_accepted_verified_and_audited"])
        self.assertEqual(set(nvidia.read(dest / "synthetic_real_leakage_audit.json")["by_split"]),
                         {"train", "validation", "holdout_test"})

    def test_run_overwrite_and_hash_mutation_rejected(self):
        with self.assertRaises(FileExistsError):
            nvidia.prepare(target=self.prepared)
        with tempfile.TemporaryDirectory() as d:
            target = nvidia.prepare(target=Path(d) / "prepared", run_id="nvidia_hash_fixture")
            with (target / "public_seeds.jsonl").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                nvidia.verify_prepared(target)

    def mock_http(self, super_error=False):
        import httpx
        seeds = {s["seed_id"]: s for s in self.seeds}
        captured = []
        async def respond(transport, request):
            payload = json.loads(request.content)
            captured.append(payload)
            text = payload["messages"][-1]["content"]
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text)
            seed_id = text.split("SEED_ID=", 1)[1].splitlines()[0]
            seed = seeds[seed_id]
            self.assertIn(seed["prompt"], text)
            self.assertEqual(str(request.url), nvidia.ENDPOINT + "/chat/completions")
            if "VARIANT_KIND=prompt_variant" in text:
                self.assertEqual(payload["model"], self.config["models"]["nvidia-text"]["model"])
                content = self.ns["NVIDIA_INTRODUCTIONS"][0]
            else:
                self.assertEqual(payload["reasoning_effort"], "medium")
                self.assertEqual(payload["response_format"]["type"], "json_schema")
                if super_error:
                    return httpx.Response(400, json={"error": {"message": "Fixture: unsupported JSON schema", "type": "invalid_request_error"}}, request=request)
                content = json.dumps(reference_proof(self.ns, seed))
            return httpx.Response(200, json={"id": "cpu-fixture-only", "object": "chat.completion", "created": 1,
                "model": payload["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 101, "completion_tokens": 21, "total_tokens": 122}}, request=request)
        return captured, respond

    def test_actual_sdk_18_requests_strict_raw_binding_and_no_secret_artifacts(self):
        import httpx
        captured, respond = self.mock_http()
        dest = self.root / "sdk18"
        dest.mkdir()
        secret = "cpu-fixture-secret-no-real-key"
        with patch.dict(os.environ, {"NVIDIA_API_KEY": secret, "NEMO_TELEMETRY_ENABLED": "false"}), patch.object(
                httpx.AsyncHTTPTransport, "handle_async_request", respond), nvidia.capture_requests(dest / "http_requests.jsonl"):
            records = []
            for i, seed in enumerate(self.seeds):
                seed_file = dest / f"seed{i}.jsonl"
                nvidia.write_rows(seed_file, [{k: seed[k] for k in ("seed_id", "task_type", "prompt")}])
                builder = self.ns["build_nvidia_data_designer"](seed_file, self.config, seed["task_type"])
                outputs = nvidia.sdk_generate(builder, dest / f"sdk{i}")
                if not outputs:
                    self.assertTrue((dest / f"sdk{i}/profiling_warning.json").exists())
                else:
                    self.assertEqual(len(outputs), 1)
                records.append(nvidia.bind_raw_response({k: seed[k] for k in ("seed_id", "task_type", "prompt")},
                    nvidia.rows(dest / "http_requests.jsonl"), seed))
            summary = self.ns["screen_nvidia_records"](records, self.seeds, dest / "screen", nvidia.read(self.prepared / "seed_checks.json"))
        self.assertEqual(len(captured), 18)
        self.assertEqual(summary["accepted"], 18, summary)
        self.assertEqual(nvidia.telemetry_summary(nvidia.rows(dest / "http_requests.jsonl"))["http_requests"], 18)
        for path in dest.rglob("*"):
            if path.is_file():
                self.assertNotIn(secret.encode(), path.read_bytes(), str(path))
        # A repaired SDK object cannot hide a duplicate key in raw text.
        corrupted = copy.deepcopy(records)
        for record in corrupted:
            record["trace_variant"] = record["trace_variant"][:-1] + ',"answer":"duplicate"}'
            record["prompt_variant__reasoning_content"] = "unverified thought"
        summary = self.ns["screen_nvidia_records"](corrupted, self.seeds, dest / "corrupted", nvidia.read(self.prepared / "seed_checks.json"))
        self.assertEqual(summary["accepted"], 0)

    def test_api_smoke_reports_model_failure_separately_and_blocks_generation(self):
        import httpx
        captured, respond = self.mock_http(super_error=True)
        target = nvidia.prepare(target=self.root / "api_failure", run_id="nvidia_api_failure_fixture")
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "cpu-fixture-only", "NEMO_TELEMETRY_ENABLED": "false"}), patch.object(
                httpx.AsyncHTTPTransport, "handle_async_request", respond):
            result = nvidia.api_smoke(target)
            self.assertEqual(result["models"]["nvidia-text"]["status"], "passed")
            self.assertEqual(result["models"]["nvidia-reasoning"]["status"], "failed")
            self.assertEqual(result["status"], "failed_api_smoke")
            with self.assertRaisesRegex(RuntimeError, "Both NVIDIA models"):
                nvidia.run(target)
        self.assertFalse((target / "generation").exists())
        self.assertTrue(any(e["http_status"] == 400 for e in nvidia.rows(target / "api_smoke/http_requests.jsonl")))

    def test_successful_three_stages_22_mock_wire_requests_and_nonoverwrite(self):
        import httpx
        target = nvidia.prepare(target=self.root / "complete_fixture", run_id="nvidia_complete_cpu_fixture")
        captured, respond = self.mock_http()
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "cpu-fixture-only", "NEMO_TELEMETRY_ENABLED": "false"}), patch.object(
                httpx.AsyncHTTPTransport, "handle_async_request", respond):
            api = nvidia.api_smoke(target)
            self.assertEqual(api["status"], "passed", api)
            result = nvidia.run(target)
            self.assertEqual(result["status"], "passed", result)
            self.assertEqual(result["quality"]["accepted"], 18)
            self.assertEqual(result["telemetry"]["http_requests"], 18)
            self.assertEqual(result["telemetry"]["by_model"]["nvidia-text"]["http_requests"], 9)
            self.assertEqual(result["telemetry"]["by_model"]["nvidia-reasoning"]["http_requests"], 9)
            self.assertEqual(len(captured), 22)
            for stage in (nvidia.api_smoke, nvidia.run):
                with self.assertRaises(FileExistsError):
                    stage(target)


if __name__ == "__main__":
    unittest.main()
