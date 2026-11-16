"""CPU/offline V2 + SFT rendering + real SDK with mocked HTTPS for Gravity.

Network sockets are forbidden throughout this suite. No credentials/models/GPU.
"""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from utils import run_nvidia_v2 as v2
from utils import run_nvidia_gravity_targeted as gravity
from utils import run_nvidia_synthetic_smoke as historical
from test_nemo_smoke import reference_proof


class V2RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network_guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden in CPU V2 task"))
        cls.network_guard.start()
        cls.addClassCleanup(cls.network_guard.stop)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.before = v2.tree_hashes(v2.ORIGINAL)
        cls.ns = v2.load_notebook("nvidia_v2_cpu_fixture", cls.root / "loader")
        cls.seeds = v2.rows(v2.ORIGINAL / "private_seeds.jsonl")
        cls.gates = v2.check_seed_gates(cls.ns, cls.seeds)
        cls.sft = v2.load_sft_render_definitions()
        cls.reaudit = cls.root / "actual_reaudit"
        cls.reaudit_summary = v2.reaudit(destination=cls.reaudit)

    @classmethod
    def tearDownClass(cls):
        v2.assert_tree_unchanged(v2.ORIGINAL, cls.before)

    def fixture(self, seed, thinking="UNVERIFIED_MODEL_THINKING_FIXTURE", proof=None, finish="stop"):
        intro = self.ns["NVIDIA_INTRODUCTIONS"][0]
        proof = reference_proof(self.ns, seed) if proof is None else proof
        content = proof if isinstance(proof, str) else json.dumps(proof)
        record = {k: seed[k] for k in ("seed_id", "task_type", "prompt")}
        record.update(introduction=intro, prompt_variant=intro + "\n" + seed["prompt"], trace_variant=content)
        http = []
        for kind, raw in (("prompt_variant", intro), ("trace_variant", content)):
            usage = {"prompt_tokens": 30, "completion_tokens": 20,
                     "completion_tokens_details": {"reasoning_tokens": 7 if thinking else 0}}
            reason = finish if kind == "trace_variant" else "stop"
            msg = {"content": raw, "reasoning_content": thinking if kind == "trace_variant" else None}
            record[kind + "__transport"] = {"verified_response_match": True, "finish_reason": reason, "usage": usage}
            record[kind + "__reasoning_content"] = msg["reasoning_content"]
            http.append({"seed_id": seed["seed_id"], "variant_kind": kind, "status": "completed", "http_status": 200,
                "usage": usage, "response": {"choices": [{"finish_reason": reason, "message": msg}]}})
        return record, http

    def screen_fixture(self, seed, record, http, name):
        destination = self.root / name
        self.ns["screen_nvidia_records_v2"]([record], [seed], destination, self.gates, http, {"models_called": 0})
        return destination, v2.rows(destination / "candidates.jsonl")

    def test_thinking_present_or_absent_does_not_change_json_acceptance(self):
        for i, thinking in enumerate(("UNVERIFIED_MODEL_THINKING_FIXTURE", None, "")):
            seed = self.seeds[0]
            record, http = self.fixture(seed, thinking)
            target, candidates = self.screen_fixture(seed, record, http, f"thinking{i}")
            self.assertEqual(sum(c["verification_status"] == "accepted" for c in candidates), 2)
            trace = candidates[1]
            self.assertFalse(trace["verification"]["reasoning_content_verified"])
            self.assertNotIn("raw_reasoning_content", trace)
            self.assertEqual(v2.rows(target / "thinking_audit.jsonl")[1]["raw_reasoning_content"], thinking)
            self.assertEqual(trace["thinking_audit"]["verification_status"], "unverified")

    def test_original_actual_reaudit_15_accepted_3_preview(self):
        self.assertEqual((self.reaudit_summary["accepted"], self.reaudit_summary["preview"]), (15, 3))
        manifest, accepted, _ = v2.validate_v2_manifest(self.reaudit, self.ns)
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["run_status"], "completed_with_previews")
        self.assertFalse(self.reaudit_summary["all_quality_passed"])
        self.assertEqual(manifest["lineage_status_counts"], {"unavailable": 18})
        self.assertEqual(manifest["numeral_query_exposure_policy"], "report_only")
        self.assertFalse(manifest["claims_no_mapping_leakage"])
        trace = [r for r in accepted if r["variant_kind"] == "trace_variant"]
        self.assertEqual(len(trace), 6)
        self.assertEqual({r["task_type"] for r in trace}, {"numeral_system", "equation_numeric"})

    def test_strict_schema_duplicates_wrong_intermediates_and_no_prose_extraction(self):
        for seed in self.seeds:
            proof = reference_proof(self.ns, seed)
            raw = json.dumps(proof)
            for bad in (raw[:-1] + ',"answer":"duplicate"}', raw[:-1] + ',"extra":1}',
                        '```json\n' + raw + '\n```', raw + '\nextraneous claim'):
                self.assertFalse(self.ns["verify_nvidia_trace_v2"](seed, bad)["verified"])
            if seed["task_type"] == "numeral_system": proof["components"][0]["arabic"] += 1
            elif seed["task_type"] == "gravity": proof["g_estimates"][0] *= 2
            else: proof["example_outputs"][0] = "999999"
            self.assertEqual(proof["answer"], reference_proof(self.ns, seed)["answer"])
            self.assertFalse(self.ns["verify_nvidia_trace_v2"](seed, json.dumps(proof))["verified"])

    def test_gravity_query_time_string_and_original_numeric_tolerance(self):
        seed = next(s for s in self.seeds if s["seed_id"] == "gravity-seed3")
        proof = reference_proof(self.ns, seed)
        self.assertEqual(proof["query_t"], "4.62")
        self.assertTrue(self.ns["verify_nvidia_trace_v2"](seed, json.dumps(proof))["verified"])
        proof["query_t"] = "4.62s"
        self.assertFalse(self.ns["verify_nvidia_trace_v2"](seed, json.dumps(proof))["verified"])
        proof["query_t"] = "4.620"
        self.assertFalse(self.ns["verify_nvidia_trace_v2"](seed, json.dumps(proof))["verified"])
        proof = reference_proof(self.ns, seed)
        proof["g_ols"] *= 1.0002
        self.assertFalse(self.ns["verify_nvidia_trace_v2"](seed, json.dumps(proof))["verified"])

    def test_http_failed_truncated_forged_or_empty_never_accepted(self):
        seed = self.seeds[0]
        for i, mode in enumerate(("length", "http_failure", "match_false", "content_mismatch", "empty", "missing_http")):
            record, http = self.fixture(seed, finish="length" if mode == "length" else "stop")
            if mode == "http_failure": http[-1]["http_status"] = 500
            if mode == "match_false": record["trace_variant__transport"]["verified_response_match"] = False
            if mode == "content_mismatch": record["trace_variant"] += " "
            if mode == "empty":
                record["trace_variant"] = ""
                http[-1]["response"]["choices"][0]["message"]["content"] = ""
            if mode == "missing_http": http.pop()
            _, candidates = self.screen_fixture(seed, record, http, "bad_http" + str(i))
            self.assertEqual(candidates[1]["verification_status"], "preview", mode)

    def test_empty_accepted_never_passes_manifest_and_no_overwrite(self):
        dest = self.root / "empty"
        self.ns["screen_nvidia_records_v2"]([], self.seeds, dest, self.gates, [], {})
        manifest = v2.read(dest / "accepted_manifest.json")
        self.assertEqual(manifest["status"], "empty_accepted")
        self.assertEqual(manifest["run_status"], "empty_accepted")
        self.assertFalse(manifest["all_accepted_verified_and_audited"])
        with self.assertRaises(FileExistsError):
            self.ns["screen_nvidia_records_v2"]([], self.seeds, dest, self.gates, [], {})
        with self.assertRaises(FileExistsError): v2.reaudit(destination=self.reaudit)

    def test_verifier_independence_and_all_examples_rules_required(self):
        for seed in self.seeds:
            proof = reference_proof(self.ns, seed)
            solvers = {k: lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Verifier called Solver"))
                       for k in ("solve_numeral_prompt", "solve_gravity_prompt", "solve_equation_numeric_prompt")}
            with patch.dict(self.ns, solvers):
                self.assertTrue(self.ns["verify_nvidia_trace_v2"](seed, json.dumps(proof))["verified"])
            if seed["task_type"] == "gravity": proof["g_estimates"].pop()
            elif seed["task_type"] == "equation_numeric": proof["example_outputs"].pop()
            else: proof["components"] = []
            self.assertFalse(self.ns["verify_nvidia_trace_v2"](seed, json.dumps(proof))["verified"])

    def test_all_three_task_mappings_boxed_deterministic_and_no_hidden_thinking(self):
        for i, task in enumerate(("numeral_system", "gravity", "equation_numeric")):
            seed = copy.deepcopy(next(s for s in self.seeds if s["task_type"] == task))
            record, http = self.fixture(seed)
            _, candidates = self.screen_fixture(seed, record, http, f"render_map{i}")
            candidate = candidates[1]
            a, _ = self.sft["render_nvidia_verified_proof"](candidate, seed, self.ns, v2.sft_training_tasks())
            b, _ = self.sft["render_nvidia_verified_proof"](candidate, seed, self.ns, v2.sft_training_tasks())
            self.assertEqual(a["completion"], b["completion"])
            self.assertIn('Final answer: \\boxed{' + a["final_answer"] + '}', a["completion"])
            self.assertNotIn('hidden_rule', a["completion"])
            self.assertNotIn('UNVERIFIED_MODEL_THINKING_FIXTURE', json.dumps(a))
            self.assertNotIn('reasoning_content', a)
            self.assertTrue(self.sft["audit_trace_record"](a)["accepted"])
            self.assertFalse(a["sft_training_eligible"])
            self.assertFalse(a["finally_entered_sft"])
            trace = a["canonical_solver_trace"]
            if task == 'numeral_system':
                self.assertEqual(set(trace), {'query_number', 'components'})
                self.assertFalse(a["task_pool_compatible"])
            elif task == 'gravity':
                self.assertEqual(set(trace), {'solver', 'g_estimates', 'g_ols', 'query_t', 'query_raw'})
            else:
                self.assertTrue(trace['candidate_rules_by_operator'])
                self.assertIn('selected_assignment', trace)

    def test_real_six_proofs_render_only_preview_and_manifest_tampering_rejected(self):
        target = self.root / 'render_actual'
        result = self.sft['render_nvidia_verified_records'](self.reaudit, target, self.ns)
        self.assertEqual(result['mathematical_proof_accepted'], 6)
        self.assertEqual(result['sft_format_audit_passed'], 6)
        self.assertEqual(result['current_task_pool_review_candidates'], 3)
        self.assertEqual(result['sft_training_eligible'], 0)
        records = v2.rows(target / 'nvidia_sft_rendered_preview.jsonl')
        self.assertEqual(len(records), 6)
        self.assertNotIn('gravity', {r['task_type'] for r in records})
        with self.assertRaises(FileExistsError):
            self.sft['render_nvidia_verified_records'](self.reaudit, target, self.ns)
        corrupt = self.root / 'corrupt_reaudit'
        shutil.copytree(self.reaudit, corrupt)
        with (corrupt / 'accepted.jsonl').open('a') as f: f.write('\n')
        with self.assertRaisesRegex(ValueError, 'V2 artifact changed'):
            v2.validate_v2_manifest(corrupt, self.ns)

    def test_training_default_and_old_notebook_functions_unchanged(self):
        self.assertEqual(v2.sft_training_tasks(), ('gravity', 'equation_numeric', 'bit_manipulation'))
        for name, new_ids in [('01_Synthetic_Data_Engineering.ipynb', {'nvidia-v2-acceptance', 'nvidia-v2-acceptance-contract',
                                  'final-train-only-protocol', 'final-train-only-functions'}),
                              ('03_SFT.ipynb', {'nvidia-v2-sft-render', 'nvidia-v2-sft-render-contract',
                                  'solver-first-preview-guide', 'solver-first-preview-render'})]:
            old = json.loads(subprocess.check_output(['git', 'show', '6f1c01e:' + name], cwd=v2.ROOT, text=True))
            current = v2.read(v2.ROOT / name)
            if name.startswith('01_'):
                # Explicitly authorized final-pilot generator interface changes.
                # All Solver/Verifier/prompt cells remain compared byte for byte.
                for cell in old['cells']:
                    text = ''.join(cell['source'])
                    if cell.get('id') == 'f4686928':
                        text = text.replace('train_numeric_operator_pool or tuple("+-*/|\\\\@$%&!?^#~`")',
                                            'train_numeric_operator_pool')
                    if cell.get('id') == 'c8de4249':
                        text = text.replace('    mode = str(rng.choice(["normal", "flip"]))',
                            '    if not numeric_operator_pool:\n        raise ValueError("Train operator support is empty; no global fallback")\n    mode = str(rng.choice(["normal", "flip"]))')
                    if cell.get('id') == '6a57d560':
                        # Compare the arithmetic/rendering body unchanged. Only
                        # the optional, sourced sampling-parameter interface differs.
                        cur = next(c for c in current['cells'] if c.get('id') == '6a57d560')
                        new_text = ''.join(cur['source'])
                        self.assertEqual(text[text.index('    example_pairs = '):],
                                         new_text[new_text.index('    example_pairs = '):])
                        text = new_text
                    if cell.get('id') in {'f4686928', 'c8de4249', '6a57d560'}:
                        cell['source'] = text.splitlines(keepends=True)
            # Explicitly authorized exact Equation repair: preserve every other
            # statement, including the Gravity arithmetic and default builders.
            authorized = {"_parse_numeric_prompt", "_numeric_candidate_rules", "_numeric_valid_assignments",
                "solve_equation_numeric_prompt", "equation_trace_from_resolution", "_numeric_verifier_format",
                "_numeric_verifier_candidate_rules", "_numeric_verifier_valid_assignments", "_numeric_verifier_parse",
                "verify_equation_numeric_prompt", "verify_equation_numeric_sample", "_equation_assignment_set",
                "audit_equation_agreement", "_solve_candidate", "_smoke_equation_resolution", "_verify_equation_proof",
                "_format_trace", "audit_trace_record"}
            def stable(nb):
                out=[]
                for c in nb:
                    if c.get('id') in new_ids or c.get('id','').startswith('equation-sft-augmentation-'):continue
                    if c['cell_type']!='code':out.append(c);continue
                    clone=dict(c)
                    tree=ast.parse(''.join(c['source']))
                    clone['source']=[ast.dump(x) for x in tree.body if not (isinstance(x,ast.FunctionDef) and x.name in authorized)]
                    out.append(clone)
                return out
            self.assertEqual(stable(old['cells']),stable(current['cells']))
        self.assertNotIn('full_trace_records', self.sft)
        self.assertNotIn('use_synthetic_data', self.sft)
        v2.assert_tree_unchanged(v2.ORIGINAL, self.before)

    def test_three_set_audit_still_rejects_actual_duplicates(self):
        for split in ('train', 'validation', 'holdout_test'):
            row = self.ns['split_df'][self.ns['split_df'].split == split].iloc[0]
            self.assertEqual(self.ns['audit_synthetic_leakage']({'prompt': row.prompt, 'task_type': row.task_type},
                                                               self.ns['real_index'])['status'], 'rejected')

    def test_gravity_conditions_same_seed_prompt_schema_only_budget_differs(self):
        output = gravity.prepare(output=self.root / 'gravity_prepared')
        c = gravity.validated_config()
        p16 = output / c['conditions']['P16K']['run_id']
        p32 = output / c['conditions']['P32K']['run_id']
        for name in ('private_seeds.jsonl', 'public_seeds.jsonl'):
            self.assertEqual((p16 / name).read_bytes(), (p32 / name).read_bytes())
        a, b = v2.read(p16 / 'compiled_config.json'), v2.read(p32 / 'compiled_config.json')
        a['seed_config']['source']['path'] = b['seed_config']['source']['path'] = 'same-frozen-public-path'
        a['model_configs'][1]['inference_parameters']['max_tokens'] = b['model_configs'][1]['inference_parameters']['max_tokens']
        self.assertEqual(a, b)
        prompt = a['columns'][0]['prompt']
        self.assertIn('Incorrect: "4.62s"', prompt)
        self.assertIn('Preserve its original decimal precision', prompt)
        self.assertEqual(v2.read(p16 / 'prepared_manifest.json')['models_called'], 0)

    def test_targeted_missing_key_cannot_make_api_request_or_create_generation(self):
        target = self.root / 'no_such_prepared_run'
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'NVIDIA_API_KEY is missing'):
                gravity.run_prepared(target)
        self.assertFalse(target.exists())

    def test_gravity_real_sdk_mock_requests_and_explicit_32768_error_no_fallback(self):
        import httpx
        output = gravity.prepare(output=self.root / 'gravity_mock_prepared')
        c = gravity.validated_config()
        seen = []
        seeds = {s['seed_id']: s for s in self.seeds}
        async def responder(transport, request):
            payload = json.loads(request.content)
            seen.append(payload)
            text = payload['messages'][-1]['content']
            if isinstance(text, list): text = ''.join(part.get('text', '') for part in text)
            self.assertIn('query_t MUST be exactly', text)
            self.assertTrue(payload['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(payload['model'], 'nvidia/nemotron-3-ultra-550b-a55b')
            seed_id = text.split('SEED_ID=', 1)[1].splitlines()[0]
            if payload['max_tokens'] == 32768:
                return httpx.Response(400, json={'error': {'message': 'Mock: token budget unsupported', 'type': 'invalid_request_error'}}, request=request)
            proof = reference_proof(self.ns, seeds[seed_id])
            return httpx.Response(200, json={'id': 'cpu-only', 'model': payload['model'], 'object': 'chat.completion', 'created': 1,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps(proof),
                    'reasoning_content': 'UNVERIFIED_MODEL_THINKING_FIXTURE'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130,
                          'completion_tokens_details': {'reasoning_tokens': 20}}}, request=request)
        with patch.dict(os.environ, {'NVIDIA_API_KEY': 'cpu-fixture-not-a-real-key', 'NEMO_TELEMETRY_ENABLED': 'false'}), patch.object(
                httpx.AsyncHTTPTransport, 'handle_async_request', responder):
            p16 = gravity.run_prepared(output / c['conditions']['P16K']['run_id'])
            self.assertTrue(p16['engineering_gate_passed'])
            self.assertEqual(p16['quality']['accepted'], 3)
            p32 = gravity.run_prepared(output / c['conditions']['P32K']['run_id'])
            self.assertFalse(p32['engineering_gate_passed'])
            self.assertEqual(p32['failed_requests'], 1)
        self.assertEqual([r['max_tokens'] for r in seen], [16384, 16384, 16384, 32768])
        for p in output.rglob('*'):
            if p.is_file(): self.assertNotIn(b'cpu-fixture-not-a-real-key', p.read_bytes())


if __name__ == '__main__':
    unittest.main()
