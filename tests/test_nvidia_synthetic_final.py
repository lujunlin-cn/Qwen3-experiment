"""CPU-only final synthetic regression. All HTTPS is mocked; no real secret or GPU.

Fixtures are never experiment evidence. The real pilot's raw mathematical
outputs and read boundaries are tested again by the registered Runner stages.
"""
import ast
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from utils import run_nvidia_final_synthetic as r


class FinalSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard=patch.object(socket.socket,"connect",side_effect=AssertionError("CPU tests forbid network"))
        cls.guard.start();cls.addClassCleanup(cls.guard.stop)
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name);cls.c=r.config();cls.ns=r.load_definitions()
        with r.train_read_boundary(cls.c,[]):cls.manifest,cls.train,_=r.read_train_only(cls.c)
        cls.prior=cls.ns["build_train_only_prior"](cls.train,cls.manifest)
        cls.ph="TEST_MANIFEST_HASH";rng=cls.ns["np"].random.default_rng(20261115)
        cls.candidates=[cls.ns["generate_train_only_candidate"](cls.prior,rng,t,f"{t}-test-{i}",cls.ph) for t in r.TASKS for i in range(20)]
        cls.valid={}
        for s in cls.candidates:
            try:
                v=cls.ns["solver_first_structured_proof"](s)
                if v["solver_proof_accepted"]:cls.valid.setdefault(s["task_type"],(s,v))
            except ValueError:pass
        if set(cls.valid)!=set(r.TASKS):raise RuntimeError("Need real independently valid CPU examples of every task")
        cls.frozen=r.rows(r.ROOT/cls.c["frozen_seed_source"])

    def new_budget(self,name):
        target=self.root/name;target.mkdir();return r.RequestBudget(target,self.c)
    def fixture(self,task="gravity",thinking="UNVERIFIED_TEST_THINKING",finish="stop",usage=True):
        s,v=self.valid[task];seed={**s,**v["sample"]}
        body={"id":"mock-response","object":"chat.completion","model":self.c["model_id"],
            "choices":[{"index":0,"finish_reason":finish,"message":{"role":"assistant","content":v["raw_proof"],"reasoning_content":thinking}}]}
        if usage:body["usage"]={"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"completion_tokens_details":{"reasoning_tokens":30}}
        return seed,body
    def http_fixture(self,seed,body):
        return [{"seed_id":seed["seed_id"],"variant_kind":"trace_variant","http_status":200,"status":"completed",
                 "response":body,"finish_reason":body["choices"][0]["finish_reason"],"usage":body.get("usage"),"latency_seconds":.01}]
    def mock_sdk(self,name,status=200,error=None,unknown_usage=False):
        import httpx
        seed,body=self.fixture(usage=not unknown_usage)
        target=self.root/name;target.mkdir();public=target/"public.jsonl"
        r.write_jsonl_new(public,[{k:seed[k] for k in ("seed_id","task_type","prompt")}])
        builder=self.ns["build_final_data_designer"](public,self.c,"gravity","A")
        calls=[]
        def response(transport,request):
            calls.append(json.loads(request.content))
            if error:raise error
            return httpx.Response(status,json=body if status==200 else {"error":{"message":"mock failure"}},request=request)
        async def async_response(transport,request):
            return response(transport,request)
        with patch.dict(os.environ,{"NVIDIA_API_KEY":"DUMMY_CPU_ONLY_SECRET","NEMO_TELEMETRY_ENABLED":"false"}), \
             patch.object(httpx.HTTPTransport,"handle_request",response), \
             patch.object(httpx.AsyncHTTPTransport,"handle_async_request",async_response):
            try:
                with r.single_request_capture(target,self.c,"A:"+seed["seed_id"],seed,"A",target/"http.jsonl"):
                    r.sdk_generate(builder,target/"sdk")
            except Exception as exc:observed=type(exc).__name__
            else:observed=None
        return target,seed,calls,observed

    def test_01_generator_only_accepts_authenticated_train_prior(self):
        bad=copy.deepcopy(self.prior);bad["source_split"]="validation"
        with self.assertRaises(ValueError):self.ns["generate_train_only_candidate"](bad,self.ns["np"].random.default_rng(1),"gravity","bad",self.ph)
        self.assertEqual(self.prior["source_fields"],["id","prompt","source_row","split","task_type"])

    def test_02_prior_rejects_non_train_rows(self):
        bad=self.train.copy();bad.loc[0,"split"]="holdout_test"
        with self.assertRaises(ValueError):self.ns["build_train_only_prior"](bad,self.manifest)

    def test_03_answer_column_cannot_change_prior_or_generation(self):
        a=self.train.copy();a["answer"]="forbidden secret labels"
        b=self.train.copy();b["answer"]="totally changed"
        pa=self.ns["build_train_only_prior"](a,self.manifest);pb=self.ns["build_train_only_prior"](b,self.manifest)
        self.assertEqual(pa,pb);self.assertEqual(pa,self.prior)
        for task in r.TASKS:
            self.assertEqual(self.ns["generate_train_only_candidate"](pa,self.ns["np"].random.default_rng(99),task,"same",self.ph),
                             self.ns["generate_train_only_candidate"](pb,self.ns["np"].random.default_rng(99),task,"same",self.ph))

    def test_04_missing_train_statistics_fails(self):
        for task,feature in (("numeral_system","numbers"),("gravity","times"),("equation_numeric","operators")):
            bad=copy.deepcopy(self.prior);bad["features"][task][feature]=[]
            bad.pop("prior_content_sha256");bad["prior_content_sha256"]=self.ns["_prior_hash"](bad)
            with self.assertRaises(ValueError):self.ns["generate_train_only_candidate"](bad,self.ns["np"].random.default_rng(99),task,"bad",self.ph)

    def test_05_no_global_operator_fallback(self):
        source=r.NOTEBOOK.read_text()
        self.assertNotIn("train_numeric_operator_pool or",source)
        self.assertEqual(self.ns["numeric_operator_pool"],())
        with self.assertRaises(ValueError):self.ns["generate_equation_numeric_sample"](self.ns["np"].random.default_rng(1))

    def test_06_every_candidate_has_complete_empirical_lineage(self):
        for s in self.candidates:
            audit=self.ns["audit_train_prior_lineage"](s,self.prior,self.ph)
            self.assertTrue(audit["train_only_provenance"],audit)
            self.assertTrue(s["prior_source_ids"]);self.assertTrue(s["prior_source_rows"])
            self.assertEqual(audit["lineage_status"],"declared")

    def test_07_forged_source_or_missing_feature_rejected(self):
        seed=self.candidates[0]
        for mode in ("id","row","feature","manifest","spec"):
            bad=copy.deepcopy(seed);key=next(iter(bad["prior_feature_lineage"]))
            if mode=="id":bad["prior_feature_lineage"][key]["source_id"]="NON_TRAIN_ID"
            if mode=="row":bad["prior_feature_lineage"][key]["source_row"]="NON_TRAIN_ROW"
            if mode=="feature":bad["prior_feature_lineage"].pop(key)
            if mode=="manifest":bad["prior_manifest_sha256"]="forged"
            if mode=="spec":bad["task_spec_prior_sha256"]="forged"
            self.assertFalse(self.ns["audit_train_prior_lineage"](bad,self.prior,self.ph)["train_only_provenance"],mode)

    def test_08_generator_cannot_read_evaluation_or_other_real_files(self):
        events=[]
        with r.train_read_boundary(self.c,events):
            for name in ("validation","holdout_test"):
                path=r.ROOT/self.manifest["files"][name]
                with self.assertRaises(PermissionError):path.read_text()
                with self.assertRaises(PermissionError):open(path).close()
            r.load_definitions()
        self.assertEqual(events,[])

    def test_09_audit_cannot_load_splits_before_pool_freeze(self):
        missing=self.root/"no_freeze";missing.mkdir()
        with patch.object(r,"verify_run",return_value=(self.c,{})),patch.object(r,"load_definitions",side_effect=AssertionError("must not load")):
            with self.assertRaises(FileNotFoundError):r.audit(missing)

    def test_10_hashes_match_and_corruption_fails(self):
        prior_hash=dict(self.prior);expected=prior_hash.pop("prior_content_sha256")
        self.assertEqual(expected,self.ns["_prior_hash"](prior_hash))
        t=self.root/"frozen";t.mkdir();r.write_jsonl_new(t/"candidate_pool_pre_audit.jsonl",self.candidates)
        r.write_json_new(t/"candidate_pool_freeze.json",{"candidate_pool_sha256":r.file_hash(t/"candidate_pool_pre_audit.jsonl")})
        r.verify_candidate_freeze(t)
        with (t/"candidate_pool_pre_audit.jsonl").open("a") as f:f.write("{}\n")
        with self.assertRaises(ValueError):r.verify_candidate_freeze(t)

    def test_11_roman_solver_verifier_positive_and_negative(self):self.check_math("numeral_system")
    def test_12_gravity_solver_verifier_positive_and_negative(self):self.check_math("gravity")
    def test_13_equation_solver_verifier_positive_and_negative(self):self.check_math("equation_numeric")
    def check_math(self,task):
        s,v=self.valid[task];seed={**s,**v["sample"]}
        self.assertTrue(self.ns["verify_nvidia_trace_v2"](seed,v["raw_proof"])["verified"])
        proof=copy.deepcopy(v["proof"])
        if task=="numeral_system":proof["components"][0]["arabic"]+=1
        elif task=="gravity":proof["g_estimates"][0]*=2
        else:proof["example_outputs"][0]="wrong"
        self.assertFalse(self.ns["verify_nvidia_trace_v2"](seed,json.dumps(proof))["verified"])

    def test_14_equation_global_constraints_and_uniqueness(self):
        s,v=self.valid["equation_numeric"]
        bad=self.ns["_numeric_prompt_text"]([("12+34","46")],"22@33")
        self.assertFalse(self.ns["_smoke_equation_resolution"](bad)["answer_unique"])
        self.assertFalse(self.ns["_numeric_verifier_valid_assignments"]({"+" : ["add"], "@" : ["add_p1"]},["+","@"]))
        self.assertTrue(v["verification"]["field_audit"]["answer_unique"])

    def test_15_exactly_twenty_candidates_per_task_without_resampling(self):
        self.assertEqual(len(self.candidates),60)
        self.assertEqual({t:sum(s["task_type"]==t for s in self.candidates) for t in r.TASKS},{t:20 for t in r.TASKS})
        self.assertEqual(len({s["candidate_id"] for s in self.candidates}),60)

    def test_16_real_duplicates_all_three_splits_are_rejected(self):
        import pandas as pd
        s,v=self.valid["gravity"];s={**s,**v["sample"]}
        for split in ("train","validation","holdout_test"):
            frame=pd.DataFrame([{**s,"id":"cloned","source_row":"999","split":split}])
            index=self.ns["build_real_index"](frame)
            a=self.ns["audit_synthetic_leakage"](s,index)
            self.assertEqual(a["status"],"rejected")
            self.assertIn("real_"+split+"_exact_duplicate",a["rejection_reasons"])

    def test_17_strict_roman_mapping_target_exposure(self):
        import pandas as pd
        s,v=self.valid["numeral_system"];s={**s,**v["sample"]}
        pairs,q=self.ns["parse_numeral_prompt"](s["prompt"])
        for split in ("validation","holdout_test"):
            frame=pd.DataFrame([{"id":"target","source_row":"999","split":split,"task_type":"numeral_system",
                "prompt":f"1 -> I\nNow, write the number {pairs[0][0]} in the Wonderland numeral system.","answer":pairs[0][1]}])
            a=self.ns["audit_final_seed_leakage"](s,self.ns["build_real_index"](frame),{"lineage_status":"declared","train_only_provenance":True})
            self.assertTrue(a["direct_mapping_leakage"]);self.assertEqual(a["status"],"rejected")

    def test_18_structure_similarity_is_report_only_and_prior_sources_are_not_clones(self):
        import pandas as pd
        s,v=self.valid["gravity"];s={**s,**v["sample"]}
        other=s["prompt"].replace("For t =", "For t =",1).replace("distance = 0.5*g*t^2","distance = 0.5*g*t^2")
        other=__import__('re').sub(r'(?<=distance = )\d+',"123",other)
        ref={"id":s["prior_source_ids"][0],"source_row":s["prior_source_rows"][0],"split":"train","task_type":"gravity","prompt":other,"answer":"0"}
        a=self.ns["audit_synthetic_leakage"](s,self.ns["build_real_index"](pd.DataFrame([ref])))
        self.assertEqual(a["status"],"passed");self.assertTrue(a["structure_similar_only"])
        self.assertEqual(a["matches"]["source_id"],[])

    def test_19_ab_only_prompt_changes_and_compact_is_shorter(self):
        for task in r.TASKS:
            builders=[]
            for arm in ("A","B"):
                d=self.root/(task+arm);d.mkdir();f=d/"public.jsonl"
                seed=next(s for s in self.frozen if s["task_type"]==task)
                r.write_jsonl_new(f,[{k:seed[k] for k in ("seed_id","task_type","prompt")}])
                b=self.ns["build_final_data_designer"](f,self.c,task,arm).build().model_dump(mode="json")
                builders.append(b)
            a,b=builders
            pa=a["columns"][0].pop("prompt");pb=b["columns"][0].pop("prompt")
            a.pop("seed_config");b.pop("seed_config")
            self.assertEqual(a,b);self.assertLess(len(pb),len(pa))
            if task=="equation_numeric":
                for rule in self.ns["NUMERIC24_RULES"]:self.assertIn('"'+rule["name"]+'"',pb)
                for constraint in self.ns["EQUATION_NUMERIC_PRIOR"]["global_constraints"]:self.assertIn(constraint,pb)

    def test_20_schemas_identical_to_original_closed_schema(self):
        for task in r.TASKS:
            schema=self.ns["smoke_trace_schema"](task)
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]),set(schema["properties"]))
            seed,v=self.valid[task]
            self.assertTrue(self.ns["verify_nvidia_trace_v2"]({**seed,**v["sample"]},v["raw_proof"])["schema_valid"])

    def test_21_low_effort_in_real_sdk_http_request(self):
        target,seed,calls,error=self.mock_sdk("sdk_low")
        self.assertEqual(len(calls),1);self.assertEqual(calls[0]["reasoning_effort"],"low")
        self.assertEqual(calls[0]["model"],self.c["model_id"])
        screened=r.screen_entry(self.ns,seed,r.rows(target/"http.jsonl"))
        self.assertTrue(screened["llm_proof_accepted"])

    def test_22_duplicate_send_blocked_durably(self):
        b=self.new_budget("duplicate");b.reserve("A:seed",100,"req")
        with self.assertRaises(r.DuplicateRequestBlocked):r.RequestBudget(b.path.parent,self.c).reserve("A:seed",100,"req")
        self.assertEqual(b.state()["request_count"],1)

    def test_23_http500_disconnect_and503_are_not_retried(self):
        import httpx
        for name,status,error in (("sdk_500",500,None),("sdk_503",503,None),("sdk_disconnect",200,httpx.ReadError("mock disconnect"))):
            target,seed,calls,observed=self.mock_sdk(name,status,error)
            self.assertEqual(len(calls),1,name)
            self.assertFalse(r.screen_entry(self.ns,seed,r.rows(target/"http.jsonl"))["llm_proof_accepted"])
            self.assertEqual(r.RequestBudget(target,self.c).state()["unknown_usage_requests"],1)

    def test_24_unknown_usage_is_not_zero(self):
        seed,body=self.fixture(usage=False);entry=self.http_fixture(seed,body)[0]
        self.assertIsNone(r.usage_fields(entry)["completion_tokens"])
        self.assertIsNone(r.usage_fields(entry)["reasoning_tokens"])
        b=self.new_budget("unknown");b.reserve("x",32768,"req");b.settle("x",32768,None)
        self.assertEqual(b.state()["completion_tokens_charged_or_reserved"],32768)
        self.assertEqual(b.state()["unknown_usage_requests"],1)

    def test_25_hard_token_and_request_budgets(self):
        b=self.new_budget("cap")
        for i in range(15):b.reserve(str(i),32768,"req");b.settle(str(i),32768,None)
        with self.assertRaises(r.BudgetStop):b.reserve("overflow",32768,"req")
        self.assertEqual(b.state()["request_count"],15)
        self.assertLessEqual(b.state()["completion_tokens_charged_or_reserved"],500000)
        b=self.new_budget("request_cap")
        for i in range(48):b.reserve(str(i),1,"req");b.settle(str(i),1,{"completion_tokens":1})
        with self.assertRaises(r.BudgetStop):b.reserve("49",1,"req")

    def test_26_thinking_cannot_change_math_acceptance(self):
        for thinking in (None,"", "UNVERIFIED_TEST_THINKING"):
            seed,body=self.fixture(thinking=thinking)
            self.assertTrue(r.screen_entry(self.ns,seed,self.http_fixture(seed,body))["llm_proof_accepted"])
        seed,body=self.fixture(finish="length")
        self.assertFalse(r.screen_entry(self.ns,seed,self.http_fixture(seed,body))["llm_proof_accepted"])

    def test_27_missing_key_no_request_and_no_key_in_archives(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):r.require_api_key()
        target,seed,calls,error=self.mock_sdk("secret_scan")
        for p in target.rglob("*"):
            if p.is_file():
                body=p.read_bytes()
                self.assertNotIn(b"DUMMY_CPU_ONLY_SECRET",body,p)
                self.assertNotIn(b'"authorization"',body.lower(),p)

    def test_28_29_30_31_32_renderer_maps_all_tasks_boxed_no_hidden_or_thinking_deterministic(self):
        # Actual local tokenizer integration runs in its installed CPU environment.
        fixture=[]
        for task,(seed,verified) in self.valid.items():
            fixture.append({"seed":{**seed,"verification_status":"accepted","leakage_audit":{"status":"passed"},
                "reasoning_content":"UNVERIFIED_TEST_THINKING","hidden_rule":"DO_NOT_RENDER"},"verified":verified})
        d=self.root/"render";d.mkdir();r.write_json_new(d/"prior.json",self.prior);r.write_jsonl_new(d/"fixture.jsonl",fixture)
        code='''
import json,socket
from pathlib import Path
from unittest.mock import patch
from utils import run_nvidia_final_synthetic as r
from utils.run_nvidia_v2 import local_tokenizer
p=Path(__import__('sys').argv[1]);n=r.load_definitions();s=r.load_renderer();tokenizer,meta=local_tokenizer()
assert tokenizer is not None
with patch.object(socket.socket,'connect',side_effect=AssertionError('no network')):
 for row in r.rows(p/'fixture.jsonl'):
  seed=row['seed'];v=row['verified'];prior=r.read(p/'prior.json')
  a,audit=s['render_solver_first_preview'](seed,v,n,prior,'TEST_MANIFEST_HASH',r.sft_training_tasks(),tokenizer)
  b,_=s['render_solver_first_preview'](seed,v,n,prior,'TEST_MANIFEST_HASH',r.sft_training_tasks(),tokenizer)
  assert a==b and audit['sft_render_audit_passed']
  assert 'Final answer: \\\\boxed{'+a['final_answer']+'}' in a['completion']
  assert 'UNVERIFIED_TEST_THINKING' not in json.dumps(a) and 'DO_NOT_RENDER' not in json.dumps(a)
  assert a['actual_token_counts']['completion_tokens']==len(tokenizer(a['completion'],add_special_tokens=False).input_ids)
  assert a['sft_training_eligible']==(seed['task_type']!='numeral_system')
print('3/3 actual tokenizer renderer checks passed')
'''
        env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",CUDA_VISIBLE_DEVICES="",TOKENIZERS_PARALLELISM="false")
        env.pop("NVIDIA_API_KEY",None)
        run=subprocess.run([str(r.ROOT/self.c["tokenizer_python"]),"-c",code,str(d)],cwd=r.ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)

    def test_33_roman_not_in_default_task_pool(self):
        self.assertEqual(r.sft_training_tasks(),("gravity","equation_numeric","bit_manipulation"))

    def test_34_sft_default_behavior_unchanged(self):
        original=json.loads(subprocess.check_output(["git","show","e305a47:03_SFT.ipynb"],cwd=r.ROOT,text=True))
        current=r.read(r.SFT)
        by_id={c.get("id"):c for c in current["cells"]}
        for cell in original["cells"]:
            self.assertEqual(cell,by_id[cell.get("id")])

    def test_35_36_37_historical_archives_are_identical_to_git(self):
        paths=[self.c["high_baseline_r4"],"data/processed/round3/nvidia_hosted_20261115_ultra_r4_reaudit_v2",
               "data/processed/round3/nvidia_gravity_targeted_v2_20261115_live_r1",
               "data/processed/round3/nvidia_gravity_targeted_v2_20261115_live_r2_continuation"]
        tracked=subprocess.check_output(["git","ls-tree","-r","--name-only","e305a47","--",*paths],cwd=r.ROOT,text=True).splitlines()
        self.assertTrue(tracked)
        for name in tracked:
            baseline=subprocess.check_output(["git","show","e305a47:"+name],cwd=r.ROOT)
            self.assertEqual((r.ROOT/name).read_bytes(),baseline,name)

    def test_38_no_gpu_training_or_model_load_entrypoint(self):
        tree=ast.parse(Path(r.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,{"cuda","backward","fit","train"})
            if isinstance(node,ast.Import):self.assertFalse(any(a.name=="torch" for a in node.names))

    def test_39_immutable_outputs_and_no_sft_merge(self):
        f=self.root/"immutable.json";r.write_json_new(f,{"status":"original"})
        with self.assertRaises(FileExistsError):r.write_json_new(f,{"status":"overwrite"})
        self.assertFalse(self.c["merge_into_sft"]);self.assertFalse(self.c["expand_to_100"])
        for name in ("empty","partial"):
            out=r.summarize_proofs([] if name=="empty" else [{"task_type":"gravity","llm_proof_accepted":False}])
            self.assertEqual(out["gravity"]["accepted"],0)
            self.assertIsNone(out["gravity"]["tokens_per_accepted_proof"])

    def test_40_dpo_ppo_grpo_unchanged(self):
        for name in r.PROTECTED:
            baseline=subprocess.check_output(["git","show","e305a47:"+name],cwd=r.ROOT)
            self.assertEqual((r.ROOT/name).read_bytes(),baseline,name)

    def test_selection_thresholds_missing_usage_and_no_cherry_picking(self):
        base=[{"task_type":t,"llm_proof_accepted":True,"reasoning_tokens":100,"completion_tokens":120} for t in r.TASKS for _ in range(3)]
        outputs=[]
        for arm,reason,completion in (("A",None,20),("B",50,70)):
            outputs.extend({"task_type":t,"condition":arm,"llm_proof_accepted":True,"reasoning_tokens":reason,"completion_tokens":completion} for t in r.TASKS for _ in range(3))
        result=r.select_conditions(self.c,base,outputs)
        for d in result["decisions"].values():
            self.assertEqual(d["selected_condition"],"A")
            self.assertFalse(d["new_seed_llm_gate_passed"]) # Missing reasoning cost cannot satisfy reduction.
            self.assertIsNone(d["reasoning_reduction"])
        for row in outputs:
            if row["condition"]=="A":row["reasoning_tokens"]=79
        result=r.select_conditions(self.c,base,outputs)
        self.assertTrue(all(d["new_seed_llm_gate_passed"] for d in result["decisions"].values()))


if __name__=="__main__":unittest.main()
