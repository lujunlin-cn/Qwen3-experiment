"""Read-only integration checks of the real frozen CPU production artifacts."""
import ast,json,unittest
from pathlib import Path
from unittest.mock import patch
from utils.run_equation_augmentation import RUN,OLD,ROOT,RECOVERED,read,rows,sha,definitions,generation_boundary

class ActualArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.n,cls.s=definitions()
 def test_frozen_500_reused_without_resampling(self):
  f=read(RUN/'candidate_pool_freeze.json');self.assertEqual(len(rows(RUN/'equation_candidates.jsonl')),500)
  self.assertEqual(sha(RUN/'equation_candidates.jsonl'),f['candidate_sha256'])
  parent=RUN.parent/'equation_sft_augmentation_20261116_r1'
  self.assertEqual((RUN/'equation_candidates.jsonl').read_bytes(),(parent/'equation_candidates.jsonl').read_bytes())
 def test_384_accepted_all_gate_records(self):
  report=read(RUN/'equation_solver_audit.json');self.assertEqual(report['accepted'],384);self.assertEqual(report['rejected'],126)
  self.assertTrue(all(r['lineage']['train_only_provenance'] for r in report['records']))
  for r in report['records']:
   if r['passed']:
    self.assertTrue(r['agreement']['passed']);self.assertTrue(r['render']['sft_render_audit_passed'])
    self.assertTrue(all(v['status']=='passed' for v in r['leakage'].values()))
 def test_exact_certificate_for_all_accepted(self):
  for r in rows(RUN/'equation_accepted.jsonl'):
   v=self.n['verify_equation_numeric_sample'](r)
   self.assertTrue(v['answer_ok'] and v['trace_ok'],r['candidate_id'])
 def test_ambiguity_is_not_solver_verifier_disagreement(self):
  from utils.run_equation_augmentation import training_gate
  seed=next(r for r in rows(RUN/'equation_rejected.jsonl') if r['deduce_or_guess']=='guess')
  prior=read(OLD/'train_prior.json');ph=sha(OLD/'train_prior_manifest.json')
  gate=training_gate(seed,self.n,{},prior,ph)
  self.assertTrue(gate['agreement']['passed']);self.assertFalse(gate['proof']['solver_proof_accepted'])
 def test_boxed_roundtrip_all604(self):
  a=read(RUN/'sft_boxed_answer_roundtrip.json');self.assertEqual(a['status'],'passed');self.assertEqual(a['records_checked'],604)
 def test_gravity_record_bytes_unchanged(self):
  old=[l for l in (OLD/'sft_rendered_preview.jsonl').read_text().splitlines() if json.loads(l)['task_type']=='gravity']
  new=[l for l in (RUN/'sft_synthetic_110.jsonl').read_text().splitlines() if json.loads(l)['task_type']=='gravity']
  self.assertEqual(old,new)
 def test_control_ids_and_token_budget(self):
  a=rows(RUN/'sft_synthetic_110.jsonl');b=rows(RUN/'sft_real_control_110.jsonl');used={r['source_id'] for r in rows(RECOVERED/'panels/sft.jsonl')}
  self.assertEqual(len(a),110);self.assertEqual(len(b),110);self.assertFalse(used&{r['source_id'] for r in b})
  self.assertLessEqual(read(RUN/'sft_data_manifest.json')['token_budget']['relative_difference'],.01)
 def test_generator_file_boundary(self):
  with generation_boundary([]):
   for p in [ROOT/'data/processed/tasksets/roman_gravity_equation_v1/splits/validation.csv',ROOT/'data/processed/splits/holdout_test.csv']:
    with self.assertRaises(PermissionError):p.read_text()
 def test_roman_preview_only(self):
  data=rows(RUN/'roman_prompt_variant_preview.jsonl');self.assertTrue(data)
  for r in data:
   self.assertEqual(r['prompt'].encode().split(b'\n',1)[1],r['original_prompt'].encode());self.assertFalse(r['sft_training_eligible'])
   self.assertFalse(r['new_mapping_knowledge']);self.assertTrue(all(r['checks'].values()));self.assertTrue(r['leakage_audits'])
 def test_official_scoring_and_protected_algorithms(self):
  a=read(RUN/'initial_state.json');self.assertTrue(all(sha(ROOT/p)==h for p,h in a['protected_sha256'].items()))
  self.assertTrue(self.n['numeric_or_text_match']('100','101'))
 def test_no_hosted_api_entrypoint(self):
  for f in ['utils/run_equation_augmentation.py','utils/run_sft_augmentation.py','utils/evaluate_sft_augmentation_vllm.py']:
   src=(ROOT/f).read_text();self.assertNotIn('require_api_key(',src);self.assertNotIn('NVIDIA_API_KEY',src);self.assertNotIn('integrate.api.nvidia.com',src)

if __name__=='__main__':unittest.main()
