"""CPU certificates for the registered natural-language Gravity DPO-B run."""
import ast,copy,hashlib,json,math,pathlib,tempfile,unittest
from unittest.mock import patch
from utils.dpo_gravity_worker import nl_api,pair_api,ROOT,PARENT,BASE,sha

PROMPT='For t = 1s, distance = 5 m\nFor t = 2s, distance = 20 m\nNow, determine the falling distance for t = 3s given d = 0.5*g*t^2.'
PROOF='''1. Parse all public time-distance examples.
2. Fit g = 2*sum(t^2*d)/sum(t^4) using only those examples.
3. The fitted g is 10; apply distance = 0.5*g*t^2 to Query t=3.
4. The unrounded Query distance is 45; format the final numeric answer.
Final answer: \\boxed{45.00}'''

class NaturalGravity(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.ns=nl_api();cls.verify=staticmethod(cls.ns['verify_gravity_natural_language'])
 def check(self,text=PROOF,prompt=PROMPT,answer='45.00',finish='stop'):return self.verify(prompt,text,answer,finish)
 def test_canonical_sft_format_without_per_example_g(self):self.assertTrue(self.check()['verified'])
 def test_paraphrase_is_not_full_string_equality(self):
  text=PROOF.replace('Parse','Read').replace('format the final numeric answer','then round the result to two decimals')
  self.assertTrue(self.check(text)['verified'])
 def test_thinking_ignored_and_not_certified(self):
  a=self.check('<think>wrong: 1+1=99</think>'+PROOF)
  self.assertTrue(a['verified']);self.assertFalse(a['thinking_verified'])
 def test_correct_thinking_does_not_rescue_wrong_final(self):self.assertFalse(self.check('<think>'+PROOF+'</think>Final answer: \\boxed{45}')['verified'])
 def test_unclosed_thinking_rejected(self):self.assertFalse(self.check('<think>'+PROOF)['verified'])
 def test_wrong_fitted_g(self):self.assertFalse(self.check(PROOF.replace('g is 10','g is 10.01'))['verified'])
 def test_one_percent_official_answer_does_not_certify_wrong_process(self):
  a=self.check(PROOF.replace('g is 10','g is 10.05').replace('distance is 45','distance is 45.225'))
  self.assertTrue(a['final_correct']);self.assertFalse(a['verified'])
 def test_wrong_query_time(self):self.assertFalse(self.check(PROOF.replace('Query t=3','Query t=2'))['verified'])
 def test_wrong_unrounded_distance(self):self.assertFalse(self.check(PROOF.replace('distance is 45','distance is 44'))['verified'])
 def test_missing_physics(self):self.assertFalse(self.check(PROOF.replace('distance = 0.5*g*t^2','distance using a formula'))['verified'])
 def test_missing_raw_is_unverifiable(self):self.assertFalse(self.check(PROOF.replace('The unrounded Query distance is 45','Compute the number'))['verified'])
 def test_final_only_not_proof(self):self.assertFalse(self.check('Final answer: \\boxed{45.00}')['verified'])
 def test_length_finish_rejected(self):self.assertFalse(self.check(finish='length')['verified'])
 def test_incorrect_extra_arithmetic(self):self.assertFalse(self.check(PROOF+'\n2*10=99')['verified'])
 def test_valid_extra_arithmetic(self):self.assertTrue(self.check(PROOF+'\n0.5*10*(3)^2=45')['verified'])
 def test_bad_symbolic_arithmetic(self):self.assertFalse(self.check(PROOF+'\nt^2=999')['verified'])
 def test_good_symbolic_arithmetic(self):self.assertTrue(self.check(PROOF+'\nt^2=9')['verified'])
 def test_modified_public_example(self):self.assertFalse(self.check(PROOF+'\nFor t = 1, d = 999')['verified'])
 def test_wrong_fitted_g_in_additional_sentence(self):self.assertFalse(self.check(PROOF+'\nThe fitted g is 999.')['verified'])
 def test_wrong_arithmetic_with_trailing_prose(self):self.assertFalse(self.check(PROOF+'\n2*10=99 is the checked result.')['verified'])
 def test_input_opened_thinking_requires_response_close(self):self.assertFalse(self.verify(PROMPT,PROOF,'45',thinking_expected=True)['verified'])
 def test_contradictory_additional_physics(self):self.assertFalse(self.check(PROOF+'\nd=g*t^3')['verified'])
 def test_numeric_query_substitution_instead_of_prose(self):self.assertTrue(self.check(PROOF.replace('The unrounded Query distance is 45; format the final numeric answer.','d = 0.5*10*(3)^2 = 45'))['verified'])
 def test_multiple_boxes(self):self.assertFalse(self.check(PROOF+'\n\\boxed{45}')['verified'])
 def test_solver_not_called(self):
  self.ns['solve_gravity_prompt']=lambda *a:(_ for _ in ()).throw(AssertionError('Solver called'))
  self.assertTrue(self.check()['verified'])
 def test_no_answer_in_public_reference(self):self.assertEqual(self.ns['gravity_nl_public_reference'](PROMPT)['g_ols'],10)
 def test_safe_math_no_code_execution(self):
  with self.assertRaises(ValueError):self.ns['_gravity_nl_eval']('__import__("os").system("true")')
 def test_synthetic_query_wording_supported(self):self.assertTrue(self.check(prompt=PROMPT.replace('determine','calculate'))['verified'])
 def test_actual_ten_gravity_teachers(self):
  p=ROOT/'data/processed/round4/equation_sft_augmentation_20261116_r2/sft_synthetic_110.jsonl'
  rs=[json.loads(l) for l in p.read_text().splitlines()];rs=[r for r in rs if r['task_type']=='gravity']
  self.assertEqual(len(rs),10)
  for r in rs:self.assertTrue(self.check(r['completion'],r['prompt'],r['final_answer'])['verified'],r['source_id'])

class PairsAndIsolation(unittest.TestCase):
 def row(self,i,n,**changes):
  r={'task_type':'gravity','sample_id':'a','prompt_token_ids':[1,2],'policy_version':'SFT_C:hash','completion_token_ids':[i]*n,
     'correct':True,'truncated':False,'finish_reason':'stop','group_index':i,'proof_audit':{'verified':True}}
  r.update(changes);return r
 def pair(self,rs):return pair_api()(rs)
 def test_shortest_and_longest_verified_only(self):
  ps=self.pair([self.row(1,100),self.row(2,200),self.row(3,400),self.row(4,600,correct=False)])
  self.assertEqual(len(ps),1);self.assertEqual(ps[0]['chosen']['group_index'],1);self.assertEqual(ps[0]['rejected']['group_index'],3)
 def test_no_wrong_or_truncated_rejected(self):
  for change in [{'correct':False},{'truncated':True},{'finish_reason':'length'},{'proof_audit':{'verified':False}}]:self.assertEqual(self.pair([self.row(1,100),self.row(2,500,**change)]),[])
 def test_gap64(self):self.assertEqual(self.pair([self.row(1,100),self.row(2,163)]),[])
 def test_fraction20(self):self.assertEqual(self.pair([self.row(1,1000),self.row(2,1200)]),[])
 def test_prompt_mismatch_raises(self):
  with self.assertRaises(ValueError):self.pair([self.row(1,100),self.row(2,300,prompt_token_ids=[3])])
 def test_policy_mismatch_raises(self):
  with self.assertRaises(ValueError):self.pair([self.row(1,100),self.row(2,300,policy_version='old')])
 def test_gravity_only(self):
  with self.assertRaises(ValueError):self.pair([self.row(1,100,task_type='equation_numeric')])
 def test_config_matches_user(self):
  c=json.loads((ROOT/'configs/dpo_gravity_sft_c_20261116.json').read_text())
  self.assertEqual((c['dpo']['beta'],c['dpo']['learning_rate'],c['dpo']['epochs'],c['dpo']['gradient_accumulation']),(.1,2e-6,2,4))
  self.assertEqual(c['sampling']['max_tokens'],16384);self.assertEqual(c['device_limit_bytes'],8000000000)
 def test_real_parent_hash(self):self.assertEqual(sha(PARENT/'adapter_model.safetensors'),'7033241d41efb086762df20789ce2b70764bb3a5c524e29b8d8f846d3254e3f4')
 def test_no_hosted_api_or_legacy_entry(self):
  for name in ['utils/dpo_gravity_worker.py','utils/run_dpo_gravity_sft_c.py']:
   t=(ROOT/name).read_text()
   self.assertNotIn('NVIDIA_API_KEY',t);self.assertNotIn('integrate.api.nvidia.com',t);self.assertNotIn('run_dpo_smoke_pipeline(',t)
 def test_response_tokens_include_thinking_and_eos(self):
  r=self.row(1,300);r['completion_token_ids']=[1]*299+[151645]
  self.assertEqual(len(r['completion_token_ids']),300)

if __name__=='__main__':unittest.main()
