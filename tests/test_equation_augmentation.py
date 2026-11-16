"""Exact-string Numeric24 regression; no network, API credentials, or GPU."""
import ast,copy,json,socket,unittest
from unittest.mock import patch
from utils.run_nvidia_final_synthetic import load_definitions,load_renderer

class ExactEquationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.n=load_definitions();cls.s=load_renderer()
 def prompt(self,pairs,q):return self.n['_numeric_prompt_text'](pairs,q)
 def sample(self,pairs,q):
  s={'task_type':'equation_numeric','prompt':self.prompt(pairs,q),'difficulty':'deduce_exact'}
  self.n['_solve_candidate'](s);return s
 def test_tolerance_counterexample(self):
  p=self.prompt([('50+50','100')],'51+50');r=self.n['solve_equation_numeric_prompt'](p)
  self.assertEqual(r['candidates_by_operator'],{'+':['add']});self.assertEqual(r['query_outputs'],['101'])
  self.assertTrue(self.n['numeric_or_text_match']('100','101'))
 def test_unique_rules_and_full_trace(self):
  s=self.sample([('50+50','100')],'51+50');v=self.n['verify_equation_numeric_sample'](s)
  self.assertTrue(v['trace_ok']);self.assertTrue(v['answer_ok']);self.assertTrue(v['identifiability']['rule_assignment_unique'])
 def test_multiple_assignments_same_answer(self):
  r=self.n['solve_equation_numeric_prompt'](self.prompt([('22+22','2222')],'33+33'))
  self.assertEqual(r['query_outputs'],['3333']);self.assertTrue(r['answer_unique']);self.assertFalse(r['rule_assignment_unique'])
  self.assertEqual({a['rules']['+'] for a in r['valid_assignments']},{'join','join_swap'})
 def test_multiple_assignments_different_answers(self):
  r=self.n['solve_equation_numeric_prompt'](self.prompt([('22+22','2222')],'33+44'))
  self.assertFalse(r['answer_unique']);self.assertEqual(r['query_outputs'],['3344','4433']);self.assertIsNone(r['selected_assignment'])
 def test_leading_zero_string_identity(self):
  for name in ['numeric_rule_output','_numeric_truth_output','_numeric_verify_output']:
   self.assertEqual(self.n[name]('join','01+02'),'0102')
  self.assertNotIn('join',self.n['_numeric_candidate_rules']([('01+02','102')],'+'))
 def test_prefix_suffix_sign_and_reverse(self):
  for rule,expr,expected in [('sub','12-34','-22'),('sub','12@34','@22'),('absdiff_prefix','12@34','@22'),('flip_absdiff_suffix','21@43','22@'),('flip_add','10+20','3'),('join','00+00','0000')]:
   for fn in ['numeric_rule_output','_numeric_truth_output','_numeric_verify_output']:
    self.assertEqual(self.n[fn](rule,expr),expected,(fn,rule))
 def test_near_outputs_not_equal(self):
  s=self.sample([('50+50','100')],'51+50');s['answer']='102';self.assertFalse(self.n['verify_equation_numeric_sample'](s)['answer_ok'])
 def test_undefined_operation(self):
  for fn in ['numeric_rule_output','_numeric_truth_output','_numeric_verify_output']:
   self.assertIsNone(self.n[fn]('maxmodmin','00+12'))
  self.assertEqual(self.n['solve_equation_numeric_prompt']('bad query')['status'],'invalid_prompt')
 def test_global_group_and_family_conflicts(self):
  for fn in ['_numeric_valid_assignments','_numeric_verifier_valid_assignments']:
   self.assertFalse(self.n[fn]({'+':['add'],'@':['add_p1']},['+','@']))
   self.assertFalse(self.n[fn]({'+':['add'],'@':['flip_mul']},['+','@']))
 def test_corrupted_example_and_query(self):
  s=self.sample([('50+50','100')],'51+50')
  for key in ['computed','expected','rule']:
   b=copy.deepcopy(s);b['trace']['example_checks'][0][key]='wrong'
   self.assertFalse(self.n['verify_equation_numeric_sample'](b)['trace_ok'])
  b=copy.deepcopy(s);b['trace']['query_outputs']=['102'];self.assertFalse(self.n['verify_equation_numeric_sample'](b)['trace_ok'])
 def test_same_assignment_count_different_set_rejected(self):
  s=self.sample([('50+50','100')],'51+50');s['trace']['valid_assignments'][0]['rules']['+']='add_p1'
  self.assertFalse(self.n['verify_equation_numeric_sample'](s)['trace_ok'])
 def test_solver_verifier_disagreement_detected(self):
  p=self.prompt([('50+50','100')],'51+50')
  with patch.dict(self.n,{'_numeric_candidate_rules':lambda *a:['add_p1']}):
   self.assertFalse(self.n['audit_equation_agreement'](p)['passed'])
 def test_verifier_independence(self):
  p=self.prompt([('50+50','100')],'51+50')
  with patch.dict(self.n,{k:lambda *a: (_ for _ in ()).throw(AssertionError('solver used')) for k in ['numeric_rule_output','_numeric_candidate_rules','_numeric_valid_assignments','_parse_numeric_prompt','_numeric_format','_numeric_operation']}):
   self.assertEqual(self.n['verify_equation_numeric_prompt'](p)['query_outputs'],['101'])
 def test_differential_all_rules_all_operand_pairs(self):
  # 24 rules x 10,000 pairs: real arithmetic/string comparison, not score tolerance.
  for a in range(100):
   for b in range(100):
    e=f'{a:02d}@{b:02d}'
    for r in self.n['NUMERIC24_RULES']:
     name=r['name'];x=self.n['numeric_rule_output'](name,e)
     self.assertEqual(x,self.n['_numeric_verify_output'](name,e),(name,e))
     self.assertEqual(x,self.n['_numeric_truth_output'](name,e),(name,e))
 def test_render_all_examples_assignments_unique_claim(self):
  s=self.sample([('22+22','2222')],'33+33');s['verification_status']='accepted'
  s,a=self.s['strict_equation_render_reference'](s,self.n)
  text=self.s['_format_trace'](s)
  self.assertIn('22+22',text);self.assertIn('rule_assignment_unique=False',text);self.assertIn('answer_unique=True',text)
  self.assertIn('Final answer: \\boxed{3333}',text);self.assertEqual(text,self.s['_format_trace'](s));self.assertNotIn('hidden_rule',text)
 def test_boxed_literal_operator_braces_roundtrip(self):
  from utils.benchmark_training import verifier
  official=verifier()
  for answer in ('}30','35}','{12','12{','012','-01','@22'):
   sample={'task_type':'equation_numeric','trace':{},'answer':answer,'prompt':'public fixture','verification_status':'accepted'}
   record=self.s['make_trace_record'](sample)
   extracted,_=official['extract_final_answer'](record['completion'],'equation_numeric')
   self.assertEqual(extracted,answer)
   self.assertTrue(self.s['audit_trace_record'](record)['accepted'])
 def test_historical_strict_equation_previews(self):
  from pathlib import Path
  p=Path('data/processed/round3/nvidia_synthetic_final_20261115_r1/sft_rendered_preview.jsonl')
  for l in p.read_text().splitlines():
   r=json.loads(l)
   if r['task_type']!='equation_numeric':continue
   s=self.sample(self.n['_parse_numeric_prompt'](r['prompt'])[0],self.n['_parse_numeric_prompt'](r['prompt'])[1])
   self.assertEqual(s['answer'],r['final_answer']);self.assertTrue(self.n['verify_equation_numeric_sample'](s)['trace_ok'])


class AugmentationTrainingMathTests(unittest.TestCase):
 def test_real_cpu_updates_full_masks_and_short_final_window(self):
  import importlib.util
  if importlib.util.find_spec('torch') is None:
   import subprocess,os
   from pathlib import Path
   root=Path(__file__).resolve().parents[1]
   env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'PYTHONPATH':str(root)+os.pathsep+str(root/'tests'),'CUDA_VISIBLE_DEVICES':'','TOKENIZERS_PARALLELISM':'false'}
   result=subprocess.run([str(root/'.venv-low-memory/bin/python'),'-m','unittest','test_equation_augmentation.AugmentationTrainingMathTests','-v'],cwd=root,env=env,capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stdout+result.stderr)
   return
  import torch
  from utils.run_sft_augmentation import api
  torch.manual_seed(7)
  class Tiny(torch.nn.Module):
   def __init__(self):
    super().__init__();self.emb=torch.nn.Embedding(11,6);self.head=torch.nn.Linear(6,11)
   def get_input_embeddings(self):return self.emb
   def forward(self,input_ids,attention_mask=None,use_cache=False):
    return self.head(self.emb(input_ids))
  ns=api();model=Tiny();before=model.head.weight.detach().clone()
  encoded=[{'input_ids':[1,2,3],'labels':[-100,2,3],'prompt_length':1},
           {'input_ids':[1,4,5,6],'labels':[-100,-100,5,6],'prompt_length':2},
           {'input_ids':[1,7,8],'labels':[-100,7,8],'prompt_length':1}]
  ns['encode_chat_example']=lambda tok,record,limit:record
  config={'seed':17,'max_length':4096,'learning_rate':.001,'weight_decay':.01,'adam_epsilon':1e-8,
          'epochs':2,'gradient_accumulation':2,'projection_chunk_size':2,'max_grad_norm':1.}
  result=ns['train_augmentation_branch'](model,None,encoded,config)
  self.assertEqual(result['optimizer_steps'],4);self.assertEqual(result['supervised_tokens'],12)
  self.assertEqual(sum(e['window_supervised_tokens'] for e in result['history']),12)
  self.assertFalse(torch.equal(before,model.head.weight))


if __name__ == "__main__": unittest.main()
