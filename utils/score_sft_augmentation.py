"""Offline official scoring + conservative independent full-trace certification."""
from __future__ import annotations
from utils.run_equation_augmentation import *
from utils.benchmark_training import verifier
import re

def reference_audit():
 n,s=definitions();out=[]
 for r in rows(RUN/'augmentation_validation_panel.jsonl'):
  sample={'task_type':r['task_type'],'prompt':r['prompt'],'difficulty':'deduce_validation'}
  try:
   proof=n['solver_first_structured_proof'](sample);sample.update(proof['sample'])
   if r['task_type']=='equation_numeric':sample,_=s['strict_equation_render_reference'](sample,n)
   out.append({'id':r['id'],'task_type':r['task_type'],'reference_verified':proof['solver_proof_accepted'],
    'solver_answer':sample['answer'],'official_answer_exact':sample['answer']==r['answer'],
    'official_answer_scored_correct':n['numeric_or_text_match'](sample['answer'],r['answer']),
    'canonical_completion':s['_format_trace'](sample)})
  except (ValueError,KeyError,IndexError) as e:
   resolution=n['solve_equation_numeric_prompt'](r['prompt']) if r['task_type']=='equation_numeric' else {}
   out.append({'id':r['id'],'task_type':r['task_type'],'reference_verified':False,'error':str(e),'resolution':resolution})
 write(RUN/'validation_reference_audit.json',{'rows':out,'all_questions_retained':True,'primary_metric':'official final accuracy on every registered question','scope_note':'Non-unique/unsupported public Numeric24 references cannot certify full proof; never change official targets.'})
 return out

def score():
 ref_path=RUN/'validation_reference_audit.json'
 refs=reference_audit() if not ref_path.exists() else read(ref_path)['rows'];by_id={r['id']:r for r in refs};official=verifier();n,s=definitions()
 for a in 'ABC':
  out=RUN/f'sft_{a}_validation';infer=read(out/'inference_result.json')
  if infer['status']!='completed' or infer['rows']!=150:raise ValueError('Incomplete inference is not an evaluation result')
  scored=[]
  for r in rows(out/'raw_rows.jsonl'):
   predicted,method=official['extract_final_answer'](r['completion'],r['task_type']);correct,why=official['verify_answer'](r['task_type'],predicted,r['answer'])
   ref=by_id[r['id']];exact=False
   if ref['reference_verified'] and not r['truncated']:
    text=r['completion']
    for token in ('<think>','</think>','<|im_end|>','<|endoftext|>'):text=text.replace(token,'')
    exact=' '.join(text.split())==' '.join(ref['canonical_completion'].split())
   subtype=None
   if r['task_type']=='equation_numeric':
    examples,query=n['_parse_numeric_prompt'](r['prompt']);subtype='deduce' if query[2] in {e[2] for e,_ in examples} else 'guess'
   scored.append({**r,'predicted':predicted,'answer_extraction_success':predicted is not None,'extraction_method':method,'verification_method':why,
    'correct':bool(correct and not r['truncated']),'strict_final_string_correct':bool(not r['truncated'] and predicted==r['answer']),
    'reference_verified':ref['reference_verified'],'full_trace_verified':exact,'full_trace_status':'verified_canonical_full_trace' if exact else 'unverified_noncanonical_or_unsolved',
    'deduce_or_guess':subtype})
  write_rows(out/'rows.jsonl',scored);write(out/'result.json',{**infer,'status':'completed','metrics_scope':'official answers + conservative canonical Full Trace certificate','scored_rows':len(scored)})
 print('Scored A/B/C, 150 fixed Validation questions each; no model/API call')
if __name__=='__main__':score()
