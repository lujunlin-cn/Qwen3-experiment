"""CPU orchestration for the isolated Equation/SFT augmentation experiment.

No hosted API entrypoint or credential handling. Core math remains in Notebooks
01/03. Historical files are read-only. Every stage refuses to overwrite output.
"""
from __future__ import annotations
import argparse,ast,contextlib,copy,hashlib,io,json,os,pathlib,random,socket,subprocess,sys,time
from collections import Counter
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RUN=ROOT/'data/processed/round4/equation_sft_augmentation_20261116_r2'
OLD=ROOT/'data/processed/round3/nvidia_synthetic_final_20261115_r1'
TASKSET=ROOT/'data/processed/tasksets/roman_gravity_equation_v1'
RECOVERED=ROOT/'data/processed/benchmarks/qwen3-0_6b/real_gpu_20261109/recovered_course_r2'
ADAPTER=ROOT/'.autodl/round2_20261114/setup/sft_r1_adapter'
SEED=20261116
MAX_LENGTH=4096

def read(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):
 with pathlib.Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_rows(p,x):
 with pathlib.Path(p).open('x') as f:
  for v in x:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
def definitions():
 # No notebook top-level data-loading or GPU cells execute.
 from utils.run_nvidia_final_synthetic import load_definitions,load_renderer
 return load_definitions(),load_renderer()
def tokenizer():
 from utils.run_nvidia_v2 import local_tokenizer
 tok,meta=local_tokenizer()
 if tok is None:raise ValueError('Actual local tokenizer required')
 return tok,meta

def indexes(n):
 import pandas as pd
 out={}
 for label,path in [('sft_actual',TASKSET/'dataset_manifest.json'),('alternate',ROOT/'data/processed/dataset_manifest.json')]:
  m=read(path);frames=[]
  for s,p in m['files'].items():
   if sha(ROOT/p)!=m['artifacts'][s]['sha256']:raise ValueError('Split hash changed')
   frames.append(pd.read_csv(ROOT/p,dtype=str,keep_default_na=False))
  out[label]=n['build_real_index'](pd.concat(frames,ignore_index=True))
 return out

def training_gate(seed,n,idx,prior,ph):
 lineage=n['audit_train_prior_lineage'](seed,prior,ph)
 leakage={name:n['audit_synthetic_leakage'](seed,index) for name,index in idx.items()}
 reasons=[]
 if not lineage['train_only_provenance']:reasons.append('lineage_failed')
 for name,a in leakage.items():
  if a['status']!='passed':reasons.append('leakage_'+name)
 agreement={'passed':False}
 try:
  agreement=n['audit_equation_agreement'](seed['prompt']) if seed['task_type']=='equation_numeric' else {'passed':True}
  proof=n['solver_first_structured_proof'](seed)
  if not agreement['passed'] or not proof['solver_proof_accepted']:reasons.append('exact_math_failed')
  if seed.get('task_spec_sampling'):
   hidden=seed['task_spec_sampling']['rules'];examples,q=n['_parse_numeric_prompt'](seed['prompt'])
   truth=all(n['_numeric_truth_output'](hidden[e[2]],e)==y for e,y in examples)
   truth=truth and n['_numeric_truth_output'](hidden[q[2]],q)==proof['sample']['answer']
   if not truth:reasons.append('generator_solver_verifier_disagreement')
 except (ValueError,KeyError,TypeError,IndexError) as e:
  proof={'solver_proof_accepted':False,'error':str(e)};reasons.append('solver_unresolved_or_nonunique')
 return {'lineage':lineage,'leakage':leakage,'proof':proof,'agreement':agreement,'rejection_reasons':reasons}

def render_candidate(seed,gate,n,s,prior,ph,tok):
 clean=copy.deepcopy(seed);clean.update(verification_status='accepted',leakage_audit={'status':'passed'})
 clean.update(gate['proof']['sample']);clean['candidate_id']=seed['candidate_id']
 # Legacy Gravity Completion must remain byte-identical. Equation is explicitly
 # re-rendered after exact Solver/reference auditing.
 record,audit=s['render_solver_first_preview'](clean,gate['proof'],n,prior,ph,('gravity','equation_numeric','bit_manipulation'),tok,max_sequence_length=MAX_LENGTH)
 return record,audit

def review():
 n,s=definitions();idx=indexes(n);prior=read(OLD/'train_prior.json');ph=sha(OLD/'train_prior_manifest.json');tok,meta=tokenizer()
 by_id={x['candidate_id']:x for x in rows(OLD/'seed_accepted.jsonl')};records=[];audits=[]
 for old in rows(OLD/'sft_rendered_preview.jsonl'):
  seed=by_id[old['candidate_id']];gate=training_gate(seed,n,idx,prior,ph)
  checks={'prompt_exact':old['prompt']==seed['prompt'],'lineage_pass':not gate['lineage']['rejection_reasons'],'math_pass':gate['proof']['solver_proof_accepted'],
   'leakage_both_manifests':all(v['status']=='passed' for v in gate['leakage'].values()),'old_format_audit':s['audit_trace_record'](old)['accepted'],
   'answer_exact':old['final_answer']==gate['proof'].get('sample',{}).get('answer'),'no_hidden_thinking':'reasoning_content' not in old and 'hidden_rule' not in old['completion']}
  old_count=len(tok(old['completion'],add_special_tokens=False).input_ids);old_encoded=s['encode_chat_example'](tok,old,MAX_LENGTH)
  checks['old_token_count_exact']=old_count==old['actual_token_counts']['completion_tokens']
  checks['old_within_sft_r1_limit']=len(old_encoded['input_ids'])<=MAX_LENGTH
  if not gate['rejection_reasons']:
   new,audit=render_candidate(seed,gate,n,s,prior,ph,tok)
   if old['task_type']=='gravity':
    checks['gravity_completion_unchanged']=new['completion']==old['completion']
    records.append(old) # Preserve original full record, not only completion.
   else:records.append(new)
   checks['new_render_audit']=audit['sft_render_audit_passed']
  audits.append({'candidate_id':seed['candidate_id'],'task_type':seed['task_type'],'checks':checks,'passed':all(checks.values()),
   'gate':gate,'old_completion_sha256':hashlib.sha256(old['completion'].encode()).hexdigest(),'actual_token_counts':{'completion':old_count,'full_chat':len(old_encoded['input_ids'])},
   'equation_rerender_reason':'Legacy preview omitted explicit example calculations and full assignments' if seed['task_type']=='equation_numeric' else None})
 write_rows(RUN/'preview_review.jsonl',audits);write_rows(RUN/'preview_reverified_records.jsonl',records)
 write(RUN/'preview_review_summary.json',{'by_task':{t:{'total':sum(a['task_type']==t for a in audits),'passed':sum(a['task_type']==t and a['passed'] for a in audits)} for t in ('gravity','equation_numeric')},'tokenizer':meta,'max_length':MAX_LENGTH,'merge_into_sft':False})
 if not all(a['passed'] for a in audits):raise ValueError('Preview audit failed; do not proceed')
 return read(RUN/'preview_review_summary.json')

@contextlib.contextmanager
def generation_boundary(events):
 import builtins
 original_builtin,original_io=builtins.open,io.open
 allowed={OLD/'train_prior.json',OLD/'train_prior_manifest.json',TASKSET/'dataset_manifest.json',TASKSET/'splits/train.csv'}
 def guard(original):
  def opened(file,mode='r',*args,**kwargs):
   if isinstance(file,(str,bytes,os.PathLike)) and not any(c in mode for c in 'wax+'):
    p=pathlib.Path(os.fsdecode(file)).resolve()
    if ROOT/'data' in p.parents:
     if p not in allowed:raise PermissionError('Generator read outside frozen Train inputs: '+str(p))
     events.append(str(p.relative_to(ROOT)))
   return original(file,mode,*args,**kwargs)
  return opened
 with patch('builtins.open',guard(original_builtin)),patch('io.open',guard(original_io)):yield

def generate():
 n,s=definitions();events=[];started=time.perf_counter()
 if (RUN/'equation_candidates.jsonl').exists():raise FileExistsError('Candidate pool frozen; no resampling')
 if (RUN/'equation_solver_audit.json').exists():raise RuntimeError('Cannot generate after audit')
 if not read(RUN/'preview_review_summary.json'):raise ValueError('Preview review required')
 with generation_boundary(events):
  prior=read(OLD/'train_prior.json');manifest=read(OLD/'train_prior_manifest.json');ph=sha(OLD/'train_prior_manifest.json')
  dataset=read(TASKSET/'dataset_manifest.json')
  if sha(TASKSET/'splits/train.csv')!=manifest['train_split_sha256'] or sha(TASKSET/'dataset_manifest.json')!=manifest['dataset_manifest_sha256']:raise ValueError('Registered prior source changed')
  if n['_prior_hash']({k:v for k,v in prior.items() if k!='prior_content_sha256'})!=prior['prior_content_sha256']:raise ValueError('Prior content changed')
  rng=n['np'].random.default_rng(SEED)
  pool=[n['generate_train_only_candidate'](prior,rng,'equation_numeric',f'equation-augmentation-{i:04d}',ph,SEED) for i in range(1,501)]
  write_rows(RUN/'equation_candidates.jsonl',pool)
 write(RUN/'candidate_pool_freeze.json',{'candidate_count':500,'candidate_sha256':sha(RUN/'equation_candidates.jsonl'),'rng_seed':SEED,'prior_manifest_sha256':ph,'prior_file_sha256':sha(OLD/'train_prior.json'),
  'generated_before_this_run_evaluation_audit':True,'evaluation_reads':0,'reads':events,'elapsed_seconds':time.perf_counter()-started,'resampling_after_audit':False,'note':'Prior selection fixed before this run. Previous historical audits existed; no evaluation-driven changes to sampling.'})
 return {'candidates':len(pool)}

def audit():
 freeze=read(RUN/'candidate_pool_freeze.json')
 if sha(RUN/'equation_candidates.jsonl')!=freeze['candidate_sha256']:raise ValueError('Frozen pool changed')
 if (RUN/'equation_solver_audit.json').exists():raise FileExistsError('Audit already exists')
 write(RUN/'equation_audit_started.json',{'candidate_freeze_sha256':sha(RUN/'candidate_pool_freeze.json'),'started_unix':time.time(),'evaluation_content_loaded_after_freeze':True})
 n,s=definitions();idx=indexes(n);prior=read(OLD/'train_prior.json');ph=sha(OLD/'train_prior_manifest.json');tok,meta=tokenizer()
 accepted=[];rejected=[];audits=[];seen=set();renders=[];eligible=[];old_seeds={x['candidate_id']:x for x in rows(OLD/'seed_accepted.jsonl')}
 # Re-audited old ten count only after the new exact certificate and renderer.
 inputs=[old_seeds[x['candidate_id']] for x in rows(RUN/'preview_reverified_records.jsonl') if x['task_type']=='equation_numeric']+rows(RUN/'equation_candidates.jsonl')
 for i,seed in enumerate(inputs):
  gate=training_gate(seed,n,idx,prior,ph);reasons=list(gate['rejection_reasons']);key=n['synthetic_prompt_fingerprint'](seed['prompt'])
  if key in seen:reasons.append('within_pool_duplicate')
  seen.add(key)
  if seed.get('generation_error'):reasons.append(seed['generation_error'])
  record=audit_record=None
  if not reasons:
   try:
    record,audit_record=render_candidate(seed,gate,n,s,prior,ph,tok)
    if not audit_record['sft_render_audit_passed']:reasons.append('render_audit_failed')
   except ValueError as e:reasons.append('render_length_or_format_failed:'+str(e))
  item={**seed,'verification_status':'rejected' if reasons else 'accepted','lineage_status':gate['lineage']['lineage_status'],'rejection_reasons':reasons}
  if not reasons:
   item.update(answer=gate['proof']['sample']['answer'],trace=gate['proof']['sample']['trace']);accepted.append(item);renders.append(record)
  else:rejected.append(item)
  audits.append({'candidate_id':seed['candidate_id'],'historical_preview':i<10,'passed':not reasons,'rejection_reasons':reasons,'lineage':gate['lineage'],
    'leakage':gate['leakage'],'agreement':gate['agreement'],'independent_verification':gate['proof'].get('sample_verification'), 'render':audit_record,
    'solver_proof':gate['proof'].get('proof')})
  if (i+1)%50==0:print('audited',i+1,'accepted',len(accepted),flush=True)
 write_rows(RUN/'equation_accepted.jsonl',accepted);write_rows(RUN/'equation_rejected.jsonl',rejected)
 write_rows(RUN/'equation_sft_rendered_preview.jsonl',renders)
 write(RUN/'equation_solver_audit.json',{'status':'passed_100_gate' if len(accepted)>=100 else 'shortfall','total_including_old':len(inputs),'accepted':len(accepted),'rejected':len(rejected),
   'rejection_counts':dict(Counter(r for a in rejected for r in a['rejection_reasons'])),'records':audits,'tokenizer':meta,'strict_intermediate_strings':True,'official_score_tolerance_changed':False})
 write(RUN/'equation_lineage_audit.json',{'status':'passed' if all(a['lineage']['train_only_provenance'] for a in audits) else 'failed','prior_manifest_sha256':ph,'all_source_rows_train':all(a['lineage']['train_only_provenance'] for a in audits),'records':[{'candidate_id':a['candidate_id'],**a['lineage']} for a in audits]})
 write(RUN/'equation_distribution_audit.json',{'all_new_candidates':n['audit_train_sampling_distribution'](prior,rows(RUN/'equation_candidates.jsonl')),'accepted':n['audit_train_sampling_distribution'](prior,accepted),'selection_bias':'Exact identifiability/length gates change the accepted distribution; no holdout resampling','holdout_tuning':False})
 if len(accepted)<100:raise ValueError('100 final accepted gate failed')
 return {'accepted':len(accepted),'rejected':len(rejected)}

def roman():
 n,s=definitions();idx=indexes(n);originals=rows(ROOT/'data/processed/round3/nvidia_hosted_20261115_ultra_r4_full/private_seeds.jsonl');out=[]
 for seed in originals:
  if seed['task_type']!='numeral_system':continue
  for j,intro in enumerate(n['NVIDIA_INTRODUCTIONS']):
   text=intro+'\n'+seed['prompt'];clean=copy.deepcopy(seed);clean['prompt']=text;n['_solve_candidate'](clean);v=n['verify_numeral_sample'](clean)
   checks={'original_prompt_utf8_suffix_unchanged':text.encode().split(b'\n',1)[1]==seed['prompt'].encode(),'answer_unchanged':clean['answer']==seed['answer'],'solver_verifier':v['answer_ok'] and v['trace_ok']}
   exposure={name:n['audit_synthetic_leakage'](clean,index) for name,index in idx.items()}
   out.append({'variant_id':seed['seed_id']+f':intro{j}','task_type':'numeral_system','prompt':text,'original_prompt':seed['prompt'],'answer':seed['answer'],'original_prompt_sha256':hashlib.sha256(seed['prompt'].encode()).hexdigest(),
     'checks':checks,'leakage_audits':exposure,'verification_status':'preview','new_mapping_knowledge':False,'sft_training_eligible':False,'merge_into_sft':False})
 write_rows(RUN/'roman_prompt_variant_preview.jsonl',out)
 return {'previews':len(out),'new_mapping_knowledge':False}

def datasets():
 import pandas as pd
 n,s=definitions();tok,meta=tokenizer();eq=rows(RUN/'equation_sft_rendered_preview.jsonl')[:100]
 if len(eq)!=100:raise ValueError('100 exact Equation training records required')
 gravity=[x for x in rows(OLD/'sft_rendered_preview.jsonl') if x['task_type']=='gravity']
 initial=read(RUN/'initial_state.json')
 if [''+l for l in initial['gravity_source_lines']]!=[l for l in (OLD/'sft_rendered_preview.jsonl').read_text().splitlines() if json.loads(l)['task_type']=='gravity']:raise ValueError('Historical Gravity changed')
 synthetic=eq+gravity
 # Actual SFT R1 Train only, excluding every original teacher ID.
 used={x['source_id'] for x in rows(RECOVERED/'panels/sft.jsonl')}
 frame=pd.read_csv(TASKSET/'splits/train.csv',dtype=str,keep_default_na=False)
 controls={t:[] for t in ('equation_numeric','gravity')};rejections=[]
 for r in sorted(frame.to_dict('records'),key=lambda x:x['id']):
  if r['task_type'] not in controls or r['id'] in used:continue
  sample={'task_type':r['task_type'],'prompt':r['prompt'],'difficulty':'deduce_real_train','source_id':r['id'],'source':'verified_real_train','verification_status':'accepted'}
  try:
   proof=n['solver_first_structured_proof'](sample);sample.update(proof['sample'])
   if not proof['solver_proof_accepted'] or sample['answer']!=r['answer']:raise ValueError('Exact public Solver answer differs from official Train answer or proof failed')
   if r['task_type']=='equation_numeric':sample,reference=s['strict_equation_render_reference'](sample,n)
   record=s['make_trace_record'](sample);record.update(source='verified_real_train',source_id=r['id'],source_row=r['source_row'],source_split='train',canonical_solver_trace=sample['trace'])
   if not s['audit_trace_record'](record)['accepted']:raise ValueError('Trace audit')
   encoded=s['encode_chat_example'](tok,record,MAX_LENGTH);record['actual_token_counts']={'completion_tokens':len(tok(record['completion'],add_special_tokens=False).input_ids),'sft_supervised_tokens':sum(x!=-100 for x in encoded['labels'][1:]),'sft_full_chat_tokens':len(encoded['input_ids'])}
   controls[r['task_type']].append(record)
  except (ValueError,KeyError,IndexError,TypeError) as e:rejections.append({'source_id':r['id'],'task_type':r['task_type'],'reason':str(e)})
 # Preserve the exact real-control IDs/order from the discarded format trial.
 # Only the literal-brace renderer is repaired; no new outcome-based selection.
 parent=ROOT/'data/processed/round4/equation_sft_augmentation_20261116_r1'
 original_control=rows(parent/'sft_real_control_110.jsonl')
 by_control={r['source_id']:r for pool in controls.values() for r in pool}
 real=[by_control[r['source_id']] for r in original_control]
 if len(real)!=110:raise ValueError('Original real-control IDs no longer pass exact math/format gates')
 total=lambda rs:sum(x['actual_token_counts']['sft_supervised_tokens'] for x in rs)
 budget={'synthetic_supervised_tokens':total(synthetic),'real_supervised_tokens':total(real),'relative_difference':abs(total(real)-total(synthetic))/total(synthetic)}
 # Never claim matched if outside 1%; require strict gate before actual training.
 write_rows(RUN/'sft_synthetic_110.jsonl',synthetic);write_rows(RUN/'sft_real_control_110.jsonl',real);write_rows(RUN/'real_control_rejected.jsonl',rejections)
 write(RUN/'sft_data_manifest.json',{'status':'ready' if budget['relative_difference']<=.01 else 'token_budget_unmatched','train_records_per_arm':110,'synthetic_by_task':dict(Counter(x['task_type'] for x in synthetic)),'real_by_task':dict(Counter(x['task_type'] for x in real)),
 'token_budget':budget,'all_records_used_once':True,'completion_truncation':False,'max_length':MAX_LENGTH,'excluded_original_teacher_ids':sorted(used),'real_ids_overlap_r1':sorted(used&{r['source_id'] for r in real}),
 'sha256':{p:sha(RUN/p) for p in ('sft_synthetic_110.jsonl','sft_real_control_110.jsonl')},'baseline_adapter_sha256':sha(ADAPTER/'adapter_model.safetensors'),'dataset_manifest_sha256':sha(TASKSET/'dataset_manifest.json'),'tokenizer':meta,'merge_into_default_sft':False,'training_executed':False})
 return read(RUN/'sft_data_manifest.json')

def isolation():
 initial=read(RUN/'initial_state.json');checks={}
 checks['protected_algorithms']=all(sha(ROOT/p)==h for p,h in initial['protected_sha256'].items())
 checks['historical_artifacts']=all(sha(ROOT/p)==h for p,h in initial['historical_tree'].items())
 # Also ensure the default SFT builder and Gravity renderer branch stayed intact.
 old=json.loads(subprocess.check_output(['git','show','a3f29dc:03_SFT.ipynb'],cwd=ROOT,text=True));new=read(ROOT/'03_SFT.ipynb')
 def func(nb,name):
  for c in nb['cells']:
   if c['cell_type']=='code':
    for x in ast.parse(''.join(c['source'])).body:
     if isinstance(x,ast.FunctionDef) and x.name==name:return x
 checks['default_sft_builder_unchanged']=ast.dump(func(old,'build_full_trace_records'))==ast.dump(func(new,'build_full_trace_records'))
 def gravity_branch(nb):
  tree=func(nb,'_format_trace')
  return next(ast.dump(x) for x in ast.walk(tree) if isinstance(x,ast.If) and ast.unparse(x.test)=="task == 'gravity'")
 # Compare the body only: chained elif includes the intentionally changed Equation.
 def gb(nb):
  return next([ast.dump(x) for x in x.body] for x in ast.walk(func(nb,'_format_trace')) if isinstance(x,ast.If) and ast.unparse(x.test)=="task == 'gravity'")
 checks['gravity_renderer_unchanged']=gb(old)==gb(new)
 result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'api_calls':0,'gpu_training':False,'merge_into_default_sft':False}
 write(RUN/'isolation_audit.json',result)
 if not all(checks.values()):raise ValueError(result)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['review','generate','audit','roman','datasets','isolation']);args=p.parse_args()
 # Fail closed if accidentally reaching any network path during CPU production.
 with patch.object(socket.socket,'connect',side_effect=RuntimeError('CPU data stages forbid network')):
  result=globals()[args.stage]()
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
