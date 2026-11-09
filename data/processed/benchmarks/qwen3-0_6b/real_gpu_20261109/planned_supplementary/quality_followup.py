"""Preregister a focused SFT/DPO follow-up without changing the active run."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from utils.course_protocol import prepare,read,write,digest,check_protocol,read_rows
from utils.course_run import Experiment

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('action',choices=('prepare','run'))
args=parser.parse_args()
source=ROOT/'data/processed/course_runs/course8h_20261109_r2_cuda_graph'
run=ROOT/'data/processed/course_runs/quality_followup_20261109_r1'
if args.action=='prepare':
 if run.exists():raise FileExistsError('Use a fresh follow-up ID')
 p=check_protocol(source);state=read(source/'state.json')
 assert not any(k.startswith('final_') for k in state['jobs']), 'Register before viewing final model results'
 c=json.loads(json.dumps(p['config']))
 c.update(total_seconds=3600,budgets_seconds={'baseline':0,'prompt':0,'sft':900,'dpo':1800,'ppo':0,'grpo':0,'final':900})
 c['sft'].update(mode='bf16_lora',learning_rate=1e-4,max_rounds=3)
 c['dpo'].update(learning_rate=5e-5,gradient_accumulation=1,prompts_per_task=2,max_rounds=3,rollout_caps=[16384])
 write(run/'config.json',c);prepare(run/'config.json',run)
 for name in ('tune','final','sft'):
  assert digest(run/f'panels/{name}.jsonl')==digest(source/f'panels/{name}.jsonl')
 write(run/'focused_manifest.json',{'status':'prepared_cpu_only','source_protocol_sha256':digest(source/'protocol.json'),
  'script_sha256':digest(Path(__file__)),'prepared_before_source_final_evaluation':True,
  'scope':'Focused SFT and DPO only; original four-method study remains unchanged. Same paired tune/final panels and shared already-measured base controls.',
  'changes':{'sft':'BF16 base,lr1e-4,up to3 epochs','dpo':'lr5e-5,microbatch1,accumulation1,2 prompts/task,4 candidates,cap16384,up to3 rounds'},
  'dpo_gate':'Tune macro accuracy >= selected original SFT, Equation accuracy >= selected original SFT, mean output <=4096 for both Gravity and Equation; otherwise skip DPO.',
  'run_only_after_original_finished':True,'maximum_trial_seconds':3600,'selection_on_final_panel':False})
 print('prepared_cpu_only',run);raise SystemExit
manifest=read(run/'focused_manifest.json')
assert digest(Path(__file__))==manifest['script_sha256']
s=read(source/'state.json');assert s.get('finished_at_unix'), 'Finish original study first'
if (run/'execution_started.json').exists():raise FileExistsError('Do not overwrite a previous attempt')
write(run/'execution_started.json',{'time_unix':time.time(),'source_finished_at_unix':s['finished_at_unix']})
e=Experiment(SimpleNamespace(run_dir=run,config=run/'config.json',model=Path('/root/autodl-tmp/huggingface/models/Qwen3-0.6B'),inference_python=ROOT/'.venv-vllm-cu128/bin/python',training_python=ROOT/'.venv/bin/python',gpu='0'))
sft=e.train_method('sft');dpo=None;gate=False
if sft:
 old=s['methods']['sft']['best_metrics'];new=e.state['methods']['sft']['best_metrics']
 gate=(sum(m['accuracy'] for m in new.values())>=sum(m['accuracy'] for m in old.values())
       and new['equation_numeric']['accuracy']>=old['equation_numeric']['accuracy']
       and all(new[t]['mean_output_tokens']<=4096 for t in ('gravity','equation_numeric')))
 if gate:dpo=e.train_method('dpo',sft)
 else:e.state['dpo_gate']='not_met_do_not_spend_rollout_budget';e.persist()
e.stage('final');evaluations={}
for variant in e.c['evaluation_variants']:
 for method,cp in [('sft',sft),('dpo',dpo)]:
  if cp:
   r=e.evaluation(f'final_{method}_{variant}',cp['adapter_path'],variant,panel='final',version=cp['policy_version'])
   if r:evaluations[f'{method}_{variant}']=r['metrics']
comparisons=[]
for variant in e.c['evaluation_variants']:
 base_job=s['jobs'].get(f'final_base_{variant}')
 if not base_job or base_job['guard']['status']!='completed':continue
 before=read(Path(base_job['output'])/'result.json')['metrics']
 for method,parent in [('sft','base'),('dpo','sft')]:
  after=evaluations.get(f'{method}_{variant}')
  a=before if parent=='base' else evaluations.get(f'sft_{variant}')
  if after is None or a is None:continue
  original_rows=read_rows(Path(base_job['output'])/'rows.jsonl')
  current_rows=read_rows(Path(e.state['jobs'][f'final_{method}_{variant}']['output'])/'rows.jsonl')
  key=lambda r:(r['sample_id'],r['task_type'],r['max_tokens'],r['variant'])
  assert {key(r) for r in original_rows}=={key(r) for r in current_rows}
  for t in after:
   comparisons.append({'method':method,'parent':parent,'task':t,'variant':variant,'delta':{k:after[t][k]-a[t][k] for k in ('accuracy','truncation_rate','mean_output_tokens')}})
e.state['finished_at_unix']=time.time();e.persist()
expected=2*(bool(sft)+bool(dpo))
passed=all(j['guard']['status']=='completed' for j in e.state['jobs'].values())
write(run/'focused_result.json',{'status':'focused_trial_completed' if passed and expected==len(evaluations) and sft else 'focused_trial_incomplete',
 'scope':'SFT/DPO follow-up, not a replacement for the original full four-method experiment',
 'dpo_gate_met':gate,'dpo_ran':bool(dpo),'evaluations':evaluations,'comparisons':comparisons,
 'methods':e.state['methods'],'source_run_id':source.name,'shared_base_controls':True,'checkpoint_selection_on_final':False,
 'wall_seconds':e.state['finished_at_unix']-e.state['started_at_unix'],
 'all_executed_jobs_device_memory_sampled_pass':passed,'continuous_memory_bound_verified':False})
print(json.dumps(read(run/'focused_result.json')),flush=True)
