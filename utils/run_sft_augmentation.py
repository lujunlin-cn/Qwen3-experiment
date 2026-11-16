"""Registered A/B/C experiment supervisor: model I/O, artifacts, stats only.

Training/decoding implementations are explicit Notebook 03 functions. Never
changes default SFT inputs or the DPO/PPO/GRPO workers. No NVIDIA hosted API.
"""
from __future__ import annotations
import argparse,ast,gc,json,math,os,pathlib,random,statistics,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from utils.run_equation_augmentation import (RUN,OLD,TASKSET,RECOVERED,ADAPTER,read,rows,sha,write,write_rows,definitions)
CONFIG=ROOT/'configs/equation_sft_augmentation_20261116.json'
BASE=ROOT/'.autodl/augmentation_base_c1899de'
SOURCES=('01_Synthetic_Data_Engineering.ipynb','03_SFT.ipynb','utils/run_equation_augmentation.py','utils/run_sft_augmentation.py','configs/equation_sft_augmentation_20261116.json')

def api():
 import torch
 from utils.run_nvidia_final_synthetic import load_renderer
 ns=load_renderer();ns.update(torch=torch)
 wanted={'train_augmentation_branch','generate_augmentation_evaluation','course_sft_loss_sum'}
 for cell in read(ROOT/'03_SFT.ipynb')['cells']:
  if cell['cell_type']!='code':continue
  for node in ast.parse(''.join(cell['source'])).body:
   if isinstance(node,ast.FunctionDef) and node.name in wanted:
    exec(compile(ast.Module(body=[node],type_ignores=[]),'03_SFT.ipynb','exec'),ns)
 return ns

def register():
 import pandas as pd
 c=read(CONFIG);data=read(RUN/'sft_data_manifest.json')
 if data['status']!='ready' or data['token_budget']['relative_difference']>.01:raise ValueError('Equal-budget data gate failed')
 import re
 cpu=(RUN/'related_cpu_final.log').read_text()
 counts=re.findall(r'Ran (\d+) tests?',cpu)
 if not counts or int(counts[-1])<85 or not cpu.rstrip().endswith('OK'):raise ValueError('CPU regression gate failed')
 if sha(ADAPTER/'adapter_model.safetensors')!=c['parent_adapter_sha256']:raise ValueError('Real parent missing')
 expected=read(RECOVERED/'base_model_fingerprint.json')
 checks={p:sha(BASE/p)==h for p,h in expected.items()}
 if not all(checks.values()):raise ValueError('Base/tokenizer revision mismatch')
 m=read(TASKSET/'dataset_manifest.json');frame=pd.read_csv(ROOT/m['files']['validation'],dtype=str,keep_default_na=False)
 panel=[]
 for i,t in enumerate(('numeral_system','gravity','equation_numeric')):
  pool=sorted((r for r in frame.to_dict('records') if r['task_type']==t),key=lambda r:r['id']);random.Random(c['seed']+i).shuffle(pool)
  panel+=pool[:c['validation_per_task']]
 write_rows(RUN/'augmentation_validation_panel.jsonl',panel)
 write(RUN/'sft_experiment_registration.json',{'status':'registered_before_training','config':c,'config_sha256':sha(CONFIG),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  'source_sha256':{p:sha(ROOT/p) for p in SOURCES},'data_sha256':data['sha256'],'validation_panel_sha256':sha(RUN/'augmentation_validation_panel.jsonl'),
  'validation_source_sha256':sha(ROOT/m['files']['validation']),'base_files_match_historical_r1':checks,'parent_weight_sha256':sha(ADAPTER/'adapter_model.safetensors'),
  'baseline_A':'same R1 adapter, no optimizer update; BF16 base inference as used in the original vLLM evaluation',
  'precision_note':'Original R1 trained NF4 QLoRA. Both new arms use identical BF16 LoRA on this host; fixed original adapter is preserved.',
  'evaluation_panel_reason':'50/task resource-bounded preregistration; no sample chosen using model outcomes','holdout_used':False,
  'evaluation_cap_note':'Caps lower than historical 8192/16384; all A/B/C use identical caps. Results only describe this registered decode budget.',
  'full_trace_verification_scope':'Only exact deterministic Full Trace templates can be certified; other wording is unverified, not mathematically false.',
  'registered_at_unix':time.time()})
 return {'registered':True,'panel':len(panel)}

def verify_registration():
 reg=read(RUN/'sft_experiment_registration.json')
 for p,h in reg['source_sha256'].items():
  if sha(ROOT/p)!=h:
   amendment=read(RUN/'loader_registration.json')
   if p!='utils/run_sft_augmentation.py' or amendment['runner_sha256']!=sha(ROOT/p) or amendment['original_registration_sha256']!=sha(RUN/'sft_experiment_registration.json'):
    raise ValueError('Registered source changed: '+p)
 for p,h in reg['data_sha256'].items():
  if sha(RUN/p)!=h:raise ValueError('Frozen training data changed: '+p)
 if sha(RUN/'augmentation_validation_panel.jsonl')!=reg['validation_panel_sha256']:raise ValueError('Validation panel changed')
 return reg

def resource_inventory():
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 info={l.split(':')[0]:int(l.split()[1])*1024 for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','MemTotal:'))}
 return {'existing_gpu_processes':text,'host_memory':info,'gpu_memory_scope':'shared integrated GB10; nvidia-smi whole-device memory is unavailable; own Torch peaks reported'}

def load_model(adapter,train,c):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 info=resource_inventory()
 if info['host_memory']['MemAvailable']<c['minimum_available_host_bytes']:raise RuntimeError('Insufficient shared-host free memory; leave existing processes intact')
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU model substitution')
 torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
 total=torch.cuda.get_device_properties(0).total_memory
 torch.cuda.set_per_process_memory_fraction(min(.90,c['device_process_reserved_limit_bytes']/total),0)
 torch.cuda.reset_peak_memory_stats()
 # Load on CPU first: bypass only Transformers' optional CUDA allocator
 # warmup (which queries unreliable physical free memory on this shared UMA
 # host). Actual CUDA allocations still enforce the 8 GB process limit.
 base=AutoModelForCausalLM.from_pretrained(str(BASE),local_files_only=True,dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='sdpa')
 base.config.use_cache=False
 model=PeftModel.from_pretrained(base,str(adapter),is_trainable=train,torch_device='cpu').to('cuda')
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 model.enable_input_require_grads()
 tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model.train(train)
 return model,tok,info

def train(arm):
 if arm not in ('B','C'):raise ValueError('A never trains')
 reg=verify_registration();c=reg['config'];out=RUN/f'sft_{arm}_train';out.mkdir(exist_ok=False)
 started=time.time()
 try:
  model,tok,inventory=load_model(ADAPTER,True,c)
  data=rows(RUN/('sft_synthetic_110.jsonl' if arm=='B' else 'sft_real_control_110.jsonl'))
  def callback(e):
   with (out/'training_curve.jsonl').open('a') as f:f.write(json.dumps(e)+'\n');f.flush()
   print(arm,e['optimizer_step'],round(e['token_mean_loss'],4),round(e['elapsed_seconds'],1),flush=True)
   if e['torch_peak_reserved_bytes']>=c['device_process_reserved_limit_bytes']:raise RuntimeError('Own-process memory guard exceeded')
  result=api()['train_augmentation_branch'](model,tok,data,c,callback)
  import torch
  state=result.pop('optimizer_state');weights=ROOT/'.autodl'/c['run_id']/arm;weights.mkdir(parents=True,exist_ok=False)
  model.save_pretrained(weights/'adapter');tok.save_pretrained(weights/'adapter');torch.save(state,weights/'optimizer.pt')
  result.update(arm=arm,parent_adapter_sha256=c['parent_adapter_sha256'],adapter_path=str(weights/'adapter'),adapter_sha256=sha(weights/'adapter/adapter_model.safetensors'),optimizer_state_sha256=sha(weights/'optimizer.pt'),
   source_registration_sha256=sha(RUN/'sft_experiment_registration.json'),own_peak_allocated_bytes=torch.cuda.max_memory_allocated(),own_peak_reserved_bytes=torch.cuda.max_memory_reserved(),resource_inventory=inventory,
   completed_at_unix=time.time(),wall_seconds=time.time()-started)
  # Check persisted tensors actually differ from the original R1 tensors.
  from safetensors.torch import load_file
  before=load_file(str(ADAPTER/'adapter_model.safetensors'));after=load_file(str(weights/'adapter/adapter_model.safetensors'))
  result['changed_tensor_count']=sum(not torch.equal(before[k],after[k]) for k in before)
  if not result['changed_tensor_count']:raise RuntimeError('No real parameter updates')
  write(out/'result.json',result)
  del model;gc.collect();torch.cuda.empty_cache()
  return result
 except BaseException as e:
  write(out/'failure.json',{'status':'failed_or_blocked','error':type(e).__name__+': '+str(e),'wall_seconds':time.time()-started,'resource_inventory':resource_inventory()});raise

def evaluate(arm):
 if arm not in ('A','B','C'):raise ValueError(arm)
 reg=verify_registration();c=reg['config'];out=RUN/f'sft_{arm}_validation';out.mkdir(exist_ok=False)
 adapter=ADAPTER if arm=='A' else pathlib.Path(read(RUN/f'sft_{arm}_train/result.json')['adapter_path'])
 model,tok,inventory=load_model(adapter,False,c);n,s=definitions();official=__import__('utils.benchmark_training',fromlist=['verifier']).verifier()
 panel=rows(RUN/'augmentation_validation_panel.jsonl');result=[];started=time.time()
 for start in range(0,len(panel),c['evaluation_batch_size']):
  # Never combine task-specific caps in a batch.
  chunk=panel[start:start+c['evaluation_batch_size']]
  if len({r['task_type'] for r in chunk})>1:chunks=[[r] for r in chunk]
  else:chunks=[chunk]
  for batch in chunks:
   cap=c['eval_caps'][batch[0]['task_type']];tic=time.perf_counter()
   outputs=api()['generate_augmentation_evaluation'](model,tok,[r['prompt'] for r in batch],cap)
   for r,o in zip(batch,outputs):
    predicted,method=official['extract_final_answer'](o['completion'],r['task_type']);correct,why=official['verify_answer'](r['task_type'],predicted,r['answer'])
    # Conservative Full Trace certificate. No model self-evaluation is used.
    exact=False;trace_status='unverified_noncanonical_or_unsolved'
    sample={'task_type':r['task_type'],'prompt':r['prompt'],'difficulty':'deduce_validation'}
    try:
     proof=n['solver_first_structured_proof'](sample);sample.update(proof['sample'])
     if r['task_type']=='equation_numeric':sample,_=s['strict_equation_render_reference'](sample,n)
     canonical=s['_format_trace'](sample)
     text=o['completion']
     for token in ('<think>','</think>','<|im_end|>','<|endoftext|>'):text=text.replace(token,'')
     exact=bool(proof['solver_proof_accepted'] and not o['truncated'] and ' '.join(text.split())==' '.join(canonical.split()))
     if exact:trace_status='verified_canonical_full_trace'
    except (ValueError,KeyError,IndexError):pass
    item={**r,**o,'arm':arm,'predicted':predicted,'answer_extraction_success':predicted is not None,'extraction_method':method,'verification_method':why,
     'correct':bool(correct and not o['truncated']),'full_trace_verified':exact,'full_trace_status':trace_status,'max_tokens':cap,'batch_seconds':time.perf_counter()-tic,
     'deduce_or_guess':('deduce' if n['_parse_numeric_prompt'](r['prompt'])[1][2] in {e[2] for e,_ in n['_parse_numeric_prompt'](r['prompt'])[0]} else 'guess') if r['task_type']=='equation_numeric' else None}
    result.append(item)
    with (out/'rows.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
   print(arm,'evaluation',len(result),'/',len(panel),'seconds',round(time.time()-started,1),flush=True)
 write(out/'result.json',{'status':'completed' if len(result)==len(panel) else 'incomplete','rows':len(result),'evaluation_seconds':time.time()-started,'resource_inventory':inventory,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
   'own_peak_allocated_bytes':__import__('torch').cuda.max_memory_allocated(),'own_peak_reserved_bytes':__import__('torch').cuda.max_memory_reserved()})
 return {'arm':arm,'rows':len(result)}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['register','train','evaluate']);p.add_argument('--arm');a=p.parse_args()
 result=register() if a.stage=='register' else globals()[a.stage](a.arm)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
