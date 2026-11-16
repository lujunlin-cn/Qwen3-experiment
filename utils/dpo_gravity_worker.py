"""Dedicated SFT-C DPO-B GPU worker. Never runs legacy Notebook GPU entries.

Natural-language mathematics and preference/loss definitions remain in notebooks.
This module provides model I/O, sampled-token evidence and strict stage isolation.
"""
from __future__ import annotations
import argparse,ast,gc,hashlib,json,math,os,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RUN=ROOT/'data/processed/round5/dpo_gravity_sft_c_20261116_r1'
BASE=ROOT/'.autodl/augmentation_base_c1899de'
PARENT=ROOT/'.autodl/equation_sft_augmentation_20261116_r2/C/adapter'
WEIGHTS=ROOT/'.autodl/dpo_gravity_sft_c_20261116_r1'
LIMIT=8_000_000_000

def read(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_rows(p,rs):
 with pathlib.Path(p).open('x') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def append(p,v):
 with pathlib.Path(p).open('a') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n');f.flush()

def nl_api():
 from utils.benchmark_training import verifier
 ns={'gravity_nl_official_verify':verifier()['verify_answer']}
 wanted={'gravity_nl_public_reference','gravity_nl_final_section','_gravity_nl_normalize','_gravity_nl_eval','verify_gravity_natural_language'}
 for c in read(ROOT/'01_Synthetic_Data_Engineering.ipynb')['cells']:
  if c['cell_type']!='code':continue
  nodes=[n for n in ast.parse(''.join(c['source'])).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
  if nodes:exec(compile(ast.Module(body=nodes,type_ignores=[]),'01_Synthetic_Data_Engineering.ipynb:gravity_nl','exec'),ns)
 if not wanted<=ns.keys():raise RuntimeError('Natural language Verifier missing')
 return ns

def pair_api():
 ns={}
 for c in read(ROOT/'04_DPO.ipynb')['cells']:
  if c['cell_type']=='code':
   for n in ast.parse(''.join(c['source'])).body:
    if isinstance(n,ast.FunctionDef) and n.name=='gravity_nl_preference_pairs':exec(compile(ast.Module(body=[n],type_ignores=[]),'04_DPO.ipynb','exec'),ns)
 return ns['gravity_nl_preference_pairs']

def audit_nl_rows(rs):
 ns=nl_api()
 for r in rs:
  audit=ns['verify_gravity_natural_language'](r['prompt'],r['completion'],r['expected_answer'],r['finish_reason'],thinking_expected=r.get('thinking_enabled',False))
  r.update(proof_audit=audit,format_valid=audit['format_valid'],exact_trace_pass=audit['verified'],
    correct=bool(audit['final_correct'] and not r['truncated'] and r['finish_reason']=='stop'),predicted=audit.get('predicted'),
    proof_verifier_version=audit['version'],verifier_version='official_nvidia_metric_v3',output_contract='natural_language_gravity_v1',variant='original')
 return rs

def validate_identity():
 c=read(RUN/'config.json');identity=read(RUN/'identity_audit.json')
 if sha(PARENT/'adapter_model.safetensors')!=c['parent_adapter_sha256']:raise ValueError('SFT C weight mismatch; no fallback')
 for p,check in identity['base_files'].items():
  if sha(BASE/p)!=check['expected']:raise ValueError('Pinned base/tokenizer mismatch: '+p)
 if (read(PARENT/'adapter_config.json')['r'],read(PARENT/'adapter_config.json')['lora_alpha'],read(PARENT/'adapter_config.json')['lora_dropout'])!=(8,16,0):raise ValueError('LoRA contract mismatch')
 reg=RUN/'registration.json'
 if reg.exists():
  sources=read(reg)['sources']
  amendment=RUN/'verification_amendment.json'
  if amendment.exists():
   amended=read(amendment)
   if amended['original_registration_sha256']!=sha(reg) or amended['cpu_tests_exit_code']!=0:raise ValueError('Invalid CPU verifier repair registration')
   if sha(RUN/amended['cpu_tests_log'])!=amended['cpu_tests_log_sha256']:raise ValueError('CPU verifier repair test evidence changed')
   sources=amended['sources']
  for p,h in sources.items():
   if sha(ROOT/p)!=h:raise ValueError('Registered source changed: '+p)
 return c

def mem_report(worker=None):
 import torch
 a=torch.cuda.max_memory_allocated();r=torch.cuda.max_memory_reserved()
 if r>LIMIT:raise RuntimeError('Own-process 8,000,000,000 byte CUDA memory guard exceeded')
 return {'peak_allocated_bytes':a,'peak_reserved_bytes':r,'limit_bytes':LIMIT,'scope':'own CUDA Torch allocator; UMA device-wide free memory not available'}

def load_training_policy(model_path,mode,adapter=None):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 from utils.course_worker import training_mode
 if mode!='bf16_lora' or adapter is None:raise ValueError('Pinned BF16 LoRA adapter required')
 if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU-training substitution')
 torch.set_num_threads(4)
 torch.cuda.set_per_process_memory_fraction(min(.95,LIMIT/torch.cuda.get_device_properties(0).total_memory))
 torch.cuda.reset_peak_memory_stats()
 # CPU loading avoids optional UMA-unaware allocator warmup, without changing dtype or training configuration.
 base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='sdpa')
 base.config.use_cache=False
 policy=PeftModel.from_pretrained(base,str(adapter),is_trainable=True,torch_device='cpu').to('cuda')
 policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});policy.enable_input_require_grads();training_mode(policy)
 tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 mem_report();return policy,tok

def infer(stage):
 import torch
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 c=validate_identity();seed=c['seed']
 torch.cuda.set_per_process_memory_fraction(min(.95,LIMIT/torch.cuda.get_device_properties(0).total_memory))
 engine=LLM(model=str(BASE),tokenizer=str(BASE),dtype='bfloat16',tensor_parallel_size=1,
  enable_lora=True,max_lora_rank=8,max_loras=1,enable_prefix_caching=False,max_model_len=20480,
  max_num_seqs=8,max_num_batched_tokens=512,kv_cache_memory_bytes=4*2**30,gpu_memory_utilization=.10,
  compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4,8],'max_cudagraph_capture_size':8},
  enforce_eager=False,seed=seed,generation_config='vllm',logprobs_mode='raw_logprobs')
 tok=engine.get_tokenizer();memory=engine.collective_rpc(mem_report)
 if stage in ('rollout8','rollout4'):
  panel=rows(RUN/'train_panel.jsonl');n=8 if stage=='rollout8' else 4;offset=0 if n==8 else 8
  targets=[('SFT_C',PARENT,panel,1.,n,offset)]
 else:
  name='tune' if stage=='tune' else 'locked_final';panel=rows(RUN/f'{name}_panel.jsonl')
  lock=read(RUN/'checkpoint_freeze.json') if stage=='final' else read(WEIGHTS/'checkpoint.json')
  if sha(WEIGHTS/'adapter/adapter_model.safetensors')!=lock['weights']['adapter_model.safetensors']:raise ValueError('DPO Checkpoint changed')
  targets=[('SFT_C',PARENT,panel,0.,1,0),('DPO',WEIGHTS/'adapter',panel,0.,1,0)]
 out=RUN/stage;out.mkdir(exist_ok=False)
 for arm,adapter,panel,temperature,n,offset in targets:
  start=time.perf_counter();count=0;fingerprint=sha(adapter/'adapter_model.safetensors')
  request=LoRARequest(stage+'_'+arm,1 if arm=='SFT_C' else 2,str(adapter))
  for start_i in range(0,len(panel),1 if n>1 else 8):
   batch=panel[start_i:start_i+(1 if n>1 else 8)]
   prompts=[tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in batch]
   if any(len(tok.encode(p,add_special_tokens=False))+16384>20480 for p in prompts):raise ValueError('Context overflow; no truncation')
   params=[SamplingParams(n=n,temperature=temperature,top_p=1.,top_k=-1,max_tokens=16384,
      logprobs=1 if n>1 else None,seed=seed+start_i+offset,skip_special_tokens=False) for _ in batch]
   outputs=engine.generate(prompts,params,lora_request=request,use_tqdm=False)
   for original,rendered,output in zip(batch,prompts,outputs):
    for k,o in enumerate(output.outputs):
     ids=list(o.token_ids);lp=[]
     if n>1:
      if o.logprobs is None or len(o.logprobs)!=len(ids):raise ValueError('Missing behavior token probabilities')
      lp=[float(l[t].logprob) for t,l in zip(ids,o.logprobs)]
      if not all(math.isfinite(v) for v in lp):raise ValueError('Nonfinite behavior probability')
     # EOS normally appears as the final generated token even though hidden by some decoders.
     is_stop=o.finish_reason=='stop'
     if is_stop and (not ids or ids[-1] not in {tok.eos_token_id,151643,151645}):raise ValueError('Stopped response missing EOS; refuse wrong length/logprob accounting')
     r={'sample_id':str(original['id']),'prompt_id':str(original['id']),'task_type':'gravity','prompt':original['prompt'],
      'expected_answer':original['answer'],'completion':o.text,'prompt_token_ids':list(output.prompt_token_ids),
      'completion_token_ids':ids,'behavior_logprobs':lp,'finish_reason':o.finish_reason,'stop_reason':o.stop_reason,
      'truncated':o.finish_reason=='length','max_tokens':16384,'group_index':k+offset,
      'policy_version':'SFT_C:'+fingerprint if arm=='SFT_C' else 'DPO:'+fingerprint,'policy_fingerprint':{'adapter_model.safetensors':fingerprint,'base_revision':c['base_revision']},
      'source':'vllm_rollout' if n>1 else 'vllm_evaluation','source_split':'train' if n>1 else 'validation',
      'provenance':original.get('provenance',{}),'fixture':False,'temperature':temperature,'top_p':1.,'top_k':-1,
      'seed':seed+start_i+offset,'variant':'original','arm':arm,'stage':stage,'thinking_enabled':True,
      'completion_tokens':len(ids),'eos_included':is_stop,'public_model_input_sha256':hashlib.sha256(rendered.encode()).hexdigest()}
     append(out/(arm+'_raw.jsonl'),r);count+=1
   engine.collective_rpc(mem_report)
  write(out/(arm+'_result.json'),{'status':'completed','records':count,'generation_seconds':time.perf_counter()-start,
   'policy_sha256':fingerprint,'memory':engine.collective_rpc(mem_report),'initial_memory':memory})
 write(out/'completion_hook.json',{'status':'completed','stage':stage,'time':time.time()})

def train():
 import torch
 import utils.course_worker as cw
 c=validate_identity();torch.manual_seed(c['seed']);cw.load_policy=load_training_policy
 WEIGHTS.mkdir(parents=True,exist_ok=False)
 job={'method':'dpo','model':str(BASE),'adapter':str(PARENT),'sft_adapter':str(PARENT),'run':str(RUN),'output':str(WEIGHTS),
  'rollout':str(RUN/'rollouts_audited.jsonl'),'policy_version':'SFT_C:'+c['parent_adapter_sha256'],'seed':c['seed'],'round':1}
 result=cw.train_dpo(job,c)
 result.update(memory=mem_report(),parent_adapter_sha256=sha(PARENT/'adapter_model.safetensors'))
 # Verify real weight changes and nonzero gradients, not only a successful save.
 from safetensors.torch import load_file
 a=load_file(str(PARENT/'adapter_model.safetensors'));b=load_file(str(WEIGHTS/'adapter/adapter_model.safetensors'))
 result['changed_tensors']=sum(not torch.equal(a[k],b[k]) for k in a)
 if not result['changed_tensors'] or not any(e['gradient_norm_before_clip']>0 for e in result['training_curve']):raise ValueError('No real DPO update')
 write(RUN/'training_result.json',result)

def reload():
 import torch
 import utils.course_worker as cw
 c=validate_identity();cw.load_policy=load_training_policy
 result=cw.reload_check({'model':str(BASE),'method':'dpo','adapter':str(WEIGHTS/'adapter')},c)
 state=torch.load(WEIGHTS/'trainer.pt',weights_only=True,map_location='cpu')
 result['optimizer_steps_total']=state['optimizer_steps_total'];result['optimizer_step_values']=sorted({int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v})
 result['memory']=mem_report();write(RUN/'reload_result.json',result)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['rollout8','rollout4','train','reload','tune','final']);a=p.parse_args()
 try:train() if a.stage=='train' else reload() if a.stage=='reload' else infer(a.stage)
 except BaseException as e:
  write(RUN/f'{a.stage}_failure_{time.time_ns()}.json',{'status':'failed','error':repr(e),'stage':a.stage,'time':time.time()});raise
if __name__=='__main__':main()
