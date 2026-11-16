"""Offline local vLLM inference worker for the pre-registered A/B/C panel.

No training, no HTTP provider, no API credential lookup. Scoring runs separately
on the host with Notebook 01's independent mathematics and official metric.
"""
import argparse,hashlib,json,pathlib,time,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
RUN=ROOT/'data/processed/round4/equation_sft_augmentation_20261116_r1'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def memory_report(worker):
 import torch
 return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
def main():
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 registration=read(RUN/'sft_experiment_registration.json');amendment=read(RUN/'evaluation_backend_registration.json');c=registration['config']
 if sha(pathlib.Path(__file__))!=amendment['worker_sha256']:raise ValueError('Worker changed after registration')
 panel=rows(RUN/'augmentation_validation_panel.jsonl')
 if sha(RUN/'augmentation_validation_panel.jsonl')!=registration['validation_panel_sha256']:raise ValueError('Panel changed')
 model=ROOT/'.autodl/augmentation_base_c1899de'
 engine=LLM(model=str(model),tokenizer=str(model),dtype='bfloat16',tensor_parallel_size=1,enable_lora=True,max_lora_rank=8,max_loras=1,
  max_model_len=8192,max_num_seqs=4,max_num_batched_tokens=512,kv_cache_memory_bytes=2*2**30,gpu_memory_utilization=.10,
  enforce_eager=False,compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4],'max_cudagraph_capture_size':4},
  seed=c['seed'],generation_config='vllm')
 tokenizer=engine.get_tokenizer()
 try:idle_memory=engine.collective_rpc(memory_report)
 except Exception as e:idle_memory={'unavailable':repr(e)}
 for a in 'ABC':
  out=RUN/f'sft_{a}_validation';out.mkdir(exist_ok=False);started=time.time();count=0
  adapter=(ROOT/'.autodl/round2_20261114/setup/sft_r1_adapter') if a=='A' else ROOT/f'.autodl/{c["run_id"]}/{a}/adapter'
  expected=c['parent_adapter_sha256'] if a=='A' else read(RUN/f'sft_{a}_train/result.json')['adapter_sha256']
  if sha(adapter/'adapter_model.safetensors')!=expected:raise ValueError('Adapter changed')
  lora=LoRARequest('augmentation_'+a,ord(a)-64,str(adapter))
  for start in range(0,len(panel),4):
   batch=panel[start:start+4];prompts=[tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in batch]
   params=[SamplingParams(temperature=0.,top_p=1.,top_k=-1,max_tokens=c['eval_caps'][r['task_type']],seed=c['seed'],skip_special_tokens=False) for r in batch]
   tic=time.perf_counter();outputs=engine.generate(prompts,params,lora_request=lora,use_tqdm=False)
   for r,p,o in zip(batch,prompts,outputs):
    completion=o.outputs[0]
    item={**r,'arm':a,'completion':completion.text,'completion_token_ids':list(completion.token_ids),'prompt_token_ids':list(o.prompt_token_ids),
     'truncated':completion.finish_reason=='length','finish_reason':completion.finish_reason,'stop_reason':completion.stop_reason,
     'max_tokens':c['eval_caps'][r['task_type']],'batch_seconds':time.perf_counter()-tic,'backend':'vllm-0.27.1','adapter_sha256':expected}
    with (out/'raw_rows.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
    count+=1
   print(a,'evaluation',count,'/',len(panel),'seconds',round(time.time()-started,1),flush=True)
  try:peak=engine.collective_rpc(memory_report)
  except Exception as e:peak={'unavailable':repr(e)}
  with (out/'inference_result.json').open('x') as f:json.dump({'status':'completed' if count==len(panel) else 'incomplete','arm':a,'rows':count,'evaluation_seconds':time.time()-started,
   'adapter_sha256':expected,'memory':peak,'initial_engine_memory':idle_memory,'backend_registration_sha256':sha(RUN/'evaluation_backend_registration.json')},f,indent=2)
if __name__=='__main__':main()
