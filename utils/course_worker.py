"""Isolated GPU worker. The supervisor alone starts this under a memory/time guard.

Losses, advantages and preference/reward definitions are loaded from teaching
notebooks; this module handles model I/O, inference and repeated epochs/rounds.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.course_protocol import (TASKSET, check_protocol, digest, metrics, prompt_text,
                                   read, read_rows, write, write_rows)

NOTEBOOKS = {'sft':'03_SFT.ipynb', 'dpo':'04_DPO.ipynb', 'ppo':'05_PPO.ipynb', 'grpo':'06_GRPO.ipynb'}

def notebook_api(method, chunk_size=64):
    """Execute config and selected definitions, never notebook smoke/GPU entrypoints."""
    module = types.ModuleType('course_' + method)
    sys.modules[module.__name__] = module
    ns = module.__dict__
    exec('from typing import *\nfrom torch.utils.checkpoint import checkpoint\nfrom torch.utils.data import Dataset, DataLoader', ns)
    notebook = read(ROOT / NOTEBOOKS[method])
    config_source=''.join(notebook['cells'][3]['source']).replace(
        '"data" / "processed" / "splits"',
        '"data" / "processed" / "tasksets" / "roman_gravity_equation_v1" / "splits"')
    exec(compile(config_source, NOTEBOOKS[method]+':config', 'exec'), ns)
    ns.update({'OFFICIAL_VERIFIER_VERSION':'official_nvidia_metric_v3', 'GAMMA':1.0, 'GAE_LAMBDA':1.0,
               'TOKEN_CHUNK_SIZE':chunk_size})
    wanted = {
        'sft':{'_format_trace','make_trace_record','encode_chat_example','collate_causal_examples','course_sft_loss_sum'},
        'dpo':{'_completion_length','course_preference_pairs','course_dpo_backward'},
        'ppo':{'PPORollout','validate_rollout_alignment','outcome_reward','_recompute','_effective_attention_mask',
               '_selected_logprob','_token_entropy','token_logprobs','entropy_from_logits','projected_policy_statistics',
               'masked_mean','compute_gae','normalize_masked','ppo_clip_loss','value_loss','approx_kl','explained_variance',
               'CausalLMValueHead','align_response_mask','align_ppo_action_axis','ppo_update',
               'build_ppo_rollout_from_records','course_ppo_outcome'},
        'grpo':{'group_advantage','grpo_clip_loss','policy_logprob_change','course_efficiency_reward'},
    }[method]
    loaded = set()
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code': continue
        tree = ast.parse(''.join(cell['source']))
        definitions = [node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in wanted]
        if definitions:
            exec(compile(ast.Module(body=definitions,type_ignores=[]), NOTEBOOKS[method]+f':cell{i}', 'exec'), ns)
            loaded.update(node.name for node in definitions)
    assert loaded == wanted, wanted-loaded
    return ns

def artifact_fingerprint(adapter):
    if not adapter: return {'kind':'base'}
    adapter = Path(adapter)
    return {p.name:digest(p) for p in sorted(adapter.glob('adapter_*')) if p.is_file()}

def training_mode(policy):
    """Keep decoder checkpointing active while retaining deterministic dropout."""
    import torch
    policy.train()
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    return policy

def load_policy(model_path, mode, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    kwargs={'local_files_only':True,'torch_dtype':torch.bfloat16,'device_map':{'':0},'attn_implementation':'sdpa'}
    if mode == 'nf4_qlora':
        kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    elif mode != 'bf16_lora': raise ValueError(mode)
    base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    base.config.use_cache=False
    if mode == 'nf4_qlora':
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={'use_reentrant':False})
        base.get_input_embeddings().to(dtype=torch.bfloat16)
        base.get_output_embeddings().to(dtype=torch.bfloat16)
    if adapter:
        policy=PeftModel.from_pretrained(base,str(adapter),is_trainable=True)
    else:
        policy=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,bias='none',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    policy.enable_input_require_grads()
    # Transformers skips decoder checkpointing in eval mode even with gradients.
    training_mode(policy)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    return policy,tokenizer

def tensor_example(row, device):
    import torch
    prompt=row['prompt_token_ids']; response=row['completion_token_ids']
    if not prompt or not response: raise ValueError('Empty rollout tokens')
    ids=torch.tensor([prompt+response],device=device)
    mask=torch.tensor([[0]*len(prompt)+[1]*len(response)],device=device)
    return ids,mask


def validate_training_rows(rows,job,c):
    import math
    allowed=c[job['method']]['tasks']
    if not rows: raise ValueError('Empty training rollout')
    for r in rows:
        if (r['task_type'] not in allowed or r.get('source_split')!='train' or r.get('fixture')
            or r.get('source')!='vllm_rollout' or r['policy_version']!=job['policy_version']):
            raise ValueError('Wrong Train policy/task/source provenance')
        response=r['completion_token_ids'];prompt=r['prompt_token_ids'];old=r['behavior_logprobs']
        if not response or not prompt or len(response)!=len(old) or not all(math.isfinite(x) for x in old):
            raise ValueError('Invalid sampled token/logprob alignment')
        if len(response)+len(prompt)>c['max_model_len']: raise ValueError('Context overflow in training')

def verify_rows(rows):
    from utils.benchmark_training import verifier
    api=verifier()
    for r in rows:
        predicted,method=api['extract_final_answer'](r['completion'],r['task_type'])
        correct,verification=api['verify_answer'](r['task_type'],predicted,r['expected_answer'])
        r.update(correct=bool(correct and not r['truncated']), predicted=predicted,
                 extraction_method=method,verification_method=verification,
                 verifier_version='official_nvidia_metric_v3')
    return rows

def generate(job, c):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows=read_rows(job['panel']); prior=read(Path(job['run'])/'public_priors.json')
    tokenizer=AutoTokenizer.from_pretrained(job['model'],local_files_only=True)
    prompts=[prompt_text(r['prompt'],r['task_type'],job['variant'],prior) for r in rows]
    rendered=[tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=True) for p in prompts]
    caps=[job.get('cap') or c['eval_caps'][r['task_type']] for r in rows]
    for text,cap in zip(rendered,caps):
        if len(tokenizer.encode(text,add_special_tokens=False))+cap > c['max_model_len']:
            raise ValueError('Context overflow: refuse silent prompt/completion truncation')
    engine=LLM(model=job['model'],tokenizer=job['model'],dtype='bfloat16',tensor_parallel_size=1,
        enable_lora=True,max_lora_rank=8,max_model_len=c['max_model_len'],max_num_seqs=c['max_num_seqs'],
        max_num_batched_tokens=c['max_num_batched_tokens'],kv_cache_memory_bytes=int(c['kv_cache_gib']*2**30),
        gpu_memory_utilization=.70,enforce_eager=True,enable_chunked_prefill=True,
        generation_config='vllm',logprobs_mode='raw_logprobs',seed=job['seed'])
    rollout=job['kind']=='rollout'
    params=[SamplingParams(n=job.get('candidates',1),temperature=1.0 if rollout else .6,
        top_p=1.0 if rollout else .95,top_k=-1 if rollout else 20,max_tokens=cap,
        logprobs=1 if rollout else None,seed=job['seed'],skip_special_tokens=False) for cap in caps]
    adapter=LoRARequest('course_policy',1,job['adapter']) if job.get('adapter') else None
    start=time.perf_counter()
    outputs=engine.generate(rendered,params,lora_request=adapter)
    result=[]
    for original, rendered_prompt, output,cap in zip(rows,prompts,outputs,caps):
        for k,out in enumerate(output.outputs):
            ids=list(out.token_ids); lps=[]
            if rollout:
                if out.logprobs is None or len(out.logprobs)!=len(ids): raise ValueError('Missing sampled token logprobs')
                for token,values in zip(ids,out.logprobs):
                    if token not in values: raise ValueError('Sampled token absent from logprobs')
                    lps.append(float(values[token].logprob))
            result.append({'sample_id':str(original['id']),'prompt_id':str(original['id']),
                'source_split':original.get('split','train'), 'task_type':original['task_type'],
                'prompt':rendered_prompt,'expected_answer':original['answer'],'completion':out.text,
                'prompt_token_ids':list(output.prompt_token_ids),'completion_token_ids':ids,
                'behavior_logprobs':lps,'finish_reason':out.finish_reason,'stop_reason':out.stop_reason,
                'truncated':out.finish_reason=='length','max_tokens':cap,'group_index':k,
                'policy_version':job['policy_version'],'source':'vllm_rollout' if rollout else 'vllm_evaluation',
                'fixture':False,'temperature':1.0 if rollout else .6,'top_p':1.0 if rollout else .95,
                'top_k':-1 if rollout else 20,'variant':job['variant']})
    result=verify_rows(result)
    write_rows(Path(job['output'])/'rows.jsonl',result)
    seconds=time.perf_counter()-start
    return {'status':'completed','metrics':metrics(result),'records':len(result),
        'generation_seconds':seconds,'output_tokens_per_second':sum(len(r['completion_token_ids']) for r in result)/seconds,
        'policy_fingerprint':artifact_fingerprint(job.get('adapter'))}

def optimizer_for(model, job, settings):
    import torch
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'])
    if job.get('trainer_state'):
        state=torch.load(job['trainer_state'],map_location='cpu',weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
    return optimizer

def alignment(policy, rows, chunk, output):
    """Audit exact sampled behavior denominators; never replace them with recomputed scores."""
    import torch
    from utils.low_memory_logprobs import causal_policy_logprobs
    device=policy.get_input_embeddings().weight.device
    stats=[]; signed=absolute=count=0; maximum=0
    with torch.no_grad():
        for row in rows:
            ids,mask=tensor_example(row,device)
            lp=causal_policy_logprobs(policy,ids,None,chunk)[mask[:,1:].bool()].float().cpu()
            old=torch.tensor(row['behavior_logprobs'])
            if lp.shape != old.shape or not torch.isfinite(old).all(): raise ValueError('Invalid behavior denominator')
            diff=lp-old; maximum=max(maximum,float(diff.abs().max()))
            signed+=float(diff.sum());absolute+=float(diff.abs().sum());count+=diff.numel()
            stats.append({'id':row['sample_id'],'group_index':row['group_index'],
                          'mean_abs_diff':float(diff.abs().mean()),'max_abs_diff':float(diff.abs().max())})
    report={'tokens':count,'mean_abs_diff':absolute/count,'max_abs_diff':maximum,
            'mean_signed_diff':signed/count,'rows':stats,'behavior_denominator':'original raw sampled logprobs'}
    # Aggregate drift also detects BF16/NF4 distribution changes and stale adapters.
    report['status']='passed' if report['mean_abs_diff'] <= .15 and maximum <= 3.0 else 'failed_alignment'
    write(output/'alignment.json',report)
    if report['status']!='passed': raise ValueError('Policy/behavior alignment failed; see alignment.json')
    return report

def sft_epoch(job,c):
    import torch
    from utils.low_memory_logprobs import causal_policy_logprobs
    ns=notebook_api('sft'); settings=c['sft']
    policy,tokenizer=load_policy(job['model'],settings['mode'],job.get('adapter'))
    records=[ns['make_trace_record'](r) for r in read_rows(Path(job['run'])/'panels/sft.jsonl')]
    encoded=[ns['encode_chat_example'](tokenizer,r,settings['max_length']) for r in records]
    optimizer=optimizer_for(policy,job,settings)
    generator=torch.Generator().manual_seed(job['seed'])
    order=torch.randperm(len(encoded),generator=generator).tolist()
    losses=[]; total_tokens=steps=0; device=policy.get_input_embeddings().weight.device
    start=time.perf_counter()
    for k in range(0,len(order),settings['gradient_accumulation']):
        window=[encoded[i] for i in order[k:k+settings['gradient_accumulation']]]
        count=sum(sum(x!=-100 for x in r['labels'][1:]) for r in window)
        optimizer.zero_grad(set_to_none=True); loss_value=0
        for record in window:
            batch={k:v.to(device) for k,v in ns['collate_causal_examples']([record],tokenizer.pad_token_id).items()}
            loss,n=ns['course_sft_loss_sum'](policy,batch,c['projection_chunk_size'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite SFT loss')
            (loss/count).backward(); loss_value+=float(loss.detach());total_tokens+=n
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);optimizer.step();steps+=1
        losses.append(loss_value/count)
    result={'status':'completed','optimizer_steps':steps,'training_seconds':time.perf_counter()-start,
            'mean_window_loss':sum(losses)/len(losses),'assistant_training_tokens':total_tokens,'records':len(records)}
    return save_training(policy,tokenizer,optimizer,job,result)

def save_training(policy,tokenizer,optimizer,job,result,value_head=None):
    import torch
    result['decoder_memory_settings'] = {
        'training_mode': policy.get_base_model().model.training,
        'gradient_checkpointing_enabled': bool(policy.is_gradient_checkpointing),
        'use_cache': policy.config.use_cache,
    }
    output=Path(job['output']); adapter=output/'adapter'
    policy.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
    torch.save({'optimizer':optimizer.state_dict()},output/'trainer.pt')
    if value_head is not None: torch.save(value_head.state_dict(),output/'value_head.pt')
    manifest={'status':'completed','adapter_path':str(adapter),'method':job['method'],
        'round':job['round'],'parent_policy_version':job['policy_version'],
        'policy_version':f"{job['method']}:round{job['round']}:{artifact_fingerprint(adapter)['adapter_model.safetensors'][:12]}",
        'parent_fingerprint':artifact_fingerprint(job.get('adapter')),'weights':artifact_fingerprint(adapter),
        'sft_reference':job.get('sft_adapter'),'base_model_path':job['model'],
        'trainer_state':str(output/'trainer.pt'),'value_head':str(output/'value_head.pt') if value_head is not None else None}
    write(output/'checkpoint.json',manifest)
    return dict(result,checkpoint=manifest)

def train_dpo(job,c):
    import torch
    from utils.low_memory_logprobs import causal_sequence_logprob
    ns=notebook_api('dpo'); settings=c['dpo']; output=Path(job['output'])
    rows=read_rows(job['rollout']); pairs=ns['course_preference_pairs'](rows)
    validate_training_rows(rows,job,c)
    write_rows(output/'preferences.jsonl',[{'sample_id':p['sample_id'],'task_type':p['task_type'],
        'preference_reason':p['preference_reason'],'chosen_group_index':p['chosen']['group_index'],
        'rejected_group_index':p['rejected']['group_index']} for p in pairs])
    if not pairs: return {'status':'no_preference_signal','optimizer_steps':0}
    policy,tokenizer=load_policy(job['model'],settings['mode'],job['adapter'])
    # Two small LoRA adapters share one frozen base. Reference remains original SFT in every round.
    policy.load_adapter(job['sft_adapter'],adapter_name='reference',is_trainable=False)
    device=policy.get_input_embeddings().weight.device
    def score(row):
        ids,mask=tensor_example(row,device)
        return causal_sequence_logprob(policy,ids,mask,c['projection_chunk_size'])
    policy.set_adapter('reference'); policy.eval()
    cached=[]
    with torch.no_grad():
        for p in pairs: cached.append([float(score(p[s])) for s in ('chosen','rejected')])
    policy.set_adapter('default');training_mode(policy)
    for name,p in policy.named_parameters():
        if '.reference.' in name: p.requires_grad_(False)
    write(output/'reference_scores.json',{'reference_fingerprint':artifact_fingerprint(job['sft_adapter']),
        'reference_mode':settings['mode'],'rows':cached})
    optimizer=optimizer_for(policy,job,settings); steps=0;losses=[];start=time.perf_counter()
    for k in range(0,len(pairs),settings['gradient_accumulation']):
        window=pairs[k:k+settings['gradient_accumulation']];optimizer.zero_grad(set_to_none=True)
        for i,pair in enumerate(window,k):
            with torch.no_grad(): values=[score(pair[s]) for s in ('chosen','rejected')]
            refs=[values[0].new_tensor([v]) for v in cached[i]]
            loss=ns['course_dpo_backward'](policy,values,refs,lambda side:score(pair[side]),scale=1/len(window))
            if not __import__('math').isfinite(loss): raise ValueError('Nonfinite DPO')
            losses.append(loss)
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);optimizer.step();steps+=1
    # Save only the trainable default adapter; keep the immutable SFT reference separate.
    result={'status':'completed','pairs':len(pairs),'optimizer_steps':steps,
            'losses':losses,'training_seconds':time.perf_counter()-start,'reference':'same frozen SFT for all rounds',
            'logprob_backend':'sequential DPO backward; checkpointed full-vocabulary projection'}
    policy.delete_adapter('reference')
    return save_training(policy,tokenizer,optimizer,job,result)

def train_on_policy(job,c):
    import torch
    from utils.low_memory_logprobs import causal_policy_logprobs
    method=job['method']; settings=c[method];ns=notebook_api(method,c['projection_chunk_size']);output=Path(job['output'])
    rows=read_rows(job['rollout'])
    validate_training_rows(rows,job,c)
    policy,tokenizer=load_policy(job['model'],settings['mode'],job['adapter']);device=policy.get_input_embeddings().weight.device
    ns['TOKEN_CHUNK_SIZE']=c['projection_chunk_size']
    audit=alignment(policy,rows,c['projection_chunk_size'],output)
    if method=='ppo':
        model=ns['CausalLMValueHead'](policy)
        if job.get('value_head'):
            model.value_head.load_state_dict(torch.load(job['value_head'],map_location=device,weights_only=True))
        optimizer=optimizer_for(model,job,settings)
        # Snapshot value estimates of the unchanged current policy before any update.
        rollouts=[]; all_adv=[];counts=[]
        for row in rows:
            r=ns['build_ppo_rollout_from_records']([row],model,tokenizer.pad_token_id)
            r=ns['course_ppo_outcome'](r,row)
            rewards,values,dones,mask=ns['align_ppo_action_axis'](r.rewards,r.old_values,r.dones,r.response_mask)
            adv,_=ns['compute_gae'](rewards,values,dones,mask)
            all_adv.append(adv[mask.bool()].cpu());counts.append(int(mask.sum()))
            # Only small token vectors, not activation graphs; keep on CPU until its microbatch.
            for key in ('input_ids','attention_mask','response_mask','old_logprobs','old_values','rewards','dones'):
                setattr(r,key,getattr(r,key).cpu())
            rollouts.append(r)
        adv=torch.cat(all_adv).to(device);stats=(adv.mean(),adv.var(unbiased=False));del adv
        logs=[];steps=0;start=time.perf_counter()
        for k in range(0,len(rollouts),settings['gradient_accumulation']):
            window=rollouts[k:k+settings['gradient_accumulation']];denom=sum(counts[k:k+len(window)])
            optimizer.zero_grad(set_to_none=True)
            for i,r in enumerate(window,k):
                for key in ('input_ids','attention_mask','response_mask','old_logprobs','old_values','rewards','dones'):
                    setattr(r,key,getattr(r,key).to(device))
                logs.append(ns['ppo_update'](model,optimizer,r,step_optimizer=False,
                    loss_scale=counts[i]/denom,advantage_stats=stats,entropy_coef=settings['entropy_coef'],kl_coef=0.0))
                for key in ('input_ids','attention_mask','response_mask','old_logprobs','old_values','rewards','dones'):
                    setattr(r,key,getattr(r,key).cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();steps+=1
        result={'status':'completed','optimizer_steps':steps,'logs':logs,'training_seconds':time.perf_counter()-start,
                'outcomes':metrics(rows),'alignment':audit,'gamma':1,'gae_lambda':1,'reference_kl_coef':0,
                'finite_horizon_length_limit_terminal':True,'includes_incorrect_and_truncated':True}
        return save_training(policy,tokenizer,optimizer,job,result,model.value_head)
    grouped={}
    for row in rows: grouped.setdefault(row['sample_id'],[]).append(row)
    ordered=[];advantages=[];variances=[]
    for group in grouped.values():
        group.sort(key=lambda r:r['group_index'])
        if len(group)!=settings['candidates'] or len({r['group_index'] for r in group})!=settings['candidates']:
            raise ValueError('Incomplete GRPO group')
        rewards=torch.tensor([[ns['course_efficiency_reward'](r) for r in group]],dtype=torch.float32)
        adv,_,std=ns['group_advantage'](rewards)
        variances.append(float(std[0]));ordered+=group;advantages+=adv.flatten().tolist()
    if not any(v>1e-6 for v in variances):
        return {'status':'no_group_signal','optimizer_steps':0,'group_stds':variances,'alignment':audit}
    optimizer=optimizer_for(policy,job,settings);logs=[];steps=0;start=time.perf_counter()
    for k in range(0,len(ordered),settings['gradient_accumulation']):
        window=ordered[k:k+settings['gradient_accumulation']]
        denom=sum(len(r['completion_token_ids']) for r in window);optimizer.zero_grad(set_to_none=True)
        for i,row in enumerate(window,k):
            ids,mask=tensor_example(row,device)
            old=torch.zeros_like(ids[:,:-1],dtype=torch.float32)
            old[mask[:,1:].bool()]=torch.tensor(row['behavior_logprobs'],device=device)
            new=causal_policy_logprobs(policy,ids,None,c['projection_chunk_size'])
            result=ns['grpo_clip_loss'](old,new,torch.tensor([advantages[i]],device=device),mask[:,1:])
            if not torch.isfinite(result['loss']): raise ValueError('Nonfinite GRPO')
            (result['loss']*len(row['completion_token_ids'])/denom).backward()
            logs.append(float(result['loss'].detach()));del new,result
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);optimizer.step();steps+=1
    result={'status':'completed','optimizer_steps':steps,'losses':logs,'group_stds':variances,
            'training_seconds':time.perf_counter()-start,'alignment':audit,'reference_kl_coef':0,
            'reward_definition':'correct complete: 1-.15*length/cap; otherwise 0','outcomes':metrics(rows)}
    return save_training(policy,tokenizer,optimizer,job,result)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True,type=Path)
    args=parser.parse_args();job=read(args.job);output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT);os.environ['BASELINE_PHASE']='off'
    for key in list(os.environ):
        if key.startswith('RUN_'): os.environ[key]='0'
    os.environ['QWEN3_MODEL_PATH']=job['model']
    c=check_protocol(Path(job['run']))['config']
    for key, sha in job.get('input_hashes',{}).items():
        if digest(job[key]) != sha: raise ValueError(f'Job input changed: {key}')
    for key in ('adapter','sft_adapter'):
        if job.get(key) and artifact_fingerprint(job[key]) != job[key+'_fingerprint']:
            raise ValueError(f'Adapter changed: {key}')
    import torch
    import numpy as np
    torch.manual_seed(job['seed']); np.random.seed(job['seed'])
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this worker')
    # Allocator cap is an extra guard, not proof of device peak (vLLM uses other allocators).
    torch.cuda.set_per_process_memory_fraction(min(.95,c['device_limit_bytes']*c['allocator_fraction_of_limit']/torch.cuda.get_device_properties(0).total_memory))
    config=read(Path(job['model'])/'config.json')
    if (config.get('model_type'),config.get('hidden_size'),config.get('num_hidden_layers'))!=('qwen3',1024,28):
        raise ValueError('Not the pinned Qwen3-0.6B architecture')
    started=time.perf_counter()
    try:
        if job['kind'] in ('evaluation','rollout'): result=generate(job,c)
        elif job['method']=='sft': result=sft_epoch(job,c)
        elif job['method']=='dpo': result=train_dpo(job,c)
        else: result=train_on_policy(job,c)
        torch.cuda.synchronize()
        result.update(worker_wall_seconds=time.perf_counter()-started,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            versions={'torch':torch.__version__,'transformers':__import__('transformers').__version__},
            base_config_sha256=digest(Path(job['model'])/'config.json'))
        write(output/'result.json',result)
    except Exception as error:
        write(output/'result.json',{'status':'failed','error':repr(error),'worker_wall_seconds':time.perf_counter()-started})
        raise

if __name__=='__main__': main()
