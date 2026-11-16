"""Isolated Gravity C2 final-proof repair, inference, and conditional DPO chain.

Core Solver/Renderer/serialization/SFT math remain in Notebooks 01/03.
DPO uses course_worker.train_dpo without modifying its objective or Reference.
"""
from __future__ import annotations
import argparse, ast, csv, gc, hashlib, json, math, os, pathlib, re, statistics, subprocess, sys, time, traceback
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from utils.dpo_gravity_worker import BASE, PARENT, LIMIT, read, rows, sha, write, write_rows, append, nl_api, pair_api, audit_nl_rows, mem_report
from utils.run_dpo_sft_c_gravity_diagnosis import actual_teachers, selected_functions, identity, ols_certified

RUN = ROOT/'data/processed/round6/gravity_sft_c2_dpo_20261116_r1'
OLD = ROOT/'data/processed/round5/dpo_gravity_sft_c_20261116_r1'
DIAG = ROOT/'data/processed/round5/dpo_sft_c_gravity_diagnosis_20261116_r1'
SFT = ROOT/'data/processed/round4/equation_sft_augmentation_20261116_r2'
TASKSET = ROOT/'data/processed/tasksets/roman_gravity_equation_v1'
WEIGHTS = ROOT/'.autodl/gravity_sft_c2_dpo_20261116_r1'
C2 = WEIGHTS/'C2/adapter'
DPO = WEIGHTS/'DPO/adapter'
PYTHON = ROOT/'.venv-low-memory/bin/python'
CONFIG = ROOT/'configs/gravity_sft_c2_20261116.json'
SOURCES = ['01_Synthetic_Data_Engineering.ipynb','03_SFT.ipynb','utils/run_gravity_sft_c2.py',
    'utils/report_gravity_sft_c2.py','tests/test_gravity_sft_c2.py','configs/gravity_sft_c2_20261116.json',
    'utils/course_worker.py','utils/low_memory_logprobs.py','utils/dpo_gravity_worker.py','04_DPO.ipynb']


def apis():
    import numpy as np
    n = selected_functions(read(ROOT/'01_Synthetic_Data_Engineering.ipynb'),
        {'_gravity_prompt_parts','solve_gravity_prompt','_gravity_verifier_fit','gravity_c2_solver_reference'}, {'np':np,'re':re})
    s = selected_functions(read(ROOT/'03_SFT.ipynb'),
        {'encode_chat_example','audit_trace_record','course_sft_loss_sum','train_augmentation_branch',
         'render_gravity_c2_final_proof','make_gravity_c2_record','encode_gravity_c2_final_example','train_gravity_c2_branch'}, {'max_length':4096})
    return n,s


def function_hashes(book, names):
    result={}
    for cell in book['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                result[node.name]=hashlib.sha256(ast.dump(node).encode()).hexdigest()
    return result


def verify_sources():
    c=read(CONFIG);reg=read(RUN/'registration.json')
    if sha(PARENT/'adapter_model.safetensors')!=c['parent_adapter_sha256']:raise ValueError('Original SFT C changed')
    for p,h in reg['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Registered source changed: '+p)
    for p,h in reg['input_hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen input changed: '+p)
    return c


def prepare():
    from transformers import AutoTokenizer
    if (RUN/'registration.json').exists():raise FileExistsError('Existing C2 Run must not be overwritten')
    initial=read(RUN/'initial_state.json');identity_result=identity();c=read(CONFIG)
    for p,h in initial['protected_hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('Protected history changed before repair: '+p)
    nl_names={name for name in nl_api() if name.startswith('gravity_nl_') or name.startswith('_gravity_nl_') or name=='verify_gravity_natural_language'}
    original_hashes=function_hashes(read(RUN/'source_before_repair/01_Synthetic_Data_Engineering.ipynb'),nl_names)
    if original_hashes!=function_hashes(read(ROOT/'01_Synthetic_Data_Engineering.ipynb'),nl_names):raise ValueError('NL Verifier must remain unchanged')
    for name in ['encode_chat_example','_format_trace','make_trace_record','build_full_trace_records','choose_sft_records','course_sft_loss_sum']:
        if function_hashes(read(RUN/'source_before_repair/03_SFT.ipynb'),{name})!=function_hashes(read(ROOT/'03_SFT.ipynb'),{name}):
            raise ValueError('Default SFT entry or loss changed: '+name)
    n,s=apis();tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);nl=nl_api();records=[];audits=[]
    with (TASKSET/'splits/train.csv').open() as f:train={r['id']:r for r in csv.DictReader(f)}
    held={}
    for split in ['validation','holdout_test']:
        with (TASKSET/f'splits/{split}.csv').open() as f:held[split]=list(csv.DictReader(f))
    normalize=lambda text:' '.join(text.casefold().split())
    for group,old,source,legacy in actual_teachers()[0]:
        source_id=old['source_id'];real=train[source_id]
        if old['prompt']!=real['prompt'] or str(source['source_row'])!=str(real['source_row']):raise ValueError('Teacher is not actual Train')
        sample=n['gravity_c2_solver_reference'](old['prompt'])
        if sample['answer']!=old['final_answer']:raise ValueError('Do not relabel existing teacher')
        record=s['make_gravity_c2_record'](sample,source_id,source['source_row'],group)
        encoded=s['encode_gravity_c2_final_example'](tok,record,c['sft']['max_length'])
        p=encoded['prompt_length'];response=tok.decode(encoded['input_ids'][p:],skip_special_tokens=False)
        final=nl['gravity_nl_final_section'](response)[0]
        proof=nl['verify_gravity_natural_language'](record['prompt'],response,record['final_answer'],thinking_expected=True)
        if not proof['verified'] or record['completion'] not in final:raise ValueError('Teacher final proof rejected: '+str(proof['reasons']))
        if not s['audit_trace_record'](record)['accepted']:raise ValueError('SFT format audit failed')
        labels=encoded['labels'];ids=encoded['input_ids']
        if any(v!=-100 for v in labels[:p]) or labels[p:]!=ids[p:]:raise ValueError('C2 loss mask omits proof tokens')
        leaks={split:[r['id'] for r in data if normalize(r['prompt'])==normalize(record['prompt'])] for split,data in held.items()}
        if any(leaks.values()):raise ValueError('C2 Train teacher overlaps actual held-out prompt')
        record['actual_token_counts']={'full_chat_tokens':len(ids),'supervised_tokens':sum(v!=-100 for v in labels[1:]),
            'visible_completion_tokens':len(tok(record['completion'],add_special_tokens=False).input_ids)}
        record['parent_teacher_completion_sha256']=hashlib.sha256(old['completion'].encode()).hexdigest()
        records.append(record)
        audits.append({'source_id':source_id,'source_row':source['source_row'],'group':group,'source':'solver_generated_rendered',
            'prompt_unchanged':True,'all_public_observations':sample['trace']['public_observations'],
            'canonical_solver_trace':sample['trace'],'proof_audit':proof,'serialized_response':response,
            'final_contains_entire_proof':record['completion'] in final,'all_response_targets_supervised':True,
            'prompt_targets_masked':True,'completion_truncated':False,'leakage_audit':leaks,
            'actual_token_counts':record['actual_token_counts']})
    if len(records)!=74 or len({r['source_id'] for r in records})!=74:raise ValueError('Exactly 74 distinct actual teachers required')
    write_rows(RUN/'gravity_c2_teachers_74.jsonl',records);write_rows(RUN/'teacher_repair_audit.jsonl',audits)
    panels={'diagnosis':rows(OLD/'train_panel.jsonl'),'tune':rows(OLD/'tune_panel.jsonl'),'locked_final':rows(OLD/'locked_final_panel.jsonl')}
    for key,rs in panels.items():
        if len(rs)!={'diagnosis':26,'tune':48,'locked_final':100}[key]:raise ValueError('Frozen panel count mismatch')
        write_rows(RUN/f'{key}_panel.jsonl',rs)
    baseline=rows(DIAG/'greedy/SFT_C_raw.jsonl')
    if len(baseline)!=26 or any(r['policy_fingerprint']['adapter_model.safetensors']!=c['parent_adapter_sha256'] for r in baseline):raise ValueError('Cached SFT C diagnostic identity mismatch')
    for a,b in zip(baseline,panels['diagnosis']):
        if a['sample_id']!=b['id'] or a['prompt']!=b['prompt']:raise ValueError('Cached SFT C prompts differ')
    write_rows(RUN/'cached_SFT_C_diagnosis_raw.jsonl',baseline)
    input_files=[RUN/'gravity_c2_teachers_74.jsonl',RUN/'cached_SFT_C_diagnosis_raw.jsonl']+[RUN/f'{x}_panel.jsonl' for x in panels]
    write(RUN/'registration.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sources':{p:sha(ROOT/p) for p in SOURCES},'config':c,'identity':identity_result,
        'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in input_files},'nl_verifier_ast_hashes_unchanged':original_hashes,
        'default_sft_entry_and_loss_unchanged':True,'teacher_count_by_origin':dict(Counter(r['teacher_source_group'] for r in records)),
        'supervised_tokens_per_epoch':sum(r['actual_token_counts']['supervised_tokens'] for r in records),
        'teacher_manifest_hashes':{'r1_actual_panel':sha(ROOT/'data/processed/benchmarks/qwen3-0_6b/real_gpu_20261109/recovered_course_r2/panels/sft.jsonl'),
            'c_actual_input':sha(SFT/'sft_real_control_110.jsonl'),'c_actual_inclusion':sha(SFT/'sft_actual_inclusion_manifest.jsonl')},
        'cached_C_diagnosis_raw_sha256':sha(DIAG/'greedy/SFT_C_raw.jsonl'),
        'validation_note':'Tune48 and Locked Final100 are frozen Validation subsets; not untouched official Holdout.',
        'created_unix':time.time(),'hyperparameter_selection_after_tune':False})
    write(RUN/'teacher_repair_summary.json',{'status':'passed','teachers':74,'final_proof_passed':74,'all_math_tokens_supervised':74,
        'max_full_chat_tokens':max(r['actual_token_counts']['full_chat_tokens'] for r in records),
        'supervised_tokens_per_epoch':sum(r['actual_token_counts']['supervised_tokens'] for r in records),
        'source':'solver_generated_rendered','old_teacher_answers_changed':False,
        'format_preflight':'Observation labels avoid ambiguity with Query t; numerical sums have descriptive labels. The existing NL verifier was not modified.'})


def probe(model):
    import torch
    from utils.low_memory_logprobs import causal_policy_logprobs
    model.eval();ids=torch.tensor([[2,3,4,5]],device=model.get_input_embeddings().weight.device)
    with torch.no_grad():return causal_policy_logprobs(model,ids,None,64).float().cpu()


def train_sft():
    import torch
    from utils.run_sft_augmentation import load_model
    from safetensors.torch import load_file
    c=verify_sources();cfg=c['sft'];out=RUN/'sft_train';out.mkdir(exist_ok=False)
    started=time.time();model,tok,inventory=load_model(PARENT,True,cfg)
    def on_step(event):
        append(out/'training_curve.jsonl',event)
        if event['torch_peak_reserved_bytes']>LIMIT:raise RuntimeError('SFT process memory guard exceeded')
    result=apis()[1]['train_gravity_c2_branch'](model,tok,rows(RUN/'gravity_c2_teachers_74.jsonl'),cfg,on_step)
    if result['optimizer_steps']!=20 or result['supervised_tokens']!=2*read(RUN/'registration.json')['supervised_tokens_per_epoch']:
        raise ValueError('C2 did not use all 74 teachers for both epochs')
    if not all(math.isfinite(x['gradient_norm']) and math.isfinite(x['token_mean_loss']) for x in result['history']):
        raise ValueError('Nonfinite C2 gradients or losses')
    saved_probe=probe(model);state=result.pop('optimizer_state');C2.parent.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(C2);tok.save_pretrained(C2);torch.save(state,C2.parent/'optimizer.pt')
    before=load_file(str(PARENT/'adapter_model.safetensors'));after=load_file(str(C2/'adapter_model.safetensors'))
    changed=sum(not torch.equal(before[k],after[k]) for k in before)
    if not changed or not any(x['gradient_norm']>0 for x in result['history']):raise ValueError('C2 has no real updates')
    result.update(adapter_sha256=sha(C2/'adapter_model.safetensors'),parent_adapter_sha256=sha(PARENT/'adapter_model.safetensors'),
        adapter_path=str(C2),optimizer_sha256=sha(C2.parent/'optimizer.pt'),changed_tensor_count=changed,
        probe_logprobs=saved_probe.tolist(),probe_input_ids=[[2,3,4,5]],memory=mem_report(),inventory=inventory,wall_seconds=time.time()-started,
        versions={'torch':torch.__version__,'transformers':__import__('transformers').__version__,'peft':__import__('peft').__version__})
    write(out/'result.json',result)
    write(RUN/'c2_checkpoint_freeze.json',{'adapter_sha256':result['adapter_sha256'],'parent_adapter_sha256':result['parent_adapter_sha256'],
        'config_sha256':sha(CONFIG),'data_sha256':sha(RUN/'gravity_c2_teachers_74.jsonl'),'optimizer_steps':20,'created_unix':time.time()})


def reload_sft():
    import torch
    from utils.run_sft_augmentation import load_model
    c=verify_sources();saved=read(RUN/'sft_train/result.json')
    if sha(C2/'adapter_model.safetensors')!=saved['adapter_sha256']:raise ValueError('C2 checkpoint changed')
    model,tok,inventory=load_model(C2,False,c['sft']);actual=probe(model);expected=torch.tensor(saved['probe_logprobs'])
    torch.testing.assert_close(actual,expected,atol=.01,rtol=1e-3)
    write(RUN/'sft_reload.json',{'status':'completed','max_logprob_delta':float((actual-expected).abs().max()),
        'adapter_sha256':sha(C2/'adapter_model.safetensors'),'memory':mem_report()})


def make_engine(c):
    import torch
    from vllm import LLM
    torch.cuda.set_per_process_memory_fraction(min(.95,LIMIT/torch.cuda.get_device_properties(0).total_memory))
    return LLM(model=str(BASE),tokenizer=str(BASE),dtype='bfloat16',tensor_parallel_size=1,
        enable_lora=True,max_lora_rank=8,max_loras=1,enable_prefix_caching=False,max_model_len=20480,
        max_num_seqs=8,max_num_batched_tokens=512,kv_cache_memory_bytes=4*2**30,gpu_memory_utilization=.10,
        compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4,8],'max_cudagraph_capture_size':8},
        enforce_eager=False,seed=c['seed'],generation_config='vllm',logprobs_mode='raw_logprobs')


def infer(stage):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    c=verify_sources();engine=make_engine(c);tok=engine.get_tokenizer();c2_hash=read(RUN/'c2_checkpoint_freeze.json')['adapter_sha256']
    if sha(C2/'adapter_model.safetensors')!=c2_hash:raise ValueError('C2 checkpoint changed')
    if stage=='comparison':
        targets=[('C2_diagnosis','SFT_C2',C2,'diagnosis',1,1,0),('C_tune','SFT_C',PARENT,'tune',1,8,0),('C2_tune','SFT_C2',C2,'tune',1,8,0)]
    elif stage in ['rollout8','rollout4']:
        n=8 if stage=='rollout8' else 4;targets=[('C2','SFT_C2',C2,'diagnosis',n,1,0 if n==8 else 8)]
    elif stage=='dpo_tune':targets=[('DPO','DPO',DPO,'tune',1,8,0)]
    else:
        frozen=read(RUN/'dpo_checkpoint_freeze.json')
        if sha(DPO/'adapter_model.safetensors')!=frozen['adapter_sha256']:raise ValueError('Frozen DPO changed before Locked Final')
        targets=[('C2','SFT_C2',C2,'locked_final',1,8,0),('DPO','DPO',DPO,'locked_final',1,8,0)]
    out=RUN/stage;out.mkdir(exist_ok=False)
    for target_index,(name,arm,adapter,panel_name,n,batch_size,offset) in enumerate(targets):
        panel=rows(RUN/f'{panel_name}_panel.jsonl');fingerprint=sha(adapter/'adapter_model.safetensors')
        request=LoRARequest(stage+'_'+name,target_index+1,str(adapter));started=time.perf_counter();count=0
        for start in range(0,len(panel),batch_size):
            batch=panel[start:start+batch_size]
            prompts=[tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in batch]
            if any(len(tok.encode(p,add_special_tokens=False))+16384>20480 for p in prompts):raise ValueError('No prompt or completion truncation allowed')
            params=[SamplingParams(n=n,temperature=1. if n>1 else 0.,top_p=1.,top_k=-1,max_tokens=16384,
                logprobs=1 if n>1 else None,seed=c['seed']+start+offset,skip_special_tokens=False) for _ in batch]
            outputs=engine.generate(prompts,params,lora_request=request,use_tqdm=False)
            for original,prompt,output in zip(batch,prompts,outputs):
                for k,o in enumerate(output.outputs):
                    ids=list(o.token_ids);lp=[]
                    if n>1:
                        if o.logprobs is None or len(o.logprobs)!=len(ids):raise ValueError('Missing sampled behavior probabilities')
                        lp=[float(v[t].logprob) for t,v in zip(ids,o.logprobs)]
                        if not all(math.isfinite(v) for v in lp):raise ValueError('Nonfinite behavior probabilities')
                    if o.finish_reason=='stop' and (not ids or ids[-1] not in {151643,151645}):raise ValueError('Stopped response missing real EOS')
                    row={'sample_id':str(original['id']),'prompt_id':str(original['id']),'task_type':'gravity',
                        'prompt':original['prompt'],'expected_answer':original['answer'],'completion':o.text,
                        'prompt_token_ids':list(output.prompt_token_ids),'completion_token_ids':ids,'completion_tokens':len(ids),
                        'behavior_logprobs':lp,'finish_reason':o.finish_reason,'stop_reason':o.stop_reason,
                        'truncated':o.finish_reason=='length','eos_included':o.finish_reason=='stop','max_tokens':16384,
                        'group_index':k+offset,'policy_version':arm+':'+fingerprint,
                        'policy_fingerprint':{'adapter_model.safetensors':fingerprint,'base_revision':c['base_revision']},
                        'source':'vllm_rollout' if n>1 else 'vllm_evaluation','source_split':'train' if panel_name=='diagnosis' else 'validation',
                        'provenance':original.get('provenance',{}),'fixture':False,'temperature':1. if n>1 else 0.,
                        'do_sample':n>1,'top_p':1.,'top_k':-1,'thinking_enabled':True,'seed':c['seed']+start+offset,
                        'arm':arm,'stage':stage,'public_model_input_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
                    append(out/f'{name}_raw.jsonl',row);count+=1
            engine.collective_rpc(mem_report)
        write(out/f'{name}_result.json',{'status':'completed','records':count,'generation_seconds':time.perf_counter()-started,
            'adapter_sha256':fingerprint,'memory':engine.collective_rpc(mem_report)})


def train_dpo():
    import torch
    import utils.course_worker as cw
    from utils.dpo_gravity_worker import load_training_policy
    from safetensors.torch import load_file
    c=verify_sources();expected=read(RUN/'c2_checkpoint_freeze.json')['adapter_sha256']
    if sha(C2/'adapter_model.safetensors')!=expected:raise ValueError('Reference C2 changed')
    if read(RUN/'readiness.json')['pairs']<16:raise ValueError('DPO readiness gate failed')
    cw.load_policy=load_training_policy;torch.manual_seed(c['seed']);DPO.parent.mkdir(parents=True,exist_ok=False)
    job={'method':'dpo','model':str(BASE),'adapter':str(C2),'sft_adapter':str(C2),'run':str(RUN),'output':str(DPO.parent),
        'rollout':str(RUN/'rollouts_audited.jsonl'),'policy_version':'SFT_C2:'+expected,'seed':c['seed'],'round':1}
    result=cw.train_dpo(job,c)
    a=load_file(str(C2/'adapter_model.safetensors'));b=load_file(str(DPO/'adapter_model.safetensors'))
    result['changed_tensors']=sum(not torch.equal(a[k],b[k]) for k in a);result['memory']=mem_report()
    if not result['changed_tensors'] or not any(x['gradient_norm_before_clip']>0 for x in result['training_curve']):raise ValueError('DPO did not update parameters')
    if sha(C2/'adapter_model.safetensors')!=expected:raise ValueError('DPO updated its parent/reference on disk')
    write(RUN/'dpo_training_result.json',result)


def reload_dpo():
    import torch
    import utils.course_worker as cw
    from utils.dpo_gravity_worker import load_training_policy
    verify_sources();cw.load_policy=load_training_policy
    result=cw.reload_check({'model':str(BASE),'method':'dpo','adapter':str(DPO)},read(CONFIG))
    state=torch.load(DPO.parent/'trainer.pt',weights_only=True,map_location='cpu')
    result.update(optimizer_steps_total=state['optimizer_steps_total'],memory=mem_report(),adapter_sha256=sha(DPO/'adapter_model.safetensors'))
    write(RUN/'dpo_reload.json',result)


def metrics(rs):
    from utils.round2_evaluation import wilson
    n=len(rs);good=[r for r in rs if r['correct']];verified=[r for r in good if r['proof_audit']['verified']]
    return {'n':n,'correct':len(good),'accuracy':len(good)/n,'wilson95':wilson(len(good),n),
        'proof_pass':sum(r['proof_audit']['verified'] for r in rs),'proof_rate':sum(r['proof_audit']['verified'] for r in rs)/n,
        'ols_certified':sum(ols_certified(r['proof_audit']) for r in rs),'ols_rate':sum(ols_certified(r['proof_audit']) for r in rs)/n,
        'format_valid':sum(r['format_valid'] for r in rs),'truncated':sum(r['truncated'] for r in rs),
        'truncation_rate':sum(r['truncated'] for r in rs)/n,'mean_tokens':statistics.mean(len(r['completion_token_ids']) for r in rs),
        'median_tokens':statistics.median(len(r['completion_token_ids']) for r in rs),
        'official_correct_mean_tokens':statistics.mean(len(r['completion_token_ids']) for r in good) if good else None,
        'correct_verified_mean_tokens':statistics.mean(len(r['completion_token_ids']) for r in verified) if verified else None,
        'total_tokens':sum(len(r['completion_token_ids']) for r in rs),
        'failure_reasons_nonexclusive':dict(Counter(x for r in rs for x in r['proof_audit']['reasons']))}


def compare(a,b):
    from utils.round2_evaluation import paired_mean_analysis,mcnemar_exact
    aa={r['sample_id']:r for r in a};bb={r['sample_id']:r for r in b}
    if set(aa)!=set(bb) or len(aa)!=len(a) or len(bb)!=len(b):raise ValueError('Unmatched evaluation questions')
    for key in aa:
        if aa[key]['prompt_token_ids']!=bb[key]['prompt_token_ids'] or aa[key]['max_tokens']!=bb[key]['max_tokens']:raise ValueError('Unmatched evaluation context')
    gained=sum(not aa[k]['correct'] and bb[k]['correct'] for k in aa);lost=sum(aa[k]['correct'] and not bb[k]['correct'] for k in aa)
    both=[k for k in aa if aa[k]['correct'] and bb[k]['correct']]
    result={'before':metrics(a),'after':metrics(b),'gained_correct':gained,'lost_correct':lost,'mcnemar_exact_pvalue':mcnemar_exact(lost,gained),
        'accuracy_delta_bootstrap':paired_mean_analysis([int(bb[k]['correct'])-int(aa[k]['correct']) for k in aa]),
        'length_delta_bootstrap':paired_mean_analysis([len(bb[k]['completion_token_ids'])-len(aa[k]['completion_token_ids']) for k in aa]),
        'both_correct_n':len(both),'both_correct_length_delta':paired_mean_analysis([len(bb[k]['completion_token_ids'])-len(aa[k]['completion_token_ids']) for k in both]) if both else None,
        'per_question':[{'id':k,'before_correct':aa[k]['correct'],'after_correct':bb[k]['correct'],'before_verified':aa[k]['proof_audit']['verified'],
            'after_verified':bb[k]['proof_audit']['verified'],'before_tokens':len(aa[k]['completion_token_ids']),'after_tokens':len(bb[k]['completion_token_ids'])} for k in aa]}
    return result


def score_comparison():
    for panel,a_path,b_path in [('diagnosis',RUN/'cached_SFT_C_diagnosis_raw.jsonl',RUN/'comparison/C2_diagnosis_raw.jsonl'),
                               ('tune',RUN/'comparison/C_tune_raw.jsonl',RUN/'comparison/C2_tune_raw.jsonl')]:
        a=audit_nl_rows(rows(a_path));b=audit_nl_rows(rows(b_path))
        write_rows(RUN/f'C_{panel}_scored.jsonl',a);write_rows(RUN/f'C2_{panel}_scored.jsonl',b)
        write(RUN/f'C_vs_C2_{panel}.json',compare(a,b))


def readiness(supplement=False):
    rs=rows(RUN/'rollout8/C2_raw.jsonl')
    if len(rs)!=208:raise ValueError('Initial readiness rollout incomplete')
    if supplement:
        extra=rows(RUN/'rollout4/C2_raw.jsonl')
        if len(extra)!=104:raise ValueError('Supplement incomplete')
        rs+=extra
    expected=read(RUN/'c2_checkpoint_freeze.json')['adapter_sha256'];panel={r['id']:r for r in rows(RUN/'diagnosis_panel.jsonl')}
    for r in rs:
        if r['prompt']!=panel[r['sample_id']]['prompt'] or r['policy_version']!='SFT_C2:'+expected:raise ValueError('Readiness rollout provenance mismatch')
        if len(r['completion_token_ids'])!=len(r['behavior_logprobs']):raise ValueError('Behavior alignment mismatch')
    audited=audit_nl_rows(rs);pairs=pair_api()(audited);suffix='12' if supplement else '8'
    write_rows(RUN/f'rollouts_audited_{suffix}.jsonl',audited)
    detail=[{'sample_id':p['sample_id'],'chosen_index':p['chosen']['group_index'],'rejected_index':p['rejected']['group_index'],
        'chosen_tokens':len(p['chosen']['completion_token_ids']),'rejected_tokens':len(p['rejected']['completion_token_ids']),
        'chosen_proof_verified':p['chosen']['proof_audit']['verified'],'rejected_proof_verified':p['rejected']['proof_audit']['verified']} for p in pairs]
    write_rows(RUN/f'preferences_{suffix}.jsonl',detail)
    result={'status':'ready' if len(pairs)>=16 else 'no_preference_signal','rollouts':len(audited),'prompts':26,'pairs':len(pairs),
        'pair_coverage':len(pairs)/26,'metrics':metrics(audited),'maximum_candidates_per_prompt':12 if supplement else 8,
        'proofs_by_prompt':{k:sum(r['sample_id']==k and r['proof_audit']['verified'] for r in audited) for k in panel}}
    write(RUN/f'readiness_{suffix}.json',result)
    if len(pairs)>=16:write_rows(RUN/'rollouts_audited.jsonl',audited)
    return result


def score_dpo(stage):
    if stage=='tune':a=rows(RUN/'C2_tune_scored.jsonl');b=audit_nl_rows(rows(RUN/'dpo_tune/DPO_raw.jsonl'))
    else:a=audit_nl_rows(rows(RUN/'final/C2_raw.jsonl'));b=audit_nl_rows(rows(RUN/'final/DPO_raw.jsonl'))
    write_rows(RUN/f'dpo_{stage}_baseline_scored.jsonl',a);write_rows(RUN/f'dpo_{stage}_scored.jsonl',b)
    result=compare(a,b);am,bm=result['before'],result['after']
    result['dpo_b_point_gate']=bm['accuracy']>=am['accuracy'] and bm['mean_tokens']<am['mean_tokens'] and bm['truncation_rate']<=am['truncation_rate'] and bm['proof_rate']>=am['proof_rate']
    result['statistical_noninferiority_claim']=False
    write(RUN/f'C2_vs_DPO_{stage}.json',result)


def invoke(stage):
    c=verify_sources();available=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    if available<c['sft']['minimum_available_host_bytes']:raise RuntimeError('Insufficient shared memory; other processes left untouched')
    if stage in ['comparison','rollout8','rollout4','dpo_tune','final']:
        image=subprocess.check_output(['docker','image','inspect',c['backend_image'],'--format','{{.Id}}'],text=True).strip()
        if image!=c['backend_image_id']:raise ValueError('Pinned inference image changed')
        cmd=['docker','run','--rm','--name','qwen-gravity-c2-'+stage,'--network','none','--gpus','all','--ipc','host','--user',f'{os.getuid()}:{os.getgid()}',
            '-e','HOME=/tmp','-e','VLLM_ENABLE_V1_MULTIPROCESSING=0','-e','HF_HUB_OFFLINE=1','-e','TOKENIZERS_PARALLELISM=false',
            '-v',f'{ROOT}:{ROOT}','-w',str(ROOT),'--entrypoint','python3',c['backend_image'],'utils/run_gravity_sft_c2.py',stage]
    else:cmd=[str(PYTHON),'utils/run_gravity_sft_c2.py',stage]
    started=time.time()
    with (RUN/f'{stage}.log').open('x') as f:
        proc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
    append(RUN/'completion_hooks.jsonl',{'stage':stage,'exit_code':proc.returncode,'start':started,'end':time.time(),'mechanism':'blocking child completion; no metric polling'})
    if proc.returncode:raise RuntimeError(stage+' failed; see preserved log')


def finish(status,error=None):
    initial=read(RUN/'initial_state.json');integrity={p:sha(ROOT/p)==h for p,h in initial['protected_hashes'].items()}
    result={'status':status,'error':error,'protected_files':len(integrity),'all_protected_unchanged':all(integrity.values()),
        'protected_mismatches':[p for p,v in integrity.items() if not v],'original_C_sha256':sha(PARENT/'adapter_model.safetensors'),
        'sft_training_completed':(RUN/'sft_train/result.json').exists(),'dpo_training_completed':(RUN/'dpo_training_result.json').exists(),
        'locked_final_completed':(RUN/'C2_vs_DPO_final.json').exists(),'hosted_api_calls':0,'time':time.time()}
    write(RUN/'final_status.json',result)
    if WEIGHTS.exists():
        write(RUN/'checkpoint_archive_manifest.json',{'files':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size}
            for p in sorted(WEIGHTS.rglob('*')) if p.is_file()},'original_C_overwritten':False})
    from utils.report_gravity_sft_c2 import report
    report()


def all_stages():
    try:
        prepare()
        with (RUN/'cpu_tests.log').open('x') as f:
            tests=subprocess.run([str(PYTHON),'-m','unittest','tests.test_gravity_sft_c2','tests.test_dpo_sft_c_gravity_diagnosis',
                'tests.test_dpo_gravity_sft_c','tests.test_round2_numerics','tests.test_course_experiment','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(ROOT)+':'+str(ROOT/'tests'),'TOKENIZERS_PARALLELISM':'false'})
        write(RUN/'cpu_gate.json',{'exit_code':tests.returncode,'log_sha256':sha(RUN/'cpu_tests.log')})
        if tests.returncode:raise RuntimeError('CPU tests failed; GPU not started')
        invoke('sft_train');invoke('sft_reload');invoke('comparison');score_comparison()
        invoke('rollout8');signal=readiness()
        if signal['pairs']<16:invoke('rollout4');signal=readiness(True)
        write(RUN/'readiness.json',signal)
        if signal['pairs']<16:finish('no_preference_signal');return
        invoke('dpo_train');invoke('dpo_reload');invoke('dpo_tune');score_dpo('tune')
        write(RUN/'dpo_checkpoint_freeze.json',{'adapter_sha256':sha(DPO/'adapter_model.safetensors'),'created_unix':time.time(),
            'tune_comparison_sha256':sha(RUN/'C2_vs_DPO_tune.json'),'parameters_changed_after_tune':False})
        invoke('final');score_dpo('final');finish('completed')
    except BaseException as e:
        write(RUN/f'failure_{time.time_ns()}.json',{'error':repr(e),'traceback':traceback.format_exc()})
        if not (RUN/'final_status.json').exists():finish('failed',repr(e))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','prepare','sft_train','sft_reload','comparison','rollout8','rollout4','dpo_train','dpo_reload','dpo_tune','final']);a=p.parse_args()
    if a.stage=='all':all_stages()
    elif a.stage=='prepare':prepare()
    elif a.stage=='sft_train':train_sft()
    elif a.stage=='sft_reload':reload_sft()
    elif a.stage=='dpo_train':train_dpo()
    elif a.stage=='dpo_reload':reload_dpo()
    else:infer(a.stage)


if __name__=='__main__':main()
