"""Isolated C3 native-Thinking repair and completion-driven SFT/DPO chain.

Core rendering, encoding, mathematics and verification remain in Notebooks 01/03.
No Hosted API, Replay, default SFT changes or edits to the DPO objective.
"""
from __future__ import annotations
import argparse, ast, copy, csv, hashlib, json, math, os, pathlib, re, shutil, statistics, subprocess, sys, time, traceback
from collections import Counter
from contextlib import contextmanager

ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from utils.dpo_gravity_worker import BASE,PARENT,LIMIT,read,rows,sha,write,write_rows,append,nl_api,pair_api,mem_report
from utils.run_dpo_sft_c_gravity_diagnosis import selected_functions,actual_teachers,identity
from utils.run_gravity_sft_c2 import function_hashes,probe,make_engine,metrics,compare

RUN=ROOT/'data/processed/round7/gravity_sft_c3_dpo_20261117_r1'
C2_RUN=ROOT/'data/processed/round6/gravity_sft_c2_dpo_20261116_r1'
WEIGHTS=ROOT/'.autodl/gravity_sft_c3_dpo_20261117_r1'
C3=WEIGHTS/'C3/adapter'
DPO=WEIGHTS/'DPO/adapter'
CONFIG=ROOT/'configs/gravity_sft_c3_20261117.json'
PYTHON=ROOT/'.venv-low-memory/bin/python'
SOURCES=['01_Synthetic_Data_Engineering.ipynb','03_SFT.ipynb','04_DPO.ipynb','05_PPO.ipynb','06_GRPO.ipynb',
 'utils/run_gravity_sft_c3.py','utils/report_gravity_sft_c3.py','tests/test_gravity_sft_c3.py','configs/gravity_sft_c3_20261117.json',
 'utils/course_worker.py','utils/low_memory_logprobs.py','utils/dpo_gravity_worker.py','utils/run_gravity_sft_c2.py',
 'utils/run_sft_augmentation.py']
CPU_SUITES=['tests.test_gravity_sft_c3','tests.test_gravity_sft_c2','tests.test_dpo_sft_c_gravity_diagnosis',
 'tests.test_dpo_gravity_sft_c','tests.test_round2_numerics','tests.test_course_experiment']


def apis():
    from utils.run_gravity_sft_c2 import apis as c2_apis
    n,s=c2_apis()
    legacy=nl_api()
    n.update(selected_functions(read(ROOT/'01_Synthetic_Data_Engineering.ipynb'),
        {'gravity_c3_split_channels','gravity_c3_numeric_certificates','verify_gravity_dual_channel'},legacy))
    s.update(selected_functions(read(ROOT/'03_SFT.ipynb'),
        {'make_gravity_c3_record','encode_gravity_c3_example','train_gravity_c3_branch'},s))
    return n,s


def audit_rows(rs):
    api=apis()[0]
    result=copy.deepcopy(rs)
    for r in result:
        legacy=api['verify_gravity_natural_language'](r['prompt'],r['completion'],r['expected_answer'],r['finish_reason'],thinking_expected=True)
        proof=api['verify_gravity_dual_channel'](r['prompt'],r['completion'],r['expected_answer'],r['finish_reason'])
        r.update(proof_audit=proof,legacy_v1_2_audit=legacy,format_valid=proof['format_valid'],exact_trace_pass=proof['verified'],
            correct=bool(legacy['final_correct'] and not r['truncated'] and r['finish_reason']=='stop'),predicted=legacy.get('predicted'),
            proof_verifier_version=proof['version'],verifier_version='official_nvidia_metric_v3',
            output_contract='gravity_native_thinking_full_trace_v1',variant='original')
    return result


def verify_sources(cpu_required=False):
    c=read(CONFIG);reg=read(RUN/'registration.json');identity()
    for category in ['sources','input_hashes']:
        for p,h in reg[category].items():
            if sha(ROOT/p)!=h:raise ValueError('Frozen '+category+' changed: '+p)
    if cpu_required and read(RUN/'cpu_gate.json')['exit_code']!=0:raise ValueError('CPU gate failed')
    return c


def prepare():
    from transformers import AutoTokenizer
    if (RUN/'registration.json').exists():raise FileExistsError('C3 Run immutable; cannot overwrite registration')
    c=read(CONFIG);initial=read(RUN/'initial_state.json');id_result=identity()
    for p,h in initial['protected_hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('Protected history changed: '+p)
    old01=read(RUN/'source_before_repair/01_Synthetic_Data_Engineering.ipynb');now01=read(ROOT/'01_Synthetic_Data_Engineering.ipynb')
    unchanged01={'solve_gravity_prompt','_gravity_prompt_parts','_gravity_verifier_fit','verify_gravity_sample','gravity_c2_solver_reference',
        'gravity_nl_public_reference','gravity_nl_final_section','_gravity_nl_normalize','_gravity_nl_eval','verify_gravity_natural_language'}
    if function_hashes(old01,unchanged01)!=function_hashes(now01,unchanged01):raise ValueError('Solver or v1.2 changed')
    unchanged03={'encode_chat_example','course_sft_loss_sum','build_full_trace_records','choose_sft_records','render_gravity_c2_final_proof','make_gravity_c2_record','encode_gravity_c2_final_example'}
    old03=read(RUN/'source_before_repair/03_SFT.ipynb');now03=read(ROOT/'03_SFT.ipynb')
    if function_hashes(old03,unchanged03)!=function_hashes(now03,unchanged03):raise ValueError('Renderer or default SFT entry changed')
    n,s=apis();tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    teachers=rows(C2_RUN/'gravity_c2_teachers_74.jsonl');actual=actual_teachers()[0]
    actual_ids={r['source_id'] for _,r,_,_ in actual}
    if {r['source_id'] for r in teachers}!=actual_ids or len(teachers)!=74:raise ValueError('Actual 74 teachers differ from C2')
    records=[];audits=[]
    for old in teachers:
        sample=n['gravity_c2_solver_reference'](old['prompt'])
        record=s['make_gravity_c3_record'](sample,old['source_id'],old['source_row'],old['teacher_source_group'])
        for key in ['prompt','completion','reasoning','final_answer','canonical_solver_trace','source','source_id','source_row','source_split','teacher_source_group']:
            if record[key]!=old[key]:raise ValueError('C3 single-variable teacher mismatch: '+key)
        encoded=s['encode_gravity_c3_example'](tok,record,c['sft']['max_length']);p=encoded['prompt_length']
        response=tok.decode(encoded['input_ids'][p:],skip_special_tokens=False)
        thinking,final=n['gravity_c3_split_channels'](response)
        proof=n['verify_gravity_dual_channel'](record['prompt'],response,record['final_answer'])
        legacy=n['verify_gravity_natural_language'](record['prompt'],response,record['final_answer'],thinking_expected=True)
        if thinking!=record['reasoning'] or final!=record['assistant_content'] or not proof['verified']:
            raise ValueError('C3 serialized teacher failed: '+record['source_id']+':'+str(proof['reasons']))
        if encoded['labels'][:p]!=[-100]*p or encoded['labels'][p:]!=encoded['input_ids'][p:]:raise ValueError('C3 loss mask mismatch')
        if not s['audit_trace_record'](record)['accepted']:raise ValueError('SFT trace audit failed')
        record['actual_token_counts']={'full_chat_tokens':len(encoded['input_ids']),
            'supervised_tokens':sum(v!=-100 for v in encoded['labels'][1:]),'thinking_text_tokens':len(tok.encode(thinking,add_special_tokens=False)),
            'final_text_tokens':len(tok.encode(final,add_special_tokens=False))}
        records.append(record)
        audits.append({'source_id':record['source_id'],'source_row':record['source_row'],'group':record['teacher_source_group'],
            'same_mathematical_proof_as_C2':True,'serialized_response':response,'dual_channel_audit':proof,'legacy_v1_2_final_only':legacy,
            'all_response_tokens_supervised':True,'prompt_masked':True,'truncated':False,'actual_token_counts':record['actual_token_counts']})
    write_rows(RUN/'gravity_c3_teachers_74.jsonl',records);write_rows(RUN/'teacher_channel_audit.jsonl',audits)
    inputs=[RUN/'gravity_c3_teachers_74.jsonl',C2_RUN/'gravity_c2_teachers_74.jsonl']
    for name in ['diagnosis','tune','locked_final']:
        data=rows(C2_RUN/f'{name}_panel.jsonl')
        if len(data)!={'diagnosis':26,'tune':48,'locked_final':100}[name]:raise ValueError('Panel count mismatch')
        write_rows(RUN/f'{name}_panel.jsonl',data);inputs.append(RUN/f'{name}_panel.jsonl')
    regression=rows(ROOT/c['regression']['source'])
    for task in c['regression']['tasks']:
        data=[r for r in regression if r['task_type']==task]
        if len(data)!=50:raise ValueError('Expected frozen regression 50/task')
        write_rows(RUN/f'{task}_panel.jsonl',data);inputs.append(RUN/f'{task}_panel.jsonl')
    # Recheck actual Train provenance and prompt isolation, rather than relying
    # solely on a historical lineage declaration.
    taskset=ROOT/'data/processed/tasksets/roman_gravity_equation_v1'
    with (taskset/'splits/train.csv').open() as f:train={r['id']:r for r in csv.DictReader(f)}
    normalize=lambda x:' '.join(x.casefold().split())
    held={}
    for split in ['validation','holdout_test']:
        with (taskset/f'splits/{split}.csv').open() as f:held[split]={normalize(r['prompt']) for r in csv.DictReader(f)}
    for r in records:
        if r['source_id'] not in train or r['prompt']!=train[r['source_id']]['prompt'] or r['source_row']!=train[r['source_id']]['source_row']:raise ValueError('Teacher is not actual Train')
        if any(normalize(r['prompt']) in pool for pool in held.values()):raise ValueError('Teacher held-out overlap')
    inputs += [taskset/'dataset_manifest.json']+[taskset/f'splits/{x}.csv' for x in ['train','validation','holdout_test']]
    write(RUN/'teacher_channel_summary.json',{'teachers':74,'historical_r1':64,'added_real_c':10,'dual_verified':sum(a['dual_channel_audit']['verified'] for a in audits),
        'legacy_final_only_verified':sum(a['legacy_v1_2_final_only']['verified'] for a in audits),'all_math_tokens_supervised':74,
        'supervised_tokens_per_epoch':sum(r['actual_token_counts']['supervised_tokens'] for r in records),
        'max_full_chat_tokens':max(r['actual_token_counts']['full_chat_tokens'] for r in records),'same_proof_as_C2':True,'replay':False,
        'train_source_and_validation_holdout_prompt_isolation_passed':True})
    write(RUN/'registration.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sources':{p:sha(ROOT/p) for p in SOURCES},'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
        'config':c,'identity':id_result,'created_unix':time.time(),'single_training_variable':'assistant channels only, same C parent/74 proofs/hyperparameters',
        'selection_rule':c['selection'],'regression_used_for_selection':False,
        'validation_note':'Tune48/Locked Final100 are historical Validation panels, not untouched official Holdout; Final evaluated once after selection and optional DPO freeze.'})


def train_sft():
    import torch
    from safetensors.torch import load_file
    from utils.run_sft_augmentation import load_model
    c=verify_sources(True);cfg=c['sft'];out=RUN/'sft_train';out.mkdir(exist_ok=False)
    started=time.time();model,tok,inventory=load_model(PARENT,True,cfg)
    previous=load_file(str(PARENT/'adapter_model.safetensors'));epoch_results=[];history=[]
    def on_step(event):
        append(out/'training_curve.jsonl',event);history.append(event)
        if event['torch_peak_reserved_bytes']>LIMIT:raise RuntimeError('SFT memory limit')
    def on_epoch(epoch,model,optimizer):
        adapter=WEIGHTS/f'epoch{epoch}/adapter';adapter.parent.mkdir(parents=True,exist_ok=False)
        was_training=model.training;saved_probe=probe(model);model.train(was_training)
        model.save_pretrained(adapter);tok.save_pretrained(adapter);torch.save(optimizer.state_dict(),adapter.parent/'optimizer.pt')
        now=load_file(str(adapter/'adapter_model.safetensors'));changed=sum(not torch.equal(previous[k],now[k]) for k in previous)
        info={'epoch':epoch,'optimizer_steps':len(history),'adapter_path':str(adapter),'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
            'optimizer_sha256':sha(adapter.parent/'optimizer.pt'),'probe_logprobs':saved_probe.tolist(),'changed_tensor_count':changed,'memory':mem_report()}
        if not changed or len(history)!=epoch*10:raise ValueError('Missing genuine epoch updates')
        epoch_results.append(info);write(out/f'epoch{epoch}.json',info)
    result=apis()[1]['train_gravity_c3_branch'](model,tok,rows(RUN/'gravity_c3_teachers_74.jsonl'),cfg,on_step,on_epoch)
    result.pop('optimizer_state');result.update(epoch_checkpoints=epoch_results,inventory=inventory,memory=mem_report(),wall_seconds=time.time()-started,
        parent_adapter_sha256=sha(PARENT/'adapter_model.safetensors'),versions={'torch':torch.__version__,'transformers':__import__('transformers').__version__,'peft':__import__('peft').__version__})
    per=read(RUN/'teacher_channel_summary.json')['supervised_tokens_per_epoch']
    if result['optimizer_steps']!=20 or result['supervised_tokens']!=per*2 or len(epoch_results)!=2:raise ValueError('Incomplete C3 training')
    if not all(math.isfinite(e['gradient_norm']) and math.isfinite(e['token_mean_loss']) for e in result['history']):raise ValueError('Nonfinite training')
    write(out/'result.json',result)


def reload_sft():
    import torch,gc
    from utils.run_sft_augmentation import load_model
    c=verify_sources(True);reports=[]
    for epoch in [1,2]:
        saved=read(RUN/f'sft_train/epoch{epoch}.json');adapter=WEIGHTS/f'epoch{epoch}/adapter'
        if sha(adapter/'adapter_model.safetensors')!=saved['adapter_sha256']:raise ValueError('Epoch weight changed')
        model,tok,_=load_model(adapter,False,c['sft']);actual=probe(model);expected=torch.tensor(saved['probe_logprobs'])
        torch.testing.assert_close(actual,expected,atol=.01,rtol=1e-3)
        reports.append({'epoch':epoch,'adapter_sha256':saved['adapter_sha256'],'max_logprob_delta':float((actual-expected).abs().max()),'memory':mem_report()})
        del model,tok;gc.collect();torch.cuda.empty_cache()
    write(RUN/'sft_reload.json',{'status':'completed','epochs':reports})


def selected_adapter():
    freeze=read(RUN/'c3_checkpoint_freeze.json')
    if sha(C3/'adapter_model.safetensors')!=freeze['adapter_sha256']:raise ValueError('Frozen C3 changed')
    return C3


def inference_targets(stage):
    if stage=='tune_epochs':return [('C_tune','SFT_C',PARENT,'tune',1,8,0)]+[(f'C3_epoch{e}_tune',f'SFT_C3_epoch{e}',WEIGHTS/f'epoch{e}/adapter','tune',1,8,0) for e in [1,2]]
    selected_adapter()
    if stage=='comparison':
        return [(arm+'_'+panel,'SFT_'+arm,adapter,panel,1,1 if panel=='diagnosis' else 8,0)
            for panel in ['diagnosis','numeral_system','equation_numeric'] for arm,adapter in [('C',PARENT),('C3',C3)]]
    if stage in ['rollout8','rollout4']:return [('C3','SFT_C3',C3,'diagnosis',8 if stage=='rollout8' else 4,1,0 if stage=='rollout8' else 8)]
    if stage=='dpo_tune':return [('DPO','DPO',DPO,'tune',1,8,0)]
    if stage=='final':
        targets=[('C','SFT_C',PARENT,'locked_final',1,8,0),('C3','SFT_C3',C3,'locked_final',1,8,0)]
        if (RUN/'dpo_checkpoint_freeze.json').exists():
            lock=read(RUN/'dpo_checkpoint_freeze.json')
            if sha(DPO/'adapter_model.safetensors')!=lock['adapter_sha256']:raise ValueError('DPO changed before Final')
            targets.append(('DPO','DPO',DPO,'locked_final',1,8,0))
        return targets
    raise ValueError('Unknown inference stage')


def infer(stage):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    c=verify_sources(True);targets=inference_targets(stage);out=RUN/stage;out.mkdir(exist_ok=False)
    # The directory's exclusive creation is the one-shot Locked Final guard.
    engine=make_engine(c);tok=engine.get_tokenizer()
    for index,(name,arm,adapter,panel_name,n,batch_size,offset) in enumerate(targets):
        panel=rows(RUN/f'{panel_name}_panel.jsonl');fingerprint=sha(adapter/'adapter_model.safetensors')
        request=LoRARequest(stage+'_'+name,index+1,str(adapter));started=time.perf_counter();count=0
        for start in range(0,len(panel),batch_size):
            batch=panel[start:start+batch_size]
            prompts=[tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in batch]
            if any(len(tok.encode(p,add_special_tokens=False))+16384>20480 for p in prompts):raise ValueError('No prompt truncation permitted')
            params=[SamplingParams(n=n,temperature=1. if n>1 else 0.,top_p=1.,top_k=-1,max_tokens=16384,
                logprobs=1 if n>1 else None,seed=c['seed']+start+offset,skip_special_tokens=False) for _ in batch]
            outputs=engine.generate(prompts,params,lora_request=request,use_tqdm=False)
            for original,prompt,output in zip(batch,prompts,outputs):
                for k,o in enumerate(output.outputs):
                    ids=list(o.token_ids);lp=[]
                    if n>1:
                        if o.logprobs is None or len(o.logprobs)!=len(ids):raise ValueError('Missing behavior logprobs')
                        lp=[float(v[t].logprob) for t,v in zip(ids,o.logprobs)]
                        if not all(math.isfinite(v) for v in lp):raise ValueError('Nonfinite behavior probability')
                    if o.finish_reason=='stop' and (not ids or ids[-1] not in {151643,151645}):raise ValueError('Missing true EOS')
                    row={'sample_id':str(original['id']),'prompt_id':str(original['id']),'task_type':original['task_type'],
                        'prompt':original['prompt'],'expected_answer':original['answer'],'completion':o.text,
                        'prompt_token_ids':list(output.prompt_token_ids),'completion_token_ids':ids,'completion_tokens':len(ids),
                        'behavior_logprobs':lp,'finish_reason':o.finish_reason,'stop_reason':o.stop_reason,'truncated':o.finish_reason=='length',
                        'eos_included':o.finish_reason=='stop','max_tokens':16384,'group_index':k+offset,
                        'policy_version':arm+':'+fingerprint,'policy_fingerprint':{'adapter_model.safetensors':fingerprint,'base_revision':c['base_revision']},
                        'source':'vllm_rollout' if n>1 else 'vllm_evaluation','source_split':'train' if panel_name=='diagnosis' else 'validation',
                        'provenance':original.get('provenance',{}),'fixture':False,'temperature':1. if n>1 else 0.,'do_sample':n>1,
                        'top_p':1.,'top_k':-1,'thinking_enabled':True,'seed':c['seed']+start+offset,'arm':arm,'stage':stage,
                        'equation_subtype':original.get('equation_subtype',''),'public_model_input_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
                    append(out/f'{name}_raw.jsonl',row);count+=1
            engine.collective_rpc(mem_report)
        write(out/f'{name}_result.json',{'status':'completed','records':count,'generation_seconds':time.perf_counter()-started,
            'adapter_sha256':fingerprint,'memory':engine.collective_rpc(mem_report)})


def score_path(raw,out):
    rs=rows(RUN/raw);audited=audit_rows(rs);write_rows(RUN/out,audited);return audited


def selection_key(epoch,result):
    return (result['proof_pass'],result['correct'],-result['truncated'],-result['mean_tokens'],-epoch)


def select_checkpoint():
    c=verify_sources(True);results={}
    baseline=score_path('tune_epochs/C_tune_raw.jsonl','C_tune_scored.jsonl')
    for epoch in [1,2]:
        data=score_path(f'tune_epochs/C3_epoch{epoch}_tune_raw.jsonl',f'C3_epoch{epoch}_tune_scored.jsonl')
        comp=compare(baseline,data);write(RUN/f'C_vs_C3_epoch{epoch}_tune.json',comp);results[epoch]=comp['after']
    best=max(results,key=lambda e:selection_key(e,results[e]))
    source=WEIGHTS/f'epoch{best}/adapter';C3.parent.mkdir(parents=True,exist_ok=False);shutil.copytree(source,C3)
    shutil.copyfile(source.parent/'optimizer.pt',C3.parent/'optimizer.pt')
    freeze={'selected_epoch':best,'adapter_sha256':sha(C3/'adapter_model.safetensors'),'parent_adapter_sha256':c['parent_adapter_sha256'],
        'selection_rule':c['selection'],'tune_metrics_by_epoch':results,'teacher_sha256':sha(RUN/'gravity_c3_teachers_74.jsonl'),
        'config_sha256':sha(CONFIG),'selected_before_locked_final':True,'created_unix':time.time()}
    write(RUN/'c3_checkpoint_freeze.json',freeze)
    write_rows(RUN/'C3_tune_scored.jsonl',rows(RUN/f'C3_epoch{best}_tune_scored.jsonl'))
    write(RUN/'C_vs_C3_tune.json',read(RUN/f'C_vs_C3_epoch{best}_tune.json'))


def score_regression(rs):
    from utils.benchmark_training import verifier
    official=verifier();api=nl_api();result=copy.deepcopy(rs)
    for r in result:
        final,_=api['gravity_nl_final_section'](r['completion'])
        if '</think>' not in r['completion']:final=''
        predicted,method=official['extract_final_answer'](final,r['task_type'])
        correct=predicted is not None and official['verify_answer'](r['task_type'],predicted,r['expected_answer'])[0]
        r.update(correct=bool(correct and r['finish_reason']=='stop' and not r['truncated']),predicted=predicted,extraction_method=method,
            format_valid=bool(re.search(r'\\boxed\{[^{}]+\}',final)),
            proof_audit={'verified':False,'checks':{},'reasons':['full_proof_not_certified_for_regression_task']})
    return result


def score_comparison():
    for name in ['diagnosis','numeral_system','equation_numeric']:
        score=audit_rows if name=='diagnosis' else score_regression
        a=score(rows(RUN/f'comparison/C_{name}_raw.jsonl'));b=score(rows(RUN/f'comparison/C3_{name}_raw.jsonl'))
        write_rows(RUN/f'C_{name}_scored.jsonl',a);write_rows(RUN/f'C3_{name}_scored.jsonl',b)
        result=compare(a,b)
        if name!='diagnosis':result['proof_metric_scope']='not certified for regression tasks; compare official final answer only'
        if name=='equation_numeric':
            result['subtypes']={k:{'C':metrics([r for r in a if r['equation_subtype']==k]),'C3':metrics([r for r in b if r['equation_subtype']==k])} for k in sorted({r['equation_subtype'] for r in a})}
        write(RUN/f'C_vs_C3_{name}.json',result)


def readiness(supplement=False):
    rs=rows(RUN/'rollout8/C3_raw.jsonl')
    if len(rs)!=208:raise ValueError('Initial rollout incomplete')
    if supplement:
        extra=rows(RUN/'rollout4/C3_raw.jsonl')
        if len(extra)!=104:raise ValueError('Supplement incomplete')
        rs+=extra
    fingerprint=read(RUN/'c3_checkpoint_freeze.json')['adapter_sha256'];panel={r['id']:r for r in rows(RUN/'diagnosis_panel.jsonl')}
    for r in rs:
        if r['prompt']!=panel[r['sample_id']]['prompt'] or r['policy_version']!='SFT_C3:'+fingerprint:raise ValueError('Wrong sampled policy/prompt')
        if len(r['completion_token_ids'])!=len(r['behavior_logprobs']):raise ValueError('Unaligned behavior logprobs')
    audited=audit_rows(rs);pairs=pair_api()(audited);suffix='12' if supplement else '8'
    write_rows(RUN/f'rollouts_audited_{suffix}.jsonl',audited)
    write_rows(RUN/f'preferences_{suffix}.jsonl',[{'sample_id':p['sample_id'],'chosen_index':p['chosen']['group_index'],'rejected_index':p['rejected']['group_index'],
        'chosen_tokens':len(p['chosen']['completion_token_ids']),'rejected_tokens':len(p['rejected']['completion_token_ids']),
        'length_gap':p['length_gap'],'length_fraction':p['length_fraction'],'verifier':'gravity-dual-channel-v2.0'} for p in pairs])
    result={'status':'ready' if len(pairs)>=16 else 'no_preference_signal','rollouts':len(audited),'prompts':26,'pairs':len(pairs),
        'pair_coverage':len(pairs)/26,'metrics':metrics(audited),'maximum_candidates_per_prompt':12 if supplement else 8,
        'proofs_by_prompt':{k:sum(r['sample_id']==k and r['proof_audit']['verified'] for r in audited) for k in panel}}
    write(RUN/f'readiness_{suffix}.json',result)
    if len(pairs)>=16:write_rows(RUN/'rollouts_audited.jsonl',audited)
    return result


@contextmanager
def dual_dpo_auditor():
    """Only this explicit C3 job uses v2 auditing; loss/reference code unchanged."""
    import utils.dpo_gravity_worker as worker
    old=worker.audit_nl_rows
    worker.audit_nl_rows=audit_rows
    try:yield
    finally:worker.audit_nl_rows=old


def train_dpo():
    import torch
    import utils.course_worker as cw
    from utils.dpo_gravity_worker import load_training_policy
    from safetensors.torch import load_file
    c=verify_sources(True);selected_adapter();expected=read(RUN/'c3_checkpoint_freeze.json')['adapter_sha256']
    if read(RUN/'readiness.json')['pairs']<16:raise ValueError('C3 DPO gate failed')
    cw.load_policy=load_training_policy;torch.manual_seed(c['seed']);DPO.parent.mkdir(parents=True,exist_ok=False)
    job={'method':'dpo','model':str(BASE),'adapter':str(C3),'sft_adapter':str(C3),'run':str(RUN),'output':str(DPO.parent),
        'rollout':str(RUN/'rollouts_audited.jsonl'),'policy_version':'SFT_C3:'+expected,'seed':c['seed'],'round':1}
    with dual_dpo_auditor():result=cw.train_dpo(job,c)
    before=load_file(str(C3/'adapter_model.safetensors'));after=load_file(str(DPO/'adapter_model.safetensors'))
    result['changed_tensors']=sum(not torch.equal(before[k],after[k]) for k in before);result['memory']=mem_report()
    if not result['changed_tensors'] or not any(x['gradient_norm_before_clip']>0 for x in result['training_curve']):raise ValueError('No genuine DPO updates')
    if sha(C3/'adapter_model.safetensors')!=expected:raise ValueError('C3 Reference changed')
    write(RUN/'dpo_training_result.json',result)


def reload_dpo():
    import torch
    import utils.course_worker as cw
    from utils.dpo_gravity_worker import load_training_policy
    verify_sources(True);cw.load_policy=load_training_policy
    result=cw.reload_check({'model':str(BASE),'method':'dpo','adapter':str(DPO)},read(CONFIG))
    state=torch.load(DPO.parent/'trainer.pt',weights_only=True,map_location='cpu')
    result.update(optimizer_steps_total=state['optimizer_steps_total'],memory=mem_report(),adapter_sha256=sha(DPO/'adapter_model.safetensors'))
    write(RUN/'dpo_reload.json',result)


def score_dpo_tune():
    baseline=rows(RUN/'C3_tune_scored.jsonl');dpo=score_path('dpo_tune/DPO_raw.jsonl','DPO_tune_scored.jsonl')
    write(RUN/'C3_vs_DPO_tune.json',compare(baseline,dpo))
    write(RUN/'dpo_checkpoint_freeze.json',{'adapter_sha256':sha(DPO/'adapter_model.safetensors'),'created_unix':time.time(),
        'reference_c3_sha256':sha(C3/'adapter_model.safetensors'),'tune_sha256':sha(RUN/'C3_vs_DPO_tune.json'),'parameters_changed_after_tune':False})


def score_final():
    c=score_path('final/C_raw.jsonl','C_final_scored.jsonl');c3=score_path('final/C3_raw.jsonl','C3_final_scored.jsonl')
    write(RUN/'C_vs_C3_final.json',compare(c,c3))
    if (RUN/'final/DPO_raw.jsonl').exists():
        dpo=score_path('final/DPO_raw.jsonl','DPO_final_scored.jsonl');comp=compare(c3,dpo);a,b=comp['before'],comp['after']
        comp['dpo_b_point_gate']=b['accuracy']>=a['accuracy'] and b['mean_tokens']<a['mean_tokens'] and b['truncation_rate']<=a['truncation_rate'] and b['proof_rate']>=a['proof_rate']
        comp['statistical_noninferiority_claim']=False;write(RUN/'C3_vs_DPO_final.json',comp)


def invoke(stage):
    c=verify_sources(True)
    available=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    if available<c['sft']['minimum_available_host_bytes']:raise RuntimeError('Shared host memory insufficient; no other process touched')
    if stage in ['tune_epochs','comparison','rollout8','rollout4','dpo_tune','final']:
        image=subprocess.check_output(['docker','image','inspect',c['backend_image'],'--format','{{.Id}}'],text=True).strip()
        if image!=c['backend_image_id']:raise ValueError('Pinned inference image changed')
        cmd=['docker','run','--rm','--name','qwen-gravity-c3-'+stage,'--network','none','--gpus','all','--ipc','host','--user',f'{os.getuid()}:{os.getgid()}',
            '-e','HOME=/tmp','-e','VLLM_ENABLE_V1_MULTIPROCESSING=0','-e','HF_HUB_OFFLINE=1','-e','TOKENIZERS_PARALLELISM=false',
            '-v',f'{ROOT}:{ROOT}','-w',str(ROOT),'--entrypoint','python3',c['backend_image'],'utils/run_gravity_sft_c3.py',stage]
    else:cmd=[str(PYTHON),'utils/run_gravity_sft_c3.py',stage]
    started=time.time()
    with (RUN/f'{stage}.log').open('x') as f:
        proc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
    append(RUN/'completion_hooks.jsonl',{'stage':stage,'exit_code':proc.returncode,'start':started,'end':time.time(),'mechanism':'blocking child completion; no metric polling'})
    if proc.returncode:raise RuntimeError(stage+' failed; preserved stage log')


def finish(status,error=None):
    protected=read(RUN/'initial_state.json')['protected_hashes'];mismatch=[p for p,h in protected.items() if sha(ROOT/p)!=h]
    result={'status':status,'error':error,'sft_training_completed':(RUN/'sft_train/result.json').exists(),
        'dpo_training_completed':(RUN/'dpo_training_result.json').exists(),'locked_final_completed':(RUN/'C_vs_C3_final.json').exists(),
        'all_protected_unchanged':not mismatch,'protected_files':len(protected),'protected_mismatches':mismatch,
        'original_C_sha256':sha(PARENT/'adapter_model.safetensors'),'hosted_api_calls':0,'replay':False,'time':time.time()}
    write(RUN/'final_status.json',result)
    if WEIGHTS.exists():write(RUN/'checkpoint_archive_manifest.json',{'files':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(WEIGHTS.rglob('*')) if p.is_file()}})
    from utils.report_gravity_sft_c3 import report
    report()


def all_stages():
    try:
        prepare()
        with (RUN/'cpu_tests.log').open('x') as f:
            proc=subprocess.run([str(PYTHON),'-m','unittest',*CPU_SUITES,'-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(ROOT)+':'+str(ROOT/'tests'),'TOKENIZERS_PARALLELISM':'false'})
        write(RUN/'cpu_gate.json',{'exit_code':proc.returncode,'log_sha256':sha(RUN/'cpu_tests.log')})
        if proc.returncode:raise RuntimeError('CPU gate failed; no GPU start')
        invoke('sft_train');invoke('sft_reload');invoke('tune_epochs');select_checkpoint()
        invoke('comparison');score_comparison()
        invoke('rollout8');signal=readiness()
        if signal['pairs']<16:invoke('rollout4');signal=readiness(True)
        write(RUN/'readiness.json',signal)
        if signal['pairs']>=16:
            invoke('dpo_train');invoke('dpo_reload');invoke('dpo_tune');score_dpo_tune()
        invoke('final');score_final()
        finish('completed_with_dpo' if signal['pairs']>=16 else 'no_preference_signal')
    except BaseException as e:
        write(RUN/f'failure_{time.time_ns()}.json',{'error':repr(e),'traceback':traceback.format_exc()})
        if not (RUN/'final_status.json').exists():finish('failed',repr(e))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','prepare','sft_train','sft_reload','tune_epochs','comparison','rollout8','rollout4','dpo_train','dpo_reload','dpo_tune','final']);a=p.parse_args()
    if a.stage=='all':all_stages()
    elif a.stage=='prepare':prepare()
    elif a.stage=='sft_train':train_sft()
    elif a.stage=='sft_reload':reload_sft()
    elif a.stage=='dpo_train':train_dpo()
    elif a.stage=='dpo_reload':reload_dpo()
    else:infer(a.stage)


if __name__=='__main__':main()
