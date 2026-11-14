"""Register and supervise independent Round 2 GPU runs; preserve all attempts.

Teaching mathematics remains in the notebooks. This module only freezes inputs,
dispatches isolated workers, budgets time, and assembles measured evidence.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.course_protocol import (TASKSET,TASKS,check_protocol,digest,metrics,priors,
                                   read,read_rows,write,write_rows)
from utils.course_run import guard_job
from utils.round2_evaluation import paired_comparison,holm

PREP=ROOT/'data/processed/round2/round2_20261110_phaseab_v1'


def register(args):
    suite=args.suite.resolve()
    if suite.exists():raise FileExistsError('Register a fresh suite; never replace a prior run')
    design=read(args.design)
    model=args.model.resolve();adapter=args.adapter.resolve()
    if read(model/'download_manifest.json')['revision']!=design['model_revision']:
        raise ValueError('Wrong model Revision')
    if digest(adapter/'adapter_model.safetensors')!=design['parent_sft_adapter_sha256']:
        raise ValueError('Wrong selected SFT weights')
    for filename,item in read(model/'download_manifest.json')['files'].items():
        if digest(model/filename)!=item['sha256']:raise ValueError('Model download fingerprint mismatch')
    for panel in ('panels','phase_c_train_panels'):
        manifest=read(PREP/panel/'manifest.json')
        hashes=manifest.get('panel_hashes') or {k+'.jsonl':v['sha256'] for k,v in manifest['panels'].items()}
        for rel,sha in hashes.items():
            if digest(PREP/panel/rel)!=sha:raise ValueError('Prepared panel changed')
    suite.mkdir(parents=True)
    c=copy.deepcopy(design['runtime'])
    c.update(schema='qwen3-round2-gpu/v1',seed=design['seed'],
        device_limit_bytes=design['device_limit_bytes'],poll_seconds=design['poll_seconds'],
        allocator_fraction_of_limit=.9,eval_caps=design['evaluation']['caps'],
        sampling=design['sampling'],evaluation=design['evaluation'])
    for method in ('dpo','ppo','grpo'):
        c[method]=dict(design[method],mode=design['runtime']['mode'])
    source_paths=list((ROOT/'utils').glob('*.py'))+list(ROOT.glob('0[0-6]_*.ipynb'))+[args.design.resolve()]
    sources={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    model_files={p.name:digest(p) for p in model.iterdir() if p.is_file()}
    runs={}
    for method in ('baseline','readiness','dpo','ppo','grpo'):
        run=suite/(f'round2_{method}_'+suite.name)
        run.mkdir();runs[method]=str(run)
        rows=read_rows(PREP/'panels/phase_c_smoke.jsonl')
        write_rows(run/'panels/evaluation.jsonl',rows)
        train_method='grpo' if method=='readiness' else method
        if train_method in ('dpo','ppo','grpo'):
            panel='dpo_b' if train_method=='dpo' else train_method
            write_rows(run/'panels/train.jsonl',read_rows(PREP/f'phase_c_train_panels/{panel}.jsonl'))
        write(run/'public_priors.json',priors())
        frozen={str(p.relative_to(run)):digest(p) for p in run.rglob('*') if p.is_file()}
        protocol={'schema':c['schema'],'config':c,'config_sha256':digest(args.design),
            'source_commit':args.source_commit,'sources':sources,'frozen_files':frozen,
            'split_sha256':{s:digest(TASKSET/f'splits/{s}.csv') for s in ('train','validation','holdout_test')},
            'traces_sha256':digest(TASKSET/'real_train_traces.jsonl'),
            'method':method,'parent_sft_fingerprint':design['parent_sft_adapter_sha256'],
            'holdout_used':False,'final_panel_used_for_selection':False,
            'gpu_authorization':'User explicitly requested testing on port 41522 on 2026-11-14',
            'design_gpu_flag_superseded_by_explicit_user_authorization':True}
        write(run/'protocol.json',protocol)
    identity={key:str(getattr(args,key).absolute()) for key in
              ('model','adapter','inference_python','training_python','nemo_python')}
    write(suite/'suite.json',{'status':'registered','runs':runs,'config':c,'design':design,
        'source_commit':args.source_commit,'identity':identity,'model_files':model_files,
        'budget_seconds':args.budget_seconds,'on_policy_rounds':2,
        'phase_c_evaluation':'fixed 8 per task, both original and closed proof; no Holdout',
        'formal_training_started':False,'phase_d_requires_phase_c_acceptance':True})
    print(json.dumps({'status':'registered','suite':str(suite)}))


class Suite:
    def __init__(self,args):
        self.root=args.suite.resolve();self.reg=read(self.root/'suite.json');self.c=self.reg['config']
        self.state_path=self.root/'state.json'
        self.state=read(self.state_path) if self.state_path.exists() else {
            'status':'running','jobs':{},'checkpoints':{},'started_at_unix':time.time(),
            'deadline_unix':time.time()+self.reg['budget_seconds']}
        self.persist()
        for run in self.reg['runs'].values():check_protocol(run)
        for name,sha in self.reg['model_files'].items():
            if digest(Path(self.reg['identity']['model'])/name)!=sha:raise ValueError('Base model changed')
        if digest(Path(self.reg['identity']['adapter'])/'adapter_model.safetensors')!=self.reg['design']['parent_sft_adapter_sha256']:
            raise ValueError('Parent SFT changed')

    def persist(self):write(self.state_path,self.state)

    def job(self,method,key,kind,adapter=None,version='base',**options):
        identity=self.reg['identity'];run=Path(self.reg['runs'][method]);out=run/'jobs'/key
        entry=self.state['jobs'].get(key)
        if entry:
            if entry['guard']['status']=='completed':
                result=read(out/'result.json')
                if digest(out/'result.json')!=entry['result_sha256']:raise ValueError('Completed result changed')
                return result
            # A failed/interrupted attempt needs a separately registered continuation.
            return None
        if self.state.get('memory_abort') or time.time()+30>=self.state['deadline_unix']:
            self.state.setdefault('not_run',{})[key]='memory_abort' if self.state.get('memory_abort') else 'budget_exhausted'
            self.persist();return None
        if out.exists():raise FileExistsError(out)
        description={'kind':kind,'method':method,'run':str(run),'output':str(out),
            'model':identity['model'],'adapter':adapter,'policy_version':version,'variant':'original',
            'seed':self.c['seed'],'round':options.get('round',1),**options}
        from utils.course_worker import artifact_fingerprint
        for field in ('adapter','sft_adapter'):
            if description.get(field):description[field+'_fingerprint']=artifact_fingerprint(description[field])
        description['input_hashes']={field:digest(description[field]) for field in
            ('panel','rollout','trainer_state','value_head') if description.get(field)}
        write(out/'job.json',description)
        python=identity['inference_python'] if kind in ('evaluation','rollout') else identity['training_python']
        self.state['jobs'][key]={'output':str(out),'guard':{'status':'running'},'method':method,'kind':kind}
        self.persist();print('START',key,flush=True)
        timeout=7200 if kind=='evaluation' else 3600
        guard=guard_job([python,str(ROOT/'utils/course_worker.py'),'--job',str(out/'job.json')],out,'0',self.c,
                        min(self.state['deadline_unix'],time.time()+timeout))
        self.state['jobs'][key]['guard']=guard
        if (out/'result.json').exists():self.state['jobs'][key]['result_sha256']=digest(out/'result.json')
        if guard['status'] in ('memory_limit_exceeded','failed_memory_monitor'):self.state['memory_abort']=key
        self.persist();print('END',key,guard['status'],guard['device_peak_sampled_bytes'],flush=True)
        return read(out/'result.json') if guard['status']=='completed' else None

    def nemo(self):
        identity=self.reg['identity']
        for arm in ('R0','A','B','C'):
            key='nemo_'+arm
            if key in self.state['jobs']:continue
            if self.state.get('memory_abort') or time.time()+120>=self.state['deadline_unix']:continue
            out=self.root/('nemo_'+arm+'_'+self.root.name)
            # round2_nemo_gpu owns its own guard; do not double-monitor/double-kill.
            command=[identity['nemo_python'],str(ROOT/'utils/round2_nemo_gpu.py'),
                '--arm',arm,'--output',str(out),'--model',identity['model'],
                '--vllm-python',identity['inference_python'],'--timeout-seconds',
                str(int(min(1800,self.state['deadline_unix']-time.time())))]
            print('START',key,flush=True)
            self.state['jobs'][key]={'output':str(out),'guard':{'status':'running'},'kind':'nemo'};self.persist()
            result=subprocess.run(command,cwd=ROOT)
            guard=read(out/'guard.json') if (out/'guard.json').exists() else {'status':'failed_before_guard','returncode':result.returncode}
            self.state['jobs'][key]['guard']=guard
            if (out/'result.json').exists():self.state['jobs'][key]['result_sha256']=digest(out/'result.json')
            if guard['status'] in ('memory_limit_exceeded','failed_memory_monitor'):self.state['memory_abort']=key
            self.persist();print('END',key,guard['status'],flush=True)

    def execute(self):
        self.nemo()
        sft=self.reg['identity']['adapter'];version='sft:r1:'+self.reg['design']['parent_sft_adapter_sha256'][:12]
        readiness=self.job('readiness','readiness_equation','rollout',sft,version,
            panel=str(Path(self.reg['runs']['readiness'])/'panels/train.jsonl'),
            candidates=8,cap=16384,output_contract='closed_trace_v1')
        # Exercise training before spending most of the session on baseline decoding.
        for method in ('dpo','ppo','grpo'):
            current=None
            for k in range(1,2 if method=='dpo' else 3):
                parent=current['adapter_path'] if current else sft
                policy_version=current['policy_version'] if current else version
                rollout=self.job(method,f'{method}_r{k}_rollout','rollout',parent,policy_version,
                    panel=str(Path(self.reg['runs'][method])/'panels/train.jsonl'),round=k,
                    seed=self.c['seed']+k,candidates=self.c[method]['candidates'],
                    cap=16384,output_contract='closed_trace_v1')
                if rollout is None:break
                outcome=self.job(method,f'{method}_r{k}_train','training',parent,policy_version,
                    round=k,seed=self.c['seed']+k,sft_adapter=sft,
                    rollout=str(Path(self.state['jobs'][f'{method}_r{k}_rollout']['output'])/'rows.jsonl'),
                    trainer_state=current['trainer_state'] if current else None,
                    value_head=current['value_head'] if current else None)
                if not outcome or outcome['status']!='completed':break
                current=outcome['checkpoint'];self.state['checkpoints'][method]=current;self.persist()
                self.job(method,f'{method}_r{k}_reload','reload',current['adapter_path'],current['policy_version'],
                         value_head=current.get('value_head'))
        policies=[('base',None,'base'),('sft',sft,version)]+[
            (m,cp['adapter_path'],cp['policy_version']) for m,cp in self.state['checkpoints'].items()]
        for contract in (None,'closed_trace_v1'):
            for name,adapter,policy_version in policies:
                self.job('baseline',f'eval_{name}_{contract or "original"}','evaluation',adapter,policy_version,
                    panel=str(Path(self.reg['runs']['baseline'])/'panels/evaluation.jsonl'),output_contract=contract)
        expected=[f'eval_{m}_{v}' for m in ('base','sft','dpo','ppo','grpo') for v in ('original','closed_trace_v1')]
        eval_complete=all(self.state['jobs'].get(k,{}).get('guard',{}).get('status')=='completed' for k in expected)
        self.state['status']='phase_c_complete' if self.acceptance()['all_method_updates_and_memory'] and eval_complete else 'phase_c_incomplete'
        self.state['execution_ended_at_unix']=time.time();self.persist()
        self.report()

    def acceptance(self):
        details={}
        for method in ('dpo','ppo','grpo'):
            jobs=[r for r in self.state['jobs'].values() if r.get('method')==method and r['kind']=='training']
            steps=0
            for job in jobs:
                p=Path(job['output'])/'result.json'
                if p.exists():steps+=read(p).get('optimizer_steps',0)
            related=[r for r in self.state['jobs'].values() if r.get('method')==method]
            memory=bool(related) and all(r['guard'].get('status')=='completed' and
                r['guard'].get('memory_check')=='sampled_pass' for r in related)
            details[method]={'optimizer_steps':steps,'sampled_memory_pass':memory,
                             'has_saved_checkpoint':method in self.state['checkpoints']}
        return {'methods':details,'all_method_updates_and_memory':all(v['optimizer_steps']>0 and
            v['sampled_memory_pass'] and v['has_saved_checkpoint'] for v in details.values())}

    def report(self):
        evaluations={};comparisons={};jobs={}
        for key,entry in self.state['jobs'].items():
            out=Path(entry['output']);r=read(out/'result.json') if (out/'result.json').exists() else None
            jobs[key]={'guard':entry['guard'],'result':r,'output':str(out)}
            if key.startswith('eval_') and entry['guard']['status']=='completed' and r and r['status']=='completed':
                evaluations[key]=read_rows(out/'rows.jsonl')
        for contract in ('original','closed_trace_v1'):
            for name in ('sft','dpo','ppo','grpo'):
                parent='base' if name=='sft' else 'sft';a=f'eval_{parent}_{contract}';b=f'eval_{name}_{contract}'
                if a in evaluations and b in evaluations:
                    comparisons[b]=paired_comparison(evaluations[a],evaluations[b])
        expected=[f'eval_{m}_{v}' for m in ('base','sft','dpo','ppo','grpo') for v in ('original','closed_trace_v1')]
        readiness=None
        if 'readiness_equation' in self.state['jobs']:
            entry=self.state['jobs']['readiness_equation'];path=Path(entry['output'])/'rows.jsonl'
            if entry['guard']['status']=='completed' and path.exists():
                from utils.course_worker import notebook_api
                import statistics
                api=notebook_api('grpo');groups={}
                for row in read_rows(path):
                    component=api['equation_reward_components'](row,row['proof_audit'],'R1')
                    groups.setdefault(row['sample_id'],[]).append(component)
                summaries=[]
                for sample_id,components in groups.items():
                    rewards=[v['combined'] for v in components];good=sum(v['final'] for v in components)
                    std=statistics.pstdev(rewards)
                    summaries.append({'sample_id':sample_id,'components':components,
                        'group_type':'all_wrong' if not good else 'all_correct' if good==len(components) else 'mixed',
                        'variance':std**2,'std_population':std,
                        'advantages':[(x-statistics.mean(rewards))/(std+1e-6) if std>1e-6 else 0. for x in rewards]})
                readiness={'groups':summaries,'effective_group_ratio':sum(g['variance']>1e-12 for g in summaries)/len(summaries)}
                write(self.root/'readiness_reward_audit.json',readiness)
        primary={}
        for name,task in (('dpo','gravity'),('ppo','equation_numeric'),('grpo','equation_numeric')):
            key=f'eval_{name}_original'
            if key in comparisons:primary[key]=comparisons[key][task]['accuracy_mcnemar_pvalue']
        payload={'status':self.state['status'],'jobs':jobs,'acceptance':self.acceptance(),'comparisons':comparisons,
            'missing_evaluations':[k for k in expected if k not in evaluations],
            'readiness':readiness,'primary_accuracy_holm_pvalues':holm(primary) if len(primary)==3 else None,
            'formal_training_started':False,'continuous_memory_bound_verified':False,
            'reward':'R1=official_correct+.20*verified_process+.05*format; no length reward',
            'holdout_used_for_tuning':False,'source_commit':self.reg['source_commit']}
        write(self.root/'report.json',payload)
        print(json.dumps({'suite':str(self.root),'status':payload['status'],'acceptance':payload['acceptance'],
                          'missing_evaluations':payload['missing_evaluations']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','run','report'))
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--design',type=Path,default=ROOT/'configs/round2_design.json')
    parser.add_argument('--model',type=Path);parser.add_argument('--adapter',type=Path)
    for name in ('inference-python','training-python','nemo-python'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--source-commit');parser.add_argument('--budget-seconds',type=int,default=28800)
    args=parser.parse_args()
    if args.action=='register':register(args)
    elif args.action=='report':Suite(args).report()
    else:
        with (args.suite/'supervisor.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);Suite(args).execute()


if __name__=='__main__':main()
