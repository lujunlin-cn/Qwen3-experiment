"""Prepare, run/resume, and report the bounded eight-hour Qwen3 experiment.

Defaults are CPU-only. `run` requires explicit model and interpreter paths.
Every inference/training job is an isolated process group with device monitoring.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.course_protocol import (TASKS, advance_stop, check_protocol, digest, metrics, prepare,
                                   read, read_rows, validation_score, write)


def gpu_memory(gpu):
    out=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,uuid,name',
                                 '--format=csv,noheader,nounits'],text=True,timeout=10)
    fields=out.strip().split(',')
    return {'used_bytes':int(fields[0].strip())*2**20,'uuid':fields[1].strip(),'name':','.join(fields[2:]).strip()}


def idle_device(gpu):
    info=gpu_memory(gpu)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                    '--format=csv,noheader,nounits'],text=True,timeout=10)
    active=[line for line in processes.splitlines() if info['uuid'] in line]
    if active: raise RuntimeError(f"GPU already has compute processes: {active}; use an idle device")
    return info


def kill_group(process):
    if process.poll() is not None: return
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def guard_job(command, output, gpu, c, deadline):
    """Guard only our process group. Sampling is not a proof of a continuous bound."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    before=idle_device(gpu)
    if before['used_bytes'] >= c['device_limit_bytes']: raise RuntimeError('Device baseline already above limit')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLBACKEND='Agg')
    started=time.time();reason=None;peak=before['used_bytes'];samples=0;process=None
    try:
        with (output/'worker.log').open('w') as log, (output/'device_memory.jsonl').open('w') as telemetry:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                try:
                    reading=gpu_memory(gpu);now=time.time();peak=max(peak,reading['used_bytes']);samples+=1
                    telemetry.write(json.dumps({'elapsed_seconds':now-started,'used_bytes':reading['used_bytes']})+'\n');telemetry.flush()
                except Exception as exc:
                    reason='failed_memory_monitor';write(output/'monitor_error.json',{'error':repr(exc)});break
                if peak >= c['device_limit_bytes']: reason='memory_limit_exceeded';break
                if now >= deadline: reason='budget_exhausted';break
                if process.poll() is not None: break
                time.sleep(c['poll_seconds'])
    except BaseException:
        reason='interrupted'
        raise
    finally:
        if process is not None:
            kill_group(process)
            # Descendant vLLM workers can outlive a failed parent. Kill this group only.
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
        report={'status':reason or ('completed' if process.returncode==0 else 'worker_failed'),
            'returncode':process.returncode if process else None,'wall_seconds':time.time()-started,
            'started_at_unix':started,'device_peak_sampled_bytes':peak,'memory_limit_bytes':c['device_limit_bytes'],
            'memory_check':'sampled_pass' if peak<c['device_limit_bytes'] and samples else 'failed',
            'samples':samples,'poll_seconds':c['poll_seconds'],'device':before,
            'scope':'entire selected GPU, including CUDA context and reserved KV; sub-poll transient peaks unverified'}
        write(output/'guard.json',report)
    return report


class Experiment:
    def __init__(self,args):
        self.args=args;self.run=args.run_dir.resolve();p=check_protocol(self.run,args.config);self.c=p['config']
        self.state_path=self.run/'state.json'
        self.state=read(self.state_path) if self.state_path.exists() else {'jobs':{},'methods':{}}
        if 'deadline_unix' not in self.state:
            self.state.update(started_at_unix=time.time(),deadline_unix=time.time()+self.c['total_seconds'])
        identity={'model':str(args.model.resolve()),'inference_python':str(args.inference_python.resolve()),
                  'training_python':str(args.training_python.resolve()),'gpu':args.gpu}
        if self.state.get('execution') and self.state['execution']!=identity: raise ValueError('Execution paths/device changed; use a new run')
        self.state['execution']=identity;self.persist()
        model_files={str(p.relative_to(args.model)):digest(p) for pattern in
            ('*.safetensors','*.index.json','config.json','tokenizer*.json','chat_template*','generation_config.json','vocab.json','merges.txt')
            for p in args.model.glob(pattern) if p.is_file()}
        if not any(name.endswith('.safetensors') for name in model_files): raise ValueError('Local model weights missing')
        manifest=self.run/'base_model_fingerprint.json'
        if manifest.exists() and read(manifest)!=model_files: raise ValueError('Base weights/tokenizer changed')
        write(manifest,model_files)
        self.stage_deadline=0

    def persist(self): write(self.state_path,self.state)

    def stage(self,name):
        self.stage_name=name
        # Preserve each stage's budget across restart; never reset its timer on resume.
        stages=self.state.setdefault('stages',{})
        if name not in stages:
            stages[name]={'started_at_unix':time.time(), 'deadline_unix':min(self.state['deadline_unix'],
                time.time()+self.c['budgets_seconds'][name])}
            self.persist()
        self.stage_deadline=stages[name]['deadline_unix']
        if name=='final': self.stage_deadline=self.state['deadline_unix']  # use time saved in earlier stages

    def job(self,key,kind,**options):
        record=self.state['jobs'].get(key)
        if record and record['guard']['status']=='running':
            path=Path(record['output'])/'guard.json'
            record['guard']=read(path) if path.exists() else {'status':'interrupted'}
            result_path=Path(record['output'])/'result.json'
            if result_path.exists(): record['result_sha256']=digest(result_path)
        if record and record['guard']['status']=='completed':
            result=read(Path(record['output'])/'result.json')
            if record.get('result_sha256') != digest(Path(record['output'])/'result.json'):
                raise ValueError(f'Result changed: {key}')
            return result
        if self.state.get('guard_abort'):
            self.state.setdefault('not_run',{})[key]='experiment_aborted_by_memory_guard';self.persist();return None
        if time.time()+15>=self.stage_deadline:
            self.state.setdefault('not_run',{})[key]='budget_exhausted';self.persist();return None
        if record and record['guard']['status']!='interrupted':
            # Failed attempts are explicit; a resume must not quietly hide OOM/limit violations.
            return None
        attempt=(record.get('attempt',0)+1) if record else 1
        if record:
            self.state.setdefault('attempt_history',[]).append({'job':key,**record})
            self.persist()
        output=self.run/'jobs'/f'{key}_attempt{attempt}'
        while output.exists():
            attempt+=1;output=self.run/'jobs'/f'{key}_attempt{attempt}'
        job={'kind':kind,'run':str(self.run),'output':str(output),'model':str(self.args.model.resolve()),
             'seed':self.c['seed']+options.get('round',0),'policy_version':'base','variant':'original',**options}
        for key_path in ('adapter','sft_adapter'):
            if job.get(key_path):
                from utils.course_worker import artifact_fingerprint
                job[key_path+'_fingerprint']=artifact_fingerprint(job[key_path])
                if not job[key_path+'_fingerprint']: raise ValueError('Missing adapter artifacts')
        job['input_hashes']={k:digest(job[k]) for k in ('panel','rollout','trainer_state','value_head') if job.get(k)}
        write(output/'job.json',job)
        python=self.args.inference_python if kind in ('evaluation','rollout') else self.args.training_python
        print(f"START {key} ({max(0,int(self.stage_deadline-time.time()))} seconds left in {self.stage_name})",flush=True)
        self.state['jobs'][key]={'output':str(output),'guard':{'status':'running'},'stage':self.stage_name,'attempt':attempt}
        self.persist()
        try:
            guard=guard_job([str(python),str(ROOT/'utils/course_worker.py'),'--job',str(output/'job.json')],
                            output,self.args.gpu,self.c,self.stage_deadline)
        except BaseException:
            path=output/'guard.json'
            self.state['jobs'][key]['guard']=read(path) if path.exists() else {'status':'interrupted'}
            self.persist()
            raise
        record={'output':str(output),'guard':guard,'stage':self.stage_name,'attempt':attempt}
        if (output/'result.json').exists(): record['result_sha256']=digest(output/'result.json')
        self.state['jobs'][key]=record;self.persist()
        if guard['status'] in ('memory_limit_exceeded','failed_memory_monitor'):
            self.state['guard_abort']={'job':key,'status':guard['status']};self.persist()
        print(f"END {key}: {guard['status']}, peak {guard['device_peak_sampled_bytes']/2**30:.2f} GiB",flush=True)
        if guard['status']=='completed': return read(output/'result.json')
        return None

    def evaluation(self,key,adapter=None,variant='original',panel='tune',cap=None,version='base'):
        return self.job(key,'evaluation',panel=str(self.run/f'panels/{panel}.jsonl'),adapter=adapter,
                        variant=variant,cap=cap,policy_version=version)

    def train_method(self,method,sft=None):
        self.stage(method);settings=self.c[method];entry=self.state['methods'].setdefault(method,{'history':[]})
        if entry.get('stop_reason'): return entry.get('best_checkpoint')
        current=entry.get('current_checkpoint')
        parent=current or sft
        # Every RL method independently starts from the same selected SFT, never from another RL run.
        base_eval=self.state['methods'].get('sft',{}).get('best_metrics') if method!='sft' else None
        for k in range(len(entry['history'])+1,settings['max_rounds']+1):
            if time.time()+15 >= self.stage_deadline: entry['stop_reason']='budget_exhausted';break
            options={'method':method,'round':k,'adapter':parent['adapter_path'] if parent else None,
                'policy_version':parent['policy_version'] if parent else 'base',
                'trainer_state':current.get('trainer_state') if current else None,
                'value_head':current.get('value_head') if current else None,
                'sft_adapter':sft['adapter_path'] if sft else None}
            if method!='sft':
                cap_index=entry.get('cap_index',0);cap=settings['rollout_caps'][cap_index]
                rollout=self.job(f'{method}_r{k}_rollout','rollout',method=method,round=k,
                    panel=str(self.run/f'panels/{method}_{k}.jsonl'),adapter=options['adapter'],
                    policy_version=options['policy_version'],variant=self.c['training_prompt'],
                    cap=cap,candidates=settings['candidates'])
                if rollout is None: entry['stop_reason']='rollout_incomplete';break
                options['rollout']=str(Path(self.state['jobs'][f'{method}_r{k}_rollout']['output'])/'rows.jsonl')
                if method=='ppo' and (sum(m['correct'] for m in rollout['metrics'].values())==0
                    and any(m['truncated'] for m in rollout['metrics'].values())):
                    entry['cap_index']=min(cap_index+1,len(settings['rollout_caps'])-1)
            result=self.job(f'{method}_r{k}_train','training',**options)
            if result is None: entry['stop_reason']='training_incomplete';break
            if result['status'] in ('no_preference_signal','no_group_signal'):
                entry['cap_index']=min(entry.get('cap_index',0)+1,len(settings['rollout_caps'])-1)
                entry['history'].append({'round':k,'status':result['status'],'validation_score':None})
                self.persist();continue
            if result['status']!='completed': entry['stop_reason']=result['status'];break
            current=result['checkpoint'];entry['current_checkpoint']=current;self.persist()
            evaluation=self.evaluation(f'{method}_r{k}_validation',adapter=current['adapter_path'],version=current['policy_version'])
            if evaluation is None: entry['stop_reason']='validation_incomplete';break
            m=evaluation['metrics'];score=validation_score(m,settings['tasks'],self.c['eval_caps'])
            old_scores=[r['validation_score'] for r in entry['history'] if r['validation_score'] is not None]
            improved,plateau=advance_stop(old_scores,score,self.c['stop'])
            entry['history'].append({'round':k,'status':'completed','validation_score':score,'metrics':m,
                                     'optimizer_steps':result['optimizer_steps']})
            if improved:
                entry.update(best_checkpoint=current,best_metrics=m,best_round=k,
                    time_until_best_seconds=time.time()-self.state['stages'][method]['started_at_unix'])
            parent=current
            # Require at least some true optimization, plus validation plateau on multiple checkpoints.
            if plateau:
                entry['stop_reason']='validation_plateau';entry['stop_at_unix']=time.time();break
            self.persist()
        if not entry.get('stop_reason'): entry['stop_reason']='max_rounds'
        entry['time_until_stop_seconds']=time.time()-self.state['stages'][method]['started_at_unix']
        entry['coarse_convergence_observed']=entry['stop_reason']=='validation_plateau'
        self.persist();return entry.get('best_checkpoint')

    def execute(self):
        self.stage('baseline')
        # Reference first, so partial grid work does not lose the primary baseline.
        pilot=self.evaluation('base_reference')
        if pilot:
            throughput=pilot['output_tokens_per_second']
            final_tokens=sum(pilot['metrics'][t]['mean_output_tokens'] for t in TASKS)*self.c['panels']['final_per_task']*10
            estimate={'observed_reference_tokens_per_second':throughput,
                'estimated_final_output_tokens_for_5_models_2_prompts':final_tokens,
                'estimated_final_generation_seconds':final_tokens/max(throughput,1e-8),
                'warning':'Planning extrapolation; training/prior prompts can change lengths. No sample size or cap silently reduced.'}
            write(self.run/'pilot_budget_estimate.json',estimate)
            print('PILOT_BUDGET',json.dumps(estimate),flush=True)
        for cap in self.c['baseline_caps']: self.evaluation(f'base_cap{cap}',cap=cap)
        self.stage('prompt');self.evaluation('prior_reference',variant='prior')
        sft=self.train_method('sft')
        if sft:
            for method in ('dpo','ppo','grpo'): self.train_method(method,sft)
        self.stage('final')
        # Final panel stays unused until checkpoints and stop decisions are frozen.
        policies=[('base',None,'base')]
        for method in ('sft','dpo','ppo','grpo'):
            cp=self.state['methods'].get(method,{}).get('best_checkpoint')
            if cp: policies.append((method,cp['adapter_path'],cp['policy_version']))
        # Original prompt first for all models; the prior route is an extra matched comparison.
        for variant in self.c['evaluation_variants']:
            for method,adapter,version in policies:
                self.evaluation(f'final_{method}_{variant}',adapter,variant,panel='final',version=version)
        self.state['finished_at_unix']=time.time();self.persist()
        report(self.run)


def report(run):
    run=Path(run);protocol=read(run/'protocol.json');c=protocol['config']
    state=read(run/'state.json') if (run/'state.json').exists() else {'jobs':{},'methods':{}}
    rows=[];timing={};phase_timing={};failures=[]
    for name,job in state['jobs'].items():
        guard=job['guard'];out=Path(job['output']);item={'job':name,'stage':job['stage'],**guard}
        r=read(out/'result.json') if (out/'result.json').exists() else {}
        item['worker_status']=r.get('status')
        timing[job['stage']]=timing.get(job['stage'],0)+guard.get('wall_seconds',0)
        description=read(out/'job.json') if (out/'job.json').exists() else {}
        phase_key=job['stage']+':'+description.get('kind','unknown')
        phase_timing[phase_key]=phase_timing.get(phase_key,0)+guard.get('wall_seconds',0)
        if guard['status']=='completed' and r.get('status')=='completed' and 'metrics' in r:
            rows.append({'job':name,'metrics':r['metrics'],'device_peak_bytes':guard['device_peak_sampled_bytes'],
                         'wall_seconds':guard['wall_seconds']})
        elif guard['status']!='completed' or r.get('status') not in ('completed','no_group_signal','no_preference_signal'):
            failures.append(item)
    for job in state.get('attempt_history',[]):
        guard=job['guard'];out=Path(job['output'])
        timing[job['stage']]=timing.get(job['stage'],0)+guard.get('wall_seconds',0)
        description=read(out/'job.json') if (out/'job.json').exists() else {}
        phase_key=job['stage']+':'+description.get('kind','unknown')
        phase_timing[phase_key]=phase_timing.get(phase_key,0)+guard.get('wall_seconds',0)
    # Paired changes only on exactly the same final IDs, prompt variant and token caps.
    comparisons=[]
    def final_rows(method,variant):
        job=state['jobs'].get(f'final_{method}_{variant}')
        if not job or job['guard']['status']!='completed': return None
        path=Path(job['output'])/'rows.jsonl'
        return read_rows(path) if path.exists() else None
    for variant in c['evaluation_variants']:
        for method,parent in [('sft','base'),('dpo','sft'),('ppo','sft'),('grpo','sft')]:
            before,after=final_rows(parent,variant),final_rows(method,variant)
            if before is None or after is None: continue
            def key(r): return (r['sample_id'],r['task_type'],r['max_tokens'],r['variant'])
            if {key(r) for r in before}!={key(r) for r in after}: raise ValueError('Unmatched final comparison')
            a,b=metrics(before),metrics(after)
            for task in TASKS:
                delta={field:b[task][field]-a[task][field] for field in ('accuracy','truncation_rate','mean_output_tokens')}
                comparisons.append({'method':method,'parent':parent,'task':task,'variant':variant,'delta':delta,
                    'criteria_observed':{'accuracy_increased':delta['accuracy']>0,'truncation_decreased':delta['truncation_rate']<0,
                                         'output_tokens_decreased':delta['mean_output_tokens']<0},
                    'paired_ids':len([r for r in after if r['task_type']==task])})
    original,prior=final_rows('base','original'),final_rows('base','prior')
    if original is not None and prior is not None:
        ids=lambda rows:{(r['sample_id'],r['task_type'],r['max_tokens']) for r in rows}
        if ids(original)!=ids(prior): raise ValueError('Unmatched prompt comparison')
        a,b=metrics(original),metrics(prior)
        for task in TASKS:
            comparisons.append({'method':'prompt_prior','parent':'base_original','task':task,
                'delta':{field:b[task][field]-a[task][field] for field in ('accuracy','truncation_rate','mean_output_tokens')},
                'paired_ids':b[task]['n']})
    methods={k:{x:v for x,v in val.items() if x not in ('best_checkpoint','current_checkpoint')} for k,val in state['methods'].items()}
    expected=[f'final_{m}_{v}' for v in c['evaluation_variants'] for m in ('base','sft','dpo','ppo','grpo')]
    missing=[name for name in expected if not any(row['job']==name for row in rows)]
    required_evals=['base_reference','prior_reference']+[f'base_cap{cap}' for cap in c['baseline_caps']]
    missing_screening=[name for name in required_evals if not any(row['job']==name for row in rows)]
    method_training_memory={}
    for method in ('sft','dpo','ppo','grpo'):
        phase={}
        kinds=('training',) if method=='sft' else ('rollout','training')
        for kind in kinds:
            suffix='_train' if kind=='training' else '_rollout'
            matches=[j for name,j in state['jobs'].items() if name.startswith(method+'_r') and name.endswith(suffix)]
            passed=bool(matches) and all(j['guard']['status']=='completed' and j['guard']['memory_check']=='sampled_pass' for j in matches)
            steps=0
            for j in matches:
                result_path=Path(j['output'])/'result.json'
                if result_path.exists(): steps+=read(result_path).get('optimizer_steps',0)
            exercised=kind=='rollout' or steps>0
            phase[kind]={'sampled_memory_pass':passed,'jobs':len(matches),
                         'workload_exercised':exercised,'optimizer_steps':steps,
                         'peak_sampled_bytes':max((j['guard'].get('device_peak_sampled_bytes',0) for j in matches),default=None)}
        method_training_memory[method]=phase
    phase_memory_pass=all(p['sampled_memory_pass'] and p['workload_exercised']
        for method in method_training_memory.values() for p in method.values())
    payload={'status':'complete' if not missing and not missing_screening and not failures else 'incomplete','evaluations':rows,
        'comparisons':comparisons,'stage_wall_seconds':timing,'phase_wall_seconds':phase_timing,'methods':methods,'failures':failures,
        'previous_attempts':state.get('attempt_history',[]),
        'missing_final_evaluations':missing,'missing_screening_evaluations':missing_screening,
        'all_training_phases_sampled_memory_pass':phase_memory_pass,
        'all_methods_reached_validation_plateau':all(state['methods'].get(m,{}).get('stop_reason')=='validation_plateau'
            for m in ('sft','dpo','ppo','grpo')),
        'not_run':state.get('not_run',{}),'phase_memory':method_training_memory,
        'continuous_memory_bound_verified':False,'memory_note':'Device sampled every 0.25s; transient overshoots between samples unverified.',
        'claims':'Acceptance outcomes reflect measurements. Budget exhaustion or max rounds do not establish convergence.',
        'holdout_used':False}
    write(run/'report.json',payload)
    lines=['# Qwen3 课程实验结果','',f"状态：{payload['status']}。显存阈值：8,000,000,000 bytes；设备峰值为采样值。",'',
           '| 对照 | 题型 | n | 正确率 | 截断率 | 平均输出 Token | 设备峰值 GiB | 秒 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for task,m in row['metrics'].items():
            lines.append(f"| {row['job']} | {task} | {m['n']} | {m['accuracy']:.1%} | {m['truncation_rate']:.1%} | {m['mean_output_tokens']:.2f} | {row['device_peak_bytes']/2**30:.2f} | {row['wall_seconds']:.1f} |")
    lines+=['','输出长度包含 Thinking 和所有生成 Token，包括截断样本；不含输入。小面板仅作课程筛查，Wilson 区间见 JSON。','',
            '停止条件与耗时（包括 rollout、train、验证、启动和保存；未达到平台期时不称为收敛）：','']
    for name,m in methods.items(): lines.append(f"- {name}: {m.get('stop_reason','not_finished')}, {m.get('time_until_stop_seconds',0):.1f} 秒")
    lines+=['',f'缺少最终对照：{missing}',f'缺少基线/先验对照：{missing_screening}',
            f'失败阶段数：{len(failures)}。原始日志与显存采样保存在 jobs/。','',
            'SFT 和 RL 的长度/截断变化均按实测报告；不预设只有 RL 才能改善这些指标。']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','report'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/course_8h.json')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--inference-python',type=Path)
    parser.add_argument('--training-python',type=Path)
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    if args.action=='prepare':
        p=prepare(args.config,args.run_dir);print(json.dumps({'status':'prepared_cpu_only','config':p['config_sha256'],'run':str(args.run_dir.resolve())}))
    elif args.action=='report':
        r=report(args.run_dir);print(json.dumps({'status':r['status'],'report':str(args.run_dir/'report.md')}))
    else:
        for path in (args.model,args.inference_python,args.training_python):
            if path is None or not path.exists(): parser.error('run requires existing --model, --inference-python, --training-python')
        args.run_dir.mkdir(parents=True,exist_ok=True)
        with (args.run_dir/'supervisor.lock').open('w') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: parser.error('A supervisor is already running in this directory')
            Experiment(args).execute()

if __name__=='__main__': main()
