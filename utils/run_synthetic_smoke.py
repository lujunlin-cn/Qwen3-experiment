"""CPU prepare/API check, then explicit NeMo -> local vLLM smoke tomorrow.

The Generator, Solver, Verifier and three-way audit remain in Notebook 01.
Preparation never calls a model or consumes GPU time. No existing run is replaced.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
DEFAULT_CONFIG=ROOT/'configs/nemo_smoke.json'
NOTEBOOK=ROOT/'01_Synthetic_Data_Engineering.ipynb'


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load_notebook(run_id, artifact_dir=None):
    """Load definitions/constants, excluding plots, report writes, pilot and GPU calls."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id): raise ValueError('Invalid Run ID')
    os.environ.update(SYNTHETIC_RUN_ID=run_id,TASKSET_VERSION='roman_gravity_equation_v1',RUN_SYNTHETIC='0',RUN_DATA_DESIGNER_PREVIEW='0',
                      RUN_DATA_DESIGNER_SMOKE='0',MPLBACKEND='Agg')
    os.chdir(ROOT)
    module=types.ModuleType('synthetic_smoke_notebook');sys.modules[module.__name__]=module
    ns=module.__dict__;notebook=read(NOTEBOOK)
    config=''.join(notebook['cells'][2]['source'])
    if artifact_dir is not None:
        ns['_smoke_artifact_dir']=Path(artifact_dir)
        config=config.replace('synthetic_dir = output_dir / "synthetic" / synthetic_run_id','synthetic_dir = _smoke_artifact_dir')
    exec(compile(config,str(NOTEBOOK)+':config','exec'),ns)
    for i,cell in enumerate(notebook['cells']):
        if i<=2 or cell['cell_type']!='code': continue
        tree=ast.parse(''.join(cell['source']))
        # These definitions end before the explicit pilot execution cell.
        if any(isinstance(x,ast.If) and any(isinstance(n,ast.Name) and n.id=='run_synthetic' for n in ast.walk(x.test)) for x in tree.body):
            break
        if i==15: continue  # Existing distribution figures/writes are for interactive execution only.
        safe=[x for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef,ast.Assign,ast.AnnAssign))]
        exec(compile(ast.Module(body=safe,type_ignores=[]),str(NOTEBOOK)+f':cell{i}','exec'),ns)
    ns['gravity_train_profile']=ns['_extract_gravity_profile'](ns['train_df'][ns['train_df'].task_type=='gravity'])
    ns['gravity_prior']['observed_train_profile']=ns['gravity_train_profile']['audit']
    assert ns['TASK_ORDER']==('numeral_system','gravity','equation_numeric')
    return ns


def validated_config(path=DEFAULT_CONFIG):
    c=read(path)
    if c['schema_version']!='nemo-qwen3-smoke/v1' or c['model_id']!='Qwen/Qwen3-0.6B': raise ValueError('Wrong model/schema')
    if (c['seeds_per_task'],c['expected_seed_rows'],c['expected_model_outputs'])!=(3,9,18): raise ValueError('Smoke must use 3 seeds/task, 18 outputs')
    if c['data_designer_version']!='0.10.0' or c['seed']!=42: raise ValueError('Use registered dependency/seed')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',c['synthetic_run_id']): raise ValueError('Invalid Run ID')
    if c['inference']!={'enable_thinking':False,'temperature':.3,'top_p':.9,'prompt_max_tokens':1024,'trace_max_tokens':2048}:
        raise ValueError('Inference contract differs from Notebook 01')
    if c['merge_into_sft']: raise ValueError('Smoke never merges into SFT')
    return c


def prepare(config=DEFAULT_CONFIG,run_id=None,artifact_dir=None):
    c=validated_config(config);run_id=run_id or c['synthetic_run_id']
    target=Path(artifact_dir) if artifact_dir else ROOT/f"data/processed/tasksets/{c['taskset']}/synthetic/{run_id}"
    if target.exists(): raise FileExistsError(f'Refuse to overwrite {target}; use a new SYNTHETIC_RUN_ID')
    target.mkdir(parents=True)
    write_new(target/'preparation_attempt.json',{'status':'started_cpu_only','synthetic_run_id':run_id})
    ns=load_notebook(run_id,target);rng=ns['np'].random.default_rng(c['seed'])
    candidates=[];seen_exact=set();seen_normalized=set();seen_structure=set();selected=[]
    for task in ns['TASK_ORDER']:
        count=0
        for k in range(c['max_seed_attempts_per_task']):
            generator=ns[{'numeral_system':'generate_numeral_sample','gravity':'generate_gravity_sample',
                          'equation_numeric':'generate_equation_numeric_sample'}[task]]
            sample=generator(rng)
            sample.update(candidate_id=f'{run_id}:{task}:{k}',candidate_index=k)
            ns['_solve_candidate'](sample)
            leakage=ns['audit_synthetic_leakage'](sample,ns['real_index'])
            screen=ns['screen_synthetic_sample'](sample,seen_exact,seen_normalized,seen_structure,leakage)
            candidates.append(sample)
            for key,seen in [('prompt_hash',seen_exact),('normalized_prompt_hash',seen_normalized),('structure_fingerprint',seen_structure)]: seen.add(screen[key])
            if screen['accepted']:
                selected.append({**sample,'seed_id':f'{task}-seed{count+1}'})
                count+=1
            if count==c['seeds_per_task']: break
        if count!=c['seeds_per_task']: raise ValueError(f'Insufficient verified synthetic seeds for {task}: {count}')
    statistics=ns['run_synthetic_pilot'](candidates)
    if statistics['accepted']<9: raise ValueError('Seed pilot has fewer than 9 Accepted')
    # Rebind selected objects to the audited, disk-backed deterministic seed records.
    audited={r['candidate_id']:r for r in rows(target/'synthetic_accepted.jsonl')}
    selected=[{**audited[r['candidate_id']],'seed_id':r['seed_id']} for r in selected]
    destination=target/'data_designer';destination.mkdir()
    ns['_write_jsonl'](destination/'private_seeds.jsonl',selected)
    ns['_write_jsonl'](destination/'public_seeds.jsonl',[{k:r[k] for k in ('seed_id','task_type','prompt')} for r in selected])
    write_new(destination/'smoke_config.json',{**c,'synthetic_run_id':run_id})
    frozen={str(p.relative_to(target)):digest(p) for p in sorted(target.rglob('*')) if p.is_file()}
    sources={str(p.relative_to(ROOT)):digest(p) for p in (NOTEBOOK,Path(__file__),ROOT/'configs/nemo_smoke_requirements.lock.txt')}
    manifest={'schema_version':'nemo-smoke-prepared/v1','status':'prepared_cpu_only','synthetic_run_id':run_id,
        'gpu_generation_run':False,'seed_counts':{t:3 for t in ns['TASK_ORDER']},'expected_outputs':18,
        'public_seed_fields':['seed_id','task_type','prompt'],'frozen_files':frozen,'sources':sources,
        'config_sha256':digest(config),'generator_unchanged':True,'solver_and_independent_verifier':True,
        'all_three_splits_audited':True,'merge_into_sft':False}
    write_new(destination/'prepared_manifest.json',manifest)
    return target


def verify_prepared_run(run):
    run=Path(run);manifest=read(run/'data_designer/prepared_manifest.json')
    for rel,sha in manifest['frozen_files'].items():
        if digest(run/rel)!=sha: raise ValueError(f'Frozen preparation file changed: {rel}')
    for rel,sha in manifest['sources'].items():
        if digest(ROOT/rel)!=sha: raise ValueError(f'Source changed since prepare: {rel}; use a new Run ID')
    accepted=read(run/'accepted_manifest.json')
    if accepted['status']!='passed' or accepted['accepted_rows']<9: raise ValueError('Seed Accepted manifest invalid')
    public=rows(run/'data_designer/public_seeds.jsonl');private=rows(run/'data_designer/private_seeds.jsonl')
    if len(public)!=9 or len(private)!=9 or len({r['seed_id'] for r in public})!=9: raise ValueError('Wrong seed cardinality')
    for a,b in zip(public,private):
        if set(a)!={'seed_id','task_type','prompt'} or any(a[k]!=b[k] for k in a): raise ValueError('Seed public/private mismatch')
    return manifest


def api_check(run):
    verify_prepared_run(run)
    c=read(Path(run)/'data_designer/smoke_config.json')
    if importlib.metadata.version('data-designer')!=c['data_designer_version']: raise ValueError('Wrong Data Designer version')
    ns=load_notebook(c['synthetic_run_id'],run)
    from data_designer.interface import DataDesigner
    builder,provider=ns['build_data_designer_smoke'](Path(run)/'data_designer/public_seeds.jsonl',c['endpoint'])
    designer=DataDesigner(artifact_path=Path(run)/'data_designer/api_validation',model_providers=[provider])
    ns['configure_smoke_runtime'](designer)
    designer.validate(builder)  # Real installed SDK and compiler, no model request.
    built=builder.build().model_dump(mode='json')
    result={'status':'passed_cpu_api_check','gpu_loaded':False,'model_requests':0,
            'data_designer_version':importlib.metadata.version('data-designer'),'columns':['prompt_variant','trace_variant'],
            'provider_endpoint':c['endpoint'],'seed_rows':9,'expected_outputs':18}
    write_new(Path(run)/'data_designer/api_check.json',result)
    write_new(Path(run)/'data_designer/compiled_config.json',built)
    return result


def local_endpoint_preflight(endpoint,model):
    url=urlparse(endpoint)
    if url.hostname not in {'127.0.0.1','localhost','::1'} or url.scheme!='http' or url.username or url.password:
        raise ValueError('Only unauthenticated loopback HTTP vLLM is supported')
    with urllib.request.urlopen(endpoint.rstrip('/')+'/models',timeout=10) as response: info=json.load(response)
    models={m['id'] for m in info.get('data',[])}
    if model not in models: raise ValueError(f'Expected local model {model}, got {models}')
    return {'model':model,'endpoint':endpoint,'status':'ready'}


@contextlib.contextmanager
def capture_local_requests(endpoint,destination):
    """Local transparent JSON proxy: retain finish reasons and actual token usage.

    No credentials are forwarded or logged. Retries are counted, not conflated
    with the 18 requested column outputs. Only non-streaming local HTTP is used.
    """
    parsed=urlparse(endpoint)
    if parsed.hostname not in {'127.0.0.1','localhost','::1'} or parsed.scheme!='http':
        raise ValueError('Loopback HTTP only')
    destination=Path(destination);lock=threading.Lock()
    with destination.open('x') as telemetry:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                started=time.perf_counter();entry={}
                try:
                    payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if payload.get('stream'): raise ValueError('Streaming not supported in smoke capture')
                    text=payload['messages'][-1]['content']
                    if isinstance(text,list): text=''.join(p.get('text','') for p in text)
                    seed_id=text.split('SEED_ID=',1)[1].splitlines()[0]
                    kind='prompt_variant' if 'Produce a safe wording variant.' in text else 'trace_variant'
                    entry.update(seed_id=seed_id,variant_kind=kind,model=payload['model'])
                    request=urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
                        data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=180) as response:
                        raw=response.read();status=response.status
                    body=json.loads(raw);choice=body.get('choices',[{}])[0]
                    entry.update(status='completed',finish_reason=choice.get('finish_reason'),
                        usage=body.get('usage',{}),response=body,
                        request_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
                except Exception as error:
                    status=502;entry.update(status='failed',error=repr(error))
                    raw=json.dumps({'error':{'message':repr(error),'type':'smoke_proxy_error'}}).encode()
                entry['latency_seconds']=time.perf_counter()-started
                with lock: telemetry.write(json.dumps(entry,ensure_ascii=False)+'\n');telemetry.flush()
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try: yield f'http://127.0.0.1:{server.server_port}/v1'
        finally: server.shutdown();server.server_close();thread.join(timeout=5)


def attach_transport(records,request_path):
    requests=rows(request_path);by_key={}
    for r in requests: by_key.setdefault((r.get('seed_id'),r.get('variant_kind')),[]).append(r)
    for record in records:
        for kind in ('prompt_variant','trace_variant'):
            matches=by_key.get((record.get('seed_id'),kind),[])
            completed=[r for r in matches if r['status']=='completed']
            content=record.get(kind)
            corresponding=[r for r in completed if r['response'].get('choices',[{}])[0].get('message',{}).get('content','').strip()==str(content).strip()]
            last=corresponding[-1] if corresponding else {}
            record[kind+'__transport']={'verified_response_match':bool(last),'request_count':len(matches),
                'finish_reason':last.get('finish_reason'),'usage':last.get('usage'),
                'latency_seconds':last.get('latency_seconds')}
    return {'http_requests':len(requests),'failed_requests':sum(r['status']!='completed' for r in requests),
            'completion_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in requests)}


def server_command(c,python,model):
    if not Path(python).exists() or not (Path(model)/'config.json').exists(): raise ValueError('Existing vLLM Python and local model required')
    cfg=read(Path(model)/'config.json')
    if (cfg.get('model_type'),cfg.get('hidden_size'),cfg.get('num_hidden_layers'))!=('qwen3',1024,28): raise ValueError('Wrong Qwen3-0.6B model path')
    server=c['server'];url=urlparse(c['endpoint'])
    return [str(python),'-m','vllm.entrypoints.openai.api_server','--model',str(model),
        '--served-model-name',c['model_id'],'--host','127.0.0.1','--port',str(url.port or 80),
        '--dtype','bfloat16','--max-model-len',str(server['max_model_len']),
        '--max-num-seqs',str(server['max_num_seqs']),'--max-num-batched-tokens',str(server['max_num_batched_tokens']),
        '--kv-cache-memory-bytes',str(int(server['kv_cache_gib']*2**30)),'--enforce-eager',
        '--enable-chunked-prefill','--generation-config','vllm']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','api-check','gpu','server-command'))
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--run-id');parser.add_argument('--model',type=Path);parser.add_argument('--vllm-python',type=Path)
    args=parser.parse_args();c=validated_config(args.config);run_id=args.run_id or c['synthetic_run_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id): parser.error('Invalid Run ID')
    run=ROOT/f"data/processed/tasksets/{c['taskset']}/synthetic/{run_id}"
    if args.action=='prepare':
        print(prepare(args.config,run_id))
    elif args.action=='api-check': print(json.dumps(api_check(run),ensure_ascii=False))
    elif args.action=='server-command':
        if args.model is None or args.vllm_python is None: parser.error('Pass --model and --vllm-python')
        print(shlex.join(server_command(c,args.vllm_python,args.model)))
    else:
        verify_prepared_run(run)
        ns=load_notebook(run_id,run)
        os.environ['LOCAL_VLLM_ENDPOINT']=c['endpoint']
        result=ns['run_data_designer_smoke']();print(json.dumps(result,ensure_ascii=False))
        if result['status']!='passed': raise SystemExit(2)

if __name__=='__main__': main()
