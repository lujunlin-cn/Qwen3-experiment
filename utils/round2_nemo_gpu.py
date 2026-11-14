"""Real NeMo R0/A/B/C requests, with an owned local vLLM and a device guard."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.course_protocol import digest, read, read_rows, write, write_rows
from utils.course_run import guard_job
from utils.run_synthetic_smoke import (capture_local_requests,attach_transport,load_notebook,
                                      local_endpoint_preflight,server_command,validated_config)


def worker(args):
    from data_designer.interface import DataDesigner
    output=args.output.resolve()
    ns=load_notebook(output.name,output/'loader')
    seed_path=ROOT/'data/processed/tasksets/roman_gravity_equation_v1/synthetic/nemo_qwen3_06b_gpu_smoke_20261109_r1/data_designer/private_seeds.jsonl'
    seeds=read_rows(seed_path)
    config=validated_config()
    endpoint=config['endpoint']
    public=[{k:s[k] for k in ('seed_id','task_type','prompt')} for s in seeds]
    if args.arm=='C':
        for row,seed in zip(public,seeds):
            row['solver_context']=json.dumps(ns['smoke_solver_context'](seed),ensure_ascii=False)
    write_rows(output/'private_seeds.jsonl',seeds)
    write(output/'registered_config.json',{'arm':args.arm,'model_revision':read(args.model/'download_manifest.json')['revision'],
        'model_fingerprint':{p.name:digest(p) for p in args.model.glob('*.safetensors')},
        'seed_sha256':digest(seed_path),'inference':config['inference'],'server':config['server'],
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in
                         (ROOT/'01_Synthetic_Data_Engineering.ipynb',ROOT/'utils/round2_nemo_gpu.py',ROOT/'utils/run_synthetic_smoke.py')},
        'merge_into_sft':False,'registered_at_unix':time.time()})
    # Only the guarded process group owns these server/engine descendants.
    with (output/'server.log').open('x') as log:
        server=subprocess.Popen(server_command(config,args.vllm_python.absolute(),args.model.resolve()),
                                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            until=time.monotonic()+600
            while True:
                if server.poll() is not None:raise RuntimeError('vLLM exited; inspect server.log')
                try:
                    local_endpoint_preflight(endpoint,config['model_id']);break
                except (OSError,ValueError):
                    if time.monotonic()>until:raise TimeoutError('vLLM readiness timed out')
                    time.sleep(1)
            groups=[(t,[r for r in public if r['task_type']==t]) for t in ns['SMOKE_TASKS']] if args.arm=='B' else [(None,public)]
            records=[];errors=[];start=time.perf_counter()
            with capture_local_requests(endpoint,output/'http_requests.jsonl') as proxy:
                for task,rows in groups:
                    shard=output/(task or 'all');shard.mkdir()
                    write_rows(shard/'public_seeds.jsonl',rows)
                    builder,provider=ns['build_round2_data_designer'](shard/'public_seeds.jsonl',proxy,args.arm,task)
                    write(shard/'compiled_config.json',builder.build().model_dump(mode='json'))
                    designer=DataDesigner(artifact_path=shard/'sdk',model_providers=[provider])
                    ns['configure_smoke_runtime'](designer);designer.validate(builder)
                    try:
                        generated=designer.preview(config_builder=builder,num_records=len(rows))
                        part=generated.dataset.to_dict(orient='records');records.extend(part)
                        write_rows(shard/'raw_sdk_outputs.jsonl',part)
                    except Exception as error:
                        errors.append({'task':task,'error':repr(error)})
                        write(shard/'sdk_error.json',errors[-1])
            transport=attach_transport(records,output/'http_requests.jsonl')
            write_rows(output/'raw_model_outputs.jsonl',records)
            summary=ns['screen_data_designer_records'](records,seeds,output/'screening',require_transport=True)
            result={'status':'completed' if not errors and transport['failed_requests']==0 and summary['transport_complete'] else 'failed_transport',
                'quality_status':summary['status'],'summary':summary,'transport':transport,'errors':errors,
                'generation_and_screening_seconds':time.perf_counter()-start,'merge_into_sft':False,
                'gpu_generation_real':True,'arm':args.arm}
            write(output/'result.json',result)
            return 0 if result['status']=='completed' else 1
        finally:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('R0','A','B','C'),required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--vllm-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker(args))
    if args.output.exists():raise FileExistsError('Use a new NeMo attempt directory')
    config=validated_config()
    try:local_endpoint_preflight(config['endpoint'],config['model_id'])
    except OSError:pass
    else:raise RuntimeError('Dedicated endpoint is already occupied')
    guard=guard_job([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
                    args.output,'0',{'device_limit_bytes':8_000_000_000,'poll_seconds':.25},
                    time.time()+args.timeout_seconds)
    print(json.dumps(guard),flush=True)
    raise SystemExit(0 if guard['status']=='completed' else 1)


if __name__=='__main__':main()
