"""Optional controlled precision replay; never edits the active course run."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from utils.course_protocol import read,write,digest,prepare,check_protocol
from utils.course_worker import artifact_fingerprint
from utils.course_run import guard_job

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('action',choices=('prepare','run'))
args=parser.parse_args()
source=ROOT/'data/processed/course_runs/course8h_20261109_r2_cuda_graph'
run=ROOT/'data/processed/course_runs/sft_precision_replay_20261109_r1'
if args.action=='prepare':
 if run.exists():raise FileExistsError('Use a fresh replay ID')
 original=check_protocol(source)
 config=json.loads(json.dumps(original['config']));config['sft']['mode']='bf16_lora'
 config_path=run/'config.json';write(config_path,config)
 p=prepare(config_path,run)
 for name in ('sft','tune'):
  assert digest(run/f'panels/{name}.jsonl')==digest(source/f'panels/{name}.jsonl')
 cp=read(source/'jobs/sft_r1_train_attempt1/checkpoint.json')
 assert read(source/'jobs/sft_r1_train_attempt1/guard.json')['status']=='completed'
 write(run/'replay_manifest.json',{'status':'prepared_cpu_only','purpose':'Continue same NF4 round1 checkpoint and Adam state for rounds2,3 using BF16 base; compare to original NF4 round3. Objectives, samples, tokenization, shuffle seeds and step counts unchanged. No changes to active main run; no final panel selection.',
  'source_protocol_sha256':digest(source/'protocol.json'),'initial_checkpoint':cp,
  'initial_fingerprint':artifact_fingerprint(cp['adapter_path']),
  'initial_optimizer_sha256':digest(cp['trainer_state']),
  'reference_result_sha256':digest(source/'jobs/sft_r3_validation_attempt1/result.json'),
  'script_sha256':digest(Path(__file__)),'rounds':[2,3]})
 print('prepared_cpu_only',run)
 raise SystemExit
manifest=read(run/'replay_manifest.json');config=check_protocol(run)['config']
assert digest(Path(__file__))==manifest['script_sha256']
cp=manifest['initial_checkpoint']
assert artifact_fingerprint(cp['adapter_path'])==manifest['initial_fingerprint']
assert digest(cp['trainer_state'])==manifest['initial_optimizer_sha256']
if (run/'execution_started.json').exists():raise FileExistsError('Replay already attempted; preserve evidence and use fresh ID')
write(run/'execution_started.json',{'time_unix':time.time(),'memory_limit_bytes':8000000000})
started=time.time();deadline=started+1800;guards=[]
for k in manifest['rounds']:
 output=run/f'bf16_r{k}_train'
 job={'kind':'training','method':'sft','round':k,'seed':42+k,'run':str(run),'output':str(output),
  'model':'/root/autodl-tmp/huggingface/models/Qwen3-0.6B','adapter':cp['adapter_path'],
  'adapter_fingerprint':artifact_fingerprint(cp['adapter_path']),
  'policy_version':cp['policy_version'],'trainer_state':cp['trainer_state'],
  'input_hashes':{'trainer_state':digest(cp['trainer_state'])}}
 write(output/'job.json',job)
 g=guard_job([str(ROOT/'.venv/bin/python'),str(ROOT/'utils/course_worker.py'),'--job',str(output/'job.json')],output,'0',config,deadline)
 guards.append(g)
 if g['status']!='completed':raise RuntimeError(g)
 r=read(output/'result.json');assert r['optimizer_steps']==24;cp=r['checkpoint']
 print('trained',k,g['device_peak_sampled_bytes'],r['mean_window_loss'],flush=True)
output=run/'bf16_r3_validation';panel=run/'panels/tune.jsonl'
job={'kind':'evaluation','seed':42,'run':str(run),'output':str(output),'model':'/root/autodl-tmp/huggingface/models/Qwen3-0.6B',
 'adapter':cp['adapter_path'],'adapter_fingerprint':artifact_fingerprint(cp['adapter_path']),
 'policy_version':cp['policy_version'],'variant':'original','panel':str(panel),'input_hashes':{'panel':digest(panel)}}
write(output/'job.json',job)
g=guard_job([str(ROOT/'.venv-vllm-cu128/bin/python'),str(ROOT/'utils/course_worker.py'),'--job',str(output/'job.json')],output,'0',config,deadline)
guards.append(g)
if g['status']!='completed':raise RuntimeError(g)
a=read(source/'jobs/sft_r3_validation_attempt1/result.json');b=read(output/'result.json')
assert digest(source/'jobs/sft_r3_validation_attempt1/result.json')==manifest['reference_result_sha256']
write(run/'replay_result.json',{'status':'completed_controlled_replay','source':'NF4 r1 weights plus same Adam state, then two BF16 epochs',
 'nf4_r3':a['metrics'],'bf16_r3':b['metrics'],'wall_seconds':time.time()-started,'guards':guards,
 'paired_tune_per_task':4,'final_panel_used':False,'adopted_by_original_run':False,
 'note':'Precision ablation on tune panel, not proof of generalization or a replacement for the original full run'})
print(json.dumps(read(run/'replay_result.json')),flush=True)
