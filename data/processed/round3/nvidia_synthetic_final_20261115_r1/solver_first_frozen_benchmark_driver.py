import json,os,socket,sys,time,statistics
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,'/home/hajimi2025/Qwen3-experiment')
from utils import run_nvidia_final_synthetic as r
from utils.run_nvidia_v2 import local_tokenizer
p=r.ROOT/'data/processed/round3/nvidia_synthetic_final_20261115_r1'
c=r.config(check_sdk=False);n=r.load_definitions();sft=r.load_renderer();records=[]
with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline benchmark')):
 tokenizer,meta=local_tokenizer()
 if tokenizer is None:raise RuntimeError('Local tokenizer required')
 for seed in r.rows(r.ROOT/c['frozen_seed_source']):
  started=time.perf_counter();v=n['solver_first_structured_proof'](seed);elapsed=time.perf_counter()-started
  reference_match=n['numeric_or_text_match'](v['sample']['answer'],seed['answer'])
  sample={**v['sample'],'candidate_id':seed['seed_id'],'verification_status':'accepted'}
  record=sft['make_trace_record'](sample) if v['solver_proof_accepted'] else None
  audit=sft['audit_trace_record'](record) if record else {'accepted':False,'reason':'independent_proof_rejected_no_rendering'}
  records.append({'seed_id':seed['seed_id'],'task_type':seed['task_type'],'prompt':seed['prompt'],
    'source':'solver_generated_rendered_benchmark_only','solver_proof_accepted':v['solver_proof_accepted'],'sft_render_audit_passed':audit['accepted'],'original_reference_answer_matches':reference_match,
    'proof':v['proof'],'verification':v['verification'],'completion':record['completion'] if record else None,'renderer_audit':audit,
    'rendered_completion_tokens':len(tokenizer(record['completion'],add_special_tokens=False).input_ids) if record else None,
    'cpu_solver_plus_verifier_seconds':elapsed,'hosted_completion_tokens':0,'hosted_reasoning_tokens':0,
    'lineage_status':'unavailable_legacy_frozen_seed','sft_training_eligible':False,'benchmark_only':True,
    'merge_into_sft':False,'finally_entered_sft':False})
r.write_jsonl_new(p/'solver_first_frozen_benchmark.jsonl',records)
r.write_json_new(p/'solver_first_frozen_benchmark_summary.json',{'source_private_seed_sha256':r.file_hash(r.ROOT/c['frozen_seed_source']),
  'public_only_solver_inputs':True,'tokenizer':meta,'http_requests':0,'gpu_used':False,
  'timing_scope':'warm CPU Solver plus independent Verifier; excludes tokenizer import/startup and renderer',
  'by_task':{t:{'proof_accepted':sum(x['task_type']==t and x['solver_proof_accepted'] for x in records),'outputs':3,
    'mean_rendered_completion_tokens':statistics.mean(x['rendered_completion_tokens'] for x in records if x['task_type']==t and x['rendered_completion_tokens'] is not None),
    'median_cpu_solver_plus_verifier_seconds':statistics.median(x['cpu_solver_plus_verifier_seconds'] for x in records if x['task_type']==t),
    'hosted_completion_tokens':0,'hosted_reasoning_tokens':0,'training_eligible':0} for t in r.TASKS}})
print('Frozen Solver-first benchmark complete; actual accepts:', {t:sum(x['task_type']==t and x['solver_proof_accepted'] for x in records) for t in r.TASKS})
