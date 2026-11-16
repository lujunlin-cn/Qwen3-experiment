"""Automatic isolated Gravity DPO-B chain with durable stage-completion hooks.

No hosted API or legacy Notebook execution. A quality-gate stop is final for
this Run: it never relaxes the verifier, fabricates a pair, or trains a fallback.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,os,pathlib,statistics,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from utils.dpo_gravity_worker import RUN,BASE,PARENT,WEIGHTS,read,rows,sha,write,write_rows,append,nl_api,pair_api,audit_nl_rows,validate_identity
CONFIG=ROOT/'configs/dpo_gravity_sft_c_20261116.json'
OLD=ROOT/'data/processed/round2/round2_20261110_phaseab_v1'
SFT=ROOT/'data/processed/round4/equation_sft_augmentation_20261116_r2'
TASKSET=ROOT/'data/processed/tasksets/roman_gravity_equation_v1'
PYTHON=ROOT/'.venv-low-memory/bin/python'
SOURCES=['01_Synthetic_Data_Engineering.ipynb','04_DPO.ipynb','utils/course_worker.py','utils/low_memory_logprobs.py','utils/dpo_gravity_worker.py','utils/run_dpo_gravity_sft_c.py','tests/test_dpo_gravity_sft_c.py','configs/dpo_gravity_sft_c_20261116.json']

def normalized(s):return ' '.join(s.casefold().split())
def csv_rows(p):
 with pathlib.Path(p).open() as f:return list(csv.DictReader(f))

def prepare():
 c=read(CONFIG);write(RUN/'config.json',c);validate_identity()
 datasets={s:csv_rows(TASKSET/f'splits/{s}.csv') for s in ['train','validation','holdout_test']}
 by_id={r['id']:r for r in datasets['train']}
 real=rows(OLD/'phase_c_train_panels/dpo_b.jsonl');synthetic=[r for r in rows(SFT/'sft_synthetic_110.jsonl') if r['task_type']=='gravity']
 if len(real)!=16 or len(synthetic)!=10:raise ValueError('Frozen 16+10 train panel absent')
 panel=[]
 for r in real:
  if r['id'] not in by_id or any(r[k]!=by_id[r['id']][k] for k in ['prompt','answer','source_row']):raise ValueError('Real panel not actual SFT Train')
  panel.append({**r,'provenance':{'source':'official_real_train','source_split':'train','source_id':r['id'],'source_row':r['source_row'],'source_file_sha256':sha(TASKSET/'splits/train.csv')}})
 prior=read(ROOT/'data/processed/round3/nvidia_synthetic_final_20261115_r1/train_prior_manifest.json')
 for r in synthetic:
  if not all(r.get(k) for k in ['solver_proof_accepted','sft_render_audit_passed','train_only_provenance_passed','leakage_audit_passed','sft_training_eligible']):raise ValueError('Synthetic seed audit missing')
  if r['lineage_status']!='declared' or not r['prior_feature_lineage']:raise ValueError('Synthetic lineage unavailable')
  if sha(ROOT/'data/processed/round3/nvidia_synthetic_final_20261115_r1/train_prior_manifest.json')!=r['prior_manifest_sha256']:raise ValueError('Train prior hash mismatch')
  for feature,src in r['prior_feature_lineage'].items():
   if src['source_id'] not in by_id or by_id[src['source_id']]['source_row']!=str(src['source_row']):raise ValueError('Synthetic source outside actual Train: '+feature)
  panel.append({'id':r['source_id'],'task_type':'gravity','prompt':r['prompt'],'answer':r['final_answer'],'split':'train',
   'provenance':{'source':'synthetic_train_prior','source_split':'train','candidate_id':r['candidate_id'],
     'source_file_sha256':sha(SFT/'sft_synthetic_110.jsonl'),'prior_manifest_sha256':r['prior_manifest_sha256'],
     'prior_source_ids':r['prior_source_ids'],'prior_source_rows':r['prior_source_rows'],'prior_feature_lineage':r['prior_feature_lineage']}})
 ns=nl_api();reference_audit=[]
 for r in panel:
  ref=ns['gravity_nl_public_reference'](r['prompt'])
  canonical=next((v['completion'] for v in synthetic if v['source_id']==r['id']),None)
  if canonical:
   checked=ns['verify_gravity_natural_language'](r['prompt'],canonical,r['answer'])
   if not checked['verified']:raise ValueError('Unchanged SFT teacher does not satisfy semantic verifier: '+str(checked))
  reference_audit.append({'sample_id':r['id'],'public_reference':ref,'answer_matches':ns['gravity_nl_official_verify']('gravity',ref['answer'],r['answer'])[0]})
 if not all(v['answer_matches'] for v in reference_audit):raise ValueError('Public OLS mismatches source answer')
 if len({r['id'] for r in panel})!=26 or len({normalized(r['prompt']) for r in panel})!=26:raise ValueError('Duplicate training prompts')
 # Cross-check against the actual SFT split. Query numeric reuse is not exact-prompt leakage in Gravity.
 leakage=[]
 for r in panel:
  for split in ['validation','holdout_test']:
   hits=[v['id'] for v in datasets[split] if normalized(v['prompt'])==normalized(r['prompt'])]
   if hits:leakage.append({'id':r['id'],'split':split,'hits':hits})
 if leakage:raise ValueError('Training prompt leak: '+str(leakage))
 write_rows(RUN/'train_panel.jsonl',panel);write(RUN/'data_audit.json',{'status':'passed','real':16,'synthetic_train_prior':10,'cross_split_duplicate_hits':leakage,'public_reference':reference_audit,
   'split_sha256':{s:sha(TASKSET/f'splits/{s}.csv') for s in datasets},'original_training_panel_sha256':sha(OLD/'phase_c_train_panels/dpo_b.jsonl'),'original_synthetic_sha256':sha(SFT/'sft_synthetic_110.jsonl')})
 panels={}
 for name,count in [('tune',48),('locked_final',100)]:
  rs=[r for r in rows(OLD/f'panels/{name}.jsonl') if r['task_type']=='gravity']
  if len(rs)!=count:raise ValueError('Frozen evaluation panel count mismatch')
  valid={r['id']:r for r in datasets['validation']}
  if any(r['id'] not in valid or valid[r['id']]['prompt']!=r['prompt'] or valid[r['id']]['answer']!=r['answer'] for r in rs):raise ValueError('Evaluation panel differs from actual Validation')
  write_rows(RUN/f'{name}_panel.jsonl',rs);panels[name]={'n':len(rs),'sha256':sha(RUN/f'{name}_panel.jsonl'),'historical_panel_sha256':sha(OLD/f'panels/{name}.jsonl')}
 if {r['id'] for r in rows(RUN/'tune_panel.jsonl')}&{r['id'] for r in rows(RUN/'locked_final_panel.jsonl')}:raise ValueError('Tune/Final overlap')
 write(RUN/'registration.json',{'status':'registered_before_gpu','config_sha256':sha(CONFIG),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  'sources':{p:sha(ROOT/p) for p in SOURCES},'panels':panels,'training_panel_sha256':sha(RUN/'train_panel.jsonl'),'base_identity_sha256':sha(RUN/'identity_audit.json'),
  'evaluation_design':'Frozen48/100 Gravity subsets both from Validation, not Holdout Test. Historical SFT used some of these validation questions; this is not untouched test evidence.',
  'formal_holdout_used':False,'correct_short_threshold_tokens':1024,'decode':'Greedy, top_p1 top_k-1 Thinking enabled, cap16384, identical chat template and prompts for both models',
  'created_unix':time.time()})

def invoke(stage):
 c=read(RUN/'config.json');available=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 if available<c['minimum_host_available_bytes']:raise RuntimeError('Insufficient free shared memory; other GPU processes left intact')
 log=RUN/f'{stage}.log';start=time.time()
 if stage in ['rollout8','rollout4','tune','final']:
  actual=subprocess.check_output(['docker','image','inspect',c['backend_image'],'--format','{{.Id}}'],text=True).strip()
  if actual!=c['backend_image_id']:raise ValueError('Inference image identity changed')
  cmd=['docker','run','--rm','--name','qwen-dpo-c-'+stage,'--network','none','--gpus','all','--ipc','host','--user',f'{os.getuid()}:{os.getgid()}',
   '-e','HOME=/tmp','-e','VLLM_ENABLE_V1_MULTIPROCESSING=0','-e','HF_HUB_OFFLINE=1','-e','TOKENIZERS_PARALLELISM=false',
   '-v',f'{ROOT}:{ROOT}','-w',str(ROOT),'--entrypoint','python3',c['backend_image'],'utils/dpo_gravity_worker.py',stage]
 else:cmd=[str(PYTHON),'utils/dpo_gravity_worker.py',stage]
 with log.open('x') as f:
  proc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
 append(RUN/'stage_hooks.jsonl',{'stage':stage,'exit_code':proc.returncode,'start':start,'end':time.time(),'log_sha256':sha(log)})
 if proc.returncode:raise RuntimeError(f'{stage} failed: exit {proc.returncode}; see {log.name}')

def audit_rollouts(supplement=False):
 rs=rows(RUN/'rollout8/SFT_C_raw.jsonl')
 if len(rs)!=208:raise ValueError('Initial rollout incomplete')
 if supplement:
  extra=rows(RUN/'rollout4/SFT_C_raw.jsonl')
  if len(extra)!=104:raise ValueError('Supplement rollout incomplete')
  rs+=extra
 c=read(RUN/'config.json');panel={r['id']:r for r in rows(RUN/'train_panel.jsonl')}
 for r in rs:
  if r['prompt']!=panel[r['sample_id']]['prompt'] or r['policy_fingerprint']['adapter_model.safetensors']!=c['parent_adapter_sha256']:raise ValueError('Rollout policy/prompt mismatch')
  if len(r['behavior_logprobs'])!=len(r['completion_token_ids']):raise ValueError('Behavior alignment missing')
 audited=audit_nl_rows(rs);pairs=pair_api()(audited)
 suffix='12' if supplement else '8';write_rows(RUN/f'rollouts_audited_{suffix}.jsonl',audited)
 detail=[]
 for p in pairs:
  item={k:v for k,v in p.items() if k not in ['chosen','rejected']}
  for side in ['chosen','rejected']:
   r=p[side];item[side]={'group_index':r['group_index'],'tokens':len(r['completion_token_ids']),'completion_sha256':hashlib.sha256(r['completion'].encode()).hexdigest(),'proof_verified':r['proof_audit']['verified']}
  detail.append(item)
 write_rows(RUN/f'preference_pairs_{suffix}.jsonl',detail)
 counts=collections.Counter(v for r in audited for v in r['proof_audit']['reasons'])
 summary={'status':'passed' if len(pairs)>=16 else 'no_preference_signal','rows':len(rs),'prompts':26,'pairs':len(pairs),'coverage_rate':len(pairs)/26,
  'official_correct_complete':sum(r['correct'] for r in audited),'natural_language_proof_passed':sum(r['proof_audit']['verified'] for r in audited),
  'truncated':sum(r['truncated'] for r in audited),'failure_reasons':dict(counts),'pairs_by_prompt':{k:sum(p['sample_id']==k for p in pairs) for k in panel},
  'proofs_by_prompt':{k:sum(r['sample_id']==k and r['proof_audit']['verified'] for r in audited) for k in panel},
  'total_completion_tokens':sum(len(r['completion_token_ids']) for r in rs),'thinking_included':True,'eos_included':True}
 write(RUN/f'rollout_quality_{suffix}.json',summary)
 if len(pairs)>=16:write_rows(RUN/'rollouts_audited.jsonl',audited)
 return summary

def score(stage):
 from utils.round2_evaluation import wilson,paired_comparison,paired_mean_analysis
 scored={};summaries={}
 for arm in ['SFT_C','DPO']:
  rs=audit_nl_rows(rows(RUN/stage/(arm+'_raw.jsonl')));scored[arm]=rs
  write_rows(RUN/stage/(arm+'_scored.jsonl'),rs)
  valid=[r for r in rs if r['correct'] and r['proof_audit']['verified'] and r['finish_reason']=='stop']
  summaries[arm]={'n':len(rs),'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),
    'accuracy_wilson95':wilson(sum(r['correct'] for r in rs),len(rs)),'truncation_rate':sum(r['truncated'] for r in rs)/len(rs),
    'mean_completion_tokens':statistics.mean(len(r['completion_token_ids']) for r in rs),'median_completion_tokens':statistics.median(len(r['completion_token_ids']) for r in rs),
    'correct_complete_n':len(valid),'correct_complete_mean_tokens':statistics.mean(len(r['completion_token_ids']) for r in valid) if valid else None,
    'natural_language_proof_verification_rate':sum(r['proof_audit']['verified'] for r in rs)/len(rs),
    'format_valid_rate':sum(r['format_valid'] for r in rs)/len(rs),'correct_short_rate_all_questions':sum(len(r['completion_token_ids'])<=1024 for r in valid)/len(rs),
    'correct_short_fraction_correct_complete':sum(len(r['completion_token_ids'])<=1024 for r in valid)/len(valid) if valid else None}
 comparison=paired_comparison(scored['SFT_C'],scored['DPO'])
 aa={r['sample_id']:r for r in scored['SFT_C']};bb={r['sample_id']:r for r in scored['DPO']}
 ids=[k for k in aa if aa[k]['correct'] and bb[k]['correct'] and aa[k]['proof_audit']['verified'] and bb[k]['proof_audit']['verified']]
 comparison['both_correct_verified_length']=paired_mean_analysis([len(bb[k]['completion_token_ids'])-len(aa[k]['completion_token_ids']) for k in ids]) if ids else None
 comparison['accuracy_delta_bootstrap']=paired_mean_analysis([int(bb[k]['correct'])-int(aa[k]['correct']) for k in aa])
 a,b=summaries['SFT_C'],summaries['DPO']
 point=bool(b['accuracy']>=a['accuracy'] and b['mean_completion_tokens']<a['mean_completion_tokens'] and b['truncation_rate']<=a['truncation_rate'] and b['natural_language_proof_verification_rate']>=a['natural_language_proof_verification_rate'])
 out={'status':'completed','metrics':summaries,'paired_analysis':comparison,'point_estimate_gate_passed':point,'statistical_noninferiority_claim':False,'note':'No noninferiority margin registered; wide intervals are inconclusive.'}
 write(RUN/stage/'comparison.json',out);return out

def report(status,error=None):
 c=read(RUN/'config.json') if (RUN/'config.json').exists() else read(CONFIG)
 q=next((read(RUN/f'rollout_quality_{s}.json') for s in ['12','8'] if (RUN/f'rollout_quality_{s}.json').exists()),{})
 train=read(RUN/'training_result.json') if (RUN/'training_result.json').exists() else None
 reload=read(RUN/'reload_result.json') if (RUN/'reload_result.json').exists() else None
 results={s:read(RUN/s/'comparison.json') for s in ['tune','final'] if (RUN/s/'comparison.json').exists()}
 protected=read(RUN/'identity_audit.json')['protected_sha256'];integrity={p:sha(ROOT/p)==h for p,h in protected.items()}
 manifest={'status':status,'error':error,'run_id':c['run_id'],'rollout_quality':q,'training_completed':bool(train),'optimizer_steps':train['optimizer_steps'] if train else 0,'checkpoint_reload':reload,
  'evaluations_completed':list(results),'protected_files_unchanged':integrity,'parent_adapter_unchanged':sha(PARENT/'adapter_model.safetensors')==c['parent_adapter_sha256'],'hosted_api_calls':0,'updated_unix':time.time()}
 write(RUN/'final_status.json',manifest)
 lines=['# Gravity DPO-B：SFT C 完整实验报告','',f'Run：`{c["run_id"]}`。最终状态：`{status}`。',
  '',f'固定父 Adapter SHA256：`{c["parent_adapter_sha256"]}`。Base Revision：`{c["base_revision"]}`。启动前全部 Base/Tokenizer 文件与历史 fingerprint 比对通过；Policy 和冻结 Reference 同为 SFT C，无回退。',
  '', '## 数据与验收合同','', '训练题为原始16道 Gravity Train +10道保留 `synthetic_train_prior` 逐特征来源的合成题。原始 Prompt 不追加 JSON/答案/证明。独立 OLS 使用全部公开示例；最终自然语言证明中的 g、Query 时间、未舍入距离、物理关系和盒装答案按 rel_tol=1e-4、abs_tol=1e-5 检查；官方最终评分器保持不变。隐藏 Thinking 不参与证明验收，但计入总长度及 DPO Sequence Log Probability。不可解析或不完整回答记录为 unverifiable。Verifier 是保守的、有限语法的数学证书检查，未认证不等于数学错误。',
  '', 'Tune48和Locked Final100均复用历史冻结 Validation 面板；Locked Final不是官方 Holdout Test。未使用 Holdout 结果调整任何设置。两者可能被历史实验评估过，不能声称首次盲测。',
  '', 'CPU附加反例在查看本轮Rollout内容前发现并修复：相互矛盾的拟合g、带后续自然语言的错误算式、输入Chat Template已开启但Completion未关闭的Thinking。最终使用gravity-natural-language-v1.1；原始注册和第一次完成链的状态/日志均保留。采样配置、原始Completion及数学容差没有改变。附加CPU门槛通过后，自动续链重新验收全部原始Rollout。',
  '', '## 实际 Rollout 与偏好数据','', '```json',json.dumps(q,ensure_ascii=False,indent=2),'```',
  '', '初始每题8条；不足16对时全26题各补4条。Chosen最短、Rejected最长，双方均正确完整且独立证明通过，长度差>=64且>=长答20%，每题最多1对。所有错误、截断、不可验证回答均排除。']
 if train:
  lines+=['','## 真实训练和重载','',f'标准 Sigmoid DPO，beta=0.1，LR=2e-6，梯度累积4，2 epochs，clip=1.0。完成 {train["optimizer_steps"]} 个真实更新，{train["training_seconds"]:.2f}s；改变 {train["changed_tensors"]} 个权重张量。Sequence Log Probability 按完整Response求和，包含Thinking和EOS；使用现有分块投影和逐侧重算梯度。Reference缓存来自训练前冻结SFT C。', '',f'Checkpoint：`{WEIGHTS}/adapter`；权重/Optimizer Hash和重载probe见训练结果与checkpoint manifest。重载：`{json.dumps(reload,ensure_ascii=False)}`。', '',f'CUDA峰值与完整曲线见`training_result.json`、`.autodl/{c["run_id"]}/training_curve.jsonl`。只报告本进程Torch CUDA分配器显存，不把共享GB10全设备占用解释为本实验显存。']
 else:
  lines+=['','## 训练门槛与实际执行','',f'实际 Optimizer Steps：0；未创建DPO权重，未执行Tune/Final模型比较。{("达到最多12条/题后仍不足16对，按协议终止，不降低Verifier标准。" if status=="no_preference_signal" else "失败原因见下方和原始日志。")}']
 for stage,res in results.items():
  lines+=['',f'## {stage} 配对结果','','|模型|Accuracy (95% Wilson)|截断率|平均Token|正确完整平均Token|证明通过率|格式率|正确短答率|','|---|---|---|---|---|---|---|---|']
  for arm,m in res['metrics'].items():
   lines.append(f'|{arm}|{m["accuracy"]:.3f} {m["accuracy_wilson95"]}|{m["truncation_rate"]:.3f}|{m["mean_completion_tokens"]:.1f}|{m["correct_complete_mean_tokens"]}|{m["natural_language_proof_verification_rate"]:.3f}|{m["format_valid_rate"]:.3f}|{m["correct_short_rate_all_questions"]:.3f}|')
  lines+=['','逐题结果与McNemar、配对Bootstrap95%区间见该阶段`*_scored.jsonl`和`comparison.json`。正确短答预注册为：正确、完整、证明通过且总Response<=1024 Token。', '',f'四项点估计门槛通过：`{res["point_estimate_gate_passed"]}`。未预注册非劣界值，不能宣称统计非劣。']
 if error:lines+=['','## 停止原因','',str(error)]
 lines+=['','## 最终结论','',('未获得足够的严格正确长短偏好信号，本轮未达到真实DPO训练和效果验收目标。没有训练或评测结果可用于声称缩短输出、保持正确率或改进推理。' if not train else '最终验收以Locked Final的四项点估计门槛及统计不确定性为准；输出变短本身不代表数学能力提升。'),
  '', 'PPO/GRPO、默认SFT入口和SFT C保持不变；无Hosted API调用。失败日志、原始Completion/Token IDs/行为Logprobs、来源、偏好筛选及完成hook均保留。', '', 'CPU回归实测日志：`cpu_tests.log`。复现入口：`python utils/run_dpo_gravity_sft_c.py all`（当前Run已经存在，禁止覆盖；新实验需另行注册Run ID）。']
 path=ROOT/'docs/DPO_GRAVITY_SFT_C_FINAL_REPORT.md'
 if path.exists():raise FileExistsError('Do not overwrite final report')
 path.write_text('\n'.join(lines)+'\n')
 # Immutable index includes all evidence, no weight/optimizer payload in Git.
 index={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and p.name not in ['driver.log','artifact_manifest.json']}
 write(RUN/'artifact_manifest.json',{'files':index,'report_sha256':sha(path),'checkpoint_path':str(WEIGHTS) if train else None})

def all_stages():
 try:
  prepare()
  with (RUN/'cpu_tests.log').open('x') as f:
   p=subprocess.run([str(PYTHON),'-m','unittest','tests.test_dpo_gravity_sft_c','tests.test_round2_numerics','tests.test_course_experiment','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(ROOT)+':'+str(ROOT/'tests')})
  if p.returncode:raise RuntimeError('CPU regression gate failed; no GPU started')
  invoke('rollout8');q=audit_rollouts()
  if q['pairs']<16:invoke('rollout4');q=audit_rollouts(True)
  if q['pairs']<16:report('no_preference_signal');return
  invoke('train');invoke('reload');invoke('tune');score('tune')
  checkpoint=read(WEIGHTS/'checkpoint.json');checkpoint['frozen_after_tune_unix']=time.time();checkpoint['tune_comparison_sha256']=sha(RUN/'tune/comparison.json')
  write(RUN/'checkpoint_freeze.json',checkpoint)
  invoke('final');score('final');report('completed')
 except BaseException as e:
  append(RUN/'failures.jsonl',{'error':repr(e),'traceback':traceback.format_exc(),'unix':time.time()})
  if not (RUN/'final_status.json').exists():report('failed',repr(e))
  raise

def resume_after_cpu_repair(wait_pid=None):
 """One completion-triggered recovery; no polling, no rollout regeneration.

 The initial chain's immutable-source check stops before another GPU phase
 because the CPU adversarial repair changed sources. Raw generation finishes
 unchanged. This hook resumes with the strengthened acceptance contract.
 """
 if wait_pid:
  import select
  try:
   if hasattr(os,'pidfd_open'):fd=os.pidfd_open(wait_pid)
   else:
    # Some aarch64 CPython builds omit the wrapper although glibc/kernel
    # provide pidfd_open. This still blocks on process completion, no polls.
    import ctypes,errno
    libc=ctypes.CDLL(None,use_errno=True);opener=libc.pidfd_open
    opener.argtypes=[ctypes.c_int,ctypes.c_uint];opener.restype=ctypes.c_int
    fd=opener(wait_pid,0)
    if fd<0:
     error=ctypes.get_errno()
     if error==errno.ESRCH:raise ProcessLookupError(wait_pid)
     raise OSError(error,os.strerror(error))
  except ProcessLookupError:fd=None
  if fd is not None:
   write(RUN/'cpu_repair_continuation_wait.json',{'waiting_pid':wait_pid,'mechanism':'pidfd/select completion hook; no periodic training monitoring','time':time.time()})
   select.select([fd],[],[]);os.close(fd)
 if not (RUN/'rollout8/completion_hook.json').exists():
  raise RuntimeError('Original rollout did not complete; preserve original failure and do not regenerate')
 import shutil
 archive=RUN/'attempts/before_cpu_adversarial_repair';archive.mkdir(parents=True,exist_ok=False)
 moved={}
 for name in ['final_status.json','artifact_manifest.json','rollouts_audited_8.jsonl','rollouts_audited_12.jsonl','rollouts_audited.jsonl',
              'preference_pairs_8.jsonl','preference_pairs_12.jsonl','rollout_quality_8.json','rollout_quality_12.json',
              'rollout4.log','train.log']:
  old=RUN/name
  if old.exists():
   moved[name]={'sha256':sha(old),'archived_path':str((archive/name).relative_to(RUN))};shutil.move(old,archive/name)
 old_report=ROOT/'docs/DPO_GRAVITY_SFT_C_FINAL_REPORT.md'
 if old_report.exists():
  moved['docs/DPO_GRAVITY_SFT_C_FINAL_REPORT.md']={'sha256':sha(old_report),'archived_path':str((archive/'initial_chain_report.md').relative_to(RUN))}
  shutil.move(old_report,archive/'initial_chain_report.md')
 write(archive/'preservation_manifest.json',moved)
 log=RUN/'cpu_verifier_repair_gate.log'
 with log.open('x') as f:
  p=subprocess.run([str(PYTHON),'-m','unittest','tests.test_dpo_gravity_sft_c','tests.test_round2_numerics','tests.test_course_experiment','-v'],cwd=ROOT,
   stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(ROOT)+':'+str(ROOT/'tests')})
 if p.returncode:report('failed','Strengthened CPU verifier regression failed; no further GPU phases');return
 write(RUN/'verification_amendment.json',{'version':'gravity-natural-language-v1.1','original_registration_sha256':sha(RUN/'registration.json'),
  'reason':'Acceptance strengthened by CPU adversarial tests before inspecting generated completion contents. Input/sampling/token caps unchanged; no rerollout.',
  'sources':{name:sha(ROOT/name) for name in SOURCES},'cpu_tests_exit_code':p.returncode,'cpu_tests_log':log.name,'cpu_tests_log_sha256':sha(log),
  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'registered_unix':time.time()})
 try:
  validate_identity()
  q=audit_rollouts()
  if q['pairs']<16:
   if not (RUN/'rollout4/completion_hook.json').exists():invoke('rollout4')
   q=audit_rollouts(True)
  if q['pairs']<16:report('no_preference_signal');return
  invoke('train');invoke('reload');invoke('tune');score('tune')
  checkpoint=read(WEIGHTS/'checkpoint.json');checkpoint['frozen_after_tune_unix']=time.time();checkpoint['tune_comparison_sha256']=sha(RUN/'tune/comparison.json')
  write(RUN/'checkpoint_freeze.json',checkpoint);invoke('final');score('final');report('completed')
 except BaseException as e:
  append(RUN/'failures.jsonl',{'error':repr(e),'traceback':traceback.format_exc(),'unix':time.time(),'continuation':'CPU adversarial repair'})
  if not (RUN/'final_status.json').exists():report('failed',repr(e))
  raise

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['all','report','resume-after-cpu-repair','finalize-offline']);p.add_argument('--wait-pid',type=int);a=p.parse_args()
 if a.stage=='all':all_stages()
 elif a.stage=='resume-after-cpu-repair':resume_after_cpu_repair(a.wait_pid)
 elif a.stage=='finalize-offline':finalize_offline()
 else:report('recovered_completed')


def finalize_offline():
 """Audit the entire immutable GPU pool; never retry or train post-hoc."""
 import re,shutil,math
 tests=RUN/'cpu_final_tests.log';log=tests.read_text()
 if not re.search(r'Ran 69 tests',log) or not re.search(r'^OK$',log,re.M):raise ValueError('Final CPU gate failed')
 old=read(RUN/'final_status.json')
 if old['status']!='no_preference_signal' or old['optimizer_steps']!=0:raise ValueError('Expected completed no-signal gate')
 frozen=read(RUN/'artifact_manifest.json')['files']
 for name in ['rollout8/SFT_C_raw.jsonl','rollout4/SFT_C_raw.jsonl']:
  if sha(RUN/name)!=frozen[name]:raise ValueError('Raw GPU output changed')
 rs=rows(RUN/'rollout8/SFT_C_raw.jsonl')+rows(RUN/'rollout4/SFT_C_raw.jsonl')
 if len(rs)!=312:raise ValueError('Complete fixed pool required')
 rs=audit_nl_rows(rs);pairs=pair_api()(rs)
 if pairs:raise ValueError('Unexpected preferences after offline diagnostic repair; no post-hoc training allowed')
 write_rows(RUN/'final_rollouts_audited_v12.jsonl',rs);write_rows(RUN/'final_preference_pairs_v12.jsonl',[])
 sources={}
 for src in ['official_real_train','synthetic_train_prior']:
  chunk=[r for r in rs if r['provenance']['source']==src]
  sources[src]={'prompts':len({r['sample_id'] for r in chunk}),'rollouts':len(chunk),'official_final_correct':sum(r['correct'] for r in chunk),'proof_certified':sum(r['proof_audit']['verified'] for r in chunk)}
 examples=[]
 for r in rs:
  if r['correct'] and len(examples)<8:
   a=r['proof_audit'];examples.append({'sample_id':r['sample_id'],'group_index':r['group_index'],'predicted':r['predicted'],'expected_answer':r['expected_answer'],
    'public_reference':a.get('reference'),'reported_g':a.get('reported_g_values'),'reasons':a['reasons'],'final_answer_text':nl_api()['gravity_nl_final_section'](r['completion'])[0]})
 write_rows(RUN/'official_correct_but_unverified_cases.jsonl',examples)
 stages={s:read(RUN/s/'SFT_C_result.json') for s in ['rollout8','rollout4']}
 groups={k:[r for r in rs if r['sample_id']==k] for k in sorted({r['sample_id'] for r in rs})}
 q={'status':'no_preference_signal','acceptance_version':'gravity-natural-language-v1.2','prompts':26,'rollouts':312,
  'official_final_correct':sum(r['correct'] for r in rs),'format_valid':sum(r['format_valid'] for r in rs),'proof_certified':sum(r['proof_audit']['verified'] for r in rs),
  'effective_pairs':0,'minimum_pairs':16,'truncated':sum(r['truncated'] for r in rs),'finish_stop':sum(r['finish_reason']=='stop' for r in rs),
  'response_tokens':sum(len(r['completion_token_ids']) for r in rs),'mean_response_tokens':statistics.mean(len(r['completion_token_ids']) for r in rs),
  'mean_official_correct_tokens':statistics.mean(len(r['completion_token_ids']) for r in rs if r['correct']),
  'all_behavior_logprobs_aligned_finite':all(len(r['completion_token_ids'])==len(r['behavior_logprobs']) and all(math.isfinite(x) for x in r['behavior_logprobs']) for r in rs),
  'pinned_eos_token_ids':read(BASE/'generation_config.json')['eos_token_id'],
  'actual_last_token_counts':dict(collections.Counter(str(r['completion_token_ids'][-1]) for r in rs)),
  'all_stopped_eos_included':all(r['completion_token_ids'][-1] in read(BASE/'generation_config.json')['eos_token_id'] for r in rs if r['finish_reason']=='stop'),
  'by_source':sources,'failure_reasons_nonexclusive':dict(collections.Counter(v for r in rs for v in r['proof_audit']['reasons'])),
  'per_prompt':{k:{'rollouts':len(v),'official_final_correct':sum(r['correct'] for r in v),'proof_certified':sum(r['proof_audit']['verified'] for r in v)} for k,v in groups.items()},
  'generation_seconds':sum(r['generation_seconds'] for r in stages.values()),'peak_reserved_bytes':max(m['peak_reserved_bytes'] for r in stages.values() for m in r['memory']),
  'optimizer_steps':0,'tune_evaluated':False,'locked_final_evaluated':False,'dpo_checkpoint_created':False,
  'note':'Official answer extraction applies to final section only, independent of proof format. Zero certified proofs does not establish every free-language proof is wrong.'}
 write(RUN/'final_quality_summary_v12.json',q)
 write(RUN/'offline_acceptance_amendment_v12.json',{'original_final_status_sha256':sha(RUN/'final_status.json'),'original_artifact_manifest_sha256':sha(RUN/'artifact_manifest.json'),
  'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'cpu_tests_log_sha256':sha(tests),
  'changes':['Use unchanged official extractor independently of proof format','Verify indexed g_i only if present','Check known Query time as exact numeric input'],
  'no_sampling_change':True,'no_tolerance_relaxation':True,'no_new_gpu_requests':True,'full_pool_reaudited':312,'outcome_unchanged':'no_preference_signal','time':time.time()})
 report_path=ROOT/'docs/DPO_GRAVITY_SFT_C_FINAL_REPORT.md'
 if not (RUN/'automatic_final_report_v11.md').exists():shutil.copyfile(report_path,RUN/'automatic_final_report_v11.md')
 report_path.write_text(f'''# Gravity DPO-B：SFT C 最终实验报告

Run：`dpo_gravity_sft_c_20261116_r1`。**最终状态：`no_preference_signal`。**

全部312条真实GPU Rollout已完成（26题×初始8条，再全题各补4条）。最终官方答案通过{q['official_final_correct']}/312，独立自然语言证明认证0/312，有效DPO-B偏好0/26，未达到16对门槛。实际Optimizer Steps为0；未创建DPO Checkpoint、未执行Tune/Locked Final比较，不能宣称DPO收益。

## SFT C权重与数据来源

Base：Qwen/Qwen3-0.6B，Revision `c1899de289a04d12100db370d81485cdf75e47ca`。Policy和计划中的冻结Reference同为SFT C。Adapter：`.autodl/equation_sft_augmentation_20261116_r2/C/adapter`。

Adapter SHA256：`7033241d41efb086762df20789ce2b70764bb3a5c524e29b8d8f846d3254e3f4`。

启动前逐文件核对Base权重、config、tokenizer、vocab、merges和generation_config，共7项匹配历史指纹。BF16 LoRA r=8、alpha=16、dropout=0，无R1或其他RL回退；原Adapter Hash保持不变。证据：`identity_audit.json`。

16条历史Gravity真实Train题逐一核对实际SFT Train的ID、Prompt、Answer、source_row；10条现有Gravity合成题保留`synthetic_train_prior`、Prior Manifest Hash和逐特征来源，来源ID/行号均属于实际Train。26题ID及Normalized Prompt无重复，与实际Validation/Holdout无Normalized Prompt重复。证据：`train_panel.jsonl`、`data_audit.json`。

Tune48及Locked Final100复用历史冻结面板，均来自Validation，并非官方Holdout Test。二者无交集，Hash已冻结；可能曾被历史实验评测，不能声称首次盲测。本轮因数据门槛失败，没有对这些面板执行推理。

## 自然语言数学Verifier

核心实现在Notebook01新增Gravity DPO-B Section；Notebook04保留DPO-B选对及数学loss教学实现。专用Runner/Worker只做调度、模型I/O、归档并接入`course_worker.train_dpo`，不执行Legacy DPO Notebook入口，不要求JSON证明。

Verifier独立解析全部公开时间/距离，用`2*math.fsum(t*t*d)/math.fsum(t**4)`重新计算OLS，不调用Generator/Solver计算函数。检查最终回答中的物理关系、全部示例拟合依据、g、Query时间、未舍入距离、最终盒装答案及可解析的中间算式。计算精度保持rel_tol=1e-4、abs_tol=1e-5；Query时间是已知输入，数值精确匹配。无需逐条打印g_i，但打印索引g_i时会独立检查。

物理表达式通过闭合AST转为符号表达式验证等价性；数值算式不执行eval。不使用模型自评，不用全文/模板字符串相等作为数学标准。原有10条SFT Gravity教师及自然语言改写正例通过CPU测试。

Thinking只保留审计：模型启用Thinking时，仅取最后一个`</think>`后的最终回答。输入Chat Template已开启但响应未关闭的Thinking也拒绝。所有Thinking、错误/截断输出和EOS仍计入Response长度以及计划中的Sequence Log Probability。

最终v1.2将**官方答案提取/比较与Proof Format分开**。原v1.1采用较窄的numeric-box提取，最终对全部原始输出使用现有官方extract_final_answer；本轮二者均为{q['official_final_correct']}条。这消除了把Proof Format当成Accuracy开关的实现问题，没有放宽证明格式、数学容差或偏好标准，证明与偏好仍为0。v1.1原始状态及报告保留。

**解析限制：**这是保守的有限语法证书检查，尚不能覆盖任意等价LaTeX/自由表述。0/312是认证通过数，不是证明所有推理都错误。不可解析、缺少关键拟合/Query计算或无法验证完整过程的回答标为unverifiable，不加入偏好。

## 真实GPU采样结果

硬件为本机共享NVIDIA GB10，非历史RTX4080。未终止其他GPU进程。模型只收到原始公开Prompt及固定Chat Template；未追加JSON指令，未发送标准答案、Solver Proof或隐藏规则。

Temperature1.0，top_p1.0，top_k=-1，Thinking开启，Max Tokens16384；每题同一采样合同。原始Completion、Prompt/Response Token IDs、逐Token Behavior Logprobs、finish_reason、Policy Fingerprint及来源均归档。

|阶段|实际回答|生成耗时（不含初始化）|
|---|---:|---:|
|初始8条/题|208|{stages['rollout8']['generation_seconds']:.2f}s|
|补充4条/题|104|{stages['rollout4']['generation_seconds']:.2f}s|
|合计|312|{q['generation_seconds']:.2f}s|

|最终离线审计|结果|
|---|---:|
|官方最终答案通过|{q['official_final_correct']}/312|
|Format Valid|{q['format_valid']}/312|
|独立完整证明认证|0/312|
|有效偏好对|0/26|
|Truncation|0/312|
|finish_reason=stop|312/312|
|Response Token总数（含Thinking/EOS）|{q['response_tokens']}|
|Mean Response Tokens|{q['mean_response_tokens']:.2f}|
|Behavior Logprobs对齐且有限|{q['all_behavior_logprobs_aligned_finite']}|
|所有stop回答包含真实EOS|{q['all_stopped_eos_included']}|

冻结generation_config中的合法EOS为151645和151643；实际分别出现311次和1次，均保留并计入Token总数。

本进程Torch CUDA峰值reserved为{q['peak_reserved_bytes']} bytes（约{q['peak_reserved_bytes']/1e9:.3f}GB），低于8,000,000,000 bytes分配器守卫。UMA设备无法可靠提供全设备显存；该值不是整机显存，也不保证涵盖所有非Torch CUDA分配。

逐来源、逐题和非互斥失败原因见`final_quality_summary_v12.json`。多个失败原因可属于同一条，不能相加当作独立失败数。逐条Verifier结果见`final_rollouts_audited_v12.jsonl`。官方答对但过程未认证的原始案例及参考值见`official_correct_but_unverified_cases.jsonl`，包括只输出答案、缺少拟合/Query计算和中间数值错误。

这些是Train Rollout质量统计，不是Validation准确率；不能以它们替代SFT C与DPO的效果比较。

## 偏好门槛、真实训练与评测

Chosen取同题最短、Rejected取最长的正确完整认证回答，差>=64Token且>=长回答20%，每题最多1对。错误、截断和不可验证回答均排除。初始208条不足16对，自动对全26题各补4条；达到每题12条上限后仍无有效对，按协议停止。没有借用教师回答、伪造Chosen或改成正确/错误偏好。

|阶段|真实状态|
|---|---|
|Optimizer Steps|0，未启动DPO训练|
|Loss/真实梯度|无训练结果，不填虚构数值|
|Checkpoint及重载|未产生，未执行|
|Tune SFT C vs DPO|未执行|
|Locked Final SFT C vs DPO|未执行|
|Wilson/McNemar/Paired Bootstrap|无两模型评测数据，未计算|
|DPO-B目标|无法验收，无收益证据|

已实现标准Sigmoid DPO、冻结SFT C Reference、完整Response求和mask、低显存分块logprob、beta0.1、LR2e-6、accum4、2epochs、clip1.0及梯度/曲线/Checkpoint重载后自动Tune/Final。CPU测试通过不代表本轮发生过真实GPU DPO更新；数据门槛阻止了它。

## CPU回归、完成hook与历史隔离

初始GPU前60项CPU回归通过；查看本轮Completion前的附加反例修复矛盾g、尾随文字中的错误算式、未关闭Thinking等问题，后续阶段前64项通过。旧source-hash守卫阻止旧完成链继续；登记v1.1后自动续链，初始208条未重生成。aarch64 Python缺少os.pidfd_open的hook启动失败日志保留，随后使用glibc pidfd_open/select等待完成，未轮询训练进度。

最终离线修复官方答案提取与Format混用，增强可选索引g和已知Query时间检查。**最终69项CPU测试实际通过，耗时4.455s**，证据`cpu_final_tests.log`。对全部312条原始输出重审，仍为no_preference_signal；未新增GPU/API请求。v1.1自动结果、旧注册、失败日志和原始回答均保留，v1.2单独归档。

SFT C、Notebook03默认入口、PPO/GRPO、历史DPO/SFT产物及原始数据保持不变；无NVIDIA Hosted API调用。完整性见`final_publication_manifest.json`；原始v1.1自动报告见`automatic_final_report_v11.md`。

## 最终判断

本轮完成312条真实GPU采样和全部严格数据审计，但未得到启动DPO-B所需的16对偏好。不能宣称完成真实DPO训练、缩短输出、保持正确率或改善数学证明。正确率、长度、截断率和必要证明率四项效果条件均没有两模型比较证据。当前瓶颈是SFT C最终回答的可验证完整性及解析覆盖，未通过降低门槛制造学习信号。
''')
 write(RUN/'final_publication_manifest.json',{'status':'no_preference_signal','authoritative_quality_summary':'final_quality_summary_v12.json','verifier_version':'gravity-natural-language-v1.2',
  'report_sha256':sha(report_path),'original_automatic_report_sha256':sha(RUN/'automatic_final_report_v11.md'),'original_final_status_sha256':sha(RUN/'final_status.json'),
  'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'protected_files_unchanged':{n:sha(ROOT/n)==h for n,h in read(RUN/'identity_audit.json')['protected_sha256'].items()},
  'parent_adapter_unchanged':sha(PARENT/'adapter_model.safetensors')==read(CONFIG)['parent_adapter_sha256'],
  'files':{str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and p.name!='final_publication_manifest.json'}})

if __name__=='__main__':main()
