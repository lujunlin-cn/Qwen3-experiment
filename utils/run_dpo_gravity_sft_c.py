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
   fd=os.pidfd_open(wait_pid)
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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['all','report','resume-after-cpu-repair']);p.add_argument('--wait-pid',type=int);a=p.parse_args()
 if a.stage=='all':all_stages()
 elif a.stage=='resume-after-cpu-repair':resume_after_cpu_repair(a.wait_pid)
 else:report('recovered_completed')
if __name__=='__main__':main()
