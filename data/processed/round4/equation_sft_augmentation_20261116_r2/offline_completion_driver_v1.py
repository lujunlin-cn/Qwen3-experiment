import pathlib,time,subprocess,os,json
r=pathlib.Path('data/processed/round4/equation_sft_augmentation_20261116_r2');deadline=time.time()+4*3600
while not all((r/f'sft_{a}_validation/inference_result.json').exists() for a in 'ABC'):
 if time.time()>deadline:raise RuntimeError('Evaluation deadline; leave incomplete evidence and do not claim completion')
 time.sleep(15)
for module in ['utils.score_sft_augmentation','utils.report_sft_augmentation']:
 with (r/(module.rsplit('.',1)[-1]+'.log')).open('x') as f:
  run=subprocess.run(['.venv-low-memory/bin/python','-m',module],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'.','MPLBACKEND':'Agg','CUDA_VISIBLE_DEVICES':''})
 if run.returncode:raise RuntimeError(module+' failed')
print('FINAL_OFFLINE_REPORTS_COMPLETE',flush=True)
