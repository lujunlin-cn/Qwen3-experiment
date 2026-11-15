import json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,'/home/hajimi2025/Qwen3-experiment')
from utils import run_nvidia_final_synthetic as r
os.environ.update(NEMO_TELEMETRY_ENABLED='false',MPLBACKEND='Agg')
p=r.ROOT/'data/processed/round3/nvidia_synthetic_final_20261115_r1'
try:
 r.write_json_new(p/'live_activation.json',{'status':'started','authorized_by_user':True,'credential_source':'previously supplied user credential injected into child environment; no file persistence','started_at_unix':time.time(),'pid':os.getpid(),'maximum_requests':48,'completion_token_budget':500000})
 calibration=r.calibrate(p)
 print('CALIBRATION_COMPLETE',json.dumps(calibration['decisions']),flush=True)
 new=r.run_new_seeds(p)
 print('NEW_SEED_STAGE_COMPLETE',json.dumps(new['by_task']),flush=True)
 r.write_json_new(p/'live_completion.json',{'status':'completed','ended_at_unix':time.time(),'ledger':r.RequestBudget(p,r.config()).state()})
except BaseException as e:
 error=r.redact(traceback.format_exc())
 r.write_json_new(p/'live_failure.json',{'status':'failed','error':error,'ended_at_unix':time.time()})
 print(error,flush=True)
 raise SystemExit(1)
