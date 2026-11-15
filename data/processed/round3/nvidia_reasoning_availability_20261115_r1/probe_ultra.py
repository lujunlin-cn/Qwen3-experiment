"""Independent capability probe, not the registered 18-output experiment."""
import json
import os
from pathlib import Path
import time

from utils import run_nvidia_synthetic_smoke as n

os.environ['NEMO_TELEMETRY_ENABLED'] = 'false'
n.require_api_key()
out = Path(__file__).resolve().parent / 'ultra_probe'
out.mkdir(exist_ok=False)
config = n.validated_config()
config['models']['nvidia-reasoning']['model'] = 'nvidia/nemotron-3-ultra-550b-a55b'
seeds, hashes = n.check_seeds(config)
ns = n.load_notebook('ultra_availability_only', out / 'loader')
seed = seeds[0]
seed_file = out / 'public_seed.jsonl'
n.write_rows(seed_file, [{k: seed[k] for k in ('seed_id', 'task_type', 'prompt')}])
builder = ns['build_nvidia_data_designer'](seed_file, config, seed['task_type'], 'trace_variant')
n.write_new(out / 'compiled_config.json', builder.build().model_dump(mode='json'))
n.write_new(out / 'manifest.json', {'status': 'capability_probe', 'same_seed_hashes': hashes,
    'model': config['models']['nvidia-reasoning'], 'official_default_claim': False,
    'parameters': 'Keep the original Super experimental parameters; change only model ID',
    'not_formal_generation': True})
start = time.perf_counter()
result = {'status': 'failed', 'formal_generation': False}
with n.capture_requests(out / 'http_requests.jsonl'):
    try:
        outputs = n.sdk_generate(builder, out / 'sdk_probe')
        n.write_rows(out / 'sdk_outputs.jsonl', outputs)
        record = n.bind_raw_response({}, n.rows(out / 'http_requests.jsonl'), seed, ('trace_variant',))
        result['transport'] = record['trace_variant__transport']
        result['verification'] = ns['verify_nvidia_trace'](seed, record['trace_variant'])
        result['has_unverified_reasoning_content'] = bool(record['trace_variant__reasoning_content'])
        if record['trace_variant__transport']['finish_reason'] == 'stop' and result['verification'].get('schema_valid'):
            result['status'] = 'passed_api_schema_probe'
    except Exception as error:
        result['error'] = n.redact(type(error).__name__ + ': ' + str(error))
result['elapsed_seconds'] = time.perf_counter() - start
n.write_new(out / 'result.json', result)
print(json.dumps(n.redact(result), ensure_ascii=False, indent=2))
