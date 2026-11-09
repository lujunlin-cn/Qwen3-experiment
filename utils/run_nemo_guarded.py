"""Run the real NeMo smoke with an owned vLLM server and device memory guard.

Prepare a fresh synthetic run with run_synthetic_smoke.py before calling this.
The control interpreter needs Data Designer; the separate vLLM interpreter owns
CUDA. Neither historical runs nor unrelated GPU processes are replaced.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.course_run import guard_job
from utils.run_synthetic_smoke import (
    DEFAULT_CONFIG, local_endpoint_preflight, server_command, validated_config,
    verify_prepared_run,
)


def worker(args):
    config = validated_config(args.config)
    output = args.output.resolve()
    with (output / 'server.log').open('x') as log:
        server = subprocess.Popen(
            # Resolving this symlink would select the base Python and lose venv.
            server_command(config, args.vllm_python.absolute(), args.model.resolve()),
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 420
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'vLLM exited with {server.returncode}; see server.log')
                try:
                    local_endpoint_preflight(config['endpoint'], config['model_id'])
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('vLLM readiness deadline exceeded')
                    time.sleep(1)
            return subprocess.call([
                sys.executable, str(ROOT / 'utils/run_synthetic_smoke.py'), 'gpu',
                '--config', str(args.config.resolve()), '--run-id', args.run_id,
            ], cwd=ROOT)
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--vllm-python', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = validated_config(args.config)
    run = ROOT / f"data/processed/tasksets/{config['taskset']}/synthetic/{args.run_id}"
    verify_prepared_run(run)
    if args.worker:
        raise SystemExit(worker(args))
    # Fail before launching anything if the endpoint is already occupied.
    try:
        local_endpoint_preflight(config['endpoint'], config['model_id'])
    except OSError:
        pass
    else:
        raise RuntimeError('Endpoint is already in use; use a dedicated free endpoint')
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite guard output; use a fresh output directory')
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    guard = guard_job(command, args.output, args.gpu,
                      {'device_limit_bytes': 8_000_000_000, 'poll_seconds': .25},
                      time.time() + args.timeout_seconds)
    print(json.dumps(guard, ensure_ascii=False), flush=True)
    raise SystemExit(0 if guard['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
