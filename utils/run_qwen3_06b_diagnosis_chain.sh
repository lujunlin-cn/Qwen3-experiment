#!/usr/bin/env bash
# Run from the AutoDL checkout used by diagnose_qwen3_06b_sft.py.
# All stage logs and the terminal event are stored under the isolated run.
set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run=data/processed/round8/qwen3_06b_sft_diagnosis_20261121_r1
script=utils/diagnose_qwen3_06b_sft.py
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_ENABLE_V1_MULTIPROCESSING=0

mkdir -p "$run"
mkdir "$run/chain"
stage=prepare
log="$run/chain/$stage.log"
finish() {
    local status=$?
    trap - EXIT
    local state=failed
    if ((status == 0)); then state=completed; fi
    printf '{"status":"%s","stage":"%s","exit_code":%d,"log":"%s"}\n' \
        "$state" "$stage" "$status" "$log" > "$run/chain/terminal.json.tmp"
    mv "$run/chain/terminal.json.tmp" "$run/chain/terminal.json"
    cat "$run/chain/terminal.json"
    if ((status != 0)) && [[ -f "$log" ]]; then tail -n 35 -- "$log" >&2; fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

for stage in prepare compact full_gravity evaluate; do
    log="$run/chain/$stage.log"
    python=.venv/bin/python
    if [[ "$stage" == evaluate ]]; then python=.venv-vllm-cu128/bin/python; fi
    "$python" -u "$script" "$stage" > "$log" 2>&1
done
test -s "$run/evaluation_result.json"
