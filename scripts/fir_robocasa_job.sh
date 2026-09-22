#!/usr/bin/env bash
# One H100 per independent condition; downloads finish in a shared CPU job first.
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/robocasa_env.sh"
cd "$FREQUENCY_PROJECT"
: "${RC_RUN_ROOT:?Set a fresh experiment archive}"
: "${RC_MODE:?Use h5, h20 or train500}"
[[ "$RC_MODE" == h5 || "$RC_MODE" == h20 || "$RC_MODE" == train500 ]]
export RC_RESULTS="$RC_RUN_ROOT/$RC_MODE"
export RC_CATALOG="$RC_RUN_ROOT/paired_episodes"
export RC_CHECKPOINT_ROOT="$RC_WORK/runs/$(basename "$RC_RUN_ROOT")/$RC_MODE"
[[ ! -e "$RC_RESULTS/provenance/training_setup.json" ]] || { echo 'Use a fresh attempt directory.' >&2; exit 2; }
mkdir -p "$RC_RESULTS/provenance" "$RC_RESULTS/logs"
PY=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$RC_VENV/bin/python")
"${PY[@]}" - "$RC_RUN_ROOT" <<'PY'
import json, os, sys
from pathlib import Path
from frequency_vla.logging_utils import digest, file_digest
from frequency_vla.robocasa_protocol import load_config
root=Path(sys.argv[1])
frozen=json.loads((root/'submission_sources.json').read_text())
assert all(file_digest(p)==sha for p,sha in frozen['sources'].items()), 'Code changed since submission'
setup=json.loads((Path(os.environ['RC_WORK'])/'provenance/setup.json').read_text())
assert setup['passed'] and setup['config_sha256']==digest(load_config()), 'CPU setup is incomplete or config changed'
PY
"${PY[@]}" scripts/check_cuda_allocation.py --output "$RC_RESULTS/provenance/cuda.json"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "$RC_RESULTS/provenance/gpu.csv"
# Fail fast on an unusable EGL node, before the much larger VLA allocation.
timeout --kill-after=10s 300s "${PY[@]}" scripts/robocasa_render_check.py \
  --output "$RC_RESULTS/provenance/egl.json" > "$RC_RESULTS/logs/egl.log" 2>&1
POLICY_PORT="$((20000 + SLURM_JOB_ID % 20000))"
TRAIN_ARGS=()
[[ "$RC_MODE" != train500 ]] || TRAIN_ARGS=(--train)
"${PY[@]}" -m frequency_vla.robocasa_server --port "$POLICY_PORT" \
  --results-dir "$RC_RESULTS" --checkpoint-root "$RC_CHECKPOINT_ROOT" "${TRAIN_ARGS[@]}" \
  > "$RC_RESULTS/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
"${PY[@]}" - "$POLICY_PORT" "$SERVER_PID" <<'PY'
import os, sys, time, urllib.request
deadline=time.monotonic()+600
while time.monotonic()<deadline:
    os.kill(int(sys.argv[2]),0)
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/healthz',timeout=2) as response:
            if response.status==200: break
    except OSError: time.sleep(3)
else: raise SystemExit('Policy startup exceeded ten minutes')
PY
if [[ "$RC_MODE" == train500 ]]; then
  "${PY[@]}" -m frequency_vla.robocasa_train --port "$POLICY_PORT" \
    --results-dir "$RC_RESULTS" --rollouts-dir "$RC_CHECKPOINT_ROOT/rollouts" --catalog "$RC_CATALOG"
else
  "${PY[@]}" -m frequency_vla.robocasa_eval --horizon "${RC_MODE#h}" --port "$POLICY_PORT" \
    --results-dir "$RC_RESULTS" --catalog "$RC_CATALOG" --condition "original_$RC_MODE"
fi
"${PY[@]}" -m frequency_vla.robocasa_analysis --root "$RC_RUN_ROOT"
