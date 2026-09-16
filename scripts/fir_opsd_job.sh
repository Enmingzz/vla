#!/usr/bin/env bash
# One allocation: bounded diagnostic, paired pre-eval, 100 updates, paired post-eval.
set -euo pipefail
FREQUENCY_PROJECT="${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$FREQUENCY_PROJECT/scripts/env.sh"
: "${RUN_RESULTS:?Use a fresh RUN_RESULTS directory}"
: "${OPSD_CHECKPOINT_ROOT:?Use a fresh OPSD_CHECKPOINT_ROOT on scratch}"
: "${GAP_RESULTS:?Set the completed paired frequency-result directory}"
if [[ -e "$RUN_RESULTS/provenance/training_setup.json" || -e "$RUN_RESULTS/training.jsonl" || -e "$OPSD_CHECKPOINT_ROOT/diagnostic_trainable" ]]; then
  echo "Refusing to overwrite an existing OPSD run; choose fresh paths." >&2
  exit 2
fi
export FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-$FREQUENCY_PROJECT/configs/prediction50_h15_h20.yaml}"
OPSD_CONFIG="${OPSD_CONFIG:-$FREQUENCY_PROJECT/configs/opsd_h20_100.yaml}"
POLICY_PORT="${POLICY_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
export PYTHONUNBUFFERED=1 MUJOCO_EGL_DEVICE_ID=0
mkdir -p "$RUN_RESULTS/logs"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - "$GAP_RESULTS" "$RUN_RESULTS" <<'PY'
import sys
from pathlib import Path
from frequency_vla.logging_utils import write_json
from frequency_vla.opsd_protocol import require_measured_gap
gate = require_measured_gap(sys.argv[1])
write_json(Path(sys.argv[2]) / 'provenance/hypothesis_gate.json', gate)
print(gate)
PY
nvidia-smi
bash "$FREQUENCY_PROJECT/scripts/serve_policy.sh" --port "$POLICY_PORT" \
  --manifest-out "$RUN_RESULTS/provenance/startup_server.json" \
  --opsd-config "$OPSD_CONFIG" --opsd-results-dir "$RUN_RESULTS" \
  --opsd-checkpoint-root "$OPSD_CHECKPOINT_ROOT" \
  > "$RUN_RESULTS/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" - "$POLICY_PORT" "$SERVER_PID" <<'PY'
import os, sys, time, urllib.request
deadline = time.monotonic() + 360
while time.monotonic() < deadline:
    try:
        os.kill(int(sys.argv[2]), 0)
    except ProcessLookupError:
        raise SystemExit('Training server exited')
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/healthz', timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(5)
else:
    raise SystemExit('Training server startup exceeded 360 seconds')
PY
run_stage() {
  local stage="$1" limit="$2"
  echo "Starting OPSD stage $stage; timeout ${limit}s"
  timeout --kill-after=10s "$limit" env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    "$LIBERO_VENV/bin/python" -m frequency_vla.opsd_client \
    --config "$OPSD_CONFIG" --port "$POLICY_PORT" --stage "$stage" \
    --results-dir "$RUN_RESULTS" --rollouts-dir "$OPSD_CHECKPOINT_ROOT/rollouts" \
    > "$RUN_RESULTS/logs/$stage.log" 2>&1
}
run_evaluation() {
  local phase="$1"
  echo "Evaluating H=20 $phase; timeout 1200s"
  timeout --kill-after=10s 1200 env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    "$LIBERO_VENV/bin/python" -m frequency_vla.opsd_evaluate --port "$POLICY_PORT" \
    --results-dir "$RUN_RESULTS/$phase" --workers 4 \
    > "$RUN_RESULTS/logs/evaluate-$phase.log" 2>&1
}
run_stage diagnostic 360
run_evaluation baseline
run_stage train 1200
run_stage save 180
run_stage student 60
run_evaluation student_100
echo "Completed the initial 100-update experiment; exiting to release the GPU."
