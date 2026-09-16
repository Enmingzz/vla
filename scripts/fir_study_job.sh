#!/usr/bin/env bash
# One GPU, immutable plan, isolated outputs, timeout and fail-fast cleanup.
set -euo pipefail
FREQUENCY_PROJECT="${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$FREQUENCY_PROJECT/scripts/env.sh"
: "${RUN_RESULTS:?Use a fresh result directory}"
: "${OPSD_CHECKPOINT_ROOT:?Use a fresh scratch checkpoint directory}"
: "${PARENT_CHECKPOINT:?Set the completed step-100 checkpoint}"
export FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-$FREQUENCY_PROJECT/configs/prediction50_round2.yaml}"
OPSD_CONFIG="$FREQUENCY_PROJECT/configs/opsd_continuation_500.yaml"
STUDY_PLAN="$FREQUENCY_PROJECT/configs/autoresearch_round2.yaml"
if [[ -e "$RUN_RESULTS/provenance/training_setup.json" || ! -e "$RUN_RESULTS/provenance/split_audit.json" ]]; then
  echo "Require a fresh run and completed CPU initial-state audit." >&2
  exit 2
fi
POLICY_PORT="${POLICY_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
export PYTHONUNBUFFERED=1 MUJOCO_EGL_DEVICE_ID=0
mkdir -p "$RUN_RESULTS/logs"
nvidia-smi
bash "$FREQUENCY_PROJECT/scripts/serve_policy.sh" --port "$POLICY_PORT" \
  --manifest-out "$RUN_RESULTS/provenance/startup_server.json" \
  --opsd-config "$OPSD_CONFIG" --opsd-results-dir "$RUN_RESULTS" \
  --opsd-checkpoint-root "$OPSD_CHECKPOINT_ROOT" > "$RUN_RESULTS/logs/server.log" 2>&1 &
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
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.study_runner --plan "$STUDY_PLAN" --training-config "$OPSD_CONFIG" \
  --parent-checkpoint "$PARENT_CHECKPOINT" --port "$POLICY_PORT" \
  --results-dir "$RUN_RESULTS" --checkpoint-root "$OPSD_CHECKPOINT_ROOT"
