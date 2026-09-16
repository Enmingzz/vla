#!/usr/bin/env bash
# One GPU; two frozen checkpoints; no optimizer or training.
set -euo pipefail
FREQUENCY_PROJECT="${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$FREQUENCY_PROJECT/scripts/env.sh"
: "${RUN_RESULTS:?Use a fresh result directory}"
: "${COMPARISON_CHECKPOINT:?Set the existing step-500 checkpoint}"
export FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-$FREQUENCY_PROJECT/configs/prediction50_libero90.yaml}"
if [[ -e "$RUN_RESULTS/provenance/frozen_comparison.json" || ! -e "$RUN_RESULTS/provenance/split_audit.json" ]]; then
  echo "Require a fresh comparison and completed CPU split audit." >&2
  exit 2
fi
POLICY_PORT="${POLICY_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
export PYTHONUNBUFFERED=1 MUJOCO_EGL_DEVICE_ID=0
mkdir -p "$RUN_RESULTS/logs"
nvidia-smi
bash "$FREQUENCY_PROJECT/scripts/serve_policy.sh" --port "$POLICY_PORT" \
  --manifest-out "$RUN_RESULTS/provenance/startup_server.json" \
  --comparison-checkpoint "$COMPARISON_CHECKPOINT" --comparison-results-dir "$RUN_RESULTS" \
  > "$RUN_RESULTS/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" - "$POLICY_PORT" "$SERVER_PID" <<'PY'
import os, sys, time, urllib.request
deadline = time.monotonic() + 420
while time.monotonic() < deadline:
    try:
        os.kill(int(sys.argv[2]), 0)
    except ProcessLookupError:
        raise SystemExit('Comparison server exited')
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/healthz', timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(5)
else:
    raise SystemExit('Comparison server startup exceeded 420 seconds')
PY
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.libero90_transfer run --plan "$FREQUENCY_PROJECT/configs/libero90_transfer.yaml" \
  --port "$POLICY_PORT" --results-dir "$RUN_RESULTS"
