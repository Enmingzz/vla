#!/usr/bin/env bash
# Submit from the project directory. Resource/account choices belong to the sbatch command.
set -euo pipefail
FREQUENCY_PROJECT="${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$FREQUENCY_PROJECT/scripts/env.sh"
MODE="${1:-diagnostic}"
if [[ $# -gt 0 ]]; then shift; fi
POLICY_PORT="${POLICY_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
RUN_RESULTS="${RUN_RESULTS:-$FREQUENCY_PROJECT/results}"
mkdir -p "$RUN_RESULTS/logs"
export PYTHONUNBUFFERED=1
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
# Record rendering/device information for diagnosis.
nvidia-smi
ldconfig -p | rg 'libEGL|libGLX|libOSMesa' || true
bash "$FREQUENCY_PROJECT/scripts/serve_policy.sh" --port "$POLICY_PORT" \
  --manifest-out "$RUN_RESULTS/logs/server-${SLURM_JOB_ID:-manual}.json" \
  > "$RUN_RESULTS/logs/server-${SLURM_JOB_ID:-manual}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" - "$POLICY_PORT" "$SERVER_PID" <<'PY'
import os, sys, time, urllib.request
for _ in range(180):
    try:
        os.kill(int(sys.argv[2]), 0)
    except ProcessLookupError:
        raise SystemExit('Policy server exited; inspect its log')
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/healthz', timeout=2) as r:
            if r.status == 200:
                break
    except OSError:
        time.sleep(5)
else:
    raise SystemExit('Policy server did not become ready within 15 minutes')
PY
if [[ "$MODE" == diagnostic ]]; then
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
    -m frequency_vla.evaluator --mode diagnostic --horizon 5 --episodes 1 --task-ids 0 \
    --port "$POLICY_PORT" --results-dir "$RUN_RESULTS" "$@"
elif [[ "$MODE" == smoke ]]; then
  bash "$FREQUENCY_PROJECT/scripts/run_smoke_test.sh" --port "$POLICY_PORT" --results-dir "$RUN_RESULTS" "$@"
elif [[ "$MODE" == main ]]; then
  bash "$FREQUENCY_PROJECT/scripts/run_frequency_sweep.sh" --port "$POLICY_PORT" --results-dir "$RUN_RESULTS" "$@"
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi
