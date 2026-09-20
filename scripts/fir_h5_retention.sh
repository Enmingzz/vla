#!/usr/bin/env bash
# Evaluation only: original and the completed EGL step-1500 student at H=5.
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$FREQUENCY_PROJECT"
: "${RUN_RESULTS:?Set the prepared, fresh evaluation archive}"
export FREQUENCY_CONFIG="$FREQUENCY_PROJECT/configs/prediction50_egl.yaml"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/evaluate_h5_retention.py verify --results-dir "$RUN_RESULTS"
if [[ -e "$RUN_RESULTS/provenance/frozen_comparison.json" ]]; then
  echo "Choose a fresh result directory; do not overwrite an earlier attempt." >&2
  exit 2
fi
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" \
  scripts/check_cuda_allocation.py --output "$RUN_RESULTS/provenance/cuda_allocation_check.json"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total --format=csv \
  > "$RUN_RESULTS/provenance/nvidia_smi.csv"
COMPARISON_CHECKPOINT=$(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$LIBERO_VENV/bin/python" -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["RUN_RESULTS"])/"provenance/study_plan.json").read_text())["checkpoint"]["path"])')
POLICY_PORT="${POLICY_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
bash scripts/serve_policy.sh --port "$POLICY_PORT" \
  --manifest-out "$RUN_RESULTS/provenance/startup_server.json" \
  --comparison-checkpoint "$COMPARISON_CHECKPOINT" --comparison-results-dir "$RUN_RESULTS" \
  > "$RUN_RESULTS/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - "$POLICY_PORT" "$SERVER_PID" <<'PY'
import os, sys, time, urllib.request
deadline=time.monotonic()+600
while time.monotonic()<deadline:
    try:
        os.kill(int(sys.argv[2]),0)
    except ProcessLookupError:
        raise SystemExit('Evaluation server exited')
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/healthz',timeout=2) as response:
            if response.status==200:
                break
    except OSError:
        time.sleep(5)
else:
    raise SystemExit('Evaluation server startup exceeded 600 seconds')
PY
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/evaluate_h5_retention.py run --results-dir "$RUN_RESULTS" --port "$POLICY_PORT"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/evaluate_h5_retention.py analyze --results-dir "$RUN_RESULTS"
