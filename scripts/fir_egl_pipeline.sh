#!/usr/bin/env bash
# One continuous server: fresh official checkpoint, 1500 updates, EGL throughout.
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$FREQUENCY_PROJECT"
: "${RUN_RESULTS:?Set a fresh EGL training archive}"
: "${OPSD_CHECKPOINT_ROOT:?Set a fresh checkpoint directory}"
export OPSD_CONFIG="$FREQUENCY_PROJECT/configs/opsd_egl_1500.yaml"
export FREQUENCY_CONFIG="$FREQUENCY_PROJECT/configs/prediction50_egl.yaml"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
if [[ -e "$RUN_RESULTS/provenance/training_setup.json" || -e "$OPSD_CHECKPOINT_ROOT/diagnostic_trainable" ]]; then
  echo "Require a fresh run; preserve completed and interrupted archives." >&2
  exit 2
fi
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" \
  scripts/check_cuda_allocation.py --output "$RUN_RESULTS/provenance/cuda_allocation_check.json"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
from frequency_vla.logging_utils import file_digest
preflight=json.loads((Path(os.environ['RUN_RESULTS'])/'provenance/preflight_checks.json').read_text())
if not preflight['passed'] or any(file_digest(p)!=sha for p,sha in preflight['sources'].items()):
    raise ValueError('Implementation changed after CPU checks')
PY
mkdir -p "$RUN_RESULTS/logs"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total --format=csv > "$RUN_RESULTS/provenance/nvidia_smi.csv"
POLICY_PORT="${POLICY_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
bash scripts/serve_policy.sh --port "$POLICY_PORT" \
  --manifest-out "$RUN_RESULTS/provenance/startup_server.json" \
  --opsd-config "$OPSD_CONFIG" --opsd-results-dir "$RUN_RESULTS" \
  --opsd-checkpoint-root "$OPSD_CHECKPOINT_ROOT" \
  > "$RUN_RESULTS/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" - "$POLICY_PORT" "$SERVER_PID" <<'PY'
import os, sys, time, urllib.request
deadline=time.monotonic()+600
while time.monotonic()<deadline:
    try:
        os.kill(int(sys.argv[2]),0)
    except ProcessLookupError:
        raise SystemExit('Training server exited')
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/healthz',timeout=2) as response:
            if response.status==200:
                break
    except OSError:
        time.sleep(5)
else:
    raise SystemExit('Training server startup exceeded 600 seconds')
PY
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.egl_pipeline run --results-dir "$RUN_RESULTS" --port "$POLICY_PORT" \
  --checkpoint-root "$OPSD_CHECKPOINT_ROOT"
