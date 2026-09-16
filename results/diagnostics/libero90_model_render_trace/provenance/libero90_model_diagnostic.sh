#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50_libero90.yaml"
export RUN_RESULTS="$PWD/results/diagnostics/libero90_model_render_trace"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MUJOCO_EGL_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1
mkdir -p "$RUN_RESULTS/logs"
PORT=$((20000 + SLURM_JOB_ID % 20000))
bash scripts/serve_policy.sh --port "$PORT" --comparison-checkpoint "$FREQUENCY_WORK/runs/autoresearch_round2_attempt2/step_500" --comparison-results-dir "$RUN_RESULTS" --manifest-out "$RUN_RESULTS/provenance/startup_server.json" > "$RUN_RESULTS/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" - "$PORT" "$SERVER_PID" <<'PY'
import os,sys,time,urllib.request
deadline=time.monotonic()+420
while time.monotonic()<deadline:
    os.kill(int(sys.argv[2]),0)
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/healthz',timeout=2) as response:
            if response.status==200: break
    except OSError: time.sleep(3)
else: raise SystemExit('Diagnostic startup timed out')
PY
export FREQUENCY_TRACE_FILE="$RUN_RESULTS/action_trace.jsonl"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH timeout --kill-after=10s 180 gdb --batch -ex 'set pagination off' -ex run -ex bt -ex 'info sharedlibrary' --args "$LIBERO_VENV/bin/python" .runtime/libero90_traced_evaluator.py --suite libero_90 --mode diagnostic --horizon 5 --episodes 1 --initial-state-start 0 --seed 37 --task-ids 0 --port "$PORT" --results-dir "$RUN_RESULTS/evaluation" > "$RUN_RESULTS/logs/gdb.log" 2>&1
