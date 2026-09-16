#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50_libero90.yaml"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
PROBE_DIR="$PWD/results/diagnostics/libero90_concurrent_replay"
mkdir -p "$PROBE_DIR/logs"
for setting in allocation busy; do
  export CUDA_PROBE_BUSY=0
  if [[ "$setting" == busy ]]; then export CUDA_PROBE_BUSY=1; fi
  "$LIBERO_VENV/bin/python" .runtime/cuda_memory_probe.py > "$PROBE_DIR/logs/cuda-$setting.log" 2>&1 &
  GPU_PID=$!
  trap 'kill "$GPU_PID" 2>/dev/null || true; wait "$GPU_PID" 2>/dev/null || true' EXIT
  for i in $(seq 1 30); do
    if rg -q CUDA_READY "$PROBE_DIR/logs/cuda-$setting.log"; then break; fi
    sleep 1
  done
  rg -q CUDA_READY "$PROBE_DIR/logs/cuda-$setting.log"
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH timeout --kill-after=5s 120 gdb --batch -ex 'set pagination off' -ex run -ex bt --args "$LIBERO_VENV/bin/python" .runtime/libero90_replay_evaluator.py --suite libero_90 --mode diagnostic --horizon 5 --episodes 1 --initial-state-start 0 --seed 37 --task-ids 0 --results-dir "$PROBE_DIR/$setting" > "$PROBE_DIR/logs/$setting.log" 2>&1 || true
  kill "$GPU_PID" 2>/dev/null || true
  wait "$GPU_PID" 2>/dev/null || true
 done
