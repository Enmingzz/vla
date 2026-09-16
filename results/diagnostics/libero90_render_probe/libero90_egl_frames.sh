#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50_libero90.yaml"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" .runtime/libero90_replay_evaluator.py --suite libero_90 --mode diagnostic --horizon 5 --episodes 1 --initial-state-start 0 --seed 37 --task-ids 0 --results-dir results/diagnostics/libero90_egl_frames
