#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50_libero90.yaml"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$FREQUENCY_WORK/native_osmesa/root/usr/lib64"
env -u PYTHONPATH -u PYTHONHOME "$LIBERO_VENV/bin/python" .runtime/libero90_render_probe.py
exec env -u PYTHONPATH -u PYTHONHOME "$LIBERO_VENV/bin/python" .runtime/libero90_replay_evaluator.py --suite libero_90 --mode diagnostic --horizon 5 --episodes 1 --initial-state-start 0 --seed 37 --task-ids 0 --results-dir results/diagnostics/libero90_osmesa_replay
