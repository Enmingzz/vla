#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" .runtime/libero90_render_probe.py
