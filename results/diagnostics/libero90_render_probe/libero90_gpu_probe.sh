#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader
exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH gdb --batch -ex 'set pagination off' -ex run -ex bt -ex 'info sharedlibrary' --args "$LIBERO_VENV/bin/python" .runtime/libero90_render_probe.py --reserve-gib 52 --action-scale 1 --steps 800
