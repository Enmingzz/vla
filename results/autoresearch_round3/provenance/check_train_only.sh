#!/usr/bin/env bash
set -euo pipefail
cd /home/enmingzz/project/vla/frequency_vla
source scripts/env.sh
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" -m pytest -q tests
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu "$SERVER_VENV/bin/python" tests/check_opsd_jax.py
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export FREQUENCY_OSMESA_LIBRARY_DIR="$FREQUENCY_WORK/native_osmesa/root/usr/lib64"
env -u PYTHONPATH -u PYTHONHOME LD_LIBRARY_PATH="$FREQUENCY_OSMESA_LIBRARY_DIR" "$LIBERO_VENV/bin/python" \
  tests/check_parallel_osmesa.py --output results/autoresearch_round3/provenance/parallel_environment_check.json
