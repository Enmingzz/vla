#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
# Use the CUDA compiler shipped by the frozen JAX dependencies, not an older site module.
FREQUENCY_CUDA_DIR="${FREQUENCY_CUDA_DIR:-$SERVER_VENV/lib/python3.11/site-packages/nvidia/cuda_nvcc}"
export PATH="$FREQUENCY_CUDA_DIR/bin:$PATH"
export CUDA_HOME="$FREQUENCY_CUDA_DIR"
export CUDA_ROOT="$FREQUENCY_CUDA_DIR"
export CUDA_PATH="$FREQUENCY_CUDA_DIR"
if [[ "${XLA_FLAGS:-}" != *xla_gpu_cuda_data_dir* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$FREQUENCY_CUDA_DIR"
fi
mkdir -p "$FREQUENCY_PROJECT/results/logs"
exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$SERVER_VENV/bin/python" -m frequency_vla.server \
  --manifest-out "$FREQUENCY_PROJECT/results/logs/server-manifest.json" "$@"
