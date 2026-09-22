#!/usr/bin/env bash
# Isolated from the archived LIBERO installations.
export FREQUENCY_PROJECT="${FREQUENCY_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export RC_WORK="${RC_WORK:-${SCRATCH:-$HOME/scratch}/frequency_vla/robocasa365}"
export RC_OPENPI="$RC_WORK/deps/openpi"
export RC_CASA="$RC_WORK/deps/robocasa"
export RC_SUITE="$RC_WORK/deps/robosuite"
export RC_VENV="$RC_WORK/venv"
export RC_CHECKPOINT="$RC_WORK/checkpoints/pi05_pretrain_human300/multitask_learning/75000"
export RC_CONFIG="${RC_CONFIG:-$FREQUENCY_PROJECT/configs/robocasa365_opsd.yaml}"
export UV_BIN="${UV_BIN:-${SCRATCH:-$HOME/scratch}/frequency_vla/bootstrap/bin/uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-$HOME/scratch}/frequency_vla/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCRATCH:-$HOME/scratch}/frequency_vla/python}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${SCRATCH:-$HOME/scratch}/frequency_vla/cache/openpi}"
export HF_HOME="$RC_WORK/cache/huggingface"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export WANDB_MODE=disabled
if [[ -d "$RC_VENV/lib/python3.11/site-packages/nvidia/cuda_nvcc" ]]; then
  export CUDA_HOME="$RC_VENV/lib/python3.11/site-packages/nvidia/cuda_nvcc"
  export CUDA_ROOT="$CUDA_HOME" CUDA_PATH="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
fi
