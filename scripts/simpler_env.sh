#!/usr/bin/env bash
export FREQUENCY_PROJECT="${FREQUENCY_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export SV_WORK="${SV_WORK:-${SCRATCH:-$HOME/scratch}/frequency_vla/simpler}"
export SV_SIMPLER="${SV_SIMPLER:-$SV_WORK/deps/SimplerEnv}"
export SV_OPENPI="${SV_OPENPI:-$SV_WORK/deps/openpi}"
export SV_CLIENT_ENV="${SV_CLIENT_ENV:-$SV_WORK/venvs/client}"
export SV_SERVER_ENV="${SV_SERVER_ENV:-$SV_WORK/venvs/server}"
export SV_CHECKPOINT="${SV_CHECKPOINT:-$SV_WORK/checkpoints/pi05_bridge}"
export SV_CONFIG="${SV_CONFIG:-$FREQUENCY_PROJECT/configs/simpler_frequency.yaml}"
export UV_BIN="${UV_BIN:-${SCRATCH:-$HOME/scratch}/frequency_vla/bootstrap/bin/uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-$HOME/scratch}/frequency_vla/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCRATCH:-$HOME/scratch}/frequency_vla/python}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${SCRATCH:-$HOME/scratch}/frequency_vla/cache/openpi}"
export HF_HOME="$SV_WORK/cache/huggingface"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg JAX_PLATFORMS=cpu
# SAPIEN uses Vulkan. These experiments do not use MuJoCo's EGL renderer.
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json}"
