#!/usr/bin/env bash
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/robocasa_env.sh"
cd "$FREQUENCY_PROJECT"
: "${RC_REPLAY_CHECK:?Set a fresh diagnostic output directory}"
: "${RC_REPLAY_CATALOG:?Set the existing immutable episode catalog}"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$RC_VENV/bin/python" \
  scripts/check_cuda_allocation.py --output "$RC_REPLAY_CHECK/cuda.json"
timeout --kill-after=15s 900s env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$RC_VENV/bin/python" scripts/check_robocasa_replay.py \
  --catalog "$RC_REPLAY_CATALOG" --output "$RC_REPLAY_CHECK"
