#!/usr/bin/env bash
# CPU-only checkpoint/provenance validation after the training allocation exits.
set -euo pipefail
FREQUENCY_PROJECT="${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$FREQUENCY_PROJECT/scripts/env.sh"
: "${RUN_RESULTS:?Set the training result directory}"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  "$FREQUENCY_PROJECT/scripts/check_training_only.py" --results-dir "$RUN_RESULTS"
