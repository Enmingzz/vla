#!/usr/bin/env bash
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$FREQUENCY_PROJECT"
: "${RUN_RESULTS:?Set the completed continuation result directory}"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  "$FREQUENCY_PROJECT/scripts/check_continuation.py" --results-dir "$RUN_RESULTS"
