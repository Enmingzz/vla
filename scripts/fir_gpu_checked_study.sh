#!/usr/bin/env bash
# Cheap driver/device check before the unchanged, CPU-validated study script.
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$FREQUENCY_PROJECT"
: "${RUN_RESULTS:?Set the fresh study result directory}"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" \
  "$FREQUENCY_PROJECT/scripts/check_cuda_allocation.py" \
  --output "$RUN_RESULTS/provenance/cuda_allocation_check.json"
exec bash "$FREQUENCY_PROJECT/scripts/fir_study_job.sh"
