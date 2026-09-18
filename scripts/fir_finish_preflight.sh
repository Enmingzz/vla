#!/usr/bin/env bash
# CPU-only checks before the final 22 updates and paired evaluation.
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$FREQUENCY_PROJECT"
: "${RUN_RESULTS:?Use a fresh completion archive}"
: "${PARENT_CHECKPOINT:?Set the step-978 checkpoint}"
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" -m pytest -q tests
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu "$SERVER_VENV/bin/python" tests/check_opsd_jax.py
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export FREQUENCY_OSMESA_LIBRARY_DIR="$FREQUENCY_WORK/native_osmesa/root/usr/lib64"
env -u PYTHONPATH -u PYTHONHOME LD_LIBRARY_PATH="$FREQUENCY_OSMESA_LIBRARY_DIR" "$LIBERO_VENV/bin/python" \
  tests/check_parallel_osmesa.py --output "$RUN_RESULTS/provenance/parallel_environment_check.json"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" -m frequency_vla.study_plan \
  --plan configs/autoresearch_round3_finish.yaml --training-config configs/opsd_continuation_1000.yaml \
  --parent-results results/opsd_h20_100 results/autoresearch_round2 results/autoresearch_round3 \
  --parent-checkpoint "$PARENT_CHECKPOINT" --output "$RUN_RESULTS/provenance/split_audit.json"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - <<'PY'
import os
from pathlib import Path
from frequency_vla.logging_utils import file_digest, write_json
files = list(Path('src/frequency_vla').glob('*.py')) + [Path('scripts/fir_study_job.sh'),
    Path('configs/autoresearch_round3_finish.yaml'), Path('configs/opsd_continuation_1000.yaml'),
    Path('configs/prediction50_round3.yaml')]
write_json(Path(os.environ['RUN_RESULTS']) / 'provenance/preflight_checks.json', dict(
    passed=True, cpu_job_id=os.environ.get('SLURM_JOB_ID'),
    unit_tests_passed=True, native_cpu_integration_passed=True,
    snapshot_load_preserves_optimizer_and_native_inference=True,
    partial_resume_equal_next_update=True, serial_parallel_observations_equal=True,
    initial_state_and_checkpoint_chain_audit_passed=True,
    sources={str(p):file_digest(p) for p in files}))
PY
