#!/usr/bin/env bash
# Validate the fixed 1000 -> 1500 continuation without allocating a GPU.
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$FREQUENCY_PROJECT"
: "${RUN_RESULTS:?Use a fresh round-four archive}"
: "${PARENT_CHECKPOINT:?Set the verified step-1000 checkpoint}"
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u FREQUENCY_CONFIG -u MUJOCO_GL -u PYOPENGL_PLATFORM \
  "$LIBERO_VENV/bin/python" -m pytest -q tests
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu "$SERVER_VENV/bin/python" tests/check_opsd_jax.py
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" -m frequency_vla.study_plan \
  --plan configs/autoresearch_round4.yaml --training-config configs/opsd_continuation_1500.yaml \
  --parent-results results/opsd_h20_100 results/autoresearch_round2 results/autoresearch_round3 results/autoresearch_round3_finish_retry1 \
  --parent-checkpoint "$PARENT_CHECKPOINT" --output "$RUN_RESULTS/provenance/split_audit.json"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path
from frequency_vla.cached_reference import prepare_reference, verify_reference
from frequency_vla.config import load_config
from frequency_vla.logging_utils import file_digest, write_json
root = Path(os.environ['RUN_RESULTS'])
plan = load_config('configs/autoresearch_round4.yaml')
source = Path(plan['cached_reference']['archive']) / 'provenance/parallel_environment_check.json'
check = json.loads(source.read_text())
if not check['passed'] or any(file_digest(p) != sha for p, sha in check['source_sha256'].items()):
    raise ValueError('The simulator implementation changed; rerun the CPU rendering comparison')
inference = load_config('configs/prediction50_round3.yaml')
if check['renderer']['library_sha256'] != inference['osmesa_library_sha256']:
    raise ValueError('The pinned renderer differs from the CPU-validated renderer')
write_json(root / 'provenance/parallel_environment_check.json', check)
write_json(root / 'provenance/reused_rendering_check.json', dict(source=str(source), sha256=file_digest(source),
    reason='Unchanged simulator/client code and pinned OSMesa library; exact serial/parallel check already passed.'))
prepare_reference(root, plan)
verify_reference(root, plan)
files = list(Path('src/frequency_vla').glob('*.py')) + [Path(p) for p in [
    'scripts/fir_study_job.sh', 'scripts/fir_gpu_checked_study.sh', 'scripts/check_cuda_allocation.py',
    'scripts/check_continuation.py', 'scripts/fir_round4_preflight.sh',
    'configs/autoresearch_round4.yaml', 'configs/opsd_continuation_1500.yaml', 'configs/prediction50_round3.yaml']]
write_json(root / 'provenance/preflight_checks.json', dict(passed=True, cpu_job_id=os.environ.get('SLURM_JOB_ID'),
    unit_tests_passed=True, native_cpu_integration_passed=True,
    partial_resume_equal_next_update=True, snapshot_load_preserves_optimizer_and_native_inference=True,
    initial_state_and_checkpoint_chain_audit_passed=True, cached_reference_checks_passed=True,
    serial_parallel_rendering_check_reused=True, sources={str(p):file_digest(p) for p in files}))
print('CPU checks passed; cached reference pinned; ready for 500 new updates and 100 new episodes.', flush=True)
PY
