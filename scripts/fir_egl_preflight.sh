#!/usr/bin/env bash
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$FREQUENCY_PROJECT"
: "${RUN_RESULTS:?Set a fresh EGL training archive}"
export OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
# Fail before expensive imports if the compute node cannot read scratch binaries.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" - <<'PY'
import base64, hashlib, importlib.metadata, json, os, socket
from pathlib import Path
package=importlib.metadata.distribution('scipy')
verified=[]
for entry in package.files:
    if str(entry).startswith('scipy/special/') and str(entry).endswith('.so'):
        actual=base64.urlsafe_b64encode(hashlib.sha256(package.locate_file(entry).read_bytes()).digest()).decode().rstrip('=')
        if entry.hash.mode!='sha256' or entry.hash.value!=actual:
            raise ValueError('Installed SciPy binary differs from its distribution: '+str(entry))
        verified.append(str(entry))
root=Path(os.environ['RUN_RESULTS'])/'provenance'
root.mkdir(parents=True,exist_ok=True)
(root/'scratch_binary_check.json').write_text(json.dumps(dict(passed=True,node=socket.gethostname(),
    scipy_version=package.version,verified_binaries=verified),indent=2)+'\n')
print('Scratch dependency binary checks passed',flush=True)
PY
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u FREQUENCY_CONFIG -u MUJOCO_GL -u PYOPENGL_PLATFORM \
  "$LIBERO_VENV/bin/python" -m pytest -q tests
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  "$SERVER_VENV/bin/python" tests/check_opsd_jax.py
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.egl_pipeline audit --results-dir "$RUN_RESULTS"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - <<'PY'
import os
from pathlib import Path
from frequency_vla.logging_utils import file_digest, write_json
files = list(Path('src/frequency_vla').glob('*.py')) + [Path(p) for p in [
    'configs/egl_pipeline.yaml', 'configs/prediction50_egl.yaml', 'configs/opsd_egl_1500.yaml', 'scripts/fir_egl_pipeline.sh',
    'tests/check_parallel_osmesa.py', 'tests/check_opsd_jax.py', 'tests/test_egl_pipeline.py',
    'scripts/fir_egl_summary.sh', 'scripts/check_cuda_allocation.py', 'scripts/serve_policy.sh',
    'scripts/env.sh', 'scripts/fir_egl_preflight.sh']]
write_json(Path(os.environ['RUN_RESULTS']) / 'provenance/preflight_checks.json', dict(
    passed=True, cpu_job_id=os.environ.get('SLURM_JOB_ID'), unit_tests_passed=True,
    native_flow_and_mask_checks_passed=True, diagnostic_rollback_preserves_original_weights=True,
    checkpoint_roundtrip_and_resume_equal_next_update=True, training_evaluation_pool_audit_passed=True,
    sources={str(p):file_digest(p) for p in files}))
print('CPU preflight passed; the GPU job must still pass both EGL runtime pilots.', flush=True)
PY
