#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
export OPENBLAS_NUM_THREADS=1
hostname
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SERVER_VENV/bin/python" - <<'PY'
import base64, hashlib, importlib.metadata, importlib.util, json, os, sys
from pathlib import Path
package=importlib.metadata.distribution('scipy')
checked=[]
for entry in package.files:
    if str(entry).startswith('scipy/special/') and str(entry).endswith('.so'):
        path=package.locate_file(entry)
        actual=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
        if entry.hash.mode!='sha256' or entry.hash.value!=actual:
            raise ValueError('Installed SciPy binary does not match RECORD: '+str(path))
        checked.append(str(entry))
print(json.dumps(dict(python=sys.version,sys_path=sys.path,scipy_version=package.version,verified_binaries=checked)),flush=True)
import scipy.special
import scipy.optimize
print('SciPy special/optimize imports passed', scipy.special.loggamma(2.),flush=True)
PY
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu "$SERVER_VENV/bin/python" tests/check_opsd_jax.py
