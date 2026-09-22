#!/usr/bin/env bash
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "$0")/.." && pwd)}}/scripts/robocasa_env.sh"
mkdir -p "$RC_WORK/provenance" "$RC_WORK/deps"
if [[ "${RC_CHECK_ONLY:-0}" != 1 ]]; then
checkout() {
  local url="$1" destination="$2" revision="$3"
  if [[ ! -d "$destination/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "$url" "$destination"
    git -C "$destination" checkout --detach "$revision"
  fi
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]]
}
checkout https://github.com/robocasa-benchmark/openpi "$RC_OPENPI" 5a6beda9ff99da30b4e1b59320f6a32971d7c397
checkout https://github.com/robocasa/robocasa "$RC_CASA" 4f8a2980def75a55dff96b990745b83540425f09
checkout https://github.com/ARISE-Initiative/robosuite "$RC_SUITE" 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
[[ -x "$RC_VENV/bin/python" ]] || "$UV_BIN" venv --python 3.11.13 --managed-python "$RC_VENV"
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$RC_VENV" UV_LINK_MODE=hardlink \
  "$UV_BIN" sync --project "$RC_OPENPI" --frozen --no-dev --python "$RC_VENV/bin/python"
"$UV_BIN" pip install --python "$RC_VENV/bin/python" \
  numpy==2.2.5 scipy==1.15.3 numba==0.61.2 mujoco==3.3.1 \
  pygame==2.6.1 'qpsolvers[quadprog]==4.8.1' hidapi==0.14.0.post4 \
  polars==1.30.0 rich==14.0.0 imageio-ffmpeg==0.6.0 PyYAML==6.0.2 pytest==8.3.5
# OpenPI's frozen lock provides the rest, including JAX 0.5.3 and Flax 0.10.2.
# tianshou is used only by unrelated baselines; no offline demonstration data is needed.
"$UV_BIN" pip install --python "$RC_VENV/bin/python" --no-deps \
  -e "$RC_OPENPI" -e "$RC_OPENPI/packages/openpi-client" -e "$RC_CASA" -e "$RC_SUITE" -e "$FREQUENCY_PROJECT"
"$RC_VENV/bin/python" - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['RC_CASA']) / 'robocasa/macros_private.py'
if not p.exists():
    p.write_text('DATASET_BASE_PATH = ' + repr(str(Path(os.environ['RC_WORK']) / 'unused_datasets')) + '\n')
PY
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$RC_VENV/bin/python" \
  "$FREQUENCY_PROJECT/scripts/download_robocasa.py"
fi
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  "$RC_VENV/bin/python" -m frequency_vla.robocasa_protocol --check-install "$RC_WORK/provenance/setup.json"
"$UV_BIN" pip freeze --python "$RC_VENV/bin/python" > "$RC_WORK/provenance/environment.freeze.txt"
echo 'RoboCasa CPU setup and file checks complete; EGL/model checks run on allocated GPUs.'
