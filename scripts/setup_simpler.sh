#!/usr/bin/env bash
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "$0")/.." && pwd)}}/scripts/simpler_env.sh"
mkdir -p "$SV_WORK/deps" "$SV_WORK/provenance"
checkout() {
  local url="$1" destination="$2" revision="$3"
  if [[ ! -d "$destination/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$url" "$destination"
    git -C "$destination" checkout --detach "$revision"
  fi
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || {
    echo "Unexpected revision: $destination" >&2; exit 2;
  }
  [[ -z "$(git -C "$destination" diff --name-only HEAD)" ]]
}
checkout https://github.com/simpler-env/SimplerEnv.git "$SV_SIMPLER" 06accaca93535902d408da4855f21cece12bceb7
git -C "$SV_SIMPLER" submodule update --init --recursive --depth 1
checkout https://github.com/Physical-Intelligence/openpi.git "$SV_OPENPI" 215abfb217dbac7d5f1273282331b9b1866c0479
[[ -x "$SV_CLIENT_ENV/bin/python" ]] || "$UV_BIN" venv --python 3.11.13 --managed-python "$SV_CLIENT_ENV"
[[ -x "$SV_SERVER_ENV/bin/python" ]] || "$UV_BIN" venv --python 3.11.13 --managed-python "$SV_SERVER_ENV"
"$UV_BIN" pip install --python "$SV_CLIENT_ENV/bin/python" \
  -c "$FREQUENCY_PROJECT/configs/simpler.constraints.txt" \
  -e "$SV_SIMPLER/ManiSkill2_real2sim" -e "$SV_SIMPLER" \
  -e "$SV_OPENPI/packages/openpi-client" -e "$FREQUENCY_PROJECT[test]" \
  matplotlib==3.7.5 setuptools==78.1.0
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$SV_SERVER_ENV" UV_LINK_MODE=hardlink \
  "$UV_BIN" sync --project "$SV_OPENPI" --frozen --no-dev --python "$SV_SERVER_ENV/bin/python"
"$UV_BIN" pip install --python "$SV_SERVER_ENV/bin/python" -e "$FREQUENCY_PROJECT" google-crc32c==1.7.1
# Install the official PyTorch compatibility files. Unlink first: cached wheels
# can be hardlinks shared with other virtual environments.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SV_SERVER_ENV/bin/python" - <<'PY'
import os, pathlib, shutil, sysconfig
src=pathlib.Path(os.environ['SV_OPENPI'])/'src/openpi/models_pytorch/transformers_replace'
dst=pathlib.Path(sysconfig.get_paths()['purelib'])/'transformers'
for p in src.rglob('*.py'):
    q=dst/p.relative_to(src)
    q.parent.mkdir(parents=True, exist_ok=True)
    q.unlink(missing_ok=True)
    shutil.copyfile(p,q)
PY
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SV_SERVER_ENV/bin/python" \
  "$FREQUENCY_PROJECT/scripts/download_simpler_checkpoint.py"
"$UV_BIN" pip freeze --python "$SV_CLIENT_ENV/bin/python" > "$SV_WORK/provenance/client.freeze.txt"
"$UV_BIN" pip freeze --python "$SV_SERVER_ENV/bin/python" > "$SV_WORK/provenance/server.freeze.txt"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SV_CLIENT_ENV/bin/python" - <<'PY'
import importlib.metadata as m
import simpler_env
print('SimplerEnv import OK;', len(simpler_env.ENVIRONMENTS), 'registered task names')
print({p:m.version(p) for p in ['numpy','sapien','gymnasium','opencv-python']})
PY
echo 'CPU setup complete. Rendering and model inference require an allocated GPU.'
