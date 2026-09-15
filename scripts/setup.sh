#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
export CC="${FREQUENCY_CC:-${CC:-cc}}"
export CXX="${FREQUENCY_CXX:-${CXX:-c++}}"
UV="${UV_BIN:-$FREQUENCY_WORK/bootstrap/bin/uv}"
mkdir -p "$FREQUENCY_WORK/bootstrap/bin" "$FREQUENCY_WORK/venvs" "$FREQUENCY_WORK/deps"
if [[ ! -x "$UV" ]]; then
  # Standalone uv avoids compiling packages against site-specific Python modules.
  python3 - "$UV" <<'PY'
import io, pathlib, sys, tarfile, urllib.request
target = pathlib.Path(sys.argv[1])
url = 'https://github.com/astral-sh/uv/releases/download/0.8.24/uv-x86_64-unknown-linux-gnu.tar.gz'
with urllib.request.urlopen(url, timeout=120) as r:
    data = r.read()
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
    target.write_bytes(tar.extractfile('uv-x86_64-unknown-linux-gnu/uv').read())
target.chmod(0o755)
PY
fi
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
if [[ ! -d "$OPENPI_DIR/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
  git -C "$OPENPI_DIR" checkout "$OPENPI_COMMIT"
fi
[[ "$(git -C "$OPENPI_DIR" rev-parse HEAD)" == "$OPENPI_COMMIT" ]] || { echo 'Unexpected OpenPI revision; refusing to reset an existing checkout.' >&2; exit 1; }
GIT_LFS_SKIP_SMUDGE=1 git -C "$OPENPI_DIR" submodule update --init --recursive
"$UV" python install 3.8.20 3.11.13
[[ -x "$SERVER_VENV/bin/python" ]] || "$UV" venv --python 3.11.13 --managed-python "$SERVER_VENV"
[[ -x "$LIBERO_VENV/bin/python" ]] || "$UV" venv --python 3.8.20 --managed-python "$LIBERO_VENV"
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$SERVER_VENV" UV_LINK_MODE=copy \
  "$UV" sync --project "$OPENPI_DIR" --frozen --no-dev --python "$SERVER_VENV/bin/python"
"$UV" pip install --python "$SERVER_VENV/bin/python" -e "$FREQUENCY_PROJECT" google-crc32c==1.7.1
"$UV" pip install --python "$LIBERO_VENV/bin/python" \
  -c "$FREQUENCY_PROJECT/configs/libero.constraints.txt" \
  -r "$OPENPI_DIR/examples/libero/requirements.txt" \
  -r "$OPENPI_DIR/third_party/libero/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
"$UV" pip install --python "$LIBERO_VENV/bin/python" \
  -c "$FREQUENCY_PROJECT/configs/libero.constraints.txt" \
  -e "$OPENPI_DIR/packages/openpi-client" -e "$OPENPI_DIR/third_party/libero" -e "$FREQUENCY_PROJECT[test]"
"$LIBERO_VENV/bin/python" - <<'PY'
import os, pathlib, yaml
root = pathlib.Path(os.environ['OPENPI_DIR']) / 'third_party/libero/libero/libero'
cfg = pathlib.Path(os.environ['LIBERO_CONFIG_PATH'])
cfg.mkdir(parents=True, exist_ok=True)
paths = {'benchmark_root': str(root), 'bddl_files': str(root/'bddl_files'),
         'init_states': str(root/'init_files'), 'assets': str(root/'assets'),
         'datasets': str(root.parent/'datasets')}
(root.parent/'datasets').mkdir(exist_ok=True)
(cfg/'config.yaml').write_text(yaml.safe_dump(paths))
PY
"$SERVER_VENV/bin/python" "$FREQUENCY_PROJECT/scripts/preflight.py" --inspect-only
if [[ "${SKIP_CHECKPOINT_DOWNLOAD:-0}" != 1 ]]; then
  "$SERVER_VENV/bin/python" "$FREQUENCY_PROJECT/scripts/download_checkpoint.py" --destination "$CHECKPOINT_DIR"
fi
"$UV" pip freeze --python "$SERVER_VENV/bin/python" > "$FREQUENCY_WORK/server.freeze.txt"
"$UV" pip freeze --python "$LIBERO_VENV/bin/python" > "$FREQUENCY_WORK/libero.freeze.txt"
echo "Setup complete. Source $FREQUENCY_PROJECT/scripts/env.sh before use."
echo 'The requested six-horizon sweep is blocked by native P=10; run preflight to see the constraint.'
