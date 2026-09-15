#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.runner --mode smoke "$@"
