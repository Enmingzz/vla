#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/simpler_env.sh"
cd "$FREQUENCY_PROJECT"
: "${SV_RUN_ROOT:?Set a fresh archive path}"
mkdir -p "$SV_RUN_ROOT/logs"
export SV_RUN_ROOT="$(realpath "$SV_RUN_ROOT")"
[[ ! -e "$SV_RUN_ROOT/submission_sources.json" ]] || { echo 'Already submitted; choose a fresh archive.' >&2; exit 2; }
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SV_CLIENT_ENV/bin/python" - <<'PY'
import os
from pathlib import Path
from frequency_vla.logging_utils import write_json
from frequency_vla.simpler_protocol import source_manifest
write_json(Path(os.environ['SV_RUN_ROOT'])/'submission_sources.json',source_manifest())
PY
DEPS=()
[[ -z "${SV_SETUP_JOB:-}" ]] || DEPS=(--dependency="afterok:$SV_SETUP_JOB" --kill-on-invalid-dep=yes)
sbatch --parsable --account="${SV_ACCOUNT:-def-btaati}" --gres=gpu:h100:1 \
  --cpus-per-task=4 --mem=48G --time="${SV_TIME:-00:30:00}" \
  --job-name="simpler_${SV_MODE:-smoke}" --output="$SV_RUN_ROOT/logs/job_%j.log" \
  "${DEPS[@]}" --export=ALL scripts/fir_simpler_job.sh | tee "$SV_RUN_ROOT/job.txt"
