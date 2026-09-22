#!/usr/bin/env bash
# Evaluation only: reuse original/step-500 weights and valid completed episodes.
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/robocasa_env.sh"
cd "$FREQUENCY_PROJECT"
: "${RC_RECOVERY_ROOT:?Use a fresh recovery archive}"
: "${RC_RECOVERY_SOURCE:?Set the interrupted experiment archive}"
: "${RC_RECOVERY_CHECKPOINT:?Set the existing step-500 checkpoint}"
: "${RC_RECOVERY_MANIFEST:?Set its verified manifest digest}"
PY=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$RC_VENV/bin/python")
"${PY[@]}" - "$RC_RECOVERY_ROOT" <<'PY'
import json,sys
from pathlib import Path
from frequency_vla.logging_utils import file_digest
root=Path(sys.argv[1])
frozen=json.loads((root/'submission_sources.json').read_text())
assert all(file_digest(p)==sha for p,sha in frozen['sources'].items()), 'Recovery code changed since submission'
assert not (root/'server/provenance/training_setup.json').exists(), 'Use a fresh attempt'
PY
mkdir -p "$RC_RECOVERY_ROOT/logs" "$RC_RECOVERY_ROOT/provenance"
"${PY[@]}" scripts/check_cuda_allocation.py --output "$RC_RECOVERY_ROOT/provenance/cuda.json"
timeout --kill-after=10s 300s "${PY[@]}" scripts/robocasa_render_check.py \
  --output "$RC_RECOVERY_ROOT/provenance/egl.json" > "$RC_RECOVERY_ROOT/logs/egl.log" 2>&1
POLICY_PORT="$((20000 + SLURM_JOB_ID % 20000))"
# --train exposes the existing snapshot loading / phase switching API. The client
# sends only load_snapshot, set_phase and status; no rollout, update, save or resume.
"${PY[@]}" -m frequency_vla.robocasa_server --port "$POLICY_PORT" --train \
  --results-dir "$RC_RECOVERY_ROOT/server" --checkpoint-root "$RC_RECOVERY_ROOT/unused_checkpoints" \
  > "$RC_RECOVERY_ROOT/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
"${PY[@]}" - "$POLICY_PORT" "$SERVER_PID" <<'PY'
import os,sys,time,urllib.request
deadline=time.monotonic()+600
while time.monotonic()<deadline:
    os.kill(int(sys.argv[2]),0)
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/healthz',timeout=2) as response:
            if response.status==200: break
    except OSError: time.sleep(3)
else: raise SystemExit('Policy startup exceeded ten minutes')
PY
"${PY[@]}" -m frequency_vla.robocasa_recover --port "$POLICY_PORT" \
  --source "$RC_RECOVERY_SOURCE" --results-dir "$RC_RECOVERY_ROOT" \
  --checkpoint "$RC_RECOVERY_CHECKPOINT" --manifest-sha256 "$RC_RECOVERY_MANIFEST"
