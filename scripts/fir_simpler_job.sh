#!/usr/bin/env bash
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/simpler_env.sh"
cd "$FREQUENCY_PROJECT"
: "${SV_RUN_ROOT:?Set a fresh experiment archive}"
mkdir -p "$SV_RUN_ROOT/provenance" "$SV_RUN_ROOT/logs"
SERVER=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SV_SERVER_ENV/bin/python")
CLIENT=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$SV_CLIENT_ENV/bin/python")
"${CLIENT[@]}" - "$SV_RUN_ROOT" <<'PY'
import json,os,sys
from pathlib import Path
from frequency_vla.logging_utils import git_commit
from frequency_vla.simpler_protocol import validate_frozen
validate_frozen(json.loads((Path(sys.argv[1])/'submission_sources.json').read_text()))
assert git_commit(os.environ['SV_SIMPLER']) == '06accaca93535902d408da4855f21cece12bceb7'
assert git_commit(Path(os.environ['SV_SIMPLER'])/'ManiSkill2_real2sim') == 'ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682'
PY
"${SERVER[@]}" scripts/check_cuda_allocation.py --output "$SV_RUN_ROOT/provenance/cuda.json"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv > "$SV_RUN_ROOT/provenance/gpu.csv"
timeout --kill-after=10s 240s "${CLIENT[@]}" scripts/simpler_render_check.py \
  --output "$SV_RUN_ROOT/provenance/vulkan.json" > "$SV_RUN_ROOT/logs/vulkan.log" 2>&1
if [[ "${SV_RENDER_ONLY:-0}" == 1 ]]; then exit 0; fi
POLICY_PORT="$((20000 + SLURM_JOB_ID % 20000))"
"${SERVER[@]}" -m frequency_vla.simpler_server --port "$POLICY_PORT" \
  --output "$SV_RUN_ROOT/provenance/server.json" > "$SV_RUN_ROOT/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
"${CLIENT[@]}" - "$POLICY_PORT" "$SERVER_PID" <<'PY'
import os,sys,time,urllib.request
deadline=time.monotonic()+480
while time.monotonic()<deadline:
    os.kill(int(sys.argv[2]),0)
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/healthz',timeout=2) as response:
            if response.status==200: break
    except OSError: time.sleep(2)
else: raise SystemExit('Server startup exceeded eight minutes')
PY
"${CLIENT[@]}" -m frequency_vla.simpler_eval --port "$POLICY_PORT" \
  --root "$SV_RUN_ROOT" --mode "${SV_MODE:-smoke}"
