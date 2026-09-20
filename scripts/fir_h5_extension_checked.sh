#!/usr/bin/env bash
# Preserve the frozen evaluator; reject a stalled EGL node before loading weights.
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$FREQUENCY_PROJECT"
: "${RUN_RESULTS:?Set a fresh prepared extension archive}"
export FREQUENCY_CONFIG="$FREQUENCY_PROJECT/configs/prediction50_egl.yaml"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1

env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ["RUN_RESULTS"])
plan = json.loads((root / "provenance/launch_checks.json").read_text())
for path, expected in plan["source_files"].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise SystemExit("Launch-check source changed: " + path)
PY

echo "Checking serial/parallel EGL rendering before loading the model (180 second bound)"
timeout --kill-after=10s 180s env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$LIBERO_VENV/bin/python" tests/check_parallel_osmesa.py \
  --training-config configs/opsd_egl_1500.yaml \
  --inference-config "$FREQUENCY_CONFIG" \
  --output "$RUN_RESULTS/provenance/render_context_check.json" \
  > "$RUN_RESULTS/logs/render_context_check.log" 2>&1

env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - <<'PY'
import json, os, signal, subprocess, sys, time
from pathlib import Path
root = Path(os.environ["RUN_RESULTS"])
check = json.loads((root / "provenance/render_context_check.json").read_text())
if not check["passed"] or check["renderer"]["backend"] != "egl":
    raise SystemExit("EGL rendering preflight failed")
print("EGL preflight passed; starting the frozen 500-episode extension", flush=True)
process = subprocess.Popen(["bash", "scripts/fir_h5_extension.sh"], start_new_session=True)
try:
    while process.poll() is None:
        stages = root / "stages.jsonl"
        if stages.exists():
            # append_record writes a complete line; ignore a concurrently written tail.
            records = [json.loads(line) for line in stages.read_text().splitlines(keepends=True)
                       if line.endswith("\n") and line.strip()]
            if records and records[-1]["event"] == "start":
                stage = records[-1]
                files = root.glob("new_evaluations/states_*/evaluations/step_*/raw/smoke/libero_10/seed_27/H_5/*.jsonl")
                latest = max([stage["unix_time"]] + [p.stat().st_mtime for p in files])
                if time.time() - latest > 180:
                    failure = dict(passed=False, reason="No completed episode for 180 seconds",
                                   stage=stage["stage"], last_progress_unix=latest, unix_time=time.time())
                    (root / "provenance/progress_timeout.json").write_text(json.dumps(failure, indent=2)+"\n")
                    raise RuntimeError(failure["reason"] + ": " + stage["stage"])
        time.sleep(10)
    sys.exit(process.returncode)
finally:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
PY
