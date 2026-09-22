#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/robocasa_env.sh"
cd "$FREQUENCY_PROJECT"
: "${RC_RUN_ROOT:?Choose a fresh results directory}"
[[ ! -e "$RC_RUN_ROOT/submission_sources.json" ]] || { echo 'Archive already submitted.' >&2; exit 2; }
mkdir -p "$RC_RUN_ROOT/logs"
export RC_RUN_ROOT="$(realpath "$RC_RUN_ROOT")"
"${RC_VENV}/bin/python" - "$RC_RUN_ROOT" <<'PY'
import os, sys
from pathlib import Path
from frequency_vla.logging_utils import file_digest, write_json, git_commit
from frequency_vla.robocasa_protocol import load_config
project=Path(os.environ['FREQUENCY_PROJECT'])
files=list((project/'src/frequency_vla').glob('robocasa_*.py'))
files += [project/'src/frequency_vla'/n for n in ['opsd_flow.py','opsd_protocol.py','logging_utils.py']]
files += [project/'scripts'/n for n in ['robocasa_env.sh','fir_robocasa_job.sh',
    'robocasa_render_check.py','check_cuda_allocation.py']]
files += [Path(os.environ['RC_CONFIG'])]
write_json(Path(sys.argv[1])/'submission_sources.json',{'git_commit':git_commit(project),
    'sources':{str(p):file_digest(p) for p in files},'config':load_config()})
PY
DEPS=()
[[ -z "${RC_SETUP_JOB:-}" ]] || DEPS=(--dependency="afterok:$RC_SETUP_JOB" --kill-on-invalid-dep=yes)
COMMON=(--parsable --account="${RC_ACCOUNT:-def-btaati}" --gres=gpu:h100:1 \
  --cpus-per-task=8 --mem=64G --exclude=fc10501,fc10511,fc10514 "${DEPS[@]}")
for mode in h5 h20 train500; do
  limit="${RC_EVAL_TIME:-01:00:00}"
  [[ "$mode" != train500 ]] || limit="${RC_TRAIN_TIME:-02:00:00}"
  job=$(sbatch "${COMMON[@]}" --job-name="rc365_$mode" --time="$limit" \
    --output="$RC_RUN_ROOT/logs/${mode}_%j.log" \
    --export="ALL,RC_MODE=$mode" scripts/fir_robocasa_job.sh)
  printf '%s %s\n' "$mode" "$job" | tee -a "$RC_RUN_ROOT/jobs.txt"
done
