#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/robocasa_env.sh"
cd "$FREQUENCY_PROJECT"
: "${RC_RECOVERY_ROOT:?Use a fresh output archive}"
: "${RC_RECOVERY_SOURCE:?Set the interrupted archive}"
: "${RC_RECOVERY_CHECKPOINT:?Set the saved step-500 checkpoint}"
: "${RC_RECOVERY_MANIFEST:?Set its verified manifest digest}"
[[ ! -e "$RC_RECOVERY_ROOT/submission_sources.json" ]] || { echo 'Recovery archive already submitted.' >&2; exit 2; }
mkdir -p "$RC_RECOVERY_ROOT/logs"
export RC_RECOVERY_ROOT="$(realpath "$RC_RECOVERY_ROOT")"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$RC_VENV/bin/python" - <<'PY'
import json,os
from pathlib import Path
from frequency_vla.logging_utils import digest,file_digest,write_json,git_commit
from frequency_vla.robocasa_protocol import load_config
project=Path(os.environ['FREQUENCY_PROJECT'])
checkpoint=Path(os.environ['RC_RECOVERY_CHECKPOINT'])
manifest=json.loads((checkpoint/'training_manifest.json').read_text())
assert manifest['step']==500 and digest(manifest)==os.environ['RC_RECOVERY_MANIFEST']
files=list((project/'src/frequency_vla').glob('robocasa_*.py'))
files += [project/'src/frequency_vla'/n for n in ['opsd_flow.py','opsd_protocol.py','logging_utils.py']]
files += [project/'scripts'/n for n in ['robocasa_env.sh','fir_robocasa_recover.sh',
    'robocasa_render_check.py','check_cuda_allocation.py']]
files += [Path(os.environ['RC_CONFIG'])]
root=Path(os.environ['RC_RECOVERY_ROOT'])
write_json(root/'submission_sources.json',{'git_commit':git_commit(project),
    'sources':{str(p):file_digest(p) for p in files},'config':load_config(),
    'source_archive':os.environ['RC_RECOVERY_SOURCE'],'snapshot':str(checkpoint),
    'manifest_sha256':os.environ['RC_RECOVERY_MANIFEST'],'additional_training_steps':0})
PY
job=$(sbatch --parsable --account="${RC_ACCOUNT:-def-btaati}" --gres=gpu:h100:1 \
  --cpus-per-task=8 --mem=64G --exclude=fc10501,fc10511,fc10514 \
  --job-name=rc365_finish_eval --time="${RC_RECOVERY_TIME:-01:15:00}" \
  --output="$RC_RECOVERY_ROOT/logs/recovery_%j.log" scripts/fir_robocasa_recover.sh)
printf '%s\n' "$job" | tee "$RC_RECOVERY_ROOT/job.txt"
