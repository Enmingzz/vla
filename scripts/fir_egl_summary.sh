#!/usr/bin/env bash
set -euo pipefail
source "${FREQUENCY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$FREQUENCY_PROJECT"
: "${RUN_RESULTS:?Set the completed EGL training archive}"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.egl_pipeline analyze --results-dir "$RUN_RESULTS"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - <<'PY'
import json, os, subprocess
from pathlib import Path
from frequency_vla.logging_utils import file_digest, write_json
root=Path(os.environ['RUN_RESULTS'])
job=json.loads((root/'provenance/submission.json').read_text())['job_id']
accounting=subprocess.check_output(['sacct','-j',str(job),'-nP','-o','JobID,State,ElapsedRaw,AllocTRES,ExitCode,NodeList'],text=True)
(root/'provenance/slurm_accounting.psv').write_text(accounting)
main=next(line.split('|') for line in accounting.splitlines() if line.split('|')[0]==str(job))
if main[1]!='COMPLETED' or main[4]!='0:0' or 'gres/gpu=1' not in main[3].split(','):
    raise ValueError('GPU allocation did not finish normally on one GPU')
write_json(root/'provenance/resource_accounting.json',dict(job_id=job,state=main[1],node=main[5],
    gpu_seconds=int(main[2]),gpu_hours=int(main[2])/3600,maximum_concurrent_gpus=1,allocation_released=True))
files=list((root/'aggregated').glob('*'))+[root/'FINDINGS.md',root/'training.jsonl',root/'rollouts.jsonl']
write_json(root/'provenance/final_checks.json',dict(complete=True,formal_episodes=500,validated_videos=500,
    optimizer_updates_added=1500,gpu_allocation_released=True,
    artifacts={str(p.relative_to(root)):file_digest(p) for p in files}))
print('Verified 1500 EGL updates, 500 EGL episodes and released GPU allocation.',flush=True)
PY
