#!/usr/bin/env bash
# Submit CPU checks -> one H100 for all training/evaluation -> CPU report.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$FREQUENCY_PROJECT"
export RUN_RESULTS="${RUN_RESULTS:-$FREQUENCY_PROJECT/results/egl_from_scratch_1500}"
export OPSD_CHECKPOINT_ROOT="${OPSD_CHECKPOINT_ROOT:-$FREQUENCY_WORK/runs/egl_from_scratch_1500}"
export EGL_CPU_ACCOUNT="${EGL_CPU_ACCOUNT:-def-btaati}"
export EGL_GPU_ACCOUNT="${EGL_GPU_ACCOUNT:-rrg-btaati}"
export EGL_EXCLUDED_NODES="${EGL_EXCLUDED_NODES:-fc10501,fc10511}"
export EGL_CPU_EXCLUDED_NODES="${EGL_CPU_EXCLUDED_NODES:-fc30560}"
export EGL_TIME_LIMIT="${EGL_TIME_LIMIT:-02:00:00}"
if [[ -e "$RUN_RESULTS" || -e "$OPSD_CHECKPOINT_ROOT" ]]; then
  echo "Choose fresh RUN_RESULTS and OPSD_CHECKPOINT_ROOT paths; existing runs are preserved." >&2
  exit 2
fi
mkdir -p "$RUN_RESULTS/logs" "$RUN_RESULTS/provenance"
export EGL_PREFLIGHT_JOB
EGL_PREFLIGHT_JOB=$(sbatch --parsable --account="$EGL_CPU_ACCOUNT" --job-name=egl-check \
  --exclude="$EGL_CPU_EXCLUDED_NODES" \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=12G --time=00:15:00 \
  --output="$RUN_RESULTS/logs/preflight-%j.log" scripts/fir_egl_preflight.sh)
export EGL_GPU_JOB
EGL_GPU_JOB=$(sbatch --parsable --account="$EGL_GPU_ACCOUNT" --job-name=egl-train-eval \
  --dependency="afterok:$EGL_PREFLIGHT_JOB" --kill-on-invalid-dep=yes \
  --nodes=1 --ntasks=1 --gpus-per-node=h100:1 --cpus-per-task=8 --mem=64G \
  --exclude="$EGL_EXCLUDED_NODES" --time="$EGL_TIME_LIMIT" \
  --output="$RUN_RESULTS/logs/gpu-%j.log" scripts/fir_egl_pipeline.sh)
export EGL_SUMMARY_JOB
EGL_SUMMARY_JOB=$(sbatch --parsable --account="$EGL_CPU_ACCOUNT" --job-name=egl-report \
  --exclude="$EGL_CPU_EXCLUDED_NODES" \
  --dependency="afterok:$EGL_GPU_JOB" --kill-on-invalid-dep=yes \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:15:00 \
  --output="$RUN_RESULTS/logs/summary-%j.log" scripts/fir_egl_summary.sh)
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" - <<'PY'
import datetime, os
from pathlib import Path
from frequency_vla.logging_utils import write_json
write_json(Path(os.environ['RUN_RESULTS'])/'provenance/submission.json', dict(
    job_id=os.environ['EGL_GPU_JOB'], cpu_preflight_job=os.environ['EGL_PREFLIGHT_JOB'],
    cpu_summary_job=os.environ['EGL_SUMMARY_JOB'], account=os.environ['EGL_GPU_ACCOUNT'],
    gpus=1, cpus=8, memory_gib=64, time_limit=os.environ['EGL_TIME_LIMIT'],
    excluded_nodes=os.environ['EGL_EXCLUDED_NODES'].split(','),
    excluded_cpu_nodes=os.environ['EGL_CPU_EXCLUDED_NODES'].split(','),
    launch_script='scripts/fir_egl_pipeline.sh',
    checkpoint_root=os.environ['OPSD_CHECKPOINT_ROOT'],
    execution_mode='official_step0_train_to_1500_all_egl',
    new_optimizer_updates=1500, new_evaluation_episodes=500, reused_evaluation_episodes=0,
    submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
PY
printf 'CPU checks: %s\nOne H100, EGL training + evaluation: %s\nCPU report: %s\n' \
  "$EGL_PREFLIGHT_JOB" "$EGL_GPU_JOB" "$EGL_SUMMARY_JOB"
