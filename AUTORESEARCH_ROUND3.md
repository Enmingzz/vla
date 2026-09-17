# Fixed continuation from 500 to 1000 updates

The user requested another 500 updates after reviewing the step-500 findings.
The endpoint is fixed at step 1000. This run changes no model architecture,
attention mask, velocity loss, teacher-tail construction, optimizer settings,
batch size, flow integration steps, or teacher/student execution horizons.
The official source checkpoint and earlier result archives remain immutable.

Restore the step-500 FP32 student weights, EMA teacher and Adam state with
manifest `c00bcab9c776c6e3794b4b86dc18a15b0e78dd9e7fcd9fb0fa1c7f2d920f18ed`.
Run exactly updates 501–1000 at P=50, H_student=20, H_teacher=5, flow steps=10,
batch size=4, AdamW learning rate=1e-5, EMA decay=0.9999. Train LIBERO-10 only,
using the existing eligible start-state pool 10–19 and beginning at index 16.
The actual task/layout coverage will be reported from recorded rollouts.

Only two formal conditions are requested: step 500 and step 1000, both H=20,
10 tasks × 10 episodes, seed 27, official state indices 30–39. The primary
comparison is paired step-1000 minus step-500 success, with a within-task paired
bootstrap interval and exact McNemar test. Also report per-task changes, action
steps, policy calls and wall time. These previously inspected layouts are
excluded from training but are not a new blind test. No intermediate evaluation
selects the endpoint. No new benchmark or horizon sweep is included.

## Renderer and compute

Use the pinned OSMesa renderer after the documented native NVIDIA EGL failures.
Both snapshots are evaluated again on one continuous server with the same
renderer. The earlier 72% EGL step-500 score is historical context and is not
substituted for this run's measured baseline. This renderer change also affects
new training images, so this is not a pure optimizer-step ablation under an
otherwise identical runtime.

Four isolated CPU processes execute the existing four training environments
concurrently. Policy inference and each optimizer update remain synchronous.
The student still controls every action. Before requesting a GPU, a CPU-only
check compares every observation hash, task/state identity, executed action count
and next-block starting observation against serial execution, including resets.
This scheduling change prevents serial software rendering from wasting GPU time.
The first CPU comparison found identical physical states but different images
in serial slots 0–2 after switching contexts. The installed robosuite render
method does not call make_current before rendering. The training environment
now explicitly activates its own GL context before stepping/closing. The failed
CPU diagnostic is retained and the corrected serial/parallel images must match
exactly. The historical EGL training archives are not modified; the extent of
any historical image effect has not been established by this OSMesa diagnostic.
Only one H100 is allocated. Request 8 CPUs / 64 GiB / 2h30 maximum; release on
completion or failure. No automatic extra training follows either outcome.

## Reproduction

Use a fresh RUN_RESULTS and OPSD_CHECKPOINT_ROOT for each actual attempt.
The CPU environment check should run on a CPU compute node, not a GPU node.

```bash
source scripts/env.sh
export RUN_RESULTS="$PWD/results/autoresearch_round3"
export OPSD_CHECKPOINT_ROOT="$FREQUENCY_WORK/runs/autoresearch_round3"
export PARENT_CHECKPOINT="$FREQUENCY_WORK/runs/autoresearch_round2_attempt2/step_500"
export FREQUENCY_CONFIG="$PWD/configs/prediction50_round3.yaml"
export OPSD_CONFIG="$PWD/configs/opsd_continuation_1000.yaml"
export STUDY_PLAN="$PWD/configs/autoresearch_round3.yaml"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FREQUENCY_OSMESA_LIBRARY_DIR="$FREQUENCY_WORK/native_osmesa/root/usr/lib64"
mkdir -p "$RUN_RESULTS/logs" "$RUN_RESULTS/provenance"

env -u PYTHONPATH -u PYTHONHOME LD_LIBRARY_PATH="$FREQUENCY_OSMESA_LIBRARY_DIR" \
  "$LIBERO_VENV/bin/python" tests/check_parallel_osmesa.py \
  --output "$RUN_RESULTS/provenance/parallel_environment_check.json"

env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.study_plan --plan "$STUDY_PLAN" --training-config "$OPSD_CONFIG" \
  --parent-results results/opsd_h20_100 results/autoresearch_round2 \
  --parent-checkpoint "$PARENT_CHECKPOINT" --output "$RUN_RESULTS/provenance/split_audit.json"

sbatch --job-name=vla-opsd-1000 --account=rrg-btaati --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=64G --time=02:30:00 \
  --output="$RUN_RESULTS/logs/slurm-%j.log" scripts/fir_study_job.sh

env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.study_analysis --results-dir "$RUN_RESULTS"
```

The existing backend verifies the complete parent checkpoint, frozen backbone,
Adam counters, FP32 master state and native flow implementation on resume.
The exported step-1000 checkpoint must reproduce native inference exactly on a
save/reload probe. All new rollouts, teacher views and actions are archived;
formal test episodes save videos and initial-state/observation fingerprints.
