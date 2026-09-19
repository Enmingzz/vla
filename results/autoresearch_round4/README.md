# Step 1000 to 1500: another 500 temporal OPSD updates

Status: **completed and verified**. GPU job **60480559** reached step 1500 and
completed all 100 new evaluations; CPU report job **60480560** passed. One H100
was allocated for **1h23m24s** and has been released. The 500-update training
stage took 49m28s; the new 100-episode evaluation took 25m15s, with the remaining
time spent loading, checking, compiling and saving.

At P=50, H=20, success rose from **78/100 at step 1000 to 90/100 at step 1500**:
**+12 percentage points**, paired 95% within-task bootstrap CI **+4 to +20 pp**,
exact McNemar **p=0.01690**, with 17 recoveries and 5 regressions. All 100
initial-state/first-observation/RNG/evaluator pairs match. The mean actions on
successful episodes stayed at 281.81 (rounded) for both checkpoints. Among the
73 episodes both solve, mean actions were 277.62 and 276.47 respectively.
These are previously inspected layouts excluded from training, so the result
supports improvement on this evaluation set without establishing broader transfer.

See [FINDINGS.md](FINDINGS.md), [condition summaries](aggregated/conditions.csv),
[paired comparison](aggregated/comparisons.csv), [per-task results](aggregated/per_task.csv),
and [the figure](figures/continuation_comparison.png). All 200 compared episode
videos and all exported checkpoint file checksums passed verification. Only the
100 step-1500 episodes consumed GPU time in this allocation; the 100 reference
episodes were reused with pinned checksums.

The user requested another 500 updates and evaluation. Resume the verified full
step-1000 checkpoint (FP32 action-expert parameters, EMA teacher, Adam moments and
counters), stop at the preselected step 1500, save it, and evaluate the student.
Simulator trajectories restart on resume. No optimizer or model settings change.

- P=50; student H=20; teacher H=5; 10 native flow steps.
- Batch 4; learning rate 1e-5; EMA 0.9999; frozen VLM/vision backbone.
- Training: LIBERO-10, official initial-state indices 10–19; seed 17.
- Evaluation: LIBERO-10, 10 tasks × 10 episodes, seed 27, indices 30–39.
- Reuse all 100 verified OSMesa step-1000 episodes from
  `results/autoresearch_round3_finish_retry1`, pinned before new training.
- Run only the 100 new step-1500 episodes. Compare matched outcomes, all-episode
  actions/calls, success-only means, and means on episodes both policies solve.
- These evaluation layouts have been inspected previously. This is exploratory
  continuation on layouts excluded from training, not a fresh blind test.
- Separate allocations mean timing differences are descriptive, not controlled
  speed measurements. Initial-state, first-observation, RNG, evaluator and core
  inference settings must still match exactly.

Resources: one H100 concurrently, 12 CPUs, 64 GiB, 1h40 maximum. Estimated actual
GPU time is 85–90 minutes, based on previous training and evaluation timings;
queue delay is additional and unknown. GPU exits immediately after evaluation.
Preflight and final analysis run on CPU-only allocations. Exclude fc10501 because
its previous allocation failed the CUDA initialization check.

The frozen plan is `configs/autoresearch_round4.yaml`; training config is
`configs/opsd_continuation_1500.yaml`. The checkpoint is saved under
`$FREQUENCY_WORK/runs/autoresearch_round4/step_1500`.

The automatically dependent CPU report will create `FINDINGS.md`, aggregated
CSVs, confidence intervals, paired exact McNemar statistics and plots. It also
checks video frame counts, all checkpoint checksums and Slurm GPU accounting.
Reference videos remain in the original archive and are linked here.

To analyze a completed run again:

```bash
source scripts/env.sh
export RUN_RESULTS="$FREQUENCY_PROJECT/results/autoresearch_round4"
bash scripts/fir_continuation_summary.sh
```

For a fresh reproduction on this configured Fir environment (run from the
repository root; choose unused result and checkpoint directories):

```bash
source scripts/env.sh
export RUN_RESULTS="$FREQUENCY_PROJECT/results/autoresearch_round4_reproduction"
export PARENT_CHECKPOINT="$FREQUENCY_WORK/runs/autoresearch_round3_finish_retry1/step_1000"
export OPSD_CHECKPOINT_ROOT="$FREQUENCY_WORK/runs/autoresearch_round4_reproduction"
export FREQUENCY_CONFIG="$FREQUENCY_PROJECT/configs/prediction50_round3.yaml"
export OPSD_CONFIG="$FREQUENCY_PROJECT/configs/opsd_continuation_1500.yaml"
export STUDY_PLAN="$FREQUENCY_PROJECT/configs/autoresearch_round4.yaml"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset BASELINE_CHECKPOINT
mkdir -p "$RUN_RESULTS/logs" "$RUN_RESULTS/provenance"
CHECK_JOB=$(sbatch --parsable --account=def-btaati --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=12G --time=00:15:00 \
  --output="$RUN_RESULTS/logs/preflight_%j.log" scripts/fir_round4_preflight.sh)
TRAIN_JOB=$(sbatch --parsable --account=rrg-btaati --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=12 --mem=64G --time=01:40:00 \
  --exclude=fc10501 --dependency="afterok:$CHECK_JOB" --kill-on-invalid-dep=yes \
  --output="$RUN_RESULTS/logs/gpu_%j.log" scripts/fir_gpu_checked_study.sh)
export TRAIN_JOB
python - <<'SUBMISSION'
import json, os
from pathlib import Path
(Path(os.environ['RUN_RESULTS']) / 'provenance/submission.json').write_text(
    json.dumps(dict(job_id=os.environ['TRAIN_JOB']), indent=2))
SUBMISSION
sbatch --account=def-btaati --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
  --time=00:10:00 --dependency="afterok:$TRAIN_JOB" --kill-on-invalid-dep=yes \
  --output="$RUN_RESULTS/logs/summary_%j.log" scripts/fir_continuation_summary.sh
```

The first CPU preflight detected test-fixture environment contamination and
stopped before any GPU allocation. Clearing evaluation overrides in the unit-test
subprocess fixed it; the failed CPU log and original submission remain archived.
The replacement CPU job passed all 59 tests, native FP32/EMA/Adam restoration and
next-update equivalence checks, and the complete state-pool/provenance audit.
Its GPU dependency is now satisfied.
