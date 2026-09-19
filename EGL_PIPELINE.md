# Fresh training and evaluation under EGL

This run restarts from the official `pi05_libero` checkpoint and uses EGL for
every training rollout and evaluation. It does not resume historical weights.
Earlier steps 0–500 used EGL, while subsequent training used OSMesa and a
corrected multi-environment context selection. Those histories cannot isolate
the effect of extra updates or support timing comparisons across renderers.
The fresh run uses the corrected implementation from its first update.

The fixed configuration is [configs/egl_pipeline.yaml](configs/egl_pipeline.yaml).
All conditions share one H100, one continuously running policy server, four
isolated simulator workers, P=50 and 10 flow steps. This is the previously
authorized P=50 extension; the official configuration's native P is 10.

| Training steps | Evaluation H | Episodes |
|---:|---:|---:|
| 0 (Original) | 5 | 100 |
| 0 (Original) | 20 | 100 |
| 500 | 20 | 100 |
| 1000 | 20 | 100 |
| 1500 | 20 | 100 |

Training keeps student H=20, teacher H=5, batch size 4, learning rate 1e-5,
EMA decay 0.9999 and the existing temporal velocity-matching OPSD objective.
The vision/language backbone stays frozen. The training seed is 17 and the
initial-state pool is indices 10–19. Each 500-update segment starts simulator
trajectories using the same state-cycling rule; weights, EMA and Adam continue
through all segments. Added updates also add on-policy experience.

Every evaluation uses LIBERO-10, seed 27 and ordered initial-state indices
30–39: 10 tasks × 10 episodes. These layouts remain outside the training pool
but have been inspected in earlier experiments, so this is not a new blind
test. Milestones are fixed in advance and the student weights are evaluated.

## Run on Fir

Dependencies and the official checkpoint use the existing [setup](README.md)
and [training environment](TRAINING.md). With those installed:

```bash
cd /path/to/frequency_vla
source scripts/env.sh
# Both directories must be new. Override FREQUENCY_WORK before sourcing if needed.
export RUN_RESULTS="$PWD/results/egl_from_scratch_1500"
export OPSD_CHECKPOINT_ROOT="$FREQUENCY_WORK/runs/egl_from_scratch_1500"
bash scripts/submit_egl_pipeline.sh
```

The submitter schedules CPU checks, then one H100 for all training and
evaluation, then a CPU summary. GPU allocation ends as soon as the work finishes;
its wall-time cap is two hours. Accounts, excluded nodes and the cap can be
overridden with `EGL_CPU_ACCOUNT`, `EGL_GPU_ACCOUNT`, `EGL_EXCLUDED_NODES` and
`EGL_TIME_LIMIT`. `provenance/submission.json` records exact jobs and resources.
No GPU computation runs on a login node.

CPU checks cover unit tests, native sampling/loss masks, diagnostic rollback,
checkpoint roundtrip and disjoint initial-state pools. On the GPU, the pipeline
checks CUDA and exact EGL serial/parallel observations, runs a diagnostic update
that must roll back completely, and runs single-worker and four-worker pilots
before the formal conditions. Pilot success does not decide whether to proceed.
An EGL exception or timeout stops the pipeline; it never switches renderer.

## Outputs and analysis

The archive contains raw episodes, videos, training/rollout logs, saved-checkpoint
identities, source hashes and GPU accounting. `aggregated/conditions.csv` and
`FINDINGS.md` contain the complete table, including mean actions and wall time
for all episodes and successful episodes separately. The CPU report checks all
500 videos, checkpoint file hashes, exact native checkpoint roundtrips, and
paired initial states, first observations, inference RNG and evaluator settings.
All five conditions must finish before a complete report is produced.

```bash
source scripts/env.sh
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.egl_pipeline analyze --results-dir "$RUN_RESULTS"
```

Episode time includes reset, settling, execution, inference and shared-server
waiting at fixed four-worker concurrency; it excludes that episode's video
encoding. It is not isolated model latency. Successful-episode averages may
refer to different episode subsets. Whole-condition wall time is also saved.
Success intervals are Wilson 95% intervals; paired changes use stratified
bootstrap intervals and exact McNemar tests. Eight exploratory comparisons share
a Holm adjustment. The results are reported whether or not training helps.
