# RoboCasa365 π0.5: fixed ten tasks, 500 OPSD updates

Status after the submitted jobs exited:

| Condition | Completed episodes | Successes | Job outcome |
|---|---:|---:|---|
| Original H=5 | 60/100 | 31/60 | Aborted on reset pairing |
| Original H=20 | 100/100 | 58/100 | Complete |
| Step-500 H=20 | 33/100 | 6/33 | Aborted on reset pairing |
| Step-500 H=5 | 0/100 | — | Not reached |

All 500 optimizer updates completed and both step-100 and step-500 checkpoints
were saved before the training job's evaluation failed. On the same 33 episodes,
the original H=20 model achieved 18/33 versus step-500's 6/33. On the same 60
episodes, original H=20 achieved 38/60 versus H=5's 31/60. These interrupted,
task-order-dependent subsets are descriptive, not complete benchmark means.

GPU diagnostic 60911009 completed in 2m40s. All eight checked native seeded
resets reproduced the exact saved physical states, metadata and policy inputs.
Seven differed only in redundant OBJ MIME annotations in the exported XML.
Direct XML replay was rejected because it changed observations. Recovery retains
all numerical/state/image comparisons and records the narrow XML equivalence.
See the [recovery protocol](../../ROBOCASA365_OPSD_PLAN.md#interrupted-run-recovery).
No additional training is needed. The saved step-500 manifest is archived in
`train500/provenance/step_500_training_manifest.json`.

Submitted on 2026-09-21 after successful CPU configuration and file checks.
No completed rollout results are available at submission. The implementation
and exact protocol are in [the design document](../../ROBOCASA365_OPSD_PLAN.md).

| Condition | Slurm job | Planned work |
|---|---|---|
| Original H=5 | 60905909 | 10 tasks × 10 episodes = 100 |
| Original H=20 | 60905913 | Same paired 100 episodes |
| OPSD H=20 | 60905918 | 500 updates, then 100 H=20 and 100 H=5 episodes |

Each requests one H100 under `def-btaati_gpu`; evaluation caps are one hour,
and the training-plus-evaluation cap is two hours. Jobs are independent and
can run concurrently. All use EGL and the same original RoboCasa π0.5 checkpoint.
They do not wait for each other's scores. GPU renderer/model checks remain
mandatory at job startup; CPU tests cannot validate those.

`submission_sources.json` freezes exact source/config hashes. `submission.json`
records the budget. `provenance/` contains the successful CPU check, installed
packages, checkpoint object SHA256 manifest and simulator archive hashes.
The shared `paired_episodes` catalog is created atomically before rollout.
`FINDINGS.md` and `aggregated/summary.csv` distinguish completed conditions from
partial progress in `aggregated/progress.json`. Videos, checkpoints and raw
training arrays remain on Fir scratch.

The earlier archive `robocasa365_fixed10_500` was cancelled before GPU
allocation while correcting an overly strict CPU normalization-shape assertion.
It consumed no GPU allocation and contains no measured policy results.

Live status:

```bash
squeue -j 60905909,60905913,60905918
```
