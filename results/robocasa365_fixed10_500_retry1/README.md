# RoboCasa365 π0.5: fixed ten tasks, 500 OPSD updates

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
`FINDINGS.md` and `aggregated/` report only completed conditions. Videos,
checkpoints and raw training arrays remain on Fir scratch.

The earlier archive `robocasa365_fixed10_500` was cancelled before GPU
allocation while correcting an overly strict CPU normalization-shape assertion.
It consumed no GPU allocation and contains no measured policy results.

Live status:

```bash
squeue -j 60905909,60905913,60905918
```
