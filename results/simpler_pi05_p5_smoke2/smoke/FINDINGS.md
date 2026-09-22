# SimplerEnv π0.5 replanning experiment

Mode: smoke. Same fixed third-party Bridge checkpoint; native P=5; flow steps=10.
All four official WidowX tasks, 5 Hz control, official prepackaged visual matching.
Primary endpoint: final simulator success at the official time limit (60/120 actions).
The upstream evaluator continues after transient success; any-hit success is recorded separately.

| H | Replanning Hz | Successes / episodes | Success rate (95% Wilson CI) | Mean policy calls | Rollout seconds |
|---|---|---|---|---|---|
| 1 | 5.00 | 1/8 | 12.5% (2.2%–47.1%) | 75.00 | 20.75 |
| 2 | 2.50 | 1/8 | 12.5% (2.2%–47.1%) | 37.50 | 10.46 |
| 5 | 1.00 | 0/8 | 0.0% (0.0%–32.4%) | 15.00 | 6.39 |

Primary H=1 minus H=5 gap: +12.50 percentage points; paired 95% bootstrap CI [+0.00, +25.00]. McNemar p=1; Holm-adjusted p=1.
H=5 saves 80.0% of policy calls per episode.

These measurements do not establish a statistically convincing positive frequency gap.
Probe/smoke results are functionality checks, not confirmatory hypothesis tests.

Per-task sensitivities and candidate H=2/H=5 gaps are in the aggregated CSVs.
No teacher/student pair is selected automatically and no OPSD training is started.

## Interpretation limits

Third-party Bridge-adapted pi0.5. The HF model card calls it RL, whereas ProphRL's release README calls it SFT. Training stage is unresolved. Released metadata records P=5 and global_step=200000. Current ProphRL config says P=6; this experiment preserves the saved checkpoint P=5. Published ProphRL scores are not reproduction targets for this artifact.
This checks H=1/2/5 at native P=5. It is not the P=50, H=5/20 experiment.
Rollout time excludes reset, checkpoint startup and video encoding, and can include first-call warmup.
The fixed task/time-limit protocol makes mean episode length constant across H.
Statistical inference concerns these tasks/layouts under this inference configuration; it does not establish a universal law.
