# Findings

Measured smoke evaluation on libero_10. Coverage: complete for the explicitly selected horizons.

## Protocol constraint

The pinned official `pi05_libero` config has native prediction horizon P=10. Requested horizons above P were not executed. No model/config override, action padding, repetition, or hidden policy calls was used.

OpenPI commit: `215abfb217dbac7d5f1273282331b9b1866c0479`. Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`. Flow steps: 10 (upstream default).

## Measured success

| H | Successes / episodes | Success | 95% Wilson CI | Calls/episode | Controlled steps/episode |
|---:|---:|---:|---:|---:|---:|
| 5 | 91/100 | 91.0% | [83.8%, 95.2%] | 56.04 | 278.4 |
| 10 | 97/100 | 97.0% | [91.5%, 99.0%] | 26.41 | 259.6 |

## H=5 reproduction gate

H=5 measured 91.0%, versus the official 92.4% reference. The predeclared diagnostic gate allows at most a 5 percentage point deficit: PASS. This gate is a debugging screen, not an equivalence test; the public reference has no reported uncertainty here.

## Paired replanning gaps and compute savings

Gap is Success(5) − Success(H), measured on matching seed/task/initial-state indices. Positive values favour H=5.

- H=10: gap -6.0 pp, paired 95% bootstrap CI [-11.0, -1.0] pp; 52.9% fewer calls per episode and 49.5% fewer calls per controlled step (100 pairs). Exact McNemar p (Holm-adjusted) = 0.07031.

## Interpretation

Measured success at H=10 was 6.0 percentage points higher than H=5.
The measured difference does not establish a monotonic decrease over all requested H values; H>10 was not measurable under this protocol.
Substantial degradation (predeclared ≥5 percentage points): none of the measured horizons.
Tasks with the largest absolute changes (exploratory; positive gaps favour H=5, negative gaps favour the larger H; no task-wise significance claim):

- Task 8 at H=10: gap -30.0 pp (H=5 50.0%, H=10 80.0%) — put both moka pots on the stove.
- Task 3 at H=10: gap -10.0 pp (H=5 90.0%, H=10 100.0%) — put the black bowl in the bottom drawer of the cabinet and close it.
- Task 6 at H=10: gap -10.0 pp (H=5 90.0%, H=10 100.0%) — put the white mug on the plate and put the chocolate pudding to the right of the plate.
- Task 9 at H=10: gap -10.0 pp (H=5 90.0%, H=10 100.0%) — put the yellow and white mug in the microwave and close it.

No teacher/student pair is recommended from the evidence currently available.
The intended H_S=20/30/50 premise remains untested because native P=10. These results alone cannot justify that proposed distillation stage; it requires a revised, explicitly authorized protocol. No training or distillation was implemented.
Smoke/partial results are diagnostic, not the requested main validation.

## Reproducibility and uncertainty

Environment settling, image rotation/resize, state conversion, chunk-prefix execution, success termination and task step limits come directly from the pinned official evaluator. Initial-state and first-policy-observation hashes, per-episode RNG seeds, and inference fingerprints are validated across H. Actual call positions must equal 0,H,2H,… .

Wilson intervals describe the episode-level binomial rate. Gap intervals use paired initial-state blocks resampled within fixed tasks (10,000 replicates, fixed analysis seed); repeated seeds of one initial state stay in the same block. Exact McNemar tests are reported for single-seed runs, with Holm correction across measured candidate horizons. The task set is fixed; intervals do not establish generalization to unseen tasks. Video encoding time is excluded from episode duration; first-episode JIT compilation is included.

A separate fixed-input/seed diagnostic made 3 repeated queries per server. Maximum within-server action differences were [0.0, 0.0]; the difference between the smoke and main server instances was 0.00217204. The precise numerical cause was not isolated. Independent GPU server instances were not bitwise reproducible in this run despite matching model/config fingerprints. Each H comparison uses one continuously running server; smoke and main episodes are analyzed separately. This diagnostic is not a benchmark episode. Details: `results/diagnostics/inference_reproducibility.json`.
