# Findings

Measured smoke evaluation on libero_10. Coverage: complete for the explicitly selected horizons.

## Protocol constraint

Explicit fixed-P extension: inference P=50, while the official `pi05_libero` config specifies P=10. The same checkpoint weights and upstream action loop are used. Only the static prediction length is overridden; flow steps and attention-mask rules are unchanged. Every compared H uses the same P=50. This is sequence-length extrapolation, not official-protocol reproduction. Even the first five actions can change when P changes because action tokens attend to one another. Results cannot be pooled with the native-P archive.

OpenPI commit: `215abfb217dbac7d5f1273282331b9b1866c0479`. Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`. Flow steps: 10 (upstream default).

## Measured success

| H | Successes / episodes | Success | 95% Wilson CI | Calls/episode | Controlled steps/episode |
|---:|---:|---:|---:|---:|---:|
| 5 | 88/100 | 88.0% | [80.2%, 93.0%] | 59.94 | 298.0 |
| 15 | 82/100 | 82.0% | [73.3%, 88.3%] | 24.23 | 356.9 |
| 20 | 46/100 | 46.0% | [36.6%, 55.7%] | 22.64 | 448.2 |

## H=5 baseline screen

H=5 measured 88.0%, versus the official 92.4% reference. The predeclared diagnostic gate allows at most a 5 percentage point deficit: PASS. This gate is a debugging screen, not an equivalence test; the public reference has no reported uncertainty here.
The official reference is contextual only because P differs. All selected H values are evaluated as diagnostics even if this screen fails; failure prevents interpreting a gap as evidence for a strong teacher.

## Paired replanning gaps and compute savings

Gap is Success(5) − Success(H), measured on matching seed/task/initial-state indices. Positive values favour H=5.

- H=15: gap 6.0 pp, paired 95% bootstrap CI [-2.0, 15.0] pp; 59.6% fewer calls per episode and 66.3% fewer calls per controlled step (100 pairs). Exact McNemar p (Holm-adjusted) = 0.2863.
- H=20: gap 42.0 pp, paired 95% bootstrap CI [33.0, 51.0] pp; 62.2% fewer calls per episode and 74.9% fewer calls per controlled step (100 pairs). Exact McNemar p (Holm-adjusted) = 9.095e-13.

## Interpretation

Measured success at H=15 was 6.0 percentage points lower than H=5.
Measured success at H=20 was 42.0 percentage points lower than H=5.
Only the measured horizons are compared; a monotonic trend over a wider sweep is not established. P is fixed across these conditions, so H is the experimental variable, but the conclusion applies to this extrapolated P=50 inference setting. This experiment does not isolate the quality of later extrapolated actions from the benefit of more frequent feedback.
Substantial degradation (predeclared ≥5 percentage points): H=15, H=20.
Tasks with the largest absolute changes (exploratory; positive gaps favour H=5, negative gaps favour the larger H; no task-wise significance claim):

- Task 4 at H=20: gap +70.0 pp (H=5 100.0%, H=20 30.0%) — put the white mug on the left plate and put the yellow and white mug on the right plate.
- Task 6 at H=20: gap +70.0 pp (H=5 100.0%, H=20 30.0%) — put the white mug on the plate and put the chocolate pudding to the right of the plate.
- Task 0 at H=20: gap +60.0 pp (H=5 90.0%, H=20 30.0%) — put both the alphabet soup and the tomato sauce in the basket.
- Task 9 at H=20: gap +60.0 pp (H=5 90.0%, H=20 30.0%) — put the yellow and white mug in the microwave and close it.
- Task 1 at H=20: gap +50.0 pp (H=5 100.0%, H=20 50.0%) — put both the cream cheese box and the butter in the basket.

No teacher/student pair is recommended from the evidence currently available.
These results concern only the measured H values under fixed inference P=50. They do not establish native P=10 performance at larger H or validate extrapolated action quality. No training or distillation was used to collect these frequency-sweep results.
Smoke/partial results are diagnostic, not the requested main validation.

## Reproducibility and uncertainty

Environment settling, image rotation/resize, state conversion, chunk-prefix execution, success termination and task step limits come directly from the pinned official evaluator. Initial-state and first-policy-observation hashes, per-episode RNG seeds, and inference fingerprints are validated across H. Actual call positions must equal 0,H,2H,… .

Wilson intervals describe the episode-level binomial rate. Gap intervals use paired initial-state blocks resampled within fixed tasks (10,000 replicates, fixed analysis seed); repeated seeds of one initial state stay in the same block. Exact McNemar tests are reported for single-seed runs, with Holm correction across measured candidate horizons. The task set is fixed; intervals do not establish generalization to unseen tasks. Episode duration excludes video encoding and any server startup warmup; inference compilation occurring after episode start is included.
