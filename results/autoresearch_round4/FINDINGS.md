# Another 500 OPSD updates: step 1000 to step 1500

The fixed additional 500 updates changed H=20 success by **+12.0 pp**, paired 95% within-task bootstrap CI **[+4.0, +20.0] pp**, exact McNemar p=**0.0169**. There were 17 recoveries and 5 regressions.

This follow-up supports improvement on the measured layouts.

| Updates | Success | Mean actions/episode | Mean calls/episode | Mean episode seconds | Whole evaluation seconds |
|---:|---:|---:|---:|---:|---:|
| 1000 | 78/100 (78%) | 334.21 | 17.05 | 54.89 | 1588.89 |
| 1500 | 90/100 (90%) | 305.63 | 15.72 | 51.89 | 1515.36 |

Episode timing includes reset/settling and inference, but excludes its own video encoding. Whole-condition timing includes process startup and videos. Eight simulator workers share one H100; summed episode durations are not whole-condition wall time.

| Updates | Successful episodes | Mean actions/successful episode | Mean calls/successful episode |
|---:|---:|---:|---:|
| 1000 | 78 | 281.81 | 14.53 |
| 1500 | 90 | 281.81 | 14.58 |

On the 73 episodes successful at both checkpoints, mean executed actions changed from 277.62 to 276.47. Success-only comparisons condition on outcomes and may have different episode membership; actions exclude the 10 settling steps and unexecuted chunk suffixes.

| Task | Step 1000 | Step 1500 | Change | Description |
|---:|---:|---:|---:|---|
| 0 | 100% | 80% | -20 pp | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 100% | 100% | +0 pp | put both the cream cheese box and the butter in the basket |
| 2 | 100% | 100% | +0 pp | turn on the stove and put the moka pot on it |
| 3 | 100% | 90% | -10 pp | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 60% | 100% | +40 pp | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 90% | 100% | +10 pp | pick up the book and place it in the back compartment of the caddy |
| 6 | 70% | 100% | +30 pp | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 70% | 80% | +10 pp | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 50% | 50% | +0 pp | put both moka pots on the stove |
| 9 | 40% | 100% | +60 pp | put the yellow and white mug in the microwave and close it |

Training restored the step-1000 FP32 weights, EMA teacher and Adam state. P=50, H_student=20, H_teacher=5, 10 flow steps, batch size 4, learning rate 1e-5, trainable parameter selection, velocity loss, first-five-block supervision and auxiliary teacher-tail sampling are unchanged. The VLM/vision backbone remains frozen. The student, not the EMA, is evaluated.

Exactly 2000 new action blocks executed 39075 actions across 100 task/layout combinations, at layout indices [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]. Training remains LIBERO-10 only, inside indices 10–19; each continuing episode contributes multiple action blocks.

The step-1000 reference reuses 100 previously validated episodes from `results/autoresearch_round3_finish_retry1`. The new allocation runs only the 100 step-1500 episodes, saving the 26.5 minutes previously spent evaluating the reference. Source shards, manifests and provenance are checksum-pinned before training; no cached outcomes are modified. The checkpoints use separate H100 allocations and server instances, so wall-clock differences are descriptive, not a controlled speed benchmark. Both conditions retain identical initial-state, first-observation, RNG, evaluator and non-parameter inference-setting checks, seed 27 and official indices 30–39.

These layouts were excluded from all added training but their previous results were already inspected; this is an exploratory continuation, not a new blind confirmation. No intermediate success score selected the step-1500 endpoint.

All current training and evaluation use pinned OSMesa. CPU checks verified exact serial/parallel simulator observations; previous checks are reused only when their implementation hashes still match. Training parallelism changes scheduling, not batch size or action execution. Simulator trajectories restart on checkpoint resume.

The continuation combines extra optimization with fresh on-policy experience. It does not isolate optimizer-step count on a fixed dataset, establish transfer to other suites, or reproduce the image-generation Flow-OPD algorithm. This remains temporal OPSD with velocity matching on the P=50 extension of the native P=10 checkpoint.
