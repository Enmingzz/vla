# Another 500 OPSD updates: step 500 to step 1000

The fixed additional 500 updates changed H=20 success by **+3.0 pp**, paired 95% within-task bootstrap CI **[-6.0, +12.0] pp**, exact McNemar p=**0.7111**. There were 16 recoveries and 13 regressions.

This follow-up does not establish a directional change at the 5% level.

| Updates | Success | Mean actions/episode | Mean calls/episode | Mean episode seconds | Whole evaluation seconds |
|---:|---:|---:|---:|---:|---:|
| 500 | 75/100 (75%) | 361.75 | 18.45 | 59.86 | 1592.59 |
| 1000 | 78/100 (78%) | 334.21 | 17.05 | 54.89 | 1588.89 |

Episode timing includes reset/settling and inference, but excludes its own video encoding. Whole-condition timing includes process startup and videos. Eight simulator workers share one H100; summed episode durations are not whole-condition wall time.

| Task | Step 500 | Step 1000 | Change | Description |
|---:|---:|---:|---:|---|
| 0 | 100% | 100% | +0 pp | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 90% | 100% | +10 pp | put both the cream cheese box and the butter in the basket |
| 2 | 70% | 100% | +30 pp | turn on the stove and put the moka pot on it |
| 3 | 100% | 100% | +0 pp | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 70% | 60% | -10 pp | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 90% | 90% | +0 pp | pick up the book and place it in the back compartment of the caddy |
| 6 | 10% | 70% | +60 pp | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 100% | 70% | -30 pp | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 30% | 50% | +20 pp | put both moka pots on the stove |
| 9 | 90% | 40% | -50 pp | put the yellow and white mug in the microwave and close it |

The continuation was saved at step 978 at the one-hour budget boundary, then resumed for the remaining 22 updates. Both segments preserve FP32 weights, EMA and Adam moments/counters. Simulator episodes restart on resume; this is not an uninterrupted simulator trajectory. Prior segment logs remain in their original archives and are joined only after checksum and contiguous-update validation.

Training restored the original step-500 FP32 weights, EMA teacher and Adam state. P=50, H_student=20, H_teacher=5, 10 flow steps, batch size 4, learning rate 1e-5, trainable parameter selection, velocity loss, first-five-block supervision and auxiliary teacher-tail sampling are unchanged. The VLM/vision backbone remains frozen. The student, not the EMA, is evaluated.

Exactly 2000 new action blocks executed 39151 actions across 100 task/layout combinations, at layout indices [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]. Training remains LIBERO-10 only, inside indices 10–19; each continuing episode contributes multiple action blocks.

Both checkpoints were re-evaluated on the same continuous server, seed 27, official indices 30–39, with identical initial-state, first-observation and inference-setting checks. These layouts were excluded from all added training but their previous results were already inspected; this is an exploratory continuation, not a new blind confirmation. No intermediate success score selected the step-1000 endpoint.

All current training and evaluation use pinned OSMesa, avoiding the prior native NVIDIA EGL failures. The step-500 result is therefore measured again under OSMesa; the earlier EGL 72% is historical context, not substituted for the current baseline. CPU checks verified exact serial/parallel simulator observations before the GPU allocation. Training parallelism changes scheduling, not batch size or action execution.

A CPU-only diagnostic found a context-selection issue in serial multi-environment OSMesa rendering. Training now explicitly makes the owning render context current before stepping/closing each environment. The corrected serial and isolated-process observations must match exactly before this run. This is an additional runtime correction; its effect on historical EGL training images has not been established.

The continuation combines extra optimization with fresh on-policy experience and a renderer change relative to the parent training. It does not isolate optimizer-step count on a fixed dataset, establish transfer to other suites, or reproduce the image-generation Flow-OPD algorithm. This remains temporal OPSD with velocity matching on the P=50 extension of the native P=10 checkpoint.
