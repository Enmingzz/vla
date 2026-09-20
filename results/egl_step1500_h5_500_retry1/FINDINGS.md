# H=5 retention over 500 episodes per model

P=50, H=5, EGL, four workers, 10 flow steps, LIBERO-10, seed 27, official layout indices 0–49. The prior 100 episodes per model (states 30–39) were reused; 400 per model were added. No training was performed.

The primary retention comparison excludes OPSD training layouts 10–19: 400 paired episodes per model. The full 500 and the 100 training-layout episodes are descriptive secondary results. The fixed tasks and layouts have been inspected previously; this is not a blind or unseen-task test.

| Subset | Model | Success | Mean actions | Mean calls | Mean seconds |
|---|---|---:|---:|---:|---:|
| heldout_400 | Original | 356/400 (89.0%) | 303.38 | 61.02 | 14.35 |
| heldout_400 | OPSD step 1500 | 345/400 (86.2%) | 301.40 | 60.63 | 14.44 |
| all_500 | Original | 444/500 (88.8%) | 303.32 | 61.02 | 14.38 |
| all_500 | OPSD step 1500 | 429/500 (85.8%) | 301.66 | 60.67 | 14.45 |
| training_layouts_100 | Original | 88/100 (88.0%) | 303.11 | 61.00 | 14.52 |
| training_layouts_100 | OPSD step 1500 | 84/100 (84.0%) | 302.68 | 60.84 | 14.45 |

Primary held-out change (trained minus original): **-2.75 pp**, paired 95% bootstrap CI **[-6.25, +1.00] pp**, exact two-sided McNemar **p=0.18485**; 23 recoveries and 34 regressions.

A non-significant difference does not establish equivalence or non-inferiority. One primary contrast was fixed before the extension; secondary and task-level analyses are exploratory. Cached and new episodes span two server instances with identical recorded inference/evaluator settings; there is no additional repeated baseline. Timing pools matched four-worker runs and includes shared-server waiting, but is not isolated inference latency. The native checkpoint configuration uses P=10; this study remains at the authorized P=50.
