# H=5 retention after 1500 EGL OPSD updates

Both checkpoints were freshly evaluated on one H100/server: P=50, H=5, EGL, four simulator workers, 10 flow steps, LIBERO-10, seed 27, ordered initial-state indices 30–39. No new training was performed.

| Model | Success | Mean actions | Mean calls | Mean seconds | Mean seconds, success |
|---|---:|---:|---:|---:|---:|
| Original | 88/100 (88%) | 306.23 | 61.59 | 14.46 | 13.59 |
| OPSD step 1500 | 86/100 (86%) | 301.72 | 60.69 | 14.55 | 13.55 |

Trained minus original: **-2.0 percentage points**, paired 95% bootstrap CI **[-10.0, +6.0] pp**; exact two-sided McNemar **p=0.814529**. There were 8 recovered and 10 regressed episodes.

This evaluation does not establish a statistically clear H=5 change. Failure to detect degradation is not evidence of equivalence or non-inferiority.

The historical original H=5 result was 88/100; the fresh reference is 88/100. Only the two freshly measured conditions define the primary training comparison.

This checks retention at the explicitly extended P=50, not the official native P=10. The fixed ten tasks and previously inspected evaluation layouts were held out from added OPSD training; this is not an unseen-task or blind evaluation. One requested follow-up contrast is reported; task-level changes are exploratory. Seconds include shared-server waiting and exclude each episode's video encoding.
