# Fresh training and evaluation entirely under EGL

All five conditions were rerun on one H100/server, EGL, four simulator workers, P=50, 10 flow steps, LIBERO-10, seed 27, official initial-state indices 30–39 (10 tasks × 10 episodes). All 1500 temporal OPSD updates started from the official checkpoint and used the same EGL training implementation.

| Training steps | H | Success | Avg. actions, all | Avg. actions, success | Avg. calls, all | Avg. seconds, all | Avg. seconds, success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (Original) | 5 | 88/100 (88%) | 306.23 | 277.08 | 61.59 | 14.30 | 13.41 |
| 0 (Original) | 20 | 54/100 (54%) | 437.05 | 366.39 | 22.14 | 10.42 | 8.79 |
| 500 | 20 | 87/100 (87%) | 335.02 | 307.38 | 17.17 | 8.13 | 7.54 |
| 1000 | 20 | 82/100 (82%) | 319.19 | 275.11 | 16.36 | 7.84 | 7.01 |
| 1500 | 20 | 87/100 (87%) | 324.59 | 295.39 | 16.61 | 8.03 | 7.45 |

Seconds are mean episode wall time under fixed four-worker concurrency, including reset, settling, inference and execution, excluding that episode's video encoding. They include shared-server waiting and are not isolated model latency. Success-only means condition on potentially different sets of episodes. Whole-condition seconds and Wilson success intervals are in conditions.csv.

## Controlled training and evaluation

This is a new run from the official checkpoint. All updates 1–1500 and every evaluation use EGL; historical OSMesa-trained checkpoints and historical evaluation outcomes are not used. Training keeps P=50, student H=20, teacher H=5, 10 flow steps, batch 4, learning rate 1e-5, EMA 0.9999, the same frozen VLM/vision backbone and the same temporal velocity-matching loss. The student, not the EMA, is evaluated. Train initial states stay within indices 10–19, seed 17; teacher views are collected from the student's on-policy trajectory. The context-selection correction is active from the start and serial/parallel observations must match exactly in the EGL check before formal training.

Checkpoints 500/1000/1500 were fixed before observing outcomes. Each 500-update segment starts fresh simulator trajectories under the same initial-state cycling rule; weights, EMA and Adam continue without resetting. More steps also add on-policy experience, so this is not a fixed-dataset optimizer-step ablation. Evaluation layouts were inspected previously and remain excluded from training; this is not a new blind test. Eight paired success comparisons form one Holm-adjusted exploratory family.

| Comparison | Step | Reference H | Success change (pp) | Paired 95% CI (pp) | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| versus_original_H5 | 0 | 5 | -34.0 | [-44.0, -24.0] | 1.405e-07 | 8.431e-07 |
| versus_original_H5 | 500 | 5 | -1.0 | [-9.0, +7.0] | 1 | 1 |
| versus_original_H20 | 500 | 20 | +33.0 | [+24.0, +42.0] | 3.609e-08 | 2.887e-07 |
| versus_original_H5 | 1000 | 5 | -6.0 | [-15.0, +2.0] | 0.2632 | 1 |
| versus_original_H20 | 1000 | 20 | +28.0 | [+17.0, +38.0] | 2.545e-05 | 0.0001272 |
| versus_original_H5 | 1500 | 5 | -1.0 | [-9.0, +7.0] | 1 | 1 |
| versus_original_H20 | 1500 | 20 | +33.0 | [+24.0, +42.0] | 3.609e-08 | 2.887e-07 |
| step1500_minus_step1000 | 1500 | 20 | +5.0 | [-2.0, +12.0] | 0.3018 | 1 |
