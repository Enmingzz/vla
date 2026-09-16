# LIBERO-90 transfer findings

This evaluation adds **zero optimizer updates**. It compares the official model with the fixed 500-update model trained only on LIBERO-10, using 10 selected LIBERO-90 tasks and three official initial layouts (40–42), seed 37.

## Primary comparison: task instructions absent from added OPSD

The primary subset contains **10 tasks / 30 paired episodes**. The trained H=20 model achieved **13.3%**, versus **20.0%** for original H=20: **-6.7 pp**, paired within-task 95% bootstrap CI **[-10.0, +0.0] pp**, exact McNemar p=**0.5**. There were 0 recoveries and 2 regressions.

An exploratory bootstrap resampling both tasks and layouts gives **[-23.3, +0.0] pp**. Among primary tasks, 0 improved, 1 declined and 9 tied (three trials/task).

The primary comparison does not establish an improvement or decline at the 5% level. These data do not validate cross-task recovery.

## Replanning gap and inference calls

Original H=5 minus original H=20 was **-3.3 pp**, paired 95% CI **[-10.0, +0.0] pp**, secondary-family Holm p=**1**. Trained H=20 minus original H=5 is **-3.3 pp**.

Trained H=20 uses **74.3% fewer calls per episode** and **74.9% fewer calls per controlled step** than original H=5. The latter controls for differing episode lengths; neither number measures wall-clock speedup.
Original H=5 does not outperform original H=20 in this screen. There is no positive measured replanning gap to express as a recovery fraction; any trained-model gain must be interpreted separately from recovery of a frequency penalty.

## All results

| Subset | Model | Successes/episodes | Success | Calls/episode | Calls/control step |
|---|---|---:|---:|---:|---:|
| selected_10 | Original H=5 | 5/30 | 16.7% | 72.87 | 0.2001 |
| selected_10 | Original H=20 | 6/30 | 20.0% | 18.10 | 0.0503 |
| selected_10 | 500 updates H=20 | 4/30 | 13.3% | 18.73 | 0.0501 |

## Task outcomes

| ID | Task | Original H=5 | Original H=20 | Trained H=20 |
|---:|---|---:|---:|---:|
| 0 | close the top drawer of the cabinet | 0/3 | 0/3 | 0/3 |
| 1 | close the top drawer of the cabinet and put the black bowl on top of it | 0/3 | 0/3 | 0/3 |
| 2 | put the black bowl in the top drawer of the cabinet | 2/3 | 3/3 | 1/3 |
| 3 | put the butter at the back in the top drawer of the cabinet and close it | 0/3 | 0/3 | 0/3 |
| 4 | put the butter at the front in the top drawer of the cabinet and close it | 0/3 | 0/3 | 0/3 |
| 5 | put the chocolate pudding in the top drawer of the cabinet and close it | 0/3 | 0/3 | 0/3 |
| 6 | open the bottom drawer of the cabinet | 0/3 | 0/3 | 0/3 |
| 7 | open the top drawer of the cabinet | 0/3 | 0/3 | 0/3 |
| 8 | open the top drawer of the cabinet and put the bowl in it | 0/3 | 0/3 | 0/3 |
| 9 | put the black bowl on the plate | 3/3 | 3/3 | 3/3 |

8 of 10 tasks had no successes in any of the three settings on the tested layouts. This low baseline success limits what the screen can say about frequency penalties and their recovery.

The three secondary comparisons (all-selected-task training effect and original replanning gaps for both subsets) receive a joint Holm correction; comparisons.csv retains raw/adjusted p values and both bootstrap intervals.

## Scope and controls

Selected LIBERO-90 task names are disjoint from the ten OPSD training task names. Exact instruction overlap occurs at task IDs []. Any overlapping tasks remain in the selected-task table and are excluded from the primary subset by the metadata rule.

Only our added OPSD is task-held-out. The official checkpoint's historical fine-tuning/pretraining exposure is not fully audited. Current public task metadata is archived separately and is not evidence that the base model never saw a task, scene, object or subskill.

All comparisons fix P=50 (an extension of the checkpoint's native P=10), ten native flow steps, preprocessing, ordered initial states, episode/call RNG and the official LIBERO-90 400-step limit. Both frozen snapshots use identical native sampler code. No optimizer or teacher updates are available in this comparison server. This evaluates the temporal OPSD Gaussian velocity-matching adaptation, not a new reproduction of the image-generation Flow-OPD algorithm.

The 500-update checkpoint was selected before any LIBERO-90 outcome. Rendering pilot episodes are excluded. All formal episodes, including failed tasks, are retained; simulator/inference exceptions abort evaluation.

After repeated native NVIDIA EGL aborts before any formal episode, all three conditions use the same pinned OSMesa software renderer. A separate exact-action replay passed; archived EGL/OSMesa images were not pixel-identical. The result is therefore specific to this documented rendering environment. No formal EGL measurements are mixed into these statistics.

Raw JSONL/manifests and videos are under evaluations/. Tables include every task and paired episode. See provenance/ for the split audit, checkpoint hashes, code, checks and GPU accounting; see LIBERO90_TRANSFER.md for exact rerun commands.

## User-requested first-ten screen

The selected official task IDs are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

These are the first ten tasks in benchmark order, not a random sample of LIBERO-90 and not the separate LIBERO-10 suite. Three layouts per task provide a small exploratory screen. Do not extrapolate the measured rate to all 90 tasks. The full-suite attempt and every outcome outside this prefix remain archived separately; the selection was requested by the user, not chosen to improve the measured effect.

The 30 completed H=5 prefix episodes were copied unchanged with checksums and videos. Both H=20 conditions use a restarted read-only server. Inference configuration, native sampler code, initial images and episode RNG are required to match despite the server restart.
