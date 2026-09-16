# Round 2 findings

## Primary confirmation on different initial layouts

The preselected step-500 H=20 model achieved **72%**, versus **49%** for the original H=20 model. The paired change is **+23.0 pp**, 95% within-task bootstrap CI **[+14.0, +32.0] pp**, exact McNemar p=**0.0006061** (one predeclared primary comparison).

There were 33 recovered episodes and 10 regressions. All 100 pairs use seed 27 and official initial-state indices 30–39, excluded from our OPSD training and the round-two screen.

| Snapshot | Deployment H | Successes/episodes | Success | Calls/episode |
|---|---:|---:|---:|---:|
| Original | 5 | 93/100 | 93% | 59.56 |
| Original | 20 | 49/100 | 49% | 22.53 |
| 100 updates | 20 | 50/100 | 50% | 21.95 |
| 500 updates | 20 | 72/100 | 72% | 18.86 |

The primary comparison provides evidence of improved task success on these held-out layouts.

On the same confirmation layouts, the original H=5 minus H=20 replanning gap was **44.0 pp**, paired CI **[35.0, 53.0] pp**, Holm-adjusted frequency-family p=2.596e-10. H=20 saved **62.2%** of calls per episode and **74.9%** of calls per controlled environment step. Episode-level savings also depend on how early a policy completes the task.

## Do more steps help?

The 100-update snapshot versus the original changed success by **+1.0 pp**, paired CI **[-9.0, +11.0] pp**, Holm-adjusted secondary-family p=1. The earlier 100-update pilot used a different evaluation split; its outcomes are not pooled with this confirmation.

On confirmation, 500 versus 100 updates changed success by **+22.0 pp**, paired CI **[+13.0, +31.0] pp**; raw p=0.000472, Holm-adjusted secondary-family p=0.01038.

The 300-update model is a screen checkpoint only. The final checkpoint was fixed at 500 before observing new outcomes; the screen did not select a winner. More updates also collect additional on-policy experience and cover more starting layouts; this is not an optimizer-step ablation on a fixed dataset.

## H=15 and H=25 deployment transfer

All added training used H=20. These are deployment-horizon tests of the same parameters on 50 paired episodes per condition, states 20–24, seed 17.

| H | Original | 100 updates | 500 updates | 500 − original | Secondary adjusted p |
|---:|---:|---:|---:|---:|---:|
| 15 | 80% | 78% | 84% | +4.0 pp | 1 |
| 20 | 54% | 52% | 78% | +24.0 pp | 0.2127 |
| 25 | 24% | 30% | 52% | +28.0 pp | 0.2132 |

## Transfer to other LIBERO suites

Only LIBERO-10 supplied our OPSD rollouts. These suites assess transfer/retention of the added OPSD; they are not claimed unseen by the official base checkpoint. Each screen has only 30 paired episodes (3/task), so small changes are inconclusive.

| Suite | Original H=5 | Original H=20 | 100 updates H=20 | 500 updates H=20 | 500 − original (paired 95% CI) | Adjusted p |
|---|---:|---:|---:|---:|---:|---:|
| libero_spatial | 100.0% | 43.3% | 70.0% | 86.7% | +43.3 [+26.7, +60.0] pp | 0.02051 |
| libero_object | 90.0% | 66.7% | 63.3% | 93.3% | +26.7 [+10.0, +43.3] pp | 0.3223 |
| libero_goal | 96.7% | 73.3% | 83.3% | 93.3% | +20.0 [+6.7, +33.3] pp | 0.8438 |

## Task-level confirmation

Exploratory, 10 episodes/task; no task-wise significance claim.

| Task | Original H=5 | Original H=20 | 100 updates H=20 | 500 updates H=20 | Description |
|---:|---:|---:|---:|---:|---|
| 0 | 100% | 40% | 40% | 100% | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 100% | 30% | 70% | 90% | put both the cream cheese box and the butter in the basket |
| 2 | 100% | 40% | 20% | 90% | turn on the stove and put the moka pot on it |
| 3 | 100% | 90% | 90% | 80% | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 100% | 30% | 50% | 80% | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 100% | 90% | 80% | 70% | pick up the book and place it in the back compartment of the caddy |
| 6 | 80% | 30% | 20% | 10% | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 100% | 100% | 100% | 100% | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 60% | 40% | 10% | 10% | put both moka pots on the stove |
| 9 | 90% | 0% | 20% | 90% | put the yellow and white mug in the microwave and close it |

The average recovery does not extend to every task: step 500 versus original H=20 declined on task 3 (-10 pp), task 5 (-20 pp), task 6 (-20 pp), task 8 (-30 pp). Broader layout coverage and retention checks on these tasks are useful next controls; the small task samples do not identify the cause of these regressions.

The largest observed original H=5 → H=20 drops were on task 9 (90 pp), task 4 (70 pp), task 1 (70 pp). This ranking is descriptive; the table above gives task descriptions and all outcomes.

## Teacher/student setting and next-stage evidence

**H_T=5, H_S=20 is the strongest validated setting in this round.** It has a confirmed initial replanning gap and a predeclared independent confirmation of recovery after training. H=15 has less initial headroom in the screen; H=25 is a deployment test of a model trained only at H=20, so it needs its own training and confirmation before a fair comparison.

The step-500 point estimate recovered 52.3% of the original confirmation gap; the remaining difference from the original H=5 reference is 21.0 pp. This is not an equivalence test against H=5.

The step-500 H=20 policy used 18.86 calls/episode versus 59.56 for original H=5: 68.3% fewer calls/episode and 74.7% fewer calls per controlled environment step.

The extra 400 updates are supported as an improvement over 100 updates after the secondary-family correction.

## Limits and reproducibility

This tests temporal OPSD with Gaussian velocity matching, an experimental continuous-action adaptation of the local OPSD pipeline; it is not the image-generation Flow-OPD clipped policy-gradient algorithm. All conditions extrapolate the official checkpoint's native P=10 to P=50, with 10 unchanged native flow steps. Evaluation calls the official LIBERO execution loop.

FP32 master parameters, EMA and Adam moments/counters were restored at step 100. Updates 101–500 kept the learning rate, loss and frozen backbone unchanged. Training used only official LIBERO-10 start indices 10–19, starting at 12 after the two simulator restarts. All tested layouts were hash-disjoint from these and the earlier 18 OPSD training layouts. All evaluated snapshots shared one continuous policy server.

The added 400 updates actually visited **36 task/layout combinations**, with layout indices [12, 13, 14, 15], using 1600 action blocks and 31829 executed actions. Layout identifiers describe the initial scene; each trajectory supplies many intermediate observations.

Wilson intervals describe individual rates; paired differences use 10,000 within-task initial-state bootstrap replicates. The 22 secondary/screen training comparisons form one Holm family; 7 frequency comparisons form a separate Holm family. The fixed task set and small single-seed screens do not establish generalization to unseen tasks. Every condition, including regressions, is retained.

See aggregated/conditions.csv, comparisons.csv, per_task.csv and episodes.csv; raw records/videos under evaluations/; all new losses and rollout identities in training.jsonl and rollouts.jsonl. Checkpoint paths and hashes, plan, split audit and resource accounting are in provenance/. No further hyperparameter sweep was triggered by these outcomes.
