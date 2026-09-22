# RoboCasa365 fixed-ten-task pilot

This is a selected ten-task pilot on the official pretrain scene/object split, not the full RoboCasa365 benchmark.
The ten tasks were fixed before observing results. Training and evaluation use distinct seed namespaces.
Each complete evaluation condition contains 10 tasks × 10 episodes = 100 episodes. OPSD uses 500 optimizer updates, batch size 4.

| Condition | H | Success | 95% CI | Calls/episode | Actions/episode | Seconds/episode |
|---|---:|---:|---:|---:|---:|---:|
| original_h5 | — | Incomplete: 31/60 completed episodes succeeded | — | — | — | — |
| original_h20 | 20 | 58/100 (58.0%) | [48.2%, 67.2%] | 25.4 | 501.0 | 20.13 |
| step500_h20 | — | Incomplete: 6/33 completed episodes succeeded | — | — | — | — |
| step500_h5 | — | Not started | — | — | — | — |

All environments use EGL. Episode seconds exclude reset/startup and include rollout video writing.
Success uses the official `info["success"]`; episode limits come directly from the native task registry.
Intervals below are paired episode bootstrap intervals, conditional on these ten tasks; p-values are unadjusted and exploratory.

Incomplete replanning comparison, restricted to the same 60 completed episodes: original_h5 31/60; original_h20 38/60. This task-order-dependent subset is descriptive and is not the full ten-task result.

Incomplete opsd_recovery comparison, restricted to the same 33 completed episodes: original_h20 18/33; step500_h20 6/33. This task-order-dependent subset is descriptive and is not the full ten-task result.

The method ports the existing temporal Gaussian velocity-matching OPSD implementation with an EMA teacher (decay 0.9999). It is not a reproduction of an image-generation Flow-OPD policy-gradient objective.
No task was selected by baseline success. Shared task identities mean this measures adaptation to new seeded episodes, not cross-task generalization.

Experiment incomplete: missing episodes are not scored as failures.
