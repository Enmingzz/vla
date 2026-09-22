# RoboCasa365 fixed-ten-task pilot

This is a selected ten-task pilot on the official pretrain scene/object split, not the full RoboCasa365 benchmark.
The ten tasks were fixed before observing results. Training and evaluation use distinct seed namespaces.
Each complete evaluation condition contains 10 tasks × 10 episodes = 100 episodes. OPSD uses 500 optimizer updates, batch size 4.

| Condition | H | Success | 95% CI | Calls/episode | Actions/episode | Seconds/episode |
|---|---:|---:|---:|---:|---:|---:|
| original_h5 | 5 | 54/100 (54.0%) | [44.3%, 63.4%] | 111.3 | 555.5 | 31.52 |
| original_h20 | 20 | 58/100 (58.0%) | [48.2%, 67.2%] | 25.4 | 501.0 | 20.13 |
| step500_h20 | 20 | 35/100 (35.0%) | [26.4%, 44.7%] | 32.0 | 633.5 | 25.49 |
| step500_h5 | — | Incomplete: 2/43 completed episodes succeeded | — | — | — | — |

All environments use EGL. Episode seconds exclude reset/startup and include rollout video writing.
Success uses the official `info["success"]`; episode limits come directly from the native task registry.
Intervals below are paired episode bootstrap intervals, conditional on these ten tasks; p-values are unadjusted and exploratory.

replanning: change = +4.0%; paired 95% CI [-7.0%, +15.0%]; exact McNemar p = 0.5966.

opsd_recovery: change = -23.0%; paired 95% CI [-34.0%, -12.0%]; exact McNemar p = 0.0002941.

Incomplete h5_retention comparison, restricted to the same 43 completed episodes: original_h5 17/43; step500_h5 2/43. This task-order-dependent subset is descriptive and is not the full ten-task result.

H=20 saves 77.2% of policy calls per episode relative to original H=5.
Largest measured task gaps (H=5 minus H=20): TurnOnSinkFaucet +40%, PickPlaceCounterToCabinet +10%, PickPlaceCounterToStove +10%.
The H=5 advantage is not statistically established in this pilot; do not assume sparse replanning caused a material accuracy loss.

The measured 500-step H=20 change is -23.0%; positive changes indicate recovery, negative changes indicate deterioration.

The method ports the existing temporal Gaussian velocity-matching OPSD implementation with an EMA teacher (decay 0.9999). It is not a reproduction of an image-generation Flow-OPD policy-gradient objective.
No task was selected by baseline success. Shared task identities mean this measures adaptation to new seeded episodes, not cross-task generalization.

Experiment incomplete: missing episodes are not scored as failures.
