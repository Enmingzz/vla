# Fixed P=50 extension archive

Requested after the original native-P=10 experiment. This is a separate inference
configuration using the same official `pi05_libero` checkpoint. The official
fine-tuning config sets P=10; this extension fixes P=50 for both H=5 and H=30.
No training, weight conversion, flow-step change, attention-rule change, action
padding, or repetition was used. The official loader successfully restored all
required parameter shapes, and every actual inference returned 50 actions.

The completed LIBERO-10 smoke test uses seed 7 and the first 10 official initial
states of every task, paired across H. One continuous policy server handled both
conditions. Four simulator task workers shared one H100 80 GB, 8 CPU cores and
96 GB host memory. Job **60035109** completed with exit `0:0` in **19m 10s** on
2026-09-15. The evaluation implementation at submission was commit `5130a60`;
only reporting and documentation changed after collection.

| Fixed P | H | Success | Mean calls/episode | Mean controlled steps/episode |
|---:|---:|---:|---:|---:|
| 50 | 5 | 86/100 = 86% | 61.43 | 305.45 |
| 50 | 30 | 12/100 = 12% | 17.26 | 498.63 |

The paired success gap is **74 percentage points**, bootstrap 95% CI **[67, 81]**;
exact two-sided McNemar p = **1.059e-22** (74 H=5-only successes, zero H=30-only).
H=30 saves **71.9%** of calls per episode, or **82.8%** per controlled environment
step. Most H=30 failures run to the 520-step task limit, explaining why episode
call savings are smaller than call-density savings.

All 200 episodes, 20 shard manifests, and 200 distinct videos are complete.
All 15 automated tests passed, including prefix execution through the actual
pinned upstream loop at P=50/H=30 and rejection of mixed-P aggregation.
Paired initial states, first observations, episode RNG seeds, inference settings,
action chunk lengths and actual policy-call schedules passed validation. Original
OpenPI and LIBERO tracked files remain unchanged. Videos and runtime logs are
saved locally and excluded from Git; all raw records, manifests, tables and
figures are included. Do not pool these episodes with the native-P archive or
treat the overlapping initial states as independent repetitions of it.

## Interpretation

P=50/H=30 is technically runnable, but it performs poorly in this smoke test.
This is a valid comparison of H **within fixed P=50**, not the official P=10
protocol. Increasing P can change even the first five predictions because action
tokens attend to one another. The experiment does not isolate degradation of
later extrapolated actions from the benefit of more frequent feedback.

P=50/H=5 also misses the predeclared 87.4% diagnostic reference screen, so it is
not an established reproduction of the published 92.4% baseline. Before any data
were collected, this extension was configured to complete both diagnostic
conditions even if that screen failed. The observed 74-point gap is much larger
than the intended moderate gap; recoverability through distillation was not
tested. No teacher/student pair is recommended, and no main 50-episode/task P=50
run or other H values were collected in this extension.

## Files and reproduction

- [Findings](FINDINGS.md)
- [Episode CSV](aggregated/smoke/libero_10/episodes.csv)
- [Aggregate CSV](aggregated/smoke/libero_10/frequency_sweep.csv)
- [Paired gaps](aggregated/smoke/libero_10/replanning_gaps.csv)
- [Validation](aggregated/smoke/libero_10/validation.json)
- [Success vs H](figures/smoke/libero_10/success_vs_replan_horizon.png)
- [Success vs calls](figures/smoke/libero_10/success_vs_policy_calls.png)
- [Relative performance drop](figures/smoke/libero_10/relative_performance_drop.png)
- [Runtime config comparison](provenance/configuration_comparison.json)
- [Scheduler record](provenance/slurm_jobs.psv)

The [paired video frames](diagnostics/paired_rollout_frames.png) show the first
task/initial-state pair, ordered by task and episode index, where H=5 succeeded
and H=30 failed. H=5 completed after 379 controlled steps and 76 calls; H=30
timed out after 520 steps and 18 calls. In the inspected H=30 rollout, the robot
picked up the wrong carton. This is one qualitative example, not a general
failure-mechanism estimate. The selection rule and both episode records are
preserved in the adjacent JSON file.

From the repository root, regenerate this archive's summaries and figures:

```bash
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50.yaml"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/aggregate_results.py --mode smoke --results-dir results/p50 \
  --findings-out results/p50/FINDINGS.md
```

The top-level README contains exact server and Fir submission commands. Use a
fresh results directory for a rerun; existing rollouts are never overwritten.
