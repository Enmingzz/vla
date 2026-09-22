# Completed RoboCasa recovery: all 400 evaluation episodes

Job `60918340` completed successfully in **23m27s**. All 57 outstanding native
reset preflight checks passed. The existing 343 measured outcomes were reused
without alteration, and the remaining 57 trained-H5 episodes completed with
15 additional successes: **step-500 H5 = 17/100**. No optimizer updates ran.

| Condition | Success | Mean calls | Mean actions | Rollout seconds |
|---|---:|---:|---:|---:|
| Original H5 | 54/100 | 111.29 | 555.48 | 31.52 |
| Original H20 | 58/100 | 25.42 | 501.04 | 20.13 |
| Step-500 H20 | 35/100 | 32.02 | 633.53 | 25.49 |
| Step-500 H5 | 17/100 | 137.96 | 689.48 | 38.97 |

The paired H5 retention change is **−37 percentage points**, 95% bootstrap CI
[−47, −27], McNemar p=1.455e-10. The frequency premise was not established on
these ten tasks and the 500-step OPSD run degraded both H settings. These
negative results are retained when moving to SimplerEnv. See [FINDINGS.md](FINDINGS.md).

## Submission record

Slurm job **60918340**, `rc365_finish_eval`: one H100, eight CPUs, 64 GiB memory,
`def-btaati`, 45-minute scheduler cap. Submitted after 15 CPU tests passed and
all 343 existing episode records were validated. No new results at submission.

Source archive: `robocasa365_fixed10_500_recovery1`. Immutable episode catalog:
`robocasa365_fixed10_500_retry1/paired_episodes`. Use the same saved step-500
checkpoint, manifest
`a8f369fa2278e0d6d46f68c8d757fc7839959544a697e7f205287924de5d6a1d`.
Zero additional optimizer updates; preserve every prior measured outcome.

| Condition | Verified completed episodes | Existing successes | Remaining |
|---|---:|---:|---:|
| Original H=5 | 100 | 54 | 0 |
| Original H=20 | 100 | 58 | 0 |
| Step-500 H=20 | 100 | 35 | 0 |
| Step-500 H=5 | 43 | 2 | 57 |

The upstream counter-region bug maps the same RNG-selected label to different
left/right placements because `Counter.get_reset_regions` uses a set of XML
elements. The recovery wrapper restores an unambiguous mapping from the saved
episode metadata using only exact native geometry. It neither changes numeric
geometry nor consumes random draws, injects simulator state or reloads XML.
Physical state, metadata, language and preprocessed observation checks remain
exact. This affects evaluation initialization only, never training or inference.

Before loading model weights, the job checks all 57 outstanding initial states
with the renderer. The preflight has a ten-minute cap and must pass. Only then
does it load the frozen original/step-500 parameters, verify cached records and
evaluate missing episodes. The original 343 records retain their measured values,
videos and inference fingerprints, with validated source-server provenance.

The completed comparisons already show no established H5 advantage on these ten
tasks and a -23-point H20 OPSD change (95% paired CI [-34, -12]). The unfinished
H5 subset scores 2/43 versus the original model's 17/43 on the same episodes;
that subset is not a full ten-task mean.

```bash
squeue -j 60918340
```
