# Complete the interrupted RoboCasa365 evaluations

## Measured outcome

Job 60911562 ran for 38m25s on fc10619. The two original conditions and the
trained H=20 condition completed all 100 paired episodes. The last H=5 condition
aborted after 43 episodes because PickPlaceSinkToCounter episode 0 failed its
strict reset checks on all three same-seed attempts. Missing episodes were not
scored as failures. No additional training took place.

| Condition | Success | Policy calls/episode | Actions/episode | Rollout seconds/episode |
|---|---:|---:|---:|---:|
| Original H=5 | 54/100 | 111.29 | 555.48 | 31.52 |
| Original H=20 | 58/100 | 25.42 | 501.04 | 20.13 |
| Step-500 H=20 | 35/100 | 32.02 | 633.53 | 25.49 |
| Step-500 H=5 | 2/43, incomplete | — | — | — |

On the same 43 completed H=5 episodes, the original model scored 17/43 and the
trained model 2/43. This task-order-dependent subset is descriptive only.

The completed original comparison does not support a sparse-replanning penalty
on these ten tasks: H=20 minus H=5 is +4 percentage points, paired 95% bootstrap
CI [-7, +15], exact McNemar p=0.5966. H=20 uses 77.2% fewer policy calls/episode.
The completed OPSD comparison shows deterioration: 35% versus 58%, a difference
of -23 percentage points, paired 95% CI [-34, -12], p=0.0002941. These exploratory
intervals/tests are conditional on the fixed ten tasks. Timings span allocations
and exclude environment reset/server startup.

The saved failure states isolated the remaining reset bug: upstream
`Counter.get_reset_regions` deduplicates XML elements through `list(set(...))`,
making the association of `geom_0`/`geom_1` with left/right counter regions
depend on element identity. The two recorded object regions were swapped despite
the same seed. This was a physical placement mismatch, not the earlier XML MIME
serialization difference. A subsequent recovery restores only the recorded
region labels before the native sampler and keeps every state/image check.

## Original submission

Slurm job **60911562**, `rc365_finish_eval`: one H100, eight CPUs, 64 GiB memory,
`def-btaati`, 75-minute scheduler cap. The job exits when the work finishes.
Submitted on 2026-09-21; no new evaluation results existed at submission.

This is evaluation only. Reuse the saved step-500 checkpoint from
`robocasa365_fixed10_500_retry1`, with manifest digest
`a8f369fa2278e0d6d46f68c8d757fc7839959544a697e7f205287924de5d6a1d`.
Perform zero additional optimizer updates. Keep the original experiment archive.

| Condition | Verified existing episodes | New episodes required |
|---|---:|---:|
| Original H=5 | 60 | 40 |
| Original H=20 | 100 | 0 |
| Step-500 H=20 | 33 | 67 |
| Step-500 H=5 | 0 | 100 |

All 193 cached rows passed checks against the original inference specification,
checkpoint, exact episode identities, raw videos and policy-call counts before
submission. Runtime repeats those checks against the actual loaded server.
Each reused row keeps its measured values and original inference fingerprint,
with a `reused_record_source` path; task provenance includes its source server.
New records keep the actual raw XML hash and a comparison hash which removes
only inferred `model/obj` MIME annotations on `.obj` mesh files. State, images,
language, metadata and all numerical XML attributes remain strict. No exported
XML is reloaded. Unpaired resets may retry the same seed at most three times
before any policy inference, preserving rejected state artifacts.

The existing backend's `load_snapshot` and `set_phase` calls select the checkpoint.
The `student` phase alias becomes `step_500`; this is the only allowed difference
when comparing saved versus current inference specifications. The native
sampler, normalization, checkpoint weights, P=50 and ten flow steps are unchanged.

Thirteen CPU tests passed. Diagnostic job 60911009 completed eight same-seed
native reset comparisons in 2m40s; all physical states, metadata and model inputs
matched exactly. Direct XML replay changed observations and was rejected.

`submission_sources.json` freezes the implementation. The job will write raw
records, aggregates, videos and `FINDINGS.md`. Existing negative results are
retained. Until each condition is complete, use only explicitly matched partial
episode comparisons; do not compare a partial mean against all 100 baseline rows.

```bash
squeue -j 60911562
```
