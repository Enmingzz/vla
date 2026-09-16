# Round 2: longer OPSD training, horizons and suite transfer

This archive follows the [predeclared protocol](../../AUTORESEARCH_ROUND2.md).
The immutable matrix is [autoresearch_round2.yaml](../../configs/autoresearch_round2.yaml),
with 1,310 evaluation episodes across LIBERO-10/spatial/object/goal. The primary
comparison is the preselected step-500 model versus the original model at H=20,
on 100 paired LIBERO-10 episodes excluded from our OPSD training and screen.

**Completed result:** original H=5 scored **93/100**. Original, step-100 and
step-500 H=20 scored **49/100, 50/100 and 72/100**. The primary 500 − original
change is **+23 pp**, paired 95% CI **[14, 32] pp**, exact McNemar p=0.000606.
The 500 − 100 change is +22 pp (CI [13, 31], secondary Holm p=0.0104).
The 500-update model recovered 52.3% of the original H=5/H=20 gap, with 68.3%
fewer calls per episode than H=5. Four task-level point estimates declined.
The [findings](FINDINGS.md) retain those regressions, the inconclusive independent
100-update result, and all horizon/suite screens with multiplicity corrections.

Code revision for training/evaluation: `1916f4c885e1fe0724bea6ea779487a2b11d8038`.
The retry job is `60080314`: one H100, eight CPU cores, 64 GiB host memory,
two-hour upper bound. It completed successfully in **1h42m58s** on `fc10601`;
the GPU allocation is released. Peak host RAM was 45.37 GiB. Exact submission,
environment and parent checkpoint are in [resource_plan.json](provenance/resource_plan.json).
Total round-two GPU allocation time is **1h56m50s**, including the aborted first
attempt; see [resource_accounting.json](provenance/resource_accounting.json).
The first attempt, `60078880`, was
cancelled for severely degraded simulator throughput before any completed benchmark
episode or formal added update; it used 13m52s and is retained in
[the diagnostic archive](../diagnostics/autoresearch_round2_attempt1/README.md).
The retry requires single/four-process runtime pilots on training layouts, each
with a 180-second bound, and uses the prior XLA memory fraction 0.65. These five
pilot episodes are excluded from statistics; the root cause of the stall remains
unresolved.

The parent is the complete [100-update pilot](../opsd_h20_100/README.md), pinned
by checkpoint manifest digest
`a7d6b7cb8cf42e2285721c007515b4a6526e4bb8c1874028f1dae841b5238a8c`.
Its FP32 master weights, EMA and Adam moments/counters are restored. Updates
101–500 keep the algorithm and optimizer unchanged; all training stays at H=20.
The simulator starts new episodes at each continuation boundary. Frozen
original/100/300/500 parameter snapshots share one policy server for evaluation.

All conditions use P=50 and 10 flow steps. This is the explicitly authorized
sequence-length extension of the official P=10 checkpoint and an experimental
temporal OPSD adaptation with Gaussian velocity matching. It is not a faithful
implementation of the image-generation Flow-OPD algorithm.

The [split audit](provenance/split_audit.json) verifies actual initial-state hashes.
Screening uses LIBERO-10 states 20–24, transfer states 20–22, and confirmation
states 30–39. Training uses LIBERO-10 indices 10–19; the prior 100 updates actually
visited indices 10–11. These indices identify initial object layouts, not single
observations or optimizer steps. The official checkpoint's original training data
is separate from this added OPSD train/evaluation split.

In the first 100 updates, task IDs 0–7 visited initial-layout indices 10 and 11;
tasks 8–9 visited index 10. Those 18 task/layout combinations produced 400
20-action training blocks and 7,961 executed actions. An update consumes one
block from each of four ongoing environments, so 100 updates do not imply
100 different starting layouts. The continuation actually visited 36 additional
task/layout combinations, indices 12–15, with 1,600 blocks and 31,829 actions.
Across all 500 updates, that is 54 distinct task/layout combinations and 39,790
executed actions. Evaluation layouts remain disjoint from all of them.

## Files

- `evaluations/{screen,transfer,confirmation}/step_*/raw/`: episode JSONL and
  complete task manifests, including source/config/checkpoint/RNG identities.
- `evaluations/*/step_*/videos/`: a video for each measured episode (local, not in Git).
- `aggregated/`: condition, task and episode CSVs; paired changes, confidence
  intervals, exact McNemar tests and multiplicity corrections.
- `figures/`: horizon, training-step, suite, compute/accuracy and loss plots.
- `figures/replanning_gap.png`: paired original H=5 minus H=15/20/25 differences.
- `training.jsonl`, `rollouts.jsonl`: every added optimizer update and rollout identity.
- `provenance/`: plan, checks, checkpoint lineage, Slurm resources and data audits.
- `stages.jsonl`: exact commands and stage timings, in execution order.
- `FINDINGS.md`: generated only from the completed measurements.

Full checkpoints and replayable training images/actions live under the scratch
path recorded in `provenance/resource_plan.json`, with milestone manifests and
file hashes. Keep that directory when moving the run; Git does not contain weights.

## Reproduce

Install dependencies and obtain the official checkpoint using the project
[README](../../README.md). The exact CPU checks, split audit, one-GPU submission,
individual H=25 evaluation and checkpoint-loading commands are in
[TRAINING.md](../../TRAINING.md#round-2-continue-to-500-updates-and-test-transfer).
Use fresh result/checkpoint paths for a rerun.

To recreate all tables, plots and findings from the completed matrix:

```bash
source scripts/env.sh
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.study_analysis --results-dir results/autoresearch_round2
```

Run from the repository root. No GPU is needed for aggregation. The analyzer
requires every planned condition, checks the actual state hashes, parameter
snapshot, one-server identity and unchanged inference settings, and refuses
incomplete or mismatched data. No screen outcome selects a checkpoint.
