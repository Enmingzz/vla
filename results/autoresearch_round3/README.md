# Fixed 500-update continuation

This archive is for the requested continuation from step 500 through step 1000.
The design and exact training commands are in
[AUTORESEARCH_ROUND3.md](../../AUTORESEARCH_ROUND3.md).
No new measured result is implied by the submission or preflight files.

GPU job **60146846** runs the diagnostic, restores the complete step-500 state,
evaluates step 500, performs exactly 500 additional updates, saves step 1000,
and evaluates step 1000. Both formal conditions use H=20, P=50, LIBERO-10,
seed 27 and official initial-state indices 30–39: 100 episodes per checkpoint,
200 total. Only one H100 is requested, and it is released when the GPU job exits.

CPU job **60147552** depends on successful completion of the GPU job. It generates
the tables, figures and findings, checks all 200 videos, verifies the exported
checkpoint metadata, and records GPU accounting. The exact postprocessing
scripts and submission details are preserved in `provenance/`.

Check execution without allocating a GPU:

```bash
squeue -j 60146846,60147552
sacct -j 60146846,60147552 -o JobID,State,Elapsed,ExitCode
tail -n 5 results/autoresearch_round3/stages.jsonl
```

Successful completion requires `provenance/final_checks.json` with
`complete: true`; preflight success alone does not establish a training result.
The completed outputs will be `FINDINGS.md`, `aggregated/conditions.csv`,
`aggregated/comparisons.csv`, `aggregated/per_task.csv`,
`aggregated/episodes.csv`, `figures/continuation_comparison.png`, and
`figures/training_loss.png`. Raw episode JSONL and videos reside under
`evaluations/confirmation/step_500/` and `step_1000/`. Training updates and
on-policy state provenance are in `training.jsonl` and `rollouts.jsonl`.

Both checkpoints are re-evaluated under pinned OSMesa on one continuous server.
The historical EGL 72% is not substituted for the current step-500 baseline.
The continuation includes a renderer/context-selection correction relative to
the parent training; these evaluation layouts have also been inspected before.
It is an exploratory continuation, not a new blind confirmation or an isolated
optimizer-step ablation on fixed data.
