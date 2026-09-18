# Fixed 500-update continuation

This archive is for the requested continuation from step 500 through step 1000.
The design and exact training commands are in
[AUTORESEARCH_ROUND3.md](../../AUTORESEARCH_ROUND3.md).
The one-hour allocation finished normally on **2026-09-17, 15:12:53–16:09:51
America/Vancouver**, using one H100 for **56m58s**. It added **478 updates**,
from step 500 through **step 978**. The preset save reserve stopped the run;
**22 updates remain** to the step-1000 target. The checkpoint was saved and all
file checksums and native inference round-trip checks passed. GPU and CPU
validation jobs are complete and the GPU is released.

The training stage took 47m55.68s; checkpoint export and verification in the GPU
job took 85.73s. The 1,912 on-policy action blocks executed 37,440 actions and
visited 100 distinct LIBERO-10 task/layout combinations across official state
indices 10–19. These training states remain disjoint from the deferred evaluation.
The extra updates also supplied fresh on-policy experience, not fixed-data
optimization alone. See [TRAINING_STATUS.md](TRAINING_STATUS.md) and
[the validation record](provenance/training_only_validation.json).

**No formal evaluation episode ran. Improvement in task success is unknown.**

At the user's request on 2026-09-17, GPU job **60146846** was amended in place to
**one hour, training only**. It ran the diagnostic, restored the complete
step-500 state, aimed for 500 additional updates, saved the completed checkpoint,
and released the single H100. Five minutes before the actual allocation end,
the client stops at a completed update so FP32/EMA/Adam state can be exported.
If fewer than 500 updates fit, the actual count and remaining work are reported.
The job ID and original submission timestamp are retained.

The original CPU analysis job **60147552** was cancelled without running because
the 200-episode evaluation is deferred. The replacement CPU validation job is
recorded in `provenance/training_validation_submission.json`; job **60252387**
verified the actual update count, all checkpoint files and GPU accounting in
13 seconds on CPU. It generated
`TRAINING_STATUS.md` and `provenance/training_only_validation.json`.

The deferred comparison remains step 500 versus step 1000, at H=20, P=50,
LIBERO-10, seed 27 and official indices 30–39: 100 episodes per checkpoint.
No success rate or training benefit can be inferred from a training-only job.

Check execution without allocating a GPU:

```bash
squeue -j 60146846
sacct -j 60146846 -o JobID,State,Elapsed,ExitCode
tail -n 5 results/autoresearch_round3/stages.jsonl
```

Completed training requires `provenance/training_only_validation.json` with
both `validation_passed: true` and `requested_updates_complete: true`.
Preflight success or a valid partial checkpoint is not 500 completed updates.
The later evaluation outputs will be `FINDINGS.md`, `aggregated/conditions.csv`,
`aggregated/comparisons.csv`, `aggregated/per_task.csv`,
`aggregated/episodes.csv`, `figures/continuation_comparison.png`, and
`figures/training_loss.png`. Raw episode JSONL and videos reside under
`evaluations/confirmation/step_500/` and `step_1000/`. Training updates and
on-policy state provenance are in `training.jsonl` and `rollouts.jsonl`.

Both checkpoints will be evaluated under pinned OSMesa on one continuous server.
The historical EGL 72% is not substituted for the current step-500 baseline.
The continuation includes a renderer/context-selection correction relative to
the parent training; these evaluation layouts have also been inspected before.
It is an exploratory continuation, not a new blind confirmation or an isolated
optimizer-step ablation on fixed data.
