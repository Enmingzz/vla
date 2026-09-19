# Fresh EGL training and evaluation

**This attempt stopped in CPU preflight; its GPU job was cancelled before allocation.**
No training or evaluation ran. The CPU fixture failed while importing SciPy; see
[the preserved traceback](provenance/preflight_failure.txt). All 64 unit tests passed.
The [fresh retry](../egl_from_scratch_1500_retry1/README.md) excludes the CPU node
that returned a scratch filesystem read error.

Submitted on 2026-09-19. This archive is a new run from the official checkpoint,
with EGL throughout training and evaluation. It contains no reused outcomes or
mixed-renderer training checkpoints. A successful submission is not a completed
experiment: completion requires `provenance/final_checks.json` with `complete=true`.

- CPU preflight: `60516007`.
- One H100 for training and evaluation: `60516008`.
- CPU validation and report: `60516009`.
- Source commit at submission: `b27a86a`.
- GPU resources: 1 H100, 8 CPU cores, 64 GiB RAM; two-hour wall-time cap.

The [fixed protocol and reproduction commands](../../EGL_PIPELINE.md) run 1500
fresh updates, with checkpoints at 500/1000/1500 and 500 formal evaluation
episodes: original H=5/H=20 plus H=20 at each trained checkpoint. All evaluations
use the same seed, state order, EGL renderer, four workers and continuous server.
There are five separate runtime-pilot episodes excluded from formal statistics.

The pipeline must pass CPU validation and EGL runtime checks before its formal
work. Renderer errors stop the run without an OSMesa fallback. On completion,
`FINDINGS.md` and `aggregated/conditions.csv` provide the unified accuracy,
actions, policy calls and time table, including successful-episode averages.
Videos stay on the local filesystem and are excluded from Git.

```bash
squeue -j 60516007,60516008,60516009
sacct -j 60516007,60516008,60516009 -o JobID,State,Elapsed,ExitCode
```
