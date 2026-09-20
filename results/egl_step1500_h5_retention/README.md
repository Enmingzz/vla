# Step-1500 H=5 retention check

Completed job **60558779** on one H100 under `def-btaati_gpu` in **20m43s**;
the allocation is released. Original H=5 achieved **88/100**, and the step-1500
student at H=5 achieved **86/100**. The paired difference is **−2 pp**, 95% CI
**[−10, +6] pp**, exact McNemar **p=0.814529**. This small evaluation does not
establish a clear degradation or prove equivalence. See [FINDINGS.md](FINDINGS.md).

The original-model repeat took **541.89 seconds**. All 100 success outcomes,
action counts, call schedules, initial states, first observations and RNG seeds
matched the historical reference exactly. The cached original H=5 data therefore
give the same success comparison here. Prefer reusing this verified reference for
future follow-ups with matching protocols unless evidence requires another control.
The comparison is recorded in `provenance/historical_reproduction_check.json`.

Evaluation-only follow-up to `../egl_from_scratch_1500_retry1`: compare the original
checkpoint and that run's student after 1500 updates, both at P=50 and H=5.
Both are freshly measured on one H100 and one continuous native sampler/server.
The saved student is evaluated, not its EMA teacher. No optimizer is instantiated.

Protocol: LIBERO-10, all ten tasks, seed 27, official initial-state indices 30–39,
ten episodes per task, EGL, four simulator workers, ten flow steps. The two
conditions total 200 episodes. The earlier original H=5 result, 88/100, is recorded
as a historical reference; the fresh original H=5 measurement is the primary
control because independent GPU compilations need not be bitwise reproducible.

The checkpoint manifest and original experiment artifacts are pinned before
submission. The existing frozen-comparison loader verifies inference-parameter
checksums, normalization assets, and equality of the frozen backbone. Analysis
requires complete paired coverage, identical initial states and first observations,
matching RNG/evaluator settings, one server, exact H=5 call schedules, and videos.
It reports a task-stratified paired bootstrap interval and exact McNemar test.
The layouts were held out from added OPSD training but were previously inspected;
this is neither an unseen-task test nor a native-P=10 retention test.

## Reproduce

Use the installed environments and checkpoint described in the project README.
Choose a fresh output directory. The checkpoint path is read from the completed
source archive's `provenance/step_1500.json`.

```bash
source scripts/env.sh
export RUN_RESULTS="$PWD/results/egl_step1500_h5_retention-rerun"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/evaluate_h5_retention.py prepare --results-dir "$RUN_RESULTS" \
  --source-results "$PWD/results/egl_from_scratch_1500_retry1"
sbatch --account=def-btaati --job-name=vla-h5-retain \
  --nodes=1 --ntasks=1 --gpus-per-node=h100:1 --cpus-per-task=8 --mem=48G \
  --exclude=fc10501,fc10511 --time=00:30:00 \
  --output="$RUN_RESULTS/logs/gpu-%j.log" scripts/fir_h5_retention.sh
```

The batch job evaluates both snapshots, stops the server, writes the paired
analysis, and exits immediately. Recreate the report without a GPU:

```bash
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/evaluate_h5_retention.py analyze --results-dir "$RUN_RESULTS"
```

Completed measurements appear in `FINDINGS.md`, `aggregated/`, and `evaluations/`.
`provenance/final_checks.json` records the completed allocation, 200 episodes and
video files, exact historical baseline reproduction, and report checksums.
