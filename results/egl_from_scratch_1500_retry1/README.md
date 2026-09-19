# Fresh EGL training and evaluation

Submitted on 2026-09-19 from source commit `bc8d9a1`. This is a fresh official
checkpoint run with EGL throughout. Completion requires
`provenance/final_checks.json` with `complete=true`; submission alone is not a
completed result.

CPU preflight `60516311` completed successfully in 155 seconds on `fc30555`:
all 64 unit tests, native sampling, diagnostic rollback, checkpoint roundtrip,
optimizer/EMA continuation and the training/evaluation initial-state audit
passed. GPU runtime rendering checks still run inside the dependent GPU job.

- CPU checks: `60516311`.
- One H100, 8 CPU cores, 64 GiB RAM, two-hour cap: `60516312`, account `def-btaati_gpu`.
- CPU report: `60516313`.

The [fixed protocol](../../EGL_PIPELINE.md) trains 1500 updates and evaluates
original H=5/H=20 and H=20 after 500/1000/1500 updates. All five conditions use
100 LIBERO-10 episodes, seed 27, initial-state indices 30–39, P=50, 10 flow steps,
EGL and four simulator workers on one continuous policy server. No historical
checkpoints or evaluation outcomes are reused. Five runtime-pilot episodes stay
outside the formal statistics.

The preceding [attempt](../egl_from_scratch_1500/README.md) stopped on CPU node
`fc30560`: reading installed SciPy binaries on scratch returned
`BrokenPipeError: [Errno 108] Cannot send after transport endpoint shutdown`.
Its dependent GPU job never allocated resources. This retry excludes that CPU
node and verifies the unchanged SciPy binaries against their installed RECORD
before running the CPU tests. No dependency, model, optimizer or renderer was
changed to bypass the failure.

Reproduction after the documented setup, using new paths:

```bash
source scripts/env.sh
export EGL_GPU_ACCOUNT=def-btaati
export RUN_RESULTS="$PWD/results/new_egl_run"
export OPSD_CHECKPOINT_ROOT="$FREQUENCY_WORK/runs/new_egl_run"
bash scripts/submit_egl_pipeline.sh
```

On completion, `FINDINGS.md` and `aggregated/conditions.csv` contain accuracy,
actions, policy calls and wall time, including successful-episode averages.
All 500 videos and saved checkpoint hashes are validated on CPU after the GPU
allocation is released. Paired comparisons include confidence intervals and
Holm-adjusted exact McNemar tests.

The pending GPU job was changed in place from `rrg-btaati_gpu` to
`def-btaati_gpu` at the user's request on 2026-09-19. Its job ID and dependent
CPU summary remain the same; see [the account-change record](provenance/account_change.json).
