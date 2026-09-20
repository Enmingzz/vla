# Fresh EGL training and evaluation

Completed on 2026-09-19: **1500 fresh updates and all 500 formal evaluation
episodes**, with EGL throughout training and evaluation. The GPU job completed
normally on `fc10413` in **1h20m44s** and released its single H100; the CPU report
completed in five minutes. `provenance/final_checks.json` records successful
validation of all 500 videos, checkpoint file hashes, native inference
roundtrips and paired evaluation. The run was submitted from source commit
`bc8d9a1`; no historical trained checkpoint or outcome was reused.

Original H=5/H=20 success was **88% / 54%**; H=20 after 500/1000/1500 updates
achieved **87% / 82% / 87%**. The full table, including actions and wall time for
all episodes and successful episodes, is in [FINDINGS.md](FINDINGS.md).

Step 1500 improved over original H=20 by **33 pp** (paired 95% CI 24–42 pp,
Holm-adjusted exact McNemar p=2.89e-7), recovering 97.1% of the observed 34 pp
replanning gap. It used 73.0% fewer policy calls per episode than original H=5.
Its mean episode time was 8.03 s versus 14.30 s at H=5 (43.8% lower); successful
episode means were 7.45 s versus 13.41 s (44.4% lower). At fixed four-worker
concurrency, whole-evaluation time was 326.26 s versus 451.42 s (27.7% lower).
These are different timing measures; episode means include shared-server
waiting and successful-episode means may use different subsets.

Step 500 already achieved 87%, so the extra 1000 updates did not increase the
aggregate success on these 100 episodes. The step-1500 minus H=5 paired
interval was -9 to +7 pp, which does not establish equivalence. The step-1500
minus step-1000 gain was +5 pp, with a -2 to +12 pp interval and exact p=0.302.
Training recovery is supported on these previously inspected layouts held out
from training; further-step benefits and new-seed/task generalization remain
unestablished.

CPU preflight `60516311` completed successfully in 155 seconds on `fc30555`:
all 64 unit tests, native sampling, diagnostic rollback, checkpoint roundtrip,
optimizer/EMA continuation and the training/evaluation initial-state audit
passed. The GPU job also passed the exact serial/parallel EGL observation check,
diagnostic rollback and both runtime pilots before formal evaluation/training.

- CPU checks: `60516311`.
- One H100, 8 CPU cores, 64 GiB RAM, two-hour cap: `60516312`, account `def-btaati_gpu`.
- CPU report: `60516313`.

The [fixed protocol](../../EGL_PIPELINE.md) trained 1500 updates and evaluated
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

`FINDINGS.md` and `aggregated/conditions.csv` contain accuracy,
actions, policy calls and wall time, including successful-episode averages.
All 500 videos and saved checkpoint hashes were validated on CPU after the GPU
allocation was released. Paired comparisons include confidence intervals and
Holm-adjusted exact McNemar tests.

The pending GPU job was changed in place from `rrg-btaati_gpu` to
`def-btaati_gpu` at the user's request on 2026-09-19. Its job ID and dependent
CPU summary remain the same; see [the account-change record](provenance/account_change.json).
