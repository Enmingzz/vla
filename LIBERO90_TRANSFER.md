# LIBERO-10 OPSD transfer to LIBERO-90

This evaluation reuses the frozen 500-update LIBERO-10 model from round two
to test transfer outside its added OPSD training tasks. No new training is
performed for this experiment.

## Fixed experiment

Evaluate **all 90 official LIBERO-90 tasks**, three ordered initial layouts
40–42 per task, seed 37. Compare original H=5, original H=20, and the preselected
step-500 model at H=20: **810 formal episodes**. The original and trained weights
share one continuous native inference server; switching snapshots does not run
an optimizer. P=50 and the 10 native flow steps stay fixed. The existing official
LIBERO evaluator determines success and uses its LIBERO-90 limit of 400 controlled
steps, plus ten settling steps. Simulator/inference errors abort the run rather
than being counted as task failures.

The exact matrix and checkpoint digests are in `configs/libero90_transfer.yaml`.
The step-500 manifest must equal
`c00bcab9c776c6e3794b4b86dc18a15b0e78dd9e7fcd9fb0fa1c7f2d920f18ed`.
No checkpoint, task or horizon is selected by these new outcomes.

## Meaning of task generalization

Before GPU use, audit the actual 500 prior training updates against LIBERO-10
task descriptions and state arrays. Check all LIBERO-90 task identities and
selected initial states. Names are distinct, but an instruction about putting a
book in a caddy also occurs in LIBERO-10. Therefore the **primary analysis uses
only task instructions not present in added OPSD training**; all 90 tasks and the
overlapping-instruction subset are also reported. The exclusion is defined from
task metadata before observing outcomes. The complete task list remains tested.

This is transfer to tasks held out from **our added OPSD**, not a claim that the
official base checkpoint never encountered the tasks. Its fine-tuning config
references `physical-intelligence/libero`. Current public task metadata is archived
at its resolved revision, but cannot establish the historical training revision or
the full earlier pretraining corpus. Shared objects, scenes or subskills may remain.
The method is the existing temporal OPSD Gaussian velocity-matching adaptation;
P=50 is the explicitly authorized extension of the official native P=10 model.

## Analysis and resources

Primary comparison: trained minus original H=20 success on the predeclared novel
instruction tasks. Report exact paired McNemar and a 10,000-replicate bootstrap
within tasks. Also report an exploratory hierarchical bootstrap resampling tasks
and layouts, since transfer should not depend on a few favorable tasks. Secondary
comparisons are the all-90 training effect and original H=5 minus H=20 gaps for
the primary subset and all 90 tasks; apply Holm to that secondary family. Task
tables include regressions. Three layouts per task is a broad, small-per-task
screen; individual task changes are descriptive.

One H100, 12 CPU cores and 64 GiB host memory; a 4h45m upper bound, with
immediate release on completion or error. Eight simulator workers each use one
software rasterizer thread. First run single/eight-worker rendering guards
(360/600-second bounds), using layout 0 and including task ID 89. These nine pilot
episodes are excluded from formal statistics. Each formal condition has a
5,400-second runtime bound. These bounds allow cold dependency imports for each
task process as well as software rendering; they do not change the 400-action
episode limit or reserve additional GPUs. Saved parameters/assets are checksum-checked;
the frozen backbone and normalization assets must match the original checkpoint.
The comparison has no optimizer or training endpoints. Record any failed GPU time.

### Rendering amendment before formal evaluation

Three EGL attempts aborted during their single-worker pilot, before any complete
episode or formal outcome. Native GDB tracing located `abort()` inside NVIDIA
`libnvidia-eglcore.so.580.159.03`, called by MuJoCo's `mjr_readPixels`. Exact action
replay without the policy model passed; disabling JAX preallocation did not fix
the co-resident model/renderer failure. The precise driver cause remains unresolved.
Diagnostics are retained under `results/diagnostics/libero90_*`; every GPU attempt
counts toward resource use, including GDB jobs whose Slurm status alone is misleading.

All formal conditions therefore use **the same OSMesa software renderer**, while
keeping the model, ten flow steps, official execution loop, image preprocessing,
task list, ordered states and statistical comparisons fixed. The runtime-only plan
amendment has digest `d3f293c11475f62a280f5e832e78d23f10d190537414d3154d71ae578235090d`;
the earlier plan is archived with the failed attempts. Renderer identity, its
library checksum, and thread settings enter every evaluation fingerprint.

The pinned EL9 Mesa 25.0.7 build is extracted into user-owned scratch storage;
the system and Python dependencies are not replaced. It requires compatible
system LLVM 20 libraries. The cluster's CVMFS OSMesa has a different glibc and
cannot be used with this standalone Python. See `scripts/setup_osmesa_fir.sh`.
Software rendering passed 400 random-action steps and the exact 261-action crash
prefix. Comparing five corresponding frames from both cameras across GPU/CPU
nodes gave mean absolute channel differences of 0.83–1.54 on the 0–255 scale.
They are **not pixel-identical**, so the measured result is specific to this
documented renderer. No EGL measurements are mixed into the formal statistics.

The first full-model software-rendering pilot reached 385 of 400 controlled
steps without a native crash, but its 180-second process timeout included cold
imports and stopped the rollout. Its partial episode is archived, not scored.
The time bounds above were extended before any formal result was observed.

## Reproduce

Install and obtain the official checkpoint with README.md; reuse the saved
step-500 checkpoint and both training archives. Choose a new result directory.

```bash
source scripts/env.sh
bash scripts/setup_osmesa_fir.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50_libero90.yaml"
export RUN_RESULTS="$PWD/results/libero90_transfer-rerun"
export COMPARISON_CHECKPOINT="$FREQUENCY_WORK/runs/autoresearch_round2_attempt2/step_500"
mkdir -p "$RUN_RESULTS/logs"

# CPU audit: verifies checkpoint ancestry and every actual training task/layout.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.libero90_transfer audit --results-dir "$RUN_RESULTS" \
  --checkpoint "$COMPARISON_CHECKPOINT" \
  --prior-results results/opsd_h20_100 results/autoresearch_round2

sbatch --job-name=vla-libero90-transfer --account=rrg-btaati --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=12 --mem=64G --time=04:45:00 \
  --output="$RUN_RESULTS/logs/slurm-%j.log" \
  scripts/fir_libero90_job.sh

# CPU aggregation after all 810 episodes complete.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.libero90_analysis --results-dir "$RUN_RESULTS"
```

For a manual GPU allocation, start the server with:

```bash
bash scripts/serve_policy.sh --port 8000 \
  --comparison-checkpoint "$COMPARISON_CHECKPOINT" \
  --comparison-results-dir "$RUN_RESULTS" \
  --manifest-out "$RUN_RESULTS/provenance/startup_server.json"
```

Then run `frequency_vla.libero90_transfer run --port 8000 --results-dir "$RUN_RESULTS"`
with the LIBERO Python and these simulator settings:

```bash
export FREQUENCY_OSMESA_LIBRARY_DIR="$FREQUENCY_WORK/native_osmesa/root/usr/lib64"
env -u PYTHONPATH -u PYTHONHOME MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH="$FREQUENCY_OSMESA_LIBRARY_DIR" \
  "$LIBERO_VENV/bin/python" -m frequency_vla.libero90_transfer run \
  --port 8000 --results-dir "$RUN_RESULTS"
```

The dedicated driver evaluates every task through
`opsd_evaluate`, including IDs above 9, and selects both frozen snapshots.
