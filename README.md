# π0.5 LIBERO replanning-frequency validation

The initial evaluation phase uses the official Physical Intelligence OpenPI
checkpoint and its existing LIBERO action-execution loop, without training,
attention-rule changes, or action repetition. The default protocol preserves the
official prediction horizon; an explicitly selected P=50 extension is documented below.
The subsequently authorized 100-step H=20 recovery experiment is specified in
[AUTORESEARCH.md](AUTORESEARCH.md) and keeps its learned weights and results separate.
Exact commands for that phase are in [TRAINING.md](TRAINING.md).

The first recovery pilot is complete: at fixed **P=50, H=20**, 100 temporal OPSD
updates increased paired LIBERO-10 success from **46% to 58%** (100 episodes;
change +12 pp, paired 95% CI +2 to +22 pp, exact McNemar p=0.0428).
This is preliminary evidence from a continuous-action adaptation of the local
OPSD pipeline, not a reproduction of the image-generation Flow-OPD algorithm.
See the [training findings](results/opsd_h20_100/FINDINGS.md),
[before/after figure](results/opsd_h20_100/figures/success_before_after_100.png),
and [complete run archive](results/opsd_h20_100/README.md).

The follow-up is complete: on 100 new paired LIBERO-10 confirmation episodes,
original H=5 achieved **93%**; at H=20, original/100/500 updates achieved
**49% / 50% / 72%**. The preselected 500-update model improved by **23 pp**
(paired 95% CI 14–32 pp, exact McNemar p=0.000606), recovering 52.3% of the
original replanning gap while using 68.3% fewer calls/episode than original H=5.
The additional 400 updates also improved over step 100 (22 pp; adjusted p=0.0104).
The 100-update benefit alone did not clearly replicate on this split.
Deployment H=15/25 and spatial/object/goal were also measured, with task-level
regressions and corrected exploratory comparisons retained. All training used H=20.
See [round-two findings](results/autoresearch_round2/FINDINGS.md) and the
[training-step figure](results/autoresearch_round2/figures/success_vs_training_steps.png).
The 1,310-episode round used one H100 at a time, totaling 1h56m50s including an
aborted rendering-stall attempt; the allocation is released. Its fixed design is
in [AUTORESEARCH_ROUND2.md](AUTORESEARCH_ROUND2.md), commands in
[TRAINING.md](TRAINING.md#round-2-continue-to-500-updates-and-test-transfer), and
outputs in the separate [round-two archive](results/autoresearch_round2/README.md).

The completed cross-task evaluation was reduced at the user's request to
**LIBERO-90 task IDs 0–9**, using the existing 500-update model without further
training. The 90-episode screen reused 30 complete H=5 episodes and added 60 H=20
episodes. Original H=5 / original H=20 / trained H=20 achieved **16.7% / 20.0% /
13.3%**; the observed training change was −6.7 pp (paired p=0.5). This small
screen shows no demonstrated transfer benefit. See [the findings](results/libero90_first10/FINDINGS.md). The [first-ten archive](results/libero90_first10/README.md) keeps this
ordered exploratory prefix separate from the [interrupted full-suite attempt](results/libero90_transfer/README.md).
The amended protocol, original full-suite plan and exact commands are in
[LIBERO90_TRANSFER.md](LIBERO90_TRANSFER.md).

The requested one-hour continuation finished at **step 978: 478 additional
updates from step 500**, leaving 22 updates to the step-1000 target. It used one
H100 for **56m58s**, stopped for the checkpoint-save reserve, and released the
allocation normally. Checkpoint verification passed. No benchmark evaluation
ran, so there is no new success-rate result yet. See the
[training status](results/autoresearch_round3/TRAINING_STATUS.md),
[execution protocol](AUTORESEARCH_ROUND3.md) and
[round-three archive](results/autoresearch_round3/README.md).

The subsequently requested completion resumes step 978 for the remaining **22
updates**, then compares **step 500 and step 1000 at H=20** on 100 paired
LIBERO-10 episodes each under the same OSMesa renderer. GPU job **60427489**
and dependent CPU validation/reporting are submitted; results are not yet
available. The [completion archive](results/autoresearch_round3_finish/README.md)
contains the fixed protocol, provenance and reproduction commands.

**Protocol constraint discovered before implementation:** at pinned OpenPI commit
`215abfb217dbac7d5f1273282331b9b1866c0479`, `pi05_libero` explicitly configures
**P=10**, not 50. Therefore only **H=5 and H=10** from the requested sweep are
valid with the unchanged config. H=20/30/40/50 fail preflight. The requested six
horizons remain in `configs/frequency_sweep.yaml`; they are never silently dropped.
The intended large-H teacher/student premise cannot be answered without revising
the protocol. A supported two-horizon run is clearly labelled a partial experiment.

The completed main result (seed 7, 50 episodes/task) is **H=5: 92.0%, H=10: 95.0%**,
with 51.8% fewer policy calls per episode at H=10. See [FINDINGS.md](FINDINGS.md)
and the [result archive](results/README.md). The measured comparison does not
support H=5 as the stronger teacher.

## Explicit P=50 extension

The completed paired smoke test (seed 7, 10 episodes/task) measured **H=5: 86%,
H=30: 12%**, with 71.9% fewer policy calls per episode at H=30. The paired gap is
74 percentage points (95% CI 67–81). This is a large diagnostic failure at H=30,
and the P=50/H=5 baseline itself failed the predeclared reference screen. These
results do not establish the desired moderate, recoverable teacher/student gap.
See the separate [P=50 findings](results/p50/FINDINGS.md) and
[P=50 archive](results/p50/README.md). The original main result above is unchanged.

Following the native-P experiment, a separate requested extension fixes inference
P=50 and compares H=5 with H=30. Select `configs/prediction50.yaml` explicitly;
the default `frequency_sweep.yaml` and original results remain the native-P protocol.
The official loader receives a dataclass copy with only `model.action_horizon=50`
changed. It restores the same checkpoint, checking all required parameter shapes.
Flow integration remains 10 steps and the official evaluator executes the first H
actions of each returned 50-step chunk. No actions are padded or repeated.

**Interpretation:** this checkpoint's official fine-tuning configuration uses P=10.
P=50 is sequence-length extrapolation, and even the first five actions can change
because action tokens attend to one another. Always compare H values at the same P;
do not attribute a comparison of P=10/H=5 versus P=50/H=30 solely to replanning.
The 92.4% official reference is only a contextual health screen for this extension.
Both smoke conditions run even if P=50/H=5 fails that screen; such failure prevents
claiming a strong teacher. This is an evaluation extension, not new training.

On Fir, after installation, submit the paired smoke test (10 episodes/task/H):

```bash
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50.yaml"
export RUN_RESULTS="$PWD/results/p50-rerun"
mkdir -p "$RUN_RESULTS/logs"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$LIBERO_VENV/bin/python" scripts/preflight.py
sbatch --job-name=freq-p50-smoke --account=rrg-btaati --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=40G --time=00:35:00 \
  --output="$RUN_RESULTS/logs/slurm-%j.log" \
  scripts/fir_job.sh smoke --horizons 5 30 --workers 4
```

For a manual GPU session, export the same `FREQUENCY_CONFIG` in **both** server
and evaluator shells. Start the server with `bash scripts/serve_policy.sh`; then:

```bash
# One H, or both paired conditions. Choose a fresh result path for each run.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.evaluator --mode smoke --horizon 30 --results-dir results/p50-one-h
bash scripts/run_smoke_test.sh --horizons 5 30 --workers 4 --results-dir results/p50-paired

# Recreate extension tables, plots and its separate findings file, without a GPU.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/aggregate_results.py --mode smoke --results-dir results/p50-paired \
  --findings-out results/p50-paired/FINDINGS.md

# Return to the original, unchanged-P protocol.
unset FREQUENCY_CONFIG
```

Every episode records both official P=10 and effective inference P=50, the actual
chunk length, and a distinct inference fingerprint. Aggregation refuses mixed P
or mixed config fingerprints. The native-P archive is not pooled with this experiment.

### Follow-up H=15 and H=20

The completed paired smoke test measured **H=5: 88%, H=15: 82%, H=20: 46%**.
H=20 has a 42 pp gap (paired 95% CI 33–51 pp), with 62.2% fewer calls per episode.
The H=15 gap is inconclusive (6 pp, CI −2 to 15 pp). See the separate
[findings](results/p50_h15_h20/FINDINGS.md) and [archive](results/p50_h15_h20/README.md).

Use `configs/prediction50_h15_h20.yaml` to compare H=5, 15 and 20 at fixed P=50.
H=5 is rerun on the same continuous server, since separate GPU server instances
were not bitwise reproducible in the earlier diagnostic. All three conditions use
seed 7 and the same first 10 initial states per task. Results live separately from
the earlier H=5/30 archive.

```bash
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50_h15_h20.yaml"
export RUN_RESULTS="$PWD/results/p50_h15_h20-rerun"
export PILOT_FIRST=1 SERVER_READY_TIMEOUT_SECONDS=300 PILOT_TIMEOUT_SECONDS=180
mkdir -p "$RUN_RESULTS/logs"
sbatch --job-name=freq-p50-h15-h20 --account=rrg-btaati --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=40G --time=00:35:00 \
  --output="$RUN_RESULTS/logs/slurm-%j.log" \
  scripts/fir_job.sh smoke --horizons 5 15 20 --workers 4

# Recreate this follow-up's summaries and plots after it finishes.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/aggregate_results.py --mode smoke --results-dir "$RUN_RESULTS" \
  --findings-out "$RUN_RESULTS/FINDINGS.md"
```

Inspected upstream sources:

- [Official LIBERO instructions and 92.4% LIBERO-10 reference](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)
- [Official evaluator: prefix execution, defaults and termination](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py)
- [Official policy construction](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/serve_policy.py)
- [Actual pi05_libero config: action_horizon=10](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py#L744)

## Install

From this repository directory, on Linux x86-64:

```bash
export FREQUENCY_WORK="${SCRATCH:-$HOME/scratch}/frequency_vla"
bash scripts/setup.sh
source scripts/env.sh
```

On Fir, use the system compiler so generated Linux input headers and the compiler
use the same sysroot (otherwise the upstream `evdev` dependency may not build):

```bash
FREQUENCY_CC=/usr/bin/gcc FREQUENCY_CXX=/usr/bin/g++ bash scripts/setup.sh
source scripts/env.sh
```

Setup pins OpenPI and its LIBERO submodule, installs separate managed Python 3.11.13
(policy) and 3.8.20 (official simulator) environments, uses OpenPI's frozen `uv.lock`
and official LIBERO requirements, and creates a private LIBERO path config to
avoid interactive dataset prompts. It does not download training demonstrations.
`configs/libero.constraints.txt` additionally pins the resolved simulator's
transitive dependency versions without changing the official requirements.
The two `*.freeze.txt` files in `$FREQUENCY_WORK` record the installed environments.
The setup/download phase needs internet and ample disk space (checkpoint alone:
12.44 GB). GPU evaluation needs an EGL-capable NVIDIA driver; the default is EGL.

Configurable paths are `FREQUENCY_WORK`, `OPENPI_DIR`, `SERVER_VENV`, `LIBERO_VENV`,
`CHECKPOINT_DIR`, `OPENPI_DATA_HOME`, `LIBERO_CONFIG_PATH`, and `UV_BIN`. They default
under `$FREQUENCY_WORK`; the repository itself can be placed anywhere. Scripts
remove inherited `PYTHONPATH`, `PYTHONHOME`, and `LD_LIBRARY_PATH` when launching
managed Python, avoiding accidental use of site Python/CUDA libraries.
The policy launcher explicitly uses the CUDA 12.9 `ptxas` shipped by the frozen
dependencies, rather than Fir's inherited CUDA 12.2 compiler. Compiler version
and XLA flags are recorded in the inference fingerprint. `FREQUENCY_CUDA_DIR`
can override that compiler location.

## Check the protocol before running

```bash
# Prints provenance and raises a clear error for the requested full six-H sweep.
"$SERVER_VENV/bin/python" scripts/preflight.py

# Explicitly validate the supported partial experiment.
"$SERVER_VENV/bin/python" scripts/preflight.py --horizons 5 10
```

P is read from the actual pinned training config and checked again after loading
the model. Flow integration remains the upstream **10-step default**, with empty
`sample_kwargs`. P, H, and flow steps are independent concepts; coincidental equal
values of 10 do not make them interchangeable.

## Obtain the unchanged checkpoint

Setup downloads it by default. To separate installation and download:

```bash
SKIP_CHECKPOINT_DOWNLOAD=1 FREQUENCY_CC=/usr/bin/gcc FREQUENCY_CXX=/usr/bin/g++ bash scripts/setup.sh
source scripts/env.sh
"$SERVER_VENV/bin/python" scripts/download_checkpoint.py --destination "$CHECKPOINT_DIR"
```

The source is exactly `gs://openpi-assets/checkpoints/pi05_libero`. Each public GCS
object is downloaded by immutable generation and verified against size and MD5,
or CRC32C for composed objects. `download_manifest.json` records the object
identities. The policy uses the official `pi05_libero` config and checkpoint
normalization assets. It is never fine-tuned or converted.

## Start the policy server

Run inside an allocated GPU compute node. On Fir, for example:

```bash
salloc --account=rrg-btaati --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=40G --time=00:35:00
# Return to this repository directory in the allocated shell, then:
source scripts/env.sh
bash scripts/serve_policy.sh --host 127.0.0.1 --port 8000
```

The server wrapper calls the unchanged `scripts/serve_policy.py:create_policy`
and uses OpenPI's WebSocket server. Its only inference adaptation is a documented,
deterministic PRNG key per `(seed,suite,task,episode,call-index)`, independent of H
and previous episode lengths. This provides paired sampling streams without
changing the native Gaussian sampler or its integration steps. It differs from
the stock server's single continuously advancing key initialized at zero.
The same protocol is used for every measured H, including the reproduction gate.

The current launcher compiles one synthetic observation before opening the server
and disables the server's WebSocket heartbeat, so a long compilation cannot expire
an evaluator connection. The synthetic call is not an episode, and every measured
call still resets its PRNG key. Both settings are recorded in provenance. Batch
startup defaults to a 300-second limit (`SERVER_READY_TIMEOUT_SECONDS`). With
`PILOT_FIRST=1`, one diagnostic episode must complete within 180 seconds
(`PILOT_TIMEOUT_SECONDS`) before the sweep begins; it is excluded from smoke/main
statistics. Any worker failure stops all remaining workers and releases the server.

Seeds do not guarantee bitwise reproducibility across independent GPU server
starts. In this Fir run, three identical fixed-input/seed requests matched exactly
within each server, but the two servers differed by up to 0.002172 in an action
component; some repeated rollouts also differed. The precise numerical
cause was not isolated. See
[`results/diagnostics/inference_reproducibility.json`](results/diagnostics/inference_reproducibility.json).
All H values within each smoke/main comparison use one continuously running
server. Smoke and main records are never pooled as independent episodes.

## Smoke test and one-H evaluation

In another shell on the compute node. For a client on a different node, start
the server with `--host 0.0.0.0` and pass that node's hostname to the client with
`--host`; the batch script uses loopback throughout.

```bash
source scripts/env.sh

# Requested smoke test: 10 episodes/task, seed 7, six horizons.
# Correctly fails because H>10 cannot be executed by the unchanged policy.
bash scripts/run_smoke_test.sh

# Explicit supported partial smoke test: 100 episodes/H, H=5 then H=10.
bash scripts/run_smoke_test.sh --horizons 5 10 --results-dir results/rerun

# One H, 50 episodes/task (use a separate results root to avoid duplicates).
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.evaluator --mode main --suite libero_10 --horizon 5 \
  --seed 7 --results-dir results/one_H
```

A one-task/one-episode environment diagnostic is available via
`--mode diagnostic --episodes 1 --task-ids 0`. It is not the smoke test or a
benchmark result. The evaluator refuses overwriting existing runs/videos; use
an explicit new `--results-dir` for retries. Do not combine overlapping all-task
and task-subset runs: aggregation rejects duplicate episodes.

This repository includes the completed benchmark records under `results/`.
Fresh-rollout examples use `results/rerun`; choose another new directory for
subsequent attempts. Regenerating statistics from archived records uses the
original `results/` directory and does not run the model again.

## Main sweep and additional suites/seeds

```bash
# Full requested sweep: deliberately fails preflight for native P=10.
bash scripts/run_frequency_sweep.sh

# Valid partial main validation: 50 episodes/task, seed 7, H=5 then H=10.
bash scripts/run_frequency_sweep.sh --horizons 5 10 --results-dir results/rerun

# Optional: four independent task processes share one policy server.
bash scripts/run_frequency_sweep.sh --horizons 5 10 --workers 4 \
  --results-dir results/parallel_run

# Optional repeated seeds, isolated from the first completed single-seed run.
bash scripts/run_frequency_sweep.sh --horizons 5 10 --seeds 7 17 27 \
  --results-dir results/three_seeds

# Later suites require flags only; use separate roots if repeating any run.
bash scripts/run_frequency_sweep.sh --suite libero_spatial --horizons 5 10
bash scripts/run_frequency_sweep.sh --suite libero_object --horizons 5 10
bash scripts/run_frequency_sweep.sh --suite libero_goal --horizons 5 10
```

For LIBERO-10, the runner first completes H=5 and checks its rate against 92.4%.
A deficit greater than **5 percentage points** stops subsequent horizons for
debugging. This predeclared gate is a practical diagnostic, not an equivalence
claim. Smoke-test sampling uncertainty is larger than main-test uncertainty.
The other suites remain runnable, but this LIBERO-10-specific gate is not applied
to them.

`--workers` changes task scheduling only. Each task still executes the official
loop, ordered initial states, and identical per-episode sampling seeds. The
upstream WebSocket server processes inference serially. Each task writes a
separate raw shard and log; aggregation verifies complete, non-overlapping
coverage before the H=5 gate or a comparison. Wall-clock durations with concurrent
clients include server queueing, so they are not standalone latency benchmarks.

## Fir batch jobs

The inspected GPU types, user account associations, interactive allocation command
and fairshare snapshot are in [FIR_NOTES.md](FIR_NOTES.md) (Chinese).

Run `sbatch` from the repository directory, or export `FREQUENCY_PROJECT` to its
absolute path. Replace the account with a GPU account you are authorized to use:

Use one GPU and one continuous policy server, and run H values sequentially.
The earlier P=50 job peaked at 26.34 GB host memory, so the examples now request
40 GB instead of 96 GB. A subsequent 4-CPU attempt produced no completed episode
in 15m15s and was cancelled; CPU versus node effects were not isolated. Retain
the previously successful 8 CPUs for four simulator workers rather than assuming
that average CPU utilization predicts peak rendering/initialization needs.
The resource objective is useful evaluation per GPU minute, not simply fewer CPUs.
Wall-time limits are upper bounds; the job releases its allocation when it exits.
The commands below are alternatives: submit only the required mode.

```bash
# Preserve the included measurements; choose a new directory for fresh rollouts.
export RUN_RESULTS="$PWD/results/rerun"
mkdir -p "$RUN_RESULTS/logs"

# One-episode real GPU/environment check.
sbatch --account=rrg-btaati --job-name=freq-check --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=40G --time=00:15:00 \
  --output="$RUN_RESULTS/logs/slurm-%j.log" scripts/fir_job.sh diagnostic

# The scripts start/stop one policy server and collect both horizons sequentially.
sbatch --account=rrg-btaati --job-name=freq-smoke --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=40G --time=00:35:00 \
  --output="$RUN_RESULTS/logs/slurm-%j.log" scripts/fir_job.sh smoke --horizons 5 10 --workers 4

sbatch --account=rrg-btaati --job-name=freq-main --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=40G --time=02:00:00 \
  --output="$RUN_RESULTS/logs/slurm-%j.log" scripts/fir_job.sh main --horizons 5 10 --workers 4
```

Create the output log directory before submission, as above. `POLICY_PORT`
can override the automatically chosen local port. No GPU computation runs on
login nodes. The exact commands and job IDs for the included runs are recorded
in [results/README.md](results/README.md).

## Aggregate results and reproduce plots

```bash
source scripts/env.sh
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/aggregate_results.py --mode smoke
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/aggregate_results.py --mode main --findings-out FINDINGS.md
```

The same command regenerates the CSVs, findings, and all three PNGs from saved
JSONL. No GPU is needed. For an incomplete run, explicitly add `--allow-partial`;
the report then labels its evidence partial. No rows or plots are fabricated for
unsupported or unmeasured horizons.

To aggregate a fresh rollout instead, add `--results-dir results/rerun` and choose
its own findings destination, for example `--findings-out results/rerun/FINDINGS.md`.

Outputs, separately for each mode and suite:

```text
results/raw/<mode>/<suite>/seed_7/H_5/all_tasks.jsonl
results/raw/<mode>/<suite>/seed_7/H_5/all_tasks.manifest.json
results/videos/<mode>/<suite>/seed_7/H_5/task_00_episode_000_success.mp4
results/aggregated/<mode>/<suite>/episodes.csv
results/aggregated/<mode>/<suite>/frequency_sweep.csv
results/aggregated/<mode>/<suite>/per_task.csv
results/aggregated/<mode>/<suite>/per_seed.csv
results/aggregated/<mode>/<suite>/replanning_gaps.csv
results/aggregated/<mode>/<suite>/task_gaps.csv
results/aggregated/<mode>/<suite>/validation.json
results/aggregated/<mode>/<suite>/FINDINGS.md
results/figures/<mode>/<suite>/success_vs_replan_horizon.png
results/figures/<mode>/<suite>/relative_performance_drop.png
results/figures/<mode>/<suite>/success_vs_policy_calls.png
```

Episode records include task identity, initial-state index/hash, first policy
observation hash, seeds, H, success, total/controlled/settling steps, exact policy
call positions/counts, actions executed per call, duration and video path.
Controlled steps exclude the 10 settling actions and include the terminal action;
thus a successful episode is not off by one. Policy/VLM calls count completed
native `infer` invocations, not internal denoising iterations. The wrapper asserts
`calls = ceil(controlled_steps/H)` and exact call positions `0,H,2H,...`.

Overall and per-task rates use 95% Wilson intervals. Gaps additionally use paired
bootstrap intervals, resampling initial-state blocks within fixed tasks; repeated
seeds of one state remain together. A single-seed run also gets exact McNemar
tests and Holm-adjusted p-values across measured candidate horizons. Both calls
per episode and calls per controlled step are reported, since a failing rollout
can be longer than a successful rollout. A 5 pp gap is the predeclared descriptive
threshold for substantial degradation. Any pair recommendation remains
exploratory, and cannot support unmeasured H=20/30/50.

## Implementation and tests

The upstream checkout stays clean. `evaluator.py` imports the official
`eval_libero` function and temporarily wraps its benchmark lookup, environment,
WebSocket client and video writer. It preserves the actual action queue,
preprocessing, limits, ordered `.pruned_init` states, and success condition.
Unlike stock logging, exceptions abort the run and are stored as errors, not
task failures. Videos are uniquely named instead of overwriting earlier episodes.

```bash
source scripts/env.sh
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" -m pytest -q
```

Tests exercise the actual pinned upstream loop using synthetic environment and
policy fixtures confined to pytest temporary directories: prefix discard,
terminal-step counting, call-frequency changes, unique videos, paired-state and
model fingerprint checks, statistical calculations and loud error handling.
They do not substitute for the real checkpoint/environment runs.
