# π0.5 LIBERO replanning-frequency validation

Evaluation only, using the official Physical Intelligence OpenPI checkpoint and its
existing LIBERO action-execution loop. No training, distillation, Flow-OPD, model
changes, attention changes, action repetition, or prediction-horizon overrides.

**Protocol constraint discovered before implementation:** at pinned OpenPI commit
`215abfb217dbac7d5f1273282331b9b1866c0479`, `pi05_libero` explicitly configures
**P=10**, not 50. Therefore only **H=5 and H=10** from the requested sweep are
valid with the unchanged config. H=20/30/40/50 fail preflight. The requested six
horizons remain in `configs/frequency_sweep.yaml`; they are never silently dropped.
The intended large-H teacher/student premise cannot be answered without revising
the protocol. A supported two-horizon run is clearly labelled a partial experiment.

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

Setup pins OpenPI and its LIBERO submodule, installs separate managed Python 3.11
(policy) and 3.8 (official simulator) environments, uses OpenPI's frozen `uv.lock`
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
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=96G --time=03:00:00
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
bash scripts/run_smoke_test.sh --horizons 5 10

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

## Main sweep and additional suites/seeds

```bash
# Full requested sweep: deliberately fails preflight for native P=10.
bash scripts/run_frequency_sweep.sh

# Valid partial main validation: 50 episodes/task, seed 7, H=5 then H=10.
bash scripts/run_frequency_sweep.sh --horizons 5 10

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

## Fir batch jobs

Run `sbatch` from the repository directory, or export `FREQUENCY_PROJECT` to its
absolute path. Replace the account with a GPU account you are authorized to use:

```bash
# One-episode real GPU/environment check.
sbatch --account=rrg-btaati --job-name=freq-check --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=96G --time=01:00:00 \
  --output=results/logs/slurm-%j.log scripts/fir_job.sh diagnostic

# The scripts start/stop one policy server and collect both horizons sequentially.
sbatch --account=rrg-btaati --job-name=freq-smoke --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=96G --time=03:00:00 \
  --output=results/logs/slurm-%j.log scripts/fir_job.sh smoke --horizons 5 10

sbatch --account=rrg-btaati --job-name=freq-main --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=96G --time=12:00:00 \
  --output=results/logs/slurm-%j.log scripts/fir_job.sh main --horizons 5 10
```

Create `results/logs` before submission (`mkdir -p results/logs`). `POLICY_PORT`
and `RUN_RESULTS` can override the batch job's automatically chosen local port
and results directory. No GPU computation runs on login nodes.

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
