# RoboCasa365: fixed-ten-task π0.5 OPSD pilot

The user authorized three independent Slurm jobs on 2026-09-21: original H=5,
original H=20, and 500 optimizer updates at student H=20. This fixed-ten-task
pilot supersedes the earlier proposed single-task screening and performance gate.
Training does not wait for either baseline's success rate. Infrastructure and
numerical checks must still pass. No success claim is made before measurement.

## Model and tasks

Use the RoboCasa team's [OpenPI fork](https://github.com/robocasa-benchmark/openpi)
and its official [π0.5 checkpoint](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000).
`pi05_pretrain_human300` is π0.5 (`pi05=True`), with native P=50, ten flow steps,
three camera inputs, 16-dimensional proprioception, and 12 physical action
coordinates padded to the model's 32 dimensions. Preserve native discrete state
input and checkpoint normalization. Both student and EMA teacher start from
this RoboCasa checkpoint. The LIBERO-trained student is not the initialization.

The ten tasks, selected before observing any result, are CloseFridge,
CoffeeSetupMug, OpenCabinet, OpenDrawer, PickPlaceCounterToCabinet,
PickPlaceCounterToStove, PickPlaceSinkToCounter, TurnOffStove,
TurnOnMicrowave, and TurnOnSinkFaucet. They are a fixed subset of Atomic-Seen;
they were not selected by measured H=5 competence.

Use `split=pretrain`, matching the official multi-task benchmark's scene/object
distribution. `target` denotes a separate scene/object split. The full benchmark's
published means do not establish the success of this subset. This pilot cannot
establish an average effect over all 365 tasks or unseen-task generalization.

## Three jobs

| Job | Starting weights | Work | GPU allocation limit |
|---|---|---|---|
| `h5` | Official RoboCasa π0.5 | H=5, 10 tasks × 10 episodes = 100 | 1 H100, 1 hour |
| `h20` | Same official checkpoint | H=20, same 100 episode identities | 1 H100, 1 hour |
| `train500` | Same official checkpoint | 500 updates, then step-500 H=20 and H=5, 100 episodes each | 1 H100, 2 hours |

The limits are scheduler caps, not measured runtime estimates. Jobs exit when
finished. All three use EGL, eight CPUs, 64 GiB host memory and `def-btaati`.
Download/install once in a separate CPU-only setup job. Each GPU job depends
only on successful setup, so the three requested jobs can run concurrently.
No GPU is reserved while downloading. A bounded EGL probe runs before loading
weights; known failed EGL nodes are excluded.

The evaluation seed namespace is 27; training is 17. The actual environment and
policy seed is `namespace * 10000000 + task_id * 10000 + episode_index`.
A fresh official Gym environment is constructed and explicitly reset with the
same seed for each episode. An atomic shared catalog checks exact initial
simulator state, XML, task metadata, language and preprocessed observation hashes
before evaluation actions. It stores the initial state/XML/metadata. Policy
keys depend only on the episode and call index, not concurrent request order.
Training-only numerical probes use episode indices 9000 and 9001; their updates
are rolled back, and formal training restarts its own environment stream.
Distinct seeded resets provide held-out initializations, not new task identities.

Use the official action conversion and `info["success"]`. Limits come directly
from `get_task_horizon`; its 1.5× update is already included. No additional
horizon multiplier, action repetition, or success-based filtering is used.
The official evaluation entry point references a nonexistent `task_soup`
argument; our small task dispatcher uses explicit task names while preserving
its preprocessing, action deque, success definition and environment limits.

## OPSD protocol

This ports the existing temporal Gaussian velocity-matching OPSD implementation;
it is not a reproduction of the image-generation Flow-OPD policy-gradient loss.

- Exactly 500 formal optimizer updates; batch of four on-policy blocks each.
- H_student=20, teacher feedback offsets 0, 5, 10 and 15; each next student query
  starts another fully supervised 20-action block.
- EMA teacher decay 0.9999, updated after each student optimizer update.
- AdamW, learning rate 1e-5, weight decay zero, gradient clipping norm one.
- Full action expert and action/time projections train; visual/language backbone
  frozen. No LoRA, architecture changes, attention changes, or flow-step changes.
- Native P=50, model dimension 32; loss uses the 12 real RoboCasa action axes and
  masks actions after episode termination. Unsampled teacher future tails use
  EMA native-flow auxiliary latents at the same flow time.
- FP32 master parameters/EMA/Adam state; BF16 compute. Save steps 100 and 500 with
  normalization assets, optimizer state, checksums and inference reload checks.
- Before training, compare traced versus native sampling, same-input teacher
  versus student fields, fresh-feedback signal versus numeric noise, nonzero
  gradients, and checkpoint roundtrip. Roll back the diagnostic update.

## Installation and reproduction

From the repository root, set scratch storage as needed:

```bash
export RC_WORK="${SCRATCH}/frequency_vla/robocasa365"
export RC_RUN_ROOT="$PWD/results/robocasa365_fixed10_500"
mkdir -p "$RC_WORK/logs"
export RC_SETUP_JOB=$(sbatch --parsable --account=def-btaati \
  --cpus-per-task=4 --mem=16G --time=01:30:00 --job-name=rc365_setup \
  --output="$RC_WORK/logs/setup_%j.log" scripts/setup_robocasa.sh)
bash scripts/submit_robocasa.sh
```

On a brand-new environment, wait for setup to install the Python environment
before running the submission script, which uses that Python for source capture.
The checkpoint downloader restricts HF downloads to inference parameters,
normalization assets and metadata at a pinned revision. Simulator assets use
the pinned official Box registry; archive SHA256 values are recorded. No offline
demonstration data or original training optimizer state is downloaded.

Source revisions and all experiment parameters are in
[configs/robocasa365_opsd.yaml](configs/robocasa365_opsd.yaml).
The server disables only the data-config fallback that searches local offline
training datasets. The official loader still loads all inference transforms and
normalization statistics from the checkpoint. Setup uses the official OpenPI
lock plus pinned RoboCasa simulation requirements; unrelated `tianshou` baselines
are omitted. Installed versions are saved in the setup provenance.

To run individual components **inside a GPU allocation**, after CPU setup:

```bash
source scripts/robocasa_env.sh
# Start server; add --train for the training job.
"$RC_VENV/bin/python" -m frequency_vla.robocasa_server --port 8000 \
  --results-dir "$RC_RUN_ROOT/manual" --checkpoint-root "$RC_WORK/runs/manual"
# In another shell, source the same environment and run one condition:
"$RC_VENV/bin/python" -m frequency_vla.robocasa_eval --port 8000 --horizon 5 \
  --results-dir "$RC_RUN_ROOT/manual" --catalog "$RC_RUN_ROOT/paired_episodes" \
  --condition original_h5
# With a training-enabled server:
"$RC_VENV/bin/python" -m frequency_vla.robocasa_train --port 8000 \
  --results-dir "$RC_RUN_ROOT/manual" --rollouts-dir "$RC_WORK/runs/manual/rollouts" \
  --catalog "$RC_RUN_ROOT/paired_episodes"
```

Never mix attempts in an existing condition directory. Submission freezes hashes
of the exact implementation/config files; later edits require a fresh submission.
Archived LIBERO modules and their source hashes remain unchanged.

CPU-only checks and aggregation:

```bash
source scripts/robocasa_env.sh
"$RC_VENV/bin/python" -m pytest -q tests/test_robocasa_protocol.py
"$RC_VENV/bin/python" -m frequency_vla.robocasa_analysis --root "$RC_RUN_ROOT"
```

Raw episode JSONL/CSV, per-task summaries, Wilson intervals, policy calls,
actions, wall-clock rollout time, successful-episode time and videos are saved.
The final report compares original H5/H20, step500 H20 recovery and step500 H5
retention, checking paired episode identities. Paired episode bootstrap CIs and
exact McNemar tests are exploratory and conditional on these ten tasks. Missing
conditions are reported as pending; simulator/runtime failures abort instead of
being silently scored as task failures. Weak, null and adverse outcomes stay in
the report. There is no checkpoint or task selection using these evaluation results.

## Interrupted-run recovery

The first measured attempt completed all 500 optimizer updates and saved step
500. Original H=20 completed 100 episodes (58 successes). Original H=5 stopped
at 60 completed episodes and trained H=20 at 33 because of initial-reset checks.
The paired partial trained comparison is negative: 6/33 versus original H=20
18/33 on exactly those episodes. This is an incomplete, task-order-dependent
subset, not the ten-task mean. Preserve the results regardless of outcome.

GPU diagnostic 60911009 took 2m40s. All eight checked identities reproduced exact
physical state, episode metadata and preprocessed policy input with native seeded
construction. Seven had different XML text, exclusively an inferred
`content_type="model/obj"` attribute on `.obj` meshes. The comparison now permits
only this serialization equivalence, retaining both the raw XML hash and a
comparison hash. All numerical attributes, assets, physical states, metadata and
observations remain strict. Replaying exported XML was tested and rejected: its
rounded geometry changed the input observations.

Recovery uses the existing server and `load_snapshot` API, with no optimizer
updates. It verifies cached rows, checkpoint/inference identity, episode catalog,
policy-call counts and saved videos. A `student` / `step_500` phase-name alias is
the only allowed inference-spec difference. The 193 completed episodes are
reused; 207 remain across the four conditions. A fresh output archive records the
source of every reused row. An unpaired native reset may be retried at most three
times with the exact same seed, before policy queries or actions; rejected states
are saved, never scored. No seed or result-based filtering is permitted.

To submit this evaluation-only recovery after setting the four paths/digest:

```bash
source scripts/robocasa_env.sh
export RC_RECOVERY_SOURCE="$PWD/results/robocasa365_fixed10_500_retry1"
export RC_RECOVERY_ROOT="$PWD/results/robocasa365_fixed10_500_recovery1"
export RC_RECOVERY_CHECKPOINT="$RC_WORK/runs/robocasa365_fixed10_500_retry1/train500/step_500"
export RC_RECOVERY_MANIFEST="a8f369fa2278e0d6d46f68c8d757fc7839959544a697e7f205287924de5d6a1d"
bash scripts/submit_robocasa_recovery.sh
```

One H100 serves all remaining conditions sequentially; the default scheduler cap
is 75 minutes, not a promised runtime. No task is selected by observed success.
Aggregated wall-clock episode times span the original and recovery allocations;
all use EGL and four evaluation workers, but these times are exploratory.

The first recovery completed original H5 (54/100), original H20 (58/100) and
step-500 H20 (35/100). It stopped during step-500 H5 at 2/43. The complete H20
training comparison therefore shows a 23-percentage-point deterioration; the
original comparison does not establish an H5 advantage on these ten tasks.

The second reset defect was isolated from saved failed metadata: upstream
`Counter.get_reset_regions` uses `list(set(valid_geoms))` on XML element objects,
which can swap left/right region labels across otherwise identically seeded
environments. `robocasa_reset.py` restores only an unambiguous mapping between
recorded labels and exact native region geometry. It does not change geometry,
consume RNG draws, replace simulation state, reload XML or affect training.
The existing exact state, metadata, language and image checks remain mandatory.
Before loading the model, `robocasa_reset_check` checks every outstanding episode
identity on the allocated GPU. Recovery now also supports a separate immutable
`RC_RECOVERY_CATALOG` and provenance-checked reuse across successive attempts.

The follow-up uses `robocasa365_fixed10_500_recovery1` as its source,
`robocasa365_fixed10_500_retry1/paired_episodes` as the catalog, and a fresh
`robocasa365_fixed10_500_recovery2` output. It reuses 343 verified episodes and
needs only 57 more H5 episodes. Fifteen CPU tests and validation of all 343 cached
rows passed. Use `RC_RECOVERY_TIME=00:45:00` for this smaller remaining job.
