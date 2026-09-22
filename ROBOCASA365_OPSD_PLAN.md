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
