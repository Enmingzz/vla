# SimplerEnv: paired replanning-frequency evaluation

This experiment measures whether less frequent replanning reduces the success
of one fixed π0.5 checkpoint on the four standard WidowX Bridge tasks. It adds
no training, distillation, model, attention, or flow-sampling changes.

## Checkpoint and prediction horizon

The checkpoint is [Fleurrr/OpenPI05-Bridge-RL](https://huggingface.co/Fleurrr/OpenPI05-Bridge-RL),
revision `241ab13fe84ae2d547b64076ad6e172e2a963691`. This is a **third-party
Bridge-adapted π0.5**, not Physical Intelligence's LIBERO or RoboCasa weights.
The downloaded weights' SHA-256 is
`ae0b7fc1c44813d6bb9be76b5ff125ff2b8c94472d1e0977fd903cc34e7fb424`.

The saved `metadata.pt` specifies **P=5**, 32 padded action dimensions,
200 language tokens, `pi05=True`, and `discrete_state_input=False`.
We preserve these settings and use 10 ordinary ODE flow steps in the official
OpenPI PyTorch model. Strict checkpoint loading must pass. There is no
prediction-horizon extrapolation, action repetition, or temporal ensembling.

The [ProphRL release README](https://github.com/LogosRoboticsGroup/ProphRL/blob/6deaf18631fc5db2d4029acec8837d502f343e4a/rl/README.md)
calls this artifact an SFT checkpoint, while its Hugging Face card calls it RL.
Its training stage is therefore **unresolved**. The current ProphRL config also
uses P=6, differing from the saved artifact's P=5. We use the saved P=5 and record
the conflicting sources. We do not label its score as an official π0.5 baseline
or a reproduction of the ProphRL paper. Its fixed weights still permit a paired
within-checkpoint frequency comparison once the policy adapter is validated.

The Bridge input mapping and bundled normalization statistics come from
ProphRL commit `6deaf18631fc5db2d4029acec8837d502f343e4a`.
The release code's data factory selects quantile normalization for π0.5;
the nested base-config metadata's default `use_quantile_norm=False` is overridden
by that factory. Inputs are the single third-person image and instruction,
with masked zero wrist images and no state tokens. We use the official OpenPI
image resize and tokenizer. The seven physical outputs are delta XYZ,
delta Euler rotation, and absolute gripper openness. The client applies the
official Simpler Bridge Euler-to-axis-angle and binary-gripper conversions.

## Fixed evaluation protocol

| Parameter | Value |
|---|---|
| SimplerEnv | `06accaca93535902d408da4855f21cece12bceb7` |
| ManiSkill2 real2sim | `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682` |
| Official OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Tasks | Spoon on towel; carrot on plate; stack blocks; eggplant in basket |
| Scene protocol | Official prepackaged visual matching; fixed robot/cameras |
| Native prediction horizon P | 5 |
| Execution horizons H | 1, 2, 5 |
| Action frequency | 5 Hz |
| Replanning frequencies | 5, 2.5, 1 Hz, respectively |
| Episode limit | 60 actions for spoon/carrot/stack; 120 for eggplant |
| Smoke | 2 object-layout IDs per task × 4 tasks × 3 H = 24 rollouts |
| Main | All 24 object-layout IDs per task × 4 tasks × 3 H = 288 rollouts |
| Seed | 7 |
| Renderer | SAPIEN 2.2.2 Vulkan, official IBL shaders |

These 24 layout IDs per task follow the official
[Bridge evaluation script](https://github.com/simpler-env/SimplerEnv/blob/06accaca93535902d408da4855f21cece12bceb7/scripts/rt1x_bridge.sh).
This does not evaluate additional background/robot-pose variations.
The whole suite is chosen before observing policy scores.

SAPIEN uses Vulkan; `MUJOCO_GL=egl` does not select its renderer. All H conditions
use the same Vulkan renderer and shader settings. LIBERO/RoboCasa installations
and archived measurements remain separate.

Success is the **final** simulator success at the official time limit, matching
Simpler's official inference loops. Transient success is logged as `ever_success`
but is not the primary endpoint. We do not end episodes at the first success.
Therefore mean episode lengths are fixed, and policy-call savings can be
compared directly. Rollout time excludes reset, model startup and video encoding;
it can include first-call warmup and is a secondary measurement.

Reset seeds and ordered object-layout IDs are identical across H. Exact initial
physical-state and input-image hashes must match. Sampling noise is keyed by
seed, task, layout and environment timestep, so shared query times use identical
noise. The first predicted chunk must match exactly across H. Every episode
checks `calls = ceil(actions/H)` and logs actual query timesteps. Actions outside
the first H of each chunk are discarded.

Each episode now calls the simulator's native `reset(...,
options={"reconfigure": True, ...})` to rebuild the scene before initialization.
The first smoke attempt (`60921197`) found that a seed alone with a reused scene
did not reproduce exact physical state or pixels; it stopped after 16 seconds,
before policy loading or any task evaluation. The retry retains exact checks
and probes both smoke layouts of every task before loading the model. No
tolerance relaxation or stored-state injection is used.

## First measured smoke

Job `60947294` completed all 24 rollouts in 7m44s on one H100. H=1, H=2 and H=5
achieved 1/8 (12.5%), 1/8 (12.5%) and 0/8 (0%), respectively. H=1 succeeded on
one carrot layout; H=2 succeeded on one eggplant layout. The H=1 minus H=5 gap
is +12.5 percentage points, but the exact paired p-value is 1.0. This smoke
does not establish a statistically convincing frequency effect or a strong
teacher baseline. In particular, transient carrot success at H=5 did not persist
until the final time limit and is correctly counted as failure.

All eight preflight reset comparisons had zero state and image difference.
All 24 saved videos have the expected frame count, and the recorded raw actions
exactly equal the executed prefixes of the saved full chunks. Mean episode
length is 75 actions at every H; mean policy calls are 75, 37.5 and 15.
The [smoke archive](../results/simpler_pi05_p5_smoke2/README.md) includes raw
records, traces, provenance, CSVs and figures. Videos remain on the cluster.

Main job `60948028` was submitted with the unchanged source/configuration for
all 24 predefined layouts of all four tasks, 96 episodes per H and 288 total.
The first two layouts are evaluated again, so smoke and main must not be pooled.
The one-hour allocation cap is a bound, not a measured runtime or queue estimate.
No OPSD training is scheduled from the smoke outcome.

## Run

All paths can be overridden before sourcing `scripts/simpler_env.sh`.
Defaults use `$SCRATCH/frequency_vla/simpler`. A standalone `uv` executable is
needed (`UV_BIN`, existing project installations already provide it).

```bash
cd /path/to/frequency_vla
export FREQUENCY_PROJECT="$PWD"
source scripts/simpler_env.sh

# CPU only: install isolated client/server dependencies and download the fixed
# checkpoint, tokenizer and statistics. This is safe to submit without a GPU.
sbatch --account=def-btaati --cpus-per-task=4 --mem=12G --time=00:30:00 \
  --output=setup_simpler_%j.log scripts/setup_simpler.sh
# On a non-Slurm workstation, run bash scripts/setup_simpler.sh directly.

# After successful setup:
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$SV_CLIENT_ENV/bin/python" -m pytest -q tests/test_simpler_protocol.py

# One H100; bounded renderer/reset preflight, strict model load, then smoke.
SV_RUN_ROOT=results/simpler_smoke SV_MODE=smoke SV_TIME=00:30:00 \
  bash scripts/submit_simpler.sh

# Run only after inspecting smoke logs/videos and validating policy behavior.
SV_RUN_ROOT=results/simpler_main SV_MODE=main SV_TIME=01:00:00 \
  bash scripts/submit_simpler.sh

# Reproduce all aggregate CSVs, intervals, paired tests and plots:
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$SV_CLIENT_ENV/bin/python" -m frequency_vla.simpler_analysis \
  --root results/simpler_main --mode main
```

The server can also be started manually **inside a GPU allocation**:

```bash
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$SV_SERVER_ENV/bin/python" -m frequency_vla.simpler_server \
  --port 8000 --output /path/to/server_provenance.json
```

The Slurm script freezes source hashes before submission. Use fresh result roots
after fixes; it fails if source/config changes before launch. GPU time is not
used for downloads. The model server and simulator/video encoder are separate
processes. Environment package versions, weights and source provenance are
recorded under `$SV_WORK/provenance` and the run's `provenance` directory.

Raw JSONL/CSV, action/chunk NPZ traces, videos, aggregate CSVs, three figures and
`FINDINGS.md` are written under `<run>/<mode>/`. The figures are
`success_vs_replan_horizon.png`, `relative_performance_drop.png` and
`success_vs_policy_calls.png`. Success intervals use Wilson; gaps use a paired
bootstrap within each fixed task; McNemar tests receive Holm correction across
the two comparisons against H=1. No performance direction is assumed.

**This native-P=5 experiment cannot answer the literal H=5 versus H=20 question.**
That comparison needs a verified Bridge/Google-robot checkpoint natively
predicting at least 20 actions. Existing LIBERO, DROID and RoboCasa checkpoints
have different embodiment/action contracts and are not used as substitutes.
