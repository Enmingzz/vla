# First H=20 recovery experiment

The paired P=50 sweep measured H=5: 88%, H=15: 82%, H=20: 46%; the H=20 gap
is 42 pp, paired 95% CI [33, 51] pp. The separately authorized recovery experiment
is specified in [AUTORESEARCH.md](AUTORESEARCH.md). It is temporal OPSD with local
Gaussian velocity matching, adapted from the user's OPSD pipeline, not the
image-generation Flow-OPD paper's policy-gradient algorithm.

The completed first pilot measured **46% → 58%** at H=20 after 100 updates
(100 paired episodes, +12 pp, paired 95% CI +2 to +22 pp, McNemar p=0.0428).
See [FINDINGS](results/opsd_h20_100/FINDINGS.md) and the
[run archive](results/opsd_h20_100/README.md). The actual training covered 18
task/start-layout combinations at zero-based indices 10–11; evaluation used
indices 0–9. These are layout identifiers, not optimizer steps or a count of
intermediate observations: the 100 updates used 400 action blocks and 7,961
executed actions. This describes our added OPSD, not the checkpoint's original
training data. These results are
preliminary and are not a fully held-out evaluation of all 50 official states.

Reuse the setup and base checkpoint from README.md. The original OpenPI checkout,
checkpoint and frequency-result archives remain intact.

## Local checks

```bash
source scripts/env.sh
env -u FREQUENCY_CONFIG -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  "$LIBERO_VENV/bin/python" -m pytest -q
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  "$SERVER_VENV/bin/python" tests/check_opsd_jax.py
```

The CPU fixture uses the native sampler and NNX/Orbax APIs on tiny arrays; it
loads no VLA checkpoint and produces no research results. The GPU job additionally
requires a successful real-environment diagnostic update, parameter save/reload,
and exact rollback before the formal baseline evaluation.

## Run on Fir

Use fresh paths; runs are never overwritten. GAP_RESULTS must contain a complete
paired H=5/15/20 sweep with a statistically convincing H=20 deficit of at least 5 pp.

```bash
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50_h15_h20.yaml"
export GAP_RESULTS="$PWD/results/p50_h15_h20"
export RUN_RESULTS="$PWD/results/opsd_h20_100-rerun"
export OPSD_CHECKPOINT_ROOT="$FREQUENCY_WORK/runs/opsd_h20_100-rerun"
mkdir -p "$RUN_RESULTS/logs"
sbatch --job-name=vla-opsd-h20-100 --account=rrg-btaati --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=48G --time=01:00:00 \
  --output="$RUN_RESULTS/logs/slurm-%j.log" scripts/fir_opsd_job.sh
```

One server and one GPU handle diagnostic/rollback, 100 held-out untrained H=20
episodes, 100 optimizer updates, checkpoint export, and the same 100 held-out
episodes with the step-100 student. Startup, diagnosis, training and evaluations
have separate timeouts; the first failing stage exits and releases the GPU. The
one-hour limit is an upper bound, not a mandatory allocation duration.

Training uses four student-controlled environments, seed 17 and official initial
state indices 10–49. Evaluation uses seed 7 and indices 0–9. Hash checks reject
overlap. Terminal masks exclude actions after episode termination. No evaluation
success values are fed to the optimizer.

Only the existing action expert and action/time projections are optimized. The
visual/language backbone is frozen; no adapters or architecture changes are added.
The config fixes AdamW at 1e-5, gradient clipping 1.0, no weight decay, EMA teacher
decay 0.9999, P=50, H_student=20, H_teacher=5 and 10 flow steps.

## Analyze without a GPU

```bash
source scripts/env.sh
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.opsd_analysis --results-dir "$RUN_RESULTS"
```

Outputs include FINDINGS.md, episode/aggregate/per-task CSVs, paired bootstrap
uncertainty, exact McNemar statistics, training_loss.png and
success_before_after_100.png. The checkpoint location/digest is recorded in
provenance/step_100.json. Replayable training observations and actions are saved
under OPSD_CHECKPOINT_ROOT/rollouts, outside the source repository.

## Load the exported student later

Inside a GPU allocation:

```bash
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50_h15_h20.yaml"
bash scripts/serve_policy.sh --port 8000 \
  --trained-checkpoint "$OPSD_CHECKPOINT_ROOT/step_100" \
  --manifest-out "$RUN_RESULTS/provenance/reloaded_student.json"
```

The loader validates the training manifest, source-checkpoint identity, inference
settings, and exported parameter/asset checksums. Use the standard evaluator at
H=20 with a fresh result path. Reloaded runs remain separate from the same-server
before/after comparison because independent GPU compilations were not bitwise
reproducible in the earlier diagnostics.

## Round 2: continue to 500 updates and test transfer

The fixed protocol is in [AUTORESEARCH_ROUND2.md](AUTORESEARCH_ROUND2.md).
It restores FP32 parameters, EMA and Adam state from the completed step-100
checkpoint, saves milestones 300 and 500, evaluates deployment H=15/20/25,
and screens spatial/object/goal. Training stays at H=20. The 1,310-episode
matrix includes a separate 100-pair confirmation split; step 500 is preselected.

```bash
source scripts/env.sh
export FREQUENCY_CONFIG="$PWD/configs/prediction50_round2.yaml"
export RUN_RESULTS="$PWD/results/autoresearch_round2-rerun"
export OPSD_CHECKPOINT_ROOT="$FREQUENCY_WORK/runs/autoresearch_round2-rerun"
export PARENT_CHECKPOINT="$FREQUENCY_WORK/runs/opsd_h20_100_attempt2/step_100"
export PARENT_RESULTS="$PWD/results/opsd_h20_100"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
mkdir -p "$RUN_RESULTS/logs"

# CPU only: check actual state hashes and the parent checkpoint identity first.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.study_plan --plan configs/autoresearch_round2.yaml \
  --training-config configs/opsd_continuation_500.yaml \
  --parent-results "$PARENT_RESULTS" --parent-checkpoint "$PARENT_CHECKPOINT" \
  --output "$RUN_RESULTS/provenance/split_audit.json"

# One GPU throughout; the two-hour request is a limit, with immediate exit on completion.
sbatch --job-name=vla-opsd-round2 --account=rrg-btaati --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --output="$RUN_RESULTS/logs/slurm-%j.log" scripts/fir_study_job.sh

# After completion, validate all records and recreate tables, figures and FINDINGS.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.study_analysis --results-dir "$RUN_RESULTS"
```

The parent checkpoint's manifest digest is pinned in the study plan. Paths may
change without changing its identity. A newly trained parent is a different
experiment: declare that identity in a separate plan before evaluating it.
The current batch script uses the round-two configs named above.

To serve a saved milestone independently, use the preceding student-loading
command with `--trained-checkpoint "$OPSD_CHECKPOINT_ROOT/step_500"` and this
round's `FREQUENCY_CONFIG`. To evaluate one condition at a different horizon,
use the official wrapper with explicit layout offset:

```bash
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.opsd_evaluate --port 8000 --suite libero_10 --horizon 25 \
  --episodes 5 --seed 17 --initial-state-start 20 --workers 4 \
  --results-dir "$PWD/results/step500-H25-rerun"
```

This standalone check is kept separate from the matrix, whose snapshots all
share one continuous server to control the observed cross-compilation variation.
