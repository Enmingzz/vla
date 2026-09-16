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
initial states at indices 10–11; evaluation used indices 0–9. These results are
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
