# H=5 retention: 500 episodes per model (retry 1)

**Completed:** job 60564883 finished normally in **1h11m08s** on one H100;
the allocation is released. All 800 new episodes plus 200 reused episodes passed
the paired-state, first-observation, RNG, inference-setting, evaluator-setting,
coverage, policy-call, and video-presence checks. No training was performed.

| Model | All 500 episodes | 400 episodes held out from added OPSD |
|---|---:|---:|
| Original | 444/500 (88.8%) | 356/400 (89.00%) |
| OPSD step 1500 | 429/500 (85.8%) | 345/400 (86.25%) |

The primary held-out change is **−2.75 pp**, paired 95% CI **[−6.25, +1.00] pp**,
exact McNemar **p=0.18485** (23 recoveries, 34 regressions). The full-500
secondary change is **−3.00 pp**, CI **[−6.40, +0.20] pp**, **p=0.10064**
(29 recoveries, 44 regressions). The point estimates decrease at H=5, but neither
contrast establishes a statistically significant decrease. These data also do
not establish equivalence or absence of degradation.

On the held-out subset, the largest task-level net decrease is task 8, putting
both moka pots on the stove: 16/40 → 12/40. Tasks 0 and 5 each lose three
successes; task 3 gains two and task 2 gains one. These task-level comparisons
are exploratory. The [full per-task CSV](aggregated/per_task.csv) includes all
tasks and both training-layout and held-out partitions.

Mean time per episode over all 500 is 14.38s → 14.45s; among each model's
successful episodes it is 13.62s → 13.31s. These are four-worker shared-server
wall times, and conditioning on each model's successes changes which episodes
enter the latter means. They do not establish an inference speedup.

Including the preceding cancelled 12m53s attempt, this extension used **1h24m01s**
of single-H100 allocation time. The failed initialization produced no episode
outcomes. Complete [findings](FINDINGS.md), [conditions](aggregated/conditions.csv),
[paired comparisons](aggregated/comparisons.csv), and
[1,000 joined episode records](aggregated/episodes.csv) are available below.

Retry after job 60563925 stalled during EGL reset with zero completed episodes. The scientific plan is unchanged. This launch performs a bounded EGL rendering preflight before loading the model and stops if evaluation produces no completed episode for 180 seconds.

Compare Original and the completed all-EGL OPSD step-1500 student at P=50, H=5,
LIBERO-10, seed 27, EGL, four simulator workers, and ten flow steps. This follows
the user's request to increase the 100-episode retention comparison to 500 episodes.
There is no new training and no additional repetition of completed episodes.

Each model uses all ten tasks × fifty official initial states (indices 0–49).
Reuse its completed 100 episodes at states 30–39 from
[`../egl_step1500_h5_retention`](../egl_step1500_h5_retention/README.md), and add
400 episodes at states 0–29 and 40–49. Both conditions therefore have 500 episodes:
1,000 total, of which 800 require new inference and 200 are checksum-pinned reuse.
Original and trained snapshots alternate within each new block on one server.

## Interpretation fixed before the extension

Added OPSD training used initial states 10–19 for all ten tasks. Report all three
subsets; do not pool training-layout and held-out performance into a generalization
claim:

| Subset | Episodes per model | Role |
|---|---:|---|
| Initial states 0–9 and 20–49 | 400 | Primary retention comparison; held out from added OPSD |
| All official initial states 0–49 | 500 | Full protocol, descriptive secondary result |
| Initial states 10–19 | 100 | Training-layout result, descriptive secondary result |

The fixed tasks/layouts have been inspected previously, so this is not a blind or
unseen-task evaluation. The comparison concerns extended P=50, not official native
P=10. State hashes from the older native-P evaluation only validate layout identities;
none of that evaluation's outcomes enter these performance tables.

Analysis requires complete coverage, matching paired state/first-observation/RNG
identities, unchanged native inference, EGL throughout, and exactly one cached
server plus one new server. Raw episode indices remain local to their blocks;
aggregation preserves them as `source_episode_index` and pairs by physical layout
index. Raw evaluator fingerprints retain their block offset. Only that explicitly
declared offset may differ between evaluator specifications. Confidence intervals
resample paired layouts within fixed tasks; the primary exact McNemar contrast is
step-1500 minus Original on the 400 held-out episodes. A nonsignificant difference
does not prove equivalence or non-inferiority.

## Reproduce

Use the existing installed environments and trained checkpoint. Choose a fresh
result directory; paths are provided through environment variables and CLI flags.

```bash
source scripts/env.sh
export RUN_RESULTS="$PWD/results/egl_step1500_h5_500-rerun"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.h5_extension prepare --results-dir "$RUN_RESULTS" \
  --reference-results "$PWD/results/egl_step1500_h5_retention" \
  --state-catalog "$PWD/results/raw/main/libero_10/seed_7/H_5"
# Freeze the additional renderer checks before submission.
python3 - <<'PY_LAUNCH'
from pathlib import Path
import hashlib, json, os
paths = ["scripts/fir_h5_extension_checked.sh", "tests/check_parallel_osmesa.py",
         "configs/opsd_egl_1500.yaml", "src/frequency_vla/opsd_client.py",
         "src/frequency_vla/opsd_parallel_env.py"]
record = dict(renderer_check_timeout_seconds=180, episode_progress_timeout_seconds=180,
              source_files={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in paths})
(Path(os.environ["RUN_RESULTS"]) / "provenance/launch_checks.json").write_text(
    json.dumps(record, indent=2) + "\n")
PY_LAUNCH
sbatch --account=def-btaati --job-name=vla-h5-500 \
  --nodes=1 --ntasks=1 --gpus-per-node=h100:1 --cpus-per-task=8 --mem=48G \
  --exclude=fc10501,fc10511,fc10514 --time=01:30:00 \
  --output="$RUN_RESULTS/logs/gpu-%j.log" scripts/fir_h5_extension_checked.sh
```

The 90-minute request is a cap; the job exits on completion or error. The previous
paired 200-episode evaluation took 996.49 seconds excluding startup. Scaling that
measurement estimates about 66 minutes for 800 new episodes, plus roughly four
minutes startup; actual task durations and queue waits can differ.

The job writes `FINDINGS.md` and `aggregated/` automatically after all eight new
100-episode blocks finish. Raw new episodes/videos live under `new_evaluations/`;
cached raw episodes/videos remain in their original archive. Recreate statistics:

```bash
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.h5_extension analyze --results-dir "$RUN_RESULTS"
```

`provenance/study_complete.json` and `aggregated/validation.json` passed their
checks. `provenance/final_checks.json` records the completed allocation and pins
the result artifacts and raw records by SHA-256.
