# Successful-episode action counts versus the H=5 reference

This is a descriptive reanalysis of existing episode records, with **no new
GPU allocation**. All conditions use LIBERO-10, P=50, ten flow steps, seed 27,
official initial-state indices 30–39, and 100 evaluated episodes per condition.
Actions mean **actually executed policy actions**, excluding the ten initial
settling steps and the unused suffix of each predicted chunk.

**Protocol limitation:** the H=5 reference and historical H=20 conditions use
the archived EGL run. The current 500/1000-update conditions use OSMesa.
Initial-state hashes and episode RNG seeds match for all 100 layouts, but
**all 100 first-observation hashes differ across the two archives**, as do
the evaluator fingerprints. Matching successful initial states does not
remove this difference. These are historical descriptive comparisons, not
a controlled equivalence test between H=5 and the current H=20 model.

Each row below averages only that condition's successful episodes:

| Policy | H | Renderer | Successes / all episodes | Mean actions on success | Mean policy calls on success |
|---|---:|---|---:|---:|---:|
| Original | 5 | EGL | 93/100 | 279.18 | 56.22 |
| Original | 20 | EGL | 49/100 | 368.33 | 18.92 |
| OPSD 500, historical | 20 | EGL | 72/100 | 311.75 | 16.08 |
| OPSD 500, current | 20 | OSMesa | 75/100 | 309.00 | 15.93 |
| OPSD 1000, current | 20 | OSMesa | 78/100 | 281.81 | 14.53 |

For example, original H=5 executed 25,964 actions across its 93 successes,
giving 25,964/93 = 279.18. Current OPSD 1000 executed 21,981 actions across
its 78 successes, giving 21,981/78 = 281.81. Failure episodes are excluded
from both sums and denominators; no timeout penalty is substituted.

Because those successful sets differ, also restrict each comparison to
initial states where **both the H=5 reference and the candidate succeeded**:

| Candidate at H=20 | Common successes | H=5 mean actions | Candidate mean actions | Candidate extra actions |
|---|---:|---:|---:|---:|
| Original, EGL | 46 | 261.80 | 360.22 | +37.6% |
| OPSD 500, historical EGL | 71 | 273.90 | 312.32 | +14.0% |
| OPSD 500, current OSMesa | 73 | 273.33 | 307.63 | +12.5% |
| OPSD 1000, current OSMesa | 75 | 269.03 | 275.73 | +2.5% |

Each row has its own intersection, hence the H=5 mean also changes between
rows. The first two comparisons have identical initial observations and
evaluator fingerprints. The last two retain the EGL/OSMesa discrepancy.
Conditioning on both policies' success selects a subset of the benchmark
and cannot establish full-benchmark equivalence.

In the saved records, the 1000-update H=20 model uses a similar number of
actions when it succeeds, while it succeeds on 78% of episodes versus the
historical H=5 reference's 93%. Successful-episode efficiency and overall
task reliability answer different questions. A controlled comparison with
the current H=20 model would require measuring H=5 under the same OSMesa
evaluation protocol.

The [success-only table](successful_episodes.csv),
[common-success table](common_success_with_H5.csv),
[episode membership](episode_membership.csv), and
[source checksums and limitations](provenance.json) are saved alongside this
report. Reproduce from the repository root:

```bash
source scripts/env.sh
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  scripts/compare_successful_actions.py
```
