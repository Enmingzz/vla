# Simpler smoke retry with native scene reconfiguration

Job `60947294`: **COMPLETED**, exit 0, **7m44s**, node `fc10506`.
One H100, four CPUs, 48 GiB, `def-btaati`, 20-minute cap. The allocation was
released. Submitted after four CPU protocol tests passed.

The retry recreates the physics scene through the official `reconfigure=True`
reset option for every episode. Its preflight checks both smoke layouts for
each of the four tasks, including state and image equality after taking an
action and resetting again. Only after all eight checks pass does it load
the fixed π0.5 checkpoint and execute the 24-rollout H=1/2/5 smoke sweep.

All eight preflight comparisons passed with exactly zero state/pixel difference.
Strict loading of all checkpoint parameters passed. All 24 paired rollouts
completed with native P=5, 10 flow steps and the same checkpoint/inference
fingerprint. No training was performed.

| H | Replanning Hz | Successes / episodes | Success rate | Mean policy calls | Mean rollout seconds |
|---|---|---|---|---|---|
| 1 | 5.0 | 1/8 | 12.5% | 75.0 | 20.75 |
| 2 | 2.5 | 1/8 | 12.5% | 37.5 | 10.46 |
| 5 | 1.0 | 0/8 | 0.0% | 15.0 | 6.39 |

Each condition contains four tasks and the first two object-layout IDs per task,
seed 7. H=1 succeeded on carrot layout 0; H=2 succeeded on eggplant layout 0.
Spoon and stack had no successes in these two layouts. H=5 had transient carrot
success on layout 1 but failed the predeclared final-success endpoint.
The two successes are too few to rank task sensitivity or select a teacher/student
pair. The H=1 versus H=5 paired exact p-value is 1.0; **the hypothesis is not
established**. The +12.5 pp point-estimate gap is exploratory.

Mean episode length is 75 actions for every H. H=2 and H=5 save 50% and 80% of
policy calls, respectively. Timing excludes reset/model loading/video encoding
and includes a first-call warmup in H=1; it is not a controlled latency benchmark.

`provenance/completed_episode_checks.json` verifies all 24 expected combinations,
identical paired states/images/first chunks, shared-step noise, exact prefix
execution, finite actions and video frame counts. The raw records and all
action/chunk traces are under `smoke/`. Videos are saved on the cluster but
excluded from Git. See [FINDINGS.md](smoke/FINDINGS.md), the
[aggregate CSV](smoke/aggregated/frequency_sweep.csv) and
[success figure](smoke/figures/success_vs_replan_horizon.png).

The initial failed reset preflight is preserved in `../simpler_pi05_p5_smoke1`.
The full-layout follow-up is a separate archive, `../simpler_pi05_p5_main1`,
job `60948028`. It uses the same four tasks and does not pool these pilot records.
