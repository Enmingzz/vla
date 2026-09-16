# Fixed P=50, H=5/15/20 follow-up

All 300 paired LIBERO-10 smoke episodes completed: seed 7, the first 10 official
initial states per task, one continuous server, unchanged official weights,
P=50 and 10 flow steps. OpenPI's official fine-tuning config remains P=10; this
is an explicitly requested inference-length extension.

| H | Success | Mean policy calls | Mean controlled steps |
|---:|---:|---:|---:|
| 5 | 88/100 | 59.94 | 298.00 |
| 15 | 82/100 | 24.23 | 356.95 |
| 20 | 46/100 | 22.64 | 448.18 |

The H=20 deficit is 42 pp, paired bootstrap 95% CI [33, 51] pp, Holm-adjusted
exact McNemar p=9.095e-13. It saves 62.2% of calls per episode and 74.9% per
controlled step. H=15 saves 59.6% of calls per episode; its 6 pp gap has CI
[-2, 15] pp and adjusted p=.2863, so the deficit is not established.

H=20 causes the largest measured drops on tasks 4 and 6 (both 100% to 30%),
followed by tasks 0 and 9 (both 90% to 30%). These task comparisons are exploratory.
The H=5 contextual baseline screen passes (88% versus a predeclared 87.4% floor).
The 42 pp H=20 gap is larger than the desired moderate gap; recoverability is
not established by evaluation. The user subsequently requested a bounded
100-step H=20 OPSD adaptation, which is tracked separately in AUTORESEARCH.md.

## Runtime and checks

Job 60042686 completed with exit 0 in 25m21s on fc10606: one H100 80 GB, 8 CPUs,
40 GiB host memory; measured host peak 27.45 GiB. The evaluation implementation
at submission was b491e55. No evaluator code changed while this job was running.
One synthetic warmup compiled before the server accepted clients (40.66s).
One 16-second successful diagnostic rollout was excluded from benchmark statistics.
All 300 unique benchmark videos plus the diagnostic video exist locally.

Initial-state, first-observation and RNG hashes, P=50 returned chunk lengths,
inference fingerprints, and actual call positions passed validation. The server
checkpoint/model/flow/package settings match the earlier P=50 experiment; startup
warmup and WebSocket heartbeat handling intentionally changed and are recorded
in provenance/configuration_comparison.json. Independent server runs are not
pooled. No training was used in this archive.

Two earlier attempts, 60040618 and 60041592, were cancelled with no completed
episodes; together they consumed 30m05s of one GPU. Including them, this follow-up
used 55m26s. Their diagnostics are under ../diagnostics/p50_h15_h20_attempt1/ and
attempt2/. The exact root cause of the earlier slowdown was not isolated; the
successful run used startup warmup, no server heartbeat, and failure propagation.

## Artifacts

- [Findings](FINDINGS.md)
- [Episode CSV](aggregated/smoke/libero_10/episodes.csv)
- [Aggregate CSV](aggregated/smoke/libero_10/frequency_sweep.csv)
- [Paired gaps](aggregated/smoke/libero_10/replanning_gaps.csv)
- [Validation](aggregated/smoke/libero_10/validation.json)
- [Success vs H](figures/smoke/libero_10/success_vs_replan_horizon.png)
- [Success vs calls](figures/smoke/libero_10/success_vs_policy_calls.png)
- [Relative drop](figures/smoke/libero_10/relative_performance_drop.png)
- [Scheduler accounting](provenance/slurm_jobs.psv)

Use configs/prediction50_h15_h20.yaml and the commands in the repository README
to reproduce or re-aggregate this run. Runtime logs and videos remain local and
are excluded from Git; raw records, manifests, CSVs and figures are tracked.
