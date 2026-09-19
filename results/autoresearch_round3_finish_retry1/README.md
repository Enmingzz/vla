# Completed step-1000 continuation and paired evaluation

The continuation reached **step 1000** and completed all **200 evaluation
episodes**. At fixed P=50, H=20, the remeasured step-500 model achieved
**75/100** successes and step 1000 achieved **78/100**. The paired difference
is **+3 pp**, 95% within-task bootstrap CI **−6 to +12 pp**, exact McNemar
**p=0.7111**. This does not establish a success-rate improvement from the
additional 500 updates. There were 16 recovered episodes and 13 regressions.

Mean executed actions fell from **361.75 to 334.21**, and policy calls from
**18.45 to 17.05** per episode (about **7.6%** fewer). Both models still use
H=20, so call density remains approximately 1/20: fewer calls reflect shorter
episodes. Among the 62 episodes where both succeeded, mean actions fell
from 300.42 to 269.79. These efficiency comparisons are descriptive.
Whole-condition evaluation time was nearly unchanged: 1592.59 vs 1588.89 s.

The largest task changes were task 6, **10% → 70%**, and task 9,
**90% → 40%**. Each task has only ten paired episodes; the opposite changes
show why the overall +3 pp should not be described as a broad improvement.
See the [full findings](FINDINGS.md), [comparison figure](figures/continuation_comparison.png),
[episode records](aggregated/episodes.csv), [task table](aggregated/per_task.csv)
and [paired statistics](aggregated/comparisons.csv).

The previous launch, job **60427489**, failed on `fc10501` before model
loading: CUDA reported no device and `nvidia-smi` displayed errors. It consumed
**3m25s of one H100 allocation**, with zero optimizer updates or evaluation
episodes. Its logs remain in the
[failed-attempt archive](../autoresearch_round3_finish/README.md).
The underlying device/driver/allocation cause has not been established.

Retry job **60469132** excluded that node and completed on **fc10512**, using
one H100 for **1h04m50s**, below its 75-minute cap. Its allocation is released.
A lightweight CUDA driver probe ran before importing the model and recorded the device's
name, visible memory and Slurm/CUDA visibility variables. It requires one full
H100 and exits promptly if initialization fails. No extra GPU diagnostic job
was requested. CPU report job **60469138** failed on an analysis-only plan
serialization issue: JSON converts integer keys into strings, changing the
sorted hash order of 500 and 1000. The reader now restores those schema-defined
integer keys before checking the original plan digest. The restored plan
equals the predeclared YAML exactly; the hash guard still rejects actual
protocol changes. Episode files and checkpoint identities were not changed.
CPU-only report/integrity job **60477541** completed in 23 seconds with the
corrected reader. All **200 videos** have the expected frame counts; every
final checkpoint file checksum passed; saved/reloaded native inference has
maximum absolute difference **0**. The [final checks](provenance/final_checks.json)
record these results and the report/CSV/figure hashes.

The research protocol is unchanged: resume the verified **step 978** FP32,
EMA and Adam state; train exactly **22 remaining updates**; save step 1000;
then evaluate **step 500 and step 1000 at P=50, H=20**, ten flow steps,
LIBERO-10 seed 27, official initial-state indices 30–39. There are 100 paired
episodes per checkpoint, **200 total**. Both evaluations use the same
continuous policy server and pinned OSMesa renderer.

At GPU startup, all training, evaluation, configuration and original runner
source hashes matched the passing CPU preflight job **60427465**. Its evidence and
initial-state/checkpoint-chain audit are copied here and explicitly marked
as reused in `provenance/reused_preflight.json`. The new startup guard passed
six tests and shell syntax checking. It does not modify the model, optimizer,
loss, training rollouts or evaluation protocol.

The retry used **one H100, 12 CPUs and 64 GiB**. The initial 478-update segment,
failed GPU launch and successful finish/evaluation together used
**2h05m13s of one-GPU allocation time**, never concurrently. The last 22
updates took 163.34 s; the prior 478 took 2875.68 s. The remaining allocation
time includes startup, diagnostics, saving and the 200 evaluations.
Check `provenance/final_checks.json` for checkpoint/video integrity checks.
The earlier EGL step-500 result of 72% is historical; both current endpoints
were measured again with OSMesa on the same server, seed and initial states.

For reproduction, use the commands in the
[completion protocol](../autoresearch_round3_finish/README.md#reproduction),
select a fresh `RUN_RESULTS` and `OPSD_CHECKPOINT_ROOT`, and replace the GPU
submission script with `scripts/fir_gpu_checked_study.sh`. Add
`--exclude=fc10501` while the observed device issue remains unresolved.

To regenerate completed statistics and plots:

```bash
source scripts/env.sh
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.study_analysis --results-dir results/autoresearch_round3_finish_retry1
```
