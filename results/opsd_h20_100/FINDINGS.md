# First 100-step temporal OPSD result

At fixed P=50 and H=20, held-out LIBERO-10 success changed from
**46/100 (46%)** to
**58/100 (58%)** after 100 optimizer updates.
The paired change is **+12.0 percentage points**, 95% within-task paired
bootstrap CI **[+2.0, +22.0] pp**; exact McNemar
p = **0.04277**.

This pilot provides paired evidence of improvement on these held-out initial states.
The measured direction is reported regardless of the training loss.
This single-seed pilot requires independent confirmation before a strong recovery
claim; no training hyperparameters were selected using its success results.

## Recovery and task changes

There were 21 failures converted to successes and 9 successes
converted to failures. Relative to the preceding H=5 reference of
88%, the trained student's remaining gap is
30 pp. The measured improvement
corresponds to 28.6% of the earlier H=5/H=20 gap. This recovery
fraction is contextual: H=5 was measured in the preceding frequency job, not
re-evaluated in this training job. The paired before/after H=20 comparison above
is the primary training result.

Each task has only 10 evaluation episodes; the following changes are exploratory.

| Task | Before | After | Change | Description |
|---:|---:|---:|---:|---|
| 0 | 30% | 60% | +30 pp | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 50% | 90% | +40 pp | put both the cream cheese box and the butter in the basket |
| 2 | 40% | 60% | +20 pp | turn on the stove and put the moka pot on it |
| 3 | 70% | 80% | +10 pp | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 30% | 50% | +20 pp | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 90% | 80% | -10 pp | pick up the book and place it in the back compartment of the caddy |
| 6 | 30% | 30% | +0 pp | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 90% | 70% | -20 pp | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 0% | 0% | +0 pp | put both moka pots on the stove |
| 9 | 30% | 60% | +30 pp | put the yellow and white mug in the microwave and close it |

## Protocol

Both conditions used the same continuous server and the native OpenPI inference
sampler, seed 7, and official initial-state indices 0–9 for every LIBERO-10 task.
Every episode executed at most 20 actions before replanning, with P=50 and 10 flow
steps. The only intended before/after variable was the student parameters.
Mean policy calls per episode: 22.64 before,
21.32 after. Calls per fixed-length trajectory remain
approximately one per 20 steps; episode lengths can change with success.

Training used 100 current-student batches of four action blocks, seed 17, and an
eligible initial-state pool of indices 10–49. Initial-state hashes were verified disjoint from
evaluation. The run collected 7,961 controlled training steps across
18 distinct initial states; per-task exposure is reported in
aggregated/training_task_coverage.csv. One diagnostic update was saved/reloaded and fully rolled back before
the formal baseline. Only the existing action expert and action/time projections
were optimized; the visual/language backbone was outside the optimizer. The
teacher was an EMA, decay 0.9999. No demonstrations or success rewards were used
in the loss. The step-100 checkpoint was selected in advance.

## What was tested

This is an experimental continuous-action adaptation of the local OPSD pipeline:
student-controlled rollouts, fresh teacher views every five physical steps,
detached teacher velocity targets on aligned student denoising states, and an EMA
teacher. Auxiliary teacher samples complete missing latent tails. The objective
matches local Gaussian velocity probes; it is not the KL of the full chunk
distribution and is not the image-generation Flow-OPD paper's clipped policy
gradient algorithm. See the repository's AUTORESEARCH.md for the specified method.

The official checkpoint was fine-tuned at P=10; both tested conditions extrapolate
to P=50. This is an initial 100-episode held-out pilot, not a 500-episode main
validation or proof of recovery on unseen tasks. Per-task changes are exploratory.
These evaluation states were excluded from optimizer training but were already
used in the preceding H sweep; they are not a fresh final test set. Expanding to
all official states later would include the training states and must not be
labelled a fully held-out evaluation without a new split.
Further tests must not reuse these results for unreported hyperparameter selection.

Raw before/after records and videos are in baseline/ and student_100/. Training
logs are training.jsonl and rollouts.jsonl; the checkpoint location and digest are
in provenance/step_100.json. Aggregated CSVs and figures are in aggregated/ and figures/.
