# Round 2: steps, deployment horizon and LIBERO-suite transfer

Authorized on 2026-09-16. The prior 100-update pilot measured H=20 success
46% → 58% on 100 paired episodes (CI +2 to +22 pp, p=0.0428). This round tests
whether that gain holds on different initial layouts, whether more updates help,
and whether the H=20-trained parameters transfer to H=15/25 and other LIBERO suites.

The algorithm stays temporal OPSD with Gaussian velocity matching, the local
OPSD adaptation described in AUTORESEARCH.md. P=50 remains an explicit extension
of the checkpoint's official P=10 config. No change to loss, learning rate,
architecture, attention, flow steps, teacher EMA or the frozen backbone is planned.
This is not a reproduction of the image-generation Flow-OPD paper.

## Fixed design

`configs/autoresearch_round2.yaml` specifies the full evaluation matrix before
the new outcomes are measured. Training resumes the saved FP32 master weights,
EMA and Adam moments at step 100; it adds 200 updates to step 300 and 200 to
step 500. Adam counters must match the checkpoint step. Training is always H=20;
H=15 and H=25 measure deployment transfer, not separately optimized students.
Simulator episodes restart at the two continuation boundaries; model/optimizer
state does not restart. This is not a bitwise continuation of the old simulator
trajectories. Snapshot 500 is the preselected final model, not the best validation
checkpoint. No automatic hyperparameter sweep follows these results.

Initial-state identifiers refer to each task's ordered official start layouts,
not to optimization steps or to individual intermediate robot observations.
The earlier 100-update run actually trained on 18 task/layout combinations at
indices 10–11. This round restricts LIBERO-10 training to indices 10–19, starting
at 12 after each continuation boundary. Hash checks protect indices 0–9 and
20–49. The original checkpoint's own fine-tuning data is a separate matter;
"held out" here means held out from our added OPSD optimization.

| Purpose | Suite | Official state indices | Seed | Episodes/condition |
|---|---|---|---:|---:|
| Horizon/step screen | LIBERO-10 | 20–24 | 17 | 50 |
| Suite transfer screen | spatial, object, goal | 20–22 | 17 | 30 each |
| Primary confirmation | LIBERO-10 | 30–39 | 27 | 100 |

The screen compares original H=5/15/20/25; step-100 and step-500 H=15/20/25;
and step-300 H=20. Each transfer suite compares original H=5/H=20 and step-100/
step-500 H=20. Confirmation compares original H=5/H=20 and step-100/step-500
H=20, after all training finishes. The full bounded design contains 1,310
evaluation episodes: 550 horizon/step, 360 transfer, and 400 confirmation.
Every condition uses the pinned official execution loop and the same ordered
initial states and per-episode/call noise protocol within its comparison.

Other LIBERO suites test preservation/transfer after LIBERO-10-only OPSD. They
are not claimed to be unseen by the official base checkpoint. Screening 30
episodes per suite/condition is too small to establish small gains; all such
results remain exploratory.

## Statistics and stopping

Primary: step 500 minus original checkpoint, H=20, on the 100 confirmation
episodes. Report the paired within-task bootstrap interval and exact McNemar
test regardless of direction. Secondary: step 500 minus step 100 on those same
episodes. Correct the family of secondary/screen checkpoint comparisons with
Holm; do not promote an uncorrected screen result into a final claim. Report
per-task regressions as well as recovered episodes, success, calls and lengths.

The shared policy server keeps frozen original/100/300/500 parameter snapshots
and the native inference sampler. A real diagnostic precedes restoration; source
checkpoint hashes, frozen-backbone equality, EMA/optimizer restoration and saved
checkpoint inference round-trips are checked. Raw actions/images, state identities,
episode records, videos, plots and Slurm usage are retained separately from round 1.

One H100, eight CPU cores, at most two hours for the bounded first job; startup,
diagnostic, checkpoint, training and evaluation stages have their own timeouts.
Request 64 GiB host RAM because the earlier export already peaked at 42.68 GiB
and this round restores optimizer/EMA state and retains extra snapshots. Do not
reserve additional GPUs. On a failure, release the allocation, diagnose on CPU
where possible, preserve completed artifacts, and include failed GPU time in the
final accounting. Do not silently repeat a whole completed study after a late error.

## Reproduction

The completed job's run archive will contain exact commands, revisions, checkpoint
lineage and measured findings. Select fresh result/checkpoint paths for reruns;
the original checkpoint and previous result directories remain intact.
