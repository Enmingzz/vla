# Simpler full-layout frequency evaluation

Submitted as job `60948028`, after the complete 24-rollout smoke and its
state/image/action/video checks passed. One H100, four CPUs, 48 GiB,
`def-btaati`, initially a one-hour walltime cap, updated to **75 minutes**.
This is a submission record; it does not
claim a completed main evaluation.

Status checked on September 22 at 12:08 PDT: pending, **0/288 episodes**, no GPU
allocated. Before the walltime update, Slurm reported `ReqNodeNotAvail` and
listed `fc10615`, `fc10713`, and `fc10715`; no start estimate was available.
The job does not request those specific nodes. The existing job ID and submit
time were retained. See `provenance/scheduling_check.txt` and
`provenance/walltime_update.json` for the snapshot and time-limit change.

The smoke's measured steady inference plus per-episode overhead scales to
approximately one hour for 288 rollouts including startup. The extra 15-minute
margin avoids a tight timeout; the script releases the single GPU when finished.
No inference or evaluation source/configuration was changed.

Protocol: fixed third-party Bridge-adapted π0.5, native P=5, H=1/2/5, ten
ordinary OpenPI flow steps, 5 Hz control, seed 7. All four predefined WidowX
tasks use all object-layout IDs 0–23: **96 episodes per H, 288 total**. Reset
states, input images, first chunks and noise at shared query timesteps must
match across H. Success is measured at the official 60/120-action time limit.

The frozen source/config hashes are in `submission_sources.json`. The
protocol and model have not changed since the completed smoke. No training
is included. The smoke's first two layouts are rerun here; do not combine
those pilot records with the main results as additional independent episodes.

The smoke scored 1/8, 1/8 and 0/8 at H=1, H=2 and H=5. That low-success pilot
does not establish the hypothesis or a suitable teacher. This broader fixed
evaluation measures the same policy without selecting tasks/layouts by outcome.
The checkpoint's SFT/RL provenance remains unresolved, as documented in
[the experiment protocol](../../docs/SIMPLER.md).

Progress is recorded in `logs/` and `main/raw/episodes.jsonl`. On successful
completion, the evaluator writes `main/FINDINGS.md`, aggregate CSVs, paired
statistics and three figures automatically. Videos are retained on the
cluster and excluded from Git.
