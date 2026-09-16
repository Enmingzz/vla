# H=20 recovery: first 100-step experiment

Authorized on 2026-09-15 after the initial evaluation-only phase. The objective is
to measure whether on-policy self-distillation recovers success lost by executing
20 actions between observations. Training is conditional on the paired P=50
H=5/15/20 experiment showing a clear H=20 deficit. A negative or inconclusive
training result is an acceptable research result; do not select a result by
repeatedly evaluating the test initial states.

## References and distinction

The user's existing OPSD project is `../opsd` relative to the parent `vla/`
directory: `/home/enmingzz/project/opsd`, commit
`28d04e440350df2037fdd34ce703f088fb4ffd97`. The relevant implementation is
`visionzip_aokvqa/train.py:opsd_nogt_step`: current-student generation, a teacher
with richer observations, detached teacher targets, a distribution-matching loss,
and an EMA teacher (decay 0.9999 in the referenced config). It operates on discrete
Qwen logits and is not directly executable on π0.5 action flows.

[Flow-OPD](https://arxiv.org/abs/2605.08063) and its
[official code](https://github.com/CostaliyA/Flow-OPD) use SDE sampling and a clipped
policy-gradient loss with a detached dense KL reward. Direct velocity regression
is **not** a reproduction of that algorithm. The initial VLA experiment will be
labelled **temporal OPSD with Gaussian velocity matching**, an experimental
continuous-action adaptation of the local OPSD pipeline. It must not be reported
as a faithful reproduction of the image-generation Flow-OPD paper.

## Proposed first adaptation

- Keep the same π0.5 architecture, P=50, H_student=20, and 10 flow integration
  steps. Start from the unchanged official LIBERO checkpoint. Keep the source
  checkpoint and evaluation archives immutable.
- Roll out the current student. During each executed 20-action block, retain the
  actual observations at offsets 0, 5, 10, and 15. The teacher can see these fresh
  observations; the student predicts the entire block from the offset-0 view.
  Teacher actions never control the student environment.
- Use the student's own denoising latent at a sampled native integration time.
  At offset k, align latent positions k onward with teacher positions 0 onward.
  Complete the missing k tail positions with an EMA-teacher latent sample at the
  same time; never repeat or zero-pad executed actions. Supervise only the
  teacher's first five velocity vectors, aligned with student positions k:k+5.
  Offset 0 uses exactly the student's full latent, so identical models and views
  must give zero offset-0 discrepancy before training.
- Minimize the mean squared velocity discrepancy (the forward KL of local
  Gaussian velocity probes with equal fixed covariance), stopping gradients
  through all rollout latents and teacher targets. This is not a claim to compute
  the KL of the complete action-chunk distributions. Terminal-action masks
  prevent training on states or actions after episode termination.
- Use an EMA teacher, decay 0.9999. For this single-GPU pilot, update the existing
  action expert and action/time projections; freeze the visual and language
  backbone. No LoRA modules, new encoders, attention changes, or KV-cache changes.
- First run: 100 optimizer updates, batch 4 current-student action blocks,
  AdamW with learning rate 1e-5, weight decay 0, gradient clipping 1.0. Preserve
  float32 master parameters/optimizer state and the original bfloat16 inference
  computation. No hyperparameter selection using evaluation success.

The temporal alignment and auxiliary-tail construction are new adaptation
choices, not established results. They will be recorded and tested explicitly.

## Validation and resource bounds

Training uses seed 17 and official initial-state indices 10–49. Evaluation uses
seed 7 and indices 0–9, 10 episodes for each of the 10 LIBERO-10 tasks. Initial
state hashes must be disjoint. The original simulator and its official evaluation
loop remain the source of the evaluation protocol.

Before full training, verify finite, nonzero gradients on real student rollouts,
frozen-backbone integrity, latent/velocity alignment, checkpoint save/reload, and
unchanged native inference behavior. A diagnostic update must be rolled back
before the formal pre-training evaluation. Evaluate untrained and step-100 H=20
policies with the same continuous server, initial states and inference seed
protocol. Preserve raw records, videos, training logs, and a separately saved
step-100 checkpoint. Report paired success change and its confidence interval;
loss reduction alone does not demonstrate recovery.

Use one H100 at a time. Complete local implementation checks before requesting
training resources. Bound startup and diagnostic stages, terminate on any
worker/gradient error, and record failed as well as successful GPU time. Stop the
first experiment after the 100-step evaluation; do not launch an open-ended
hyperparameter sweep or silently expand the budget.

## Numerical diagnostic refinement before any formal training

The first real-model diagnostic (job 60044854, 4m55s, no formal updates or
evaluation) stopped because the offset-0 MSE from the differentiated program
exceeded 1e-5. Its raw MSE was not captured before the exception, so the cause
was not yet established. The refinement compares identical teacher/student
inputs through the **same compiled forward function** (MSE tolerance 1e-8), and
separately records the difference between that function and the primal used by
the gradient program. Fresh-observation supervision must exceed this measured
numeric MSE by a factor of five before proceeding. This diagnoses bfloat16/XLA
differences without conflating them with temporal alignment errors. No success
rate was used to revise this diagnostic.

The refined real-model diagnostic passed in job `60045256`: same-program
offset-0 MSE was exactly 0; forward-versus-backward-primal MSE was
9.49e-6; teacher-target MSE at offsets 5–15 was 0.0540 (ratio 5,695). This
signal includes fresh observations, latent reindexing and auxiliary-tail
completion; it does not isolate the contribution of each choice. Traced and native
sampling gave exactly identical actions. The single diagnostic update had
finite nonzero gradients, passed parameter save/reload, and was rolled back
before the formal baseline. These checks establish the implementation signal
for this pilot, not a task-success benefit. Full values are retained in
`results/opsd_h20_100/provenance/numeric_diagnostic.json`.

## Completed first result

Job `60045256` completed the specified 100 updates and both 100-episode H=20
evaluations. Success increased from 46% to 58%: +12 pp, within-task paired 95%
bootstrap CI [+2, +22] pp, exact McNemar p=0.04277. There were 21 recovered
episodes and 9 regressions. The unchanged H=20 baseline matched the preceding
frequency sweep in all 100 outcomes, episode lengths and policy-call counts.

This is an initial positive signal, not robust confirmation from multiple seeds
or a fresh final test set. The improvement corresponds contextually to 28.6%
of the prior 42 pp H=5/H=20 gap; the prior H=5 reference remains 30 pp higher.
Training and evaluation details, task regressions, raw data and limitations are
reported in `results/opsd_h20_100/FINDINGS.md`. No further training or parameter
sweep was launched after observing these results.

The successful job used one H100 for 22m12s. Including the earlier 4m55s failed
diagnostic, this training trial consumed 27m07s of single-GPU time. It ended and
released its GPU. The preceding H=5/15/20 frequency follow-up, including its own
failed attempts, consumed a separate 55m26s.
