# Retry the step-1000 completion and paired evaluation

The previous launch, job **60427489**, failed on `fc10501` before model
loading: CUDA reported no device and `nvidia-smi` displayed errors. It consumed
**3m25s of one H100 allocation**, with zero optimizer updates or evaluation
episodes. Its logs remain in the
[failed-attempt archive](../autoresearch_round3_finish/README.md).
The underlying device/driver/allocation cause has not been established.

Retry job **60469132** excludes that node and runs a lightweight CUDA driver
probe before importing the model. The probe records the allocated device's
name, visible memory and Slurm/CUDA visibility variables. It requires one full
H100 and exits promptly if initialization fails. No extra GPU diagnostic job
is requested. CPU report job **60469138** follows successful completion.

The research protocol is unchanged: resume the verified **step 978** FP32,
EMA and Adam state; train exactly **22 remaining updates**; save step 1000;
then evaluate **step 500 and step 1000 at P=50, H=20**, ten flow steps,
LIBERO-10 seed 27, official initial-state indices 30–39. There are 100 paired
episodes per checkpoint, **200 total**. Both evaluations use the same
continuous policy server and pinned OSMesa renderer.

All training, evaluation, configuration and original runner source hashes
still match the passing CPU preflight job **60427465**. Its evidence and
initial-state/checkpoint-chain audit are copied here and explicitly marked
as reused in `provenance/reused_preflight.json`. The new startup guard passed
six tests and shell syntax checking. It does not modify the model, optimizer,
loss, training rollouts or evaluation protocol.

The retry again requests **one H100, 12 CPUs, 64 GiB, at most 75 minutes**,
and releases resources immediately after completion or error. Formal success
rates are available only after evaluation; none are inferred from training
loss or the previous run. Check `provenance/final_checks.json` for a validated
completion and `FINDINGS.md` for the measured comparison once generated.

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
