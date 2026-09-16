# First 100-update H=20 recovery pilot

Completed on Fir, job `60045256`, one H100. Fixed P=50, H=20, 10 native flow
steps, original π0.5 LIBERO initialization; action expert and action/time
projections optimized, visual/language backbone frozen.

**Result:** 46/100 → 58/100 paired successes, +12 pp (95% paired CI +2 to +22 pp;
exact McNemar p=0.04277). The prior H=5 reference was 88%; the remaining gap is
30 pp. This is a preliminary result from temporal OPSD with Gaussian velocity
matching, not an implementation of the image-generation Flow-OPD paper.

- [FINDINGS.md](FINDINGS.md): interpretation, task changes and limitations.
- [aggregated/comparison.json](aggregated/comparison.json): paired statistics and validation.
- [aggregated/episodes.csv](aggregated/episodes.csv): all 200 evaluation records.
- [aggregated/per_task.csv](aggregated/per_task.csv): 10 task-level comparisons.
- [aggregated/training_task_coverage.csv](aggregated/training_task_coverage.csv): training exposure.
- [figures/success_before_after_100.png](figures/success_before_after_100.png): success and uncertainty.
- [figures/training_loss.png](figures/training_loss.png): all 100 optimizer steps.
- `baseline/` and `student_100/`: 100 raw episode records, 10 completed task manifests
  and 100 videos per condition. Videos and runtime logs are present locally and
  excluded from Git; JSONL records and manifests are versioned.
- `training.jsonl`, `rollouts.jsonl`, `rollout_inputs.jsonl`: gradients/losses,
  student versions, training state identities and raw-file references.
- `diagnostic_*.jsonl`: one separately labelled diagnostic update, fully rolled
  back before either formal condition; never included in the 100 updates.
- `provenance/`: configs, model/source fingerprints, numerical diagnostics,
  checkpoint and rollout hashes, and Slurm resource accounting.

The 100 batches contain 400 student-controlled action blocks, 7,961 environment
actions and 18 distinct training initial states across all 10 tasks. Training
seed was 17, actual initial-state indices 10–11. Evaluation used seed 7 and
indices 0–9, with identical ordered states, initial observations and RNG protocol
before and after. The state hashes were disjoint. Evaluation states had already
been used in the frequency sweep; this is not an independent final test set.

The executed training/evaluation revision was `90b8368` (full revision in
`provenance/resource_plan.json`). Actual training source hashes are in
`provenance/training_setup.json` and `provenance/exported_training_manifest.json`.
Post-run reporting additionally validates checkpoint identity by path and
manifest digest: the native WebSocket server had added a `server_timing`
annotation to the returned checkpoint dictionary. Returning a copy now prevents
that annotation from entering future checkpoint metadata; optimization and
inference are unaffected by this reporting fix.

The complete step-100 checkpoint, optimizer/EMA state and normalization assets are at:

```text
/scratch/enmingzz/frequency_vla/runs/opsd_h20_100_attempt2/step_100
```

Its manifest digest is
`a7d6b7cb8cf42e2285721c007515b4a6526e4bb8c1874028f1dae841b5238a8c`.
Native inference after reloading the checkpoint exactly matched the in-memory
student on the saved diagnostic probe. The 100 replayable training-array files
(483,315,800 bytes total) are in the sibling `rollouts/train/` directory, with
paths, sizes and SHA256 hashes in `provenance/rollout_files.json`.

From the repository root, reproduce tables, findings and plots without a GPU:

```bash
source scripts/env.sh
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.opsd_analysis --results-dir results/opsd_h20_100
```

Setup, rerun and exported-policy loading commands are in [TRAINING.md](../../TRAINING.md).
Use new result/checkpoint paths for a rerun. The successful job used 22m12s;
including failed diagnostic `60044854` (archived in
`../diagnostics/opsd_h20_100_attempt1/`), this trial used **27m07s single-GPU time**.
Peak host memory was 42.68 GiB of the requested 48 GiB. The GPU was released on
completion. No further training was started after this preselected step-100 result.
