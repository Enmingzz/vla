# Result archive

These are real OpenPI/LIBERO rollouts on Fir H100 GPUs, collected on 2026-09-15.
Only H=5 and H=10 are supported by the unchanged official `pi05_libero` config
(native prediction horizon P=10). No results exist for H=20/30/40/50.

| Mode | Slurm job | Episodes per task per H | Seed | Task workers | Status |
|---|---:|---:|---:|---:|---|
| Smoke | 60008811 | 10 | 7 | 1 | Complete: 200 episodes |
| Main | 60009027 | 50 | 7 | 4 | In progress |

Each job used one H100 80 GB GPU, 8 allocated CPU cores and 96 GB host memory.
The smoke job completed in 27 minutes 57 seconds. Each mode keeps one policy
server running across both horizons. The main command was:

```bash
sbatch --job-name=freq-vla-main --account=rrg-btaati --nodes=1 --ntasks=1 \
  --gpus-per-node=h100:1 --cpus-per-task=8 --mem=96G --time=04:00:00 \
  --output=results/logs/slurm-%j.log scripts/fir_job.sh main --horizons 5 10 --workers 4
```

Raw records and complete runtime manifests are under `raw/<mode>/libero_10/`.
Aggregates, paired comparisons and generated findings are under
`aggregated/<mode>/libero_10/`; the three figures are under the matching
`figures/` directory. Video paths in each record are relative to this directory.
Videos and runtime logs are saved locally and excluded from Git; raw records,
manifests, CSVs, findings and figures are included. Smoke and main observations
overlap in initial states and seeds and must not be pooled as independent data.
`provenance/checkpoint_manifest.json` preserves the exact public GCS object
generations and checksums used by both benchmark servers; its canonical digest
matches the value recorded in their inference manifests. Checkpoint weights are
stored outside this repository at the configurable `CHECKPOINT_DIR`.

The inference reproducibility probe in `diagnostics/` uses synthetic fixed
inputs and is explicitly excluded from benchmark episode counts. It documents
small output differences between independent server instances despite identical
model/config fingerprints. Within-server repetitions matched exactly.

The one-episode preliminary run in `raw/diagnostic/` (job 60007504) is also
excluded from all benchmark summaries. It preceded the CUDA setup correction:
JAX selected ptxas 12.2.140 through Fir's inherited `CUDA_ROOT`, while that old
manifest's compiler field recorded the 12.9 executable on `PATH`. Both formal
benchmark jobs use and record JAX's actual ptxas 12.9.41. The diagnostic's compiler
field therefore must not be used as evidence of the formal run configuration.
