# Aborted resource-sizing diagnostic

Job `60040618` requested one H100, 4 CPUs, 40 GiB host memory and a 35-minute
limit on node `fc10605`. It was cancelled after **15m15s**, with **zero completed
episodes**, because simulator throughput was severely below the earlier run.
This allocation is included in resource accounting, not benchmark episode counts.

The model loaded with the identical P=50 inference fingerprint. No simulator or
inference exception was recorded before cancellation. Connection counters showed
very few policy requests; initialization/settling/rendering was progressing far
too slowly. GPU utilization samples reached 100% despite no episode throughput,
so utilization alone was not evidence of efficient evaluation. CUDA and EGL both
selected device 0, matching this allocation. The peak recorded host memory was
about 26.7 GiB, below the 40 GiB request. CPU versus node effects were not isolated.

The retry restores the previously successful 8-CPU allocation, retains one H100
and 40 GiB, and excludes `fc10605`. All H values are rerun together on a fresh
continuous server. No interrupted episode is treated as a task failure.

Runtime logs remain under `logs/` locally. Incomplete manifests are preserved
under `raw/` for diagnosis only and must not be combined with benchmark results.
The resource request, observations, cancellation record and `seff` output are
under `provenance/`. No checkpoint weights or model inference settings changed.
