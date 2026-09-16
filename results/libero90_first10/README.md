# LIBERO-90 first-ten transfer screen

Official task IDs 0–9, three ordered layouts (40–42), seed 37, P=50 and ten flow steps. Compare original H=5, original H=20 and the preselected 500-update LIBERO-10 OPSD model at H=20. Total 90 episodes; the original H=5 prefix is reused with checksums from the user-interrupted full-suite attempt. No new training.

This is an ordered exploratory prefix, not the separate LIBERO-10 suite or an estimate for all LIBERO-90 tasks. Completed: original H=5 **5/30 (16.7%)**, original H=20 **6/30 (20.0%)**, trained H=20 **4/30 (13.3%)**. The observed training change is −6.7 pp (exact paired p=0.5); there is no demonstrated transfer benefit or positive original H=5 replanning gap in this small screen. All 90 videos and paired episode records passed validation.

See [FINDINGS.md](FINDINGS.md), [episode CSV](aggregated/episodes.csv), [success plot](figures/libero90_success.png) and [calls plot](figures/success_vs_policy_calls.png). Reproduce statistics and plots with the command in [the protocol](../../LIBERO90_TRANSFER.md).

The supplementary job used one H100 for **19m51s** and released it. Total allocated GPU time for this LIBERO-90 effort, including all failed diagnostics and the interrupted full-suite run, was **1h16m37s**; allocations never overlapped. Details: [resource accounting](provenance/resource_accounting.json).
