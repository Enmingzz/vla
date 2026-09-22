# Simpler CPU setup

CPU Slurm job `60920522` completed successfully in 2m46s, without requesting a GPU.
It installed isolated Simpler/OpenPI environments, downloaded and SHA-256-verified
the fixed Bridge checkpoint and tokenizer, and checked environment imports.

Four CPU protocol tests passed. Saved checkpoint metadata was checked directly:
native P=5, π0.5, 32 padded action dimensions, no state tokens. The safetensors
header contains 812 tensors; the action projection shape is `[32, 1024]`.

Follow-up smoke job: `60921197`, one H100, four CPUs, 48 GiB, 30-minute cap,
`def-btaati`. Its separate archive is `../simpler_pi05_p5_smoke1`.
At submission there were no Simpler policy outcomes to report.

See [the protocol and exact commands](../../docs/SIMPLER.md). Published artifact
provenance conflicts about SFT versus RL remain unresolved and are recorded.
