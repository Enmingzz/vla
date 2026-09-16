# Cancelled H=15/20 follow-up attempt 2

Job 60041592 used one H100, 8 CPUs and 40 GiB on fc10519 for 14m50s.
No benchmark episodes completed. Explicit WebSocket keepalive and opening-handshake
timeouts occurred. The old ordered worker queue started new tasks despite a
later-index task failure. These diagnostics are excluded from every result table.

The benchmark memory cgroup reported no OOM; a separate 1 GiB debugger step
(60041592.2) was OOM-killed and is visible in Slurm accounting. The debugger was
not an inference or simulator worker. The root cause of unusually slow startup
and rollout was not yet isolated.

Archived manifest arguments retain the original result directory path. Runtime
logs remain locally available under logs/ and are ignored by Git.
