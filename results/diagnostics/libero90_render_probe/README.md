# LIBERO-90 runtime diagnostics

These runs contain no formal evaluation results and no training updates. Two full-model attempts and one traced model run reproduced a native NVIDIA EGL abort. The traced run stopped in `libnvidia-eglcore.so.580.159.03` during `mjr_readPixels`, at environment step 260 (including ten settling steps). Its 261-action prefix is stored in the sibling `libero90_model_render_trace` archive.

Rendering 400 random-action steps alone passed on fc10517. Reserving 52 GiB in the same process also passed for 800 steps. Exact policy-action replay without the policy model passed. A separate CUDA allocation process also allowed exact replay; a continuously busy CUDA memset process made rendering slow enough to hit the 120-second bound. That artificial stress test did not reproduce the same abort and does not establish a precise root cause.

CPU OSMesa initially failed because the cluster CVMFS library is incompatible with this standalone Python's system glibc. An EL9-compatible Mesa 25.0.7 RPM extracted only into the user's scratch directory fixed loading. Both 400 random-action steps and the exact 261-action prefix then completed. EGL/OSMesa image comparisons are in `image_comparison.json`; they are not pixel-identical. Software rendering took about 0.12–0.14 s/step, so disabling JAX preallocation is being tested before adopting a slower renderer for all formal conditions.

The diagnostic scripts and Slurm accounting are retained. A Slurm `COMPLETED` status for a GDB run does not mean its inferior survived: consult the trace and completion markers. Conversely, the standalone action-replay job ended with Slurm `FAILED` because GDB's `bt` command had no stack after the replay had exited normally. No such exit status is scored as task success or failure.

CPU jobs use `def-btaati`; one-GPU jobs use `rrg-btaati`. Only one GPU allocation runs at a time. All diagnostic GPU time, including failed starts, belongs in the final resource accounting.
