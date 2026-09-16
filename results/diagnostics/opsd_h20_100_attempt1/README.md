# First temporal OPSD diagnostic, stopped before formal training

Job 60044854 used one H100, 8 CPUs, 48 GiB for 4m55s and exited on its
initial numeric consistency guard. The gradient-program offset-0 MSE exceeded
1e-5; its value was not recorded before the exception. No optimizer update was
committed and no formal before/after evaluation started.

The same-input guard is now separated from a measured comparison of forward
and differentiated-program numeric errors. See AUTORESEARCH.md. These files
are engineering diagnostics and must not be aggregated as task outcomes.
The original result paths are retained in the archived runtime metadata.
Raw diagnostic views/actions remain at
/scratch/enmingzz/frequency_vla/runs/opsd_h20_100/rollouts/diagnostic/step_001.npz.
