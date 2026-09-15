# Findings — experiment in progress

No success-rate claim is made yet. Real GPU evaluation is being prepared.

The requested protocol has an upstream constraint: official `pi05_libero` at
OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479` sets native prediction
horizon P=10. Keeping that config unchanged permits H=5 and H=10, but makes
H=20/30/40/50 impossible. Preflight and runtime checks reject unsupported H.

The full sparse-replanning premise, substantial-degradation horizon, task
sensitivity, and a large-H teacher/student pair remain unestablished. No model
changes, training or distillation have been implemented. Measured results will
replace this progress note when evaluation completes.
