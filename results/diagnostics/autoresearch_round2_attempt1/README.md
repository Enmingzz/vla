# Aborted round-two runtime diagnostic

Job 60078880 on fc10511 was cancelled after 13m52s with zero completed benchmark
episodes and zero added formal optimizer updates. The native-flow diagnostic,
parameter rollback and full step-100 FP32/EMA/Adam restoration passed.

Four simulator workers then showed environment resets of 84–119 seconds (versus
about 1.3 seconds in the prior successful run); their first episodes remained
unfinished after several minutes. A graphics process held 99% GPU utilization
while the inference server was idle. The cause was not isolated. This is a runtime
failure, not measured task failure. Its GPU allocation is included in total cost.

Paths inside original manifests/logs refer to the original results directory.
All these records are excluded from research statistics. The retry adds one- and
four-worker runtime guards on training layouts, each with a 180-second bound,
restores the previously used XLA allocation fraction 0.65, and avoids this node.
The comparison matrix, update budget, loss, data split and checkpoint selection
remain unchanged. Runtime improvements are not attributed to a specific cause.
