# Simpler smoke retry with native scene reconfiguration

Job `60947294`: one H100, four CPUs, 48 GiB, `def-btaati`, 20-minute cap.
Submitted after four CPU protocol tests passed.

The retry recreates the physics scene through the official `reconfigure=True`
reset option for every episode. Its preflight checks both smoke layouts for
each of the four tasks, including state and image equality after taking an
action and resetting again. Only after all eight checks pass does it load
the fixed π0.5 checkpoint and execute the 24-rollout H=1/2/5 smoke sweep.

This record describes the submitted protocol, not a completed evaluation.
No result is claimed at submission. See the raw logs and `provenance/vulkan.json`
for measured progress. The original failed attempt is preserved separately.
