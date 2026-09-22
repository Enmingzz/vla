# Failed initialization probe: no policy evaluation

Slurm job `60921197` exited with status 1 after 16 seconds on `fc10609`.
The allocated H100 and Vulkan renderer were usable. The first task rendered
successfully, but repeated same-seed initialization in a reused physical scene
produced different physical-state and first-image SHA-256 hashes. The strict
pairing guard stopped the job before model loading.

**Completed policy episodes: 0. Policy calls: 0. No success rate is available.**
This is an environment initialization failure, not a measured policy failure.

The first state hash was
`176016ef1ee024cec2c4802d5c4612443a023de7adcd4923f494a2ef2873d43b`;
the repeated-reset state hash was
`f3dfba6f08eccb1c7dd7d6edfa145f7d9a0d52ca40628d8f30c2b466089e5f4d`.
The first and repeated image hashes also differed. Numeric states were not saved
by this initial probe, so their physical difference cannot be quantified from
these hashes. The retry saves numeric diagnostics if a mismatch recurs.

The next attempt is `../simpler_pi05_p5_smoke2` (job `60947294`). It uses the
upstream native scene-reconfiguration reset option for every episode and retains
the exact state/image requirements. Model weights, P=5, H=1/2/5, seeds, task
identities, layouts, flow steps and success definition are unchanged.
