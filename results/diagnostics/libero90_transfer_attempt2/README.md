# Second aborted LIBERO-90 startup attempt

Job 60097833 used one H100 on fc10513 for 7m05s. The single-worker pilot again aborted with SIGABRT, with faulthandler locating the abort at mujoco.mjr_readPixels in robosuite. No complete episode or formal measurement was produced. The GPU was released automatically. Both attempts count toward resource use, but neither contributes a scored task failure. The next diagnostic isolates rendering without loading the VLA.
