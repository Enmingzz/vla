"""Bounded GPU probe of official scenes and exact paired resets; no model calls."""
import argparse
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np

from frequency_vla.logging_utils import write_json
from frequency_vla.simpler_eval import make_env, reset
from frequency_vla.simpler_protocol import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run rendering probes in a GPU allocation')
    cfg, checks = load_config(), []
    for task_id, task in enumerate(cfg['tasks']):
        t = time.monotonic()
        env = make_env(task)
        try:
            _, image, identity = reset(env, cfg['seed'], task_id, 0)
            assert image.ndim == 3 and image.shape[2] == 3 and np.std(image) > 5
            args.output.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(args.output.parent/f'{task}.png', image)
            env.step(np.array([0., 0., 0., 0., 0., 0., 1.]))
            _, image2, identity2 = reset(env, cfg['seed'], task_id, 0)
            if identity != identity2:
                raise RuntimeError(f'Same-seed native reset mismatch: {task}: {identity} / {identity2}')
            checks.append({'task': task, 'identity': identity, 'shape': list(image.shape),
                'seconds': time.monotonic()-t, 'control_frequency_hz': env.unwrapped.control_freq,
                'episode_limit': env.spec.max_episode_steps})
        finally:
            env.close()
        print(checks[-1], flush=True)
    write_json(args.output, {'passed': True, 'renderer': 'sapien_vulkan_ibl',
                            'render_device': os.environ.get('SV_RENDER_DEVICE', 'cuda:0'), 'checks': checks})


if __name__ == '__main__':
    main()
