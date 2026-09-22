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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task_id, task in enumerate(cfg['tasks']):
        t = time.monotonic()
        env = make_env(task)
        try:
            for index in range(cfg['episodes_per_task']['smoke']):
                _, image, identity = reset(env, cfg['seed'], task_id, index)
                state = env.unwrapped.get_state().copy()
                assert image.ndim == 3 and image.shape[2] == 3 and np.std(image) > 5
                imageio.imwrite(args.output.parent/f'{task}_{index}.png', image)
                env.step(np.array([0., 0., 0., 0., 0., 0., 1.]))
                _, image2, identity2 = reset(env, cfg['seed'], task_id, index)
                state2 = env.unwrapped.get_state().copy()
                check = {'task': task, 'episode_index': index, 'identity': identity,
                    'repeated_identity': identity2, 'shape': list(image.shape),
                    'state_max_abs_difference': float(np.max(np.abs(state2-state))),
                    'image_max_abs_difference': int(np.max(np.abs(image2.astype(int)-image.astype(int)))),
                    'passed': identity == identity2, 'seconds': time.monotonic()-t,
                    'control_frequency_hz': env.unwrapped.control_freq,
                    'episode_limit': env.spec.max_episode_steps}
                checks.append(check)
                write_json(args.output, {'passed': all(c['passed'] for c in checks),
                    'complete': False, 'renderer': 'sapien_vulkan_ibl', 'checks': checks})
                print(check, flush=True)
                if identity != identity2:
                    np.savez_compressed(args.output.parent/f'{task}_{index}_reset_mismatch.npz',
                                        state=state, repeated_state=state2, image=image, repeated_image=image2)
                    raise RuntimeError(f'Same-seed native reset mismatch: {task}; saved numeric diagnostic')
        finally:
            env.close()
    write_json(args.output, {'passed': True, 'complete': True, 'renderer': 'sapien_vulkan_ibl',
                            'render_device': os.environ.get('SV_RENDER_DEVICE', 'cuda:0'), 'checks': checks})


if __name__ == '__main__':
    main()
