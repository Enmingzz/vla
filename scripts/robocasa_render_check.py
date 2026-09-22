"""Bounded EGL and deterministic-reset probe, before loading model weights."""
import argparse
import os
from pathlib import Path
import time

import numpy as np

from frequency_vla.logging_utils import write_json
from frequency_vla.robocasa_env import Episode, IMAGE_KEYS
from frequency_vla.robocasa_protocol import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    config = load_config()
    started = time.monotonic()
    identities = []
    for _ in range(2):
        env = Episode(config,0,9001,training=True)
        try:
            identities.append(env.identity)
            for key in IMAGE_KEYS:
                frame = env.obs[key]
                if frame.shape != (224,224,3) or frame.dtype != np.uint8 or frame.std() < 1:
                    raise RuntimeError('Invalid rendered camera '+key)
            from OpenGL import GL
            env.base.sim._render_context_offscreen.gl_ctx.make_current()
            renderer = GL.glGetString(GL.GL_RENDERER).decode()
            if 'NVIDIA' not in renderer:
                raise RuntimeError('Expected NVIDIA EGL renderer: '+renderer)
            env.step(np.zeros(12))
            env.get_observation()
        finally:
            env.close()
    if identities[0] != identities[1]:
        raise RuntimeError('A fresh same-seed constructor does not reproduce the episode')
    write_json(args.output,{'passed':True,'renderer':renderer,'seconds':time.monotonic()-started,
        'duplicate_initial_state_equal':True,'probe_training_seed':identities[0]['episode_seed'],
        'formal_evaluation_episode':False,'hostname':os.uname().nodename})


if __name__ == '__main__':
    main()
