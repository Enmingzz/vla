"""Small, explicit contracts for the SimplerEnv frequency experiment."""
from collections import deque
import hashlib
import os
from pathlib import Path

import numpy as np
import yaml

from .config import validate_horizons
from .logging_utils import digest, file_digest, git_commit


def load_config(path=None):
    cfg = yaml.safe_load(Path(path or os.environ['SV_CONFIG']).read_text())
    validate_horizons(cfg['horizons'], cfg['native_prediction_horizon'])
    if cfg['prediction_horizon'] != cfg['native_prediction_horizon']:
        raise ValueError('This Simpler experiment must preserve the checkpoint prediction horizon')
    if cfg['native_prediction_horizon'] != 5 or cfg['flow_steps'] != 10:
        raise ValueError('Reaudit the released checkpoint before changing P or flow steps')
    return cfg


def array_digest(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def episode_seed(seed, task_id, episode_index):
    return int(np.random.SeedSequence([seed, task_id, episode_index]).generate_state(1)[0])


def noise_at_step(seed, task_id, episode_index, step, prediction_horizon=5):
    # Same Gaussian noise at shared query times, independent of H and job order.
    rng = np.random.default_rng(np.random.SeedSequence([seed, task_id, episode_index, step, 9183]))
    return rng.standard_normal((prediction_horizon, 32), dtype=np.float32)


class ChunkPrefix:
    def __init__(self, horizon, prediction_horizon):
        validate_horizons([horizon], prediction_horizon)
        self.horizon, self.prediction_horizon = horizon, prediction_horizon
        self.queue = deque()
        self.call_steps = []

    def next(self, step, query):
        if not self.queue:
            chunk = np.asarray(query())
            if chunk.shape != (self.prediction_horizon, 7) or not np.isfinite(chunk).all():
                raise ValueError(f'Invalid full action chunk {chunk.shape}; expected ({self.prediction_horizon}, 7)')
            self.call_steps.append(step)
            self.queue.extend(chunk[:self.horizon].copy())
        return self.queue.popleft()


def bridge_action(action):
    """Official Simpler Bridge convention: delta XYZ/Euler, absolute gripper."""
    from transforms3d.euler import euler2axangle
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (7,) or not np.isfinite(a).all():
        raise ValueError('Expected seven finite physical action components')
    axis, angle = euler2axangle(*a[3:6])
    # Matches simpler_env.policies.octo.OctoInference's WidowX conversion.
    gripper = 1.0 if a[6] > 0.5 else -1.0
    return np.r_[a[:3], axis * angle, gripper]


def source_manifest():
    project = Path(os.environ['FREQUENCY_PROJECT'])
    files = list((project/'src/frequency_vla').glob('simpler_*.py'))
    files += list((project/'scripts').glob('*simpler*.sh'))
    files += list((project/'scripts').glob('*simpler*.py'))
    files += [project/'scripts/check_cuda_allocation.py']
    files += [project/'src/frequency_vla'/n for n in ['analysis.py','config.py','evaluator.py','logging_utils.py']]
    files += [Path(os.environ['SV_CONFIG']), project/'configs/simpler.constraints.txt']
    return {'project_commit': git_commit(project),
            'sources': {str(p.relative_to(project)): file_digest(p) for p in sorted(set(files))},
            'config': load_config()}


def validate_frozen(frozen):
    now = source_manifest()
    if now['sources'] != frozen['sources'] or now['config'] != frozen['config']:
        raise RuntimeError('Simpler code/config changed since submission; create a fresh run')
    return digest({'sources': frozen['sources'], 'config': frozen['config']})
