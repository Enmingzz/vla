"""Fixed RoboCasa pilot protocol; no simulator imports needed for config checks."""
import argparse
import dataclasses
import inspect
import os
from pathlib import Path

import yaml

from .logging_utils import digest, git_commit, write_json
from .opsd_protocol import align_teacher_latent


def load_config(path=None):
    with open(path or os.environ['RC_CONFIG']) as f:
        config = yaml.safe_load(f)
    validate_training_config(config)
    return config


def validate_training_config(config):
    required = {'prediction_horizon':50, 'student_horizon':20, 'teacher_horizon':5,
        'flow_steps':10, 'loss_action_dimensions':12, 'batch_size':4,
        'teacher_strategy':'ema', 'ema_decay':0.9999, 'renderer':'egl', 'split':'pretrain'}
    if any(config.get(k) != v for k,v in required.items()):
        raise ValueError('Unexpected RoboCasa pilot method or native inference settings')
    if len(config['tasks']) != 10 or len(set(config['tasks'])) != 10:
        raise ValueError('The pilot fixes ten distinct tasks before measuring results')
    if config['optimizer_steps'] != 500 or config['checkpoint_steps'] != [100,500]:
        raise ValueError('The authorized pilot is 500 optimizer updates')
    if config['train_seed'] == config['evaluation_seed']:
        raise ValueError('Training and evaluation seed namespaces must differ')
    if config['environment_workers'] != 4 or config['evaluation_workers'] not in range(1,5):
        raise ValueError('At most four simulator workers on one allocated GPU')


def episode_seed(config, task_id, episode_index, training=False):
    if not 0 <= task_id < len(config['tasks']) or not 0 <= episode_index < 10000:
        raise ValueError('Episode identity outside the fixed seed namespace')
    base = config['train_seed' if training else 'evaluation_seed']
    value = base*10000000 + task_id*10000 + episode_index
    if not 0 <= value < 2**32:
        raise ValueError('Seed overflow')
    return value


def check_chunk(actions, config):
    import numpy as np
    a = np.asarray(actions)
    if a.shape != (config['prediction_horizon'], config['loss_action_dimensions']) or not np.isfinite(a).all():
        raise ValueError('Expected finite native (50,12) action chunk; got '+str(a.shape))
    return a


def check_install(output):
    config = load_config()
    print('Checking pinned sources and importing official dependencies...',flush=True)
    revisions = {name:git_commit(os.environ[env]) for name,env in
        [('openpi','RC_OPENPI'),('robocasa','RC_CASA'),('robosuite','RC_SUITE')]}
    if any(config[name+'_commit'] != commit for name,commit in revisions.items()):
        raise ValueError('An upstream checkout differs from the pinned protocol')
    import mujoco
    import numpy as np
    from openpi.training import config as official
    from openpi.models.pi0 import Pi0
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from openpi.shared import normalize
    from openpi.policies import robocasa_policy
    c = official.get_config(config['training_config'])
    assert (c.model.action_horizon,c.model.action_dim,c.model.max_token_len) == (50,32,200)
    assert c.model.pi05 and c.model.discrete_state_input
    assert inspect.signature(Pi0.sample_actions).parameters['num_steps'].default == 10
    assert mujoco.__version__ == '3.3.1' and np.__version__ == '2.2.5'
    stats = normalize.load(Path(os.environ['RC_CHECKPOINT'])/'assets')
    # Skip lookup of offline training data; checkpoint stats still pass through the official loader.
    data = dataclasses.replace(c.data, data_dirs=None).create(c.assets_dirs,c.model)
    sample = {'observation/state':np.zeros(16), 'prompt':'installation check',
        **{k:np.zeros((224,224,3),np.uint8) for k in
           ['observation/image','observation/wrist_image','observation/right_image']}}
    observation = robocasa_policy.RobocasaInputs(32,c.model.model_type)(sample)
    assert all(observation['image_mask'].values())
    # The official checkpoint stores statistics after padding to model width.
    # Physical observations/actions remain 16/12 dimensions at the Gym boundary.
    for key,width in [('actions',12),('state',16)]:
        assert stats[key].mean.shape == stats[key].std.shape == (32,)
        np.testing.assert_allclose(stats[key].mean[width:],0,atol=1e-12)
        np.testing.assert_allclose(stats[key].std[width:],1,atol=1e-12)
    assert not data.use_quantile_norm  # This checkpoint has mean/std, not quantiles.
    registered = {t for tasks in TASK_SET_REGISTRY.values() for t in tasks}
    assert set(config['tasks']) <= registered
    write_json(output, {'passed':True,'config_sha256':digest(config),'revisions':revisions,
        'model_config':dataclasses.asdict(c.model), 'physical_action_dimensions':12,
        'normalization_dimensions':32,
        'native_task_horizons':{t:get_task_horizon(t) for t in config['tasks']},
        'quantile_normalization':data.use_quantile_norm, 'GPU_checks_pending':True})
    print('Official pi0.5 config, padding, normalization and task checks passed.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-install',required=True)
    check_install(parser.parse_args().check_install)
