"""RoboCasa's official Gym interface, preprocessing, horizons and success predicate."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .logging_utils import digest, write_json
from .robocasa_protocol import episode_seed
from .robocasa_pairing import xml_comparison_hash


IMAGE_KEYS = {'observation/image':'video.robot0_agentview_left',
    'observation/wrist_image':'video.robot0_eye_in_hand',
    'observation/right_image':'video.robot0_agentview_right'}
STATE_KEYS = ['state.end_effector_position_relative','state.end_effector_rotation_relative',
    'state.base_position','state.base_rotation','state.gripper_qpos']


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode()+str(value.dtype).encode()+value.tobytes()).hexdigest()


def observation(obs):
    from openpi_client import image_tools
    result = {key:image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[source]),224,224)) for key,source in IMAGE_KEYS.items()}
    result['observation/state'] = np.concatenate([obs[k] for k in STATE_KEYS],axis=0)
    result['prompt'] = obs['annotation.human.task_description']
    if result['observation/state'].shape != (16,):
        raise ValueError('RoboCasa proprioception must have 16 dimensions')
    return result


def observation_hash(obs):
    return digest({k:array_hash(v) if isinstance(v,np.ndarray) else v for k,v in obs.items()})


class UnpairedResetError(RuntimeError):
    pass


class Episode:
    def __init__(self,config,task_id,index,training=False):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run simulations inside a compute allocation')
        if os.environ.get('MUJOCO_GL') != 'egl':
            raise RuntimeError('This comparison requires EGL for every simulator')
        import robocasa  # Registers official environments.
        import gymnasium as gym
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        seed = episode_seed(config,task_id,index,training)
        random.seed(seed)
        np.random.seed(seed)
        self.config, self.task_id, self.index = config,task_id,index
        self.name, self.seed = config['tasks'][task_id],seed
        self.limit = get_task_horizon(self.name)
        started = time.monotonic()
        # Fresh constructor per episode removes dependence on preceding rollout
        # length, worker scheduling, and constructor-owned random generators.
        self.env = gym.make('robocasa/'+self.name,split=config['split'],seed=seed)
        try:
            self.raw, _ = self.env.reset(seed=seed)
            self.obs = observation(self.raw)
            self.base = self.env.unwrapped.env
            if self.base.control_freq != 20:
                raise ValueError('Unexpected native control frequency')
            self.state = np.asarray(self.base.sim.get_state().flatten())
            self.xml = self.base.sim.model.get_xml()
            self.meta = json.loads(json.dumps(self.base.get_ep_meta(),default=lambda x:
                x.tolist() if isinstance(x,np.ndarray) else x.item()))
            self.identity = {'task_id':task_id,'task_name':self.name,'episode_index':index,
                'episode_seed':seed,'training':training,'initial_state_sha256':array_hash(self.state),
                'initial_xml_sha256':hashlib.sha256(self.xml.encode()).hexdigest(),
                'initial_observation_sha256':observation_hash(self.obs),
                'environment_metadata_sha256':digest(self.meta),'task_horizon':self.limit,
                'task_description':self.obs['prompt']}
            self.steps, self.success, self.done = 0,False,False
            self.reset_seconds = time.monotonic()-started
        except BaseException:
            self.env.close()
            raise

    def pair(self,catalog,allow_obj_mime_equivalence=False):
        """Atomic cross-job comparison before any evaluation actions are taken."""
        if self.identity['training']:
            raise ValueError('Training must not write the evaluation catalog')
        root = Path(catalog)/self.name/('episode_%03d'%self.index)
        root.mkdir(parents=True,exist_ok=True)
        with (root/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path = root/'identity.json'
            if path.exists():
                expected = json.loads(path.read_text())
                differing = [k for k in expected if expected[k] != self.identity.get(k)]
                if allow_obj_mime_equivalence:
                    xml = (root/'model.xml').read_text()
                    if hashlib.sha256(xml.encode()).hexdigest() != expected['initial_xml_sha256']:
                        raise ValueError('The recorded XML changed')
                    comparable = xml_comparison_hash(xml)
                    if xml_comparison_hash(self.xml) == comparable and 'initial_xml_sha256' in differing:
                        differing.remove('initial_xml_sha256')
                if differing:
                    raise UnpairedResetError('{} episode {}: unpaired evaluation reset: {}'.format(
                        self.name,self.index,differing))
                if allow_obj_mime_equivalence:
                    self.identity['initial_xml_comparison_sha256'] = comparable
                    self.identity['paired_catalog_identity_sha256'] = digest(expected)
            else:
                if allow_obj_mime_equivalence:
                    raise FileNotFoundError('Recovery requires an existing episode catalog: '+str(path))
                np.savez(root/'initial_state.npz',state=self.state)
                (root/'model.xml').write_text(self.xml)
                write_json(root/'environment_metadata.json',self.meta)
                write_json(path,self.identity)

    def step(self,action):
        from robocasa.utils.env_utils import convert_action
        if self.done:
            raise RuntimeError('Attempted action after episode termination')
        self.raw,_,_,_,info = self.env.step(convert_action(np.asarray(action)))
        self.steps += 1
        self.success = bool(info['success'])
        self.done = self.success or self.steps >= self.limit
        # Only success or the registry horizon ends an official RoboCasa rollout.
        self.obs = None
        return self.done

    def get_observation(self):
        if self.obs is None:
            self.obs = observation(self.raw)
        return self.obs

    def frame(self):
        return np.ascontiguousarray(self.env.render())

    def close(self):
        self.env.close()
