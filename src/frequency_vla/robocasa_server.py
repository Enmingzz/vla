"""Official RoboCasa loader with a common native sampler for all three jobs."""
import argparse
import dataclasses
import functools
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch
import uuid

from .logging_utils import digest, file_digest, git_commit, write_json
from .robocasa_protocol import check_chunk, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--checkpoint-root', required=True)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = load_config()
    import jax
    import numpy as np
    from openpi.models.pi0 import Pi0
    from openpi.policies import policy_config
    from openpi.training import config as official
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    import websockets.asyncio.server as websocket_server
    from .robocasa_backend import TemporalOPSD

    if not os.environ.get('SLURM_JOB_ID') or not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('Model execution requires an allocated GPU compute node')
    revisions = {name:git_commit(os.environ[env]) for name,env in
        [('openpi','RC_OPENPI'),('robocasa','RC_CASA'),('robosuite','RC_SUITE')]}
    if any(config[name+'_commit'] != rev for name,rev in revisions.items()):
        raise ValueError('Upstream revision changed')
    checkpoint = Path(os.environ['RC_CHECKPOINT'])
    manifest = json.loads((checkpoint/'download_manifest.json').read_text())
    source = 'hf://'+config['checkpoint_repo']+'/'+config['checkpoint_subdir']
    if manifest['revision'] != config['checkpoint_revision'] or manifest['source'] != source:
        raise ValueError('Checkpoint revision changed')
    # Full SHA checks ran on CPU during download; recheck bytes and small stats here.
    for item in manifest['objects']:
        path = checkpoint/item['name']
        if path.stat().st_size != item['size'] or (item['name'].startswith('assets/')
                and file_digest(path) != item['sha256']):
            raise ValueError('Checkpoint file changed: '+str(path))
    native = official.get_config(config['training_config'])
    if not (native.model.pi05 and native.model.discrete_state_input
            and native.model.action_horizon == 50 and native.model.action_dim == 32):
        raise ValueError('Unexpected native RoboCasa model configuration')
    if inspect.signature(Pi0.sample_actions).parameters['num_steps'].default != config['flow_steps']:
        raise ValueError('Native flow sampler default changed')
    # The factory otherwise tries unavailable offline dataset paths, even when
    # loading a checkpoint. Only disable that fallback; official transforms and
    # checkpoint normalization assets are still loaded by create_trained_policy.
    effective = dataclasses.replace(native, data=dataclasses.replace(native.data, data_dirs=None))
    policy = policy_config.create_trained_policy(effective, checkpoint)
    if policy._sample_kwargs:
        raise ValueError('Unexpected sampling override')
    packages = {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    spec = {'checkpoint':manifest['source'], 'checkpoint_revision':manifest['revision'],
        'checkpoint_object_manifest_sha256':digest(manifest), 'upstream_revisions':revisions,
        'training_config':config['training_config'], 'model_config':dataclasses.asdict(native.model),
        'prediction_horizon':50, 'flow_steps':10, 'sample_kwargs':{}, 'physical_action_dimensions':12,
        'resize_size':224, 'image_preprocessing':'official resize_with_pad and uint8; no extra flip',
        'rng_protocol':'fold_in(episode_seed,call_index); native Policy.infer split',
        'dtype':'bfloat16', 'native_sampler':'Pi0.sample_actions with dynamic parameters in every condition',
        'inference_packages':{k:v for k,v in packages.items() if k.lower() in
            {'jax','jaxlib','flax','numpy','mujoco','robosuite','robocasa','pillow','opencv-python',
             'sentencepiece','orbax-checkpoint','tensorstore','ml-dtypes'}},
        'source_sha256':{n:file_digest(Path(__file__).parent/n) for n in
            ['robocasa_server.py','robocasa_backend.py','robocasa_protocol.py','opsd_flow.py']}}
    metadata = {'experiment_spec':spec, 'inference_fingerprint':digest(spec),
        'server_instance_id':str(uuid.uuid4()), 'hostname':socket.gethostname(),
        'checkpoint_local_path':str(checkpoint), 'packages':packages,
        'devices':[{'device':str(d),'kind':d.device_kind} for d in jax.devices()]}
    backend = TemporalOPSD(policy, metadata, config, args.results_dir, args.checkpoint_root, checkpoint)
    started = time.monotonic()
    warmup = {'observation/state':np.zeros(16), 'prompt':'warmup',
        **{k:np.zeros((224,224,3),np.uint8) for k in
            ['observation/image','observation/wrist_image','observation/right_image']}}
    output = backend.infer(dict(warmup, _frequency_vla={'episode_seed':0,'call_index':0}))
    check_chunk(output['actions'], config)
    # Also establish agreement with the official Policy.infer wrapper.
    policy._rng = jax.random.fold_in(jax.random.key(0),0)
    official_actions = policy.infer(warmup)['actions']
    difference = float(np.max(np.abs(output['actions']-official_actions)))
    if difference > 0.01:
        raise RuntimeError('Common native sampler differs from official wrapper: '+str(difference))
    metadata['warmup'] = {'seconds':time.monotonic()-started,
        'official_wrapper_max_abs_difference':difference, 'benchmark_episode':False}
    write_json(Path(args.results_dir)/'provenance/startup_server.json',metadata)

    class PairedPolicy:
        def infer(self, observation):
            if not args.train and '_flow_opsd' in observation:
                raise ValueError('This baseline server permits inference only')
            return backend.infer(observation)

    logging.info('Ready: native P=50, actions=12, flow=10; training=%s',args.train)
    with patch.object(websocket_server,'serve',functools.partial(websocket_server.serve,ping_interval=None)):
        WebsocketPolicyServer(PairedPolicy(),host='127.0.0.1',port=args.port,
                              metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
