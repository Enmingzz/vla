"""Paired chunk-prefix evaluation on official Simpler prepackaged environments."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .evaluator import validate_call_schedule
from .logging_utils import append_record, digest, write_json
from .simpler_protocol import (ChunkPrefix, array_digest, bridge_action, episode_seed,
                               load_config, noise_at_step, validate_frozen)


def make_env(task):
    import simpler_env
    # Explicitly preserve the official default IBL shaders. SAPIEN is Vulkan.
    return simpler_env.make(task, obs_mode='rgbd', shader_dir='ibl',
                           renderer_kwargs={'device': os.environ.get('SV_RENDER_DEVICE', 'cuda:0')})


def reset(env, seed, task_id, index):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    rng_seed = episode_seed(seed, task_id, index)
    # Rebuild through the simulator's native reset API. Clearing velocities in
    # a reused PhysX scene does not recreate its contact/solver state.
    obs, info = env.reset(seed=rng_seed, options={
        'reconfigure': True, 'obj_init_options': {'episode_id': index}})
    image = get_image_from_maniskill2_obs_dict(env, obs)
    identity = {'initial_state_sha256': array_digest(env.unwrapped.get_state()),
                'first_observation_sha256': array_digest(image), 'episode_rng_seed': rng_seed,
                'task_description': env.unwrapped.get_language_instruction(),
                'object_episode_id': int(info['episode_id']),
                'reset_protocol': 'native_reconfigure_each_episode'}
    return obs, image, identity


def run_episode(env, client, cfg, task_id, index, h, output, identity, initial_image):
    import imageio.v2 as imageio
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    task = cfg['tasks'][task_id]
    image = initial_image
    prefix = ChunkPrefix(h, cfg['native_prediction_horizon'])
    frames, raw_actions, executed_actions = [image], [], []
    chunks, call_noise_hashes = [], []
    predicted_success, ever_success, truncated = False, False, False
    step = 0
    started = time.monotonic()
    while not truncated:
        def query():
            noise = noise_at_step(cfg['seed'], task_id, index, step)
            chunk = client.infer({'image': image, 'prompt': identity['task_description'], 'noise': noise})['actions']
            chunks.append(chunk)
            call_noise_hashes.append(array_digest(noise))
            return chunk
        raw = prefix.next(step, query)
        action = bridge_action(raw)
        obs, reward, terminated, truncated, info = env.step(action)
        # Follow the official evaluator: final success at time limit, not any-hit
        # success or early termination. Also keep any-hit success as a diagnostic.
        predicted_success = bool(terminated)
        if predicted_success != bool(info['success']):
            raise RuntimeError('Simulator termination/success signals disagree')
        ever_success |= predicted_success
        image = get_image_from_maniskill2_obs_dict(env, obs)
        frames.append(image)
        raw_actions.append(raw)
        executed_actions.append(action)
        step += 1
        if step > env.spec.max_episode_steps:
            raise RuntimeError('Official time-limit wrapper did not truncate the episode')
    elapsed = time.monotonic() - started
    validate_call_schedule(step, len(prefix.call_steps), h, prefix.call_steps)
    assert step == env.spec.max_episode_steps
    stem = f'{task}/episode_{index:03d}_H{h}'
    video = output/'videos'/f'{stem}.mp4'
    trajectory = output/'trajectories'/f'{stem}.npz'
    video.parent.mkdir(parents=True, exist_ok=True)
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video, frames, fps=cfg['control_frequency_hz'], macro_block_size=1)
    np.savez_compressed(trajectory, raw_actions=raw_actions, executed_actions=executed_actions,
                        predicted_chunks=chunks, policy_call_control_steps=prefix.call_steps)
    return {'task_suite': cfg['benchmark'], 'task_id': task_id, 'task_name': task,
            'episode_index': index, 'seed': cfg['seed'], 'replan_steps': h,
            'native_prediction_horizon': 5, 'prediction_horizon': 5, 'action_chunk_length': 5,
            'success': predicted_success, 'ever_success': ever_success,
            'environment_steps': step, 'controlled_environment_steps': step, 'settling_steps': 0,
            'policy_calls': len(prefix.call_steps), 'policy_call_control_steps': prefix.call_steps,
            'average_executed_actions_per_policy_call': step/len(prefix.call_steps),
            'wall_clock_seconds': elapsed, 'video': str(video), 'trajectory': str(trajectory),
            'first_chunk_sha256': array_digest(chunks[0]), 'policy_noise_sha256': call_noise_hashes,
            'control_frequency_hz': cfg['control_frequency_hz'],
            'replanning_frequency_hz': cfg['control_frequency_hz']/h,
            'success_protocol': 'final_success_at_official_time_limit', **identity}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mode', choices=['probe','smoke','main'], required=True)
    args = p.parse_args()
    cfg = load_config()
    frozen = json.loads((args.root/'submission_sources.json').read_text())
    evaluation_fingerprint = validate_frozen(frozen)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host='127.0.0.1', port=args.port)
    metadata = client.get_server_metadata()
    if metadata['native_prediction_horizon'] != cfg['native_prediction_horizon']:
        raise RuntimeError('Server has the wrong prediction horizon')
    output = args.root/args.mode
    output.mkdir(parents=True, exist_ok=True)
    record_path = output/'raw/episodes.jsonl'
    if record_path.exists():
        raise RuntimeError('Use a fresh output directory; no implicit reuse of measured rows')
    write_json(output/'provenance/server.json', metadata)
    # Rotate H order between paired episodes to balance machine warmup/drift.
    reference, first_chunks = {}, {}
    for task_id, task in enumerate(cfg['tasks']):
        env = make_env(task)
        try:
            assert env.unwrapped.control_freq == cfg['control_frequency_hz']
            for index in range(cfg['episodes_per_task'][args.mode]):
                shift = (task_id + index) % len(cfg['horizons'])
                order = cfg['horizons'][shift:] + cfg['horizons'][:shift]
                for h in order:
                    _, image, identity = reset(env, cfg['seed'], task_id, index)
                    key = (task_id, index)
                    if key in reference and reference[key] != identity:
                        raise RuntimeError(f'Unpaired reset at {key}: {reference[key]} != {identity}')
                    reference[key] = identity
                    row = run_episode(env, client, cfg, task_id, index, h, output, identity, image)
                    if key in first_chunks and first_chunks[key] != row['first_chunk_sha256']:
                        raise RuntimeError('Identical initial observations/noise produced different chunks')
                    first_chunks[key] = row['first_chunk_sha256']
                    row.update(inference_fingerprint=metadata['inference_fingerprint'],
                               evaluation_fingerprint=evaluation_fingerprint)
                    append_record(record_path, row)
                    print(json.dumps({k: row[k] for k in ['task_name','episode_index','replan_steps','success',
                                                          'ever_success','policy_calls','wall_clock_seconds']}), flush=True)
        finally:
            env.close()
    from .simpler_analysis import aggregate
    aggregate(args.root, args.mode)


if __name__ == '__main__':
    main()
