"""Paired RoboCasa evaluation with the official action deque and success signal."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import time

from .logging_utils import append_record, digest, write_json
from .robocasa_protocol import check_chunk, load_config


def evaluate_task(config,task_id,horizon,port,root,catalog,condition):
    import imageio.v2 as imageio
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from .robocasa_env import Episode
    root = Path(root)
    task_name = config['tasks'][task_id]
    raw = root/'raw'/condition/(task_name+'.jsonl')
    if raw.exists():
        raise FileExistsError('Refusing to mix evaluation attempts: '+str(raw))
    video_dir = root/'videos'/condition/task_name
    video_dir.mkdir(parents=True,exist_ok=True)
    client = WebsocketClientPolicy('127.0.0.1',port)
    metadata = client.get_server_metadata()
    write_json(root/'provenance'/condition/(task_name+'_server.json'),metadata)
    fingerprint = metadata['inference_fingerprint']
    records = []
    try:
        for index in range(config['evaluation_episodes_per_task']):
            episode = Episode(config,task_id,index)
            try:
                episode.pair(catalog)
                calls,query_seconds = 0,0.0
                plan = deque()
                video = video_dir/('episode_%03d.mp4'%index)
                started = time.monotonic()
                with imageio.get_writer(video,fps=20,codec='libx264',
                                        output_params=['-threads','1']) as writer:
                    while not episode.done:
                        if not plan:
                            item = dict(episode.get_observation(),_frequency_vla={
                                'episode_seed':episode.seed,'call_index':calls})
                            query_start = time.monotonic()
                            response = client.infer(item)
                            query_seconds += time.monotonic()-query_start
                            if response['inference_fingerprint'] != fingerprint:
                                raise ValueError('Policy changed during a frozen evaluation')
                            actions = check_chunk(response['actions'],config)
                            plan.extend(actions[:horizon])
                            calls += 1
                        episode.step(plan.popleft())
                        if (episode.steps-1)%2 == 0 or episode.done:
                            writer.append_data(episode.frame())
                seconds = time.monotonic()-started
                if calls != math.ceil(episode.steps/horizon):
                    raise RuntimeError('Incorrect replanning frequency/action queue')
                record = {**episode.identity,'task_suite':'robocasa365_fixed10',
                    'split':config['split'],'seed':config['evaluation_seed'],'H':horizon,
                    'condition':condition,'success':episode.success,'env_steps':episode.steps,
                    'policy_calls':calls,'actions_per_policy_call':episode.steps/calls,
                    'episode_seconds':seconds,'reset_seconds':episode.reset_seconds,
                    'policy_query_seconds':query_seconds,'inference_fingerprint':fingerprint,
                    'prediction_horizon':50,'flow_steps':10,'renderer':'egl','control_frequency_hz':20,
                    'video':str(video),'paired_catalog':str(catalog),'config_sha256':digest(config)}
                append_record(raw,record)
                records.append(record)
                logging.warning('%s %s %s/%s success=%s steps=%s calls=%s seconds=%.1f',
                    condition,task_name,index+1,config['evaluation_episodes_per_task'],
                    episode.success,episode.steps,calls,seconds)
            finally:
                episode.close()
    finally:
        client._ws.close()
    return records


def evaluate(config,horizon,port,root,catalog,condition):
    if horizon not in (5,20):
        raise ValueError('This pilot compares H=5 and H=20')
    root = Path(root)
    tasks = config['tasks']
    records = []
    with ProcessPoolExecutor(max_workers=config['evaluation_workers'],
                             mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(evaluate_task,config,i,horizon,port,str(root),str(catalog),condition)
                   for i in range(len(tasks))]
        try:
            for future in as_completed(futures):
                records.extend(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            for process in pool._processes.values():
                process.terminate()
            raise
    records.sort(key=lambda x:(x['task_id'],x['episode_index']))
    if len(records) != len(tasks)*config['evaluation_episodes_per_task']:
        raise RuntimeError('Incomplete evaluation')
    folder = root/'aggregated'
    folder.mkdir(exist_ok=True)
    with (folder/(condition+'_episodes.csv')).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    successes = sum(r['success'] for r in records)
    n,p,z = len(records),successes/len(records),1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    average = lambda items,key:sum(r[key] for r in items)/len(items) if items else None
    successful = [r for r in records if r['success']]
    summary = {'condition':condition,'H':horizon,'episodes':n,'successes':successes,
        'success_rate':p,'wilson_ci95':[center-half,center+half],
        'mean_policy_calls':average(records,'policy_calls'),
        'mean_actions':average(records,'env_steps'),
        'mean_seconds':average(records,'episode_seconds'),
        'successful_mean_actions':average(successful,'env_steps'),
        'successful_mean_seconds':average(successful,'episode_seconds'),
        'per_task':{t:sum(r['success'] for r in records if r['task_name']==t)/
                        config['evaluation_episodes_per_task'] for t in tasks}}
    write_json(folder/(condition+'.json'),summary)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon',type=int,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--results-dir',required=True)
    parser.add_argument('--catalog',required=True)
    parser.add_argument('--condition',required=True)
    args = parser.parse_args()
    evaluate(load_config(),args.horizon,args.port,args.results_dir,args.catalog,args.condition)


if __name__ == '__main__':
    main()
