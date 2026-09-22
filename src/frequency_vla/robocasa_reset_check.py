"""Check every outstanding reset before loading policy weights on the GPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

from .logging_utils import write_json
from .robocasa_analysis import CONDITIONS
from .robocasa_protocol import load_config


def check_task(config,task_id,indices,catalog,output):
    from .robocasa_env import Episode
    rows = []
    for index in indices:
        episode = Episode(config,task_id,index,reset_catalog=catalog)
        try:
            episode.pair(catalog,allow_obj_mime_equivalence=True)
            rows.append({'identity':episode.identity,'region_relabels':episode.reset_region_events})
            write_json(Path(output)/('reset_preflight_task_%02d.json'%task_id),rows)
        finally:
            episode.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--catalog',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    config = load_config()
    tasks = []
    for task_id,name in enumerate(config['tasks']):
        missing = set()
        for folder,condition in CONDITIONS:
            path = Path(args.source)/folder/'raw'/condition/(name+'.jsonl')
            done = {json.loads(line)['episode_index'] for line in path.read_text().splitlines() if line} if path.exists() else set()
            missing.update(set(range(config['evaluation_episodes_per_task']))-done)
        if missing:
            tasks.append((task_id,sorted(missing)))
    rows = []
    with ProcessPoolExecutor(max_workers=config['evaluation_workers'],mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(check_task,config,task,indices,args.catalog,args.output) for task,indices in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())
    write_json(Path(args.output)/'reset_preflight.json',{'passed':True,'episodes':len(rows),
        'policy_calls':0,'xml_replay_used':False,'rows':rows})
    print('Exact recorded reset checks passed for all {} outstanding episode identities.'.format(len(rows)),flush=True)


if __name__ == '__main__':
    main()
