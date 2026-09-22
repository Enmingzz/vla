"""Diagnose reset mismatches and verify official XML/state replay on allocated EGL."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

import numpy as np

from frequency_vla.logging_utils import digest, write_json
from frequency_vla.robocasa_env import Episode, array_hash, observation, observation_hash
from frequency_vla.robocasa_protocol import load_config


def capture(episode):
    raw = episode.base._get_observations(force_update=True)
    episode.raw = episode.env.unwrapped.get_observation(raw)
    episode.obs = observation(episode.raw)
    return {'initial_state_sha256':array_hash(episode.base.sim.get_state().flatten()),
        'initial_xml_sha256':hashlib.sha256(episode.base.sim.model.get_xml().encode()).hexdigest(),
        'initial_observation_sha256':observation_hash(episode.obs),
        'environment_metadata_sha256':digest(json.loads(json.dumps(episode.base.get_ep_meta(),
            default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    config = load_config()
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import reset_to
    records=[]
    # Includes the first missing episodes in both failed conditions; no task score selection.
    for task,index in [(4,0),(0,8),(7,1),(4,8),(5,7),(6,4),(2,7),(3,8)]:
        task_name=config['tasks'][task]
        source=Path(args.catalog)/task_name/('episode_%03d'%index)
        expected=json.loads((source/'identity.json').read_text())
        folder=output/task_name/('episode_%03d'%index)
        folder.mkdir(parents=True,exist_ok=True)
        env=Episode(config,task,index)
        try:
            before=dict(env.identity)
            (folder/'before.xml').write_text(env.xml)
            write_json(folder/'before_metadata.json',env.meta)
            np.savez(folder/'before_state.npz',state=env.state)
            write_json(folder/'expected.json',expected)
            changed=[k for k,v in expected.items() if before[k]!=v]
            xml_diff=list(difflib.unified_diff((source/'model.xml').read_text().splitlines(),env.xml.splitlines()))
            (folder/'xml_diff.txt').write_text('\n'.join(xml_diff)+'\n')
            state=np.load(source/'initial_state.npz')['state']
            reset_to(env.base,{'states':state,'model':(source/'model.xml').read_text(),
                'ep_meta':(source/'environment_metadata.json').read_text()})
            after=capture(env)
            after_changed=[k for k,v in after.items() if expected[k]!=v]
            write_json(folder/'after.json',after)
            write_json(folder/'after_metadata.json',json.loads(json.dumps(env.base.get_ep_meta(),
                default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())))
            (folder/'after.xml').write_text(env.base.sim.model.get_xml())
            np.savez(folder/'after_state.npz',state=env.base.sim.get_state().flatten(),
                proprio=env.obs['observation/state'])
            record={'task':task_name,'episode_index':index,'before_changed':changed,
                'after_changed':after_changed,'xml_diff_lines':len(xml_diff),
                'restored_state_max_abs':float(np.max(np.abs(state-env.base.sim.get_state().flatten())))}
            print(json.dumps(record),flush=True)
            records.append(record)
            write_json(output/'checks.json',records)
        finally:
            env.close()


if __name__=='__main__':
    main()
