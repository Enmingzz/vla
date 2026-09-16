import json, os, sys
from pathlib import Path
os.environ.setdefault('FREQUENCY_CONFIG',str(Path('configs/prediction50_libero90.yaml').resolve()))
import frequency_vla.evaluator as ev
from openpi_client import websocket_client_policy
import numpy as np
root=Path('results/diagnostics/libero90_model_render_trace')
events=[json.loads(s) for s in (root/'action_trace.jsonl').read_text().splitlines()]
chunks=[r['actions'] for r in events if r['event']=='inference']
actions=[r['action'] for r in events if r['event']=='before_step']
metadata=json.loads((root/'provenance/server_step_0.json').read_text())
class Socket:
    def close(self): pass
class Replay:
    _ws=Socket()
    def get_server_metadata(self): return metadata
    def infer(self,obs):
        i=obs['_frequency_vla']['call_index']
        return {'actions':np.array(chunks[i]),'inference_fingerprint':metadata['inference_fingerprint']}
websocket_client_policy.WebsocketClientPolicy=lambda *args,**kw:Replay()
step=ev.TrackedEnv.step
def replay_step(self,action):
    index=self.tracker.current['environment_steps']
    if not np.array_equal(np.asarray(action),np.array(actions[index])):
        raise ValueError('Replay action differs at '+str(index))
    result=step(self,action)
    if index in [0,9,100,200,260]:
        image_root=Path(sys.argv[sys.argv.index('--results-dir')+1])/'render_frames'
        image_root.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(image_root/('step_'+str(index)+'.npz'),
            agent=result[0]['agentview_image'],wrist=result[0]['robot0_eye_in_hand_image'],
            eef=result[0]['robot0_eef_pos'])
    if index%20==0: print('REPLAY_STEP',index,flush=True)
    if index+1==len(actions):
        print('REPLAY_COMPLETE',len(actions),'exact actions, no policy model loaded',flush=True)
        raise SystemExit(0)
    return result
ev.TrackedEnv.step=replay_step
ev.main()
