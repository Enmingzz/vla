"""Four on-policy RoboCasa environments, EMA velocity targets and 500 updates."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np

from .logging_utils import append_record, write_json
from .robocasa_protocol import check_chunk, load_config


def execute_block(env,actions):
    """Fresh views at 0/5/10/15; execute nothing after a terminal action."""
    start = env.steps
    views = []
    for block in range(4):
        views.append(env.get_observation())
        for k in range(5):
            if not env.done:
                env.step(actions[block*5+k])
    return (views,env.steps-start,dict(env.identity,start_step=start,
        end_step=env.steps,episode_ended=env.done,success=env.success),env.get_observation())


def _worker(connection,config):
    from .robocasa_env import Episode
    env = None
    try:
        while True:
            operation,payload = connection.recv()
            if operation == 'close':
                break
            if operation == 'reset':
                if env is not None:
                    env.close()
                    env = None
                task,index = payload
                env = Episode(config,task,index,training=True)
                result = env.get_observation(),env.identity
            elif operation == 'execute':
                actions = check_chunk(payload,config)
                result = execute_block(env,actions)
            else:
                raise ValueError('Unknown simulator operation')
            connection.send((True,result))
    except EOFError:
        pass
    except BaseException:
        connection.send((False,traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        connection.close()


class Environments:
    def __init__(self,config,diagnostic=False):
        self.config = config
        self.context = mp.get_context('spawn')
        self.workers,self.connections = [],[]
        self.done = [True]*config['batch_size']
        self.observations = [None]*config['batch_size']
        # The single discarded numeric probe has its own training-only seed pool.
        self.indices = [9000 if diagnostic else 0]*len(config['tasks'])
        self.next_task = 0
        for _ in self.done:
            parent,child = self.context.Pipe()
            worker = self.context.Process(target=_worker,args=(child,config),daemon=True)
            worker.start()
            child.close()
            self.connections.append(parent)
            self.workers.append(worker)

    def receive(self,slot):
        pipe = self.connections[slot]
        if not pipe.poll(300):
            raise TimeoutError('RoboCasa worker exceeded five minutes: '+str(slot))
        okay,response = pipe.recv()
        if not okay:
            raise RuntimeError(response)
        return response

    def prepare(self):
        pending = []
        for slot,done in enumerate(self.done):
            if done:
                task = self.next_task%len(self.config['tasks'])
                self.next_task += 1
                index = self.indices[task]
                self.indices[task] += 1
                self.connections[slot].send(('reset',(task,index)))
                pending.append(slot)
        for slot in pending:
            self.observations[slot],_ = self.receive(slot)
            self.done[slot] = False
        return self.observations

    def execute(self,actions):
        if len(actions) != len(self.connections):
            raise ValueError('Training batch size changed')
        for pipe,chunk in zip(self.connections,actions):
            pipe.send(('execute',chunk))
        future,valid,identities = [[] for _ in range(4)],[],[]
        for slot in range(len(self.connections)):
            views,count,identity,self.observations[slot] = self.receive(slot)
            for block in range(4):
                future[block].append(views[block])
            self.done[slot] = identity['episode_ended']
            valid.append(count)
            identities.append(identity)
        return future,valid,identities

    def close(self):
        for pipe in self.connections:
            try:
                pipe.send(('close',None))
            except (OSError,EOFError):
                pass
        for worker in self.workers:
            worker.join(timeout=3)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=3)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=3)
        for pipe in self.connections:
            pipe.close()


def train(config,port,root,rollouts):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from .robocasa_env import array_hash
    root,rollouts = Path(root),Path(rollouts)
    client = WebsocketClientPolicy('127.0.0.1',port)
    def request(operation,payload=None,**control):
        return client.infer(dict(payload or {},_flow_opsd={'operation':operation,**control}))
    try:
        for diagnostic,count in [(True,1),(False,config['optimizer_steps'])]:
            environments = Environments(config,diagnostic)
            folder = rollouts/('diagnostic' if diagnostic else 'train')
            folder.mkdir(parents=True,exist_ok=False)
            try:
                for step in range(count):
                    started = time.monotonic()
                    obs = environments.prepare()
                    response = request('rollout',{'observations':obs},
                                       expected_step=step,diagnostic=diagnostic)
                    actions = np.asarray(response['actions'])
                    future,valid,identities = environments.execute(actions)
                    arrays = {'student_actions':actions,'valid_steps':np.asarray(valid)}
                    for block,batch in enumerate(future):
                        for key in ['observation/image','observation/wrist_image',
                                    'observation/right_image','observation/state']:
                            arrays['offset_%s_%s'%(block*5,key.replace('/','_'))] = np.stack([x[key] for x in batch])
                    path = folder/('step_%04d.npz'%(step+1))
                    np.savez(path,**arrays)
                    result = request('update',{'future_observations':future},
                        rollout_token=response['rollout_token'],valid_steps=valid)
                    record = {'optimizer_step':step+1,'diagnostic':diagnostic,'identities':identities,
                        'action_chunk_sha256':array_hash(actions),'rollout_path':str(path),
                        'seconds':time.monotonic()-started,'training':result}
                    append_record(root/('diagnostic_rollouts.jsonl' if diagnostic else 'rollouts.jsonl'),record)
                    print('OPSD update',step+1,'diagnostic',diagnostic,'loss',result['loss'],
                          'seconds',record['seconds'],flush=True)
                    if not diagnostic and step+1 in config['checkpoint_steps']:
                        print('Saved checkpoint:',json.dumps(request('save')),flush=True)
                if diagnostic:
                    print('Diagnostic rollback:',json.dumps(request('reset_after_diagnostic')),flush=True)
            finally:
                environments.close()
        status = request('status')
        if status['step'] != 500 or 500 not in status['saved_snapshots']:
            raise RuntimeError('500-step checkpoint not complete')
        request('set_phase',phase='student')
        write_json(root/'provenance/training_complete.json',status)
    finally:
        client._ws.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--results-dir',required=True)
    parser.add_argument('--rollouts-dir',required=True)
    parser.add_argument('--catalog',required=True)
    args = parser.parse_args()
    config = load_config()
    train(config,args.port,args.results_dir,args.rollouts_dir)
    from .robocasa_eval import evaluate
    for horizon in config['post_training_horizons']:
        evaluate(config,horizon,args.port,args.results_dir,args.catalog,'step500_h'+str(horizon))


if __name__ == '__main__':
    main()
