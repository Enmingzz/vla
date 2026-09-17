"""Run the existing four student environments in isolated CPU render processes.

Policy sampling, task assignment, action execution, observations and loss are
unchanged. This prevents software rendering from serializing all four slots.
"""
import multiprocessing as mp
import random
import traceback

import numpy as np

from .evaluator import array_hash
from .opsd_client import TrainingEnvironments, import_official


def _worker(connection, config, openpi_dir):
    environments = None
    try:
        random.seed(config["train_seed"])
        np.random.seed(config["train_seed"])
        import torch
        torch.manual_seed(config["train_seed"])
        environments = TrainingEnvironments(import_official(openpi_dir), dict(config, batch_size=1))
        while True:
            operation, payload = connection.recv()
            if operation == "close":
                break
            if operation == "reset":
                task, index = payload
                environments.next_task = task
                environments.state_indices[task] = index
                environments.reset(0)
                result = environments.observations[0], environments.identities[0]
            elif operation == "execute":
                result = (*environments.execute(np.asarray(payload)[None]), environments.observations[0])
            else:
                raise ValueError("Unknown environment operation: " + operation)
            connection.send((True, result))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if environments is not None:
            environments.close()
        connection.close()


class ParallelTrainingEnvironments(TrainingEnvironments):
    def __init__(self, official, config, openpi_dir):
        super().__init__(official, config)
        self.processes, self.connections = [], []
        context = mp.get_context("spawn")
        try:
            for _ in self.environments:
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, config, openpi_dir), daemon=True)
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
        except BaseException:
            self.close()
            raise

    def _receive(self, slot):
        connection = self.connections[slot]
        if not connection.poll(180):
            raise TimeoutError("Software rendering worker {} exceeded 180 seconds".format(slot))
        success, result = connection.recv()
        if not success:
            raise RuntimeError("Environment worker {} failed:\n{}".format(slot, result))
        return result

    def prepare(self):
        pending = []
        for slot in range(len(self.environments)):
            if not self.done[slot]:
                continue
            task = self.next_task % 10
            self.next_task += 1
            index = self.state_indices[task]
            self.state_indices[task] += 1
            if self.state_indices[task] >= self.config["train_initial_state_stop"]:
                self.state_indices[task] = self.config["train_initial_state_start"]
            expected_hash = array_hash(self.suite.get_task_init_states(task)[index])
            if expected_hash in self.eval_hashes:
                raise RuntimeError("Training initial state duplicates held-out evaluation")
            self.connections[slot].send(("reset", (task, index)))
            pending.append((slot, task, index, expected_hash))
        for slot, task, index, expected_hash in pending:
            observation, identity = self._receive(slot)
            if (identity["task_id"], identity["initial_state_index"], identity["initial_state_sha256"]) != (task, index, expected_hash):
                raise RuntimeError("Worker reset differs from the assigned training task/state")
            self.observations[slot], self.identities[slot] = observation, identity
            self.steps[slot], self.done[slot] = 0, False
        return self.observations

    def execute(self, actions):
        if len(actions) != len(self.environments):
            raise ValueError("Unexpected action batch")
        for connection, chunk in zip(self.connections, actions):
            connection.send(("execute", chunk))
        views = [[] for _ in range(4)]
        valid, identities = [], []
        for slot in range(len(self.environments)):
            result_views, result_valid, result_identities, final_observation = self._receive(slot)
            for block, block_views in enumerate(result_views):
                views[block].append(block_views[0])
            identity = result_identities[0]
            valid.append(result_valid[0])
            identities.append(identity)
            self.steps[slot] += result_valid[0]
            self.done[slot] = identity["episode_ended"]
            self.observations[slot] = final_observation
        return views, valid, identities

    def close(self):
        for connection in getattr(self, "connections", []):
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in getattr(self, "processes", []):
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        for connection in getattr(self, "connections", []):
            connection.close()
