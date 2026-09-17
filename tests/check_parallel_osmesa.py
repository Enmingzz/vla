"""CPU-node check: serial and parallel training simulators receive identical actions.

Uses no policy/checkpoint/GPU and never creates formal research outcomes.
"""
import argparse
import os
import random
import time

import numpy as np

from frequency_vla.config import load_config
from frequency_vla.evaluator import array_hash, rendering_spec
from frequency_vla.logging_utils import file_digest, write_json
from frequency_vla.opsd_client import TrainingEnvironments, import_official
from frequency_vla.opsd_parallel_env import ParallelTrainingEnvironments


def signature(observations):
    return [{k: v if isinstance(v, str) else array_hash(v) for k, v in row.items()} for row in observations]


def run(parallel, config):
    random.seed(config["train_seed"])
    np.random.seed(config["train_seed"])
    official = import_official(os.environ["OPENPI_DIR"])
    envs = ParallelTrainingEnvironments(official, config, os.environ["OPENPI_DIR"]) if parallel else TrainingEnvironments(official, config)
    results, durations = [], []
    try:
        for index in range(3):
            print("{} block {}".format("parallel" if parallel else "serial", index), flush=True)
            if index == 2:
                # Exercise assignment of the next four tasks after a reset.
                envs.done = [True] * 4
            before = signature(envs.prepare())
            actions = np.random.RandomState(900 + index).uniform(-.15, .15, (4, 50, 7))
            actions[:, :, -1] = 1
            started = time.monotonic()
            views, valid, identities = envs.execute(actions)
            durations.append(time.monotonic() - started)
            results.append(dict(before=before, teacher_views=[signature(x) for x in views],
                                after=signature(envs.observations), valid=valid, identities=identities))
        return results, durations
    finally:
        envs.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    config = load_config("configs/opsd_continuation_1000.yaml")
    renderer = rendering_spec(load_config("configs/prediction50_round3.yaml"))
    serial, serial_s = run(False, config)
    parallel, parallel_s = run(True, config)
    if serial != parallel:
        write_json(args.output, dict(passed=False, serial=serial, parallel=parallel))
        raise AssertionError("Serial/parallel observations, state identity or action execution changed")
    result = dict(passed=True, renderer=renderer, action_blocks=12, simulated_actions=240,
                  exact_observation_hashes_equal=True, task_assignment_equal=True,
                  serial_execute_seconds=serial_s, parallel_execute_seconds=parallel_s,
                  serial_to_parallel_speedup=sum(serial_s)/sum(parallel_s),
                  source_sha256={p:file_digest(p) for p in ["src/frequency_vla/opsd_client.py", "src/frequency_vla/opsd_parallel_env.py"]})
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
