"""Instrument the pinned official evaluator; its observation/action loop stays intact."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from unittest.mock import patch

from .config import episode_seed, load_config, prediction_horizon, upstream_spec, validate_horizons
from .logging_utils import append_record, digest, file_digest, write_json


def array_hash(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def validate_call_schedule(control_steps, policy_calls, horizon, call_steps):
    expected = math.ceil(control_steps / horizon)
    if policy_calls != expected or call_steps != list(range(0, control_steps, horizon)):
        raise RuntimeError("Policy call schedule differs from official chunk-prefix execution: "
                           + str((control_steps, policy_calls, horizon, call_steps)))


def check_chunk(actions, prediction_length, required_horizon):
    import numpy as np
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != 7 or a.shape[0] != prediction_length or a.shape[0] < required_horizon:
        raise RuntimeError("Invalid action chunk {}; require configured P={} and maximum H={}".format(a.shape, prediction_length, required_horizon))
    if not np.isfinite(a).all():
        raise RuntimeError("Non-finite policy actions")


class EpisodeTracker:
    def __init__(self, args, metadata):
        self.args, self.metadata = args, metadata
        self.task_id = None
        self.task_description = None
        self.episode_index = 0
        self.current = None
        self.env = None
        self.records = []

    def start(self):
        if self.current is not None:
            raise RuntimeError("An episode was reset without a finalized record")
        self.started = time.monotonic()
        self.current = {"task_suite": self.args.suite, "task_id": self.task_id,
                        "task_description": self.task_description, "episode_index": self.episode_index,
                        "seed": self.args.seed, "replan_steps": self.args.horizon, "mode": self.args.mode,
                        "environment_steps": 0, "controlled_environment_steps": 0, "settling_steps": 0,
                        "policy_calls": 0, "policy_call_control_steps": [], "success": False,
                        "inference_fingerprint": self.metadata["inference_fingerprint"],
                        "evaluation_fingerprint": self.metadata["evaluation_fingerprint"],
                        "native_prediction_horizon": self.metadata["experiment_spec"]["native_prediction_horizon"],
                        "prediction_horizon": prediction_horizon(self.metadata["experiment_spec"]),
                        "episode_rng_seed": episode_seed(self.args.seed, self.args.suite, self.task_id, self.episode_index)}


class TrackedEnv:
    def __init__(self, env, tracker):
        self.env, self.tracker = env, tracker

    def reset(self):
        self.tracker.start()
        return self.env.reset()

    def set_init_state(self, state):
        self.tracker.current["initial_state_sha256"] = array_hash(state)
        return self.env.set_init_state(state)

    def step(self, action):
        result = self.env.step(action)
        row = self.tracker.current
        settling = row["environment_steps"] < self.tracker.args.num_steps_wait
        row["environment_steps"] += 1
        if settling:
            row["settling_steps"] += 1
        else:
            row["controlled_environment_steps"] += 1
            row["success"] = bool(result[2])
        return result

    def close(self):
        self.env.close()


class CountingClient:
    def __init__(self, client, tracker):
        self.client, self.tracker = client, tracker

    def infer(self, observation):
        row = self.tracker.current
        if row["controlled_environment_steps"] != row["policy_calls"] * row["replan_steps"]:
            raise RuntimeError("Unexpected replanning position")
        if row["policy_calls"] == 0:
            row["first_observation_sha256"] = digest({
                k: array_hash(v) if k != "prompt" else v for k, v in observation.items()
            })
        request = dict(observation)
        request["_frequency_vla"] = {"episode_seed": row["episode_rng_seed"], "call_index": row["policy_calls"]}
        result = self.client.infer(request)
        if result.get("inference_fingerprint") != row["inference_fingerprint"]:
            raise RuntimeError("Policy changed during evaluation")
        check_chunk(result["actions"], row["prediction_horizon"], self.tracker.args.required_horizon)
        row["action_chunk_length"] = len(result["actions"])
        row["policy_call_control_steps"].append(row["controlled_environment_steps"])
        row["policy_calls"] += 1
        return result


class FatalUpstreamErrors(logging.Handler):
    def emit(self, record):
        # Upstream catches Exception inside the rollout. Abort instead of treating crashes as failures.
        if record.getMessage().startswith("Caught exception:"):
            raise RuntimeError("Upstream evaluation error (not a task failure): " + record.getMessage())


def run(args):
    config = load_config()
    spec = upstream_spec(args.openpi_dir, config)
    validate_horizons([args.horizon, args.required_horizon] if args.horizon != args.required_horizon else [args.horizon], prediction_horizon(spec))
    if args.suite not in config["supported_suites"]:
        raise ValueError("Unsupported suite")
    if args.episodes < 1 or args.episodes > 50:
        raise ValueError("Use the first 1..50 official initial states without cycling")
    args.num_steps_wait = config["num_steps_wait"]
    root = Path(args.results_dir).resolve()
    relative = Path(args.mode) / args.suite / ("seed_" + str(args.seed)) / ("H_" + str(args.horizon))
    raw_dir = root / "raw" / relative
    video_dir = root / "videos" / relative
    # Task subsets get separate shards to avoid collisions and permit independent jobs.
    shard = "tasks_" + "_".join(str(t) for t in args.task_ids) if args.task_ids else "all_tasks"
    raw_file = raw_dir / (shard + ".jsonl")
    manifest_file = raw_dir / (shard + ".manifest.json")
    if raw_file.exists() or manifest_file.exists():
        raise FileExistsError("Refusing to overwrite an evaluation; choose a new --results-dir: " + str(raw_file))
    sys.path.insert(0, str(Path(args.openpi_dir) / "third_party/libero"))
    path = Path(args.openpi_dir) / "examples/libero/main.py"
    module_spec = importlib.util.spec_from_file_location("frequency_vla_official_libero", path)
    official = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = official
    module_spec.loader.exec_module(official)
    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    client = official._websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    server_spec = metadata.get("experiment_spec", {})
    for key in spec:
        if server_spec.get(key) != spec[key]:
            raise ValueError("Server/evaluator provenance mismatch: " + key)
    if digest(server_spec) != metadata["inference_fingerprint"]:
        raise ValueError("Invalid server fingerprint")
    evaluator_packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    metadata["evaluation_spec"] = {
        "source_sha256": {name: file_digest(Path(__file__).parent / name) for name in ["evaluator.py", "config.py", "logging_utils.py"]},
        "packages": evaluator_packages, "python_version": platform.python_version(),
        "suite": args.suite, "resize_size": config["resize_size"], "num_steps_wait": config["num_steps_wait"]}
    metadata["evaluation_fingerprint"] = digest(metadata["evaluation_spec"])
    tracker = EpisodeTracker(args, metadata)
    original_suite_class = official.benchmark.get_benchmark_dict()[args.suite]
    original_get_env = official._get_libero_env
    original_mimwrite = official.imageio.mimwrite
    manifest = {"status": "running", "arguments": vars(args), "server": metadata, "project_config": config,
                "evaluator_packages": evaluator_packages,
                "initial_state_protocol": "official get_task_init_states(task_id)[episode_index], no cycling",
                "error_policy": "abort; never score simulator/inference exceptions as failures"}
    write_json(manifest_file, manifest)

    class SelectedSuite:
        def __init__(self):
            self.base = original_suite_class()
            self.ids = args.task_ids if args.task_ids else list(range(self.base.n_tasks))
            if len(self.ids) != len(set(self.ids)) or any(i < 0 or i >= self.base.n_tasks for i in self.ids):
                raise ValueError("Invalid or duplicate task IDs")
            self.n_tasks = len(self.ids)

        def get_task(self, i):
            tracker.task_id = self.ids[i]
            tracker.episode_index = 0
            return self.base.get_task(self.ids[i])

        def get_task_init_states(self, i):
            states = self.base.get_task_init_states(self.ids[i])
            if len(states) < args.episodes:
                raise ValueError("Not enough official initial states")
            return states

    def get_env(task, resolution, seed):
        if tracker.env is not None:
            tracker.env.close()
            tracker.env = None
        env, description = original_get_env(task, resolution, seed)
        tracker.task_description = description
        tracker.env = TrackedEnv(env, tracker)
        return tracker.env, description

    def save_video(_original_filename, frames, **kwargs):
        row = tracker.current
        row["wall_clock_seconds"] = time.monotonic() - tracker.started
        validate_call_schedule(row["controlled_environment_steps"], row["policy_calls"], args.horizon, row["policy_call_control_steps"])
        if row["settling_steps"] != args.num_steps_wait:
            raise RuntimeError("Settling protocol changed")
        row["average_executed_actions_per_policy_call"] = row["controlled_environment_steps"] / row["policy_calls"]
        row["policy_calls_per_controlled_step"] = row["policy_calls"] / row["controlled_environment_steps"]
        video_dir.mkdir(parents=True, exist_ok=True)
        filename = video_dir / ("task_{:02d}_episode_{:03d}_{}.mp4".format(row["task_id"], row["episode_index"], "success" if row["success"] else "failure"))
        if filename.exists():
            raise FileExistsError("Refusing video overwrite: " + str(filename))
        original_mimwrite(filename, frames, **kwargs)
        row["video_path"] = str(filename.relative_to(root))
        append_record(raw_file, row)
        tracker.records.append(row)
        logging.info("RECORDED task=%s ep=%s H=%s success=%s steps=%s calls=%s", row["task_id"], row["episode_index"], args.horizon, row["success"], row["controlled_environment_steps"], row["policy_calls"])
        tracker.current = None
        tracker.episode_index += 1

    errors = FatalUpstreamErrors(logging.ERROR)
    logging.getLogger().addHandler(errors)
    try:
        with patch.object(official.benchmark, "get_benchmark_dict", lambda: {args.suite: SelectedSuite}), \
             patch.object(official, "_get_libero_env", get_env), \
             patch.object(official._websocket_client_policy, "WebsocketClientPolicy", lambda *a, **kw: CountingClient(client, tracker)), \
             patch.object(official.imageio, "mimwrite", save_video):
            official.eval_libero(official.Args(host=args.host, port=args.port, resize_size=config["resize_size"],
                replan_steps=args.horizon, task_suite_name=args.suite, num_steps_wait=config["num_steps_wait"],
                num_trials_per_task=args.episodes, video_out_path=str(video_dir), seed=args.seed))
        expected = (len(args.task_ids) if args.task_ids else 10) * args.episodes
        if len(tracker.records) != expected:
            raise RuntimeError("Incomplete evaluation: {} of {} episodes".format(len(tracker.records), expected))
        manifest.update(status="complete", total_episodes=len(tracker.records), total_successes=sum(r["success"] for r in tracker.records))
    except BaseException as exc:
        manifest.update(status="error", error=repr(exc), completed_episodes=len(tracker.records))
        write_json(raw_dir / (shard + ".error.json"), {"error": repr(exc), "incomplete_episode": tracker.current})
        raise
    finally:
        logging.getLogger().removeHandler(errors)
        write_json(manifest_file, manifest)
        if tracker.env is not None:
            tracker.env.close()
        client._ws.close()


def main():
    config = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openpi-dir", default=os.environ.get("OPENPI_DIR"), required=not os.environ.get("OPENPI_DIR"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mode", choices=["smoke", "main", "diagnostic"], default="smoke")
    p.add_argument("--suite", choices=config["supported_suites"], default=config["suite"])
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--required-horizon", type=int)
    p.add_argument("--episodes", type=int)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--task-ids", type=int, nargs="+")
    p.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[2] / "results"))
    args = p.parse_args()
    args.episodes = args.episodes or config["modes"].get(args.mode, 1)
    args.required_horizon = args.required_horizon or args.horizon
    logging.basicConfig(level=logging.INFO, force=True)
    run(args)


if __name__ == "__main__":
    main()
