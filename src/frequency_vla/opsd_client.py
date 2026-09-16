"""Student-controlled LIBERO rollouts with fresh, training-only teacher views."""
import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from .config import load_config
from .evaluator import array_hash, check_chunk
from .logging_utils import append_record, digest
from .opsd_protocol import validate_training_config


def import_official(root):
    sys.path.insert(0, str(Path(root) / "third_party/libero"))
    spec = importlib.util.spec_from_file_location("opsd_official_libero", Path(root) / "examples/libero/main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def policy_observation(official, observation, prompt):
    tools = official.image_tools
    return {
        "observation/image": tools.convert_to_uint8(tools.resize_with_pad(
            np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]), 224, 224)),
        "observation/wrist_image": tools.convert_to_uint8(tools.resize_with_pad(
            np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224)),
        "observation/state": np.concatenate([observation["robot0_eef_pos"],
            official._quat2axisangle(observation["robot0_eef_quat"]), observation["robot0_gripper_qpos"]]),
        "prompt": str(prompt)}


class TrainingEnvironments:
    def __init__(self, official, config):
        self.official, self.config = official, config
        self.suite = official.benchmark.get_benchmark_dict()["libero_10"]()
        self.next_task = 0
        self.state_indices = [config.get("train_initial_state_cursor", config["train_initial_state_start"])] * 10
        self.environments = [None] * config["batch_size"]
        self.observations, self.identities = [None] * len(self.environments), [None] * len(self.environments)
        self.steps, self.done = [0] * len(self.environments), [True] * len(self.environments)
        # Protect the actual states, as well as their indices, from evaluation leakage.
        protected = config.get("protected_eval_ranges", [[0, config["evaluation_episodes_per_task"]]])
        self.eval_hashes = {array_hash(state) for task in range(10) for start, stop in protected
            for state in self.suite.get_task_init_states(task)[start:stop]}

    def reset(self, slot):
        if self.environments[slot] is not None:
            self.environments[slot].close()
        task_id = self.next_task % 10
        self.next_task += 1
        index = self.state_indices[task_id]
        self.state_indices[task_id] += 1
        if self.state_indices[task_id] >= self.config["train_initial_state_stop"]:
            self.state_indices[task_id] = self.config["train_initial_state_start"]
        state = self.suite.get_task_init_states(task_id)[index]
        state_hash = array_hash(state)
        if state_hash in self.eval_hashes:
            raise RuntimeError("Training initial state duplicates a held-out evaluation state")
        env, prompt = self.official._get_libero_env(self.suite.get_task(task_id), 256, self.config["train_seed"])
        self.environments[slot] = env
        env.reset()
        obs = env.set_init_state(state)
        for _ in range(10):
            obs, _, _, _ = env.step(self.official.LIBERO_DUMMY_ACTION)
        self.observations[slot] = policy_observation(self.official, obs, prompt)
        self.identities[slot] = {"task_id": task_id, "initial_state_index": index,
            "initial_state_sha256": state_hash, "seed": self.config["train_seed"], "task_description": prompt}
        self.steps[slot], self.done[slot] = 0, False

    def prepare(self):
        for slot in range(len(self.environments)):
            if self.done[slot]:
                self.reset(slot)
        return self.observations

    def execute(self, actions):
        for chunk in actions:
            check_chunk(chunk, 50, 20)
        start_steps = list(self.steps)
        # Offset 0 is the exact stale observation from which the student sampled.
        views = [list(self.observations)]
        valid = [0] * len(self.environments)
        success = [False] * len(self.environments)
        for offset in range(20):
            if offset in (5, 10, 15):
                views.append(list(self.observations))
            for slot, env in enumerate(self.environments):
                if self.done[slot]:
                    continue
                obs, _, done, _ = env.step(actions[slot, offset].tolist())
                self.steps[slot] += 1
                valid[slot] += 1
                success[slot] = bool(done)
                self.done[slot] = bool(done) or self.steps[slot] >= 520
                self.observations[slot] = policy_observation(self.official, obs, self.identities[slot]["task_description"])
        identities = [dict(identity, controlled_steps_at_start=start_steps[i],
            executed_steps=valid[i], episode_ended=self.done[i], task_success=success[i])
            for i, identity in enumerate(self.identities)]
        return views, valid, identities

    def close(self):
        for env in self.environments:
            if env is not None:
                env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--openpi-dir", default=os.environ.get("OPENPI_DIR"), required=not os.environ.get("OPENPI_DIR"))
    parser.add_argument("--stage", choices=["diagnostic", "train", "save", "baseline", "student", "status", "resume", "snapshot"], required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--rollouts-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--snapshot", type=int)
    parser.add_argument("--end-step", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    config = load_config(args.config)
    validate_training_config(config)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(args.host, args.port)

    def request(operation, payload=None, **control):
        return client.infer(dict(payload or {}, _flow_opsd=dict(operation=operation, **control)))

    environments = None
    try:
        if args.stage == "resume":
            if not args.checkpoint:
                raise ValueError("Resume requires --checkpoint")
            print(request("resume", checkpoint=args.checkpoint), flush=True)
            return
        if args.stage == "snapshot":
            if args.snapshot is None:
                raise ValueError("Select --snapshot 0, 100, 300 or 500")
            print(request("set_phase", phase="baseline" if args.snapshot == 0 else "step_" + str(args.snapshot)), flush=True)
            return
        if args.stage in ("baseline", "student"):
            print(request("set_phase", phase=args.stage), flush=True)
            return
        if args.stage in ("save", "status"):
            print(request(args.stage), flush=True)
            return
        if not args.rollouts_dir:
            raise ValueError("Training requires --rollouts-dir for replayable observations and actions")
        current_step = int(request("status")["step"])
        end_step = current_step + 1 if args.stage == "diagnostic" else (args.end_step or config["optimizer_steps"])
        if not current_step < end_step <= config["optimizer_steps"]:
            raise ValueError("Training endpoint must advance within the fixed update budget")
        rollouts = Path(args.rollouts_dir) / (args.stage if current_step == 0 else "train_{}_{}".format(current_step + 1, end_step))
        rollouts.mkdir(parents=True, exist_ok=False)
        official = import_official(args.openpi_dir)
        random.seed(config["train_seed"])
        np.random.seed(config["train_seed"])
        import torch
        torch.manual_seed(config["train_seed"])
        environments = TrainingEnvironments(official, config)
        diagnostic = args.stage == "diagnostic"
        for step in range(current_step, end_step):
            started = time.monotonic()
            observations = environments.prepare()
            response = request("rollout", {"observations": observations}, expected_step=step, diagnostic=diagnostic)
            actions = np.asarray(response["actions"])
            future, valid, identities = environments.execute(actions)
            # Raw arrays are kept on scratch, without costly per-step compression.
            arrays = {"student_actions": actions, "valid_steps": np.asarray(valid)}
            for block, batch in enumerate(future):
                for key in ["observation/image", "observation/wrist_image", "observation/state"]:
                    arrays["offset_{}_{}".format(5 * block, key.replace("/", "_"))] = np.stack([x[key] for x in batch])
            filename = rollouts / "step_{:03d}.npz".format(step + 1)
            if filename.exists():
                raise FileExistsError(str(filename))
            np.savez(filename, **arrays)
            append_record(Path(args.results_dir) / "rollout_inputs.jsonl", {
                "optimizer_step": step + 1, "diagnostic": diagnostic, "initial_states": identities,
                "action_chunk_sha256": array_hash(actions), "rollout_path": str(filename.resolve()),
                "flow_time_index": response["flow_time_index"]})
            result = request("update", {"future_observations": future},
                             rollout_token=response["rollout_token"], valid_steps=valid)
            record = {"optimizer_step": step + 1, "diagnostic": diagnostic, "initial_states": identities,
                      "action_chunk_sha256": array_hash(actions), "rollout_path": str(filename.resolve()),
                      "seconds": time.monotonic() - started, "training": result}
            append_record(Path(args.results_dir) / ("diagnostic_rollouts.jsonl" if diagnostic else "rollouts.jsonl"), record)
            logging.info("Completed optimizer update %s; loss %.6g", step + 1, result["loss"])
        if diagnostic:
            print(request("reset_after_diagnostic"), flush=True)
    finally:
        if environments is not None:
            environments.close()
        client._ws.close()


if __name__ == "__main__":
    main()
