"""One-GPU temporal OPSD backend; native inference and trainable-state isolation."""
import copy
import json
import logging
from pathlib import Path
import shutil
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from openpi.models import model as model_lib
from openpi.shared import nnx_utils

from .logging_utils import append_record, digest, file_digest, write_json
from .opsd_flow import sample_with_trace, velocity
from .opsd_protocol import align_teacher_latent, validate_training_config


class TemporalOPSD:
    def __init__(self, policy, metadata, config, results_dir, checkpoint_root, source_checkpoint):
        validate_training_config(config)
        self.policy, self.metadata, self.config = policy, metadata, config
        self.root, self.checkpoint_root = Path(results_dir), Path(checkpoint_root)
        if (self.root / "provenance/training_setup.json").exists():
            raise FileExistsError("Refusing to overwrite an OPSD run: " + str(self.root))
        self.source_checkpoint = Path(source_checkpoint)
        self.base_spec = copy.deepcopy(metadata["experiment_spec"])
        self.step, self.pending, self.phase = 0, None, "baseline"
        self.diagnostic_complete = False
        self.saved_checkpoint = None
        self.teacher_filter = nnx.All(nnx.Param, nnx.Any(
            nnx_utils.PathRegex(".*llm.*_1.*"),
            nnx_utils.PathRegex("(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.*")))
        graph, trainable, frozen = nnx.split(policy._model, self.teacher_filter, ...)
        self.graph, self.frozen = graph, frozen
        self.initial = jax.tree.map(lambda x: x.astype(jnp.float32), trainable)
        self.master, self.ema = self.initial, self.initial
        self.tx = optax.chain(optax.clip_by_global_norm(config["clip_gradient_norm"]),
            optax.adamw(config["learning_rate"], weight_decay=config["weight_decay"]))
        self.opt_state = self.tx.init(self.master)
        names = ["/".join(map(str, p)) for p in trainable.flat_state()]
        if any("/img/" in name or ("/llm/" in name and "_1" not in name) for name in names):
            raise RuntimeError("The frozen visual/language backbone entered the optimizer")
        if not names or not any("_1" in name for name in names):
            raise RuntimeError("No action-expert parameters selected")
        self.parameter_count = sum(x.size for x in jax.tree.leaves(trainable))
        self.sources = {name: file_digest(Path(__file__).parent / name)
                        for name in ["opsd_backend.py", "opsd_flow.py", "opsd_protocol.py"]}
        write_json(self.root / "provenance/training_setup.json", {
            "config": config, "sources": self.sources,
            "trainable_parameter_count": self.parameter_count, "trainable_paths": names,
            "base_experiment_spec": self.base_spec,
            "training_checkpoint_root": str(self.checkpoint_root.resolve())})

        def make_model(master, frozen):
            return nnx.merge(graph, jax.tree.map(lambda x: x.astype(jnp.bfloat16), master), frozen)

        def native(master, frozen, observation, key):
            return make_model(master, frozen).sample_actions(key, observation, num_steps=config["flow_steps"])

        def trace(master, frozen, observation, key):
            model = make_model(master, frozen)
            noise = jax.random.normal(key, (observation.state.shape[0], model.action_horizon, model.action_dim))
            return sample_with_trace(model, observation, noise, config["flow_steps"])

        def field(master, frozen, observation, latent, timestep):
            return velocity(make_model(master, frozen), observation, latent, timestep)

        def update(master, opt_state, frozen, observation, latent, timestep, targets, mask):
            latent, targets = jax.lax.stop_gradient(latent), jax.lax.stop_gradient(targets)

            def loss_fn(params):
                prediction = field(params, frozen, observation, latent, timestep)[:, :20, :7]
                squared = jnp.square(prediction.astype(jnp.float32) - targets.astype(jnp.float32))
                loss = 0.5 * jnp.sum(squared * mask[..., None]) / (jnp.sum(mask) * 7)
                per_offset = jnp.sum((squared * mask[..., None]).reshape((-1, 4, 5, 7)), axis=(0, 2, 3))
                per_offset /= jnp.maximum(1, jnp.sum(mask.reshape((-1, 4, 5)), axis=(0, 2)) * 7)
                return loss, (per_offset, prediction)

            (loss, (per_offset, prediction)), grads = jax.value_and_grad(loss_fn, has_aux=True)(master)
            updates, state = self.tx.update(grads, opt_state, master)
            return optax.apply_updates(master, updates), state, {
                "loss": loss, "grad_norm": optax.global_norm(grads),
                "update_norm": optax.global_norm(updates), "offset_mse": per_offset,
                "forward_prediction": prediction}

        self.native, self.trace, self.field, self.update = map(jax.jit, [native, trace, field, update])
        self.set_phase("baseline")

    def observations(self, items):
        inputs = [self.policy._input_transform(jax.tree.map(lambda x: x, item)) for item in items]
        batch = jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *inputs)
        return model_lib.Observation.from_dict(batch)

    def environment_actions(self, observation, actions):
        actions = np.asarray(actions)
        return np.stack([self.policy._output_transform({"state": np.asarray(observation.state[i]),
                         "actions": actions[i]})["actions"] for i in range(len(actions))])

    def set_phase(self, phase):
        if self.pending is not None:
            raise RuntimeError("Cannot switch evaluation phase with a pending training rollout")
        if phase == "student" and (self.step != 100 or self.saved_checkpoint is None):
            raise RuntimeError("Only the saved step-100 student is evaluated")
        if phase not in {"baseline", "student"}:
            raise ValueError("Unknown evaluation phase")
        self.phase = phase
        spec = copy.deepcopy(self.base_spec)
        spec["temporal_opsd"] = {
            "algorithm": self.config["algorithm"], "sources": self.sources,
            "phase": phase, "optimizer_step": 0 if phase == "baseline" else self.step,
            "checkpoint": None if phase == "baseline" else self.saved_checkpoint,
            "evaluation_sampler": "unchanged Pi0.sample_actions; dynamic parameter arguments"}
        self.metadata["experiment_spec"] = spec
        self.metadata["inference_fingerprint"] = digest(spec)
        write_json(self.root / "provenance" / ("server_" + phase + ".json"), self.metadata)
        return {"phase": phase, "inference_fingerprint": self.metadata["inference_fingerprint"]}

    def infer(self, observation):
        observation = dict(observation)
        control = observation.pop("_flow_opsd", None)
        if control is not None:
            operation = control["operation"]
            if operation == "rollout":
                return self.rollout(observation["observations"], control)
            if operation == "update":
                return self.learn(observation["future_observations"], control)
            if operation == "reset_after_diagnostic":
                return self.reset_after_diagnostic()
            if operation == "save":
                return self.save()
            if operation == "set_phase":
                return self.set_phase(control["phase"])
            if operation == "status":
                return {"step": self.step, "phase": self.phase, "diagnostic_complete": self.diagnostic_complete}
            raise ValueError("Unknown OPSD request")
        control = observation.pop("_frequency_vla", None)
        if control is None:
            raise ValueError("An instrumented evaluation or training client is required")
        inputs = self.observations([observation])
        rng = jax.random.fold_in(jax.random.key(int(control["episode_seed"])), int(control["call_index"]))
        _, rng = jax.random.split(rng)  # Exactly the native Policy.infer RNG protocol.
        params = self.initial if self.phase == "baseline" else self.master
        result = self.environment_actions(inputs, self.native(params, self.frozen, inputs, rng))[0]
        return {"actions": result, "inference_fingerprint": self.metadata["inference_fingerprint"]}

    def rollout(self, observations, control):
        if self.pending is not None or self.step >= self.config["optimizer_steps"]:
            raise RuntimeError("Pending rollout or exhausted optimizer budget")
        diagnostic = bool(control.get("diagnostic", False))
        if not diagnostic and not self.diagnostic_complete:
            raise RuntimeError("A successful diagnostic and rollback are required before training")
        if len(observations) != self.config["batch_size"] or int(control["expected_step"]) != self.step:
            raise ValueError("Training batch size or student version mismatch")
        inputs = self.observations(observations)
        key = jax.random.fold_in(jax.random.key(self.config["train_seed"]), self.step + (100000 if diagnostic else 0))
        sample_key, time_key, teacher_key = jax.random.split(key, 3)
        actions, trace, times, count = self.trace(self.master, self.frozen, inputs, sample_key)
        actions.block_until_ready()
        if int(count) != self.config["flow_steps"] or not np.isfinite(np.asarray(actions)).all():
            raise RuntimeError("Invalid native flow rollout")
        if diagnostic:
            self.diagnostic_observation = jax.tree.map(lambda x: x[:1], inputs)
            reference = self.native(self.master, self.frozen, inputs, sample_key)
            difference = float(jnp.max(jnp.abs(reference - actions)))
            if difference > 0.01:
                raise RuntimeError("Traced sampler differs from native inference: max abs=" + str(difference))
            self.trace_check = {"max_abs_difference": difference, "tolerance": 0.01,
                                "same_noise_key": True, "native_flow_steps": int(count)}
        index = int(jax.random.randint(time_key, (), 0, self.config["flow_steps"]))
        self.pending = {"observation": inputs, "latent": trace[index], "time": times[index],
                        "time_index": index, "teacher_key": teacher_key, "diagnostic": diagnostic,
                        "token": digest([self.step, diagnostic, index]), "started": time.monotonic()}
        return {"actions": self.environment_actions(inputs, actions), "rollout_token": self.pending["token"],
                "optimizer_step": self.step, "flow_time_index": index}

    def learn(self, future_observations, control):
        pending = self.pending
        if pending is None or control["rollout_token"] != pending["token"]:
            raise RuntimeError("Missing or stale on-policy rollout")
        valid = np.asarray(control["valid_steps"], dtype=np.int32)
        if valid.shape != (self.config["batch_size"],) or np.any((valid < 1) | (valid > 20)):
            raise ValueError("Invalid executed-action counts")
        if len(future_observations) != 4 or any(len(x) != len(valid) for x in future_observations):
            raise ValueError("Require four aligned fresh-observation batches")
        student_z = pending["latent"]
        timestep = jnp.broadcast_to(pending["time"], len(valid))
        targets = []
        for block, observations in enumerate(future_observations):
            inputs = self.observations(observations)
            offset = block * 5
            if offset == 0:
                if not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                           zip(jax.tree.leaves(inputs), jax.tree.leaves(pending["observation"]))):
                    raise ValueError("Teacher offset-0 input differs from the student's rollout input")
                teacher_z = student_z
            else:
                _, auxiliary, _, _ = self.trace(self.ema, self.frozen, inputs,
                                                jax.random.fold_in(pending["teacher_key"], offset))
                teacher_z = align_teacher_latent(student_z, auxiliary[pending["time_index"]], offset, jnp)
            v = self.field(self.ema, self.frozen, inputs, teacher_z, timestep)
            targets.append(v[:, :5, :7])
        target = jax.lax.stop_gradient(jnp.concatenate(targets, axis=1))
        mask = jnp.arange(20)[None, :] < jnp.asarray(valid)[:, None]
        params, optimizer, info = self.update(self.master, self.opt_state, self.frozen,
            pending["observation"], student_z, timestep, target, mask)
        info = jax.device_get(info)
        if not all(np.isfinite(np.asarray(x)).all() for x in info.values()):
            raise RuntimeError("Non-finite training loss or gradient; aborting before state update")
        gradient_program_prediction = info.pop("forward_prediction")
        if float(info["grad_norm"]) == 0 or float(info["update_norm"]) == 0:
            raise RuntimeError("Zero distillation gradient/update; verify the teacher information advantage")
        if pending["diagnostic"]:
            # Compare like with like. XLA can compile a different bfloat16 primal
            # when producing backward residuals; that is a separate numeric check.
            forward = np.asarray(self.field(self.master, self.frozen,
                pending["observation"], student_z, timestep), dtype=np.float32)[:, :20, :7]
            target_array, mask_array = np.asarray(target, dtype=np.float32), np.asarray(mask)
            same_program_mse = float(np.mean(np.square(forward[:, :5] - target_array[:, :5])))
            backward_primal_mse = float(np.mean(np.square(forward - gradient_program_prediction.astype(np.float32))))
            feedback_mse = float(np.sum(np.square(forward[:, 5:] - target_array[:, 5:]) * mask_array[:, 5:, None])
                                 / max(1, np.sum(mask_array[:, 5:]) * 7))
            metrics = {"same_program_offset0_mse": same_program_mse,
                "forward_vs_backward_primal_mse": backward_primal_mse,
                "fresh_feedback_mse": feedback_mse,
                "required_signal_to_numeric_ratio": 5,
                "signal_to_numeric_ratio": feedback_mse / max(backward_primal_mse, 1e-20),
                "gradient_program_offset_mse": np.asarray(info["offset_mse"]).tolist(),
                "flow_time_index": pending["time_index"], "trace_check": self.trace_check}
            write_json(self.root / "provenance/numeric_diagnostic.json", metrics)
            logging.info("OPSD numeric diagnostic: %s", metrics)
            if same_program_mse > 1e-8:
                raise RuntimeError("Identical teacher/student inputs differ in the same compiled forward function")
            if feedback_mse <= max(1e-10, 5 * backward_primal_mse):
                raise RuntimeError("Fresh-observation supervision does not exceed compiled numeric noise sufficiently")
        self.master, self.opt_state = params, optimizer
        self.ema = jax.tree.map(lambda old, new: self.config["ema_decay"] * old +
                               (1 - self.config["ema_decay"]) * new, self.ema, self.master)
        self.step += 1
        record = {"optimizer_step": self.step, "diagnostic": pending["diagnostic"],
                  "flow_time_index": pending["time_index"], "flow_time": float(pending["time"]),
                  "valid_controlled_steps": valid.tolist(), "student_behavior_version": self.step - 1,
                  "seconds_including_environment": time.monotonic() - pending["started"],
                  **{k: np.asarray(v).tolist() for k, v in info.items()}}
        append_record(self.root / ("diagnostic_training.jsonl" if pending["diagnostic"] else "training.jsonl"), record)
        logging.info("OPSD step=%s diagnostic=%s loss=%.6g grad_norm=%.6g", self.step,
                     pending["diagnostic"], record["loss"], record["grad_norm"])
        self.pending = None
        return record

    def reset_after_diagnostic(self):
        if self.step != 1 or self.pending is not None or self.diagnostic_complete:
            raise RuntimeError("Only the single diagnostic update can be rolled back")
        path = (self.checkpoint_root / "diagnostic_trainable").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(path, {"params": self.master.to_pure_dict()})
        restored = model_lib.restore_params(path, dtype=jnp.float32)
        expected = self.master.to_pure_dict()
        equal = jax.tree.all(jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), expected, restored))
        if not equal:
            raise RuntimeError("Diagnostic parameter save/reload failed")
        self.master, self.ema = self.initial, self.initial
        self.opt_state = self.tx.init(self.master)
        self.step = 0
        self.diagnostic_complete = True
        # Compile the batch-one native evaluator before four clients connect.
        probe = self.native(self.initial, self.frozen, self.diagnostic_observation, jax.random.key(123))
        if not np.isfinite(np.asarray(probe)).all():
            raise RuntimeError("Invalid restored baseline inference")
        result = {"passed": True, "checkpoint_roundtrip_equal": True,
                  "diagnostic_updates_rolled_back": 1, "optimizer_steps": 0,
                  "trace_check": self.trace_check, "frozen_backbone_outside_optimizer": True}
        write_json(self.root / "provenance/diagnostic.json", result)
        return result

    def save(self):
        if self.step != 100 or self.pending is not None:
            raise RuntimeError("Only the completed 100-step student can be exported")
        path = (self.checkpoint_root / "step_100").resolve()
        path.mkdir(parents=True, exist_ok=False)
        params = nnx.merge(self.graph, self.master, self.frozen)
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(path / "params", {"params": nnx.state(params).to_pure_dict()})
            checkpointer.save(path / "training_state", {"ema": self.ema,
                              "optimizer": self.opt_state, "step": np.asarray(self.step)})
        shutil.copytree(self.source_checkpoint / "assets", path / "assets")
        restored = model_lib.restore_params(path / "params", dtype=jnp.bfloat16)
        loaded_state = nnx.state(self.policy._model)
        loaded_state.replace_by_pure_dict(restored)
        loaded = nnx.merge(self.graph, loaded_state)
        _, loaded_trainable, loaded_frozen = nnx.split(loaded, self.teacher_filter, ...)
        loaded_trainable = jax.tree.map(lambda x: x.astype(jnp.float32), loaded_trainable)
        key = jax.random.key(123)
        expected = self.native(self.master, self.frozen, self.diagnostic_observation, key)
        actual = self.native(loaded_trainable, loaded_frozen, self.diagnostic_observation, key)
        reload_difference = float(jnp.max(jnp.abs(expected - actual)))
        if reload_difference != 0:
            raise RuntimeError("Exported checkpoint inference round-trip differs: " + str(reload_difference))
        manifest = {"step": self.step, "config": self.config, "sources": self.sources,
                    "base_checkpoint": self.base_spec["checkpoint"],
                    "base_object_manifest_sha256": self.base_spec["checkpoint_object_manifest_sha256"],
                    "reloaded_native_inference_max_abs_difference": reload_difference,
                    "files": {str(p.relative_to(path)): file_digest(p) for p in sorted(path.rglob("*")) if p.is_file()}}
        write_json(path / "training_manifest.json", manifest)
        self.saved_checkpoint = {"path": str(path), "manifest_sha256": digest(manifest)}
        write_json(self.root / "provenance/step_100.json", self.saved_checkpoint)
        # Native WebSocket transport mutates replies to append server timing.
        return dict(self.saved_checkpoint)
