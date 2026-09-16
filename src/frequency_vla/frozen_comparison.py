"""Read-only original/trained snapshot comparison through one native sampler."""
import copy
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils

from .logging_utils import digest, file_digest, write_json


class FrozenComparison:
    """No optimizer, teacher updates, rollout collection or training operations."""
    def __init__(self, policy, metadata, checkpoint, results_dir, source_checkpoint):
        self.policy, self.metadata = policy, metadata
        self.root = Path(results_dir)
        self.base_spec = copy.deepcopy(metadata["experiment_spec"])
        path = Path(checkpoint).resolve()
        manifest = json.loads((path / "training_manifest.json").read_text())
        if manifest["base_object_manifest_sha256"] != self.base_spec["checkpoint_object_manifest_sha256"]:
            raise ValueError("Comparison checkpoint has a different original checkpoint")
        for key in ["prediction_horizon", "flow_steps"]:
            if manifest["config"][key] != self.base_spec[key]:
                raise ValueError("Comparison changed inference settings: " + key)
        for name, expected in manifest["files"].items():
            if name.startswith(("params/", "assets/")) and file_digest(path / name) != expected:
                raise ValueError("Comparison checkpoint checksum mismatch: " + name)
            if name.startswith("assets/") and file_digest(Path(source_checkpoint) / name) != expected:
                raise ValueError("Comparison changed the official normalization assets")
        expert_filter = nnx.All(nnx.Param, nnx.Any(nnx_utils.PathRegex(".*llm.*_1.*"),
            nnx_utils.PathRegex("(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.*")))
        graph, original, frozen = nnx.split(policy._model, expert_filter, ...)
        restored = model_lib.restore_params(path / "params")
        loaded_state = nnx.state(policy._model)
        loaded_state.replace_by_pure_dict(restored)
        _, trained, loaded_frozen = nnx.split(nnx.merge(graph, loaded_state), expert_filter, ...)
        if not jax.tree.all(jax.tree.map(lambda a,b: bool(jnp.array_equal(a,b)), frozen, loaded_frozen)):
            raise ValueError("The trained checkpoint changed the supposedly frozen backbone")
        self.frozen = frozen
        self.step = int(manifest["step"])
        self.snapshots = {0: jax.tree.map(lambda x: x.astype(jnp.float32), original),
                          self.step: jax.tree.map(lambda x: x.astype(jnp.float32), trained)}
        self.checkpoint = dict(path=str(path), manifest_sha256=digest(manifest))
        flow_steps = self.base_spec["flow_steps"]

        def native(params, fixed, observation, key):
            model = nnx.merge(graph, jax.tree.map(lambda x: x.astype(jnp.bfloat16), params), fixed)
            return model.sample_actions(key, observation, num_steps=flow_steps)

        self.native = jax.jit(native)
        write_json(self.root / "provenance/frozen_comparison.json", dict(
            original_checkpoint=self.base_spec["checkpoint"], trained_checkpoint=self.checkpoint,
            trained_step=self.step, frozen_backbone_equal=True, normalization_assets_equal=True,
            inference_parameter_checksums_verified=True, training_operations_available=False,
            native_sampler="unchanged Pi0.sample_actions with dynamic snapshot parameters",
            source_sha256=file_digest(__file__)))
        write_json(self.root / "provenance/exported_training_manifest.json", manifest)
        self.select(0)

    def select(self, step):
        if step not in self.snapshots:
            raise ValueError("Unknown frozen parameter snapshot")
        self.active_step = step
        spec = copy.deepcopy(self.base_spec)
        spec["frozen_comparison"] = dict(optimizer_step=step,
            checkpoint=self.checkpoint if step else None,
            source_sha256=file_digest(__file__), sampler="unchanged Pi0.sample_actions")
        self.metadata["experiment_spec"] = spec
        self.metadata["inference_fingerprint"] = digest(spec)
        write_json(self.root / "provenance" / ("server_step_" + str(step) + ".json"), self.metadata)
        return dict(step=step, inference_fingerprint=self.metadata["inference_fingerprint"])

    def infer(self, observation):
        observation = dict(observation)
        selection = observation.pop("_frozen_comparison", None)
        if selection is not None:
            if selection.get("operation") == "select":
                return self.select(int(selection["step"]))
            if selection.get("operation") == "status":
                return dict(step=self.active_step, snapshots=sorted(self.snapshots),
                            training_operations_available=False, gpu_memory=jax.devices()[0].memory_stats() or {})
            raise ValueError("Frozen comparison supports only snapshot selection and status")
        control = observation.pop("_frequency_vla", None)
        if control is None:
            raise ValueError("An instrumented evaluation client is required")
        inputs = self.policy._input_transform(observation)
        inputs = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs))
        key = jax.random.fold_in(jax.random.key(int(control["episode_seed"])), int(control["call_index"]))
        _, key = jax.random.split(key)
        result = self.native(self.snapshots[self.active_step], self.frozen, inputs, key)
        outputs = self.policy._output_transform(dict(state=np.asarray(inputs.state[0]), actions=np.asarray(result[0])))
        return dict(actions=outputs["actions"], inference_fingerprint=self.metadata["inference_fingerprint"])
