"""Small CPU integration test against the pinned native sampler and NNX/Orbax APIs.

Run with the server Python; fixtures and checkpoints exist only in a temp directory.
No VLA checkpoint is loaded and no benchmark result is created.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

from pathlib import Path
import tempfile

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import Pi0
from frequency_vla.config import load_config
from frequency_vla.opsd_backend import TemporalOPSD


class TinyLLM(nnx.Module):
    def __init__(self):
        self.scale_1 = nnx.Param(jnp.asarray(.6))
        self.prefix_scale = nnx.Param(jnp.asarray(.4))

    def __call__(self, inputs, *, mask, positions, kv_cache=None, adarms_cond=None):
        prefix, suffix = inputs
        context = prefix.mean(axis=1, keepdims=True) * self.prefix_scale.value if prefix is not None else kv_cache
        output = None if suffix is None else self.scale_1.value * suffix + context
        return (prefix, output), context


class TinyModel(nnx.Module):
    action_horizon, action_dim = 50, 32
    sample_actions = Pi0.sample_actions

    def __init__(self):
        self.PaliGemma = nnx.Dict(llm=TinyLLM(), img=nnx.Param(jnp.asarray(.7)))
        self.action_out_proj = nnx.Linear(32, 32, rngs=nnx.Rngs(1))

    def embed_prefix(self, observation):
        b = observation.state.shape[0]
        return observation.state[:, None, :], jnp.ones((b, 1), bool), jnp.zeros(1, bool)

    def embed_suffix(self, observation, latent, timestep):
        b = observation.state.shape[0]
        ar = jnp.arange(50) == 0
        return latent + .01 * timestep[:, None, None], jnp.ones((b, 50), bool), ar, None


class TinyPolicy:
    def __init__(self):
        self._model = TinyModel()

    @staticmethod
    def _input_transform(item):
        return {"state": np.pad(item["observation/state"], (0, 24)),
                "image": {k: np.zeros((224, 224, 3), dtype=np.uint8)
                          for k in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]},
                "image_mask": {k: True for k in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]}}

    @staticmethod
    def _output_transform(item):
        return {"actions": item["actions"][..., :7]}


def observation(value):
    return {"observation/state": np.full(8, value, dtype=np.float32)}


def main():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/opsd_h20_100.yaml")
    with tempfile.TemporaryDirectory(prefix="opsd-api-test-") as directory:
        root = Path(directory)
        metadata = {"experiment_spec": {"checkpoint": "SYNTHETIC_TEST_ONLY",
                    "checkpoint_object_manifest_sha256": "SYNTHETIC_TEST_ONLY"}}
        trainer = TemporalOPSD(TinyPolicy(), metadata, config, root / "results", root / "checkpoints", root)
        initial = jax.tree.map(lambda x: np.array(x), trainer.initial)
        frozen = jax.tree.map(lambda x: np.array(x), trainer.frozen)
        batch = [observation(i * .1) for i in range(4)]
        result = trainer.rollout(batch, {"expected_step": 0, "diagnostic": True})
        assert result["actions"].shape == (4, 50, 7)
        assert trainer.trace_check["max_abs_difference"] <= .01
        views = [[observation(i * .1 + block * .2) for i in range(4)] for block in range(4)]
        info = trainer.learn(views, {"rollout_token": result["rollout_token"], "valid_steps": [20, 17, 5, 1]})
        assert info["loss"] > 0 and info["grad_norm"] > 0 and info["update_norm"] > 0
        assert trainer.step == 1
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(frozen), jax.tree.leaves(trainer.frozen)))
        assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(trainer.master)))
        updated_fixture = trainer.master
        restored = trainer.reset_after_diagnostic()
        assert restored["passed"] and trainer.step == 0 and trainer.diagnostic_complete
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(trainer.master)))
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(trainer.ema)))
        # Exercise the export path with a tiny synthetic snapshot, not a research run.
        (root / "assets").mkdir()
        (root / "assets/test-only.txt").write_text("SYNTHETIC TEST FIXTURE")
        trainer.master, trainer.step = updated_fixture, 100
        saved = trainer.save()
        assert Path(saved["path"]).joinpath("training_manifest.json").is_file()
        assert trainer.set_phase("student")["phase"] == "student"
        print("PASS: native trace, nonzero detached-target update, frozen backbone, exact rollback, and complete checkpoint inference round-trip")
        # A second tiny, synthetic-only experiment checks a real optimizer resume:
        # one update, save, restore all state, then equal next updates in both copies.
        small_config = dict(config, optimizer_steps=3, checkpoint_steps=[1, 2, 3])
        def make_trainer(name):
            value = TemporalOPSD(TinyPolicy(), {"experiment_spec": dict(metadata["experiment_spec"])},
                small_config, root / name / "results", root / name / "checkpoints", root)
            rollout = value.rollout(batch, {"expected_step": 0, "diagnostic": True})
            value.learn(views, {"rollout_token": rollout["rollout_token"], "valid_steps": [20, 17, 5, 1]})
            value.reset_after_diagnostic()
            return value

        def advance(value):
            rollout = value.rollout(batch, {"expected_step": value.step})
            return value.learn(views, {"rollout_token": rollout["rollout_token"], "valid_steps": [20, 17, 5, 1]})

        def equal_trees(left, right):
            assert jax.tree.structure(left) == jax.tree.structure(right)
            assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)))

        uninterrupted = make_trainer("continuation-original")
        advance(uninterrupted)
        saved_one = uninterrupted.save()
        resumed = make_trainer("continuation-restored")
        resume_info = resumed.resume(saved_one["path"])
        assert resume_info["adam_update_counters"] == [1] and resumed.step == 1
        for key in ["master", "ema", "opt_state", "frozen"]:
            equal_trees(getattr(uninterrupted, key), getattr(resumed, key))
        frozen_one = jax.tree.map(lambda x: np.array(x), resumed.snapshots[1]["params"])
        advance(uninterrupted)
        advance(resumed)
        for key in ["master", "ema", "opt_state"]:
            equal_trees(getattr(uninterrupted, key), getattr(resumed, key))
        resumed.save()
        resumed.set_phase("step_1")
        equal_trees(frozen_one, resumed.evaluation_parameters)
        resumed.set_phase("step_2")
        equal_trees(resumed.master, resumed.evaluation_parameters)
        print("PASS: FP32/EMA/Adam resume, identical next update, and frozen milestone evaluation")


if __name__ == "__main__":
    main()
