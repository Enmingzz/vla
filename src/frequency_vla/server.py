"""Official policy construction/server with provenance and paired inference RNG only."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys

from .config import load_config, upstream_spec
from .logging_utils import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openpi-dir", default=os.environ.get("OPENPI_DIR"), required=not os.environ.get("OPENPI_DIR"))
    p.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR"), required=not os.environ.get("CHECKPOINT_DIR"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--manifest-out", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    config = load_config()
    spec = upstream_spec(args.openpi_dir, config)
    checkpoint = Path(args.checkpoint_dir).resolve()
    download_manifest = json.loads((checkpoint / "download_manifest.json").read_text())
    if download_manifest["source"] != config["checkpoint"]:
        raise ValueError("Checkpoint provenance does not match the requested official checkpoint")
    if (checkpoint / "model.safetensors").exists():
        raise ValueError("This experiment uses the unconverted official JAX checkpoint")
    for entry in download_manifest["objects"]:
        local = checkpoint / entry["name"].split("checkpoints/pi05_libero/", 1)[1]
        if local.stat().st_size != int(entry["size"]):
            raise ValueError("Checkpoint file size changed: " + str(local))
    sys.path.insert(0, str(Path(args.openpi_dir).resolve()))
    import jax
    from jax._src import lib as jax_lib
    from openpi.training import config as official_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from scripts import serve_policy

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("A GPU allocation is required; refusing CPU inference on a login node")
    native_config = official_config.get_config(config["training_config"])
    if native_config.model.action_horizon != spec["native_prediction_horizon"]:
        raise ValueError("Runtime model configuration differs from source preflight")
    # Same factory and no sample_kwargs overrides as scripts/serve_policy.py.
    policy = serve_policy.create_policy(serve_policy.Args(
        env=serve_policy.EnvMode.LIBERO,
        policy=serve_policy.Checkpoint(config=config["training_config"], dir=str(checkpoint)),
    ))
    if policy._model.action_horizon != spec["native_prediction_horizon"] or policy._sample_kwargs:
        raise ValueError("Native prediction horizon or sampling defaults changed")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    core_packages = {k: v for k, v in packages.items() if k.lower() in {
        "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "numpy", "pillow",
        "opencv-python", "sentencepiece", "orbax-checkpoint", "tensorstore", "ml-dtypes"}}
    spec.update({"checkpoint_object_manifest_sha256": digest(download_manifest),
                 "model_config": dataclasses.asdict(native_config.model),
                 "sample_kwargs": {}, "backend": "jax", "dtype": "bfloat16",
                 "rng_protocol": "fold_in(episode_key,call_index); native Policy.infer split; v1",
                 "resize_size": config["resize_size"], "num_steps_wait": config["num_steps_wait"],
                 "python_version": platform.python_version(), "inference_packages": core_packages,
                 "ptxas_version": subprocess.check_output([str(Path(jax_lib.cuda_path) / "bin/ptxas"), "--version"], text=True).strip(),
                 "jax_cuda_root": str(jax_lib.cuda_path),
                 "xla_flags": os.environ.get("XLA_FLAGS", "")})
    fingerprint = digest(spec)
    metadata = {"experiment_spec": spec, "inference_fingerprint": fingerprint,
                "checkpoint_local_path": str(checkpoint),
                "server_packages": packages,
                "devices": [{"device": str(d), "kind": d.device_kind} for d in jax.devices()]}
    write_json(args.manifest_out, metadata)

    class PairedPolicy:
        def infer(self, observation):
            observation = dict(observation)
            control = observation.pop("_frequency_vla", None)
            if control is None:
                raise ValueError("Use the instrumented evaluator so inference RNG and provenance are recorded")
            # Only the RNG key changes. Weights, preprocessing, sampler, and P are untouched.
            policy._rng = jax.random.fold_in(jax.random.key(int(control["episode_seed"])), int(control["call_index"]))
            result = policy.infer(observation)
            result["inference_fingerprint"] = fingerprint
            return result

    WebsocketPolicyServer(PairedPolicy(), host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
