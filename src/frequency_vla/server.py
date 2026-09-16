"""Official policy loader with provenance, paired RNG, and explicit fixed-P extensions."""
from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import socket
import uuid
from unittest.mock import patch

from .config import load_config, prediction_horizon, upstream_spec
from .logging_utils import digest, file_digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openpi-dir", default=os.environ.get("OPENPI_DIR"), required=not os.environ.get("OPENPI_DIR"))
    p.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR"), required=not os.environ.get("CHECKPOINT_DIR"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--manifest-out", required=True)
    p.add_argument("--opsd-config")
    p.add_argument("--opsd-results-dir")
    p.add_argument("--opsd-checkpoint-root")
    p.add_argument("--trained-checkpoint", help="Separately exported OPSD checkpoint, with its training manifest")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
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
    active_checkpoint = checkpoint
    trained_manifest = None
    if args.trained_checkpoint:
        if args.opsd_config:
            raise ValueError("The initial 100-step training run must start from the official checkpoint")
        active_checkpoint = Path(args.trained_checkpoint).resolve()
        trained_manifest = json.loads((active_checkpoint / "training_manifest.json").read_text())
        if trained_manifest["base_object_manifest_sha256"] != digest(download_manifest):
            raise ValueError("Trained checkpoint has a different source checkpoint")
        for key in ["prediction_horizon", "flow_steps"]:
            if trained_manifest["config"][key] != (prediction_horizon(spec) if key == "prediction_horizon" else spec[key]):
                raise ValueError("Trained checkpoint inference setting mismatch: " + key)
        for name, expected in trained_manifest["files"].items():
            if name.startswith(("params/", "assets/")) and file_digest(active_checkpoint / name) != expected:
                raise ValueError("Trained checkpoint checksum mismatch: " + name)
    sys.path.insert(0, str(Path(args.openpi_dir).resolve()))
    import jax
    from jax._src import lib as jax_lib
    from openpi.training import config as official_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    import websockets.asyncio.server as websocket_server
    from scripts import serve_policy

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("A GPU allocation is required; refusing CPU inference on a login node")
    native_config = official_config.get_config(config["training_config"])
    if native_config.model.action_horizon != spec["native_prediction_horizon"]:
        raise ValueError("Runtime model configuration differs from source preflight")
    effective_config = native_config
    if "prediction_horizon" in config:
        # Same official loader, weights, transforms and sampler. Only the static
        # sequence length changes; never mutate the upstream checkout/config.
        from openpi.policies import policy_config
        effective_config = dataclasses.replace(native_config,
            model=dataclasses.replace(native_config.model, action_horizon=prediction_horizon(spec)))
        policy = policy_config.create_trained_policy(effective_config, str(active_checkpoint))
    else:
        policy = serve_policy.create_policy(serve_policy.Args(
            env=serve_policy.EnvMode.LIBERO,
            policy=serve_policy.Checkpoint(config=config["training_config"], dir=str(active_checkpoint)),
        ))
    if policy._model.action_horizon != prediction_horizon(spec) or policy._sample_kwargs:
        raise ValueError("Configured prediction horizon or sampling defaults changed")
    # Compile before opening any evaluator connections. Otherwise a long first
    # inference blocks the server event loop and can trigger keepalive timeouts.
    import numpy as np
    from .evaluator import check_chunk
    logging.info("Compiling one synthetic warmup inference before accepting connections")
    warmup_started = time.monotonic()
    warmup = policy.infer({
        "observation/image": np.zeros((config["resize_size"], config["resize_size"], 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((config["resize_size"], config["resize_size"], 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float64), "prompt": "warmup",
    })
    check_chunk(warmup["actions"], prediction_horizon(spec), prediction_horizon(spec))
    warmup_seconds = time.monotonic() - warmup_started
    logging.info("Warmup complete: P=%s in %.2f seconds", prediction_horizon(spec), warmup_seconds)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    core_packages = {k: v for k, v in packages.items() if k.lower() in {
        "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "numpy", "pillow",
        "opencv-python", "sentencepiece", "orbax-checkpoint", "tensorstore", "ml-dtypes"}}
    spec.update({"checkpoint_object_manifest_sha256": digest(download_manifest),
                 "model_config": dataclasses.asdict(effective_config.model),
                 "sample_kwargs": {}, "backend": "jax", "dtype": "bfloat16",
                 "rng_protocol": "fold_in(episode_key,call_index); native Policy.infer split; v1",
                 "startup_warmup_protocol": "one zero observation before listening; all episode calls reseeded",
                 "websocket_ping_interval": None,
                 "resize_size": config["resize_size"], "num_steps_wait": config["num_steps_wait"],
                 "python_version": platform.python_version(), "inference_packages": core_packages,
                 "ptxas_version": subprocess.check_output([str(Path(jax_lib.cuda_path) / "bin/ptxas"), "--version"], text=True).strip(),
                 "jax_cuda_root": str(jax_lib.cuda_path),
                 "xla_flags": os.environ.get("XLA_FLAGS", "")})
    if trained_manifest is not None:
        spec.update(trained_checkpoint_manifest_sha256=digest(trained_manifest),
                    trained_optimizer_step=trained_manifest["step"],
                    trained_checkpoint_local_path=str(active_checkpoint))
    fingerprint = digest(spec)
    metadata = {"experiment_spec": spec, "inference_fingerprint": fingerprint,
                "server_instance_id": str(uuid.uuid4()), "hostname": socket.gethostname(),
                "checkpoint_local_path": str(active_checkpoint),
                "warmup": {"seconds": warmup_seconds, "synthetic": True, "benchmark_episode": False},
                "server_packages": packages,
                "devices": [{"device": str(d), "kind": d.device_kind} for d in jax.devices()]}
    trainer = None
    if args.opsd_config:
        if not args.opsd_results_dir or not args.opsd_checkpoint_root:
            raise ValueError("OPSD requires separate results and checkpoint directories")
        from .opsd_backend import TemporalOPSD
        trainer = TemporalOPSD(policy, metadata, load_config(args.opsd_config),
                               args.opsd_results_dir, args.opsd_checkpoint_root, checkpoint)
    write_json(args.manifest_out, metadata)

    class PairedPolicy:
        def infer(self, observation):
            if trainer is not None:
                return trainer.infer(observation)
            observation = dict(observation)
            control = observation.pop("_frequency_vla", None)
            if control is None:
                raise ValueError("Use the instrumented evaluator so inference RNG and provenance are recorded")
            # Only RNG changes between calls; P is fixed for this entire server.
            policy._rng = jax.random.fold_in(jax.random.key(int(control["episode_seed"])), int(control["call_index"]))
            result = policy.infer(observation)
            result["inference_fingerprint"] = fingerprint
            return result

    # Keep the official handler and serialization. Physics initialization and
    # blocking inference need not satisfy a 20-second heartbeat deadline.
    serve_without_heartbeat = functools.partial(websocket_server.serve, ping_interval=None)
    with patch.object(websocket_server, "serve", serve_without_heartbeat):
        WebsocketPolicyServer(PairedPolicy(), host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
