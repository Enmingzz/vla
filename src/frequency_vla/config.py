from __future__ import annotations

import ast
import io
import os
from pathlib import Path
import tokenize

import yaml

from .logging_utils import file_digest, git_commit


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "frequency_sweep.yaml"


def load_config(path=None):
    path = path or os.environ.get("FREQUENCY_CONFIG") or CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def upstream_spec(openpi_dir, config):
    """Inspect actual upstream source without importing JAX or modifying a config."""
    root = Path(openpi_dir)
    commit = git_commit(root)
    if commit != config["openpi_commit"]:
        raise ValueError("OpenPI commit differs from the pinned, inspected commit: " + commit)
    libero_commit = git_commit(root / "third_party/libero")
    if libero_commit != config["libero_commit"]:
        raise ValueError("LIBERO submodule commit differs from the pinned protocol")
    # Refuse tracked edits; untracked virtual environments are harmless.
    for repo in [root, root / "third_party/libero"]:
        import subprocess
        changes = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
        if changes.strip():
            raise ValueError("Upstream checkout has tracked modifications: " + changes)
    # Simulator Python is 3.8; training config contains Python 3.10 match syntax.
    # Tokenize out constructor expressions so we never import/parse unrelated code.
    source = (root / "src/openpi/training/config.py").read_text()
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    constructors = []
    for i, token in enumerate(tokens[:-1]):
        if token.type != tokenize.NAME or token.string != "TrainConfig" or tokens[i + 1].string != "(":
            continue
        depth = 0
        for end in tokens[i + 1:]:
            if end.type == tokenize.OP and end.string == "(":
                depth += 1
            elif end.type == tokenize.OP and end.string == ")":
                depth -= 1
                if depth == 0:
                    start_offset = offsets[token.start[0] - 1] + token.start[1]
                    end_offset = offsets[end.end[0] - 1] + end.end[1]
                    constructors.append(ast.parse(source[start_offset:end_offset], mode="eval").body)
                    break
    selected = []
    for node in constructors:
        if isinstance(node, ast.Call):
            kws = {kw.arg: kw.value for kw in node.keywords}
            name = kws.get("name")
            if isinstance(name, ast.Constant) and name.value == config["training_config"]:
                selected.append(kws)
    if len(selected) != 1:
        raise ValueError("Cannot uniquely identify the official training config")
    model = selected[0]["model"]
    model_kws = {kw.arg: ast.literal_eval(kw.value) for kw in model.keywords}
    horizon = model_kws.get("action_horizon")
    if not isinstance(horizon, int):
        raise ValueError("Native action horizon is not explicit; inspect upstream before proceeding")
    model_tree = ast.parse((root / "src/openpi/models/pi0.py").read_text())
    sample = next(n for n in ast.walk(model_tree) if isinstance(n, ast.FunctionDef) and n.name == "sample_actions")
    flow_steps = dict((k.arg, ast.literal_eval(v)) for k, v in zip(sample.args.kwonlyargs, sample.args.kw_defaults))["num_steps"]
    if flow_steps != config["flow_steps"]:
        raise ValueError("Upstream default flow steps changed")
    files = ["examples/libero/main.py", "examples/libero/README.md", "scripts/serve_policy.py",
             "src/openpi/training/config.py", "src/openpi/models/pi0.py", "uv.lock"]
    spec = {"openpi_commit": commit, "libero_commit": libero_commit,
            "native_prediction_horizon": horizon, "flow_steps": flow_steps,
            "training_config": config["training_config"], "checkpoint": config["checkpoint"],
            "source_sha256": {p: file_digest(root / p) for p in files}}
    if "prediction_horizon" in config:
        requested = config["prediction_horizon"]
        if type(requested) is not int or requested < 1:
            raise ValueError("prediction_horizon must be a positive integer")
        spec.update(prediction_horizon=requested, protocol="fixed_prediction_horizon_extension")
    return spec


def prediction_horizon(spec):
    return spec.get("prediction_horizon", spec["native_prediction_horizon"])


def validate_horizons(horizons, prediction_horizon):
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Horizons must be positive integers")
    if len(horizons) != len(set(horizons)):
        raise ValueError("Duplicate horizons would duplicate observations")
    invalid = [h for h in horizons if h > prediction_horizon]
    if invalid:
        raise ValueError(
            "PROTOCOL BLOCKED: configured prediction horizon P={}. "
            "Requested H={} exceeds its action chunk. No padding, action repetition, hidden "
            "policy calls, or automatic prediction-horizon override is allowed. "
            "Select supported horizons or explicitly select an extension config.".format(prediction_horizon, invalid)
        )


def episode_seed(seed, suite, task_id, episode_index):
    from .logging_utils import digest
    # Independent of H, run mode, prior episode length, and task sharding.
    return int(digest([int(seed), suite, int(task_id), int(episode_index)])[:8], 16)
