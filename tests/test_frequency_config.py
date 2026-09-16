"""Synthetic fixtures are confined to pytest temp directories; never research results."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from frequency_vla import evaluator
from frequency_vla.analysis import exact_mcnemar, holm, paired_interval, validate_records, wilson
from frequency_vla.config import episode_seed, load_config, prediction_horizon, upstream_spec, validate_horizons
from frequency_vla.logging_utils import digest


def test_full_requested_sweep_is_preserved_and_fails_for_native_ten():
    assert load_config()["horizons"] == [5, 10, 20, 30, 40, 50]
    with pytest.raises(ValueError, match="PROTOCOL BLOCKED"):
        validate_horizons(load_config()["horizons"], 10)
    validate_horizons([5, 10], 10)


@pytest.mark.parametrize("horizons", [[0], [-1], [5, 5], [True], []])
def test_invalid_horizons(horizons):
    with pytest.raises(ValueError):
        validate_horizons(horizons, 10)


def test_episode_rng_does_not_depend_on_h_or_run_history():
    assert episode_seed(7, "libero_10", 3, 4) == episode_seed(7, "libero_10", 3, 4)
    assert episode_seed(7, "libero_10", 3, 4) != episode_seed(7, "libero_10", 3, 5)
    assert episode_seed(7, "libero_10", 3, 4) != episode_seed(17, "libero_10", 3, 4)


def test_chunk_guard_rejects_padding_short_chunks_and_nan():
    evaluator.check_chunk(np.zeros((10, 7)), 10, 10)
    for actions, required in [(np.zeros((10, 7)), 50), (np.zeros((50, 7)), 5), (np.zeros((10, 8)), 5), (np.full((10, 7), np.nan), 5)]:
        with pytest.raises(RuntimeError):
            evaluator.check_chunk(actions, 10, required)


def test_exact_call_count_includes_terminal_action_and_partial_chunk():
    evaluator.validate_call_schedule(13, 3, 5, [0, 5, 10])
    evaluator.validate_call_schedule(13, 2, 10, [0, 10])
    with pytest.raises(RuntimeError):
        evaluator.validate_call_schedule(13, 2, 5, [0, 5])


def test_wilson_and_paired_significance():
    low, high = wilson(0, 10)
    assert low == pytest.approx(0)
    assert high == pytest.approx(0.2775328)
    assert wilson(10, 10)[1] == pytest.approx(1)
    assert exact_mcnemar(0, 0) == 1
    assert exact_mcnemar(10, 0) == pytest.approx(0.001953125)
    assert exact_mcnemar(10, 10) == 1
    assert holm({10: 0.01, 20: 0.04, 30: 0.03}) == {10: 0.03, 30: 0.06, 20: 0.06}


def test_bootstrap_pairs_keep_repeated_seeds_in_same_state_block():
    pairs = []
    for seed in [7, 17]:
        for ep in range(5):
            t = {"task_id": 0, "episode_index": ep, "seed": seed, "success": True}
            pairs.append((t, dict(t, success=False)))
    assert paired_interval(pairs, replicates=100, seed=1) == (1, 1)
    assert paired_interval([(s, t) for t, s in pairs], replicates=100, seed=1) == (-1, -1)


@pytest.fixture
def official_fixture(monkeypatch, request):
    root = Path(os.environ.get("OPENPI_DIR", "/nonexistent"))
    if not (root / "examples/libero/main.py").exists():
        pytest.skip("Set OPENPI_DIR to run integration tests against the actual pinned upstream loop")
    options = getattr(request, "param", {})
    config = load_config()
    if options.get("prediction_horizon"):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/prediction50.yaml")
        monkeypatch.setattr(evaluator, "load_config", lambda: config)
    spec = upstream_spec(root, config)
    assert spec["native_prediction_horizon"] == 10
    assert spec["flow_steps"] == 10
    metadata = {"experiment_spec": spec, "inference_fingerprint": digest(spec)}
    created, requests = [], []
    fail_infer = [False]

    def observation():
        return {"agentview_image": np.arange(8*8*3, dtype=np.uint8).reshape(8, 8, 3),
                "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                "robot0_gripper_qpos": np.zeros(2)}

    class FakeEnv:
        def __init__(self, **kwargs):
            self.actions, self.episodes = [], []
            self.closed = False
            created.append(self)

        def seed(self, value):
            self.seed_value = value

        def reset(self):
            self.actions = []
            self.episodes.append(self.actions)
            return observation()

        def set_init_state(self, state):
            return observation()

        def step(self, action):
            self.actions.append(action)
            return observation(), 0., len(self.actions) == options.get("terminal_step", 23), {}

        def close(self):
            self.closed = True

    class FakeSuite:
        n_tasks = 1

        def get_task(self, task):
            return types.SimpleNamespace(language="synthetic integration fixture", problem_folder="fake", bddl_file="fake.bddl")

        def get_task_init_states(self, task):
            return np.arange(100).reshape(50, 2)

    class FakeClient:
        def __init__(self, *args):
            self._ws = types.SimpleNamespace(close=lambda: None)

        def get_server_metadata(self):
            return metadata

        def infer(self, request):
            requests.append(request)
            if fail_infer[0]:
                raise RuntimeError("synthetic inference failure")
            return {"actions": np.tile(np.arange(prediction_horizon(spec))[:, None], (1, 7)).astype(float), "inference_fingerprint": metadata["inference_fingerprint"]}

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    benchmark = module("libero.libero.benchmark", get_benchmark_dict=lambda: {"libero_10": FakeSuite})
    envs = module("libero.libero.envs", OffScreenRenderEnv=FakeEnv)
    inner = module("libero.libero", benchmark=benchmark, get_libero_path=lambda key: str(root), envs=envs)
    module("libero", libero=inner)
    image_tools = module("openpi_client.image_tools", convert_to_uint8=lambda x: x, resize_with_pad=lambda x, *args: x)
    socket = module("openpi_client.websocket_client_policy", WebsocketClientPolicy=FakeClient)
    module("openpi_client", image_tools=image_tools, websocket_client_policy=socket)
    module("imageio", mimwrite=lambda path, frames, **kw: Path(path).write_bytes(b"SYNTHETIC TEST ONLY"))
    module("tqdm", tqdm=lambda x: x)
    module("tyro", cli=lambda f: None)
    module("torch", manual_seed=lambda seed: None)
    return root, created, requests, fail_infer


def arguments(root, tmp, h):
    return argparse.Namespace(openpi_dir=str(root), suite="libero_10", seed=7, horizon=h,
        required_horizon=10, episodes=2, task_ids=[0], mode="diagnostic", host="fake", port=8000,
        results_dir=str(tmp))


def test_real_upstream_loop_discards_tail_and_writes_unique_paired_records(official_fixture, tmp_path):
    root, created, requests, _ = official_fixture
    for h in [5, 10]:
        evaluator.run(arguments(root, tmp_path, h))
    assert len(created) == 2 and all(e.closed for e in created)
    h5, h10 = created
    assert [a[0] for a in h5.episodes[0][10:]] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]
    assert [a[0] for a in h10.episodes[0][10:]] == list(range(10)) + [0, 1, 2]
    rows = [json.loads(line) for path in tmp_path.rglob("*.jsonl") for line in path.read_text().splitlines()]
    assert len(rows) == 4
    assert all(r["environment_steps"] == 23 and r["controlled_environment_steps"] == 13 for r in rows)
    assert all(r["policy_calls"] == math.ceil(13 / r["replan_steps"]) for r in rows)
    assert len({r["video_path"] for r in rows}) == 4
    validate_records(rows)
    changed = copy.deepcopy(rows)
    changed[-1]["initial_state_sha256"] = "different"
    with pytest.raises(ValueError, match="Unpaired"):
        validate_records(changed)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_records(rows + [rows[0]])
    changed = copy.deepcopy(rows)
    changed[-1]["inference_fingerprint"] = "different model"
    with pytest.raises(ValueError, match="changed checkpoints"):
        validate_records(changed)


def test_official_exception_handler_cannot_silently_score_errors_as_failures(official_fixture, tmp_path):
    root, created, _, fail = official_fixture
    fail[0] = True
    with pytest.raises(RuntimeError, match="not a task failure"):
        evaluator.run(arguments(root, tmp_path, 5))
    assert not list(tmp_path.rglob("*.jsonl"))
    manifest = json.loads(next(tmp_path.rglob("*.manifest.json")).read_text())
    assert manifest["status"] == "error"
    assert created[0].closed


def test_aggregation_and_all_three_plots_from_actual_loop_fixtures(official_fixture, tmp_path, monkeypatch):
    from frequency_vla import analysis
    root, _, _, _ = official_fixture
    for h in [5, 10]:
        evaluator.run(arguments(root, tmp_path, h))
    monkeypatch.setattr(sys, "argv", ["aggregate", "--results-dir", str(tmp_path), "--mode", "diagnostic", "--allow-partial"])
    analysis.main()
    aggregated = tmp_path / "aggregated/diagnostic/libero_10"
    validation = json.loads((aggregated / "validation.json").read_text())
    assert validation["measured_horizons"] == [5, 10]
    assert not validation["complete_for_measured_horizons"]
    assert validation["unsupported_requested_horizons"] == [20, 30, 40, 50]
    assert (aggregated / "frequency_sweep.csv").is_file()
    assert (aggregated / "replanning_gaps.csv").is_file()
    for name in ["success_vs_replan_horizon.png", "success_vs_policy_calls.png", "relative_performance_drop.png"]:
        assert (tmp_path / "figures/diagnostic/libero_10" / name).read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("official_fixture", [{"prediction_horizon": 50, "terminal_step": 73}], indirect=True)
def test_fixed_p50_executes_thirty_then_discards_twenty_and_pairs_with_h5(official_fixture, tmp_path, monkeypatch):
    from frequency_vla import analysis
    root, created, requests, _ = official_fixture
    for h in [5, 30]:
        args = arguments(root, tmp_path, h)
        args.required_horizon = 30
        evaluator.run(args)
    h5, h30 = created
    assert [a[0] for a in h5.episodes[0][10:]] == list(range(5)) * 12 + [0, 1, 2]
    assert [a[0] for a in h30.episodes[0][10:]] == list(range(30)) * 2 + [0, 1, 2]
    rows = [json.loads(line) for path in tmp_path.rglob("*.jsonl") for line in path.read_text().splitlines()]
    assert all(r["native_prediction_horizon"] == 10 and r["prediction_horizon"] == 50 and r["action_chunk_length"] == 50 for r in rows)
    validate_records(rows)
    changed = copy.deepcopy(rows)
    changed[0]["prediction_horizon"] = 10
    with pytest.raises(ValueError, match="different prediction horizons"):
        validate_records(changed)
    monkeypatch.setenv("FREQUENCY_CONFIG", str(Path(__file__).resolve().parents[1] / "configs/prediction50.yaml"))
    monkeypatch.setattr(sys, "argv", ["aggregate", "--results-dir", str(tmp_path), "--mode", "diagnostic", "--allow-partial"])
    analysis.main()
    out = tmp_path / "aggregated/diagnostic/libero_10"
    validation = json.loads((out / "validation.json").read_text())
    assert validation["prediction_horizon"] == 50
    assert validation["measured_horizons"] == [5, 30]
    assert validation["unsupported_requested_horizons"] == []
    assert not validation["official_prediction_horizon_unchanged"]
    text = (out / "FINDINGS.md").read_text()
    assert "Explicit fixed-P extension" in text
    assert "No model/config override" not in text
