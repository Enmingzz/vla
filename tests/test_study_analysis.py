"""Guard paired outcome direction and layout/observation matching."""
import copy
from pathlib import Path

import pytest

from frequency_vla.config import load_config
from frequency_vla.logging_utils import digest, write_json
from frequency_vla.study_analysis import analyze, load_archived_plan, paired_change


@pytest.mark.parametrize("filename", ["autoresearch_round2.yaml", "autoresearch_round3.yaml", "autoresearch_round3_finish.yaml", "autoresearch_round4.yaml"])
def test_archived_plan_preserves_original_integer_key_digest(tmp_path, filename):
    plan = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    path = tmp_path / "plan.json"
    write_json(path, plan)
    restored = load_archived_plan(path)
    assert restored == plan
    assert digest(restored) == digest(plan)


def test_archived_plan_still_rejects_real_protocol_changes(tmp_path):
    plan = load_config(Path(__file__).resolve().parents[1] / "configs/autoresearch_round3_finish.yaml")
    expected_digest = digest(plan)
    write_json(tmp_path / "provenance/split_audit.json", dict(plan_sha256=expected_digest))
    write_json(tmp_path / "provenance/study_complete.json", dict(complete=True, plan_sha256=expected_digest))
    plan["confirmation"]["seed"] += 1
    write_json(tmp_path / "provenance/study_plan.json", plan)
    with pytest.raises(ValueError, match="incomplete or changed"):
        analyze(tmp_path)


def test_recovery_uses_left_minus_right_and_rejects_unpaired_observations():
    before, after = [], []
    for index in range(20):
        row = dict(seed=27, task_id=index // 10, episode_index=index % 10,
                   initial_state_index=30 + index % 10,
                   initial_state_sha256=str(index), first_observation_sha256=str(index),
                   episode_rng_seed=index, evaluation_fingerprint="same evaluator",
                   task_description="task {}".format(index // 10), prediction_horizon=50,
                   success=index >= 8)
        before.append(row)
        after.append(dict(row, success=index < 8 or index >= 10))
    plan = dict(bootstrap_replicates=1000, analysis_seed=7)
    change = paired_change(after, before, plan)
    assert change["recovered_episodes"] == 8
    assert change["regressed_episodes"] == 2
    assert change["success_change"] == .3
    reverse = paired_change(before, after, plan)
    assert reverse["success_change"] == -.3
    assert reverse["p_exact"] == change["p_exact"]
    mismatched = copy.deepcopy(after)
    mismatched[0]["first_observation_sha256"] = "different image"
    with pytest.raises(ValueError, match="Unpaired comparison"):
        paired_change(mismatched, before, plan)
    with pytest.raises(ValueError, match="Unequal paired episode coverage"):
        paired_change(after[:-1], before, plan)
