"""Protocol checks run without allocating a GPU or loading a VLA model."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from frequency_vla.config import load_config
from frequency_vla.opsd_protocol import align_teacher_latent, require_measured_gap, validate_training_config
from frequency_vla.study_plan import conditions


def test_temporal_alignment_preserves_the_student_suffix_and_samples_only_missing_tail():
    student = np.arange(50).reshape(1, 50, 1)
    auxiliary = (100 + np.arange(50)).reshape(1, 50, 1)
    for offset in [0, 5, 10, 15]:
        result = align_teacher_latent(student, auxiliary, offset, np)
        assert result.shape == student.shape
        assert result[0, :5, 0].tolist() == list(range(offset, offset + 5))
        assert np.array_equal(result[:, :50-offset], student[:, offset:])
        if offset:
            assert np.array_equal(result[:, -offset:], auxiliary[:, -offset:])
        else:
            assert result is student


def test_training_indices_must_not_overlap_held_out_evaluation():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/opsd_h20_100.yaml")
    validate_training_config(config)
    config["train_initial_state_start"] = 9
    with pytest.raises(ValueError, match="overlap"):
        validate_training_config(config)


def test_continuation_split_protects_confirmation_and_caps_updates():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/opsd_continuation_500.yaml")
    validate_training_config(config)
    config["train_initial_state_stop"] = 31
    with pytest.raises(ValueError, match="overlap"):
        validate_training_config(config)
    config["train_initial_state_stop"] = 20
    config["optimizer_steps"] = 1501
    with pytest.raises(ValueError, match="at-most-1500"):
        validate_training_config(config)


def test_third_round_keeps_the_method_and_exact_additional_update_budget():
    base = Path(__file__).resolve().parents[1]
    previous = load_config(base / "configs/opsd_continuation_500.yaml")
    current = load_config(base / "configs/opsd_continuation_1000.yaml")
    validate_training_config(current)
    plan = load_config(base / "configs/autoresearch_round3.yaml")
    assert current["optimizer_steps"] - plan["resume_step"] == 500
    for key in ["algorithm", "prediction_horizon", "student_horizon", "teacher_horizon", "flow_steps",
                "batch_size", "learning_rate", "ema_decay", "trainable_parameters", "teacher_tail"]:
        assert current[key] == previous[key]
    assert current["checkpoint_steps"] == plan["milestones"] == [500, 1000]
    matrix = conditions(plan)
    assert len(matrix) == 2
    assert sum(c["episodes_per_task"] * 10 for c in matrix) == 200
    assert all(c["horizon"] == 20 and c["initial_state_start"] == 30 and c["seed"] == 27 for c in matrix)


def test_research_matrix_keeps_confirmation_separate_and_preselects_final_step():
    plan = load_config(Path(__file__).resolve().parents[1] / "configs/autoresearch_round2.yaml")
    matrix = conditions(plan)
    assert sum(c["episodes_per_task"] * 10 for c in matrix) == 1310
    screen = {i for c in matrix if c["split"] != "confirmation"
              for i in range(c["initial_state_start"], c["initial_state_start"] + c["episodes_per_task"])}
    confirmation = {i for c in matrix if c["split"] == "confirmation"
                    for i in range(c["initial_state_start"], c["initial_state_start"] + c["episodes_per_task"])}
    assert not screen & confirmation
    assert {c["step"] for c in matrix if c["split"] == "confirmation"} == {0, 100, 500}


def test_training_gate_rejects_inconclusive_and_incomplete_frequency_results(tmp_path):
    root = tmp_path / "aggregated/smoke/libero_10"
    root.mkdir(parents=True)
    validation = {"complete_for_measured_horizons": True, "paired_checks_passed": True,
                  "measured_horizons": [5, 15, 20], "prediction_horizon": 50}
    (root / "validation.json").write_text(json.dumps(validation))
    gap = {"student_H": 20, "replanning_gap": .1, "gap_ci95_low": -.01, "mcnemar_p_holm": .08}

    def write_gap():
        with (root / "replanning_gaps.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(gap))
            writer.writeheader()
            writer.writerow(gap)

    write_gap()
    with pytest.raises(ValueError, match="not yet convincingly"):
        require_measured_gap(tmp_path)
    gap.update(gap_ci95_low=.03, mcnemar_p_holm=.01)
    write_gap()
    assert require_measured_gap(tmp_path)["gap"]["student_H"] == "20"
    validation["complete_for_measured_horizons"] = False
    (root / "validation.json").write_text(json.dumps(validation))
    with pytest.raises(ValueError, match="incomplete"):
        require_measured_gap(tmp_path)
