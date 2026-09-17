"""No evaluation may run inside the user's one-hour training allocation."""
import json
import time

import pytest

from frequency_vla.config import load_config
from frequency_vla.logging_utils import digest, write_json
from frequency_vla import study_runner


@pytest.mark.parametrize("last_step", [731, 1000])
def test_training_only_exports_actual_step_and_defers_all_evaluation(tmp_path, monkeypatch, last_step):
    from pathlib import Path
    project = Path(__file__).resolve().parents[1]
    plan_path = project / "configs/autoresearch_round3.yaml"
    plan = load_config(plan_path)
    root = tmp_path / "results"
    (root / "logs").mkdir(parents=True)
    write_json(root / "provenance/split_audit.json", dict(plan_sha256=digest(plan)))
    write_json(root / "provenance/parallel_environment_check.json", dict(passed=True, source_sha256={}))
    write_json(root / "provenance/execution_request.json", dict(
        mode="train_only", wall_time_limit_seconds=3600, checkpoint_reserve_seconds=300))
    start = int(time.time()) - 200
    monkeypatch.setenv("SLURM_JOB_START_TIME", str(start))
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(start + 3600))
    operations = []

    def execute(command, **kwargs):
        assert "frequency_vla.opsd_evaluate" not in command
        operation = command[command.index("--stage") + 1]
        operations.append(operation)
        if operation == "train":
            assert float(command[command.index("--stop-at-unix-time") + 1]) == start + 3300
            write_json(root / "provenance/training_progress.json", dict(
                start_step=500, last_completed_step=last_step, requested_end_step=1000,
                optimizer_updates_added=last_step - 500, requested_updates_complete=last_step == 1000,
                stopped_for_walltime=last_step < 1000))
        if operation == "save":
            write_json(root / ("provenance/step_{}.json".format(last_step)), dict(path="SYNTHETIC_TEST_ONLY"))

    monkeypatch.setattr(study_runner.subprocess, "run", execute)
    monkeypatch.setattr(study_runner.sys, "argv", ["study_runner", "--plan", str(plan_path),
        "--training-config", str(project / "configs/opsd_continuation_1000.yaml"),
        "--parent-checkpoint", "SYNTHETIC_TEST_ONLY", "--port", "12345",
        "--results-dir", str(root), "--checkpoint-root", str(tmp_path / "checkpoints")])
    study_runner.main()
    result = json.loads((root / "provenance/training_complete.json").read_text())
    assert operations == ["diagnostic", "resume", "train", "save", "status"]
    assert result["complete"] == (last_step == 1000)
    assert result["last_completed_step"] == last_step
    assert result["evaluation_complete"] is False
    assert len(result["deferred_conditions"]) == 2
    assert not (root / "provenance/study_complete.json").exists()
