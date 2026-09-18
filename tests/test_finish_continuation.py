"""The 978 -> 1000 finish must preserve the 500 -> 1000 paired comparison."""
import json
from pathlib import Path

import pytest

from frequency_vla import study_runner
from frequency_vla.config import load_config
from frequency_vla.continuation_records import audit_prior_segments, training_records
from frequency_vla.logging_utils import digest, write_json


def test_finish_saves_before_evaluation_and_loads_baseline_without_resuming_it(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    plan_path = project / "configs/autoresearch_round3_finish.yaml"
    plan = load_config(plan_path)
    root = tmp_path / "results"
    (root / "logs").mkdir(parents=True)
    write_json(root / "provenance/split_audit.json", dict(plan_sha256=digest(plan)))
    write_json(root / "provenance/parallel_environment_check.json", dict(passed=True, source_sha256={}))
    write_json(root / "provenance/preflight_checks.json", dict(passed=True, sources={}))
    operations = []

    def execute(command, **kwargs):
        if "frequency_vla.opsd_evaluate" in command:
            operations.append("evaluate")
            assert command[command.index("--episodes") + 1] == "10"
            assert command[command.index("--initial-state-start") + 1] == "30"
            assert command[command.index("--horizon") + 1] == "20"
            return
        operation = command[command.index("--stage") + 1]
        operations.append(operation)
        if operation == "resume":
            assert command[command.index("--checkpoint") + 1] == "SYNTHETIC_STEP_978"
        if operation == "train":
            assert command[command.index("--end-step") + 1] == "1000"
        if operation == "load_snapshot":
            assert command[command.index("--checkpoint") + 1] == "SYNTHETIC_STEP_500"
            assert command[command.index("--manifest-sha256") + 1] == plan["comparison_manifest_sha256"]

    monkeypatch.setattr(study_runner.subprocess, "run", execute)
    monkeypatch.setattr(study_runner.sys, "argv", ["study_runner", "--plan", str(plan_path),
        "--training-config", str(project / "configs/opsd_continuation_1000.yaml"),
        "--parent-checkpoint", "SYNTHETIC_STEP_978", "--baseline-checkpoint", "SYNTHETIC_STEP_500",
        "--port", "12345", "--results-dir", str(root), "--checkpoint-root", str(tmp_path / "checkpoints")])
    study_runner.main()
    assert operations == ["diagnostic", "resume", "train", "save", "status", "load_snapshot",
                          "snapshot", "evaluate", "snapshot", "evaluate"]
    completed = json.loads((root / "provenance/study_complete.json").read_text())
    assert completed["optimizer_steps_added"] == 22
    assert completed["comparison_optimizer_steps_added"] == 500
    assert len(completed["completed_conditions"]) == 2


def test_segment_join_checks_chain_hashes_and_exact_update_coverage(tmp_path):
    prior, current, checkpoint = [tmp_path / name for name in ["prior", "current", "checkpoint"]]
    resume = dict(step=500, manifest_sha256="BASELINE", fp32_master_restored=True,
                  ema_and_optimizer_restored=True, frozen_backbone_equal=True)
    manifest = dict(step=978, resumed_from=resume, reloaded_native_inference_max_abs_difference=0)
    write_json(checkpoint / "training_manifest.json", manifest)
    write_json(prior / "provenance/resume.json", resume)
    write_json(prior / "provenance/training_setup.json", {})
    write_json(prior / "provenance/step_978.json", dict(path=str(checkpoint), manifest_sha256=digest(manifest)))
    for root, start, end in [(prior, 501, 979), (current, 979, 1001)]:
        root.mkdir(exist_ok=True)
        for name in ["training.jsonl", "rollouts.jsonl"]:
            (root / name).write_text("\n".join(json.dumps(dict(optimizer_step=s)) for s in range(start, end)))
    plan = dict(comparison_step=500, comparison_manifest_sha256="BASELINE", resume_step=978,
                resume_manifest_sha256=digest(manifest), prior_continuation_results=[str(prior)], milestones=[500, 1000])
    audit = dict(continuation_segments=audit_prior_segments(plan))
    training, rollouts = training_records(current, plan, audit)
    assert len(training) == len(rollouts) == 500
    assert [r["optimizer_step"] for r in training] == list(range(501, 1001))
    with pytest.raises(ValueError, match="chain is broken"):
        audit_prior_segments(dict(plan, comparison_manifest_sha256="WRONG"))
    last = current / "training.jsonl"
    last.write_text(last.read_text() + '\n{"optimizer_step": 1000}')
    with pytest.raises(ValueError, match="exactly the remaining"):
        training_records(current, plan, audit)
    (prior / "training.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="segment changed"):
        training_records(current, plan, audit)
