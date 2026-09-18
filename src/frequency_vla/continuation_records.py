"""Verify and join completed continuation segments without rewriting their archives."""
import json
from pathlib import Path

from .logging_utils import digest, file_digest


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit_prior_segments(plan):
    expected_step = plan.get("comparison_step", plan["resume_step"])
    expected_manifest = plan.get("comparison_manifest_sha256", plan["resume_manifest_sha256"])
    segments = []
    for directory in plan.get("prior_continuation_results", []):
        root = Path(directory).resolve()
        resume = json.loads((root / "provenance/resume.json").read_text())
        training, rollouts = [read_rows(root / name) for name in ["training.jsonl", "rollouts.jsonl"]]
        if not training or resume["step"] != expected_step or resume["manifest_sha256"] != expected_manifest:
            raise ValueError("Continuation checkpoint chain is broken")
        if not all(resume[k] for k in ["fp32_master_restored", "ema_and_optimizer_restored", "frozen_backbone_equal"]):
            raise ValueError("A prior continuation did not restore complete training state")
        final = training[-1]["optimizer_step"]
        expected = list(range(expected_step + 1, final + 1))
        if any([r["optimizer_step"] for r in rows] != expected for rows in [training, rollouts]):
            raise ValueError("A prior continuation has missing or duplicate updates")
        identity_path = root / "provenance" / ("step_" + str(final) + ".json")
        identity = json.loads(identity_path.read_text())
        manifest_path = Path(identity["path"]) / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if digest(manifest) != identity["manifest_sha256"] or manifest["step"] != final:
            raise ValueError("A prior continuation checkpoint manifest differs")
        if manifest["resumed_from"] != resume or manifest["reloaded_native_inference_max_abs_difference"] != 0:
            raise ValueError("A prior continuation lacks a verified checkpoint roundtrip")
        files = {name: root / name for name in ["training.jsonl", "rollouts.jsonl", "provenance/resume.json", "provenance/training_setup.json"]}
        files.update(checkpoint_identity=identity_path, checkpoint_manifest=manifest_path)
        segments.append(dict(start_step=expected_step, final_step=final,
            parent_manifest_sha256=expected_manifest, final_manifest_sha256=identity["manifest_sha256"],
            files={key: dict(path=str(p), sha256=file_digest(p)) for key, p in files.items()}))
        expected_step, expected_manifest = final, identity["manifest_sha256"]
    if expected_step != plan["resume_step"] or expected_manifest != plan["resume_manifest_sha256"]:
        raise ValueError("Prior continuation segments do not reach the declared resume checkpoint")
    return segments


def training_records(root, plan, audit):
    training, rollouts = [], []
    segments = audit.get("continuation_segments", [])
    if len(segments) != len(plan.get("prior_continuation_results", [])):
        raise ValueError("Missing audited continuation segments")
    for segment in segments:
        for item in segment["files"].values():
            if file_digest(item["path"]) != item["sha256"]:
                raise ValueError("An audited continuation segment changed")
        training.extend(read_rows(segment["files"]["training.jsonl"]["path"]))
        rollouts.extend(read_rows(segment["files"]["rollouts.jsonl"]["path"]))
    current_training, current_rollouts = [read_rows(Path(root) / name) for name in ["training.jsonl", "rollouts.jsonl"]]
    expected_current = list(range(plan["resume_step"] + 1, plan["milestones"][-1] + 1))
    if any([r["optimizer_step"] for r in rows] != expected_current for rows in [current_training, current_rollouts]):
        raise ValueError("Current continuation must contain exactly the remaining updates")
    training.extend(current_training)
    rollouts.extend(current_rollouts)
    expected_all = list(range(plan.get("comparison_step", plan["resume_step"]) + 1, plan["milestones"][-1] + 1))
    if any([r["optimizer_step"] for r in rows] != expected_all for rows in [training, rollouts]):
        raise ValueError("Combined continuation has missing or duplicate updates")
    return training, rollouts
