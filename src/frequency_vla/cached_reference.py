"""Reuse a predeclared, checksum-pinned evaluation without spending GPU time."""
import json
from pathlib import Path
import shutil

from .logging_utils import append_record, digest, file_digest, write_json
from .study_plan import condition_id, conditions


def reference_inventory(archive, step):
    archive = Path(archive)
    folder = archive / "evaluations/confirmation" / ("step_" + str(step))
    files = sorted((folder / "raw").rglob("*.json*")) + [
        archive / "provenance" / ("step_" + str(step) + ".json"),
        archive / "provenance/final_checks.json", archive / "stages.jsonl"]
    if len(files) != 23:
        raise ValueError("Cached reference requires ten complete episode shards and manifests")
    return {str(p.relative_to(archive)): file_digest(p) for p in files}


def audit_reference(plan):
    declaration = plan["cached_reference"]
    step = declaration["step"]
    if step != plan["resume_step"] or step != plan.get("comparison_step", step):
        raise ValueError("Only the resumed comparison checkpoint may reuse an evaluation")
    selected = [c for c in conditions(plan) if c["step"] == step]
    if len(selected) != 1 or selected[0]["split"] != "confirmation":
        raise ValueError("Exactly one cached confirmation condition is required")
    archive = Path(declaration["archive"])
    inventory = reference_inventory(archive, step)
    if digest(inventory) != declaration["inventory_sha256"]:
        raise ValueError("Cached reference archive changed after declaration")
    final = json.loads((archive / "provenance/final_checks.json").read_text())
    checkpoint = json.loads((archive / "provenance" / ("step_" + str(step) + ".json")).read_text())
    if not final["complete"] or not final["all_checkpoint_file_checksums_verified"] or checkpoint["manifest_sha256"] != plan["resume_manifest_sha256"]:
        raise ValueError("Cached reference is not the verified resumed checkpoint")
    folder = archive / "evaluations/confirmation" / ("step_" + str(step))
    manifests = [json.loads(p.read_text()) for p in (folder / "raw").rglob("*.manifest.json")]
    servers = {m["server"]["server_instance_id"] for m in manifests}
    if len(servers) != 1 or any(m["status"] != "complete" or
            m["server"]["experiment_spec"]["temporal_opsd"]["optimizer_step"] != step or
            m["server"]["experiment_spec"]["temporal_opsd"]["checkpoint"] != checkpoint or
            m["server"]["evaluation_spec"]["rendering"]["backend"] != plan["renderer"] for m in manifests):
        raise ValueError("Cached reference server/snapshot/renderer is inconsistent")
    identifier = condition_id(selected[0])
    stages = [json.loads(line) for line in (archive / "stages.jsonl").read_text().splitlines()]
    stage = next(r for r in stages if r["stage"] == identifier and r["event"] == "complete")
    return dict(condition=identifier, step=step, archive=str(archive.resolve()),
                files_sha256=inventory, inventory_sha256=digest(inventory),
                server_instance_id=next(iter(servers)), evaluation_stage_seconds=stage["seconds"],
                episodes=selected[0]["episodes_per_task"] * 10)


def prepare_reference(root, plan):
    root = Path(root)
    audit = audit_reference(plan)
    archive = Path(audit["archive"])
    prefix = "evaluations/confirmation/step_{}".format(audit["step"])
    for relative in audit["files_sha256"]:
        if relative.startswith(prefix + "/raw/") or relative == "provenance/step_{}.json".format(audit["step"]):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("Cached reference destination must be fresh")
            shutil.copyfile(archive / relative, target)
    # Keep the original video files; do not duplicate several GB of artifacts.
    (root / prefix / "videos").symlink_to(archive / prefix / "videos", target_is_directory=True)
    write_json(root / "provenance/cached_reference.json", audit)
    append_record(root / "stages.jsonl", dict(stage=audit["condition"], event="reused",
        seconds=audit["evaluation_stage_seconds"], source_archive=audit["archive"],
        charged_to_current_allocation=False))
    return audit


def verify_reference(root, plan):
    root = Path(root)
    audit = audit_reference(plan)
    if json.loads((root / "provenance/cached_reference.json").read_text()) != audit:
        raise ValueError("Cached reference provenance changed")
    prefix = "evaluations/confirmation/step_{}".format(audit["step"])
    expected = {p: sha for p, sha in audit["files_sha256"].items()
                if p.startswith(prefix + "/raw/") or p == "provenance/step_{}.json".format(audit["step"])}
    actual = {str(p.relative_to(root)) for p in (root / prefix / "raw").rglob("*.json*")}
    if actual != {p for p in expected if p.startswith(prefix)} or any(file_digest(root / p) != sha for p, sha in expected.items()):
        raise ValueError("Copied reference measurements changed")
    return audit
