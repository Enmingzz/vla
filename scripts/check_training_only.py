"""Validate the saved continuation on CPU; success-rate evaluation is deferred."""
import argparse
import json
from pathlib import Path
import subprocess

from frequency_vla.logging_utils import digest, file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)
    finished = json.loads((root / "provenance/training_complete.json").read_text())
    first, last = finished["start_step"], finished["last_completed_step"]
    assert first == 500 and first <= last <= 1000
    assert finished["optimizer_updates_added"] == last - first
    assert finished["complete"] == (last == 1000)
    assert not finished["evaluation_complete"]
    assert not list(root.glob("evaluations/**/raw/**/*.jsonl"))
    audit = json.loads((root / "provenance/split_audit.json").read_text())
    eval_hashes = {s["initial_state_sha256"] for states in audit["evaluation_initial_states"].values() for s in states}
    records = []
    for filename in ["training.jsonl", "rollouts.jsonl"]:
        path = root / filename
        rows = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
        assert [r["optimizer_step"] for r in rows] == list(range(first + 1, last + 1))
        assert not any(r["diagnostic"] for r in rows)
        records.append(rows)
    training, rollouts = records
    for row, rollout in zip(training, rollouts):
        assert row["student_behavior_version"] == row["optimizer_step"] - 1
        for state in rollout["initial_states"]:
            assert 10 <= state["initial_state_index"] < 20
            assert state["initial_state_sha256"] not in eval_hashes
    saved = finished["checkpoint"]
    checkpoint = Path(saved["path"])
    manifest = json.loads((checkpoint / "training_manifest.json").read_text())
    assert digest(manifest) == saved["manifest_sha256"] and manifest["step"] == last
    assert manifest["reloaded_native_inference_max_abs_difference"] == 0
    if last > first:
        assert manifest["resumed_from"]["step"] == first
        assert manifest["resumed_from"]["manifest_sha256"] == audit["parent_manifest_sha256"]
    for name, expected in manifest["files"].items():
        assert file_digest(checkpoint / name) == expected, name
    resume = json.loads((root / "provenance/resume.json").read_text())
    assert all(resume[k] for k in ["fp32_master_restored", "ema_and_optimizer_restored", "frozen_backbone_equal"])
    job = json.loads((root / "provenance/submission.json").read_text())["job_id"]
    accounting = subprocess.check_output(["sacct", "-j", job, "-nP", "-o",
        "JobID,State,ElapsedRaw,TimelimitRaw,AllocTRES,ExitCode,NodeList"], text=True)
    (root / "provenance/training_slurm_accounting.psv").write_text(accounting)
    main_row = next(line.split("|") for line in accounting.splitlines() if line.split("|")[0] == job)
    assert main_row[1] == "COMPLETED" and main_row[5] == "0:0"
    assert int(main_row[3]) == 60 and "gres/gpu=1" in main_row[4].split(",")
    validation = dict(validation_passed=True, requested_updates_complete=last == 1000,
        start_step=first, last_completed_step=last, optimizer_updates_added=last-first,
        formal_evaluation_episodes=0, evaluation_deferred=True,
        checkpoint=saved, all_checkpoint_files_verified=True, native_reload_max_abs_difference=0,
        training_evaluation_states_disjoint=True, gpu_job_id=job, gpu_seconds=int(main_row[2]),
        maximum_concurrent_gpus=1, allocation_released=True)
    write_json(root / "provenance/training_only_validation.json", validation)
    write_json(root / "provenance" / ("exported_training_manifest_step_{}.json".format(last)), manifest)
    text = ["# Training first: one-hour allocation", "",
        "Completed **{} additional updates**, from step {} to step {} (target: step 1000).".format(last-first, first, last), "",
        "The requested 500 updates are complete." if last == 1000 else
        "The time budget stopped training early; the completed progress is saved. The 500-update request is not yet complete.", "",
        "Checkpoint: `{}`.".format(saved["path"]), "",
        "All checkpoint files, FP32/EMA/Adam restoration and native inference round-trip were verified. "
        "The single H100 allocation has ended after {:.1f} minutes.".format(int(main_row[2])/60), "",
        "**No success-rate evaluation ran in this allocation.** Whether performance improved remains unknown. "
        "The paired step-500 versus step-1000 evaluation remains deferred; training loss is not task success.", ""]
    (root / "TRAINING_STATUS.md").write_text("\n".join(text))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
