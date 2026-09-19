"""CPU-only final analysis, video/checkpoint integrity and allocation accounting."""
import argparse
import json
from pathlib import Path
import subprocess

from frequency_vla.logging_utils import digest, file_digest, write_json
from frequency_vla.study_analysis import analyze
from frequency_vla.continuation_analysis import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir).resolve()
    plan, summaries, comparisons, tasks, training, validation = analyze(root)
    report(root, plan, summaries, tasks, training, validation)
    assert validation["complete"] and validation["optimizer_updates_added"] == 500
    assert validation["evaluation_episodes"] == 200
    videos = []
    for shard in sorted(root.glob("evaluations/confirmation/step_*/raw/smoke/libero_10/seed_27/H_20/*.jsonl")):
        for line in shard.read_text().splitlines():
            row = json.loads(line)
            path = shard.parents[5] / row["video_path"]
            if not path.is_file() or not path.stat().st_size:
                raise ValueError("Missing episode video: " + str(path))
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)], text=True))["streams"][0]
            assert (info["width"], info["height"], info["r_frame_rate"]) == (224, 224, "10/1"), path
            assert int(info["nb_frames"]) == row["controlled_environment_steps"], path
            videos.append(dict(video=str(path.relative_to(root)), frames=int(info["nb_frames"])))
    assert len(videos) == len({v["video"] for v in videos}) == 200
    write_json(root / "provenance/video_checks.json", dict(passed=True, videos=200, all_frame_counts_match=True, checks=videos))
    endpoint = plan["milestones"][-1]
    step = json.loads((root / "provenance" / ("step_" + str(endpoint) + ".json")).read_text())
    checkpoint = Path(step["path"])
    manifest = json.loads((checkpoint / "training_manifest.json").read_text())
    assert digest(manifest) == step["manifest_sha256"] and manifest["step"] == endpoint
    assert manifest["resumed_from"]["step"] == plan["resume_step"]
    assert manifest["resumed_from"]["manifest_sha256"] == plan["resume_manifest_sha256"]
    assert manifest["reloaded_native_inference_max_abs_difference"] == 0
    for name, expected in manifest["files"].items():
        if file_digest(checkpoint / name) != expected:
            raise ValueError("Final checkpoint checksum mismatch: " + name)
    write_json(root / "provenance" / ("exported_training_manifest_step_" + str(endpoint) + ".json"), manifest)
    job = json.loads((root / "provenance/submission.json").read_text())["job_id"]
    accounting = subprocess.check_output(["sacct", "-j", str(job), "-nP", "-o",
        "JobID,State,ElapsedRaw,AllocTRES,ExitCode,NodeList"], text=True)
    (root / "provenance/slurm_accounting.psv").write_text(accounting)
    main_job = next(line.split("|") for line in accounting.splitlines() if line.split("|")[0] == str(job))
    assert main_job[1] == "COMPLETED" and main_job[4] == "0:0", main_job
    assert "gres/gpu=1" in main_job[3].split(","), main_job
    write_json(root / "provenance/resource_accounting.json", dict(job_id=job, state=main_job[1],
        gpu_seconds=int(main_job[2]), gpu_hours=int(main_job[2])/3600, maximum_concurrent_gpus=1,
        allocation_released=True, node=main_job[5], alloc_tres=main_job[3]))
    artifacts = list((root / "aggregated").glob("*")) + list((root / "figures").glob("*")) + [
        root / "FINDINGS.md", root / "training.jsonl", root / "rollouts.jsonl"]
    write_json(root / "provenance/final_checks.json", dict(complete=True, new_optimizer_updates=500,
        updates_in_final_allocation=validation["updates_in_final_allocation"], paired_evaluation_episodes=200,
        new_evaluation_episodes=validation["new_evaluation_episodes"],
        reused_evaluation_episodes=validation["reused_evaluation_episodes"],
        validated_videos=200, checkpoint_manifest_sha256=step["manifest_sha256"],
        all_checkpoint_file_checksums_verified=True, native_reload_max_abs_difference=0,
        gpu_allocation_released=True, artifacts={str(p.relative_to(root)):file_digest(p) for p in artifacts if p.is_file()}))
    print(json.dumps(dict(complete=True, primary=validation["primary"], checkpoint=step, gpu_seconds=int(main_job[2])), indent=2))


if __name__ == "__main__":
    main()
