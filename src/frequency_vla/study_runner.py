"""Execute the frozen research matrix; task outcomes never change the schedule."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import load_config
from .logging_utils import append_record, digest, file_digest, write_json
from .study_plan import condition_id, conditions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--training-config", required=True)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--baseline-checkpoint", help="Read-only comparison checkpoint when resuming a partial continuation")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--checkpoint-root", required=True)
    args = p.parse_args()
    root, plan = Path(args.results_dir), load_config(args.plan)
    write_json(root / "provenance/study_plan.json", plan)
    audit = json.loads((root / "provenance/split_audit.json").read_text())
    if audit["plan_sha256"] != digest(plan):
        raise ValueError("Plan changed after initial-state audit")
    if plan.get("prior_continuation_results") or plan.get("require_cpu_preflight"):
        preflight = json.loads((root / "provenance/preflight_checks.json").read_text())
        if not preflight["passed"] or any(file_digest(p) != sha for p, sha in preflight["sources"].items()):
            raise ValueError("Completion implementation changed after CPU validation")
    if plan.get("continuation_only"):
        check = json.loads((root / "provenance/parallel_environment_check.json").read_text())
        if not check["passed"] or any(file_digest(p) != sha for p, sha in check["source_sha256"].items()):
            raise ValueError("Parallel software rendering was not verified for this implementation")
    matrix = conditions(plan)
    seen = set()
    reference = None
    if plan.get("cached_reference"):
        from .cached_reference import verify_reference
        reference = verify_reference(root, plan)
        if audit.get("cached_reference") != reference:
            raise ValueError("Cached reference differs from the preflight audit")
        seen.add(reference["condition"])

    def run(name, command, timeout):
        started = time.monotonic()
        append_record(root / "stages.jsonl", {"stage": name, "event": "start", "unix_time": time.time(), "command": command})
        print("Starting {} ({}s bound)".format(name, timeout), flush=True)
        logfile = root / "logs" / (name + ".log")
        with logfile.open("x") as log:
            subprocess.run(["timeout", "--kill-after=10s", str(timeout)] + command,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        append_record(root / "stages.jsonl", {"stage": name, "event": "complete", "unix_time": time.time(),
            "seconds": time.monotonic() - started})

    def client(stage, *extra, limit=300, name=None):
        run(name or stage, [sys.executable, "-m", "frequency_vla.opsd_client", "--stage", stage,
            "--config", args.training_config, "--port", str(args.port), "--results-dir", str(root),
            "--rollouts-dir", str(Path(args.checkpoint_root) / "rollouts")] + list(extra), limit)

    def evaluate(step, confirmation=False):
        client("snapshot", "--snapshot", str(step), name="select_{}_{}".format(step, "confirm" if confirmation else "screen"), limit=60)
        for c in matrix:
            if c["step"] != step or (c["split"] == "confirmation") != confirmation:
                continue
            identifier = condition_id(c)
            output = root / "evaluations" / c["split"] / ("step_" + str(step))
            run(identifier, [sys.executable, "-m", "frequency_vla.opsd_evaluate",
                "--port", str(args.port), "--results-dir", str(output), "--workers", str(plan.get("evaluation_workers", 4)),
                "--suite", c["suite"], "--horizon", str(c["horizon"]),
                "--episodes", str(c["episodes_per_task"]), "--seed", str(c["seed"]),
                "--initial-state-start", str(c["initial_state_start"])], plan.get("evaluation_timeout_seconds", 1200))
            seen.add(identifier)

    if plan.get("continuation_only"):
        request_path = root / "provenance/execution_request.json"
        execution = json.loads(request_path.read_text()) if request_path.exists() else None
        deadline = None
        if execution is not None:
            from .execution_budget import training_deadline
            deadline = training_deadline(execution, os.environ)
        client("diagnostic", limit=plan["diagnostic_timeout_seconds"])
        client("resume", "--checkpoint", args.parent_checkpoint, limit=300)
        if execution is not None:
            endpoint = plan["milestones"][-1]
            client("train", "--end-step", str(endpoint), "--stop-at-unix-time", str(deadline),
                   name="train_to_" + str(endpoint), limit=max(60, int(deadline - time.time()) + 60))
            progress = json.loads((root / "provenance/training_progress.json").read_text())
            actual = progress["last_completed_step"]
            if actual > plan["resume_step"]:
                client("save", name="save_" + str(actual), limit=240)
            client("status", name="memory_at_" + str(actual), limit=30)
            checkpoint = json.loads((root / "provenance" / ("step_" + str(actual) + ".json")).read_text())
            write_json(root / "provenance/training_complete.json", {
                **progress, "complete": progress["requested_updates_complete"],
                "checkpoint": checkpoint, "plan_sha256": digest(plan),
                "execution_request_sha256": digest(execution), "evaluation_complete": False,
                "deferred_conditions": sorted(condition_id(c) for c in matrix)})
            print("Training-only allocation finished at step {}; checkpoint saved, evaluation deferred.".format(actual), flush=True)
            return
        first = plan.get("comparison_step", plan["resume_step"])
        if first != plan["resume_step"] and not args.baseline_checkpoint:
            raise ValueError("Finishing a partial continuation requires --baseline-checkpoint")
        if first == plan["resume_step"] and reference is None:
            evaluate(first, confirmation=True)
        endpoint = plan["milestones"][-1]
        client("train", "--end-step", str(endpoint), name="train_to_" + str(endpoint),
               limit=plan["training_timeout_seconds"])
        client("save", name="save_" + str(endpoint), limit=240)
        client("status", name="memory_at_" + str(endpoint), limit=60)
        if first != plan["resume_step"]:
            client("load_snapshot", "--checkpoint", args.baseline_checkpoint,
                   "--manifest-sha256", plan["comparison_manifest_sha256"], "--snapshot", str(first), limit=240)
            evaluate(first, confirmation=True)
        evaluate(endpoint, confirmation=True)
        if seen != {condition_id(c) for c in matrix}:
            raise RuntimeError("Incomplete continuation comparison")
        write_json(root / "provenance/study_complete.json", {
            "complete": True, "plan_sha256": digest(plan), "completed_conditions": sorted(seen),
            "optimizer_steps_added": endpoint - plan["resume_step"], "final_optimizer_step": endpoint,
            "reused_conditions": [reference["condition"]] if reference else [],
            "comparison_optimizer_steps_added": endpoint - first})
        print("Continuation and paired evaluation complete; exit to release GPU.", flush=True)
        return

    # Runtime guards use training layouts and never enter benchmark statistics.
    # A single context first warms rendering; four processes then check concurrency.
    for label, tasks, workers in [("single", [0], 1), ("parallel", [0, 1, 2, 3], 4)]:
        run("runtime_pilot_" + label, [sys.executable, "-m", "frequency_vla.opsd_evaluate",
            "--port", str(args.port), "--results-dir", str(root / ("runtime_pilot_" + label)),
            "--workers", str(workers), "--suite", "libero_10", "--horizon", "5",
            "--episodes", "1", "--seed", "17", "--initial-state-start", "10",
            "--task-ids"] + [str(t) for t in tasks], 180)
    client("diagnostic", limit=420)
    client("resume", "--checkpoint", args.parent_checkpoint, limit=300)
    client("status", name="memory_after_resume", limit=60)
    evaluate(0)
    evaluate(100)
    for milestone in [300, 500]:
        client("train", "--end-step", str(milestone), name="train_to_" + str(milestone), limit=1200)
        client("save", name="save_" + str(milestone), limit=240)
        client("status", name="memory_at_" + str(milestone), limit=60)
        evaluate(milestone)
    for step in [0, 100, 500]:
        evaluate(step, confirmation=True)
    if seen != {condition_id(c) for c in matrix}:
        raise RuntimeError("Incomplete research matrix")
    write_json(root / "provenance/study_complete.json", {"complete": True, "plan_sha256": digest(plan),
        "completed_conditions": sorted(seen), "optimizer_steps_added": 400, "final_optimizer_step": 500})
    print("All planned conditions completed; exit to release GPU.", flush=True)


if __name__ == "__main__":
    main()
