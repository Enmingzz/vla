"""Execute the frozen research matrix; task outcomes never change the schedule."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import load_config
from .logging_utils import append_record, digest, write_json
from .study_plan import condition_id, conditions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--training-config", required=True)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--checkpoint-root", required=True)
    args = p.parse_args()
    root, plan = Path(args.results_dir), load_config(args.plan)
    write_json(root / "provenance/study_plan.json", plan)
    audit = json.loads((root / "provenance/split_audit.json").read_text())
    if audit["plan_sha256"] != digest(plan):
        raise ValueError("Plan changed after initial-state audit")
    matrix = conditions(plan)
    seen = set()

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
                "--port", str(args.port), "--results-dir", str(output), "--workers", "4",
                "--suite", c["suite"], "--horizon", str(c["horizon"]),
                "--episodes", str(c["episodes_per_task"]), "--seed", str(c["seed"]),
                "--initial-state-start", str(c["initial_state_start"])], 1200)
            seen.add(identifier)

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
