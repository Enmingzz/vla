"""Evaluate LIBERO-10 OPSD transfer to all LIBERO-90 tasks without training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import load_config
from .evaluator import array_hash
from .logging_utils import append_record, digest, file_digest, write_json


def normalize_instruction(text):
    return " ".join(text.lower().strip().split())


def conditions(plan):
    return [dict(step=int(step), horizon=int(h), suite=plan["suite"], seed=plan["seed"],
                 episodes_per_task=plan["episodes_per_task"], initial_state_start=plan["initial_state_start"])
            for step, horizons in plan["snapshots"].items() for h in horizons]


def condition_id(c):
    return "step{step}_H{horizon}".format(**c)


def validate_plan(plan):
    if plan["suite"] != "libero_90" or plan["all_tasks"] != 90 or plan["maximum_concurrent_gpus"] != 1:
        raise ValueError("This frozen evaluation covers all 90 tasks on at most one GPU")
    if {(c["step"], c["horizon"]) for c in conditions(plan)} != {(0, 5), (0, 20), (500, 20)}:
        raise ValueError("The original H=5/H=20 and step-500 H=20 controls are required")
    if not 0 <= plan["initial_state_start"] < plan["initial_state_start"]+plan["episodes_per_task"] <= 50:
        raise ValueError("Invalid ordered initial-state range")


def audit(args, plan):
    checkpoint = Path(args.checkpoint)
    manifest = json.loads((checkpoint / "training_manifest.json").read_text())
    if digest(manifest) != plan["checkpoint_manifest_sha256"] or manifest["step"] != 500:
        raise ValueError("The preselected trained checkpoint changed")
    if manifest["resumed_from"]["manifest_sha256"] != plan["parent_checkpoint_manifest_sha256"]:
        raise ValueError("Incorrect step-100 checkpoint ancestry")
    if manifest["base_object_manifest_sha256"] != plan["base_checkpoint_manifest_sha256"]:
        raise ValueError("Incorrect official base checkpoint")
    sys.path.insert(0, str(Path(os.environ["OPENPI_DIR"]) / "third_party/libero"))
    from libero.libero import benchmark
    source = benchmark.get_benchmark_dict()["libero_10"]()
    target = benchmark.get_benchmark_dict()["libero_90"]()
    if source.n_tasks != 10 or target.n_tasks != 90:
        raise ValueError("Unexpected benchmark task count")
    source_names = set(source.get_task_names())
    instructions = {normalize_instruction(source.get_task(t).language) for t in range(10)}
    source_states = {t: source.get_task_init_states(t) for t in range(10)}
    rolls, sources = [], []
    for directory in args.prior_results:
        p = Path(directory) / "rollouts.jsonl"
        rolls.extend(json.loads(line) for line in p.read_text().splitlines() if line)
        sources.append(dict(path=str(p.resolve()), sha256=file_digest(p)))
    if [r["optimizer_step"] for r in rolls] != list(range(1, 501)):
        raise ValueError("Need complete actual rollout provenance for all 500 training updates")
    train_hashes = set()
    for row in rolls:
        if row["diagnostic"]:
            raise ValueError("Diagnostic updates cannot enter training provenance")
        for s in row["initial_states"]:
            t, i = s["task_id"], s["initial_state_index"]
            if s["task_description"] != source.get_task(t).language or s["initial_state_sha256"] != array_hash(source_states[t][i]):
                raise ValueError("Recorded training state/task differs from LIBERO-10")
            train_hashes.add(s["initial_state_sha256"])
    public_metadata = Path(args.results_dir) / "provenance/base_dataset_tasks.jsonl"
    public_instructions = set()
    if public_metadata.exists():
        public_instructions = {normalize_instruction(json.loads(line)["task"]) for line in public_metadata.read_text().splitlines() if line}
    tasks, states = [], []
    for task_id in range(90):
        task = target.get_task(task_id)
        name = task.name
        if name in source_names:
            raise ValueError("LIBERO-90 contains an identical LIBERO-10 task name")
        bddl_sha = file_digest(target.get_task_bddl_file_path(task_id))
        tasks.append(dict(task_id=task_id, name=name, description=task.language, bddl_sha256=bddl_sha,
            instruction_seen_in_added_opsd=normalize_instruction(task.language) in instructions,
            instruction_in_current_public_base_dataset=normalize_instruction(task.language) in public_instructions))
        initial = target.get_task_init_states(task_id)
        for i in range(plan["initial_state_start"], plan["initial_state_start"]+plan["episodes_per_task"]):
            sha = array_hash(initial[i])
            if sha in train_hashes:
                raise ValueError("Evaluation exactly duplicates an actual OPSD training state")
            states.append(dict(task_id=task_id, initial_state_index=i, initial_state_sha256=sha))
    novel = [t["task_id"] for t in tasks if not t["instruction_seen_in_added_opsd"]]
    return dict(plan_sha256=digest(plan), checkpoint_manifest_sha256=digest(manifest),
        train_suite="libero_10", test_suite="libero_90", added_optimizer_updates=500,
        actual_training_distinct_layouts=len(train_hashes), prior_training_rollout_files=sources,
        distinct_task_names=True, actual_training_evaluation_state_hashes_disjoint=True,
        primary_task_ids=novel, overlapping_instruction_task_ids=[t["task_id"] for t in tasks if t["instruction_seen_in_added_opsd"]],
        tasks=tasks, evaluation_initial_states=states, total_evaluation_episodes=len(states)*3,
        added_optimizer_updates_in_this_study=0,
        base_exposure_caveat="Current public task metadata is not the historical fine-tuning revision or a complete pretraining audit.")


def execute(args, plan):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    root = Path(args.results_dir)
    audit_result = json.loads((root / "provenance/split_audit.json").read_text())
    if audit_result["plan_sha256"] != digest(plan):
        raise ValueError("Plan changed after the CPU split audit")
    setup = json.loads((root / "provenance/frozen_comparison.json").read_text())
    if setup["trained_checkpoint"]["manifest_sha256"] != plan["checkpoint_manifest_sha256"] or setup["training_operations_available"]:
        raise ValueError("Wrong checkpoint or training-enabled server")
    write_json(root / "provenance/study_plan.json", plan)

    def run(name, command, limit):
        started = time.monotonic()
        append_record(root / "stages.jsonl", dict(stage=name, event="start", command=command, unix_time=time.time()))
        print("Starting {} ({}s bound)".format(name, limit), flush=True)
        with (root / "logs" / (name + ".log")).open("x") as log:
            subprocess.run(["timeout", "--kill-after=10s", str(limit)] + command,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        append_record(root / "stages.jsonl", dict(stage=name, event="complete", seconds=time.monotonic()-started, unix_time=time.time()))

    def evaluate(name, step, h, episodes, start, ids=None, workers=4, limit=1500):
        client = WebsocketClientPolicy("127.0.0.1", args.port)
        try:
            selection = client.infer({"_frozen_comparison": {"operation": "select", "step": step}})
            if selection["step"] != step:
                raise ValueError("Snapshot selection failed")
        finally:
            client._ws.close()
        output = root / name if name.startswith("runtime_pilot_") else root / "evaluations" / ("step_"+str(step))
        command = [sys.executable, "-m", "frequency_vla.opsd_evaluate", "--port", str(args.port),
            "--results-dir", str(output), "--workers", str(workers), "--suite", plan["suite"],
            "--horizon", str(h), "--episodes", str(episodes), "--seed", str(plan["seed"]),
            "--initial-state-start", str(start)]
        if ids is not None:
            command += ["--task-ids"] + [str(t) for t in ids]
        run(name, command, limit)

    evaluate("runtime_pilot_single", 0, 5, 1, 0, [0], workers=1, limit=180)
    evaluate("runtime_pilot_parallel", 0, 5, 1, 0, [0, 30, 60, 89], workers=4, limit=180)
    completed = []
    for c in conditions(plan):
        name = condition_id(c)
        evaluate(name, c["step"], c["horizon"], c["episodes_per_task"], c["initial_state_start"])
        completed.append(name)
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    try:
        write_json(root / "provenance/final_server_status.json", client.infer({"_frozen_comparison": {"operation": "status"}}))
    finally:
        client._ws.close()
    write_json(root / "provenance/study_complete.json", dict(complete=True, plan_sha256=digest(plan),
        completed_conditions=completed, optimizer_updates_added=0))
    print("LIBERO-90 comparison complete; release GPU.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["audit", "run"])
    p.add_argument("--plan", default="configs/libero90_transfer.yaml")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--prior-results", nargs="+")
    p.add_argument("--port", type=int)
    args = p.parse_args()
    plan = load_config(args.plan)
    validate_plan(plan)
    if args.stage == "audit":
        if not args.checkpoint or not args.prior_results:
            p.error("audit requires --checkpoint and --prior-results")
        result = audit(args, plan)
        write_json(Path(args.results_dir) / "provenance/split_audit.json", result)
        print(json.dumps({k:v for k,v in result.items() if k not in ["tasks", "evaluation_initial_states"]}, indent=2))
    else:
        if args.port is None:
            p.error("run requires --port")
        execute(args, plan)


if __name__ == "__main__":
    main()
