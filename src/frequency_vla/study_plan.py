"""Predeclared round-two matrix and initial-state leakage audit (no GPU needed)."""
import argparse
import json
import os
from pathlib import Path
import sys

from .config import load_config
from .evaluator import array_hash
from .logging_utils import digest, write_json
from .opsd_protocol import validate_training_config


def conditions(plan):
    result = []
    for split in ["screen", "transfer", "confirmation"]:
        if split not in plan:
            continue
        block = plan[split]
        suites = block.get("suites", [block.get("suite")])
        for step, horizons in block["horizons"].items():
            for suite in suites:
                for horizon in horizons:
                    result.append({"split": split, "step": int(step), "suite": suite, "horizon": int(horizon),
                        "episodes_per_task": block["episodes_per_task"], "seed": block["seed"],
                        "initial_state_start": block["initial_state_start"]})
    return result


def condition_id(condition):
    return "{split}_{suite}_step{step}_H{horizon}".format(**condition)


def audit(plan, training_config, parent_results, parent_checkpoint, openpi_dir):
    validate_training_config(training_config)
    if isinstance(parent_results, (str, Path)):
        parent_results = [parent_results]
    parent_checkpoint = Path(parent_checkpoint)
    manifest = json.loads((parent_checkpoint / "training_manifest.json").read_text())
    if digest(manifest) != plan["resume_manifest_sha256"] or manifest["step"] != plan["resume_step"]:
        raise ValueError("Parent checkpoint differs from the predeclared continuation")
    from .logging_utils import file_digest
    parent_files = [Path(p) / "rollouts.jsonl" for p in parent_results]
    rows = [json.loads(line) for p in parent_files for line in p.read_text().splitlines() if line]
    if [r["optimizer_step"] for r in rows] != list(range(1, plan["resume_step"] + 1)):
        raise ValueError("Parent training provenance is incomplete")
    prior_hashes = {s["initial_state_sha256"] for r in rows for s in r["initial_states"]}
    if plan.get("continuation_only"):
        first = plan.get("comparison_step", plan["resume_step"])
        final = first + 500
        if first not in [500, 1000] or not first <= plan["resume_step"] < final or training_config["optimizer_steps"] != final or plan["milestones"] != [first, final]:
            raise ValueError("The authorized continuation must contain exactly 500 additional updates")
        if {c["step"] for c in conditions(plan)} != {first, final} or any(c["horizon"] != 20 or c["suite"] != "libero_10" for c in conditions(plan)):
            raise ValueError("This continuation compares the two H=20 LIBERO-10 snapshots")
    reference = None
    if plan.get("cached_reference"):
        from .cached_reference import audit_reference
        reference = audit_reference(plan)
    from .continuation_records import audit_prior_segments
    segments = audit_prior_segments(plan)
    sys.path.insert(0, str(Path(openpi_dir) / "third_party/libero"))
    from libero.libero import benchmark
    suites = {name: benchmark.get_benchmark_dict()[name]() for name in
              ["libero_10", "libero_spatial", "libero_object", "libero_goal"]}
    new_hashes = {array_hash(state) for task in range(10) for state in
        suites["libero_10"].get_task_init_states(task)[training_config["train_initial_state_start"]:training_config["train_initial_state_stop"]]}
    selections = {}
    for condition in conditions(plan):
        start, count = condition["initial_state_start"], condition["episodes_per_task"]
        key = (condition["split"], condition["suite"])
        if key in selections:
            continue
        selected = []
        for task in range(10):
            states = suites[condition["suite"]].get_task_init_states(task)
            if start < 0 or start + count > len(states):
                raise ValueError("Evaluation range exceeds official initial-state pool")
            for index in range(start, start + count):
                sha = array_hash(states[index])
                if sha in prior_hashes or sha in new_hashes:
                    raise ValueError("Training/evaluation state collision: " + str((key, task, index)))
                selected.append({"task_id": task, "initial_state_index": index, "initial_state_sha256": sha})
        selections[key] = selected
    screen_hashes = {r["initial_state_sha256"] for (split, _), rows in selections.items() if split != "confirmation" for r in rows}
    confirmation_hashes = {r["initial_state_sha256"] for (split, _), rows in selections.items() if split == "confirmation" for r in rows}
    if screen_hashes & confirmation_hashes:
        raise ValueError("Screen and confirmation initial states overlap")
    return {"plan_sha256": digest(plan), "parent_manifest_sha256": digest(manifest),
        "cached_reference": reference,
        "continuation_segments": segments,
        "prior_rollout_sources": [{"path": str(p.resolve()), "sha256": file_digest(p)} for p in parent_files],
        "prior_training_distinct_states": len(prior_hashes), "new_training_pool_distinct_states": len(new_hashes),
        "old_and_new_training_disjoint_from_all_evaluations": True, "screen_confirmation_disjoint": True,
        "planned_evaluation_episodes": sum(c["episodes_per_task"] * 10 for c in conditions(plan)),
        "evaluation_initial_states": {"/".join(key): value for key, value in selections.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--training-config", required=True)
    p.add_argument("--parent-results", nargs="+", required=True)
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = audit(load_config(args.plan), load_config(args.training_config), args.parent_results,
                   args.parent_checkpoint, os.environ["OPENPI_DIR"])
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k != "evaluation_initial_states"})


if __name__ == "__main__":
    main()
