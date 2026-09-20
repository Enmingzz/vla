"""Extend the paired H=5 retention check to all 50 official layouts per task."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .analysis import pair_key, validate_records, write_csv
from .egl_pipeline import summarize_condition
from .logging_utils import append_record, digest, file_digest, git_commit, write_json
from .study_analysis import paired_change


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(folder):
    return [json.loads(line) for p in sorted(Path(folder).glob("*.jsonl"))
            for line in p.read_text().splitlines() if line.strip()]


def condition_folder(base, step):
    return Path(base) / "evaluations" / ("step_" + str(step)) / "raw/smoke/libero_10/seed_27/H_5"


def normalize_rows(rows, cached):
    """Use the physical layout index when joining independently numbered blocks."""
    return [dict(r, source_episode_index=r["episode_index"], episode_index=r["initial_state_index"],
                 reused_episode=cached) for r in rows]


def subset_rows(rows, seen):
    return {"heldout_400": [r for r in rows if (r["task_id"], r["initial_state_index"]) not in seen],
            "all_500": rows,
            "training_layouts_100": [r for r in rows if (r["task_id"], r["initial_state_index"]) in seen]}


def evaluation_core(server, start):
    spec = copy.deepcopy(server["evaluation_spec"])
    if digest(spec) != server["evaluation_fingerprint"] or spec.pop("initial_state_start") != start:
        raise ValueError("Evaluator fingerprint or declared block offset changed")
    return spec


def prepare(root, reference, state_catalog):
    if root.exists():
        raise ValueError("Use a fresh output directory")
    final = read_json(reference / "provenance/final_checks.json")
    if not final["complete"] or final["formal_episodes"] != 200:
        raise ValueError("Require both completed P=50/H=5 100-episode conditions")
    for name, sha in final["artifacts"].items():
        if file_digest(reference / name) != sha:
            raise ValueError("Cached result changed: " + name)
    old_plan = read_json(reference / "provenance/study_plan.json")
    checkpoint = old_plan["checkpoint"]
    manifest = read_json(Path(checkpoint["path"]) / "training_manifest.json")
    if digest(manifest) != checkpoint["manifest_sha256"] or manifest["step"] != 1500:
        raise ValueError("Wrong trained checkpoint")
    prior_training = Path(old_plan["source_results"])
    split = read_json(prior_training / "provenance/split_audit.json")
    catalog_rows = read_rows(state_catalog)
    expected = {(t, i) for t in range(10) for i in range(50)}
    if len(catalog_rows) != 500 or {(r["task_id"], r["episode_index"]) for r in catalog_rows} != expected:
        raise ValueError("Require all 50 official initial-state hashes per task")
    states = {(r["task_id"], r["episode_index"]): r["initial_state_sha256"] for r in catalog_rows}
    for r in split["training_initial_states"] + split["evaluation_initial_states"]:
        if states[r["task_id"], r["initial_state_index"]] != r["initial_state_sha256"]:
            raise ValueError("State catalog differs from the fresh EGL experiment")
    seen = {(r["task_id"], r["initial_state_index"]) for r in split["training_initial_states"]}
    if seen != {(t, i) for t in range(10) for i in range(10, 20)}:
        raise ValueError("Unexpected OPSD training-state pool")
    reference_servers = set()
    for step in [0, 1500]:
        rows = read_rows(condition_folder(reference, step))
        validate_records(rows)
        if len(rows) != 100 or {(r["seed"], r["task_id"], r["initial_state_index"]) for r in rows} != {
                (27, t, i) for t in range(10) for i in range(30, 40)}:
            raise ValueError("Cached coverage differs from states 30–39, seed 27")
        if any(r["prediction_horizon"] != 50 or r["replan_steps"] != 5 for r in rows):
            raise ValueError("Cannot reuse a different P or H")
        for p in condition_folder(reference, step).glob("*.manifest.json"):
            m = read_json(p)
            if m["status"] != "complete":
                raise ValueError("Incomplete cached task")
            reference_servers.add(m["server"]["server_instance_id"])
    if len(reference_servers) != 1:
        raise ValueError("Cached conditions must come from the completed paired run")
    plan = dict(suite="libero_10", prediction_horizon=50, horizon=5, flow_steps=10,
        renderer="egl", seed=27, workers=4, steps=[0, 1500], checkpoint=checkpoint,
        reference_results=str(reference), cached_initial_state_start=30,
        new_initial_state_starts=[0, 10, 20, 40], episodes_per_block_per_task=10,
        total_episodes_per_condition=500, cached_episodes=200, new_episodes=800,
        primary_subset="heldout_400", training_initial_state_range=[10, 20],
        condition_timeout_seconds=1200, bootstrap_replicates=10000, analysis_seed=20260920,
        optimizer_updates_added=0, original_git_commit=git_commit(Path(__file__).resolve().parents[2]))
    paths = [reference / "provenance/final_checks.json", reference / "provenance/study_plan.json",
             prior_training / "provenance/split_audit.json"]
    for step in plan["steps"]:
        paths += sorted(condition_folder(reference, step).glob("*.json*"))
    paths += sorted(state_catalog.glob("*.jsonl"))
    project = Path(__file__).resolve().parents[2]
    code = ["src/frequency_vla/h5_extension.py", "scripts/fir_h5_extension.sh", "scripts/serve_policy.sh",
            "src/frequency_vla/server.py", "src/frequency_vla/frozen_comparison.py",
            "src/frequency_vla/evaluator.py", "src/frequency_vla/opsd_evaluate.py",
            "src/frequency_vla/config.py", "src/frequency_vla/runner.py", "configs/prediction50_egl.yaml"]
    paths += [project / p for p in code]
    (root / "logs").mkdir(parents=True)
    write_json(root / "provenance/study_plan.json", plan)
    write_json(root / "provenance/preflight.json", dict(passed=True, plan_sha256=digest(plan),
        input_files={str(p): file_digest(p) for p in paths},
        cached_server_instance=next(iter(reference_servers)),
        initial_states=[dict(task_id=t, initial_state_index=i, initial_state_sha256=states[t, i],
                            seen_in_added_training=(t, i) in seen) for t, i in sorted(states)],
        state_catalog_usage="Only initial-state identities; no native-P=10 outcomes enter this study"))
    print(json.dumps(plan, indent=2), flush=True)


def verify(root):
    plan = read_json(root / "provenance/study_plan.json")
    audit = read_json(root / "provenance/preflight.json")
    if not audit["passed"] or audit["plan_sha256"] != digest(plan):
        raise ValueError("Evaluation plan changed")
    for name, sha in audit["input_files"].items():
        if file_digest(name) != sha:
            raise ValueError("Frozen input changed: " + name)
    manifest = read_json(Path(plan["checkpoint"]["path"]) / "training_manifest.json")
    if digest(manifest) != plan["checkpoint"]["manifest_sha256"]:
        raise ValueError("Trained checkpoint changed")
    return plan, audit


def run(root, port):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    plan, _ = verify(root)
    setup = read_json(root / "provenance/frozen_comparison.json")
    if setup["training_operations_available"] or setup["trained_checkpoint"] != plan["checkpoint"]:
        raise ValueError("Wrong checkpoint or optimizer-enabled server")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise ValueError("Require EGL without renderer fallback")
    completed = []
    for start in plan["new_initial_state_starts"]:
        for step in plan["steps"]:
            client = WebsocketClientPolicy("127.0.0.1", port)
            try:
                selected = client.infer({"_frozen_comparison": {"operation": "select", "step": step}})
                if selected["step"] != step:
                    raise ValueError("Snapshot selection failed")
            finally:
                client._ws.close()
            current = read_json(root / "provenance" / ("server_step_"+str(step)+".json"))
            cached = read_json(condition_folder(Path(plan["reference_results"]), step) / "tasks_0.manifest.json")["server"]
            if current["experiment_spec"] != cached["experiment_spec"]:
                raise ValueError("Inference differs from the cached checkpoint; abort before evaluating")
            base = root / "new_evaluations" / ("states_{:02d}_{:02d}".format(start, start+9))
            output = base / "evaluations" / ("step_" + str(step))
            name = "states{}_step{}_H5".format(start, step)
            command = [sys.executable, "-m", "frequency_vla.opsd_evaluate", "--port", str(port),
                "--results-dir", str(output), "--workers", "4", "--suite", "libero_10",
                "--horizon", "5", "--episodes", "10", "--seed", "27", "--initial-state-start", str(start)]
            append_record(root / "stages.jsonl", dict(stage=name, event="start", command=command, unix_time=time.time()))
            print("Starting " + name, flush=True)
            began = time.monotonic()
            with (root / "logs" / (name + ".log")).open("x") as log:
                subprocess.run(["timeout", "--kill-after=10s", str(plan["condition_timeout_seconds"])] + command,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = read_rows(condition_folder(base, step))
            validate_records(rows)
            if len(rows) != 100 or {(r["seed"], r["task_id"], r["initial_state_index"]) for r in rows} != {
                    (27, t, i) for t in range(10) for i in range(start, start+10)}:
                raise ValueError("Incomplete new block")
            completed.append(name)
            append_record(root / "stages.jsonl", dict(stage=name, event="complete", seconds=time.monotonic()-began,
                unix_time=time.time(), episodes=100, successes=sum(r["success"] for r in rows), workers=4))
            print("Completed {}: {}/100".format(name, sum(r["success"] for r in rows)), flush=True)
    write_json(root / "provenance/study_complete.json", dict(complete=True, plan_sha256=digest(plan),
        new_episodes=800, reused_episodes=200, optimizer_updates_added=0, completed_blocks=completed))


def analyze(root):
    plan, audit = verify(root)
    done = read_json(root / "provenance/study_complete.json")
    if not done["complete"] or done["plan_sha256"] != digest(plan) or done["optimizer_updates_added"]:
        raise ValueError("Incomplete evaluation or unexpected training")
    states = {(r["task_id"], r["initial_state_index"]): r for r in audit["initial_states"]}
    seen = {k for k, r in states.items() if r["seen_in_added_training"]}
    reference = Path(plan["reference_results"])
    blocks = [(reference, 30, True)] + [(root / "new_evaluations" / "states_{:02d}_{:02d}".format(i, i+9), i, False)
                                      for i in plan["new_initial_state_starts"]]
    data, core, evaluator_core, current_servers, cached_servers = {}, None, None, set(), set()
    for step in plan["steps"]:
        joined = []
        for base, start, cached in blocks:
            folder = condition_folder(base, step)
            rows = read_rows(folder)
            validate_records(rows)
            if len(rows) != 100 or {(r["seed"], r["task_id"], r["initial_state_index"]) for r in rows} != {
                    (27, t, i) for t in range(10) for i in range(start, start+10)}:
                raise ValueError("Incorrect block coverage")
            manifests = [read_json(p) for p in folder.glob("*.manifest.json")]
            if len(manifests) != 10 or any(m["status"] != "complete" for m in manifests):
                raise ValueError("Incomplete task manifests")
            for m in manifests:
                server = m["server"]
                spec = copy.deepcopy(server["experiment_spec"])
                if digest(spec) != server["inference_fingerprint"] or any(r["inference_fingerprint"] != server["inference_fingerprint"] for r in rows):
                    raise ValueError("Inconsistent inference provenance")
                snapshot = spec.pop("frozen_comparison")
                if snapshot["optimizer_step"] != step or snapshot["checkpoint"] != (plan["checkpoint"] if step else None):
                    raise ValueError("Wrong frozen snapshot")
                if spec["prediction_horizon"] != 50 or spec["flow_steps"] != 10 or server["evaluation_spec"]["rendering"]["backend"] != "egl":
                    raise ValueError("Changed inference or rendering protocol")
                if core is not None and spec != core:
                    raise ValueError("Non-parameter inference settings changed across cached/new blocks")
                core = spec
                current_evaluator = evaluation_core(server, start)
                if evaluator_core is not None and current_evaluator != evaluator_core:
                    raise ValueError("Evaluator changed beyond the declared initial-state block offset")
                if any(r["evaluation_fingerprint"] != server["evaluation_fingerprint"] for r in rows):
                    raise ValueError("Episode evaluator differs from its block manifest")
                evaluator_core = current_evaluator
                (cached_servers if cached else current_servers).add(server["server_instance_id"])
            for r in rows:
                key = r["task_id"], r["initial_state_index"]
                if r["initial_state_index"] != start+r["episode_index"] or r["initial_state_sha256"] != states[key]["initial_state_sha256"]:
                    raise ValueError("Wrong ordered initial state")
                if r["prediction_horizon"] != 50 or r["replan_steps"] != 5 or r["task_suite"] != "libero_10":
                    raise ValueError("Wrong evaluation condition")
                video = base / "evaluations" / ("step_"+str(step)) / r["video_path"]
                if not video.is_file() or not video.stat().st_size:
                    raise ValueError("Missing video: " + str(video))
            joined.extend(normalize_rows(rows, cached))
        # Each raw block was validated above. Evaluator fingerprints intentionally
        # retain their declared offsets; only that metadata field may differ.
        if len({r["inference_fingerprint"] for r in joined}) != 1:
            raise ValueError("Checkpoint inference changed across cached/new blocks")
        if len(joined) != 500 or {pair_key(r) for r in joined} != {(27, t, i) for t in range(10) for i in range(50)}:
            raise ValueError("Combined condition is not exactly 500 distinct ordered layouts")
        data[step] = subset_rows(joined, seen)
    if cached_servers != {audit["cached_server_instance"]} or len(current_servers) != 1:
        raise ValueError("Unexpected server restart or cached reference")
    summaries, comparisons, per_task, episodes = [], [], [], []
    for subset, count in [("heldout_400", 400), ("all_500", 500), ("training_layouts_100", 100)]:
        for step in plan["steps"]:
            rows = data[step][subset]
            if len(rows) != count:
                raise ValueError("Incorrect training/held-out partition")
            summaries.append(dict(subset=subset, **summarize_condition(dict(step=step, horizon=5), rows)))
            for task in range(10):
                tr = [r for r in rows if r["task_id"] == task]
                per_task.append(dict(subset=subset, task_id=task, task_description=tr[0]["task_description"],
                    **summarize_condition(dict(step=step, horizon=5), tr)))
        comparisons.append(dict(subset=subset, primary=subset==plan["primary_subset"],
            **paired_change(data[1500][subset], data[0][subset], plan)))
    for step in plan["steps"]:
        episodes.extend(dict(optimizer_step=step, seen_in_added_training=(r["task_id"],r["initial_state_index"]) in seen, **r)
                        for r in data[step]["all_500"])
    for name, rows in [("conditions.csv", summaries), ("comparisons.csv", comparisons),
                       ("per_task.csv", per_task), ("episodes.csv", episodes)]:
        write_csv(root / "aggregated" / name, rows)
    primary = comparisons[0]
    write_json(root / "aggregated/validation.json", dict(complete=True, episodes=1000, new_episodes=800,
        reused_episodes=200, optimizer_updates_added=0, cached_server_instances=sorted(cached_servers),
        current_server_instances=sorted(current_servers), primary=primary,
        source_local_episode_indices_preserved=True, paired_states_observations_rng_evaluator=True,
        inference_settings_equal_across_cached_and_new_blocks=True,
        evaluation_settings_equal_except_declared_initial_state_start=True, video_files_present=1000))
    lines = ["# H=5 retention over 500 episodes per model", "",
        "P=50, H=5, EGL, four workers, 10 flow steps, LIBERO-10, seed 27, official layout indices 0–49. "
        "The prior 100 episodes per model (states 30–39) were reused; 400 per model were added. No training was performed.", "",
        "The primary retention comparison excludes OPSD training layouts 10–19: 400 paired episodes per model. "
        "The full 500 and the 100 training-layout episodes are descriptive secondary results. "
        "The fixed tasks and layouts have been inspected previously; this is not a blind or unseen-task test.", "",
        "| Subset | Model | Success | Mean actions | Mean calls | Mean seconds |",
        "|---|---|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append("| {} | {} | {}/{} ({:.1%}) | {:.2f} | {:.2f} | {:.2f} |".format(s["subset"],
            "Original" if s["step"]==0 else "OPSD step 1500", s["total_successes"], s["total_episodes"],
            s["success_rate"], s["mean_episode_length"], s["mean_policy_calls"], s["mean_wall_clock_seconds"]))
    lines += ["", "Primary held-out change (trained minus original): **{:+.2f} pp**, paired 95% bootstrap CI "
        "**[{:+.2f}, {:+.2f}] pp**, exact two-sided McNemar **p={:.6g}**; {} recoveries and {} regressions.".format(
            100*primary["success_change"], 100*primary["ci95_low"], 100*primary["ci95_high"], primary["p_exact"],
            primary["recovered_episodes"], primary["regressed_episodes"]), "",
        "A non-significant difference does not establish equivalence or non-inferiority. "
        "One primary contrast was fixed before the extension; secondary and task-level analyses are exploratory. "
        "Cached and new episodes span two server instances with identical recorded inference/evaluator settings; "
        "there is no additional repeated baseline. Timing pools matched four-worker runs and includes shared-server "
        "waiting, but is not isolated inference latency. The native checkpoint configuration uses P=10; this study remains at the authorized P=50.", ""]
    (root / "FINDINGS.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "verify", "run", "analyze"])
    p.add_argument("--results-dir", required=True)
    p.add_argument("--reference-results")
    p.add_argument("--state-catalog")
    p.add_argument("--port", type=int)
    args = p.parse_args()
    root = Path(args.results_dir).resolve()
    if args.stage == "prepare":
        if not args.reference_results or not args.state_catalog:
            p.error("prepare needs --reference-results and --state-catalog")
        prepare(root, Path(args.reference_results).resolve(), Path(args.state_catalog).resolve())
    elif args.stage == "verify":
        verify(root)
    elif args.stage == "run":
        if args.port is None:
            p.error("run requires --port")
        run(root, args.port)
    else:
        analyze(root)


if __name__ == "__main__":
    main()
