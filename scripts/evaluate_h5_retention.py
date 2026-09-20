"""Evaluate the completed EGL step-1500 student and original checkpoint at H=5."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from frequency_vla.analysis import pair_key, validate_records, write_csv
from frequency_vla.egl_pipeline import summarize_condition
from frequency_vla.logging_utils import append_record, digest, file_digest, git_commit, write_json
from frequency_vla.study_analysis import paired_change


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(folder):
    return [json.loads(line) for p in sorted(Path(folder).glob("*.jsonl"))
            for line in p.read_text().splitlines() if line.strip()]


def folder(root, step):
    return Path(root) / "evaluations" / ("step_" + str(step)) / "raw/smoke/libero_10/seed_27/H_5"


def prepare(root, source):
    if root.exists():
        raise ValueError("Choose a fresh result directory; preserve previous runs")
    final = read_json(source / "provenance/final_checks.json")
    if not final["complete"] or final["optimizer_updates_added"] != 1500:
        raise ValueError("Require the completed fresh EGL 1500-update experiment")
    for name, sha in final["artifacts"].items():
        if file_digest(source / name) != sha:
            raise ValueError("Archived experiment changed: " + name)
    checkpoint = read_json(source / "provenance/step_1500.json")
    manifest = read_json(Path(checkpoint["path"]) / "training_manifest.json")
    if digest(manifest) != checkpoint["manifest_sha256"] or manifest["step"] != 1500:
        raise ValueError("Wrong trained checkpoint")
    if manifest["config"]["prediction_horizon"] != 50 or manifest["config"]["flow_steps"] != 10:
        raise ValueError("Checkpoint inference configuration changed")
    runtime = read_json(source / "provenance/training_runtime.json")
    if runtime["renderer"]["backend"] != "egl" or runtime["official_start_step"] != 0:
        raise ValueError("Require training entirely under EGL from the original checkpoint")
    baseline = read_rows(folder(source, 0))
    validate_records(baseline)
    expected = {(27, t, e) for t in range(10) for e in range(10)}
    if len(baseline) != 100 or {pair_key(r) for r in baseline} != expected:
        raise ValueError("Incomplete historical H=5 reference")
    plan = dict(suite="libero_10", prediction_horizon=50, horizon=5, flow_steps=10,
        renderer="egl", seed=27, episodes_per_task=10, initial_state_start=30, workers=4,
        steps=[0, 1500], checkpoint=checkpoint, source_results=str(source),
        optimizer_updates_added=0, formal_episodes=200, condition_timeout_seconds=900,
        bootstrap_replicates=10000, analysis_seed=20260920,
        primary_comparison="step1500_H5_minus_original_H5_on_one_server",
        historical_H5_successes=sum(r["success"] for r in baseline),
        original_git_commit=git_commit(Path(__file__).resolve().parents[1]))
    source_files = ["provenance/final_checks.json", "provenance/step_1500.json",
                    "provenance/split_audit.json"]
    source_files += [str(p.relative_to(source)) for p in sorted(folder(source, 0).glob("*.json*"))]
    project = Path(__file__).resolve().parents[1]
    implementation = ["scripts/evaluate_h5_retention.py", "scripts/fir_h5_retention.sh",
        "scripts/serve_policy.sh", "configs/prediction50_egl.yaml",
        "src/frequency_vla/server.py", "src/frequency_vla/frozen_comparison.py",
        "src/frequency_vla/evaluator.py", "src/frequency_vla/opsd_evaluate.py",
        "src/frequency_vla/config.py", "src/frequency_vla/runner.py"]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    write_json(root / "provenance/study_plan.json", plan)
    write_json(root / "provenance/preflight.json", dict(passed=True, plan_sha256=digest(plan),
        sources={str(project / p): file_digest(project / p) for p in implementation},
        historical_files={str(source / p): file_digest(source / p) for p in source_files},
        evaluation_initial_states=read_json(source / "provenance/split_audit.json")["evaluation_initial_states"]))
    print(json.dumps(plan, indent=2), flush=True)


def verify(root):
    plan = read_json(root / "provenance/study_plan.json")
    audit = read_json(root / "provenance/preflight.json")
    if not audit["passed"] or digest(plan) != audit["plan_sha256"]:
        raise ValueError("Evaluation plan changed after preparation")
    for group in ["sources", "historical_files"]:
        for name, sha in audit[group].items():
            if file_digest(name) != sha:
                raise ValueError("Frozen input changed: " + name)
    manifest = read_json(Path(plan["checkpoint"]["path"]) / "training_manifest.json")
    if digest(manifest) != plan["checkpoint"]["manifest_sha256"]:
        raise ValueError("Checkpoint manifest changed")
    return plan, audit


def run(root, port):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    plan, _ = verify(root)
    setup = read_json(root / "provenance/frozen_comparison.json")
    if (setup["training_operations_available"] or setup["trained_step"] != 1500
            or setup["trained_checkpoint"] != plan["checkpoint"]):
        raise ValueError("Wrong checkpoint or training-enabled server")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise ValueError("Require EGL for every condition")
    for step in plan["steps"]:
        client = WebsocketClientPolicy("127.0.0.1", port)
        try:
            selected = client.infer({"_frozen_comparison": {"operation": "select", "step": step}})
            if selected["step"] != step:
                raise ValueError("Snapshot selection failed")
        finally:
            client._ws.close()
        output = root / "evaluations" / ("step_" + str(step))
        command = [sys.executable, "-m", "frequency_vla.opsd_evaluate", "--port", str(port),
            "--results-dir", str(output), "--workers", "4", "--suite", "libero_10",
            "--horizon", "5", "--episodes", "10", "--seed", "27", "--initial-state-start", "30"]
        name = "step{}_H5".format(step)
        append_record(root / "stages.jsonl", dict(stage=name, event="start", command=command, unix_time=time.time()))
        print("Starting " + name, flush=True)
        began = time.monotonic()
        with (root / "logs" / (name + ".log")).open("x") as log:
            subprocess.run(["timeout", "--kill-after=10s", str(plan["condition_timeout_seconds"])] + command,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = read_rows(folder(root, step))
        validate_records(rows)
        if len(rows) != 100 or {pair_key(r) for r in rows} != {(27, t, e) for t in range(10) for e in range(10)}:
            raise ValueError("Incomplete evaluation; do not report partial success")
        append_record(root / "stages.jsonl", dict(stage=name, event="complete", seconds=time.monotonic()-began,
            unix_time=time.time(), episodes=len(rows), successes=sum(r["success"] for r in rows), workers=4))
        print("Completed {}: {}/100".format(name, sum(r["success"] for r in rows)), flush=True)
    write_json(root / "provenance/study_complete.json", dict(complete=True, plan_sha256=digest(plan),
        formal_episodes=200, optimizer_updates_added=0))


def analyze(root):
    plan, audit = verify(root)
    done = read_json(root / "provenance/study_complete.json")
    if not done["complete"] or done["plan_sha256"] != digest(plan) or done["optimizer_updates_added"]:
        raise ValueError("Incomplete evaluation or unexpected training")
    setup = read_json(root / "provenance/frozen_comparison.json")
    if setup["training_operations_available"] or setup["trained_checkpoint"] != plan["checkpoint"]:
        raise ValueError("Unexpected comparison checkpoint")
    states = {(r["task_id"], r["initial_state_index"]): r["initial_state_sha256"] for r in audit["evaluation_initial_states"]}
    data, summaries, tasks, all_rows, servers = {}, [], [], [], set()
    core = None
    for step in plan["steps"]:
        rows = read_rows(folder(root, step))
        validate_records(rows)
        if len(rows) != 100 or {pair_key(r) for r in rows} != {(27, t, e) for t in range(10) for e in range(10)}:
            raise ValueError("Wrong paired coverage")
        manifests = [read_json(p) for p in folder(root, step).glob("*.manifest.json")]
        if len(manifests) != 10 or any(m["status"] != "complete" for m in manifests):
            raise ValueError("Incomplete evaluator manifests")
        for m in manifests:
            server = m["server"]
            spec = copy.deepcopy(server["experiment_spec"])
            if digest(spec) != server["inference_fingerprint"] or any(r["inference_fingerprint"] != server["inference_fingerprint"] for r in rows):
                raise ValueError("Inconsistent inference provenance")
            snap = spec.pop("frozen_comparison")
            if snap["optimizer_step"] != step or snap["checkpoint"] != (plan["checkpoint"] if step else None):
                raise ValueError("Wrong model snapshot")
            if spec["prediction_horizon"] != 50 or spec["flow_steps"] != 10:
                raise ValueError("Changed prediction horizon or sampler")
            if server["evaluation_spec"]["rendering"]["backend"] != "egl":
                raise ValueError("Changed rendering backend")
            if core is not None and core != spec:
                raise ValueError("Non-parameter inference settings changed")
            core = spec
            servers.add(server["server_instance_id"])
        for r in rows:
            index = 30 + r["episode_index"]
            if r["initial_state_index"] != index or r["initial_state_sha256"] != states[r["task_id"], index]:
                raise ValueError("Initial state differs from the historical evaluation")
            if r["replan_steps"] != 5 or r["prediction_horizon"] != 50 or r["task_suite"] != "libero_10":
                raise ValueError("Wrong evaluation condition")
            video = root / "evaluations" / ("step_" + str(step)) / r["video_path"]
            if not video.is_file() or not video.stat().st_size:
                raise ValueError("Missing evaluation video: " + str(video))
        data[step] = rows
        summaries.append(summarize_condition(dict(step=step, horizon=5), rows))
        all_rows.extend(dict(optimizer_step=step, **r) for r in rows)
        for task in range(10):
            selected = [r for r in rows if r["task_id"] == task]
            tasks.append(dict(task_id=task, task_description=selected[0]["task_description"],
                **summarize_condition(dict(step=step, horizon=5), selected)))
    if len(servers) != 1:
        raise ValueError("Both checkpoints must use one continuous server")
    change = paired_change(data[1500], data[0], plan)
    historical = read_rows(folder(Path(plan["source_results"]), 0))
    repeat = paired_change(data[0], historical, plan)
    for name, rows in [("conditions.csv", summaries), ("per_task.csv", tasks), ("episodes.csv", all_rows),
                       ("comparison.csv", [change]), ("historical_baseline_repeat.csv", [repeat])]:
        write_csv(root / "aggregated" / name, rows)
    validation = dict(complete=True, primary=change, historical_baseline_repeat=repeat,
        formal_episodes=200, video_files_present=200, optimizer_updates_added=0,
        single_server_instance=next(iter(servers)), paired_states_observations_rng_evaluator=True,
        checkpoint_manifest_sha256=plan["checkpoint"]["manifest_sha256"], renderer="egl", workers=4)
    write_json(root / "aggregated/validation.json", validation)
    lines = ["# H=5 retention after 1500 EGL OPSD updates", "",
        "Both checkpoints were freshly evaluated on one H100/server: P=50, H=5, EGL, four simulator workers, "
        "10 flow steps, LIBERO-10, seed 27, ordered initial-state indices 30–39. No new training was performed.", "",
        "| Model | Success | Mean actions | Mean calls | Mean seconds | Mean seconds, success |",
        "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        successful_seconds = "NA" if s["mean_successful_seconds"] is None else "{:.2f}".format(s["mean_successful_seconds"])
        lines.append("| {} | {}/100 ({:.0%}) | {:.2f} | {:.2f} | {:.2f} | {} |".format(
            "Original" if s["step"] == 0 else "OPSD step 1500", s["total_successes"], s["success_rate"],
            s["mean_episode_length"], s["mean_policy_calls"], s["mean_wall_clock_seconds"], successful_seconds))
    lines += ["", "Trained minus original: **{:+.1f} percentage points**, paired 95% bootstrap CI "
        "**[{:+.1f}, {:+.1f}] pp**; exact two-sided McNemar **p={:.6g}**. "
        "There were {} recovered and {} regressed episodes.".format(100*change["success_change"],
            100*change["ci95_low"], 100*change["ci95_high"], change["p_exact"],
            change["recovered_episodes"], change["regressed_episodes"]), ""]
    if change["success_change"] < 0 and change["p_exact"] < .05 and change["ci95_high"] < 0:
        lines.append("This paired evaluation provides evidence of degraded H=5 performance after H=20 training.")
    elif change["success_change"] > 0 and change["p_exact"] < .05 and change["ci95_low"] > 0:
        lines.append("This paired evaluation provides evidence of improved H=5 performance after H=20 training.")
    else:
        lines.append("This evaluation does not establish a statistically clear H=5 change. "
                     "Failure to detect degradation is not evidence of equivalence or non-inferiority.")
    lines += ["", "The historical original H=5 result was {}/100; the fresh reference is {}/100. "
        "Only the two freshly measured conditions define the primary training comparison.".format(
            plan["historical_H5_successes"], summaries[0]["total_successes"]), "",
        "This checks retention at the explicitly extended P=50, not the official native P=10. "
        "The fixed ten tasks and previously inspected evaluation layouts were held out from added OPSD training; "
        "this is not an unseen-task or blind evaluation. One requested follow-up contrast is reported; "
        "task-level changes are exploratory. Seconds include shared-server waiting and exclude each episode's video encoding.", ""]
    (root / "FINDINGS.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    return validation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "verify", "run", "analyze"])
    p.add_argument("--results-dir", required=True)
    p.add_argument("--source-results")
    p.add_argument("--port", type=int)
    args = p.parse_args()
    root = Path(args.results_dir).resolve()
    if args.stage == "prepare":
        if not args.source_results:
            p.error("prepare requires --source-results")
        prepare(root, Path(args.source_results).resolve())
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
