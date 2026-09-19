"""Fresh temporal OPSD training and paired evaluation, entirely under EGL."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .analysis import holm, pair_key, summarize, validate_records, write_csv
from .config import load_config
from .evaluator import array_hash
from .logging_utils import append_record, digest, file_digest, write_json
from .study_analysis import paired_change


def condition_id(c):
    return "step{step}_H{horizon}".format(**c)


def validate_plan(plan):
    expected = [(0, 5), (0, 20), (500, 20), (1000, 20), (1500, 20)]
    if [(c["step"], c["horizon"]) for c in plan["conditions"]] != expected:
        raise ValueError("Require all five predeclared conditions in the fixed order")
    fixed = dict(suite="libero_10", prediction_horizon=50, flow_steps=10, renderer="egl",
                 workers=4, seed=27, initial_state_start=30, episodes_per_task=10,
                 maximum_concurrent_gpus=1, optimizer_updates_added=1500, start_from_official_checkpoint=True)
    if any(plan[k] != v for k, v in fixed.items()):
        raise ValueError("The common EGL evaluation protocol changed")
    if plan["milestones"] != [500, 1000, 1500]:
        raise ValueError("All three predeclared training milestones are required")


def audit(root, plan):
    from .opsd_protocol import validate_training_config
    validate_plan(plan)
    root = Path(root)
    config = load_config(plan["training_config"])
    validate_training_config(config)
    if config["optimizer_steps"] != 1500 or config["checkpoint_steps"] != plan["milestones"] or config.get("environment_workers") != 4:
        raise ValueError("Require the fixed fresh-training configuration")
    sys.path.insert(0, str(Path(os.environ["OPENPI_DIR"]) / "third_party/libero"))
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[plan["suite"]]()
    states = {t: suite.get_task_init_states(t) for t in range(10)}
    training_states = [dict(task_id=t, initial_state_index=i, initial_state_sha256=array_hash(states[t][i]))
                       for t in range(10) for i in range(config["train_initial_state_start"], config["train_initial_state_stop"])]
    train_hashes = {r["initial_state_sha256"] for r in training_states}
    selected = []
    for task in range(10):
        for index in range(30, 40):
            sha = array_hash(states[task][index])
            if sha in train_hashes:
                raise ValueError("Training/evaluation initial-state collision")
            selected.append(dict(task_id=task, initial_state_index=index, initial_state_sha256=sha))
    manifest = json.loads((Path(os.environ["CHECKPOINT_DIR"]) / "download_manifest.json").read_text())
    if manifest["source"] != "gs://openpi-assets/checkpoints/pi05_libero":
        raise ValueError("Fresh training must start from the official checkpoint")
    result = dict(plan_sha256=digest(plan), training_config=config,
        training_config_sha256=digest(config), base_checkpoint_manifest_sha256=digest(manifest),
        evaluation_initial_states=selected, training_initial_states=training_states,
        training_evaluation_states_disjoint=True, training_renderer="egl", evaluation_renderer="egl",
        planned_new_evaluation_episodes=500, optimizer_updates_added=1500)
    write_json(root / "provenance/split_audit.json", result)
    write_json(root / "provenance/study_plan.json", plan)
    return result


def execute(root, plan, port, checkpoint_root):
    from .evaluator import rendering_spec
    root = Path(root)
    validate_plan(plan)
    audit_result = json.loads((root / "provenance/split_audit.json").read_text())
    preflight = json.loads((root / "provenance/preflight_checks.json").read_text())
    if audit_result["plan_sha256"] != digest(plan) or not preflight["passed"] or any(
            file_digest(p) != sha for p, sha in preflight["sources"].items()):
        raise ValueError("Plan or implementation changed after CPU checks")
    setup = json.loads((root / "provenance/training_setup.json").read_text())
    if digest(setup["config"]) != audit_result["training_config_sha256"] or setup["base_experiment_spec"]["checkpoint_object_manifest_sha256"] != audit_result["base_checkpoint_manifest_sha256"]:
        raise ValueError("Wrong official checkpoint or changed training configuration")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise ValueError("No fallback to another renderer is permitted")
    write_json(root / "provenance/training_runtime.json", dict(renderer=rendering_spec(load_config()),
        environment_workers=setup["config"]["environment_workers"], official_start_step=0,
        planned_optimizer_steps=1500, configuration_fixed_for_all_updates=True))

    def run(name, command, limit, **metadata):
        began = time.monotonic()
        append_record(root / "stages.jsonl", dict(stage=name, event="start", command=command,
            unix_time=time.time(), **metadata))
        print("Starting {} ({} second bound)".format(name, limit), flush=True)
        with (root / "logs" / (name + ".log")).open("x") as log:
            subprocess.run(["timeout", "--kill-after=10s", str(limit)] + command,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        append_record(root / "stages.jsonl", dict(stage=name, event="complete", seconds=time.monotonic()-began,
            unix_time=time.time(), **metadata))

    def client(stage, *extra, name=None, limit=120):
        run(name or stage, [sys.executable, "-m", "frequency_vla.opsd_client", "--port", str(port),
            "--config", plan["training_config"], "--results-dir", str(root), "--stage", stage,
            "--rollouts-dir", str(Path(checkpoint_root) / "rollouts")] + list(extra), limit)

    def evaluate(name, step, horizon, episodes, start, seed, workers, task_ids=None, limit=900):
        client("snapshot", "--snapshot", str(step), name="select_" + name)
        pilot = name.startswith("pilot_")
        output = root / "pilots" / name if pilot else root / "evaluations" / ("step_" + str(step))
        command = [sys.executable, "-m", "frequency_vla.opsd_evaluate", "--port", str(port),
            "--results-dir", str(output), "--workers", str(workers), "--suite", plan["suite"],
            "--horizon", str(horizon), "--episodes", str(episodes), "--seed", str(seed),
            "--initial-state-start", str(start)]
        if task_ids is not None:
            command += ["--task-ids"] + [str(t) for t in task_ids]
        run(name, command, limit, workers=workers, pilot=pilot)
        raw = output / "raw/smoke" / plan["suite"] / ("seed_" + str(seed)) / ("H_" + str(horizon))
        records = [json.loads(line) for p in raw.glob("*.jsonl") for line in p.read_text().splitlines() if line]
        validate_records(records)
        expected = {(seed, t, e) for t in (task_ids if task_ids is not None else range(10)) for e in range(episodes)}
        if len(records) != len(expected) or {pair_key(r) for r in records} != expected:
            raise ValueError("Incomplete runtime pilot or formal evaluation")
        manifests = [json.loads(p.read_text()) for p in raw.glob("*.manifest.json")]
        if len(manifests) != len(expected) // episodes or any(m["status"] != "complete" for m in manifests):
            raise ValueError("Incomplete evaluator manifests")

    run("egl_render_context_check", [sys.executable, "tests/check_parallel_osmesa.py", "--training-config", plan["training_config"],
        "--inference-config", os.environ["FREQUENCY_CONFIG"], "--output", str(root / "provenance/render_context_check.json")],
        plan["renderer_check_timeout_seconds"])
    render_check = json.loads((root / "provenance/render_context_check.json").read_text())
    if not render_check["passed"] or render_check["renderer"]["backend"] != "egl":
        raise ValueError("EGL serial/parallel rendering check failed")
    client("diagnostic", limit=plan["diagnostic_timeout_seconds"])
    diagnostic = json.loads((root / "provenance/diagnostic.json").read_text())
    if not diagnostic["passed"] or diagnostic["optimizer_steps"] != 0:
        raise ValueError("Fresh diagnostic must restore the official step-zero weights and optimizer")
    # Pilot task success/failure never determines whether the runtime passes.
    evaluate("pilot_single", 0, 5, 1, plan["pilot_initial_state_start"], plan["pilot_seed"], 1, [0], plan["pilot_timeout_seconds"])
    evaluate("pilot_parallel", 0, 5, 1, plan["pilot_initial_state_start"], plan["pilot_seed"],
             plan["workers"], [0, 1, 2, 3], plan["pilot_timeout_seconds"])
    write_json(root / "provenance/runtime_pilots.json", dict(passed=True, episodes=5,
        renderer="egl", formal_statistics=False, selected_by_task_success=False))
    for c in plan["conditions"]:
        if c["step"]:
            client("train", "--end-step", str(c["step"]), name="train_to_" + str(c["step"]),
                   limit=plan["training_segment_timeout_seconds"])
            client("save", name="save_" + str(c["step"]), limit=240)
        evaluate(condition_id(c), c["step"], c["horizon"], plan["episodes_per_task"],
                 plan["initial_state_start"], plan["seed"], plan["workers"], limit=plan["condition_timeout_seconds"])
    client("status", name="final_server_status")
    write_json(root / "provenance/study_complete.json", dict(complete=True, plan_sha256=digest(plan),
        completed_conditions=[condition_id(c) for c in plan["conditions"]], formal_episodes=500,
        optimizer_updates_added=1500, renderer="egl", workers=plan["workers"], final_optimizer_step=1500))
    print("Fresh 1500-update EGL training and all 500 evaluation episodes complete; release GPU.", flush=True)


def summarize_condition(condition, rows):
    successful = [r for r in rows if r["success"]]
    result = dict(condition, **summarize(condition["horizon"], rows))
    for label, field in [("actions", "controlled_environment_steps"), ("calls", "policy_calls"), ("seconds", "wall_clock_seconds")]:
        result["mean_successful_" + label] = sum(r[field] for r in successful) / len(successful) if successful else None
    return result


def analyze(root):
    root = Path(root)
    plan = json.loads((root / "provenance/study_plan.json").read_text())
    validate_plan(plan)
    audit_result = json.loads((root / "provenance/split_audit.json").read_text())
    complete = json.loads((root / "provenance/study_complete.json").read_text())
    setup = json.loads((root / "provenance/training_setup.json").read_text())
    if not complete["complete"] or complete["plan_sha256"] != audit_result["plan_sha256"] or digest(plan) != complete["plan_sha256"]:
        raise ValueError("Incomplete or changed EGL comparison")
    if complete["completed_conditions"] != [condition_id(c) for c in plan["conditions"]] or complete["optimizer_updates_added"] != 1500:
        raise ValueError("Require all conditions and exactly 1500 fresh updates")
    if digest(setup["config"]) != audit_result["training_config_sha256"] or (root / "provenance/resume.json").exists():
        raise ValueError("Training did not use the fixed configuration from the official checkpoint")
    training = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines() if line]
    rollouts = [json.loads(line) for line in (root / "rollouts.jsonl").read_text().splitlines() if line]
    if any([r["optimizer_step"] for r in rows] != list(range(1,1501)) for rows in [training, rollouts]):
        raise ValueError("Training records must contain exactly updates 1 through 1500")
    pool = {(r["task_id"],r["initial_state_index"]):r["initial_state_sha256"] for r in audit_result["training_initial_states"]}
    for row, rollout in zip(training, rollouts):
        if row["diagnostic"] or rollout["diagnostic"] or row["student_behavior_version"] != row["optimizer_step"]-1:
            raise ValueError("Diagnostic or off-policy update entered training")
        if len(rollout["initial_states"]) != setup["config"]["batch_size"]:
            raise ValueError("Training rollout batch size changed")
        for state in rollout["initial_states"]:
            if pool.get((state["task_id"],state["initial_state_index"])) != state["initial_state_sha256"]:
                raise ValueError("Training departed from the declared layout pool")
    runtime=json.loads((root / "provenance/training_runtime.json").read_text())
    if runtime["renderer"]["backend"] != "egl" or runtime["official_start_step"] != 0:
        raise ValueError("Training did not use EGL from step zero")
    checkpoints={}
    for step in plan["milestones"]:
        identity=json.loads((root / "provenance" / ("step_"+str(step)+".json")).read_text())
        checkpoint=Path(identity["path"])
        manifest=json.loads((checkpoint / "training_manifest.json").read_text())
        if digest(manifest) != identity["manifest_sha256"] or manifest["step"] != step or manifest.get("resumed_from"):
            raise ValueError("Wrong saved milestone or mixed historical training checkpoint")
        if digest(manifest["config"]) != audit_result["training_config_sha256"] or manifest["base_object_manifest_sha256"] != audit_result["base_checkpoint_manifest_sha256"]:
            raise ValueError("Training configuration or base checkpoint changed")
        if manifest["reloaded_native_inference_max_abs_difference"] != 0:
            raise ValueError("Native checkpoint roundtrip differs")
        for name, sha in manifest["files"].items():
            if file_digest(checkpoint/name) != sha:
                raise ValueError("Saved checkpoint checksum mismatch")
        write_json(root / "provenance" / ("exported_training_manifest_step_"+str(step)+".json"),manifest)
        checkpoints[str(step)]=identity
    expected_hashes = {(r["task_id"], r["initial_state_index"]): r["initial_state_sha256"] for r in audit_result["evaluation_initial_states"]}
    stages = {r["stage"]: r for line in (root / "stages.jsonl").read_text().splitlines() if (r := json.loads(line))["event"] == "complete"}
    summaries, all_rows, task_rows, videos = [], [], [], []
    data, servers, core = {}, set(), None
    for c in plan["conditions"]:
        folder = root / "evaluations" / ("step_" + str(c["step"])) / "raw/smoke/libero_10/seed_27" / ("H_" + str(c["horizon"]))
        rows = [json.loads(line) for p in sorted(folder.glob("*.jsonl")) for line in p.read_text().splitlines() if line]
        manifests = [json.loads(p.read_text()) for p in folder.glob("*.manifest.json")]
        validate_records(rows)
        if len(rows) != 100 or {pair_key(r) for r in rows} != {(27, t, e) for t in range(10) for e in range(10)} or len(manifests) != 10:
            raise ValueError("Unequal/duplicate episode or manifest coverage")
        for r in rows:
            index = 30 + r["episode_index"]
            if r["initial_state_index"] != index or r["initial_state_sha256"] != expected_hashes[r["task_id"], index]:
                raise ValueError("Measured initial state differs from the audited layout")
            if r["task_suite"] != "libero_10" or r["replan_steps"] != c["horizon"] or r["prediction_horizon"] != 50:
                raise ValueError("Measured condition differs from plan")
            video = folder.parents[4] / r["video_path"]
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(video)], text=True))["streams"][0]
            if (info["width"], info["height"], info["r_frame_rate"], int(info["nb_frames"])) != (224, 224, "10/1", r["controlled_environment_steps"]):
                raise ValueError("Video does not match executed actions")
            videos.append(dict(path=str(video.relative_to(root)), frames=int(info["nb_frames"])))
        for manifest in manifests:
            server = manifest["server"]
            if manifest["status"] != "complete" or server["evaluation_spec"]["rendering"]["backend"] != "egl":
                raise ValueError("Incomplete or non-EGL condition")
            spec = copy.deepcopy(server["experiment_spec"])
            if digest(spec) != server["inference_fingerprint"] or any(r["inference_fingerprint"] != server["inference_fingerprint"] for r in rows):
                raise ValueError("Inference provenance changed within a condition")
            snapshot = spec.pop("temporal_opsd")
            if snapshot["optimizer_step"] != c["step"] or snapshot["checkpoint"] != checkpoints.get(str(c["step"])):
                raise ValueError("Wrong frozen parameter snapshot")
            if snapshot["sources"] != setup["sources"] or snapshot["evaluation_sampler"] != "unchanged Pi0.sample_actions; dynamic parameter arguments":
                raise ValueError("Native sampler changed")
            if core is not None and core != spec:
                raise ValueError("Non-parameter inference settings changed")
            core = spec
            servers.add(server["server_instance_id"])
        data[c["step"], c["horizon"]] = rows
        stage = stages[condition_id(c)]
        if stage["workers"] != plan["workers"] or stage["pilot"]:
            raise ValueError("Concurrency changed or pilot mixed into formal statistics")
        summary = summarize_condition(c, rows)
        summary["evaluation_stage_seconds"] = stage["seconds"]
        summaries.append(summary)
        all_rows.extend(dict(optimizer_step=c["step"], **r) for r in rows)
        task_rows.extend(dict(task_id=t, **summarize_condition(c, [r for r in rows if r["task_id"] == t])) for t in range(10))
    if len(servers) != 1 or core["flow_steps"] != 10:
        raise ValueError("All conditions must share one continuous native sampler/server")
    comparisons = []
    for c in plan["conditions"][1:]:
        key = (c["step"], c["horizon"])
        for label, reference in [("versus_original_H5", (0, 5))] + ([("versus_original_H20", (0, 20))] if c["step"] else []):
            comparisons.append(dict(comparison=label, step=c["step"], H=c["horizon"],
                reference_step=reference[0], reference_H=reference[1],
                **paired_change(data[key], data[reference], plan)))
    comparisons.append(dict(comparison="step1500_minus_step1000", step=1500, H=20, reference_step=1000, reference_H=20,
                            **paired_change(data[1500,20], data[1000,20], plan)))
    for i, value in holm({i:r["p_exact"] for i,r in enumerate(comparisons)}).items():
        comparisons[i]["p_holm_exploratory_family"] = value
    for name, values in [("conditions", summaries), ("comparisons", comparisons), ("per_task", task_rows), ("episodes", all_rows)]:
        write_csv(root / "aggregated" / (name + ".csv"), values)
    validation = dict(complete=True, plan_sha256=digest(plan), formal_episodes=500, conditions=5,
        same_renderer="egl", same_workers=4, single_server_instance=next(iter(servers)),
        optimizer_updates_added=1500, trained_from_official_step_zero=True,
        initial_states_first_observations_rng_and_evaluator_paired=True,
        training_evaluation_states_disjoint=True, training_renderer="egl", evaluation_renderer="egl",
        all_checkpoint_checksums_verified=True, native_checkpoint_roundtrip_exact=True,
        validated_videos=len(videos))
    write_json(root / "aggregated/validation.json", validation)
    write_json(root / "provenance/video_checks.json", dict(passed=True, checks=videos))
    report(root, summaries, comparisons)
    return validation


def report(root, summaries, comparisons):
    text = ["# Fresh training and evaluation entirely under EGL", "",
        "All five conditions were rerun on one H100/server, EGL, four simulator workers, P=50, 10 flow steps, "
        "LIBERO-10, seed 27, official initial-state indices 30–39 (10 tasks × 10 episodes). "
        "All 1500 temporal OPSD updates started from the official checkpoint and used the same EGL training implementation.", "",
        "| Training steps | H | Success | Avg. actions, all | Avg. actions, success | Avg. calls, all | Avg. seconds, all | Avg. seconds, success |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        text.append("| {} | {} | {}/100 ({:.0%}) | {:.2f} | {} | {:.2f} | {:.2f} | {} |".format(
            "0 (Original)" if r["step"] == 0 else r["step"], r["horizon"], r["total_successes"], r["success_rate"],
            r["mean_episode_length"], "{:.2f}".format(r["mean_successful_actions"]) if r["total_successes"] else "N/A",
            r["mean_policy_calls"], r["mean_wall_clock_seconds"], "{:.2f}".format(r["mean_successful_seconds"]) if r["total_successes"] else "N/A"))
    text += ["", "Seconds are mean episode wall time under fixed four-worker concurrency, including reset, settling, inference and execution, "
             "excluding that episode's video encoding. They include shared-server waiting and are not isolated model latency. "
             "Success-only means condition on potentially different sets of episodes. Whole-condition seconds and Wilson success intervals are in conditions.csv.", "",
             "## Controlled training and evaluation", "",
             "This is a new run from the official checkpoint. All updates 1–1500 and every evaluation use EGL; "
             "historical OSMesa-trained checkpoints and historical evaluation outcomes are not used. "
             "Training keeps P=50, student H=20, teacher H=5, 10 flow steps, batch 4, learning rate 1e-5, "
             "EMA 0.9999, the same frozen VLM/vision backbone and the same temporal velocity-matching loss. "
             "The student, not the EMA, is evaluated. Train initial states stay within indices 10–19, seed 17; "
             "teacher views are collected from the student's on-policy trajectory. The context-selection correction is active from the start "
             "and serial/parallel observations must match exactly in the EGL check before formal training.", "",
             "Checkpoints 500/1000/1500 were fixed before observing outcomes. Each 500-update segment starts fresh simulator trajectories "
             "under the same initial-state cycling rule; weights, EMA and Adam continue without resetting. More steps also add on-policy experience, "
             "so this is not a fixed-dataset optimizer-step ablation. Evaluation layouts were inspected previously and remain excluded from training; "
             "this is not a new blind test. Eight paired success comparisons form one Holm-adjusted exploratory family.", "",
             "| Comparison | Step | Reference H | Success change (pp) | Paired 95% CI (pp) | Exact p | Holm p |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in comparisons:
        text.append("| {} | {} | {} | {:+.1f} | [{:+.1f}, {:+.1f}] | {:.4g} | {:.4g} |".format(
            r["comparison"], r["step"], r["reference_H"], 100*r["success_change"], 100*r["ci95_low"], 100*r["ci95_high"],
            r["p_exact"], r["p_holm_exploratory_family"]))
    (root / "FINDINGS.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "run", "analyze"])
    parser.add_argument("--plan", default="configs/egl_pipeline.yaml")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--checkpoint-root")
    args = parser.parse_args()
    plan = load_config(args.plan)
    if args.stage == "audit":
        audit(args.results_dir, plan)
    elif args.stage == "run":
        if not args.port or not args.checkpoint_root:
            parser.error("run requires --port and --checkpoint-root")
        execute(args.results_dir, plan, args.port, args.checkpoint_root)
    else:
        print(json.dumps(analyze(args.results_dir), indent=2))


if __name__ == "__main__":
    main()
