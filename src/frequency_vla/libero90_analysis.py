"""Validate and analyze frozen LIBERO-90 transfer, including every regression."""
import argparse
import copy
import json
from pathlib import Path

from .analysis import holm, pair_key, summarize, validate_records, write_csv
from .libero90_transfer import condition_id, conditions, validate_plan
from .logging_utils import digest, file_digest, write_json
from .study_analysis import paired_change


def task_and_layout_interval(left, right, replicates, seed):
    """Exploratory uncertainty resampling both tasks and their paired layouts."""
    import numpy as np
    a, b = {pair_key(r): r for r in left}, {pair_key(r): r for r in right}
    if set(a) != set(b):
        raise ValueError("Hierarchical bootstrap needs equal paired coverage")
    ids = sorted({r["task_id"] for r in left})
    values = [np.array([int(a[k]["success"])-int(b[k]["success"]) for k in sorted(a) if k[1] == task]) for task in ids]
    if len({len(v) for v in values}) != 1:
        raise ValueError("This task bootstrap requires balanced layout counts")
    values = np.stack(values)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, replicates, 500):
        n = min(500, replicates-start)
        tasks = rng.integers(0, len(ids), size=(n, len(ids), 1))
        layouts = rng.integers(0, values.shape[1], size=(n, len(ids), values.shape[1]))
        draws.extend(values[tasks, layouts].mean(axis=(1, 2)).tolist())
    return tuple(float(x) for x in np.quantile(draws, [.025, .975]))


def analyze(root):
    root = Path(root)
    plan = json.loads((root / "provenance/study_plan.json").read_text())
    validate_plan(plan)
    audit = json.loads((root / "provenance/split_audit.json").read_text())
    complete = json.loads((root / "provenance/study_complete.json").read_text())
    if not complete["complete"] or not complete["plan_sha256"] == audit["plan_sha256"] == digest(plan):
        raise ValueError("Incomplete or changed evaluation plan")
    if complete["optimizer_updates_added"] or (root / "training.jsonl").exists():
        raise ValueError("This study permits no new training")
    setup = json.loads((root / "provenance/frozen_comparison.json").read_text())
    if setup["training_operations_available"] or setup["trained_checkpoint"]["manifest_sha256"] != plan["checkpoint_manifest_sha256"]:
        raise ValueError("Wrong comparison checkpoint or optimizer-enabled server")
    for src in audit["prior_training_rollout_files"]:
        if file_digest(src["path"]) != src["sha256"]:
            raise ValueError("Prior training provenance changed")
    expected_hashes = {(r["task_id"], r["initial_state_index"]): r["initial_state_sha256"] for r in audit["evaluation_initial_states"]}
    task_info = {t["task_id"]: t for t in audit["tasks"]}
    primary_ids = set(audit["primary_task_ids"])
    if primary_ids != {t["task_id"] for t in audit["tasks"] if not t["instruction_seen_in_added_opsd"]}:
        raise ValueError("Primary task selection differs from the predeclared instruction rule")
    data, all_rows, per_task = {}, [], []
    core_reference, servers = None, set()
    for c in conditions(plan):
        folder = root / "evaluations" / ("step_"+str(c["step"])) / "raw/smoke/libero_90" / ("seed_"+str(c["seed"])) / ("H_"+str(c["horizon"]))
        manifests = [json.loads(p.read_text()) for p in folder.glob("*.manifest.json")]
        rows = [json.loads(line) for p in sorted(folder.glob("*.jsonl")) for line in p.read_text().splitlines() if line]
        if len(manifests) != 90 or any(m["status"] != "complete" for m in manifests):
            raise ValueError("Incomplete 90-task condition: " + condition_id(c))
        validate_records(rows)
        expected = {(c["seed"], t, e) for t in range(90) for e in range(c["episodes_per_task"])}
        if len(rows) != len(expected) or {pair_key(r) for r in rows} != expected:
            raise ValueError("Unequal/duplicate task or layout coverage")
        for r in rows:
            i = c["initial_state_start"] + r["episode_index"]
            if r["initial_state_index"] != i or r["initial_state_sha256"] != expected_hashes[r["task_id"], i]:
                raise ValueError("Initial state differs from the CPU audit")
            if r["task_description"] != task_info[r["task_id"]]["description"]:
                raise ValueError("Task instruction changed")
            if r["task_suite"] != "libero_90" or r["replan_steps"] != c["horizon"] or r["prediction_horizon"] != 50:
                raise ValueError("Wrong task suite or horizons")
            if not 1 <= r["controlled_environment_steps"] <= 400:
                raise ValueError("Wrong official LIBERO-90 episode budget")
        for m in manifests:
            server = m["server"]
            if server["evaluation_spec"].get("rendering", {}).get("backend") != plan["renderer"]:
                raise ValueError("Renderer differs from the fixed runtime amendment")
            spec = copy.deepcopy(server["experiment_spec"])
            if digest(spec) != server["inference_fingerprint"] or any(r["inference_fingerprint"] != server["inference_fingerprint"] for r in rows):
                raise ValueError("Changed inference fingerprint within a condition")
            snapshot = spec.pop("frozen_comparison")
            if snapshot["optimizer_step"] != c["step"]:
                raise ValueError("Wrong parameter snapshot")
            if c["step"] and snapshot["checkpoint"] != setup["trained_checkpoint"]:
                raise ValueError("Wrong trained checkpoint identity")
            if core_reference is None:
                core_reference = spec
            if spec != core_reference or spec["flow_steps"] != 10:
                raise ValueError("Inference settings differ between snapshots")
            servers.add(server["server_instance_id"])
        data[c["step"], c["horizon"]] = rows
        all_rows.extend(dict(condition=condition_id(c), optimizer_step=c["step"],
                            primary_novel_instruction=r["task_id"] in primary_ids, **r) for r in rows)
        for task in range(90):
            tr = [r for r in rows if r["task_id"] == task]
            per_task.append(dict(step=c["step"], horizon=c["horizon"], task_id=task,
                task_description=tr[0]["task_description"], primary_novel_instruction=task in primary_ids,
                **summarize(c["horizon"], tr)))
    if len(servers) != 1:
        raise ValueError("Require one continuous native sampler/server for every condition")
    tables, comparisons = [], []
    subsets = {"all_90": set(range(90)), "novel_instruction": primary_ids}
    for label, ids in subsets.items():
        selected = {k: [r for r in rows if r["task_id"] in ids] for k, rows in data.items()}
        for (step,h), rows in selected.items():
            tables.append(dict(subset=label, tasks=len(ids), step=step, **summarize(h, rows)))
        for name, left, right in [("training_benefit", (500,20), (0,20)), ("replanning_gap", (0,5), (0,20))]:
            change = paired_change(selected[left], selected[right], plan)
            lo, hi = task_and_layout_interval(selected[left], selected[right], plan["bootstrap_replicates"], plan["analysis_seed"])
            family = "primary" if label == "novel_instruction" and name == "training_benefit" else "secondary"
            comparisons.append(dict(subset=label, comparison=name, family=family,
                left_step=left[0], left_H=left[1], right_step=right[0], right_H=right[1],
                hierarchical_ci95_low=lo, hierarchical_ci95_high=hi, **change))
    for family in ["primary", "secondary"]:
        adjusted = holm({i:r["p_exact"] for i,r in enumerate(comparisons) if r["family"] == family})
        for i,p in adjusted.items():
            comparisons[i]["p_holm"] = p
    recovery = []
    for label in subsets:
        by_model = {(r["step"], r["H"]): r for r in tables if r["subset"] == label}
        teacher, original, trained = [by_model[k] for k in [(0,5), (0,20), (500,20)]]
        gap = teacher["success_rate"] - original["success_rate"]
        benefit = trained["success_rate"] - original["success_rate"]
        recovery.append(dict(subset=label, original_replanning_gap=gap,
            trained_H20_gain=benefit,
            remaining_gap_to_original_H5=teacher["success_rate"]-trained["success_rate"],
            fraction_of_original_gap_recovered=benefit/gap if gap > 0 else None,
            trained_calls_per_episode_saved_vs_original_H5=1-trained["mean_policy_calls"]/teacher["mean_policy_calls"],
            trained_calls_per_control_step_saved_vs_original_H5=1-trained["policy_calls_per_environment_step"]/teacher["policy_calls_per_environment_step"]))
    changes = []
    for task in range(90):
        values = {(r["step"],r["horizon"]):r for r in per_task if r["task_id"] == task}
        changes.append(dict(task_id=task, description=task_info[task]["description"],
            primary_novel_instruction=task in primary_ids,
            original_H5=values[0,5]["success_rate"], original_H20=values[0,20]["success_rate"],
            trained_H20=values[500,20]["success_rate"],
            success_change=values[500,20]["success_rate"]-values[0,20]["success_rate"]))
    validation = dict(complete=True, plan_sha256=digest(plan), conditions=3, formal_episodes=len(all_rows),
        primary_tasks=len(primary_ids), primary_episodes=len(primary_ids)*plan["episodes_per_task"],
        excluded_instruction_overlap_task_ids=audit["overlapping_instruction_task_ids"],
        no_new_optimizer_updates=True, exact_policy_call_schedule=True,
        all_initial_states_match_audit=True, single_server_instance=next(iter(servers)),
        primary=next(r for r in comparisons if r["family"] == "primary"),
        gap_recovery=recovery,
        task_changes={k:sum((r["success_change"]>0 if k=="improved" else r["success_change"]<0 if k=="declined" else r["success_change"]==0)
                           for r in changes if r["primary_novel_instruction"]) for k in ["improved","declined","tied"]})
    for name, rows in [("conditions.csv",tables), ("comparisons.csv",comparisons), ("per_task.csv",per_task),
                       ("task_changes.csv",changes), ("gap_recovery.csv",recovery), ("episodes.csv",all_rows)]:
        write_csv(root / "aggregated" / name, rows)
    write_json(root / "aggregated/validation.json", validation)
    return plan, audit, tables, comparisons, changes, validation


def report(root, plan, audit, summaries, comparisons, changes, validation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    labels = ["Original H=5", "Original H=20", "500 updates H=20"]
    colors = ["#64748b", "#ca8a04", "#059669"]
    fig, axes = plt.subplots(1,2,figsize=(11,4.7),sharey=True)
    for ax, subset in zip(axes,["all_90","novel_instruction"]):
        rows = [next(r for r in summaries if r["subset"]==subset and (r["step"],r["H"])==key) for key in [(0,5),(0,20),(500,20)]]
        ax.bar(range(3),[100*r["success_rate"] for r in rows],color=colors)
        ax.vlines(range(3),[100*r["ci95_low"] for r in rows],[100*r["ci95_high"] for r in rows],color="black")
        for i,r in enumerate(rows):
            ax.text(i,100*r["ci95_high"]+2,"{}/{}".format(r["total_successes"],r["total_episodes"]),ha="center",fontsize=9)
        ax.set(xticks=range(3),xticklabels=labels,ylim=(0,110),title="{} tasks · {} episodes".format(rows[0]["tasks"],rows[0]["total_episodes"]))
        ax.tick_params(axis="x",labelrotation=12)
    axes[0].set_ylabel("LIBERO-90 task success (%)")
    fig.suptitle("Cross-task transfer · P=50 · 95% Wilson intervals\nRight: instructions absent from added OPSD training")
    fig.tight_layout()
    fig.savefig(output / "libero90_success.png",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4.5))
    ax.bar([r["task_id"] for r in changes],[100*r["success_change"] for r in changes],
           color=["#059669" if r["success_change"]>0 else "#dc2626" if r["success_change"]<0 else "#64748b" for r in changes])
    for task in audit["overlapping_instruction_task_ids"]:
        ax.axvspan(task-.45,task+.45,color="gray",alpha=.25)
    ax.axhline(0,color="black",linewidth=.7)
    ax.set(xlabel="Official LIBERO-90 task ID",ylabel="Trained − original H=20 success (pp)",
           title="All tasks retained · 3 layouts/task · shaded task excluded from primary analysis",ylim=(-105,105))
    fig.tight_layout()
    fig.savefig(output / "libero90_task_changes.png",dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4.7))
    for label, color, key in zip(labels,colors,[(0,5),(0,20),(500,20)]):
        r=next(r for r in summaries if r["subset"]=="novel_instruction" and (r["step"],r["H"])==key)
        ax.errorbar(r["policy_calls_per_environment_step"],100*r["success_rate"],
                    yerr=[[100*(r["success_rate"]-r["ci95_low"])],[100*(r["ci95_high"]-r["success_rate"])]],
                    fmt="o",capsize=4,color=color,label=label)
    ax.set(xlabel="Policy calls per controlled environment step",ylabel="Task success (%)",
           title="LIBERO-90 transfer · primary task subset · 95% Wilson intervals",ylim=(0,105))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "success_vs_policy_calls.png",dpi=180)
    plt.close(fig)
    p=validation["primary"]
    lookup={(r["subset"],r["step"],r["H"]):r for r in summaries}
    text=["# LIBERO-90 transfer findings","",
        "This evaluation adds **zero optimizer updates**. It compares the official model with the fixed 500-update "
        "model trained only on LIBERO-10, using all 90 LIBERO-90 tasks and three official initial layouts (40–42), seed 37.","",
        "## Primary comparison: task instructions absent from added OPSD","",
        "The primary subset contains **{} tasks / {} paired episodes**. The trained H=20 model achieved **{:.1%}**, "
        "versus **{:.1%}** for original H=20: **{:+.1f} pp**, paired within-task 95% bootstrap CI **[{:+.1f}, {:+.1f}] pp**, "
        "exact McNemar p=**{:.4g}**. There were {} recoveries and {} regressions.".format(validation["primary_tasks"],p["episodes"],
            p["left_success_rate"],p["right_success_rate"],100*p["success_change"],100*p["ci95_low"],100*p["ci95_high"],
            p["p_exact"],p["recovered_episodes"],p["regressed_episodes"]),"",
        "An exploratory bootstrap resampling both tasks and layouts gives **[{:+.1f}, {:+.1f}] pp**. "
        "Among primary tasks, {} improved, {} declined and {} tied (three trials/task).".format(
            100*p["hierarchical_ci95_low"],100*p["hierarchical_ci95_high"],validation["task_changes"]["improved"],
            validation["task_changes"]["declined"],validation["task_changes"]["tied"]),""]
    if p["ci95_low"]>0 and p["p_exact"]<.05:
        text += ["The primary comparison supports improved success on these tasks held out from added OPSD. " +
                 ("The task-resampling interval also stays above zero." if p["hierarchical_ci95_low"]>0 else
                  "The task-resampling interval includes zero, so broader task-level generalization remains uncertain.")]
    elif p["ci95_high"]<0 and p["p_exact"]<.05:
        text += ["The primary comparison indicates harmful transfer: the LIBERO-10 update reduced success on these held-out tasks."]
    else:
        text += ["The primary comparison does not establish an improvement or decline at the 5% level. These data do not validate cross-task recovery."]
    recovery=next(r for r in validation["gap_recovery"] if r["subset"]=="novel_instruction")
    frequency=next(r for r in comparisons if r["subset"]=="novel_instruction" and r["comparison"]=="replanning_gap")
    text += ["","## Replanning gap and inference calls","",
        "Original H=5 minus original H=20 was **{:+.1f} pp**, paired 95% CI **[{:+.1f}, {:+.1f}] pp**, "
        "secondary-family Holm p=**{:.4g}**. Trained H=20 remains **{:+.1f} pp** below original H=5.".format(
            100*frequency["success_change"],100*frequency["ci95_low"],100*frequency["ci95_high"],
            frequency["p_holm"],100*recovery["remaining_gap_to_original_H5"]),"",
        "Trained H=20 uses **{:.1%} fewer calls per episode** and **{:.1%} fewer calls per controlled step** than original H=5. "
        "The latter controls for differing episode lengths; neither number measures wall-clock speedup.".format(
            recovery["trained_calls_per_episode_saved_vs_original_H5"],recovery["trained_calls_per_control_step_saved_vs_original_H5"])]
    if recovery["fraction_of_original_gap_recovered"] is not None:
        text += ["The point estimate recovers **{:.1%}** of the original replanning gap.".format(
            recovery["fraction_of_original_gap_recovered"])]
    text += ["","## All results","","| Subset | Model | Successes/episodes | Success | Calls/episode | Calls/control step |",
             "|---|---|---:|---:|---:|---:|"]
    for subset in ["all_90","novel_instruction"]:
        for label,(step,h) in zip(labels,[(0,5),(0,20),(500,20)]):
            r=lookup[subset,step,h]
            text.append("| {} | {} | {}/{} | {:.1%} | {:.2f} | {:.4f} |".format(subset,label,r["total_successes"],r["total_episodes"],r["success_rate"],r["mean_policy_calls"],r["policy_calls_per_environment_step"]))
    text += ["","The three secondary comparisons (complete-suite training effect and original replanning gaps for both subsets) "
             "receive a joint Holm correction; comparisons.csv retains raw/adjusted p values and both bootstrap intervals.","",
             "## Scope and controls","",
             "LIBERO-90 task names are disjoint from the ten OPSD training task names. Exact instruction overlap occurs at task IDs {}. "
             "Those tasks are retained in the all-90 table and excluded from the primary subset using a metadata rule fixed before evaluation.".format(audit["overlapping_instruction_task_ids"]),"",
             "Only our added OPSD is task-held-out. The official checkpoint's historical fine-tuning/pretraining exposure is not fully audited. "
             "Current public task metadata is archived separately and is not evidence that the base model never saw a task, scene, object or subskill.","",
             "All comparisons fix P=50 (an extension of the checkpoint's native P=10), ten native flow steps, preprocessing, ordered initial states, "
             "episode/call RNG and the official LIBERO-90 400-step limit. Both frozen snapshots share one continuous native sampler. "
             "No optimizer or teacher updates are available in this comparison server. This evaluates the temporal OPSD Gaussian velocity-matching "
             "adaptation, not a new reproduction of the image-generation Flow-OPD algorithm.","",
             "The 500-update checkpoint was selected before any LIBERO-90 outcome. Rendering pilot episodes are excluded. "
             "All formal episodes, including failed tasks, are retained; simulator/inference exceptions abort evaluation.","",
             "After repeated native NVIDIA EGL aborts before any formal episode, all three conditions use the same pinned OSMesa software renderer. "
             "A separate exact-action replay passed; archived EGL/OSMesa images were not pixel-identical. The result is therefore specific to "
             "this documented rendering environment. No formal EGL measurements are mixed into these statistics.","",
             "Raw JSONL/manifests and videos are under evaluations/. Tables include every task and paired episode. "
             "See provenance/ for the split audit, checkpoint hashes, code, checks and GPU accounting; see LIBERO90_TRANSFER.md for exact rerun commands.",""]
    (root/"FINDINGS.md").write_text("\n".join(text))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir",required=True)
    args=p.parse_args()
    result=analyze(args.results_dir)
    report(args.results_dir,*result)
    print(json.dumps(result[-1],indent=2))


if __name__=="__main__":
    main()
