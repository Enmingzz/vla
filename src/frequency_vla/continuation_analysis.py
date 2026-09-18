"""Report the fixed 500 -> 1000 follow-up using shared paired-data validation."""
import json
from pathlib import Path

from .analysis import pair_key, write_csv
from .logging_utils import digest, write_json


def summarize_continuation(root, plan, data, summaries, task_rows, all_rows, servers, training, rollouts):
    from .study_analysis import paired_change
    first, final = plan.get("comparison_step", plan["resume_step"]), plan["milestones"][-1]
    before = data["confirmation", "libero_10", first, 20]
    after = data["confirmation", "libero_10", final, 20]
    comparison = dict(comparison=plan["primary_comparison"], left_step=final, right_step=first,
                      **paired_change(after, before, plan))
    stages = {r["stage"]:r for line in (root / "stages.jsonl").read_text().splitlines()
              if (r := json.loads(line))["event"] == "complete"}
    for summary in summaries:
        rows = data[summary["split"], summary["suite"], summary["step"], summary["horizon"]]
        success = [r for r in rows if r["success"]]
        failure = [r for r in rows if not r["success"]]
        summary["mean_successful_episode_steps"] = sum(r["controlled_environment_steps"] for r in success) / len(success) if success else None
        summary["mean_failed_episode_steps"] = sum(r["controlled_environment_steps"] for r in failure) / len(failure) if failure else None
        stage = "confirmation_libero_10_step{}_H20".format(summary["step"])
        summary["evaluation_stage_seconds"] = stages[stage]["seconds"]
    pairs_before = {pair_key(r): r for r in before}
    both_success = [(pairs_before[pair_key(r)], r) for r in after if r["success"] and pairs_before[pair_key(r)]["success"]]
    comparison["both_success_episodes"] = len(both_success)
    for label, i in [("before", 0), ("after", 1)]:
        comparison[label + "_mean_steps_on_both_success"] = sum(p[i]["controlled_environment_steps"] for p in both_success) / len(both_success) if both_success else None
    output = root / "aggregated"
    for name, rows in [("conditions.csv", summaries), ("comparisons.csv", [comparison]), ("per_task.csv", task_rows), ("episodes.csv", all_rows)]:
        write_csv(output / name, rows)
    training_config = json.loads((root / "provenance/training_setup.json").read_text())["config"]
    resume = json.loads((root / "provenance/resume.json").read_text())
    if resume["manifest_sha256"] != plan["resume_manifest_sha256"] or not all(resume[k] for k in ["fp32_master_restored", "ema_and_optimizer_restored", "frozen_backbone_equal"]):
        raise ValueError("Incomplete restoration of the declared parent checkpoint")
    if training_config["optimizer_steps"] != final or len(training) != 500:
        raise ValueError("Training exceeded or failed to reach the authorized 500 additional updates")
    validation = dict(complete=True, plan_sha256=digest(plan), evaluation_episodes=len(all_rows),
        conditions=len(summaries), single_server_instance=list(servers)[0], renderer=plan["renderer"],
        optimizer_updates_added=len(training), first_step=first, final_step=final,
        resumed_at_step=plan["resume_step"], updates_in_final_allocation=final - plan["resume_step"],
        prior_continuation_archives=plan.get("prior_continuation_results", []),
        training_evaluation_disjoint=True, all_initial_states_match_audit=True,
        initial_state_indices_previously_evaluated=True,
        new_training_distinct_task_layouts=len({(s["task_id"], s["initial_state_index"]) for r in rollouts for s in r["initial_states"]}),
        new_training_layout_indices=sorted({s["initial_state_index"] for r in rollouts for s in r["initial_states"]}),
        new_training_action_blocks=sum(len(r["initial_states"]) for r in rollouts),
        new_training_executed_actions=sum(s["executed_steps"] for r in rollouts for s in r["initial_states"]),
        primary=comparison)
    write_json(output / "validation.json", validation)
    return plan, summaries, [comparison], task_rows, training, validation


def report(root, plan, summaries, tasks, training, validation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    first, final = validation["first_step"], validation["final_step"]
    p = validation["primary"]
    ordered = sorted(summaries, key=lambda r: r["step"])
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    labels = [str(r["step"]) for r in ordered]
    axes[0].bar(labels, [100*r["success_rate"] for r in ordered], color=["#64748b", "#0284c7"])
    axes[0].errorbar(labels, [100*r["success_rate"] for r in ordered],
                    yerr=[[100*(r["success_rate"]-r["ci95_low"]) for r in ordered],
                          [100*(r["ci95_high"]-r["success_rate"]) for r in ordered]],
                    fmt="none", color="black", capsize=4)
    axes[0].set(ylabel="Success (%)", ylim=(0, 100))
    axes[1].bar(labels, [r["mean_episode_length"] for r in ordered], color=["#64748b", "#0284c7"])
    axes[1].set(ylabel="Mean executed actions / episode")
    axes[2].bar(labels, [r["mean_policy_calls"] for r in ordered], color=["#64748b", "#0284c7"])
    axes[2].set(ylabel="Mean policy calls / episode")
    for ax in axes:
        ax.set_xlabel("Total training updates")
    fig.suptitle("LIBERO-10 · H=20 · 100 paired episodes per checkpoint")
    fig.tight_layout()
    fig.savefig(figures / "continuation_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot([r["optimizer_step"] for r in training], [r["loss"] for r in training], linewidth=.7)
    ax.set(xlabel="Total training updates", ylabel="Velocity matching loss",
           title="New on-policy observations and sampled flow times each update")
    fig.tight_layout()
    fig.savefig(figures / "training_loss.png", dpi=180)
    plt.close(fig)
    if p["ci95_low"] > 0 and p["p_exact"] < .05:
        verdict = "This follow-up supports improvement on the measured layouts."
    elif p["ci95_high"] < 0 and p["p_exact"] < .05:
        verdict = "This follow-up supports degradation on the measured layouts."
    else:
        verdict = "This follow-up does not establish a directional change at the 5% level."
    text = ["# Another 500 OPSD updates: step 500 to step 1000", "",
        "The fixed additional 500 updates changed H=20 success by **{:+.1f} pp**, paired 95% within-task bootstrap CI "
        "**[{:+.1f}, {:+.1f}] pp**, exact McNemar p=**{:.4g}**. There were {} recoveries and {} regressions.".format(
            100*p["success_change"], 100*p["ci95_low"], 100*p["ci95_high"], p["p_exact"], p["recovered_episodes"], p["regressed_episodes"]),
        "", verdict, "", "| Updates | Success | Mean actions/episode | Mean calls/episode | Mean episode seconds | Whole evaluation seconds |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in ordered:
        text.append("| {} | {}/{} ({:.0%}) | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
            r["step"], r["total_successes"], r["total_episodes"], r["success_rate"], r["mean_episode_length"],
            r["mean_policy_calls"], r["mean_wall_clock_seconds"], r["evaluation_stage_seconds"]))
    text += ["", "Episode timing includes reset/settling and inference, but excludes its own video encoding. Whole-condition timing includes process startup and videos. "
             "Eight simulator workers share one H100; summed episode durations are not whole-condition wall time.", "",
             "| Task | Step 500 | Step 1000 | Change | Description |", "|---:|---:|---:|---:|---|"]
    for task in range(10):
        a, b = [next(r for r in tasks if r["task_id"] == task and r["step"] == step) for step in [first, final]]
        text.append("| {} | {:.0%} | {:.0%} | {:+.0f} pp | {} |".format(task, a["success_rate"], b["success_rate"],
                    100*(b["success_rate"]-a["success_rate"]), a["task_description"]))
    if validation.get("resumed_at_step", first) != first:
        text += ["", "The continuation was saved at step {} at the one-hour budget boundary, then resumed for the remaining {} updates. "
                 "Both segments preserve FP32 weights, EMA and Adam moments/counters. Simulator episodes restart on resume; "
                 "this is not an uninterrupted simulator trajectory. Prior segment logs remain in their original archives and "
                 "are joined only after checksum and contiguous-update validation.".format(
                     validation["resumed_at_step"], validation["updates_in_final_allocation"])]
    text += ["", "Training restored the original step-500 FP32 weights, EMA teacher and Adam state. P=50, H_student=20, "
        "H_teacher=5, 10 flow steps, batch size 4, learning rate 1e-5, trainable parameter selection, "
        "velocity loss, first-five-block supervision and auxiliary teacher-tail sampling are unchanged. "
        "The VLM/vision backbone remains frozen. The student, not the EMA, is evaluated.", "",
        "Exactly {} new action blocks executed {} actions across {} task/layout combinations, at layout indices {}. "
        "Training remains LIBERO-10 only, inside indices 10–19; each continuing episode contributes multiple action blocks.".format(
            validation["new_training_action_blocks"], validation["new_training_executed_actions"],
            validation["new_training_distinct_task_layouts"], validation["new_training_layout_indices"]), "",
        "Both checkpoints were re-evaluated on the same continuous server, seed 27, official indices 30–39, "
        "with identical initial-state, first-observation and inference-setting checks. "
        "These layouts were excluded from all added training but their previous results were already inspected; "
        "this is an exploratory continuation, not a new blind confirmation. No intermediate success score selected the step-1000 endpoint.", "",
        "All current training and evaluation use pinned OSMesa, avoiding the prior native NVIDIA EGL failures. "
        "The step-500 result is therefore measured again under OSMesa; the earlier EGL 72% is historical context, "
        "not substituted for the current baseline. CPU checks verified exact serial/parallel simulator observations "
        "before the GPU allocation. Training parallelism changes scheduling, not batch size or action execution.", "",
        "A CPU-only diagnostic found a context-selection issue in serial multi-environment OSMesa rendering. "
        "Training now explicitly makes the owning render context current before stepping/closing each environment. "
        "The corrected serial and isolated-process observations must match exactly before this run. "
        "This is an additional runtime correction; its effect on historical EGL training images has not been established.", "",
        "The continuation combines extra optimization with fresh on-policy experience and a renderer change relative to the parent training. "
        "It does not isolate optimizer-step count on a fixed dataset, establish transfer to other suites, "
        "or reproduce the image-generation Flow-OPD algorithm. "
        "This remains temporal OPSD with velocity matching on the P=50 extension of the native P=10 checkpoint.", ""]
    (root / "FINDINGS.md").write_text("\n".join(text))
