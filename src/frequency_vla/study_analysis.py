"""Analyze the frozen round-two matrix without selecting checkpoints by success."""
import argparse
import copy
import json
from pathlib import Path

from .analysis import exact_mcnemar, holm, paired_interval, pair_key, summarize, validate_records, write_csv
from .logging_utils import digest, write_json
from .study_plan import condition_id, conditions


def paired_change(left, right, plan):
    a, b = {pair_key(r): r for r in left}, {pair_key(r): r for r in right}
    if set(a) != set(b):
        raise ValueError("Unequal paired episode coverage")
    for key in a:
        for field in ["initial_state_index", "initial_state_sha256", "first_observation_sha256",
                      "episode_rng_seed", "evaluation_fingerprint", "task_description", "prediction_horizon"]:
            if a[key][field] != b[key][field]:
                raise ValueError("Unpaired comparison: " + str((key, field)))
    pairs = [(a[k], b[k]) for k in sorted(a)]
    won = sum(x["success"] and not y["success"] for x, y in pairs)
    lost = sum(y["success"] and not x["success"] for x, y in pairs)
    low, high = paired_interval(pairs, plan["bootstrap_replicates"], plan["analysis_seed"])
    return {"episodes": len(pairs), "left_success_rate": sum(r["success"] for r in left) / len(left),
        "right_success_rate": sum(r["success"] for r in right) / len(right),
        "success_change": (won - lost) / len(pairs), "ci95_low": low, "ci95_high": high,
        "recovered_episodes": won, "regressed_episodes": lost, "p_exact": exact_mcnemar(won, lost)}


def analyze(root):
    root = Path(root)
    plan = json.loads((root / "provenance/study_plan.json").read_text())
    audit = json.loads((root / "provenance/split_audit.json").read_text())
    completed = json.loads((root / "provenance/study_complete.json").read_text())
    if not completed["complete"] or not (completed["plan_sha256"] == audit["plan_sha256"] == digest(plan)):
        raise ValueError("The declared research matrix is incomplete or changed")
    matrix = conditions(plan)
    data, summaries, task_rows, all_rows, servers = {}, [], [], [], set()
    core_reference = None
    expected_hashes = {key: {(r["task_id"], r["initial_state_index"]): r["initial_state_sha256"] for r in value}
                       for key, value in audit["evaluation_initial_states"].items()}
    for c in matrix:
        folder = root / "evaluations" / c["split"] / ("step_" + str(c["step"])) / "raw/smoke" / c["suite"] / ("seed_" + str(c["seed"])) / ("H_" + str(c["horizon"]))
        rows = [json.loads(line) for p in sorted(folder.glob("*.jsonl")) for line in p.read_text().splitlines() if line]
        manifests = [json.loads(p.read_text()) for p in folder.glob("*.manifest.json")]
        if len(manifests) != 10 or any(m["status"] != "complete" for m in manifests):
            raise ValueError("Incomplete condition: " + condition_id(c))
        validate_records(rows)
        expected = {(c["seed"], t, e) for t in range(10) for e in range(c["episodes_per_task"])}
        if len(rows) != len(expected) or {pair_key(r) for r in rows} != expected:
            raise ValueError("Unexpected coverage: " + condition_id(c))
        for row in rows:
            index = c["initial_state_start"] + row["episode_index"]
            if row["replan_steps"] != c["horizon"] or row["prediction_horizon"] != 50 or row["task_suite"] != c["suite"]:
                raise ValueError("Measured condition differs from the declared horizon/suite")
            if row["initial_state_index"] != index or row["initial_state_sha256"] != expected_hashes[c["split"] + "/" + c["suite"]][row["task_id"], index]:
                raise ValueError("Actual evaluation differs from the audited initial states")
        for manifest in manifests:
            server = manifest["server"]
            if digest(server["experiment_spec"]) != server["inference_fingerprint"] or any(r["inference_fingerprint"] != server["inference_fingerprint"] for r in rows):
                raise ValueError("Episode inference fingerprint differs from its manifest")
            servers.add(server["server_instance_id"])
            core = copy.deepcopy(server["experiment_spec"])
            snapshot = core.pop("temporal_opsd")
            if snapshot["optimizer_step"] != c["step"]:
                raise ValueError("Wrong parameter snapshot was evaluated")
            if c["step"]:
                saved = json.loads((root / "provenance" / ("step_" + str(c["step"]) + ".json")).read_text())
                if {k: snapshot["checkpoint"].get(k) for k in saved} != saved:
                    raise ValueError("Snapshot checkpoint identity mismatch")
            if core_reference is None:
                core_reference = core
            if core_reference != core:
                raise ValueError("Non-parameter inference settings changed within the study")
        key = (c["split"], c["suite"], c["step"], c["horizon"])
        data[key] = rows
        summary = dict(c, **summarize(c["horizon"], rows))
        summaries.append(summary)
        all_rows += [dict(condition=condition_id(c), split=c["split"], optimizer_step=c["step"], **row) for row in rows]
        for task in range(10):
            subset = [r for r in rows if r["task_id"] == task]
            task_rows.append(dict(c, task_id=task, task_description=subset[0]["task_description"],
                                  **summarize(c["horizon"], subset)))
    if len(servers) != 1:
        raise ValueError("This predeclared comparison requires one continuous policy server")
    training = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines() if line]
    rollouts = [json.loads(line) for line in (root / "rollouts.jsonl").read_text().splitlines() if line]
    if [r["optimizer_step"] for r in training] != list(range(101, 501)) or [r["optimizer_step"] for r in rollouts] != list(range(101, 501)):
        raise ValueError("Continuation must contain exactly updates 101–500")
    eval_hashes = {r["initial_state_sha256"] for r in all_rows}
    for row, rollout in zip(training, rollouts):
        if row["diagnostic"] or rollout["diagnostic"] or row["student_behavior_version"] != row["optimizer_step"] - 1:
            raise ValueError("Incorrect on-policy student version")
        for state in rollout["initial_states"]:
            if not 10 <= state["initial_state_index"] < 20 or state["initial_state_sha256"] in eval_hashes:
                raise ValueError("Continuation leaked into evaluation initial states")
    comparisons = []

    def compare(kind, left, right, family):
        comparisons.append({"comparison": kind, "split": left[0], "suite": left[1],
            "left_step": left[2], "left_H": left[3], "right_step": right[2], "right_H": right[3],
            "family": family, **paired_change(data[left], data[right], plan)})

    for key in sorted(data):
        split, suite, step, horizon = key
        baseline = (split, suite, 0, horizon)
        if step and baseline in data:
            primary = split == "confirmation" and step == 500 and horizon == 20
            compare("training_benefit", key, baseline, "primary" if primary else "secondary")
        early = (split, suite, 100, horizon)
        if step > 100 and early in data:
            compare("more_updates", key, early, "secondary")
        high_frequency = (split, suite, 0, 5)
        if step == 0 and horizon != 5 and high_frequency in data:
            compare("replanning_gap", high_frequency, key, "frequency")
    for family in ["primary", "secondary", "frequency"]:
        adjusted = holm({i: r["p_exact"] for i, r in enumerate(comparisons) if r["family"] == family})
        for i, p in adjusted.items():
            comparisons[i]["p_holm"] = p
    primary = next(r for r in comparisons if r["family"] == "primary")
    extra = next(r for r in comparisons if r["split"] == "confirmation" and r["comparison"] == "more_updates")
    output = root / "aggregated"
    for name, rows in [("conditions.csv", summaries), ("comparisons.csv", comparisons), ("per_task.csv", task_rows), ("episodes.csv", all_rows)]:
        write_csv(output / name, rows)
    validation = {"complete": True, "plan_sha256": digest(plan), "evaluation_episodes": len(all_rows),
        "conditions": len(matrix), "single_server_instance": list(servers)[0], "optimizer_updates_added": len(training),
        "all_initial_states_match_audit": True, "training_evaluation_disjoint": True,
        "primary": primary, "additional_updates_confirmation": extra}
    write_json(output / "validation.json", validation)
    return plan, summaries, comparisons, task_rows, training, validation


def plots(root, summaries, training, validation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(root) / "figures"
    output.mkdir(exist_ok=True)
    colors = {0: "#64748b", 100: "#0284c7", 300: "#ca8a04", 500: "#059669"}

    def curve(ax, values, xkey, label, color):
        values = sorted(values, key=lambda s: s[xkey])
        if not values:
            return
        y = np.array([s["success_rate"] for s in values]) * 100
        error = np.array([[100 * max(0., s["success_rate"] - s["ci95_low"]) for s in values],
                          [100 * max(0., s["ci95_high"] - s["success_rate"]) for s in values]])
        ax.errorbar([s[xkey] for s in values], y, yerr=error, fmt="o-", capsize=3, label=label, color=color)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for step in [0, 100, 500]:
        curve(ax, [s for s in summaries if s["split"] == "screen" and s["step"] == step],
              "horizon", "Original" if step == 0 else "{} updates (trained at H=20)".format(step), colors[step])
    ax.set(xlabel="Deployment horizon H", ylabel="LIBERO-10 success (%)", ylim=(0, 100),
           title="Horizon transfer screen · P=50 · 50 episodes/condition", xticks=[5, 15, 20, 25])
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save(fig, "success_vs_replan_horizon.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, color, label in [("screen", "#64748b", "Screen: states 20–24, seed 17, n=50"),
                                ("confirmation", "#059669", "Confirmation: states 30–39, seed 27, n=100")]:
        curve(ax, [s for s in summaries if s["split"] == split and s["horizon"] == 20], "step", label, color)
    ax.set(xlabel="Total optimizer updates", ylabel="LIBERO-10 success (%)", ylim=(0, 100),
           title="More temporal OPSD updates · P=50, H=20", xticks=[0, 100, 300, 500])
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save(fig, "success_vs_training_steps.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    suites = ["libero_spatial", "libero_object", "libero_goal"]
    for index, step in enumerate([0, 100, 500]):
        values = [next(s for s in summaries if s["split"] == "transfer" and s["suite"] == suite and s["step"] == step and s["horizon"] == 20) for suite in suites]
        x = np.arange(3) + (index - 1) * .24
        ax.bar(x, [100*s["success_rate"] for s in values], width=.23,
               label="Original" if step == 0 else "{} updates".format(step), color=colors[step])
        ax.vlines(x, [100*s["ci95_low"] for s in values], [100*s["ci95_high"] for s in values], color="black")
    ax.set(xticks=np.arange(3), xticklabels=[s.replace("libero_", "") for s in suites],
           ylabel="Success (%)", ylim=(0, 100), title="LIBERO suite transfer · P=50, H=20 · n=30 each")
    ax.legend(fontsize=8)
    save(fig, "success_across_suites.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for step in [0, 100, 500]:
        rows = [s for s in summaries if s["split"] == "screen" and s["step"] == step]
        ax.scatter([s["policy_calls_per_environment_step"] for s in rows], [100*s["success_rate"] for s in rows],
                   label="Original" if step == 0 else "{} updates".format(step), color=colors[step])
        for s in rows:
            ax.annotate("H={}".format(s["horizon"]), (s["policy_calls_per_environment_step"], 100*s["success_rate"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Policy calls / controlled environment step", ylabel="Success (%)", ylim=(0, 100),
           title="Compute and accuracy · LIBERO-10 horizon screen")
    ax.legend(fontsize=8)
    save(fig, "success_vs_policy_calls.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["optimizer_step"] for r in training], [r["loss"] for r in training], linewidth=.7)
    ax.axvline(300, color="gray", linestyle="--", label="Saved checkpoint / environment restart")
    ax.set(xlabel="Total optimizer updates", ylabel="Gaussian velocity matching loss",
           title="Continuation uses changing on-policy data and flow times")
    ax.legend(fontsize=8)
    save(fig, "training_loss.png")


def findings(root, summaries, comparisons, tasks, validation):
    p, extra = validation["primary"], validation["additional_updates_confirmation"]
    lookup = {(r["split"], r["suite"], r["step"], r["horizon"]): r for r in summaries}
    confirmation = [lookup["confirmation", "libero_10", s, h] for s, h in [(0, 5), (0, 20), (100, 20), (500, 20)]]
    text = ["# Round 2 findings", "", "## Primary confirmation on different initial layouts", "",
        "The preselected step-500 H=20 model achieved **{:.0%}**, versus **{:.0%}** for the original H=20 model. "
        "The paired change is **{:+.1f} pp**, 95% within-task bootstrap CI **[{:+.1f}, {:+.1f}] pp**, "
        "exact McNemar p=**{:.4g}** (one predeclared primary comparison).".format(p["left_success_rate"], p["right_success_rate"],
          100*p["success_change"], 100*p["ci95_low"], 100*p["ci95_high"], p["p_exact"]), "",
        "There were {} recovered episodes and {} regressions. All 100 pairs use seed 27 and official initial-state "
        "indices 30–39, excluded from our OPSD training and the round-two screen.".format(p["recovered_episodes"], p["regressed_episodes"]), "",
        "| Snapshot | Deployment H | Successes/episodes | Success | Calls/episode |", "|---|---:|---:|---:|---:|"]
    for s in confirmation:
        text.append("| {} | {} | {}/{} | {:.0%} | {:.2f} |".format("Original" if s["step"] == 0 else str(s["step"]) + " updates", s["horizon"], s["total_successes"], s["total_episodes"], s["success_rate"], s["mean_policy_calls"]))
    text += ["", "## Do more steps help?", "",
        "On confirmation, 500 versus 100 updates changed success by **{:+.1f} pp**, paired CI **[{:+.1f}, {:+.1f}] pp**; "
        "raw p={:.4g}, Holm-adjusted secondary-family p={:.4g}.".format(100*extra["success_change"],
         100*extra["ci95_low"], 100*extra["ci95_high"], extra["p_exact"], extra["p_holm"]), "",
        "The 300-update model is a screen checkpoint only. The final checkpoint was fixed at 500 before observing new outcomes; "
        "the screen did not select a winner.", "", "## H=15 and H=25 deployment transfer", "",
        "All added training used H=20. These are deployment-horizon tests of the same parameters on 50 paired episodes per condition, states 20–24, seed 17.", "",
        "| H | Original | 100 updates | 500 updates | 500 − original | Secondary adjusted p |", "|---:|---:|---:|---:|---:|---:|"]
    for h in [15, 20, 25]:
        values = [lookup["screen", "libero_10", s, h]["success_rate"] for s in [0, 100, 500]]
        c = next(c for c in comparisons if c["split"] == "screen" and c["comparison"] == "training_benefit" and c["left_step"] == 500 and c["left_H"] == h)
        text.append("| {} | {:.0%} | {:.0%} | {:.0%} | {:+.1f} pp | {:.4g} |".format(h, *values, 100*c["success_change"], c["p_holm"]))
    text += ["", "## Transfer to other LIBERO suites", "",
        "Only LIBERO-10 supplied our OPSD rollouts. These suites assess transfer/retention of the added OPSD; they are not claimed unseen by the official base checkpoint. "
        "Each screen has only 30 paired episodes (3/task), so small changes are inconclusive.", "",
        "| Suite | Original H=5 | Original H=20 | 100 updates H=20 | 500 updates H=20 |", "|---|---:|---:|---:|---:|"]
    for suite in ["libero_spatial", "libero_object", "libero_goal"]:
        values = [lookup["transfer", suite, step, h]["success_rate"] for step, h in [(0, 5), (0, 20), (100, 20), (500, 20)]]
        text.append("| {} | {:.1%} | {:.1%} | {:.1%} | {:.1%} |".format(suite, *values))
    text += ["", "## Task-level confirmation", "", "Exploratory, 10 episodes/task; no task-wise significance claim.", "",
             "| Task | Original H=20 | 100 updates | 500 updates | Description |", "|---:|---:|---:|---:|---|"]
    for task in range(10):
        values = [next(r for r in tasks if r["split"] == "confirmation" and r["step"] == s and r["horizon"] == 20 and r["task_id"] == task) for s in [0, 100, 500]]
        text.append("| {} | {:.0%} | {:.0%} | {:.0%} | {} |".format(task, *(v["success_rate"] for v in values), values[0]["task_description"]))
    text += ["", "## Limits and reproducibility", "",
        "This tests temporal OPSD with Gaussian velocity matching, an experimental continuous-action adaptation of the local OPSD pipeline; "
        "it is not the image-generation Flow-OPD clipped policy-gradient algorithm. All conditions extrapolate the official checkpoint's native P=10 to P=50, "
        "with 10 unchanged native flow steps. Evaluation calls the official LIBERO execution loop.", "",
        "FP32 master parameters, EMA and Adam moments/counters were restored at step 100. Updates 101–500 kept the learning rate, loss and frozen backbone unchanged. "
        "Training used only official LIBERO-10 start indices 10–19, starting at 12 after the two simulator restarts. All tested layouts were hash-disjoint from these "
        "and the earlier 18 OPSD training layouts. All evaluated snapshots shared one continuous policy server.", "",
        "Wilson intervals describe individual rates; paired differences use 10,000 within-task initial-state bootstrap replicates. "
        "Secondary/screen training comparisons form one Holm family; frequency comparisons form a separate Holm family. "
        "The fixed task set and small single-seed screens do not establish generalization to unseen tasks. Every condition, including regressions, is retained.", "",
        "See aggregated/conditions.csv, comparisons.csv, per_task.csv and episodes.csv; raw records/videos under evaluations/; "
        "all new losses and rollout identities in training.jsonl and rollouts.jsonl. Checkpoint paths and hashes, plan, split audit and "
        "resource accounting are in provenance/. No further hyperparameter sweep was triggered by these outcomes.", ""]
    (Path(root) / "FINDINGS.md").write_text("\n".join(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()
    plan, summaries, comparisons, tasks, training, validation = analyze(args.results_dir)
    plots(args.results_dir, summaries, training, validation)
    findings(args.results_dir, summaries, comparisons, tasks, validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
