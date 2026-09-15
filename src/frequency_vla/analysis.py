"""Raw-data validation, Wilson intervals, paired gaps, and reproducible plots."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from .config import load_config
from .evaluator import validate_call_schedule
from .logging_utils import write_json


def wilson(successes, n, z=1.959963984540054):
    if n <= 0:
        raise ValueError("An interval requires measured episodes")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar(teacher_only, student_only):
    discordant = teacher_only + student_only
    if discordant == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(teacher_only, student_only) + 1)) / 2**discordant)


def holm(pvalues):
    result, previous = {}, 0.0
    ordered = sorted(pvalues, key=pvalues.get)
    for rank, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * pvalues[key]))
        result[key] = previous
    return result


def paired_interval(pairs, replicates=10000, seed=20260915):
    import numpy as np
    # Fixed benchmark tasks; bootstrap initial-state blocks within each task.
    # Repeated evaluation seeds of the same initial state stay together.
    blocks = defaultdict(lambda: defaultdict(list))
    for teacher, student in pairs:
        blocks[teacher["task_id"]][teacher["episode_index"]].append(int(teacher["success"]) - int(student["success"]))
    if not blocks:
        raise ValueError("No paired observations")
    rng = np.random.default_rng(seed)
    draws = np.zeros(replicates)
    total = sum(len(states) for states in blocks.values())
    for task_id in sorted(blocks):
        values = np.array([np.mean(v) for _, v in sorted(blocks[task_id].items())])
        draws += values[rng.integers(0, len(values), size=(replicates, len(values)))].sum(axis=1)
    draws /= total
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def pair_key(row):
    return row["seed"], row["task_id"], row["episode_index"]


def validate_records(records, allow_partial=False):
    if not records:
        raise ValueError("No measured episode records. No placeholder results will be generated.")
    fingerprints = {r["inference_fingerprint"] for r in records}
    if len(fingerprints) != 1:
        raise ValueError("Cannot aggregate changed checkpoints/configs/inference settings")
    if len({r["evaluation_fingerprint"] for r in records}) != 1:
        raise ValueError("Cannot aggregate changed evaluator code or simulator dependencies")
    by_h = defaultdict(dict)
    reference_states = {}
    for row in records:
        h, key = row["replan_steps"], pair_key(row)
        if key in by_h[h]:
            raise ValueError("Duplicate episode: " + str((h, key)))
        if type(row["success"]) is not bool:
            raise ValueError("Success must be a measured boolean")
        validate_call_schedule(row["controlled_environment_steps"], row["policy_calls"], h, row["policy_call_control_steps"])
        if row["environment_steps"] != row["controlled_environment_steps"] + row["settling_steps"]:
            raise ValueError("Inconsistent environment step counts")
        identity = (row["initial_state_sha256"], row["first_observation_sha256"], row["episode_rng_seed"], row["task_description"])
        if key in reference_states and reference_states[key] != identity:
            raise ValueError("Unpaired initial state/first observation/RNG: " + str(key))
        reference_states[key] = identity
        by_h[h][key] = row
    if not allow_partial:
        keys = [set(group) for group in by_h.values()]
        if any(k != keys[0] for k in keys[1:]):
            raise ValueError("Horizons have different episode coverage; wait for completion")
    return by_h


def mean(rows, field):
    return sum(r[field] for r in rows) / len(rows)


def summarize(h, rows):
    successes = sum(r["success"] for r in rows)
    low, high = wilson(successes, len(rows))
    return {"H": h, "total_episodes": len(rows), "total_successes": successes,
            "success_rate": successes / len(rows), "ci95_low": low, "ci95_high": high,
            "mean_policy_calls": mean(rows, "policy_calls"),
            "mean_episode_length": mean(rows, "controlled_environment_steps"),
            "mean_environment_steps_including_settling": mean(rows, "environment_steps"),
            "policy_calls_per_environment_step": sum(r["policy_calls"] for r in rows) / sum(r["controlled_environment_steps"] for r in rows),
            "mean_executed_actions_per_policy_call": mean(rows, "average_executed_actions_per_policy_call"),
            "mean_wall_clock_seconds": mean(rows, "wall_clock_seconds")}


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def build_tables(records, config, allow_partial=False):
    by_h = validate_records(records, allow_partial)
    summaries, per_task, per_seed, gaps = [], [], [], []
    for h, group in sorted(by_h.items()):
        rows = list(group.values())
        summary = summarize(h, rows)
        for task_id in sorted({r["task_id"] for r in rows}):
            task_rows = [r for r in rows if r["task_id"] == task_id]
            task_summary = summarize(h, task_rows)
            task_summary.update(task_id=task_id, task_description=task_rows[0]["task_description"])
            per_task.append(task_summary)
            summary["task_{}_success_rate".format(task_id)] = task_summary["success_rate"]
        for seed in sorted({r["seed"] for r in rows}):
            per_seed.append(dict(summarize(h, [r for r in rows if r["seed"] == seed]), seed=seed))
        summaries.append(summary)
    teacher = by_h.get(config["teacher_horizon"], {})
    task_gaps = []
    for h, group in sorted(by_h.items()):
        if h == config["teacher_horizon"] or not teacher:
            continue
        common = sorted(set(teacher) & set(group))
        pairs = [(teacher[k], group[k]) for k in common]
        if not pairs:
            continue
        trows, srows = [p[0] for p in pairs], [p[1] for p in pairs]
        teacher_only = sum(t["success"] and not s["success"] for t, s in pairs)
        student_only = sum(s["success"] and not t["success"] for t, s in pairs)
        low, high = paired_interval(pairs, config["bootstrap_replicates"], config["analysis_seed"])
        t_density = sum(r["policy_calls"] for r in trows) / sum(r["controlled_environment_steps"] for r in trows)
        s_density = sum(r["policy_calls"] for r in srows) / sum(r["controlled_environment_steps"] for r in srows)
        gap = {"teacher_H": config["teacher_horizon"], "student_H": h, "paired_episodes": len(pairs),
               "teacher_success_rate": mean(trows, "success"), "student_success_rate": mean(srows, "success"),
               "replanning_gap": mean(trows, "success") - mean(srows, "success"),
               "gap_ci95_low": low, "gap_ci95_high": high,
               "teacher_only_successes": teacher_only, "student_only_successes": student_only,
               "policy_calls_saved_per_episode": mean(trows, "policy_calls") - mean(srows, "policy_calls"),
               "relative_policy_calls_saved": 1 - mean(srows, "policy_calls") / mean(trows, "policy_calls"),
               "relative_call_density_saved": 1 - s_density / t_density}
        if len({r["seed"] for r in trows}) == 1:
            gap["mcnemar_p_two_sided"] = exact_mcnemar(teacher_only, student_only)
        gaps.append(gap)
        for task_id in sorted({r["task_id"] for r in trows}):
            tpairs = [(t, s) for t, s in pairs if t["task_id"] == task_id]
            task_gaps.append({"student_H": h, "task_id": task_id,
                              "task_description": tpairs[0][0]["task_description"], "paired_episodes": len(tpairs),
                              "replanning_gap": sum(int(t["success"]) - int(s["success"]) for t, s in tpairs) / len(tpairs)})
    adjusted = holm({g["student_H"]: g["mcnemar_p_two_sided"] for g in gaps if "mcnemar_p_two_sided" in g})
    for g in gaps:
        if g["student_H"] in adjusted:
            g["mcnemar_p_holm"] = adjusted[g["student_H"]]
    return summaries, per_task, per_seed, gaps, task_gaps


def plots(summaries, gaps, output, suite, mode, native_p, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x = [s["H"] for s in summaries]
    y = [100 * s["success_rate"] for s in summaries]
    errors = [[100 * max(0., s["success_rate"] - s["ci95_low"]) for s in summaries],
              [100 * max(0., s["ci95_high"] - s["success_rate"]) for s in summaries]]
    title = "{} · {} · native P={}".format(suite, mode, native_p)
    note = "H > {} is unsupported with the unchanged official checkpoint config.".format(native_p) if max(config["horizons"]) > native_p else ""

    def save(fig, filename):
        if note:
            fig.text(0.5, 0.015, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, y, yerr=errors, fmt="o-", capsize=4, label="Measured success; 95% Wilson CI")
    if suite == "libero_10":
        ax.axhline(config["official_libero_10_success"] * 100, color="gray", linestyle="--", label="Official reference: 92.4%")
    ax.set(xlabel="Replanning horizon H (actions)", ylabel="Success rate (%)", title=title, ylim=(0, 100))
    ax.set_xticks(config["horizons"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, "success_vs_replan_horizon.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="gray", linewidth=1)
    if gaps:
        gx = [g["student_H"] for g in gaps]
        gy = [100 * g["replanning_gap"] for g in gaps]
        # Percentile intervals can exclude a discrete point estimate; render bounds directly.
        ax.plot([5] + gx, [0] + gy, "o-")
        ax.vlines(gx, [100 * g["gap_ci95_low"] for g in gaps], [100 * g["gap_ci95_high"] for g in gaps])
    else:
        ax.text(0.5, 0.5, "No paired comparison measured yet", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Replanning horizon H (actions)", ylabel="Success(5) − Success(H) (percentage points)", title=title)
    ax.set_xticks(config["horizons"])
    ax.grid(alpha=0.25)
    save(fig, "relative_performance_drop.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    calls = [s["mean_policy_calls"] for s in summaries]
    ax.errorbar(calls, y, yerr=errors, fmt="o", capsize=4)
    for cx, cy, h in zip(calls, y, x):
        ax.annotate("H={}".format(h), (cx, cy), xytext=(5, -12), textcoords="offset points")
    ax.set(xlabel="Mean policy calls per episode", ylabel="Success rate (%)", title=title, ylim=(0, 100))
    ax.grid(alpha=0.25)
    save(fig, "success_vs_policy_calls.png")


def findings(summaries, gaps, task_gaps, config, suite, mode, native_p, complete):
    teacher = next((s for s in summaries if s["H"] == 5), None)
    baseline_ok = bool(teacher and suite == "libero_10" and teacher["success_rate"] >= config["official_libero_10_success"] - config["baseline_max_absolute_drop"])
    lines = ["# Findings", "", "Measured {} evaluation on {}. Coverage: {}.".format(mode, suite, "complete for the explicitly selected horizons" if complete else "PARTIAL; exploratory only"), "",
             "## Protocol constraint", "", "The pinned official `pi05_libero` config has native prediction horizon P={}. Requested horizons above P were not executed. No model/config override, action padding, repetition, or hidden policy calls was used.".format(native_p), "",
             "OpenPI commit: `{}`. Checkpoint: `{}`. Flow steps: {} (upstream default).".format(config["openpi_commit"], config["checkpoint"], config["flow_steps"]), "",
             "## Measured success", "", "| H | Successes / episodes | Success | 95% Wilson CI | Calls/episode | Controlled steps/episode |", "|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append("| {H} | {total_successes}/{total_episodes} | {rate:.1%} | [{low:.1%}, {high:.1%}] | {mean_policy_calls:.2f} | {mean_episode_length:.1f} |".format(**s, rate=s["success_rate"], low=s["ci95_low"], high=s["ci95_high"]))
    lines.extend(["", "## H=5 reproduction gate", ""])
    if teacher and suite == "libero_10":
        lines.append("H=5 measured {:.1%}, versus the official 92.4% reference. The predeclared diagnostic gate allows at most a {:.0f} percentage point deficit: {}. This gate is a debugging screen, not an equivalence test; the public reference has no reported uncertainty here.".format(teacher["success_rate"], 100*config["baseline_max_absolute_drop"], "PASS" if baseline_ok else "FAIL — investigate before interpreting gaps"))
    else:
        lines.append("No applicable LIBERO-10 H=5 baseline is available; reproduction is not established.")
    lines.extend(["", "## Paired replanning gaps and compute savings", "", "Gap is Success(5) − Success(H), measured on matching seed/task/initial-state indices. Positive values favour H=5.", ""])
    for g in gaps:
        lines.append("- H={}: gap {:.1f} pp, paired 95% bootstrap CI [{:.1f}, {:.1f}] pp; {:.1%} fewer calls per episode and {:.1%} fewer calls per controlled step ({} pairs).{}".format(g["student_H"], 100*g["replanning_gap"], 100*g["gap_ci95_low"], 100*g["gap_ci95_high"], g["relative_policy_calls_saved"], g["relative_call_density_saved"], g["paired_episodes"], " Exact McNemar p (Holm-adjusted) = {:.4g}.".format(g["mcnemar_p_holm"]) if "mcnemar_p_holm" in g else ""))
    if not gaps:
        lines.append("No comparison has been measured. No frequency effect can be inferred.")
    lines.extend(["", "## Interpretation", ""])
    substantial = [g for g in gaps if g["replanning_gap"] >= config["substantial_absolute_drop"]]
    convincing = [g for g in substantial if g["gap_ci95_low"] > 0 and g.get("mcnemar_p_holm", 0) < 0.05]
    if gaps:
        lines.append("The measured difference {} a monotonic decrease over all requested H values; H>{} was not measurable under this protocol.".format("does not establish", native_p))
        lines.append("Substantial degradation (predeclared ≥5 percentage points): {}.".format(", ".join("H="+str(g["student_H"]) for g in substantial) or "none of the measured horizons"))
        lines.append("Tasks with the largest observed positive gaps (exploratory; no task-wise significance claim):")
        positives = sorted([g for g in task_gaps if g["replanning_gap"] > 0], key=lambda g: -g["replanning_gap"])
        for g in positives[:5]:
            lines.append("- Task {} at H={}: {:.1f} pp — {}.".format(g["task_id"], g["student_H"], 100*g["replanning_gap"], g["task_description"]))
        if not positives:
            lines.append("- No task has a positive measured gap.")
    candidates = [g for g in convincing if baseline_ok and g["relative_call_density_saved"] >= 0.3 and 0 < g["replanning_gap"] <= 0.25]
    if candidates and complete and mode == "main":
        candidate = max(candidates, key=lambda g: g["relative_call_density_saved"])
        lines.append("Among measured settings, (H_T,H_S)=(5,{}) is an exploratory candidate: at least 30% fewer calls per controlled step, a 5–25 pp gap, and paired evidence favouring H=5. Candidate selection is exploratory and needs confirmation.".format(candidate["student_H"]))
    else:
        lines.append("No teacher/student pair is recommended from the evidence currently available.")
    lines.append("The intended H_S=20/30/50 premise remains untested because native P={}. These results alone cannot justify that proposed distillation stage; it requires a revised, explicitly authorized protocol. No training or distillation was implemented.".format(native_p))
    if mode != "main" or not complete:
        lines.append("Smoke/partial results are diagnostic, not the requested main validation.")
    lines.extend(["", "## Reproducibility and uncertainty", "", "Environment settling, image rotation/resize, state conversion, chunk-prefix execution, success termination and task step limits come directly from the pinned official evaluator. Initial-state and first-policy-observation hashes, per-episode RNG seeds, and inference fingerprints are validated across H. Actual call positions must equal 0,H,2H,… .", "", "Wilson intervals describe the episode-level binomial rate. Gap intervals use paired initial-state blocks resampled within fixed tasks (10,000 replicates, fixed analysis seed); repeated seeds of one initial state stay in the same block. Exact McNemar tests are reported for single-seed runs, with Holm correction across measured candidate horizons. The task set is fixed; intervals do not establish generalization to unseen tasks. Video encoding time is excluded from episode duration; first-episode JIT compilation is included.", ""])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[2] / "results"))
    p.add_argument("--mode", choices=["smoke", "main", "diagnostic"], default="main")
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--findings-out")
    args = p.parse_args()
    root, config = Path(args.results_dir), load_config()
    source = root / "raw" / args.mode / args.suite
    records = [json.loads(line) for path in sorted(source.rglob("*.jsonl")) for line in path.read_text().splitlines() if line.strip()]
    summaries, tasks, seeds, gaps, task_gaps = build_tables(records, config, args.allow_partial)
    complete = True
    expected_n = config["modes"].get(args.mode)
    for h in {r["replan_steps"] for r in records}:
        for seed in {r["seed"] for r in records}:
            coverage = {(r["task_id"], r["episode_index"]) for r in records if r["replan_steps"] == h and r["seed"] == seed}
            if expected_n is None or coverage != {(t, e) for t in range(10) for e in range(expected_n)}:
                complete = False
    manifests = [json.loads(path.read_text()) for path in source.rglob("*.manifest.json")]
    if any(m.get("project_config") != config for m in manifests):
        raise ValueError("Analysis configuration differs from the recorded protocol; restore the recorded config before aggregation")
    if not manifests or any(m["status"] != "complete" for m in manifests):
        complete = False
    if not complete and not args.allow_partial:
        raise ValueError("Incomplete protocol coverage; use --allow-partial only for explicitly labelled diagnostics")
    out = root / "aggregated" / args.mode / args.suite
    for name, rows in [("frequency_sweep.csv", summaries), ("per_task.csv", tasks), ("per_seed.csv", seeds), ("replanning_gaps.csv", gaps), ("task_gaps.csv", task_gaps), ("episodes.csv", records)]:
        write_csv(out / name, rows)
    native_p = records[0]["native_prediction_horizon"]
    baseline = next((s for s in summaries if s["H"] == 5), None)
    gate = bool(baseline and args.suite == "libero_10" and baseline["success_rate"] >= config["official_libero_10_success"] - config["baseline_max_absolute_drop"])
    write_json(out / "validation.json", {"complete_for_measured_horizons": complete, "baseline_gate_passed": gate,
        "measured_horizons": [s["H"] for s in summaries], "unsupported_requested_horizons": [h for h in config["horizons"] if h > native_p],
        "paired_checks_passed": True, "native_prediction_horizon": native_p, "inference_fingerprint": records[0]["inference_fingerprint"]})
    plots(summaries, gaps, root / "figures" / args.mode / args.suite, args.suite, args.mode, native_p, config)
    text = findings(summaries, gaps, task_gaps, config, args.suite, args.mode, native_p, complete)
    (out / "FINDINGS.md").write_text(text)
    if args.findings_out:
        Path(args.findings_out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
