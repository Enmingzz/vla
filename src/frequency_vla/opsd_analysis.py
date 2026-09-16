"""Compare the held-out, paired H=20 policy before and after 100 updates."""
import argparse
import json
from pathlib import Path

from .analysis import exact_mcnemar, paired_interval, pair_key, summarize, validate_records, write_csv
from .logging_utils import write_json


def load_condition(root):
    source = root / "raw/smoke/libero_10/seed_7/H_20"
    manifests = [json.loads(p.read_text()) for p in source.glob("*.manifest.json")]
    rows = [json.loads(line) for p in sorted(source.glob("*.jsonl")) for line in p.read_text().splitlines() if line]
    if len(manifests) != 10 or any(m["status"] != "complete" for m in manifests):
        raise ValueError("Incomplete held-out evaluation: " + str(root))
    validate_records(rows)
    expected = {(7, task, episode) for task in range(10) for episode in range(10)}
    if len(rows) != 100 or {pair_key(r) for r in rows} != expected or any(r["replan_steps"] != 20 for r in rows):
        raise ValueError("Require exactly 100 paired H=20 episodes")
    return {pair_key(r): r for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)
    before, after = load_condition(root / "baseline"), load_condition(root / "student_100")
    server_specs = [json.loads((root / "provenance" / ("server_" + phase + ".json")).read_text())["experiment_spec"]
                    for phase in ["baseline", "student"]]
    phases = [spec.pop("temporal_opsd") for spec in server_specs]
    if server_specs[0] != server_specs[1]:
        raise ValueError("Inference configuration changed between baseline and trained evaluation")
    if phases[0]["phase"] != "baseline" or phases[0]["optimizer_step"] != 0 or phases[1]["phase"] != "student" or phases[1]["optimizer_step"] != 100:
        raise ValueError("Evaluation phases do not identify the initial and step-100 snapshots")
    exported = json.loads((root / "provenance/step_100.json").read_text())
    evaluated = phases[1]["checkpoint"]
    # The native WebSocket server adds server_timing to RPC return dictionaries.
    # Compare checkpoint identity, not transport timing annotations.
    if {key: evaluated.get(key) for key in exported} != exported:
        raise ValueError("Evaluated student does not identify the exported step-100 checkpoint")
    for field in ["algorithm", "sources", "evaluation_sampler"]:
        if phases[0][field] != phases[1][field]:
            raise ValueError("Evaluation implementation changed: " + field)
    for key in sorted(before):
        for field in ["initial_state_sha256", "first_observation_sha256", "episode_rng_seed", "evaluation_fingerprint",
                      "prediction_horizon", "native_prediction_horizon", "task_description"]:
            if before[key][field] != after[key][field]:
                raise ValueError("Unpaired before/after evaluation: " + str((key, field)))
        if before[key]["inference_fingerprint"] == after[key]["inference_fingerprint"]:
            raise ValueError("Before/after parameter provenance was not changed")
    training = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines() if line]
    if [r["optimizer_step"] for r in training] != list(range(1, 101)) or any(r["diagnostic"] for r in training):
        raise ValueError("Require exactly 100 formal optimizer updates")
    rollouts = [json.loads(line) for line in (root / "rollouts.jsonl").read_text().splitlines() if line]
    if len(rollouts) != 100:
        raise ValueError("Missing on-policy rollout provenance")
    eval_hashes = {r["initial_state_sha256"] for r in before.values()}
    train_hashes = {s["initial_state_sha256"] for r in rollouts for s in r["initial_states"]}
    if eval_hashes & train_hashes:
        raise ValueError("Training/evaluation initial-state leakage")
    if [r["optimizer_step"] for r in rollouts] != list(range(1, 101)) or any(r["diagnostic"] for r in rollouts):
        raise ValueError("Rollout versions do not match the 100 formal updates")
    training_steps = sum(s["executed_steps"] for r in rollouts for s in r["initial_states"])
    task_coverage = [{"task_id": task,
                      "action_blocks": sum(s["task_id"] == task for r in rollouts for s in r["initial_states"]),
                      "controlled_steps": sum(s["executed_steps"] for r in rollouts for s in r["initial_states"] if s["task_id"] == task),
                      "unique_initial_states": len({s["initial_state_sha256"] for r in rollouts for s in r["initial_states"] if s["task_id"] == task})}
                     for task in range(10)]
    pairs = [(after[key], before[key]) for key in sorted(before)]
    baseline, trained = summarize(20, list(before.values())), summarize(20, list(after.values()))
    only_after = sum(a["success"] and not b["success"] for a, b in pairs)
    only_before = sum(b["success"] and not a["success"] for a, b in pairs)
    low, high = paired_interval(pairs)
    delta = trained["success_rate"] - baseline["success_rate"]
    prior_gap = json.loads((root / "provenance/hypothesis_gate.json").read_text())["gap"]
    teacher_reference = float(prior_gap["teacher_success_rate"])
    recovered_fraction = delta / float(prior_gap["replanning_gap"])
    stats = {"paired_episodes": 100, "optimizer_steps": 100,
             "baseline_success_rate": baseline["success_rate"], "trained_success_rate": trained["success_rate"],
             "success_change": delta, "change_ci95_low": low, "change_ci95_high": high,
             "trained_only_successes": only_after, "baseline_only_successes": only_before,
             "mcnemar_p_two_sided": exact_mcnemar(only_after, only_before),
             "train_eval_initial_states_disjoint": True, "inference_config_unchanged": True,
             "exported_step_100_evaluated": True, "same_H": 20, "same_P": 50,
             "training_controlled_steps": training_steps, "unique_training_initial_states": len(train_hashes),
             "prior_H5_success_reference": teacher_reference,
             "contextual_fraction_of_prior_gap_recovered": recovered_fraction,
             "mean_calls_before": baseline["mean_policy_calls"], "mean_calls_after": trained["mean_policy_calls"]}
    output = root / "aggregated"
    write_json(output / "comparison.json", stats)
    write_csv(output / "comparison.csv", [dict(condition="untrained", **baseline), dict(condition="step_100", **trained)])
    write_csv(output / "training_task_coverage.csv", task_coverage)
    write_csv(output / "episodes.csv", [dict(condition=name, **r) for name, rows in [("untrained", before), ("step_100", after)] for r in rows.values()])
    tasks = []
    for task in range(10):
        b = [r for r in before.values() if r["task_id"] == task]
        a = [r for r in after.values() if r["task_id"] == task]
        tasks.append({"task_id": task, "task_description": b[0]["task_description"],
                      "success_before": sum(r["success"] for r in b) / 10,
                      "success_after": sum(r["success"] for r in a) / 10})
    write_csv(output / "per_task.csv", tasks)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    summaries = [baseline, trained]
    values = [s["success_rate"] * 100 for s in summaries]
    errors = [[100 * (s["success_rate"] - s["ci95_low"]) for s in summaries],
              [100 * (s["ci95_high"] - s["success_rate"]) for s in summaries]]
    ax.bar(["Before training", "After 100 updates"], values, yerr=errors, capsize=5, color=["#64748b", "#0284c7"])
    for x, value in enumerate(values):
        ax.text(x, value / 2, f"{value:.0f}%", ha="center", va="center", color="white", fontsize=14)
    ax.set(ylabel="LIBERO-10 success (%)", ylim=(0, 100), title="Temporal OPSD · fixed P=50, H=20")
    fig.text(.5, .02, f"Paired change {delta*100:+.0f} pp; 95% CI [{low*100:+.0f}, {high*100:+.0f}] pp. Bars: Wilson 95% CI.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(figures / "success_before_after_100.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["optimizer_step"] for r in training], [r["loss"] for r in training])
    ax.set(xlabel="Optimizer update", ylabel="Gaussian velocity-matching loss", title="100 on-policy updates; no evaluation samples used")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures / "training_loss.png", dpi=180)
    plt.close(fig)
    convincing = delta > 0 and low > 0 and stats["mcnemar_p_two_sided"] < .05
    task_table = "\n".join(f"| {row['task_id']} | {row['success_before']:.0%} | {row['success_after']:.0%} | "
                           f"{100*(row['success_after']-row['success_before']):+.0f} pp | {row['task_description']} |"
                           for row in tasks)
    text = f'''# First 100-step temporal OPSD result

At fixed P=50 and H=20, held-out LIBERO-10 success changed from
**{baseline['total_successes']}/100 ({baseline['success_rate']:.0%})** to
**{trained['total_successes']}/100 ({trained['success_rate']:.0%})** after 100 optimizer updates.
The paired change is **{delta*100:+.1f} percentage points**, 95% within-task paired
bootstrap CI **[{low*100:+.1f}, {high*100:+.1f}] pp**; exact McNemar
p = **{stats['mcnemar_p_two_sided']:.4g}**.

{'This pilot provides paired evidence of improvement on these held-out initial states.' if convincing else 'This pilot does not establish a statistically convincing improvement.'}
The measured direction is reported regardless of the training loss.
This single-seed pilot requires independent confirmation before a strong recovery
claim; no training hyperparameters were selected using its success results.

## Recovery and task changes

There were {only_after} failures converted to successes and {only_before} successes
converted to failures. Relative to the preceding H=5 reference of
{teacher_reference:.0%}, the trained student's remaining gap is
{100*(teacher_reference-trained['success_rate']):.0f} pp. The measured improvement
corresponds to {recovered_fraction:.1%} of the earlier H=5/H=20 gap. This recovery
fraction is contextual: H=5 was measured in the preceding frequency job, not
re-evaluated in this training job. The paired before/after H=20 comparison above
is the primary training result.

Each task has only 10 evaluation episodes; the following changes are exploratory.

| Task | Before | After | Change | Description |
|---:|---:|---:|---:|---|
{task_table}

## Protocol

Both conditions used the same continuous server and the native OpenPI inference
sampler, seed 7, and official initial-state indices 0–9 for every LIBERO-10 task.
Every episode executed at most 20 actions before replanning, with P=50 and 10 flow
steps. The only intended before/after variable was the student parameters.
Mean policy calls per episode: {baseline['mean_policy_calls']:.2f} before,
{trained['mean_policy_calls']:.2f} after. Calls per fixed-length trajectory remain
approximately one per 20 steps; episode lengths can change with success.

Training used 100 current-student batches of four action blocks, seed 17, and an
eligible initial-state pool of indices 10–49. Initial-state hashes were verified disjoint from
evaluation. The run collected {training_steps:,} controlled training steps across
{len(train_hashes)} distinct initial states; per-task exposure is reported in
aggregated/training_task_coverage.csv. One diagnostic update was saved/reloaded and fully rolled back before
the formal baseline. Only the existing action expert and action/time projections
were optimized; the visual/language backbone was outside the optimizer. The
teacher was an EMA, decay 0.9999. No demonstrations or success rewards were used
in the loss. The step-100 checkpoint was selected in advance.

## What was tested

This is an experimental continuous-action adaptation of the local OPSD pipeline:
student-controlled rollouts, fresh teacher views every five physical steps,
detached teacher velocity targets on aligned student denoising states, and an EMA
teacher. Auxiliary teacher samples complete missing latent tails. The objective
matches local Gaussian velocity probes; it is not the KL of the full chunk
distribution and is not the image-generation Flow-OPD paper's clipped policy
gradient algorithm. See the repository's AUTORESEARCH.md for the specified method.

The official checkpoint was fine-tuned at P=10; both tested conditions extrapolate
to P=50. This is an initial 100-episode held-out pilot, not a 500-episode main
validation or proof of recovery on unseen tasks. Per-task changes are exploratory.
These evaluation states were excluded from optimizer training but were already
used in the preceding H sweep; they are not a fresh final test set. Expanding to
all official states later would include the training states and must not be
labelled a fully held-out evaluation without a new split.
Further tests must not reuse these results for unreported hyperparameter selection.

Raw before/after records and videos are in baseline/ and student_100/. Training
logs are training.jsonl and rollouts.jsonl; the checkpoint location and digest are
in provenance/step_100.json. Aggregated CSVs and figures are in aggregated/ and figures/.
'''
    (root / "FINDINGS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
