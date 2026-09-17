"""Keep the measured continuation direction, snapshot labels and counts correct."""
import csv
import json

from frequency_vla.analysis import summarize
from frequency_vla.continuation_analysis import summarize_continuation, report
from frequency_vla.logging_utils import write_json


def test_continuation_report_keeps_parent_and_final_measurements_separate(tmp_path):
    plan = dict(resume_step=500, milestones=[500, 1000], renderer="osmesa",
                resume_manifest_sha256="parent", primary_comparison="step1000_minus_step500_H20",
                bootstrap_replicates=1000, analysis_seed=7)
    data, summaries, tasks, all_rows = {}, [], [], []
    stage_records = []
    for step, success_limit in [(500, 6), (1000, 8)]:
        rows = []
        for task in range(10):
            for episode in range(10):
                identity = "{}:{}".format(task, episode)
                rows.append(dict(seed=27, task_id=task, episode_index=episode, initial_state_index=30+episode,
                    initial_state_sha256=identity, first_observation_sha256=identity,
                    episode_rng_seed=episode, evaluation_fingerprint="same", task_description="task "+str(task),
                    prediction_horizon=50, native_prediction_horizon=10, replan_steps=20, success=episode < success_limit,
                    controlled_environment_steps=100 if episode < success_limit else 520,
                    environment_steps=110 if episode < success_limit else 530, policy_calls=5 if episode < success_limit else 26,
                    average_executed_actions_per_policy_call=20, wall_clock_seconds=1.0))
        condition = dict(split="confirmation", suite="libero_10", step=step, horizon=20)
        data["confirmation", "libero_10", step, 20] = rows
        summaries.append(dict(condition, **summarize(20, rows)))
        all_rows.extend(dict(optimizer_step=step, **r) for r in rows)
        tasks.extend(dict(condition, task_id=t, task_description="task "+str(t),
                          **summarize(20, [r for r in rows if r["task_id"] == t])) for t in range(10))
        stage_records.append(dict(stage="confirmation_libero_10_step{}_H20".format(step), event="complete", seconds=100))
    (tmp_path / "stages.jsonl").write_text("\n".join(json.dumps(r) for r in stage_records))
    write_json(tmp_path / "provenance/training_setup.json", dict(config=dict(optimizer_steps=1000)))
    write_json(tmp_path / "provenance/resume.json", dict(manifest_sha256="parent", fp32_master_restored=True,
               ema_and_optimizer_restored=True, frozen_backbone_equal=True))
    training = [dict(optimizer_step=s, loss=.01) for s in range(501, 1001)]
    rollouts = [dict(initial_states=[dict(task_id=t, initial_state_index=16, executed_steps=20) for t in range(4)]) for _ in training]
    result = summarize_continuation(tmp_path, plan, data, summaries, tasks, all_rows, {"one server"}, training, rollouts)
    validation = result[-1]
    assert validation["primary"]["success_change"] == .2
    assert validation["primary"]["recovered_episodes"] == 20
    assert validation["primary"]["regressed_episodes"] == 0
    assert validation["optimizer_updates_added"] == 500
    assert validation["evaluation_episodes"] == 200
    with (tmp_path / "aggregated/conditions.csv").open() as f:
        measured = list(csv.DictReader(f))
    assert [(r["step"], r["total_successes"]) for r in measured] == [("500", "60"), ("1000", "80")]
    report(tmp_path, plan, summaries, tasks, training, validation)
    assert "60/100" in (tmp_path / "FINDINGS.md").read_text()
    assert "80/100" in (tmp_path / "FINDINGS.md").read_text()
    assert (tmp_path / "figures/continuation_comparison.png").stat().st_size > 0
