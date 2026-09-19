"""Describe successful-episode action counts without pooling renderer protocols."""
import argparse
import json
from pathlib import Path

from frequency_vla.analysis import pair_key, validate_records, write_csv
from frequency_vla.logging_utils import file_digest, write_json


def mean(rows, field):
    return sum(row[field] for row in rows) / len(rows) if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", default="results/autoresearch_round2")
    parser.add_argument("--continuation-results", default="results/autoresearch_round3_finish_retry1")
    parser.add_argument("--output", default="results/successful_actions")
    args = parser.parse_args()
    output = Path(args.output)
    configurations = [
        ("Original H5", args.reference_results, 0, 5, "EGL (archived protocol)"),
        ("Original H20", args.reference_results, 0, 20, "EGL (archived protocol)"),
        ("OPSD 500 H20 historical", args.reference_results, 500, 20, "EGL (archived protocol)"),
        ("OPSD 500 H20 current", args.continuation_results, 500, 20, "OSMesa"),
        ("OPSD 1000 H20 current", args.continuation_results, 1000, 20, "OSMesa"),
    ]
    summaries, data, source_files, manifest_files, selected_episodes = [], {}, {}, {}, []
    for label, root, step, horizon, renderer in configurations:
        folder = Path(root) / "evaluations/confirmation" / ("step_" + str(step)) / "raw/smoke/libero_10/seed_27" / ("H_" + str(horizon))
        files = sorted(folder.glob("*.jsonl"))
        rows = [json.loads(line) for p in files for line in p.read_text().splitlines() if line]
        validate_records(rows)
        manifests = sorted(folder.glob("*.manifest.json"))
        if len(manifests) != 10 or any(json.loads(p.read_text())["status"] != "complete" for p in manifests):
            raise ValueError("Incomplete condition: " + label)
        if len(rows) != 100 or {pair_key(r) for r in rows} != {(27, t, e) for t in range(10) for e in range(10)}:
            raise ValueError("Unexpected episode coverage: " + label)
        if any(r["task_suite"] != "libero_10" or r["prediction_horizon"] != 50 or
               r["replan_steps"] != horizon or r["initial_state_index"] != 30 + r["episode_index"] for r in rows):
            raise ValueError("Unexpected evaluation protocol: " + label)
        data[label] = {pair_key(r): r for r in rows}
        successful = [r for r in rows if r["success"]]
        summaries.append(dict(condition=label, step=step, H=horizon, P=50, renderer=renderer,
            total_episodes=len(rows), successes=len(successful),
            success_rate=len(successful)/len(rows),
            successful_action_sum=sum(r["controlled_environment_steps"] for r in successful),
            mean_actions_success=mean(successful, "controlled_environment_steps"),
            mean_policy_calls_success=mean(successful, "policy_calls")))
        source_files.update({str(p): file_digest(p) for p in files})
        manifest_files.update({str(p): file_digest(p) for p in manifests})
        selected_episodes.extend(dict(condition=label, task_id=r["task_id"], episode_index=r["episode_index"],
            initial_state_index=r["initial_state_index"], success=r["success"],
            executed_actions=r["controlled_environment_steps"], policy_calls=r["policy_calls"]) for r in rows)
    reference = data["Original H5"]
    comparisons = []
    for label, rows in data.items():
        if label == "Original H5":
            continue
        for key in reference:
            for field in ["initial_state_index", "initial_state_sha256", "episode_rng_seed", "task_description", "prediction_horizon"]:
                if reference[key][field] != rows[key][field]:
                    raise ValueError("Cannot match initial layouts/RNG: " + str((label, key, field)))
        common = sorted(k for k in reference if reference[k]["success"] and rows[k]["success"])
        before, after = [[group[k] for k in common] for group in [reference, rows]]
        left, right = [mean(group, "controlled_environment_steps") for group in [before, after]]
        comparisons.append(dict(condition=label, common_success_episodes=len(common),
            H5_mean_actions=left, candidate_mean_actions=right,
            candidate_relative_extra_actions=right/left-1 if left else None,
            H5_mean_policy_calls=mean(before, "policy_calls"), candidate_mean_policy_calls=mean(after, "policy_calls"),
            initial_states_and_episode_rng_equal=True,
            first_observations_equal=sum(reference[k]["first_observation_sha256"] == rows[k]["first_observation_sha256"] for k in reference),
            evaluator_fingerprint_equal=all(reference[k]["evaluation_fingerprint"] == rows[k]["evaluation_fingerprint"] for k in reference)))
    write_csv(output / "successful_episodes.csv", summaries)
    write_csv(output / "common_success_with_H5.csv", comparisons)
    write_csv(output / "episode_membership.csv", selected_episodes)
    write_json(output / "provenance.json", dict(
        analysis="Descriptive means conditional on success; no significance or equivalence claim.",
        action_definition="Executed policy actions, excluding the ten initial settling actions and unused predicted actions.",
        historical_renderer_source="Archived round-two execution protocol; old manifests did not contain a rendering field.",
        limitation="H5 uses historical EGL data. Current H20 uses OSMesa; initial states and RNG match but all first-observation hashes differ. Common-success matching does not remove this protocol difference or success-selection bias.",
        raw_sources=source_files, manifest_sources=manifest_files, script_sha256=file_digest(__file__), gpu_jobs_submitted=0))
    print(json.dumps(dict(success_only=summaries, common_success=comparisons), indent=2))


if __name__ == "__main__":
    main()
