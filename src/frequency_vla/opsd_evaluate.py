"""Evaluate one frozen parameter snapshot through the official wrapper."""
import argparse
import os
from pathlib import Path
import sys

from .runner import run_task_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--suite", default="libero_10", choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-state-start", type=int, default=0)
    parser.add_argument("--task-ids", type=int, nargs="+")
    args = parser.parse_args()
    maximum_workers = 8 if os.environ.get("MUJOCO_GL") == "osmesa" else 4
    if not 1 <= args.workers <= maximum_workers:
        raise ValueError("Use at most {} simulator workers for this renderer".format(maximum_workers))
    command = [sys.executable, "-m", "frequency_vla.evaluator", "--mode", "smoke",
               "--horizon", str(args.horizon), "--required-horizon", str(args.horizon),
               "--episodes", str(args.episodes), "--seed", str(args.seed),
               "--initial-state-start", str(args.initial_state_start),
               "--suite", args.suite, "--port", str(args.port), "--results-dir", args.results_dir]
    sys.path.insert(0, str(Path(os.environ["OPENPI_DIR"]) / "third_party/libero"))
    from libero.libero import benchmark
    count = benchmark.get_benchmark_dict()[args.suite]().n_tasks
    ids = args.task_ids if args.task_ids is not None else list(range(count))
    if len(ids) != len(set(ids)) or any(t < 0 or t >= count for t in ids):
        parser.error("Invalid or duplicate task IDs for " + args.suite)
    run_task_group(command, ids, args.workers,
                   Path(args.results_dir) / "logs" / args.suite / ("H_" + str(args.horizon)))


if __name__ == "__main__":
    main()
