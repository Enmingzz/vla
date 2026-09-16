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
    parser.add_argument("--suite", default="libero_10", choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-state-start", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("The first training comparison uses at most four simulator workers")
    command = [sys.executable, "-m", "frequency_vla.evaluator", "--mode", "smoke",
               "--horizon", str(args.horizon), "--required-horizon", str(args.horizon),
               "--episodes", str(args.episodes), "--seed", str(args.seed),
               "--initial-state-start", str(args.initial_state_start),
               "--suite", args.suite, "--port", str(args.port), "--results-dir", args.results_dir]
    run_task_group(command, range(10), args.workers,
                   Path(args.results_dir) / "logs" / args.suite / ("H_" + str(args.horizon)))


if __name__ == "__main__":
    main()
