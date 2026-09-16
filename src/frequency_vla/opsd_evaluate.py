"""Evaluate one frozen H=20 parameter snapshot through the official wrapper."""
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
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("The first training comparison uses at most four simulator workers")
    command = [sys.executable, "-m", "frequency_vla.evaluator", "--mode", "smoke",
               "--horizon", "20", "--required-horizon", "20", "--episodes", "10", "--seed", "7",
               "--suite", "libero_10", "--port", str(args.port), "--results-dir", args.results_dir]
    run_task_group(command, range(10), args.workers, Path(args.results_dir) / "logs")


if __name__ == "__main__":
    main()
