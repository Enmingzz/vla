"""Run explicitly selected valid horizons, gating comparisons on H=5 reproduction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import load_config, prediction_horizon, upstream_spec, validate_horizons


def run_task_group(command, task_ids, workers, log_dir):
    """Bound simulator concurrency and stop every child on the first error."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    remaining, active = iter(task_ids), {}
    exhausted = False
    try:
        while active or not exhausted:
            # Check all exits before launching another task. Ordered map() used
            # to hide later failures behind an earlier, still-running rollout.
            for task_id, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    if code:
                        raise subprocess.CalledProcessError(code, process.args)
                    del active[task_id]
                    print("Completed task={}".format(task_id), flush=True)
            while not exhausted and len(active) < workers:
                task_id = next(remaining, None)
                if task_id is None:
                    exhausted = True
                    break
                with (log_dir / "task_{}.log".format(task_id)).open("x") as log:
                    active[task_id] = subprocess.Popen(command + ["--task-ids", str(task_id)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if active:
                time.sleep(0.1)
    finally:
        # Include descendants such as video encoders; don't leave GPU clients
        # running while an exception propagates to the batch server cleanup.
        for process in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 3
        for process in active.values():
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def main():
    config = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "main"], required=True)
    p.add_argument("--suite", choices=config["supported_suites"], default=config["suite"])
    p.add_argument("--horizons", type=int, nargs="+", default=config["horizons"])
    p.add_argument("--seeds", type=int, nargs="+", default=config["seeds"])
    p.add_argument("--workers", type=int, default=1, help="Independent task processes sharing the same policy server (1..10)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[2] / "results"))
    p.add_argument("--openpi-dir", default=os.environ.get("OPENPI_DIR"), required=not os.environ.get("OPENPI_DIR"))
    args = p.parse_args()
    spec = upstream_spec(args.openpi_dir, config)
    validate_horizons(args.horizons, prediction_horizon(spec))
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Duplicate seeds")
    if not 1 <= args.workers <= 10:
        raise ValueError("workers must be between 1 and 10")
    if 5 not in args.horizons:
        raise ValueError("A sweep must include H=5 for its baseline gate; use evaluator for one H")
    for h in [5] + [h for h in args.horizons if h != 5]:
        for seed in args.seeds:
            command = [sys.executable, "-m", "frequency_vla.evaluator", "--openpi-dir", args.openpi_dir,
                "--mode", args.mode, "--suite", args.suite, "--horizon", str(h), "--required-horizon", str(max(args.horizons)),
                "--seed", str(seed), "--host", args.host, "--port", str(args.port), "--results-dir", args.results_dir]
            if args.workers == 1:
                subprocess.run(command, check=True)
            else:
                log_dir = Path(args.results_dir) / "logs" / args.mode / args.suite / ("seed_" + str(seed)) / ("H_" + str(h))
                # Only simulator processes overlap; native server inference is serial.
                # Per-episode/call RNG makes task scheduling independent of policy noise.
                run_task_group(command, range(10), args.workers, log_dir)
        subprocess.run([sys.executable, "-m", "frequency_vla.analysis", "--mode", args.mode, "--suite", args.suite,
                        "--results-dir", args.results_dir], check=True)
        if h == 5 and args.suite == "libero_10":
            gate_file = Path(args.results_dir) / "aggregated" / args.mode / args.suite / "validation.json"
            if not json.loads(gate_file.read_text())["baseline_gate_passed"]:
                if config.get("baseline_gate_policy", "stop") == "report_only":
                    print("H=5 baseline screen failed. Explicit extension protocol continues diagnostics; do not interpret this as official baseline reproduction.", flush=True)
                else:
                    raise RuntimeError("H=5 reproduction gate failed. Comparisons stopped; inspect baseline videos/environment first.")


if __name__ == "__main__":
    main()
