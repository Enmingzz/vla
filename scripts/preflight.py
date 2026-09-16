#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frequency_vla.config import load_config, prediction_horizon, upstream_spec, validate_horizons

p = argparse.ArgumentParser()
p.add_argument("--openpi-dir", default=os.environ.get("OPENPI_DIR"), required=not os.environ.get("OPENPI_DIR"))
p.add_argument("--horizons", type=int, nargs="+")
p.add_argument("--inspect-only", action="store_true")
a = p.parse_args()
c = load_config()
s = upstream_spec(a.openpi_dir, c)
print(json.dumps(s, indent=2))
if not a.inspect_only:
    validate_horizons(a.horizons or c["horizons"], prediction_horizon(s))
    print("Preflight passed. Checkpoint unchanged; inference P={} (official P={}).".format(prediction_horizon(s), s["native_prediction_horizon"]))
