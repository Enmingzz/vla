from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
