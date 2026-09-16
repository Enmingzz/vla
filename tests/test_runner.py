"""A failing task must not strand the GPU allocation behind other workers."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from frequency_vla.runner import run_task_group


def test_failure_stops_running_child_and_does_not_start_queued_task(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text('''import os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
task = int(sys.argv[-1])
if task == 0:
    (root / "running.pid").write_text(str(os.getpid()))
    time.sleep(60)
elif task == 1:
    while not (root / "running.pid").exists():
        time.sleep(0.01)
    raise SystemExit(7)
else:
    (root / "should_not_start").touch()
''')
    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_task_group([sys.executable, str(script), str(tmp_path)], range(3), 2, tmp_path / "logs")
    assert error.value.returncode == 7
    assert time.monotonic() - started < 10
    assert not (tmp_path / "should_not_start").exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "running.pid").read_text()), 0)


def test_success_collects_every_requested_task(tmp_path):
    run_task_group([sys.executable, "-c", "import sys; print(sys.argv[-1])"], [0, 1, 2], 2, tmp_path)
    assert [(tmp_path / "task_{}.log".format(i)).read_text().strip() for i in range(3)] == ["0", "1", "2"]
