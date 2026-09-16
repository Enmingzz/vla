"""A 90-task evaluation must neither truncate to ten tasks nor accept ID 90."""
import sys
from types import SimpleNamespace

import pytest

from frequency_vla import opsd_evaluate
from frequency_vla.libero90_analysis import task_and_layout_interval
from frequency_vla.analysis import paired_interval


@pytest.mark.parametrize("requested, expected", [([], list(range(90))), (["--task-ids", "89"], [89])])
def test_evaluator_uses_benchmark_task_count(monkeypatch, tmp_path, requested, expected):
    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_90": lambda: SimpleNamespace(n_tasks=90)})
    monkeypatch.setitem(sys.modules, "libero.libero", SimpleNamespace(benchmark=benchmark))
    monkeypatch.setenv("OPENPI_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--port", "8000", "--results-dir", str(tmp_path), "--suite", "libero_90"] + requested)
    calls = []
    monkeypatch.setattr(opsd_evaluate, "run_task_group", lambda command, ids, workers, logs: calls.append(list(ids)))
    opsd_evaluate.main()
    assert calls == [expected]


def test_out_of_range_task_fails_before_starting_workers(monkeypatch, tmp_path):
    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_90": lambda: SimpleNamespace(n_tasks=90)})
    monkeypatch.setitem(sys.modules, "libero.libero", SimpleNamespace(benchmark=benchmark))
    monkeypatch.setenv("OPENPI_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--port", "8000", "--results-dir", str(tmp_path), "--suite", "libero_90", "--task-ids", "90"])
    with pytest.raises(SystemExit) as error:
        opsd_evaluate.main()
    assert error.value.code == 2


def test_task_resampling_detects_uncertainty_hidden_by_fixed_tasks():
    left = [dict(seed=37, task_id=t, episode_index=e, success=t == 0) for t in range(2) for e in range(3)]
    right = [dict(r, success=not r["success"]) for r in left]
    pairs = list(zip(left, right))
    assert paired_interval(pairs, 1000, 7) == (0., 0.)
    assert task_and_layout_interval(left, right, 1000, 7) == (-1., 1.)
    with pytest.raises(ValueError, match="equal paired coverage"):
        task_and_layout_interval(left[:-1], right, 1000, 7)
