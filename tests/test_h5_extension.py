"""Avoid false pairings and training-layout contamination when joining blocks."""
from frequency_vla.analysis import pair_key
import pytest

from frequency_vla.h5_extension import evaluation_core, normalize_rows, subset_rows
from frequency_vla.logging_utils import digest


def test_block_local_episode_numbers_do_not_merge_different_layouts():
    raw = [dict(seed=27, task_id=0, episode_index=0, initial_state_index=i) for i in [0, 20, 30, 40]]
    joined = normalize_rows(raw, cached=False)
    assert len({pair_key(r) for r in joined}) == 4
    assert [r["episode_index"] for r in raw] == [0, 0, 0, 0]
    assert all(r["source_episode_index"] == 0 for r in joined)


def test_primary_partition_excludes_every_training_layout_and_retains_cached_rows():
    raw = [dict(seed=27, task_id=t, episode_index=i % 10, initial_state_index=i)
           for t in range(10) for i in range(50)]
    rows = normalize_rows(raw, cached=False)
    seen = {(t, i) for t in range(10) for i in range(10, 20)}
    parts = subset_rows(rows, seen)
    assert {k:len(v) for k,v in parts.items()} == {"heldout_400":400,"all_500":500,"training_layouts_100":100}
    assert not {(r["task_id"],r["initial_state_index"]) for r in parts["heldout_400"]} & seen
    assert sum(30 <= r["initial_state_index"] < 40 for r in parts["heldout_400"]) == 100


def test_only_the_declared_layout_offset_may_differ_between_evaluator_blocks():
    def server(start, backend="egl"):
        spec=dict(initial_state_start=start,rendering=dict(backend=backend),packages=dict(numpy="pinned"))
        return dict(evaluation_spec=spec,evaluation_fingerprint=digest(spec))
    assert evaluation_core(server(0),0)==evaluation_core(server(30),30)
    assert evaluation_core(server(0),0)!=evaluation_core(server(30,"osmesa"),30)
    with pytest.raises(ValueError,match="block offset"):
        evaluation_core(server(30),0)
    changed=server(30)
    changed["evaluation_spec"]["packages"]["numpy"]="changed"
    with pytest.raises(ValueError,match="fingerprint"):
        evaluation_core(changed,30)
