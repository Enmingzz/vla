import math
import numpy as np
import pytest

from frequency_vla.evaluator import validate_call_schedule
from frequency_vla.simpler_protocol import ChunkPrefix, bridge_action, noise_at_step


def test_executed_prefix_discards_tail_and_counts_actual_calls():
    for horizon in [1, 2, 5]:
        controller = ChunkPrefix(horizon, 5)
        count = [0]
        def query():
            count[0] += 1
            return np.repeat((np.arange(5) + count[0]*100)[:, None], 7, axis=1)
        executed = [controller.next(step, query)[0] for step in range(13)]
        expected = [100*(step//horizon+1) + step % horizon for step in range(13)]
        assert executed == expected
        assert count[0] == math.ceil(13/horizon)
        validate_call_schedule(13, count[0], horizon, controller.call_steps)


def test_unsupported_horizon_and_bad_chunks_fail():
    with pytest.raises(ValueError):
        ChunkPrefix(20, 5)
    controller = ChunkPrefix(5, 5)
    for chunk in [np.zeros((4, 7)), np.full((5, 7), np.nan), np.zeros((5, 8))]:
        with pytest.raises(ValueError):
            controller.next(0, lambda: chunk)


def test_bridge_rotation_and_gripper_semantics():
    np.testing.assert_allclose(bridge_action([.01, 0, 0, 0, 0, np.pi/2, .9]),
                               [.01, 0, 0, 0, 0, np.pi/2, 1])
    np.testing.assert_allclose(bridge_action([0, 0, 0, 0, 0, 0, .1]), [0, 0, 0, 0, 0, 0, -1])


def test_noise_key_does_not_depend_on_number_of_previous_queries():
    every_step = {s: noise_at_step(7, 0, 3, s) for s in range(11)}
    every_five = {s: noise_at_step(7, 0, 3, s) for s in range(0, 11, 5)}
    for step in every_five:
        np.testing.assert_array_equal(every_step[step], every_five[step])
    assert not np.array_equal(every_step[0], every_step[1])
