"""The scheduler cap covers startup, training and checkpoint export together."""
import pytest

from frequency_vla.execution_budget import training_deadline


def test_deadline_is_relative_to_allocation_start_and_reserves_checkpoint_time():
    request = dict(mode="train_only", wall_time_limit_seconds=3600, checkpoint_reserve_seconds=300)
    assert training_deadline(request, dict(SLURM_JOB_START_TIME="10000", SLURM_JOB_END_TIME="13600")) == 13300


def test_unamended_two_and_half_hour_job_is_rejected():
    request = dict(mode="train_only", wall_time_limit_seconds=3600, checkpoint_reserve_seconds=300)
    with pytest.raises(ValueError, match="Actual Slurm allocation"):
        training_deadline(request, dict(SLURM_JOB_START_TIME="10000", SLURM_JOB_END_TIME="19000"))
