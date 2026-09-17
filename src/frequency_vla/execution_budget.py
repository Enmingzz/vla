"""Use the actual Slurm allocation deadline, including model startup time."""


def training_deadline(execution, environ):
    if execution["mode"] != "train_only":
        raise ValueError("Expected an explicit training-only execution request")
    limit = execution["wall_time_limit_seconds"]
    reserve = execution["checkpoint_reserve_seconds"]
    if not 0 < reserve < limit <= 3600:
        raise ValueError("The requested allocation is capped at one hour with a save reserve")
    start, end = int(environ["SLURM_JOB_START_TIME"]), int(environ["SLURM_JOB_END_TIME"])
    if not start < end <= start + limit:
        raise ValueError("Actual Slurm allocation exceeds the requested one-hour limit")
    return end - reserve
