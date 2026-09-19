"""Small, simulator-independent checks for the temporal OPSD experiment."""
from pathlib import Path
import csv
import json


def validate_training_config(config):
    if config["prediction_horizon"] != 50 or config["student_horizon"] != 20 or config["teacher_horizon"] != 5:
        raise ValueError("This first experiment is fixed at P=50, H_student=20, H_teacher=5")
    if config["flow_steps"] != 10 or type(config["optimizer_steps"]) is not int or not 1 <= config["optimizer_steps"] <= 1500:
        raise ValueError("Keep the 10-step sampler and the bounded, at-most-1500-update experiment")
    checkpoints = config.get("checkpoint_steps", [config["optimizer_steps"]])
    if not checkpoints or checkpoints != sorted(set(checkpoints)) or any(type(s) is not int or not 1 <= s <= config["optimizer_steps"] for s in checkpoints):
        raise ValueError("Checkpoint steps must be ordered, unique and inside the update budget")
    if config["train_initial_state_start"] < config["evaluation_episodes_per_task"]:
        raise ValueError("Training and evaluation initial-state indices overlap")
    pool = set(range(config["train_initial_state_start"], config["train_initial_state_stop"]))
    if not pool or not pool <= set(range(50)):
        raise ValueError("Invalid training initial-state pool")
    for start, stop in config.get("protected_eval_ranges", [[0, config["evaluation_episodes_per_task"]]]):
        if not 0 <= start < stop <= 50 or pool & set(range(start, stop)):
            raise ValueError("Training pool overlaps a protected evaluation range")
    if config.get("train_initial_state_cursor", config["train_initial_state_start"]) not in pool:
        raise ValueError("Training cursor is outside its pool")
    if not 0 < config["ema_decay"] < 1 or config["teacher_strategy"] != "ema":
        raise ValueError("The selected OPSD protocol requires an EMA teacher")
    if config["batch_size"] != 4 or config["loss_action_dimensions"] != 7:
        raise ValueError("Unexpected first-experiment batch or LIBERO action dimension")
    if config.get("environment_workers", 1) not in (1, 4):
        raise ValueError("Use the serial simulator or one process per batch environment")


def require_measured_gap(root):
    root = Path(root) / "aggregated/smoke/libero_10"
    validation = json.loads((root / "validation.json").read_text())
    if not validation["complete_for_measured_horizons"] or not validation["paired_checks_passed"]:
        raise ValueError("The paired frequency evaluation is incomplete")
    if validation["measured_horizons"] != [5, 15, 20] or validation["prediction_horizon"] != 50:
        raise ValueError("Require the fixed-P50 H=5/15/20 comparison")
    with (root / "replanning_gaps.csv").open() as f:
        gap = next(r for r in csv.DictReader(f) if int(r["student_H"]) == 20)
    if not (float(gap["replanning_gap"]) >= 0.05 and float(gap["gap_ci95_low"]) > 0
            and float(gap["mcnemar_p_holm"]) < 0.05):
        raise ValueError("H=20 degradation is not yet convincingly established; do not train")
    return {"validation": validation, "gap": gap}


def align_teacher_latent(student_latent, teacher_auxiliary, offset, xp):
    """Teacher local position 0 corresponds to student physical offset k.

    The missing future tail is sampled, not repeated or filled with zeros.
    Only the teacher's first five velocity positions are used as targets.
    """
    if student_latent.shape != teacher_auxiliary.shape:
        raise ValueError("Student and auxiliary teacher must have equal native chunk shapes")
    if not 0 <= offset < student_latent.shape[-2]:
        raise ValueError("Invalid temporal offset")
    if offset == 0:
        return student_latent
    return xp.concatenate([student_latent[..., offset:, :], teacher_auxiliary[..., -offset:, :]], axis=-2)
