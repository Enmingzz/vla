"""Small, simulator-independent checks for the temporal OPSD experiment."""
from pathlib import Path
import csv
import json


def validate_training_config(config):
    if config["prediction_horizon"] != 50 or config["student_horizon"] != 20 or config["teacher_horizon"] != 5:
        raise ValueError("This first experiment is fixed at P=50, H_student=20, H_teacher=5")
    if config["flow_steps"] != 10 or config["optimizer_steps"] != 100:
        raise ValueError("Keep the 10-step sampler and the bounded 100-update experiment")
    if config["train_initial_state_start"] < config["evaluation_episodes_per_task"]:
        raise ValueError("Training and evaluation initial-state indices overlap")
    if not 0 < config["ema_decay"] < 1 or config["teacher_strategy"] != "ema":
        raise ValueError("The selected OPSD protocol requires an EMA teacher")
    if config["batch_size"] != 4 or config["loss_action_dimensions"] != 7:
        raise ValueError("Unexpected first-experiment batch or LIBERO action dimension")


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
