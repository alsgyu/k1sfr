from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Walk20kCurriculumStage:
    """Fixed 20k curriculum stage for command timing, disturbance, and randomization."""

    iteration_start: int
    iteration_end: int
    resampling_time_range: tuple[float, float]
    disturbance_level: float
    randomization_level: float
    nominal_ratio: float
    push_interval_s: float
    push_duration_s: float


WALK_20K_CURRICULUM_STAGES: tuple[Walk20kCurriculumStage, ...] = (
    # [20%] 0~2k: randomization 20%, disturbance 0%, nominal 100%.
    Walk20kCurriculumStage(0, 2_000, (3.0, 8.0), 0.00, 0.20, 1.00, 9.0, 0.0),
    # [35%] 2k~5k: randomization 35%, disturbance 20%, nominal 80%.
    Walk20kCurriculumStage(2_000, 5_000, (2.0, 6.0), 0.20, 0.35, 0.80, 9.0, 0.3),
    # [55%] 5k~9k: randomization 55%, disturbance 45%, nominal 60%.
    Walk20kCurriculumStage(5_000, 9_000, (1.0, 4.0), 0.45, 0.55, 0.60, 7.0, 0.5),
    # [75%] 9k~14k: randomization 75%, disturbance 70%, nominal 40%.
    Walk20kCurriculumStage(9_000, 14_000, (0.5, 3.0), 0.70, 0.75, 0.40, 6.0, 0.7),
    # [90%] 14k~18k: randomization 90%, disturbance 90%, nominal 25%.
    Walk20kCurriculumStage(14_000, 18_000, (0.3, 2.0), 0.90, 0.90, 0.25, 5.5, 0.9),
    # [100%] 18k~20k+: final scratch_symmetry disturbance/randomization, nominal 15~20%.
    Walk20kCurriculumStage(18_000, 20_000, (0.2, 2.0), 1.00, 1.00, 0.20, 5.0, 1.0),
)


def get_training_iteration(env, num_steps_per_iteration: int = 24) -> int:
    """Return PPO iteration estimated from common_step_counter."""

    return int(env.common_step_counter) // max(num_steps_per_iteration, 1)


def get_walk_20k_curriculum_stage(
    env,
    num_steps_per_iteration: int = 24,
) -> Walk20kCurriculumStage:
    """Return the fixed 20k stage for the current training iteration."""

    iteration = get_training_iteration(env, num_steps_per_iteration)
    for stage in WALK_20K_CURRICULUM_STAGES:
        if stage.iteration_start <= iteration < stage.iteration_end:
            return stage
    return WALK_20K_CURRICULUM_STAGES[-1]


def scale_additive_range(
    value_range: tuple[float, float],
    level: float,
) -> tuple[float, float]:
    """Scale additive uniform range around zero."""

    return (value_range[0] * level, value_range[1] * level)


def scale_gaussian_std_range(
    value_range: tuple[float, float],
    level: float,
) -> tuple[float, float]:
    """Scale gaussian (mean, std) range by changing only std."""

    return (value_range[0], value_range[1] * level)


def scale_multiplicative_range(
    value_range: tuple[float, float],
    level: float,
) -> tuple[float, float]:
    """Scale multiplicative uniform range around 1.0."""

    return (
        1.0 + (value_range[0] - 1.0) * level,
        1.0 + (value_range[1] - 1.0) * level,
    )
