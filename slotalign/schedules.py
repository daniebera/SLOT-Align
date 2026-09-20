"""Interpolation schedules for partial Gaussian transport."""

from __future__ import annotations

import math


TAU_SCHEDULES = ("cosine_anneal", "cosine_anneal_warm_restarts")


def cosine_anneal(
    step: int,
    total_steps: int,
    tau_max: float,
    tau_min: float,
) -> float:
    """Historical cosine interpolation used by the pre-refactor experiments."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    clamped_step = max(0, min(step, total_steps))
    return tau_min + 0.5 * (tau_max - tau_min) * (
        1.0 + math.cos(math.pi * clamped_step / total_steps)
    )


def cosine_anneal_warm_restarts(
    step: int,
    tau_max: float,
    tau_min: float,
    initial_period: int = 50,
    period_multiplier: int = 1,
) -> float:
    """Historical cosine schedule with fixed-length warm-restart periods."""

    if initial_period <= 0 or period_multiplier < 1:
        raise ValueError("initial_period must be positive and period_multiplier at least one")
    period = initial_period
    period_step = step
    while period_step >= period:
        period_step -= period
        period *= period_multiplier
    return tau_min + 0.5 * (tau_max - tau_min) * (
        1.0 + math.cos(math.pi * period_step / period)
    )


def compute_tau_schedule(
    name: str,
    num_steps: int,
    tau_max: float,
    tau_min: float,
) -> list[float]:
    """Return the reversed schedule used by the original training script.

    The original implementation generated a sequence for steps
    ``0 .. num_steps - 1`` and then reversed it. For the simple cosine schedule,
    the first value is near (but generally not equal to) ``tau_min`` and the
    final value is exactly ``tau_max``.
    """

    if name not in TAU_SCHEDULES:
        raise ValueError(f"Unsupported tau schedule {name!r}; choose one of {TAU_SCHEDULES}")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not 0.0 <= tau_min <= 1.0 or not 0.0 <= tau_max <= 1.0:
        raise ValueError("tau_min and tau_max must each lie in [0, 1]")

    if name == "cosine_anneal":
        values = [
            cosine_anneal(step, num_steps, tau_max, tau_min)
            for step in range(num_steps)
        ]
    else:
        values = [
            cosine_anneal_warm_restarts(step, tau_max, tau_min)
            for step in range(num_steps)
        ]
    return list(reversed(values))
