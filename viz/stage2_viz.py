"""Stage 2 Rerun visualization entry point."""

from __future__ import annotations

import rerun as rr

from viz.blueprint import build_stage2_blueprint
from viz.primitives import (
    log_animated_trajectory,
    log_body_legs,
    log_com_velocity_arrows,
    log_foot_polygons,
    log_pendulum_rod,
    log_phase_transitions,
    log_scalar_timeseries,
    log_spatial_trajectory,
    log_torso_box,
    log_world,
)


def visualize_stage2(
    world,
    footsteps,
    schedule,
    traj,
    foot_length: float,
    foot_width: float,
    inflation_margin: float,
    com_height: float = 0.80,
    body: str = "rod",
) -> None:
    """Visualise Stage 2 output in Rerun.

    Parameters
    ----------
    body : "rod" | "model"
        "rod"   — inverted-pendulum rod from ZMP to CoM (default, fast)
        "model" — 2-link stick figure with torso box and bending legs
    """
    from robot.config import DEFAULT_ROBOT

    rr.init("bipedal-stage2", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.send_blueprint(build_stage2_blueprint())

    # Static spatial geometry
    log_world("world/occupancy", world)
    log_foot_polygons("planning/footsteps/left", "planning/footsteps/right", footsteps, foot_length, foot_width)

    # CoM and ZMP spatial overview strips
    log_spatial_trajectory("spatial/com", "spatial/zmp", traj, com_height)

    # Animated markers + velocity arrow
    log_animated_trajectory("spatial/com/marker", "spatial/zmp/marker", traj, com_height)
    log_com_velocity_arrows("spatial/com/velocity", traj, com_height)

    # Body representation
    if body == "model":
        cfg = DEFAULT_ROBOT
        log_torso_box("spatial/body/torso", traj, com_height, cfg)
        log_body_legs("spatial/body/legs", traj, footsteps, schedule, com_height, cfg)
    else:
        log_pendulum_rod("spatial/pendulum", traj, com_height)

    # Scalar time-series
    log_scalar_timeseries(traj, schedule)
    log_phase_transitions("trajectory/phase/kind", schedule)
