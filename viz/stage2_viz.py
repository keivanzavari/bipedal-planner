"""Stage 2 Rerun visualization entry point."""

from __future__ import annotations

import rerun as rr

from viz.blueprint import build_stage2_blueprint
from viz.primitives import (
    log_animated_trajectory,
    log_com_velocity_arrows,
    log_foot_polygons,
    log_pendulum_rod,
    log_phase_transitions,
    log_scalar_timeseries,
    log_spatial_trajectory,
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
) -> None:
    rr.init("bipedal-stage2", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.send_blueprint(build_stage2_blueprint())

    # Static spatial geometry
    log_world("world/occupancy", world)
    log_foot_polygons("planning/footsteps/left", "planning/footsteps/right", footsteps, foot_length, foot_width)

    # CoM and ZMP spatial overview strips
    log_spatial_trajectory("spatial/com", "spatial/zmp", traj, com_height)

    # Animated markers + pendulum rod + velocity arrow (scrub timeline to animate)
    log_animated_trajectory("spatial/com/marker", "spatial/zmp/marker", traj, com_height)
    log_pendulum_rod("spatial/pendulum", traj, com_height)
    log_com_velocity_arrows("spatial/com/velocity", traj, com_height)

    # Scalar time-series
    log_scalar_timeseries(traj, schedule)
    log_phase_transitions("trajectory/phase/kind", schedule)
