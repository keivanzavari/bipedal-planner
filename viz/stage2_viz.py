"""Stage 2 Rerun visualization entry point."""

from __future__ import annotations

import rerun as rr

from viz.blueprint import build_stage2_blueprint
from viz.primitives import (
    log_foot_polygons,
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
) -> None:
    rr.init("bipedal-stage2", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, timeless=True)
    rr.send_blueprint(build_stage2_blueprint())

    log_world("world/occupancy", world)
    log_foot_polygons("planning/footsteps/left", "planning/footsteps/right", footsteps, foot_length, foot_width)
    log_spatial_trajectory("trajectory/com/spatial", "trajectory/zmp/spatial", traj)
    log_scalar_timeseries(traj, schedule)
    log_phase_transitions("trajectory/phase/kind", schedule)
