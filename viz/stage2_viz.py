"""Stage 2 Rerun visualization entry point."""

from __future__ import annotations

import rerun as rr

from viz.blueprint import build_stage2_blueprint
from viz.primitives import (
    log_animated_trajectory,
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
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.send_blueprint(build_stage2_blueprint())

    # Static spatial geometry
    log_world("world/occupancy", world)
    log_foot_polygons("planning/footsteps/left", "planning/footsteps/right", footsteps, foot_length, foot_width)

    # CoM and ZMP spatial overview + animated scrub markers (under spatial/**)
    log_spatial_trajectory("spatial/com", "spatial/zmp", traj)
    log_animated_trajectory("spatial/com/marker", "spatial/zmp/marker", traj)

    # Scalar time-series (trajectory/**)
    log_scalar_timeseries(traj, schedule)
    log_phase_transitions("trajectory/phase/kind", schedule)
