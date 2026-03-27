"""Stage 3 Rerun visualization entry point."""

from __future__ import annotations

import rerun as rr

from stage1.footstep import Footstep
from stage1.world import World
from stage2.contact_schedule import ContactSchedule
from stage2.preview_controller import CoMTrajectory
from stage3.simulator import TrackingResult
from viz.blueprint import build_stage3_blueprint
from viz.primitives import (
    log_body_legs,
    log_foot_polygons,
    log_grf_arrows,
    log_phase_transitions,
    log_scalar_timeseries,
    log_spatial_trajectory,
    log_torso_box,
    log_tracking_error_timeseries,
    log_tracking_overlay,
    log_world,
)


def visualize_stage3(
    world: World,
    footsteps: list[Footstep],
    schedule: ContactSchedule,
    traj: CoMTrajectory,
    result: TrackingResult,
    foot_length: float,
    foot_width: float,
    inflation_margin: float,
    com_height: float = 0.80,
) -> None:
    """Visualise Stage 3 closed-loop tracking output in Rerun.

    Shows the Stage 2 reference trajectory alongside the actual tracked
    trajectory, with tracking error time-series and GRF arrows (non-zero
    only for WBC).
    """
    from robot.config import DEFAULT_ROBOT

    cfg = DEFAULT_ROBOT

    rr.init("bipedal-stage3", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.send_blueprint(build_stage3_blueprint())

    # Static geometry
    log_world("world/occupancy", world)
    log_foot_polygons("planning/footsteps/left", "planning/footsteps/right", footsteps, foot_length, foot_width)

    # Stage 2 reference trajectory
    log_spatial_trajectory("spatial/com", "spatial/zmp", traj, com_height)
    log_scalar_timeseries(traj, schedule)
    log_phase_transitions("trajectory/phase/kind", schedule)

    # Body animation (2-link model)
    log_torso_box("spatial/body/torso", traj, com_height, cfg)
    log_body_legs("spatial/body/legs", traj, footsteps, schedule, com_height, cfg)

    # Stage 3 overlays
    log_tracking_overlay("tracking/com", result, com_height)
    log_grf_arrows("tracking/grf", result, footsteps, schedule, cfg)
    log_tracking_error_timeseries(result)
