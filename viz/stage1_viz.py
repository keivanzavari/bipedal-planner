"""Stage 1 Rerun visualization entry point."""

from __future__ import annotations

import rerun as rr

from viz.blueprint import build_stage1_blueprint
from viz.primitives import (
    log_com_stability_points,
    log_foot_polygons,
    log_support_polygons,
    log_waypoints,
    log_world,
)


def visualize_stage1(
    world,
    start: tuple,
    goal: tuple,
    path: list,
    footsteps,
    phases,
    foot_length: float,
    foot_width: float,
    inflation_margin: float,
    planner_name: str,
) -> None:
    rr.init("bipedal-stage1", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.send_blueprint(build_stage1_blueprint())

    log_world("world/occupancy", world)
    rr.log(
        "world/start",
        rr.Points3D([[start[0], start[1], 0.0]], colors=[[46, 204, 113, 255]], radii=0.08),
        static=True,
    )
    rr.log(
        "world/goal",
        rr.Points3D([[goal[0], goal[1], 0.0]], colors=[[231, 76, 60, 255]], radii=0.08),
        static=True,
    )

    log_waypoints("planning/global_path", path)
    log_foot_polygons("planning/footsteps/left", "planning/footsteps/right", footsteps, foot_length, foot_width)
    log_support_polygons("stability/support_polygons/stable", "stability/support_polygons/unstable", phases)
    log_com_stability_points("stability/com_points/stable", "stability/com_points/unstable", phases)
