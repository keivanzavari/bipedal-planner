"""Stage 4 Rerun visualization entry point.

Reuses existing Stage 3 visualization primitives and adds Stage 4–specific
channels: joint angles/torques, ZMP comparison (LIPM vs GRF), CoM height
drift, and energy monitoring.
"""

from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb

from robot.config import DEFAULT_ROBOT
from stage1.footstep import Footstep
from stage2.contact_schedule import ContactSchedule
from stage2.preview_controller import CoMTrajectory
from stage4.mujoco_result import MujocoResult
from viz.primitives import (
    log_active_support_polygon,
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
    log_zmp_vs_bounds,
)


def _stride(T: int, target: int) -> int:
    return max(1, T // target)


# --- Stage 4–specific log helpers ---

_JOINT_NAMES = ["hip_L", "knee_L", "ankle_L", "hip_R", "knee_R", "ankle_R"]
_JOINT_COLORS = [
    [230, 126, 34],  # hip_L — orange
    [46, 204, 113],  # knee_L — green
    [155, 89, 182],  # ankle_L — purple
    [52, 152, 219],  # hip_R — blue
    [26, 188, 156],  # knee_R — teal
    [231, 76, 60],  # ankle_R — red
]


def _log_joint_angles(result: MujocoResult) -> None:
    """Log joint angles as time-indexed scalar channels."""
    for name, color in zip(_JOINT_NAMES, _JOINT_COLORS, strict=True):
        rr.log(f"stage4/joints/{name}/angle", rr.SeriesLine(color=color, name=f"{name} angle"), static=True)

    T = len(result.t)
    s = _stride(T, 5000)
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(result.t[i]))
        for j, name in enumerate(_JOINT_NAMES):
            rr.log(f"stage4/joints/{name}/angle", rr.Scalar(float(result.q[i, j])))


def _log_joint_torques(result: MujocoResult) -> None:
    """Log joint torques as time-indexed scalar channels."""
    for name, color in zip(_JOINT_NAMES, _JOINT_COLORS, strict=True):
        rr.log(f"stage4/joints/{name}/torque", rr.SeriesLine(color=color, name=f"{name} torque"), static=True)

    T = len(result.t)
    s = _stride(T, 5000)
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(result.t[i]))
        for j, name in enumerate(_JOINT_NAMES):
            rr.log(f"stage4/joints/{name}/torque", rr.Scalar(float(result.tau[i, j])))


def _log_zmp_comparison(result: MujocoResult) -> None:
    """Log ZMP from LIPM formula vs GRF side-by-side (sagittal x-axis)."""
    rr.log("stage4/zmp_comparison/from_grf_x", rr.SeriesLine(color=[231, 76, 60], name="ZMP GRF x"), static=True)
    rr.log("stage4/zmp_comparison/from_lipm_x", rr.SeriesLine(color=[52, 152, 219], name="ZMP LIPM x"), static=True)
    rr.log("stage4/zmp_comparison/from_grf_y", rr.SeriesLine(color=[192, 57, 43], name="ZMP GRF y"), static=True)
    rr.log("stage4/zmp_comparison/from_lipm_y", rr.SeriesLine(color=[41, 128, 185], name="ZMP LIPM y"), static=True)

    T = len(result.t)
    s = _stride(T, 5000)
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(result.t[i]))
        rr.log("stage4/zmp_comparison/from_grf_x", rr.Scalar(float(result.zmp_from_grf[i, 0])))
        rr.log("stage4/zmp_comparison/from_lipm_x", rr.Scalar(float(result.zmp_from_lipm[i, 0])))
        rr.log("stage4/zmp_comparison/from_grf_y", rr.Scalar(float(result.zmp_from_grf[i, 1])))
        rr.log("stage4/zmp_comparison/from_lipm_y", rr.Scalar(float(result.zmp_from_lipm[i, 1])))


def _log_com_height(result: MujocoResult) -> None:
    """Log actual CoM height from MuJoCo for drift monitoring."""
    rr.log("stage4/com_height", rr.SeriesLine(color=[46, 204, 113], name="CoM z"), static=True)

    T = len(result.t)
    s = _stride(T, 5000)
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(result.t[i]))
        rr.log("stage4/com_height", rr.Scalar(float(result.com_height_actual[i])))


def _log_energy(result: MujocoResult) -> None:
    """Log total energy (kinetic + potential) from MuJoCo."""
    rr.log("stage4/energy", rr.SeriesLine(color=[243, 156, 18], name="Energy"), static=True)

    T = len(result.t)
    s = _stride(T, 5000)
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(result.t[i]))
        rr.log("stage4/energy", rr.Scalar(float(result.energy[i])))


# --- Blueprint ---


def _build_stage4_blueprint() -> rrb.Blueprint:
    """4-panel layout: 3D spatial + 4 stacked time-series views."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/",
                contents=["world/**", "planning/**", "spatial/**", "tracking/**"],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/",
                    contents=["tracking/error/**", "stage4/com_height"],
                    name="Tracking Error & CoM Height",
                ),
                rrb.TimeSeriesView(
                    origin="/",
                    contents=["stage4/zmp_comparison/**"],
                    name="ZMP: GRF vs LIPM",
                ),
                rrb.TimeSeriesView(
                    origin="/",
                    contents=["stage4/joints/*/angle"],
                    name="Joint Angles",
                ),
                rrb.TimeSeriesView(
                    origin="/",
                    contents=["stage4/joints/*/torque", "stage4/energy"],
                    name="Torques & Energy",
                ),
            ),
            column_shares=[1, 1],
        ),
    )


# --- Entry point ---


def visualize_stage4(
    world,
    footsteps: list[Footstep],
    schedule: ContactSchedule,
    traj: CoMTrajectory,
    result: MujocoResult,
    foot_length: float,
    foot_width: float,
    inflation_margin: float,
    com_height: float = 0.80,
) -> None:
    """Visualise Stage 4 MuJoCo simulation output in Rerun."""
    cfg = DEFAULT_ROBOT

    rr.init("bipedal-stage4", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.send_blueprint(_build_stage4_blueprint())

    # Static geometry (reused from Stage 3)
    log_world("world/occupancy", world)
    log_foot_polygons(
        "planning/footsteps/left",
        "planning/footsteps/right",
        footsteps,
        foot_length,
        foot_width,
    )

    # Stage 2 reference trajectory
    log_spatial_trajectory("spatial/com", "spatial/zmp", traj, com_height)
    log_scalar_timeseries(traj, schedule)
    log_phase_transitions("trajectory/phase/kind", schedule)

    # Body animation
    log_torso_box("spatial/body/torso", traj, com_height, cfg)
    log_body_legs("spatial/body/legs", traj, footsteps, schedule, com_height, cfg)

    # Stage 3–compatible overlays
    log_tracking_overlay("tracking/com", result, com_height)
    log_grf_arrows("tracking/grf", result, footsteps, schedule, cfg)
    log_active_support_polygon("tracking/support_polygon", result, schedule, footsteps, foot_length, foot_width)
    log_tracking_error_timeseries(result)
    log_zmp_vs_bounds(result)

    # Stage 4–specific channels
    _log_joint_angles(result)
    _log_joint_torques(result)
    _log_zmp_comparison(result)
    _log_com_height(result)
    _log_energy(result)
