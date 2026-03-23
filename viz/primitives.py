"""Low-level Rerun logging helpers.

All functions operate on world-coordinate data and log timeless by default,
except log_scalar_timeseries and log_phase_transitions which are time-indexed.
"""

from __future__ import annotations

import math

import numpy as np
import rerun as rr


def _stride(T: int, target: int) -> int:
    """Return a step size that keeps at most `target` points from T total."""
    return max(1, T // target)


def log_world(entity_path: str, world) -> None:
    """Log obstacles as LineStrips2D batches and the world boundary."""
    if world.obstacles:
        strips = []
        for obs in world.obstacles:
            x, y, w, h = obs.x, obs.y, obs.w, obs.h
            strip = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]],
                dtype=np.float32,
            )
            strips.append(strip)
        rr.log(
            f"{entity_path}/obstacles",
            rr.LineStrips2D(strips, colors=[[80, 80, 80, 220]] * len(strips)),
            timeless=True,
        )

    W, H = world.width, world.height
    boundary = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]], dtype=np.float32)
    rr.log(
        f"{entity_path}/boundary",
        rr.LineStrips2D([boundary], colors=[[0, 0, 0, 255]]),
        timeless=True,
    )


def log_waypoints(entity_path: str, path: list) -> None:
    if not path:
        return
    points = np.array(path, dtype=np.float32)
    rr.log(
        entity_path,
        rr.LineStrips2D([points], colors=[[52, 152, 219, 255]]),
        timeless=True,
    )


def _foot_corners(x: float, y: float, theta: float, foot_length: float, foot_width: float) -> np.ndarray:
    """Return the 4 corners of a foot rectangle in world coordinates."""
    hl, hw = foot_length / 2, foot_width / 2
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (local @ R.T) + np.array([x, y])


def log_foot_polygons(
    path_left: str,
    path_right: str,
    footsteps,
    foot_length: float,
    foot_width: float,
) -> None:
    """Log all left/right foot rectangles as closed LineStrips2D, batched per side."""
    left_strips: list[np.ndarray] = []
    right_strips: list[np.ndarray] = []

    for fs in footsteps:
        corners = _foot_corners(fs.x, fs.y, fs.theta, foot_length, foot_width)
        closed = np.concatenate([corners, corners[:1]], axis=0).astype(np.float32)
        if fs.side == "L":
            left_strips.append(closed)
        else:
            right_strips.append(closed)

    if left_strips:
        rr.log(
            path_left,
            rr.LineStrips2D(left_strips, colors=[[52, 152, 219, 200]] * len(left_strips)),
            timeless=True,
        )
    if right_strips:
        rr.log(
            path_right,
            rr.LineStrips2D(right_strips, colors=[[231, 76, 60, 200]] * len(right_strips)),
            timeless=True,
        )


def log_support_polygons(path_stable: str, path_unstable: str, phases) -> None:
    """Log support polygon convex hulls, split by stability."""
    stable_strips: list[np.ndarray] = []
    unstable_strips: list[np.ndarray] = []

    for phase in phases:
        poly = phase.support_polygon
        if len(poly) < 3:
            continue
        closed = np.concatenate([poly, poly[:1]], axis=0).astype(np.float32)
        if phase.stable:
            stable_strips.append(closed)
        else:
            unstable_strips.append(closed)

    if stable_strips:
        rr.log(
            path_stable,
            rr.LineStrips2D(stable_strips, colors=[[46, 204, 113, 180]] * len(stable_strips)),
            timeless=True,
        )
    if unstable_strips:
        rr.log(
            path_unstable,
            rr.LineStrips2D(unstable_strips, colors=[[231, 76, 60, 180]] * len(unstable_strips)),
            timeless=True,
        )


def log_com_stability_points(path_stable: str, path_unstable: str, phases) -> None:
    """Log CoM positions at each stance phase, coloured by stability."""
    stable_pts: list[np.ndarray] = []
    unstable_pts: list[np.ndarray] = []

    for phase in phases:
        com = phase.com[:2]
        if phase.stable:
            stable_pts.append(com)
        else:
            unstable_pts.append(com)

    if stable_pts:
        rr.log(
            path_stable,
            rr.Points2D(np.array(stable_pts, dtype=np.float32), colors=[[46, 204, 113, 255]], radii=0.03),
            timeless=True,
        )
    if unstable_pts:
        rr.log(
            path_unstable,
            rr.Points2D(np.array(unstable_pts, dtype=np.float32), colors=[[231, 76, 60, 255]], radii=0.03),
            timeless=True,
        )


def log_spatial_trajectory(path_com: str, path_zmp: str, traj) -> None:
    """Log CoM and ZMP paths as downsampled spatial overview strips (timeless)."""
    T = len(traj.t)
    s = _stride(T, 2000)
    com_pts = np.column_stack([traj.x[::s], traj.y[::s]]).astype(np.float32)
    zmp_pts = np.column_stack([traj.zmp_x[::s], traj.zmp_y[::s]]).astype(np.float32)

    rr.log(path_com, rr.LineStrips2D([com_pts], colors=[[230, 126, 34, 255]]), timeless=True)
    rr.log(path_zmp, rr.LineStrips2D([zmp_pts], colors=[[155, 89, 182, 255]]), timeless=True)


def log_scalar_timeseries(traj, schedule) -> None:
    """Log all scalar channels time-indexed via rr.set_time_seconds."""
    T = len(traj.t)
    s = _stride(T, 5000)

    for i in range(0, T, s):
        rr.set_time_seconds("t", float(traj.t[i]))
        rr.log("trajectory/com/position/x", rr.Scalar(float(traj.x[i])))
        rr.log("trajectory/com/position/y", rr.Scalar(float(traj.y[i])))
        rr.log("trajectory/com/velocity/x", rr.Scalar(float(traj.vx[i])))
        rr.log("trajectory/com/velocity/y", rr.Scalar(float(traj.vy[i])))
        rr.log("trajectory/com/acceleration/x", rr.Scalar(float(traj.ax[i])))
        rr.log("trajectory/com/acceleration/y", rr.Scalar(float(traj.ay[i])))
        rr.log("trajectory/zmp/x", rr.Scalar(float(traj.zmp_x[i])))
        rr.log("trajectory/zmp/y", rr.Scalar(float(traj.zmp_y[i])))
        rr.log("trajectory/zmp_ref/x", rr.Scalar(float(schedule.zmp_x[i])))
        rr.log("trajectory/zmp_ref/y", rr.Scalar(float(schedule.zmp_y[i])))


def log_phase_transitions(entity_path: str, schedule) -> None:
    """Log phase kind (single/double) as TextLog only at transitions."""
    prev_kind = None
    for i, kind in enumerate(schedule.kind):
        if kind != prev_kind:
            rr.set_time_seconds("t", float(schedule.t[i]))
            rr.log(entity_path, rr.TextLog(kind))
            prev_kind = kind
