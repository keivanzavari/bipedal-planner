"""Low-level Rerun logging helpers — all spatial geometry is logged in 3D.

Ground-plane entities (world, feet, support polygons, waypoints) live at z=0.
CoM entities live at z=com_height (the LIPM pendulum height).

Static geometry (world, footsteps, trajectory overview) is logged with static=True.
Time-indexed data (animated markers, scalars) uses rr.set_time_seconds("t", ...).
"""

from __future__ import annotations

import math

import numpy as np
import rerun as rr


def _stride(T: int, target: int) -> int:
    """Return a step size that keeps at most `target` points from T total."""
    return max(1, T // target)


def log_world(entity_path: str, world, obstacle_height: float = 1.0) -> None:
    """Log obstacles as solid Boxes3D and the world boundary at z=0.

    Obstacles are extruded to `obstacle_height` (default 1 m — taller than the
    CoM at 0.8 m) so they read as walls in the 3D view.
    """
    if world.obstacles:
        centers = []
        half_sizes = []
        for obs in world.obstacles:
            x, y, w, h = obs.x, obs.y, obs.w, obs.h
            centers.append([x + w / 2, y + h / 2, obstacle_height / 2])
            half_sizes.append([w / 2, h / 2, obstacle_height / 2])
        rr.log(
            f"{entity_path}/obstacles",
            rr.Boxes3D(
                centers=np.array(centers, dtype=np.float32),
                half_sizes=np.array(half_sizes, dtype=np.float32),
                colors=[[80, 80, 80, 160]],
                fill_mode=rr.components.FillMode.Solid,
            ),
            static=True,
        )

    W, H = world.width, world.height
    boundary = np.array(
        [[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0], [0, 0, 0]], dtype=np.float32
    )
    rr.log(
        f"{entity_path}/boundary",
        rr.LineStrips3D([boundary], colors=[[0, 0, 0, 255]]),
        static=True,
    )


def log_waypoints(entity_path: str, path: list) -> None:
    if not path:
        return
    pts = np.array(path, dtype=np.float32)
    points_3d = np.column_stack([pts, np.zeros(len(pts), dtype=np.float32)])
    rr.log(
        entity_path,
        rr.LineStrips3D([points_3d], colors=[[52, 152, 219, 255]]),
        static=True,
    )


def _foot_corners(
    x: float, y: float, theta: float, foot_length: float, foot_width: float
) -> np.ndarray:
    """Return the 4 corners of a foot rectangle in world coordinates (2D)."""
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
    """Log all left/right foot rectangles as closed LineStrips3D at z=0."""
    left_strips: list[np.ndarray] = []
    right_strips: list[np.ndarray] = []

    for fs in footsteps:
        corners = _foot_corners(fs.x, fs.y, fs.theta, foot_length, foot_width)
        corners_3d = np.column_stack([corners, np.zeros(4)])
        closed = np.concatenate([corners_3d, corners_3d[:1]], axis=0).astype(np.float32)
        if fs.side == "L":
            left_strips.append(closed)
        else:
            right_strips.append(closed)

    if left_strips:
        rr.log(
            path_left,
            rr.LineStrips3D(left_strips, colors=[[52, 152, 219, 200]] * len(left_strips)),
            static=True,
        )
    if right_strips:
        rr.log(
            path_right,
            rr.LineStrips3D(right_strips, colors=[[231, 76, 60, 200]] * len(right_strips)),
            static=True,
        )


def log_support_polygons(path_stable: str, path_unstable: str, phases) -> None:
    """Log support polygon convex hulls at z=0, split by stability."""
    stable_strips: list[np.ndarray] = []
    unstable_strips: list[np.ndarray] = []

    for phase in phases:
        poly = phase.support_polygon
        if len(poly) < 3:
            continue
        poly_3d = np.column_stack([poly, np.zeros(len(poly))])
        closed = np.concatenate([poly_3d, poly_3d[:1]], axis=0).astype(np.float32)
        if phase.stable:
            stable_strips.append(closed)
        else:
            unstable_strips.append(closed)

    if stable_strips:
        rr.log(
            path_stable,
            rr.LineStrips3D(stable_strips, colors=[[46, 204, 113, 180]] * len(stable_strips)),
            static=True,
        )
    if unstable_strips:
        rr.log(
            path_unstable,
            rr.LineStrips3D(unstable_strips, colors=[[231, 76, 60, 180]] * len(unstable_strips)),
            static=True,
        )


def log_com_stability_points(
    path_stable: str, path_unstable: str, phases, com_height: float = 0.0
) -> None:
    """Log CoM positions at each stance phase as Points3D, coloured by stability."""
    stable_pts: list[np.ndarray] = []
    unstable_pts: list[np.ndarray] = []

    for phase in phases:
        com = phase.com[:2]
        pt = np.array([com[0], com[1], com_height], dtype=np.float32)
        if phase.stable:
            stable_pts.append(pt)
        else:
            unstable_pts.append(pt)

    if stable_pts:
        rr.log(
            path_stable,
            rr.Points3D(
                np.array(stable_pts, dtype=np.float32),
                colors=[[46, 204, 113, 255]],
                radii=0.03,
            ),
            static=True,
        )
    if unstable_pts:
        rr.log(
            path_unstable,
            rr.Points3D(
                np.array(unstable_pts, dtype=np.float32),
                colors=[[231, 76, 60, 255]],
                radii=0.03,
            ),
            static=True,
        )


def log_spatial_trajectory(
    path_com: str, path_zmp: str, traj, com_height: float
) -> None:
    """Log CoM (at com_height) and ZMP (at z=0) as downsampled static overview strips."""
    T = len(traj.t)
    s = _stride(T, 2000)
    idx = range(0, T, s)

    com_pts = np.column_stack(
        [traj.x[::s], traj.y[::s], np.full(len(traj.x[::s]), com_height)]
    ).astype(np.float32)
    zmp_pts = np.column_stack(
        [traj.zmp_x[::s], traj.zmp_y[::s], np.zeros(len(list(idx)))]
    ).astype(np.float32)

    rr.log(path_com, rr.LineStrips3D([com_pts], colors=[[230, 126, 34, 255]]), static=True)
    rr.log(path_zmp, rr.LineStrips3D([zmp_pts], colors=[[155, 89, 182, 255]]), static=True)


def log_animated_trajectory(
    path_com: str, path_zmp: str, traj, com_height: float
) -> None:
    """Log CoM (at com_height) and ZMP (at z=0) as time-indexed Points3D.

    Creates moving markers that animate when the Rerun timeline is scrubbed.
    """
    T = len(traj.t)
    s = _stride(T, 1000)
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(traj.t[i]))
        rr.log(
            path_com,
            rr.Points3D(
                [[float(traj.x[i]), float(traj.y[i]), com_height]],
                colors=[[230, 126, 34, 255]],
                radii=0.05,
            ),
        )
        rr.log(
            path_zmp,
            rr.Points3D(
                [[float(traj.zmp_x[i]), float(traj.zmp_y[i]), 0.0]],
                colors=[[155, 89, 182, 255]],
                radii=0.05,
            ),
        )


def log_pendulum_rod(entity_path: str, traj, com_height: float) -> None:
    """Animated inverted-pendulum rod connecting ZMP (z=0) to CoM (z=com_height).

    Logged time-indexed so it animates together with the trajectory markers.
    """
    T = len(traj.t)
    s = _stride(T, 1000)
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(traj.t[i]))
        rod = np.array(
            [
                [float(traj.zmp_x[i]), float(traj.zmp_y[i]), 0.0],
                [float(traj.x[i]), float(traj.y[i]), com_height],
            ],
            dtype=np.float32,
        )
        rr.log(entity_path, rr.LineStrips3D([rod], colors=[[255, 255, 255, 200]], radii=0.008))


def log_com_velocity_arrows(entity_path: str, traj, com_height: float) -> None:
    """Animated velocity arrow at the CoM position, pointing in the direction of travel."""
    T = len(traj.t)
    s = _stride(T, 1000)
    scale = 0.25  # m / (m/s) — visual length per unit velocity
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(traj.t[i]))
        rr.log(
            entity_path,
            rr.Arrows3D(
                origins=[[float(traj.x[i]), float(traj.y[i]), com_height]],
                vectors=[[float(traj.vx[i]) * scale, float(traj.vy[i]) * scale, 0.0]],
                colors=[[46, 204, 113, 220]],
            ),
        )


def log_scalar_timeseries(traj, schedule) -> None:
    """Log all scalar channels time-indexed via rr.set_time_seconds."""
    _styles: list[tuple[str, list[int], str]] = [
        ("trajectory/com/position/x", [230, 126, 34], "CoM x"),
        ("trajectory/com/position/y", [243, 156, 18], "CoM y"),
        ("trajectory/zmp/x", [155, 89, 182], "ZMP x"),
        ("trajectory/zmp/y", [142, 68, 173], "ZMP y"),
        ("trajectory/zmp_ref/x", [149, 165, 166], "ZMP ref x"),
        ("trajectory/zmp_ref/y", [127, 140, 141], "ZMP ref y"),
        ("trajectory/com/velocity/x", [46, 204, 113], "vel x"),
        ("trajectory/com/velocity/y", [39, 174, 96], "vel y"),
        ("trajectory/com/acceleration/x", [52, 152, 219], "acc x"),
        ("trajectory/com/acceleration/y", [41, 128, 185], "acc y"),
    ]
    for path, color, name in _styles:
        rr.log(path, rr.SeriesLine(color=color, name=name), static=True)

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


def log_torso_box(entity_path: str, traj, com_height: float, cfg) -> None:
    """Animated torso box centred above the CoM, yaw-aligned with velocity."""
    T = len(traj.t)
    s = _stride(T, 1000)
    prev_yaw = 0.0
    center_z = com_height - cfg.pelvis_offset + cfg.torso_height / 2
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(traj.t[i]))
        vx, vy = float(traj.vx[i]), float(traj.vy[i])
        speed = math.hypot(vx, vy)
        yaw = math.atan2(vy, vx) if speed > 0.01 else prev_yaw
        prev_yaw = yaw
        rr.log(
            entity_path,
            rr.Boxes3D(
                centers=[[float(traj.x[i]), float(traj.y[i]), center_z]],
                half_sizes=[[cfg.torso_width / 2, cfg.torso_depth / 2, cfg.torso_height / 2]],
                rotations=[rr.Quaternion(xyzw=[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])],
                colors=[[200, 200, 200, 180]],
                fill_mode=rr.components.FillMode.Solid,
            ),
        )


def log_body_legs(entity_path: str, traj, footsteps, schedule, com_height: float, cfg) -> None:
    """Animated 2-link legs (hip → knee → foot) for left and right sides."""
    from robot.kinematics import active_feet_at, compute_phase_progress, two_link_knee

    phase_alpha = compute_phase_progress(schedule)
    T = len(traj.t)
    s = _stride(T, 1000)
    hip_z = com_height - cfg.pelvis_offset
    for i in range(0, T, s):
        rr.set_time_seconds("t", float(traj.t[i]))
        cx, cy = float(traj.x[i]), float(traj.y[i])
        hip_l = np.array([cx - cfg.hip_width, cy, hip_z])
        hip_r = np.array([cx + cfg.hip_width, cy, hip_z])

        vx, vy = float(traj.vx[i]), float(traj.vy[i])
        speed = math.hypot(vx, vy)
        fwd = np.array([vx / speed, vy / speed, 0.0]) if speed > 0.01 else np.array([1.0, 0.0, 0.0])

        foot_l, foot_r = active_feet_at(i, footsteps, schedule, cfg, phase_alpha)
        knee_l = two_link_knee(hip_l, foot_l, cfg.upper_leg, cfg.lower_leg, fwd)
        knee_r = two_link_knee(hip_r, foot_r, cfg.upper_leg, cfg.lower_leg, fwd)

        rr.log(
            f"{entity_path}/left",
            rr.LineStrips3D(
                [np.array([hip_l, knee_l, foot_l], dtype=np.float32)],
                colors=[[52, 152, 219, 220]],
                radii=0.012,
            ),
        )
        rr.log(
            f"{entity_path}/right",
            rr.LineStrips3D(
                [np.array([hip_r, knee_r, foot_r], dtype=np.float32)],
                colors=[[231, 76, 60, 220]],
                radii=0.012,
            ),
        )


def log_phase_transitions(entity_path: str, schedule) -> None:
    """Log phase kind (single/double) as TextLog only at transitions."""
    prev_kind = None
    for i, kind in enumerate(schedule.kind):
        if kind != prev_kind:
            rr.set_time_seconds("t", float(schedule.t[i]))
            rr.log(entity_path, rr.TextLog(kind))
            prev_kind = kind
