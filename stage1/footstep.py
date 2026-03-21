from dataclasses import dataclass

import numpy as np

from stage1.world import World


@dataclass
class Footstep:
    side: str  # 'L' or 'R'
    x: float  # foot centre, world coords
    y: float
    theta: float  # heading (radians) at this step


def _resample_path(
    waypoints: list[tuple[float, float]],
    step_length: float,
) -> list[tuple[float, float, float]]:
    """
    Walk along the polyline defined by waypoints and emit evenly-spaced
    points every `step_length` metres.

    Returns list of (x, y, theta) where theta is the heading at that point.
    """
    samples = []
    if len(waypoints) < 2:
        return samples

    accumulated = 0.0
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        seg_len = np.hypot(x1 - x0, y1 - y0)
        if seg_len == 0:
            continue
        theta = np.arctan2(y1 - y0, x1 - x0)
        dx, dy = (x1 - x0) / seg_len, (y1 - y0) / seg_len

        # How far into this segment do we start emitting?
        t = (step_length - accumulated) % step_length if samples else 0.0
        while t <= seg_len:
            samples.append((x0 + dx * t, y0 + dy * t, theta))
            t += step_length
        accumulated = seg_len - (t - step_length)

    return samples


def _foot_corners(x: float, y: float, theta: float, foot_length: float, foot_width: float) -> np.ndarray:
    """
    Return the 4 corners of a foot rectangle centred at (x, y) with
    heading theta. Shape: (4, 2).
    """
    half_l = foot_length / 2
    half_w = foot_width / 2
    # Corners in local frame (forward, lateral)
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ]
    )
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([x, y])


def _foot_is_free(
    x: float,
    y: float,
    theta: float,
    foot_length: float,
    foot_width: float,
    world: World,
    grid: np.ndarray,
) -> bool:
    """Check that every grid cell under the foot rectangle is free in `grid`."""
    corners = _foot_corners(x, y, theta, foot_length, foot_width)
    steps = max(3, int(foot_length / world.resolution))
    alphas = np.linspace(0, 1, steps)
    for a in alphas:
        for b in alphas:
            p = (corners[0] * (1 - a) + corners[1] * a) * (1 - b) + (corners[3] * (1 - a) + corners[2] * a) * b
            row, col = world.world_to_cell(p[0], p[1])
            if not world.in_bounds(row, col) or grid[row, col] == 1:
                return False
    return True


def plan_footsteps(
    waypoints: list[tuple[float, float]],
    world: World,
    step_length: float = 0.25,
    step_width: float = 0.10,
    foot_length: float = 0.16,
    foot_width: float = 0.08,
    foot_clearance: float = 0.05,
    first_side: str = "L",
) -> list[Footstep]:
    """
    Generate alternating foot placements along the CoM waypoint path.

    Parameters
    ----------
    waypoints      : smoothed CoM path from the A* planner
    world          : World instance (used for collision checking)
    step_length    : forward distance between consecutive steps (m)
    step_width     : lateral offset of each foot from CoM centreline (m)
    foot_length    : foot rectangle length along heading direction (m)
    foot_width     : foot rectangle width perpendicular to heading (m)
    foot_clearance : extra margin kept between foot edge and obstacles (m)
    first_side     : which foot steps first, 'L' or 'R'

    Returns
    -------
    Ordered list of Footstep objects.
    """
    # Inflate obstacles by foot_clearance so feet never touch the boundary
    clearance_grid = world.inflated_grid(foot_clearance)

    samples = _resample_path(waypoints, step_length)
    footsteps = []
    side = first_side

    for x, y, theta in samples:
        perp = np.array([-np.sin(theta), np.cos(theta)])
        offset = step_width * perp if side == "L" else -step_width * perp
        fx, fy = x + offset[0], y + offset[1]

        if _foot_is_free(fx, fy, theta, foot_length, foot_width, world, clearance_grid):
            footsteps.append(Footstep(side=side, x=fx, y=fy, theta=theta))

        side = "R" if side == "L" else "L"

    return footsteps
