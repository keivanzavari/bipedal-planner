from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from footstep import Footstep, _foot_corners


@dataclass
class StancePhase:
    index: int  # step index
    kind: str  # 'double' or 'single'
    planted: list[Footstep]  # 1 or 2 planted feet
    com: np.ndarray  # (2,) CoM position used for check
    support_polygon: np.ndarray  # (N, 2) convex hull vertices (ordered)
    stable: bool


def _foot_polygon(fs: Footstep, foot_length: float, foot_width: float) -> np.ndarray:
    """Return the 4 corners of a foot as (4, 2) array."""
    return _foot_corners(fs.x, fs.y, fs.theta, foot_length, foot_width)


def _convex_hull_points(points: np.ndarray) -> np.ndarray:
    """Return ordered convex hull vertices from a (N, 2) point array."""
    if len(points) < 3:
        return points
    hull = ConvexHull(points)
    return points[hull.vertices]


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Ray-casting test: is `point` inside `polygon`?
    polygon is (N, 2) ordered vertices.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def check_stability(
    footsteps: list[Footstep],
    foot_length: float = 0.16,
    foot_width: float = 0.08,
) -> list[StancePhase]:
    """
    Evaluate static stability for each stance phase along the footstep sequence.

    Walk cycle modelled as alternating:
      - Double support: both current and previous foot on ground
      - Single support: only current foot planted, next foot swinging
                        CoM = midpoint between current and next footstep

    Parameters
    ----------
    footsteps   : ordered list of Footstep objects
    foot_length : foot rectangle length (m)
    foot_width  : foot rectangle width (m)

    Returns
    -------
    List of StancePhase objects, one per step.
    """
    phases = []

    for i, fs in enumerate(footsteps):
        corners = _foot_polygon(fs, foot_length, foot_width)

        # --- Double support phase: current + previous foot both planted ---
        if i > 0:
            prev_corners = _foot_polygon(footsteps[i - 1], foot_length, foot_width)
            ds_points = np.vstack([corners, prev_corners])
            ds_polygon = _convex_hull_polygon(ds_points)
            ds_com = np.array([(fs.x + footsteps[i - 1].x) / 2, (fs.y + footsteps[i - 1].y) / 2])
            phases.append(
                StancePhase(
                    index=i,
                    kind="double",
                    planted=[footsteps[i - 1], fs],
                    com=ds_com,
                    support_polygon=ds_polygon,
                    stable=_point_in_polygon(ds_com, ds_polygon),
                )
            )

        # --- Single support phase: only current foot planted, next swings ---
        if i < len(footsteps) - 1:
            ss_polygon = _convex_hull_polygon(corners)
            ss_com = np.array([(fs.x + footsteps[i + 1].x) / 2, (fs.y + footsteps[i + 1].y) / 2])
            phases.append(
                StancePhase(
                    index=i,
                    kind="single",
                    planted=[fs],
                    com=ss_com,
                    support_polygon=ss_polygon,
                    stable=_point_in_polygon(ss_com, ss_polygon),
                )
            )

    return phases


def _convex_hull_polygon(points: np.ndarray) -> np.ndarray:
    """Convex hull of points, returned as ordered (N, 2) vertices."""
    if len(points) < 3:
        return points
    hull = ConvexHull(points)
    return points[hull.vertices]


def stability_summary(phases: list[StancePhase]) -> dict:
    total = len(phases)
    unstable = [p for p in phases if not p.stable]
    return {
        "total_phases": total,
        "stable": total - len(unstable),
        "unstable": len(unstable),
        "unstable_indices": [p.index for p in unstable],
    }
