"""
Contact schedule: assigns timing to each footstep and builds the
piecewise ZMP reference trajectory used by the preview controller.

Walk cycle per step i:
  - Double support (if i > 0): ZMP linearly interpolates from foot[i-1] to foot[i]
  - Single support:            ZMP held at foot[i] centre
"""

import numpy as np
from dataclasses import dataclass

from stage1.footstep import Footstep, _foot_corners


@dataclass
class ContactSchedule:
    t:       np.ndarray   # (T,)  time in seconds
    zmp_x:   np.ndarray   # (T,)  ZMP x reference
    zmp_y:   np.ndarray   # (T,)  ZMP y reference
    phase:   np.ndarray   # (T,)  footstep index active at this timestep
    kind:    list[str]    # (T,)  'single' or 'double'


def build_contact_schedule(
    footsteps: list[Footstep],
    t_single:  float = 0.4,    # single support duration (s)
    t_double:  float = 0.1,    # double support duration (s)
    dt:        float = 0.005,  # timestep (s)
) -> ContactSchedule:
    """
    Build a time-stamped ZMP reference trajectory from an ordered footstep list.

    Parameters
    ----------
    footsteps : ordered list of Footstep from stage 1
    t_single  : how long each foot is planted while the other swings
    t_double  : how long both feet are on the ground between steps
    dt        : trajectory timestep (should match LIPM dt)

    Returns
    -------
    ContactSchedule with arrays aligned to the trajectory timestep grid.
    """
    ts, zx, zy, ph, kn = [], [], [], [], []
    t = 0.0

    for i, fs in enumerate(footsteps):

        # --- Double support: ZMP slides from previous foot to current ---
        if i > 0:
            prev = footsteps[i - 1]
            n_ds = max(1, round(t_double / dt))
            for j in range(n_ds):
                alpha = j / n_ds
                ts.append(t);  t += dt
                zx.append(prev.x + alpha * (fs.x - prev.x))
                zy.append(prev.y + alpha * (fs.y - prev.y))
                ph.append(i);  kn.append("double")

        # --- Single support: ZMP held at current foot centre ---
        n_ss = max(1, round(t_single / dt))
        for _ in range(n_ss):
            ts.append(t);  t += dt
            zx.append(fs.x)
            zy.append(fs.y)
            ph.append(i);  kn.append("single")

    return ContactSchedule(
        t=np.array(ts),
        zmp_x=np.array(zx),
        zmp_y=np.array(zy),
        phase=np.array(ph, dtype=int),
        kind=kn,
    )


def support_polygon_at(
    schedule: ContactSchedule,
    k: int,
    footsteps: list[Footstep],
    foot_length: float = 0.16,
    foot_width:  float = 0.08,
) -> np.ndarray:
    """
    Return the (N, 2) support polygon at timestep k.
    Single support → planted foot corners.
    Double support → convex hull of both feet corners.
    """
    from scipy.spatial import ConvexHull

    i = int(schedule.phase[k])
    corners = _foot_corners(
        footsteps[i].x, footsteps[i].y, footsteps[i].theta,
        foot_length, foot_width,
    )

    if schedule.kind[k] == "double" and i > 0:
        prev_corners = _foot_corners(
            footsteps[i - 1].x, footsteps[i - 1].y, footsteps[i - 1].theta,
            foot_length, foot_width,
        )
        pts = np.vstack([corners, prev_corners])
        hull = ConvexHull(pts)
        return pts[hull.vertices]

    return corners
