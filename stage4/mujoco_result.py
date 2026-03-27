"""MujocoResult dataclass — superset of Stage 3 TrackingResult.

Adds joint-level data, real GRF from MuJoCo contacts, ZMP comparison
(LIPM formula vs GRF-derived), CoM height tracking, and energy monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MujocoResult:
    """Full result from a Stage 4 MuJoCo simulation run."""

    # --- Same as TrackingResult (Stage 3 compatibility) ---
    t: np.ndarray  # (T,)
    x: np.ndarray  # (T,)  actual CoM x
    y: np.ndarray  # (T,)  actual CoM y (LIPM-analytical for lateral)
    vx: np.ndarray  # (T,)
    vy: np.ndarray  # (T,)
    ref_x: np.ndarray  # (T,)  Stage 2 reference CoM x
    ref_y: np.ndarray  # (T,)  Stage 2 reference CoM y
    err_x: np.ndarray  # (T,)  position error x (actual - ref)
    err_y: np.ndarray  # (T,)  position error y
    u_x: np.ndarray  # (T,)  applied jerk x
    u_y: np.ndarray  # (T,)  applied jerk y
    grf_left: np.ndarray  # (T, 3)  [Fx, Fy, Fz] from MuJoCo contacts
    grf_right: np.ndarray  # (T, 3)
    zmp_x: np.ndarray  # (T,)  actual ZMP x from GRF
    zmp_y: np.ndarray  # (T,)  actual ZMP y from GRF
    zmp_lb_x: np.ndarray  # (T,)  support-polygon lower bound x
    zmp_ub_x: np.ndarray  # (T,)  support-polygon upper bound x
    zmp_lb_y: np.ndarray  # (T,)  support-polygon lower bound y
    zmp_ub_y: np.ndarray  # (T,)  support-polygon upper bound y
    friction: np.ndarray  # (T,)  always 1.0 for Stage 4 (flat ground)

    # --- Stage 4 specific ---
    q: np.ndarray  # (T, 6)  joint angles [hip_L, knee_L, ankle_L, hip_R, knee_R, ankle_R]
    qdot: np.ndarray  # (T, 6)  joint velocities
    tau: np.ndarray  # (T, 6)  applied torques
    zmp_from_lipm: np.ndarray  # (T, 2)  ZMP via LIPM formula (x, y) for comparison
    zmp_from_grf: np.ndarray  # (T, 2)  ZMP from GRF (x, y) — ground truth
    com_height_actual: np.ndarray  # (T,)  real CoM z from MuJoCo (should stay ~0.80 m)
    energy: np.ndarray  # (T,)  total energy (kinetic + potential) from MuJoCo
