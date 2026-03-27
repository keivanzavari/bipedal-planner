"""Inverse-kinematics angle extraction for the planar biped.

Given world-frame positions of hip, knee, and foot (from ``two_link_knee``),
extract joint angles ``(theta_hip, theta_knee, theta_ankle)`` in the sagittal
(x-z) plane, consistent with the MJCF joint conventions in ``stage4/models/biped.xml``.

Sign conventions (all hinge axes are ``0 1 0``):
- **hip_pitch**: 0 = upper leg points straight down; positive = leg swings forward (foot ahead of hip).
- **knee**: 0 = straight leg; negative = knee flexion (MuJoCo range ``[-2.5, -0.05]``).
- **ankle**: compensates hip + knee so that the foot stays parallel to the ground.
"""

from __future__ import annotations

import math

import numpy as np


def sagittal_ik(
    hip_pos: np.ndarray,
    foot_pos: np.ndarray,
    l1: float,
    l2: float,
) -> tuple[float, float, float]:
    """Sagittal-plane 2-link IK that always picks the knee-forward (flexion) solution.

    Unlike ``angles_from_ik`` (which extracts angles from pre-computed 3-D
    positions), this function solves the IK directly in the x-z plane and
    chooses the solution where the knee protrudes anteriorly (+x), guaranteeing
    **negative** knee angles compatible with the MuJoCo joint range ``[-2.5, -0.05]``.

    Parameters
    ----------
    hip_pos  : (3,) hip joint position in world frame
    foot_pos : (3,) foot/ankle position in world frame
    l1       : upper leg length
    l2       : lower leg length

    Returns
    -------
    theta_hip, theta_knee, theta_ankle  (all in rad)
    """
    dx = foot_pos[0] - hip_pos[0]
    dz = foot_pos[2] - hip_pos[2]
    L = math.sqrt(dx * dx + dz * dz)
    L_eff = min(L, l1 + l2 - 1e-6) if L > 1e-9 else 1e-6

    # Angle of hip→foot vector from the downward vertical
    phi = math.atan2(dx, -dz)

    # Law of cosines: half-angle at hip
    cos_alpha = (l1 * l1 + L_eff * L_eff - l2 * l2) / (2.0 * l1 * L_eff)
    alpha = math.acos(max(-1.0, min(1.0, cos_alpha)))

    # Two candidate hip angles; pick the one placing the knee more forward (+x)
    th1 = phi + alpha
    th2 = phi - alpha
    knee_x1 = hip_pos[0] + l1 * math.sin(th1)
    knee_x2 = hip_pos[0] + l1 * math.sin(th2)
    theta_hip = th1 if knee_x1 >= knee_x2 else th2

    # Knee world position (for lower-leg angle extraction)
    knee_x = hip_pos[0] + l1 * math.sin(theta_hip)
    knee_z = hip_pos[2] - l1 * math.cos(theta_hip)

    # Lower-leg absolute angle from vertical
    dx_lower = foot_pos[0] - knee_x
    dz_lower = foot_pos[2] - knee_z
    theta_lower_abs = math.atan2(dx_lower, -dz_lower)

    theta_knee = theta_lower_abs - theta_hip
    theta_ankle = -(theta_hip + theta_knee)

    return theta_hip, theta_knee, theta_ankle


def angles_from_ik(
    hip_pos: np.ndarray,
    knee_pos: np.ndarray,
    foot_pos: np.ndarray,
) -> tuple[float, float, float]:
    """Extract sagittal-plane joint angles from world-frame IK positions.

    Parameters
    ----------
    hip_pos  : (3,) hip joint position in world frame
    knee_pos : (3,) knee joint position in world frame (from ``two_link_knee``)
    foot_pos : (3,) foot/ankle position in world frame

    Returns
    -------
    theta_hip   : hip pitch angle (rad) — positive = leg forward
    theta_knee  : knee angle (rad) — negative = flexion
    theta_ankle : ankle angle (rad) — compensates to keep foot flat
    """
    # Upper-leg vector in x-z plane (ignore y)
    dx_upper = knee_pos[0] - hip_pos[0]
    dz_upper = knee_pos[2] - hip_pos[2]
    # Hip angle from vertical: atan2(forward, -down)
    theta_hip = math.atan2(dx_upper, -dz_upper)

    # Lower-leg vector in x-z plane
    dx_lower = foot_pos[0] - knee_pos[0]
    dz_lower = foot_pos[2] - knee_pos[2]
    # Absolute angle of lower leg from vertical
    theta_lower_abs = math.atan2(dx_lower, -dz_lower)

    # Knee angle = relative angle between upper and lower leg
    # With straight leg = 0, flexion = negative
    theta_knee = theta_lower_abs - theta_hip

    # Ankle compensates so the foot stays flat (parallel to ground)
    # Total sagittal rotation at the foot = hip + knee + ankle = 0 for flat foot
    theta_ankle = -(theta_hip + theta_knee)

    return theta_hip, theta_knee, theta_ankle
