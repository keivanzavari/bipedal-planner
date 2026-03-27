"""
Kinematic helpers for the 2-link bipedal stick figure.

All geometry is inferred from CoM position and the known footstep schedule —
no physics simulation or joint-angle tracking is required.

Coordinate convention: x forward, y left, z up (RIGHT_HAND_Z_UP).
"""

from __future__ import annotations

import numpy as np

from robot.config import RobotConfig


# ---------------------------------------------------------------------------
# 2-link inverse kinematics
# ---------------------------------------------------------------------------

def _knee_bend_direction(d_hat: np.ndarray, fallback_forward: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to d_hat, biased upward (places knee anterior).

    Computes the component of world-up that is perpendicular to the leg axis.
    When the leg is near-vertical this degenerates, so we fall back to the
    locomotion forward direction.
    """
    z_up = np.array([0.0, 0.0, 1.0])
    z_perp = z_up - np.dot(z_up, d_hat) * d_hat
    norm = float(np.linalg.norm(z_perp))
    if norm > 1e-6:
        return z_perp / norm
    # Leg near-vertical: knee goes in the locomotion forward direction
    f_perp = fallback_forward - np.dot(fallback_forward, d_hat) * d_hat
    f_norm = float(np.linalg.norm(f_perp))
    if f_norm > 1e-6:
        return f_perp / f_norm
    return np.array([1.0, 0.0, 0.0])


def two_link_knee(
    hip: np.ndarray,
    foot: np.ndarray,
    l1: float,
    l2: float,
    forward: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 3-D knee position for a 2-link leg using closed-form IK.

    Parameters
    ----------
    hip     : (3,) hip joint position in world frame
    foot    : (3,) foot (ankle) position in world frame
    l1      : upper leg (femur) length in metres
    l2      : lower leg (tibia) length in metres
    forward : (3,) unit vector giving the locomotion forward direction,
              used only when the leg is near-vertical (degenerate case).
              Defaults to [1, 0, 0] when None.

    Returns
    -------
    knee : (3,) knee joint position
    """
    fwd = forward if forward is not None else np.array([1.0, 0.0, 0.0])

    d = foot - hip
    L = float(np.linalg.norm(d))
    # Clamp reach to avoid gimbal at full extension
    L_eff = min(L, l1 + l2 - 1e-6) if L > 1e-9 else 1e-6
    d_hat = d / max(L, 1e-9)

    # Law of cosines: angle at hip joint
    cos_alpha = (l1**2 + L_eff**2 - l2**2) / (2.0 * l1 * L_eff)
    alpha = float(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))

    perp = _knee_bend_direction(d_hat, fwd)
    return hip + l1 * (d_hat * np.cos(alpha) + perp * np.sin(alpha))


# ---------------------------------------------------------------------------
# Foot position helpers
# ---------------------------------------------------------------------------

def compute_phase_progress(schedule) -> np.ndarray:
    """Pre-compute fractional progress α ∈ [0, 1) within each support phase.

    Each contiguous block of identical (phase-index, kind) is one phase segment.
    Returns an array of shape (T,) aligned to schedule.t.
    """
    T = len(schedule.t)
    alpha = np.zeros(T)
    k = 0
    while k < T:
        k_start = k
        phase_i = int(schedule.phase[k])
        phase_kind = schedule.kind[k]
        while (
            k < T
            and int(schedule.phase[k]) == phase_i
            and schedule.kind[k] == phase_kind
        ):
            k += 1
        n = k - k_start
        for j in range(n):
            alpha[k_start + j] = j / n
    return alpha


def active_feet_at(
    k: int,
    footsteps: list,
    schedule,
    cfg: RobotConfig,
    phase_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_xyz, right_xyz) foot positions at timestep index k.

    During single support the swing foot follows a parabolic arc from its
    previous planted position to its next target, parameterised by the
    fractional phase progress stored in ``phase_alpha`` (see
    :func:`compute_phase_progress`).

    Parameters
    ----------
    k           : timestep index into schedule arrays
    footsteps   : ordered Footstep list from Stage 1
    schedule    : ContactSchedule from Stage 2
    cfg         : RobotConfig (provides foot_clearance)
    phase_alpha : (T,) array from compute_phase_progress

    Returns
    -------
    left_xyz, right_xyz : (3,) arrays in world frame, z=0 for planted feet
    """
    i = int(schedule.phase[k])
    kind = schedule.kind[k]
    n = len(footsteps)

    if kind == "double":
        # Both feet planted at their footstep positions
        fs_curr = footsteps[i]
        fs_prev = footsteps[i - 1] if i > 0 else fs_curr
        left_xyz: np.ndarray | None = None
        right_xyz: np.ndarray | None = None
        for fs in (fs_curr, fs_prev):
            xyz = np.array([fs.x, fs.y, 0.0])
            if fs.side == "L":
                left_xyz = xyz
            else:
                right_xyz = xyz
        # Fallback: both steps happen to be same side (shouldn't occur with
        # alternating L/R, but guard against edge cases)
        if left_xyz is None:
            left_xyz = right_xyz  # type: ignore[assignment]
        if right_xyz is None:
            right_xyz = left_xyz
        return left_xyz, right_xyz  # type: ignore[return-value]

    # Single support: stance foot is planted, swing foot follows a parabola
    stance_xyz = np.array([footsteps[i].x, footsteps[i].y, 0.0])
    alpha = float(phase_alpha[k])

    # Swing source: previous footstep of the swing side (= footsteps[i-1])
    # Swing destination: next footstep of the swing side (= footsteps[i+1])
    if i > 0:
        src = footsteps[i - 1]
    else:
        # First step: no previous footstep; start swing at destination
        src = footsteps[i + 1] if i + 1 < n else footsteps[i]

    dst = footsteps[i + 1] if i + 1 < n else src

    swing_x = (1.0 - alpha) * src.x + alpha * dst.x
    swing_y = (1.0 - alpha) * src.y + alpha * dst.y
    swing_z = cfg.foot_clearance * 4.0 * alpha * (1.0 - alpha)
    swing_xyz = np.array([swing_x, swing_y, swing_z])

    if footsteps[i].side == "L":
        return stance_xyz, swing_xyz   # left=stance, right=swing
    else:
        return swing_xyz, stance_xyz   # left=swing, right=stance
