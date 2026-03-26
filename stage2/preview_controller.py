"""
ZMP Preview Controller (Kajita 2003).

Offline: solve a discrete-time LQR on the augmented LIPM system to obtain
         state-feedback gain K and preview gains Gp[0..N-1].

Online:  at each timestep, apply the control law:
         u[k] = -K @ X_aug[k] - Gp @ p_ref[k+1 : k+1+N]

The x and y axes are decoupled — run independently with the same gains.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_are

from stage1.footstep import Footstep
from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams, lipm_matrices


@dataclass
class PreviewGains:
    K: np.ndarray  # (n+1,)  augmented state-feedback gain
    Gp: np.ndarray  # (N,)    preview gains
    A: np.ndarray  # (3, 3)  LIPM A matrix (stored for online use)
    B: np.ndarray  # (3,)    LIPM B matrix
    C: np.ndarray  # (3,)    LIPM C matrix


@dataclass
class CoMTrajectory:
    t: np.ndarray  # (T,)  time
    x: np.ndarray  # (T,)  CoM x
    y: np.ndarray  # (T,)  CoM y
    vx: np.ndarray  # (T,)  CoM x velocity
    vy: np.ndarray  # (T,)  CoM y velocity
    ax: np.ndarray  # (T,)  CoM x acceleration
    ay: np.ndarray  # (T,)  CoM y acceleration
    zmp_x: np.ndarray  # (T,)  ZMP x
    zmp_y: np.ndarray  # (T,)  ZMP y


def compute_gains(
    params: LIPMParams,
    Q_e: float = 1.0,
    R: float = 1e-6,
    N_preview: int = 200,
) -> PreviewGains:
    """
    Compute LQR state-feedback and preview gains offline.

    Augmented system (1D):
        X̃ = [e_int, pos, vel, acc]   (e_int = running ZMP error integral)

        X̃[k+1] = Ã @ X̃[k] + B̃ * u[k] + f * p_ref[k+1]

    where:
        Ã = [[1, C@A],   B̃ = [[C@B],   f = [[-1],
             [0,   A]]         [ B  ]]        [ 0 ]]

    Cost:  J = Σ Q_e * e_int² + R * u²
    """
    A, B, C = lipm_matrices(params)
    n = A.shape[0]  # = 3

    # --- Augmented system ---
    A_aug = np.block(
        [
            [np.ones((1, 1)), (C @ A).reshape(1, -1)],
            [np.zeros((n, 1)), A],
        ]
    )  # (4, 4)
    B_aug = np.concatenate([[C @ B], B])  # (4,)
    f_aug = np.array([-1.0, 0.0, 0.0, 0.0])  # (4,)

    Q_aug = np.zeros((n + 1, n + 1))
    Q_aug[0, 0] = Q_e

    # --- Solve DARE ---
    P = solve_discrete_are(
        A_aug,
        B_aug.reshape(-1, 1),
        Q_aug,
        np.array([[R]]),
    )  # (4, 4)

    # --- State-feedback gain ---
    BtPB = float(B_aug @ P @ B_aug) + R
    K = (B_aug @ P @ A_aug) / BtPB  # (4,)

    # --- Preview gains ---
    A_cl = A_aug - np.outer(B_aug, K)  # closed-loop augmented
    Gp = np.zeros(N_preview)
    Ptf = P @ f_aug  # (4,)
    for j in range(N_preview):
        Gp[j] = -(B_aug @ Ptf) / BtPB
        Ptf = A_cl.T @ Ptf

    return PreviewGains(K=K, Gp=Gp, A=A, B=B, C=C)


def _run_1d(
    zmp_ref: np.ndarray,
    com_init: float,
    gains: PreviewGains,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run 1D preview control along one axis.

    Returns: pos (T,), vel (T,), acc (T,), zmp (T,)
    """
    T = len(zmp_ref)
    N = len(gains.Gp)
    A, B, C = gains.A, gains.B, gains.C

    state = np.array([com_init, 0.0, 0.0])
    e_int = 0.0

    pos = np.zeros(T)
    vel = np.zeros(T)
    acc = np.zeros(T)
    zmp = np.zeros(T)

    for k in range(T):
        p = float(C @ state)
        e_int += p - zmp_ref[k]

        X_aug = np.concatenate([[e_int], state])

        # Preview window — pad with last ZMP reference at trajectory end
        end = min(k + 1 + N, T)
        preview = zmp_ref[k + 1 : end]
        if len(preview) < N:
            preview = np.append(
                preview,
                np.full(N - len(preview), zmp_ref[-1]),
            )

        u = -float(gains.K @ X_aug) + float(gains.Gp @ preview)
        state = A @ state + B * u

        pos[k] = state[0]
        vel[k] = state[1]
        acc[k] = state[2]
        zmp[k] = float(C @ state)

    return pos, vel, acc, zmp


def run_preview_control(
    schedule: ContactSchedule,
    footsteps: list[Footstep],
    gains: PreviewGains,
) -> CoMTrajectory:
    """
    Run preview control for both x and y axes.

    Initial CoM is set to the midpoint between the first two footsteps,
    which is a reasonable double-support starting position.
    """
    com_init_x = (footsteps[0].x + footsteps[1].x) / 2.0
    com_init_y = (footsteps[0].y + footsteps[1].y) / 2.0

    px, vx, ax_arr, zx = _run_1d(schedule.zmp_x, com_init_x, gains)
    py, vy, ay_arr, zy = _run_1d(schedule.zmp_y, com_init_y, gains)

    return CoMTrajectory(
        t=schedule.t,
        x=px,
        y=py,
        vx=vx,
        vy=vy,
        ax=ax_arr,
        ay=ay_arr,
        zmp_x=zx,
        zmp_y=zy,
    )


def validate_zmp(
    traj: CoMTrajectory,
    schedule: ContactSchedule,
    footsteps: list[Footstep],
    foot_length: float = 0.16,
    foot_width: float = 0.08,
) -> dict[str, Any]:
    """
    Check that the ZMP stays inside the support polygon at every timestep.
    Returns a summary dict.
    """
    from stage1.stability import _point_in_polygon
    from stage2.contact_schedule import support_polygon_at

    T = len(traj.t)
    n_fail = 0
    fail_idx = []

    for k in range(T):
        poly = support_polygon_at(schedule, k, footsteps, foot_length, foot_width)
        pt = np.array([traj.zmp_x[k], traj.zmp_y[k]])
        if not _point_in_polygon(pt, poly):
            n_fail += 1
            if len(fail_idx) < 20:  # cap stored indices
                fail_idx.append(k)

    return {
        "total_steps": T,
        "zmp_violations": n_fail,
        "violation_rate": n_fail / T,
        "first_failures": fail_idx,
    }
