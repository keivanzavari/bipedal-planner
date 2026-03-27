"""LQR closed-loop CoM tracking controller.

Offline: solve DARE on the augmented LIPM system (same formulation as
stage2/preview_controller.py:compute_gains) to obtain a state-feedback
gain K.

Online: at each step k apply
    u = -K_state @ (state - ref_state) + u_ff[k]

where ref_state = [ref_pos, ref_vel, ref_acc] and u_ff is the reference
jerk estimated from the Stage 2 trajectory via forward finite-differences:
    u_ff[k] = (ax[k+1] - ax[k]) / dt   (zero at the last step)

The x and y axes are decoupled and run with the same gains.
"""

import numpy as np
from scipy.linalg import solve_discrete_are

from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams, lipm_matrices
from stage2.preview_controller import CoMTrajectory


class LQRController:
    """State-feedback LQR with feedforward jerk from the Stage 2 reference."""

    def __init__(self, Q_e: float = 1.0, R: float = 1e-6) -> None:
        self._Q_e = Q_e
        self._R = R
        # Populated by reset()
        self._K_state: np.ndarray | None = None  # (3,) — state-error gain
        self._ref_x: np.ndarray | None = None  # (T, 3)
        self._ref_y: np.ndarray | None = None  # (T, 3)
        self._u_ff_x: np.ndarray | None = None  # (T,)
        self._u_ff_y: np.ndarray | None = None  # (T,)

    def reset(self, traj: CoMTrajectory, schedule: ContactSchedule, params: LIPMParams) -> None:
        A, B, C = lipm_matrices(params)
        dt = params.dt

        # --- Augmented system (same as preview_controller.compute_gains) ---
        n = A.shape[0]  # 3
        A_aug = np.block(
            [
                [np.ones((1, 1)), (C @ A).reshape(1, -1)],
                [np.zeros((n, 1)), A],
            ]
        )  # (4, 4)
        B_aug = np.concatenate([[C @ B], B])  # (4,)

        Q_aug = np.zeros((n + 1, n + 1))
        Q_aug[0, 0] = self._Q_e

        P = solve_discrete_are(A_aug, B_aug.reshape(-1, 1), Q_aug, np.array([[self._R]]))

        BtPB = float(B_aug @ P @ B_aug) + self._R
        K_aug = (B_aug @ P @ A_aug) / BtPB  # (4,)

        # Drop the integral-error component — keep only the 3 state components
        self._K_state = K_aug[1:]  # (3,)

        # --- Reference arrays ---
        T = len(traj.t)
        self._ref_x = np.column_stack([traj.x, traj.vx, traj.ax])  # (T, 3)
        self._ref_y = np.column_stack([traj.y, traj.vy, traj.ay])  # (T, 3)

        # Forward finite-difference jerk feedforward; zero at last step
        self._u_ff_x = np.empty(T)
        self._u_ff_y = np.empty(T)
        self._u_ff_x[:-1] = (traj.ax[1:] - traj.ax[:-1]) / dt
        self._u_ff_y[:-1] = (traj.ay[1:] - traj.ay[:-1]) / dt
        self._u_ff_x[-1] = 0.0
        self._u_ff_y[-1] = 0.0

    def step(self, k: int, state_x: np.ndarray, state_y: np.ndarray) -> tuple[float, float]:
        assert self._K_state is not None, "reset() must be called before step()"
        ux = float(-self._K_state @ (state_x - self._ref_x[k]) + self._u_ff_x[k])
        uy = float(-self._K_state @ (state_y - self._ref_y[k]) + self._u_ff_y[k])
        return ux, uy
