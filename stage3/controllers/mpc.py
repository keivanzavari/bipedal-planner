"""MPC (Model Predictive Control) closed-loop CoM tracking controller.

Receding-horizon QP solved at each step via OSQP. The x and y axes are
decoupled, so two independent N-variable QPs are solved per step.

Error-state formulation (per axis, at step k):
  Variables:  du ∈ R^N  (corrective jerk over prediction horizon)
  Error state: e = state - ref[k]   (tracking error)
  Error ZMP:  zmp_err = P @ du + zmp_free_err
              P            — (N, N) propagation matrix (fixed)
              zmp_free_err — (N,) free ZMP from error state e
  Objective:  min ½ du' H du + q' du
              H = 2(Q_e * P'P + R * I)        [fixed — assembled once in reset()]
              q = 2 * Q_e * P' * zmp_free_err  [updated each step]
  Constraints: l ≤ P @ du ≤ u   (ZMP error bounds, always feasible at zero error)
  Output:     u[k] = u_ff[k] + du[0]

The ZMP bounds are computed from the foot support polygon but expanded to
contain the reference ZMP at every step, ensuring the QP is always feasible
at perfect tracking (zero error state). This is necessary because the preview
controller reference trajectory can have ZMP outside the foot bounding box
during the initial transient (large preview-driven acceleration).

The Hessian H and constraint matrix P are constant, so OSQP is set up once in
reset() and only q / l / u are updated at each step, preserving warm-start.
"""

from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sp

from stage1.footstep import Footstep
from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams, lipm_matrices
from stage2.preview_controller import CoMTrajectory
from stage2.traj_optimizer import _compute_zmp_bounds, build_propagation_matrix, free_response


class MPCController:
    """Receding-horizon MPC with always-feasible ZMP error constraints."""

    def __init__(
        self,
        footsteps: list[Footstep],
        foot_length: float,
        foot_width: float,
        N_horizon: int = 100,
        Q_e: float = 1.0,
        R: float = 1e-6,
    ) -> None:
        self._footsteps = footsteps
        self._foot_length = foot_length
        self._foot_width = foot_width
        self._N = N_horizon
        self._Q_e = Q_e
        self._R = R
        # Populated by reset()
        self._A: np.ndarray | None = None
        self._C: np.ndarray | None = None
        self._P: np.ndarray | None = None      # (N, N) propagation matrix
        self._ref_x: np.ndarray | None = None  # (T, 3) reference state
        self._ref_y: np.ndarray | None = None
        self._u_ff_x: np.ndarray | None = None # (T,) feedforward jerks
        self._u_ff_y: np.ndarray | None = None
        self._zmp_ref_x: np.ndarray | None = None  # (T,) reference ZMP
        self._zmp_ref_y: np.ndarray | None = None
        # Expanded ZMP bounds: always contain the reference ZMP
        self._expanded_lb_x: np.ndarray | None = None
        self._expanded_ub_x: np.ndarray | None = None
        self._expanded_lb_y: np.ndarray | None = None
        self._expanded_ub_y: np.ndarray | None = None
        self._T: int = 0
        self._solver_x: osqp.OSQP | None = None
        self._solver_y: osqp.OSQP | None = None

    def reset(self, traj: CoMTrajectory, schedule: ContactSchedule, params: LIPMParams) -> None:
        A, B, C = lipm_matrices(params)
        N = self._N
        T = len(traj.t)
        dt = params.dt

        self._A = A
        self._C = C
        self._T = T

        # Reference state arrays (T, 3)
        self._ref_x = np.column_stack([traj.x, traj.vx, traj.ax])
        self._ref_y = np.column_stack([traj.y, traj.vy, traj.ay])

        # Feedforward jerks: u_ff[k] = (ax[k+1] - ax[k]) / dt
        u_ff_x = np.empty(T)
        u_ff_y = np.empty(T)
        u_ff_x[:-1] = (traj.ax[1:] - traj.ax[:-1]) / dt
        u_ff_x[-1] = 0.0
        u_ff_y[:-1] = (traj.ay[1:] - traj.ay[:-1]) / dt
        u_ff_y[-1] = 0.0
        self._u_ff_x = u_ff_x
        self._u_ff_y = u_ff_y

        # Reference ZMP (= C @ ref_state, stays within schedule target by design)
        self._zmp_ref_x = traj.zmp_x
        self._zmp_ref_y = traj.zmp_y

        # Raw support polygon bounds
        lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(
            schedule, self._footsteps, self._foot_length, self._foot_width
        )

        # Expand bounds to always contain the reference ZMP (with 1mm slack).
        # This ensures the QP is feasible at zero tracking error, even when the
        # preview controller's reference ZMP overshoots the foot polygon during
        # the initial acceleration transient.
        _eps = 0.001
        self._expanded_lb_x = np.minimum(lb_x, traj.zmp_x) - _eps
        self._expanded_ub_x = np.maximum(ub_x, traj.zmp_x) + _eps
        self._expanded_lb_y = np.minimum(lb_y, traj.zmp_y) - _eps
        self._expanded_ub_y = np.maximum(ub_y, traj.zmp_y) + _eps

        # Propagation matrix (N×N, fixed across all steps)
        P = build_propagation_matrix(A, B, C, N)
        self._P = P

        # --- Fixed QP Hessian: H = 2(Q_e * P'P + R * I) ---
        H_dense = 2.0 * (self._Q_e * P.T @ P + self._R * np.eye(N))
        H_sp = sp.csc_matrix(np.triu(H_dense))  # OSQP expects upper triangular

        # Fixed constraint matrix: A_c = P
        A_sp = sp.csc_matrix(P)

        # Dummy initial q / l / u — replaced at first step via update()
        q0 = np.zeros(N)
        l0 = np.full(N, -1e9)
        u0 = np.full(N, 1e9)

        solver_kwargs = dict(verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000)

        self._solver_x = osqp.OSQP()
        self._solver_x.setup(H_sp, q0, A_sp, l0, u0, **solver_kwargs)

        self._solver_y = osqp.OSQP()
        self._solver_y.setup(H_sp, q0, A_sp, l0, u0, **solver_kwargs)

    def _window(self, k: int, arr: np.ndarray) -> np.ndarray:
        """Extract N values from arr starting at k, padding end with last value."""
        end = min(k + self._N, self._T)
        w = arr[k:end]
        if len(w) < self._N:
            w = np.concatenate([w, np.full(self._N - len(w), arr[-1])])
        return w

    def step(self, k: int, state_x: np.ndarray, state_y: np.ndarray) -> tuple[float, float]:
        assert self._solver_x is not None, "reset() must be called before step()"
        P = self._P
        Q_e = self._Q_e

        # Tracking error state: e = actual - reference
        e_x = state_x - self._ref_x[k]
        e_y = state_y - self._ref_y[k]

        # Free ZMP error response: how the error ZMP evolves without correction
        zmp_free_x = free_response(self._A, self._C, e_x, self._N)
        zmp_free_y = free_response(self._A, self._C, e_y, self._N)

        # ZMP error bounds relative to reference ZMP over horizon window.
        # The expanded bounds guarantee feasibility at zero error.
        ref_zmp_x = self._window(k, self._zmp_ref_x)
        ref_zmp_y = self._window(k, self._zmp_ref_y)
        elb_x = self._window(k, self._expanded_lb_x)
        eub_x = self._window(k, self._expanded_ub_x)
        elb_y = self._window(k, self._expanded_lb_y)
        eub_y = self._window(k, self._expanded_ub_y)

        # Constraint: (expanded_lb - ref_zmp) - zmp_free ≤ P@du ≤ (expanded_ub - ref_zmp) - zmp_free
        l_x = (elb_x - ref_zmp_x) - zmp_free_x
        u_x = (eub_x - ref_zmp_x) - zmp_free_x
        l_y = (elb_y - ref_zmp_y) - zmp_free_y
        u_y = (eub_y - ref_zmp_y) - zmp_free_y

        # Linear cost: q = 2*Q_e*P'*zmp_free_err  (drive error ZMP to zero)
        q_x = 2.0 * Q_e * (P.T @ zmp_free_x)
        q_y = 2.0 * Q_e * (P.T @ zmp_free_y)

        self._solver_x.update(q=q_x, l=l_x, u=u_x)
        res_x = self._solver_x.solve()

        self._solver_y.update(q=q_y, l=l_y, u=u_y)
        res_y = self._solver_y.solve()

        # First corrective jerk; fallback to 0 on solver failure
        _OSQP_INF = 1e6
        du_x = float(res_x.x[0]) if (
            res_x.x is not None and np.isfinite(res_x.x[0]) and abs(res_x.x[0]) < _OSQP_INF
        ) else 0.0
        du_y = float(res_y.x[0]) if (
            res_y.x is not None and np.isfinite(res_y.x[0]) and abs(res_y.x[0]) < _OSQP_INF
        ) else 0.0

        return self._u_ff_x[k] + du_x, self._u_ff_y[k] + du_y
