"""MPC (Model Predictive Control) closed-loop CoM tracking controller.

Receding-horizon QP solved at each step via OSQP. The x and y axes are
decoupled, so two independent N-variable QPs are solved per step.

Error-state formulation with DARE terminal cost (per axis, at step k):
  Variables:  du ∈ R^N  (corrective jerk over prediction horizon)
  Error state: e = state - ref[k]   (tracking error)
  Error ZMP:  zmp_err = P @ du + zmp_free_err
              P            — (N, N) propagation matrix (fixed)
              zmp_free_err — (N,) free ZMP from error state e
  Objective:  min ½ du' H du + q' du
              H = 2(Q_e * P'P + R * I + Γ' P_∞ Γ)   [fixed]
              q = 2 Q_e P' zmp_free_err + 2 Γ' P_∞ A^N e  [updated per step]
  Terminal:   e[N]' P_∞ e[N]  (DARE cost-to-go, P_∞ from ZMP-tracking DARE)
  Output:     u[k] = u_ff[k] + du[0]
  Constraints: l ≤ P @ du ≤ u  (expanded ZMP bounds — always feasible at zero error)

The DARE terminal cost makes the finite-horizon MPC equivalent to the
infinite-horizon LQR for the ZMP-tracking problem, guaranteeing stability
for any horizon length N ≥ 1. Without it, the ZMP-only objective has a
null space (states where ZMP error = 0 but position/velocity drift) that
causes divergence.

ZMP bounds are expanded to include the reference ZMP at every step (the
preview controller trajectory can overshoot the foot bounding box during
the initial acceleration transient).
"""

from __future__ import annotations

import numpy as np
import osqp
import scipy.linalg
import scipy.sparse as sp

from stage1.footstep import Footstep
from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams, lipm_matrices
from stage2.preview_controller import CoMTrajectory
from stage2.traj_optimizer import _compute_zmp_bounds, build_propagation_matrix, free_response


class MPCController:
    """Receding-horizon MPC with DARE terminal cost and always-feasible ZMP constraints."""

    def __init__(
        self,
        footsteps: list[Footstep],
        foot_length: float,
        foot_width: float,
        N_horizon: int = 20,
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
        self._P: np.ndarray | None = None          # (N, N) ZMP propagation matrix
        self._K_terminal: np.ndarray | None = None  # (N, 3) precomputed terminal gradient
        self._ref_x: np.ndarray | None = None       # (T, 3) reference state
        self._ref_y: np.ndarray | None = None
        self._u_ff_x: np.ndarray | None = None      # (T,) feedforward jerks
        self._u_ff_y: np.ndarray | None = None
        self._zmp_ref_x: np.ndarray | None = None   # (T,) reference ZMP
        self._zmp_ref_y: np.ndarray | None = None
        self._expanded_lb_x: np.ndarray | None = None  # (T,) expanded ZMP bounds
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

        self._zmp_ref_x = traj.zmp_x
        self._zmp_ref_y = traj.zmp_y

        # ----------------------------------------------------------------
        # DARE terminal cost: solve P_∞ for the infinite-horizon ZMP problem
        #   min Σ [Q_e * (C e[j])² + R * du[j]²]
        # ----------------------------------------------------------------
        Q_state = self._Q_e * np.outer(C, C)      # 3×3
        P_inf = scipy.linalg.solve_discrete_are(
            A, B.reshape(-1, 1), Q_state, np.array([[self._R]])
        )                                           # 3×3

        # Γ[:,j] = A^{N-1-j} B  (effect of du[j] on e[N])
        Gamma = np.empty((3, N))
        Ak = np.eye(3)
        for j in range(N - 1, -1, -1):
            Gamma[:, j] = Ak @ B
            if j > 0:
                Ak = A @ Ak

        # Precompute terminal gradient: K_terminal = Γ' P_∞ A^N  (N×3)
        A_N = np.linalg.matrix_power(A, N)
        self._K_terminal = Gamma.T @ P_inf @ A_N   # (N, 3)

        # ----------------------------------------------------------------
        # ZMP propagation matrix and Hessian
        # ----------------------------------------------------------------
        P = build_propagation_matrix(A, B, C, N)
        self._P = P

        # H = 2 * (Q_e * P'P  +  R * I  +  Γ' P_∞ Γ)   [fixed]
        H_dense = 2.0 * (
            self._Q_e * P.T @ P
            + self._R * np.eye(N)
            + Gamma.T @ P_inf @ Gamma
        )
        H_sp = sp.csc_matrix(np.triu(H_dense))

        # ----------------------------------------------------------------
        # Expanded ZMP bounds: always contain the reference ZMP
        # ----------------------------------------------------------------
        lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(
            schedule, self._footsteps, self._foot_length, self._foot_width
        )
        _eps = 0.001
        self._expanded_lb_x = np.minimum(lb_x, traj.zmp_x) - _eps
        self._expanded_ub_x = np.maximum(ub_x, traj.zmp_x) + _eps
        self._expanded_lb_y = np.minimum(lb_y, traj.zmp_y) - _eps
        self._expanded_ub_y = np.maximum(ub_y, traj.zmp_y) + _eps

        # ----------------------------------------------------------------
        # OSQP setup  (constraint matrix = P, fixed across all steps)
        # ----------------------------------------------------------------
        A_sp = sp.csc_matrix(P)
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

        # Tracking error state
        e_x = state_x - self._ref_x[k]
        e_y = state_y - self._ref_y[k]

        # Free ZMP error response (how the error ZMP grows without correction)
        zmp_free_x = free_response(self._A, self._C, e_x, self._N)
        zmp_free_y = free_response(self._A, self._C, e_y, self._N)

        # Linear cost: ZMP tracking term + DARE terminal term
        q_x = 2.0 * Q_e * (P.T @ zmp_free_x) + 2.0 * (self._K_terminal @ e_x)
        q_y = 2.0 * Q_e * (P.T @ zmp_free_y) + 2.0 * (self._K_terminal @ e_y)

        # ZMP error constraint bounds (expanded to guarantee feasibility at zero error)
        ref_zmp_x = self._window(k, self._zmp_ref_x)
        ref_zmp_y = self._window(k, self._zmp_ref_y)
        elb_x = self._window(k, self._expanded_lb_x)
        eub_x = self._window(k, self._expanded_ub_x)
        elb_y = self._window(k, self._expanded_lb_y)
        eub_y = self._window(k, self._expanded_ub_y)

        l_x = (elb_x - ref_zmp_x) - zmp_free_x
        u_x = (eub_x - ref_zmp_x) - zmp_free_x
        l_y = (elb_y - ref_zmp_y) - zmp_free_y
        u_y = (eub_y - ref_zmp_y) - zmp_free_y

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
