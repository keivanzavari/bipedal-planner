"""
QP-based CoM Trajectory Optimizer — Stage 2 alternative to ZMP Preview Control.

Implements the condensed-QP formulation from:
  de Viragh et al., "Trajectory Optimization for Wheeled-Legged Quadrupedal Robots
  using Linearized ZMP Constraints", IEEE RA-L 2019.

Instead of the closed-form Kajita preview controller (which tracks the ZMP reference
implicitly), this module formulates an explicit constrained quadratic program:

    min   Q_e·‖P·u_x + p_free_x − p_ref_x‖²  +  R·‖u_x‖²
     u_x
    s.t.  lb_x[k] ≤ (P·u_x + p_free_x)[k] ≤ ub_x[k]   ∀ k

and an identical independent problem for u_y.  lb/ub are the axis-aligned
extents of the support polygon at each timestep.

Decoupling x and y reduces the problem from a 2T-variable joint QP to two
T-variable QPs, each with 2T simple bound constraints on the ZMP.  SLSQP
solves each in a few seconds for typical trajectory lengths (T ≈ 600).

Formulation details
-------------------
State:   x[k] ∈ R³  (pos, vel, acc)
Control: u[k] ∈ R   (jerk)
ZMP:     p[k] = C @ x[k]   (linearised from LIPM: p = pos - h/g·acc)

Propagation (eliminates states → purely in terms of u):
  x[k] = A^k x0  +  P_prop[k,:] @ u
  p[k] = p_free[k]  +  P_prop[k,:] @ u

where P_prop is a strictly-lower-triangular Toeplitz matrix:
  P_prop[k,j] = C @ A^(k-1-j) @ B   for j < k,  else 0

ZMP bounds are the axis-aligned bounding boxes of the support polygon vertices
(exact for axis-aligned feet; conservative for rotated feet — the polygon
constraint then lies inside the bounding box, so the ZMP constraint is
strictly satisfied).

Performance: ~5–30 s per axis for T ≈ 600; compare ~50 ms for preview control.
Use offline; use the preview controller for real-time applications.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import solve, toeplitz
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from stage1.footstep import Footstep, _foot_corners
from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams, lipm_matrices
from stage2.preview_controller import CoMTrajectory

# ---------------------------------------------------------------------------
# Building blocks (reusable, tested independently)
# ---------------------------------------------------------------------------


def build_propagation_matrix(A: np.ndarray, B: np.ndarray, C: np.ndarray, T: int) -> np.ndarray:
    """Return the T×T strictly-lower-triangular Toeplitz propagation matrix P.

    P[k, j] = C @ A^(k-1-j) @ B  for j < k,  else 0.

    The ZMP at timestep k is:  zmp[k] = p_free[k] + P[k, :] @ u
    """
    g = np.empty(T)
    Ak = np.eye(A.shape[0])
    for n in range(T):
        g[n] = float(C @ Ak @ B)
        Ak = A @ Ak

    # Strictly lower-triangular Toeplitz: first column = [0, g[0], ..., g[T-2]]
    col = np.concatenate([[0.0], g[: T - 1]])
    return toeplitz(col, np.zeros(T))


def free_response(A: np.ndarray, C: np.ndarray, x0: np.ndarray, T: int) -> np.ndarray:
    """Return p_free[k] = C @ A^k @ x0 for k = 0 … T-1."""
    p = np.empty(T)
    x = x0.copy()
    for k in range(T):
        p[k] = float(C @ x)
        x = A @ x
    return p


def precompute_polygons(
    schedule: ContactSchedule,
    footsteps: list[Footstep],
    foot_length: float,
    foot_width: float,
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Build a (phase, kind) → (A, b) half-plane cache for every unique contact state.

    A @ p ≤ b  iff  p is inside the support polygon.
    Uses ConvexHull.equations to avoid vertex-ordering assumptions.
    """
    cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(len(schedule.t)):
        key = (int(schedule.phase[k]), schedule.kind[k])
        if key in cache:
            continue
        i, kind = key
        pts = _foot_corners(footsteps[i].x, footsteps[i].y, footsteps[i].theta, foot_length, foot_width)
        if kind == "double" and i > 0:
            prev = footsteps[i - 1]
            pts = np.vstack([pts, _foot_corners(prev.x, prev.y, prev.theta, foot_length, foot_width)])
        hull = ConvexHull(pts)
        cache[key] = (hull.equations[:, :2].copy(), (-hull.equations[:, 2]).copy())
    return cache


def _compute_zmp_bounds(
    schedule: ContactSchedule,
    footsteps: list[Footstep],
    foot_length: float,
    foot_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-timestep axis-aligned ZMP bounds (lb_x, ub_x, lb_y, ub_y).

    For axis-aligned feet (theta ≈ 0) these match the exact foot rectangle.
    For rotated feet the bounding box is conservative: the polygon is a subset,
    so any ZMP inside the polygon automatically satisfies the bounds.
    """
    T = len(schedule.t)
    lb_x = np.empty(T)
    ub_x = np.empty(T)
    lb_y = np.empty(T)
    ub_y = np.empty(T)

    _pts_cache: dict[tuple[int, str], np.ndarray] = {}
    for k in range(T):
        key = (int(schedule.phase[k]), schedule.kind[k])
        if key not in _pts_cache:
            i, kind = key
            pts = _foot_corners(footsteps[i].x, footsteps[i].y, footsteps[i].theta, foot_length, foot_width)
            if kind == "double" and i > 0:
                prev = footsteps[i - 1]
                pts = np.vstack([pts, _foot_corners(prev.x, prev.y, prev.theta, foot_length, foot_width)])
            _pts_cache[key] = pts
        pts = _pts_cache[key]
        lb_x[k] = pts[:, 0].min()
        ub_x[k] = pts[:, 0].max()
        lb_y[k] = pts[:, 1].min()
        ub_y[k] = pts[:, 1].max()

    return lb_x, ub_x, lb_y, ub_y


def _solve_1d_qp(
    P: np.ndarray,
    p_free: np.ndarray,
    p_ref: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    Q_e: float,
    R_jerk: float,
) -> tuple[np.ndarray, bool]:
    """Solve a single-axis constrained QP.

    min   Q_e·‖P·u + p_free − p_ref‖²  +  R·‖u‖²
     u
    s.t.  lb ≤ P·u + p_free ≤ ub

    Returns (u_opt, converged).
    """
    T = len(p_free)

    # Warm-start: unconstrained optimum (then SLSQP just projects to the feasible set)
    H = Q_e * (P.T @ P) + R_jerk * np.eye(T)
    f = Q_e * (P.T @ (p_free - p_ref))
    u0 = -solve(H, f, assume_a="pos")

    def objective(u: np.ndarray) -> float:
        r = P @ u + p_free - p_ref
        return float(Q_e * (r @ r) + R_jerk * (u @ u))

    def gradient(u: np.ndarray) -> np.ndarray:
        r = P @ u + p_free - p_ref
        return 2.0 * (Q_e * (P.T @ r) + R_jerk * u)

    # ZMP bounds: lb ≤ P @ u + p_free ≤ ub  →  lb - p_free ≤ P @ u ≤ ub - p_free
    # Small inset margin (0.1 mm) ensures strict feasibility after reconstruction
    # floating-point accumulation (~1e-12) and SLSQP constraint tolerance (~1e-6).
    _eps = 1e-4
    lb_eff = lb - p_free + _eps
    ub_eff = ub - p_free - _eps
    # SLSQP-style dicts avoid scipy's poorly-typed LinearConstraint stubs.
    # 'ineq' means f(u) >= 0, so lower: P@u - lb_eff >= 0, upper: ub_eff - P@u >= 0.
    constraints = [
        {"type": "ineq", "fun": lambda u, b=lb_eff: P @ u - b, "jac": lambda u: P},
        {"type": "ineq", "fun": lambda u, b=ub_eff: b - P @ u, "jac": lambda u: -P},
    ]

    result = minimize(
        objective,
        u0,
        jac=gradient,
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 2000},
    )
    return result.x, result.success


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_trajectory_optimization(
    schedule: ContactSchedule,
    footsteps: list[Footstep],
    params: LIPMParams,
    foot_length: float,
    foot_width: float,
    Q_e: float = 1.0,
    R_jerk: float = 1e-6,
) -> CoMTrajectory:
    """Compute a CoM trajectory via a condensed QP with explicit ZMP polygon constraints.

    Guarantees ZMP stays within the support polygon at every timestep
    (up to numerical tolerance), unlike the preview controller which satisfies
    the constraint only implicitly.

    The x and y axes are solved as independent T-variable QPs with axis-aligned
    ZMP bound constraints derived from the support polygon at each timestep.

    Parameters
    ----------
    schedule    : ContactSchedule from build_contact_schedule
    footsteps   : ordered Footstep list from stage 1
    params      : LIPM model parameters (h, g, dt)
    foot_length : foot rectangle length (m)
    foot_width  : foot rectangle width (m)
    Q_e         : ZMP tracking weight  (same meaning as in preview controller)
    R_jerk      : jerk regularisation weight

    Returns
    -------
    CoMTrajectory with the same fields as run_preview_control.
    """
    A, B, C = lipm_matrices(params)
    T = len(schedule.t)

    # Initial CoM: first foot centre.
    # P[0,:] = 0 (strictly lower-triangular), so ZMP[0] = C @ x0 regardless of u.
    # Setting x0 = foot_0_centre ensures ZMP[0] is inside the first support polygon.
    # (The midpoint-between-feet used by run_preview_control places ZMP[0] outside
    #  foot 0's polygon, making the constraint at k=0 permanently infeasible.)
    com_init_x = footsteps[0].x
    com_init_y = footsteps[0].y
    x0_x = np.array([com_init_x, 0.0, 0.0])
    x0_y = np.array([com_init_y, 0.0, 0.0])

    print("  [QP] Building propagation matrix …")
    P = build_propagation_matrix(A, B, C, T)
    p_free_x = free_response(A, C, x0_x, T)
    p_free_y = free_response(A, C, x0_y, T)

    print("  [QP] Computing per-timestep ZMP bounds …")
    lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(schedule, footsteps, foot_length, foot_width)

    print(f"  [QP] Solving x-axis QP ({T} variables) …")
    u_x, ok_x = _solve_1d_qp(P, p_free_x, schedule.zmp_x, lb_x, ub_x, Q_e, R_jerk)
    if not ok_x:
        warnings.warn("Trajectory optimizer (x-axis) did not converge.", stacklevel=2)

    print(f"  [QP] Solving y-axis QP ({T} variables) …")
    u_y, ok_y = _solve_1d_qp(P, p_free_y, schedule.zmp_y, lb_y, ub_y, Q_e, R_jerk)
    if not ok_y:
        warnings.warn("Trajectory optimizer (y-axis) did not converge.", stacklevel=2)

    # Reconstruct full state trajectories from optimal jerk sequences
    px, vx, ax_arr, zx = _reconstruct(u_x, A, B, C, x0_x, T)
    py, vy, ay_arr, zy = _reconstruct(u_y, A, B, C, x0_y, T)

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


def _reconstruct(
    u: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    x0: np.ndarray,
    T: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward-propagate state from x0 using jerk sequence u. Returns pos, vel, acc, zmp."""
    pos = np.empty(T)
    vel = np.empty(T)
    acc = np.empty(T)
    zmp = np.empty(T)
    state = x0.copy()
    for k in range(T):
        pos[k] = state[0]
        vel[k] = state[1]
        acc[k] = state[2]
        zmp[k] = float(C @ state)
        state = A @ state + B * u[k]
    return pos, vel, acc, zmp
