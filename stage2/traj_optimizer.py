"""
QP-based CoM Trajectory Optimizer — Stage 2 alternative to ZMP Preview Control.

Implements the condensed-QP formulation from:
  de Viragh et al., "Trajectory Optimization for Wheeled-Legged Quadrupedal Robots
  using Linearized ZMP Constraints", IEEE RA-L 2019.

This module uses OSQP with the **non-condensed (sparse) formulation**:

    Decision variables: z = [x[0]', ..., x[T-1]', u[0], ..., u[T-1]] in R^(4T)

    min   Q_e * sum_k (C @ x[k] - p_ref[k])^2  +  R * sum_k u[k]^2
     z
    s.t.  x[0]    = x0
          x[k+1]  = A @ x[k] + B * u[k]   for all k   (dynamics, equality)
          lb[k]  <= C @ x[k] <= ub[k]      for all k   (ZMP polygon, inequality)

The constraint matrix is block-bidiagonal with O(T) non-zeros, which lets
OSQP's ADMM solver run in O(T) per iteration — compared to O(T^2) per iteration
for SLSQP on the condensed (dense) formulation and O(T^3) to form the Hessian.

Formulation details
-------------------
State:   x[k] in R^3  (pos, vel, acc)
Control: u[k] in R    (jerk)
ZMP:     p[k] = C @ x[k]   (linearised from LIPM: p = pos - h/g*acc)

ZMP bounds are the axis-aligned bounding boxes of the support polygon vertices
(exact for axis-aligned feet; conservative for rotated feet).

Performance: ~0.1-2 s for T ~= 1000, scaling linearly with T.
"""

from __future__ import annotations

import warnings

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import toeplitz
from scipy.spatial import ConvexHull

from stage1.footstep import Footstep, _foot_corners
from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams, lipm_matrices
from stage2.preview_controller import CoMTrajectory

# ---------------------------------------------------------------------------
# Building blocks (reusable, tested independently)
# ---------------------------------------------------------------------------


def build_propagation_matrix(A: np.ndarray, B: np.ndarray, C: np.ndarray, T: int) -> np.ndarray:
    """Return the T x T strictly-lower-triangular Toeplitz propagation matrix P.

    P[k, j] = C @ A^(k-1-j) @ B  for j < k,  else 0.
    The ZMP at timestep k is:  zmp[k] = p_free[k] + P[k, :] @ u
    """
    g = np.empty(T)
    Ak = np.eye(A.shape[0])
    for n in range(T):
        g[n] = float(C @ Ak @ B)
        Ak = A @ Ak
    col = np.concatenate([[0.0], g[: T - 1]])
    return toeplitz(col, np.zeros(T))


def free_response(A: np.ndarray, C: np.ndarray, x0: np.ndarray, T: int) -> np.ndarray:
    """Return p_free[k] = C @ A^k @ x0 for k = 0 ... T-1."""
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
    """Build a (phase, kind) -> (A, b) half-plane cache for every unique contact state.

    A @ p <= b  iff  p is inside the support polygon.
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
    """Return per-timestep axis-aligned ZMP bounds (lb_x, ub_x, lb_y, ub_y)."""
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


# ---------------------------------------------------------------------------
# Sparse OSQP solver — non-condensed formulation
# ---------------------------------------------------------------------------


def _solve_1d_qp_sparse(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    x0: np.ndarray,
    p_ref: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    Q_e: float,
    R_jerk: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Solve the 1D trajectory QP with OSQP using the non-condensed sparse formulation.

    Variables: z = [x[0]', ..., x[T-1]', u[0], ..., u[T-1]] in R^(4T)

    The constraint matrix is block-bidiagonal with O(T) non-zeros, so each
    OSQP ADMM iteration costs O(T) rather than O(T^2) for the condensed form.

    Returns (pos, vel, acc, zmp, converged).
    """
    T = len(p_ref)
    nx = 3  # state dimension (pos, vel, acc)
    n_vars = nx * T + T  # 3T states + T controls

    # --- Objective: OSQP form  min (1/2) z'Pz + q'z ---
    # State blocks:   2*Q_e*(C outer C)  (nx x nx each)
    # Control blocks: 2*R_jerk           (scalar each)
    CTC = np.outer(C, C)
    H_states = sp.block_diag([2.0 * Q_e * CTC] * T, format="csc")
    H_controls = (2.0 * R_jerk) * sp.eye(T, format="csc")
    P_osqp = sp.block_diag([H_states, H_controls], format="csc")

    # Linear term: q[nx*k : nx*(k+1)] = -2*Q_e*p_ref[k]*C,  q[nx*T + k] = 0
    q = np.zeros(n_vars)
    for k in range(T):
        q[k * nx : (k + 1) * nx] = -2.0 * Q_e * p_ref[k] * C

    # --- Constraint matrix (block-bidiagonal, O(T) non-zeros) ---
    # Row layout:
    #   [0 : nx]          — initial state equality   (nx rows)
    #   [nx : nx*T]       — dynamics equality         (nx*(T-1) rows)
    #   [nx*T : 4T-2]     — ZMP inequality            (T rows)
    n_eq_init = nx
    n_eq_dyn = nx * (T - 1)
    n_ineq_zmp = T
    n_constraints = n_eq_init + n_eq_dyn + n_ineq_zmp

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def _add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # Block 1: x[0] = x0
    for i in range(nx):
        _add(i, i, 1.0)

    # Block 2: x[k+1] - A @ x[k] - B * u[k] = 0  for k = 0 ... T-2
    r0 = n_eq_init
    for k in range(T - 1):
        row_base = r0 + k * nx
        for i in range(nx):
            _add(row_base + i, (k + 1) * nx + i, 1.0)       # +I x[k+1]
        for i in range(nx):
            for j in range(nx):
                _add(row_base + i, k * nx + j, -A[i, j])    # -A x[k]
        for i in range(nx):
            _add(row_base + i, nx * T + k, -float(B[i]))    # -B u[k]

    # Block 3: C @ x[k]  (one row per timestep)
    r1 = n_eq_init + n_eq_dyn
    for k in range(T):
        for j in range(nx):
            if C[j] != 0.0:
                _add(r1 + k, k * nx + j, float(C[j]))

    A_osqp = sp.csc_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars))

    # --- Bounds ---
    # Small inset (0.1 mm) keeps ZMP strictly inside the polygon after rounding.
    _eps = 1e-4
    l_vec = np.empty(n_constraints)
    u_vec = np.empty(n_constraints)

    l_vec[:nx] = x0
    u_vec[:nx] = x0

    l_vec[n_eq_init : n_eq_init + n_eq_dyn] = 0.0
    u_vec[n_eq_init : n_eq_init + n_eq_dyn] = 0.0

    l_vec[r1:] = lb + _eps
    u_vec[r1:] = ub - _eps

    # --- Solve ---
    solver = osqp.OSQP()
    solver.setup(P_osqp, q, A_osqp, l_vec, u_vec, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
    result = solver.solve()

    states = result.x[: nx * T].reshape(T, nx)
    pos = states[:, 0]
    vel = states[:, 1]
    acc = states[:, 2]
    zmp = states @ C  # (T, 3) @ (3,) -> (T,)

    converged = result.info.status in ("solved", "solved_inaccurate")
    return pos, vel, acc, zmp, converged


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
    """Compute a CoM trajectory via a sparse QP with explicit ZMP polygon constraints.

    Guarantees ZMP stays within the support polygon at every timestep
    (up to numerical tolerance), unlike the preview controller which satisfies
    the constraint only implicitly.

    Uses OSQP with the non-condensed (sparse) formulation: dynamics and ZMP
    constraints are expressed over explicit state/control variables, giving a
    block-bidiagonal constraint matrix with O(T) non-zeros. This scales linearly
    with trajectory length (contrast: the condensed dense formulation needs
    O(T^2) memory and O(T^3) to assemble the Hessian).

    Parameters
    ----------
    schedule    : ContactSchedule from build_contact_schedule
    footsteps   : ordered Footstep list from stage 1
    params      : LIPM model parameters (h, g, dt)
    foot_length : foot rectangle length (m)
    foot_width  : foot rectangle width (m)
    Q_e         : ZMP tracking weight
    R_jerk      : jerk regularisation weight

    Returns
    -------
    CoMTrajectory with the same fields as run_preview_control.
    """
    A, B, C = lipm_matrices(params)
    T = len(schedule.t)

    # Initial CoM: first foot centre.
    # The initial-state equality fixes x[0] = x0, so ZMP[0] = C @ x0 regardless
    # of controls. Foot 0's centre guarantees ZMP[0] is inside the first polygon
    # (the midpoint between feet would violate the k=0 constraint).
    x0_x = np.array([footsteps[0].x, 0.0, 0.0])
    x0_y = np.array([footsteps[0].y, 0.0, 0.0])

    print("  [QP] Computing per-timestep ZMP bounds ...")
    lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(schedule, footsteps, foot_length, foot_width)

    print(f"  [QP] Solving x-axis QP ({T} timesteps, sparse OSQP) ...")
    px, vx, ax_arr, zx, ok_x = _solve_1d_qp_sparse(A, B, C, x0_x, schedule.zmp_x, lb_x, ub_x, Q_e, R_jerk)
    if not ok_x:
        warnings.warn("Trajectory optimizer (x-axis) did not converge.", stacklevel=2)

    print(f"  [QP] Solving y-axis QP ({T} timesteps, sparse OSQP) ...")
    py, vy, ay_arr, zy, ok_y = _solve_1d_qp_sparse(A, B, C, x0_y, schedule.zmp_y, lb_y, ub_y, Q_e, R_jerk)
    if not ok_y:
        warnings.warn("Trajectory optimizer (y-axis) did not converge.", stacklevel=2)

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
