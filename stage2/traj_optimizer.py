"""
QP-based CoM Trajectory Optimizer — Stage 2 alternative to ZMP Preview Control.

Implements the formulation from:
  de Viragh et al., "Trajectory Optimization for Wheeled-Legged Quadrupedal Robots
  using Linearized ZMP Constraints", IEEE RA-L 2019.

This module uses OSQP with the **non-condensed (sparse) joint 2D formulation**:

    Decision variables:
        z = [x_x[0..T-1], u_x[0..T-1], x_y[0..T-1], u_y[0..T-1]] in R^(8T)
        where x_axis[k] in R^3 (pos, vel, acc) and u_axis[k] in R (jerk)

    min   Q_e * sum_k [(C@x_x[k] - zmp_ref_x[k])^2 + (C@x_y[k] - zmp_ref_y[k])^2]
     z         + R * sum_k [u_x[k]^2 + u_y[k]^2]

    s.t.  x_x[0]   = x0_x,   x_y[0]   = x0_y          (initial state)
          x_x[k+1] = A@x_x[k] + B*u_x[k]   for all k   (x dynamics)
          x_y[k+1] = A@x_y[k] + B*u_y[k]   for all k   (y dynamics)
          A_k @ [C@x_x[k], C@x_y[k]] <= b_k for all k   (ZMP polygon, exact)

The x and y axes are decoupled in the objective and dynamics, but the support
polygon constraints couple them. Using the exact half-plane constraints (instead
of axis-aligned bounding boxes) guarantees ZMP stays inside the true polygon even
during double-support phases where the polygon is a non-rectangular hexagon.

The constraint matrix has O(T) non-zeros, so each OSQP ADMM iteration is O(T).

Formulation details
-------------------
State:   x[k] in R^3  (pos, vel, acc)
Control: u[k] in R    (jerk)
ZMP:     p[k] = C @ x[k]   (LIPM: p = pos - h/g*acc)

Performance: ~0.5-3 s for T ~= 1000-4000, scaling linearly with T.
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
# Joint 2D sparse OSQP solver
# ---------------------------------------------------------------------------


def _solve_2d_qp_sparse(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    x0_x: np.ndarray,
    x0_y: np.ndarray,
    zmp_ref_x: np.ndarray,
    zmp_ref_y: np.ndarray,
    poly_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    schedule: ContactSchedule,
    Q_e: float,
    R_jerk: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Solve the joint 2D trajectory QP using OSQP with exact polygon constraints.

    Variables: z = [x_x_states(3T), u_x(T), x_y_states(3T), u_y(T)] in R^(8T)

    The x and y axes are coupled only through the support polygon constraints,
    which are expressed as exact half-planes from the convex hull of each
    support polygon. This prevents ZMP violations that arise when using
    axis-aligned bounding boxes during non-rectangular double-support phases.

    Returns (px, vx, ax, zmp_x, py, vy, ay, zmp_y, converged).
    """
    T = len(zmp_ref_x)
    nx = 3
    # Variable block offsets
    ox = 0  # x-axis states: z[ox + 3k : ox + 3(k+1)]
    ou = 3 * T  # x-axis controls: z[ou + k]
    oy = 4 * T  # y-axis states: z[oy + 3k : oy + 3(k+1)]
    ov = 7 * T  # y-axis controls: z[ov + k]
    n_vars = 8 * T

    # --- Objective: OSQP form  min (1/2) z'Pz + q'z ---
    CTC = np.outer(C, C)
    H_block = 2.0 * Q_e * CTC
    H_states = sp.block_diag([H_block] * T, format="csc")
    H_ctrl = (2.0 * R_jerk) * sp.eye(T, format="csc")
    P_osqp = sp.block_diag([H_states, H_ctrl, H_states, H_ctrl], format="csc")

    q = np.zeros(n_vars)
    for k in range(T):
        q[ox + k * nx : ox + (k + 1) * nx] = -2.0 * Q_e * zmp_ref_x[k] * C
        q[oy + k * nx : oy + (k + 1) * nx] = -2.0 * Q_e * zmp_ref_y[k] * C

    # --- Constraint matrix ---
    # Row layout:
    #   x initial state:  nx rows
    #   x dynamics:       nx*(T-1) rows
    #   y initial state:  nx rows
    #   y dynamics:       nx*(T-1) rows
    #   polygon:          sum_k m_k rows  (exact half-plane constraints)
    n_eq_x_init = nx
    n_eq_x_dyn = nx * (T - 1)
    n_eq_y_init = nx
    n_eq_y_dyn = nx * (T - 1)

    # Count total polygon rows
    poly_rows_per_k = [len(poly_cache[(int(schedule.phase[k]), schedule.kind[k])][1]) for k in range(T)]
    n_poly = sum(poly_rows_per_k)

    n_constraints = n_eq_x_init + n_eq_x_dyn + n_eq_y_init + n_eq_y_dyn + n_poly

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def _add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # x initial state
    for i in range(nx):
        _add(i, ox + i, 1.0)

    # x dynamics: x_x[k+1] - A@x_x[k] - B*u_x[k] = 0
    r0x = n_eq_x_init
    for k in range(T - 1):
        rb = r0x + k * nx
        for i in range(nx):
            _add(rb + i, ox + (k + 1) * nx + i, 1.0)
        for i in range(nx):
            for j in range(nx):
                _add(rb + i, ox + k * nx + j, -A[i, j])
        for i in range(nx):
            _add(rb + i, ou + k, -float(B[i]))

    # y initial state
    r_y_init = n_eq_x_init + n_eq_x_dyn
    for i in range(nx):
        _add(r_y_init + i, oy + i, 1.0)

    # y dynamics: x_y[k+1] - A@x_y[k] - B*u_y[k] = 0
    r0y = r_y_init + n_eq_y_init
    for k in range(T - 1):
        rb = r0y + k * nx
        for i in range(nx):
            _add(rb + i, oy + (k + 1) * nx + i, 1.0)
        for i in range(nx):
            for j in range(nx):
                _add(rb + i, oy + k * nx + j, -A[i, j])
        for i in range(nx):
            _add(rb + i, ov + k, -float(B[i]))

    # Polygon constraints: A_k @ [C@x_x[k], C@x_y[k]] <= b_k
    # Row r for half-plane (a_x, a_y) at timestep k:
    #   a_x * (C @ x_x[k]) + a_y * (C @ x_y[k]) <= b
    #   => sum_j (a_x*C[j]) * x_x[k,j] + sum_j (a_y*C[j]) * x_y[k,j] <= b
    r_poly = r_y_init + n_eq_y_init + n_eq_y_dyn
    cur_row = r_poly
    for k in range(T):
        key = (int(schedule.phase[k]), schedule.kind[k])
        Ak, bk = poly_cache[key]  # Ak: (m_k, 2), bk: (m_k,)
        for r in range(len(bk)):
            ax, ay = float(Ak[r, 0]), float(Ak[r, 1])
            for j in range(nx):
                cj = float(C[j])
                if cj != 0.0:
                    if ax != 0.0:
                        _add(cur_row + r, ox + k * nx + j, ax * cj)
                    if ay != 0.0:
                        _add(cur_row + r, oy + k * nx + j, ay * cj)
        cur_row += len(bk)

    A_osqp = sp.csc_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars))

    # --- Bounds ---
    _eps = 1e-4
    l_vec = np.full(n_constraints, -np.inf)
    u_vec = np.empty(n_constraints)

    l_vec[:nx] = x0_x
    u_vec[:nx] = x0_x

    l_vec[n_eq_x_init : n_eq_x_init + n_eq_x_dyn] = 0.0
    u_vec[n_eq_x_init : n_eq_x_init + n_eq_x_dyn] = 0.0

    l_vec[r_y_init : r_y_init + nx] = x0_y
    u_vec[r_y_init : r_y_init + nx] = x0_y

    l_vec[r0y : r0y + n_eq_y_dyn] = 0.0
    u_vec[r0y : r0y + n_eq_y_dyn] = 0.0

    # Polygon upper bounds: b_k - eps (one-sided: A@p <= b - eps)
    poly_b = np.concatenate([poly_cache[(int(schedule.phase[k]), schedule.kind[k])][1] for k in range(T)])
    u_vec[r_poly:] = poly_b - _eps

    # --- Solve ---
    solver = osqp.OSQP()
    solver.setup(
        P_osqp,
        q,
        A_osqp,
        l_vec,
        u_vec,
        verbose=False,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=10000,
    )
    result = solver.solve()

    states_x = result.x[ox:ou].reshape(T, nx)
    states_y = result.x[oy:ov].reshape(T, nx)

    px, vx, ax_arr = states_x[:, 0], states_x[:, 1], states_x[:, 2]
    py, vy, ay_arr = states_y[:, 0], states_y[:, 1], states_y[:, 2]
    zx = states_x @ C
    zy = states_y @ C

    converged = result.info.status in ("solved", "solved_inaccurate")
    return px, vx, ax_arr, zx, py, vy, ay_arr, zy, converged


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
    """Compute a CoM trajectory via a sparse QP with exact ZMP polygon constraints.

    Guarantees ZMP stays within the support polygon at every timestep
    (up to numerical tolerance). Uses exact half-plane polygon constraints
    (not axis-aligned bounding boxes) so the ZMP guarantee holds for all
    polygon shapes, including non-rectangular double-support phases.

    Uses OSQP with the non-condensed (sparse) joint 2D formulation: x and y axes
    share a single QP coupled through the polygon constraints. The constraint
    matrix has O(T) non-zeros, giving O(T) cost per OSQP ADMM iteration.

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
    # of controls. Foot 0's centre guarantees ZMP[0] is inside the first polygon.
    x0_x = np.array([footsteps[0].x, 0.0, 0.0])
    x0_y = np.array([footsteps[0].y, 0.0, 0.0])

    print("  [QP] Building polygon cache ...")
    poly_cache = precompute_polygons(schedule, footsteps, foot_length, foot_width)

    print(f"  [QP] Solving joint 2D QP ({T} timesteps, sparse OSQP) ...")
    px, vx, ax_arr, zx, py, vy, ay_arr, zy, ok = _solve_2d_qp_sparse(
        A,
        B,
        C,
        x0_x,
        x0_y,
        schedule.zmp_x,
        schedule.zmp_y,
        poly_cache,
        schedule,
        Q_e,
        R_jerk,
    )
    if not ok:
        warnings.warn("Trajectory optimizer did not converge.", stacklevel=2)

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
