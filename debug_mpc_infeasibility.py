# ruff: noqa
"""Debug script: investigate why MPC QP reports primal infeasible with slippery zones."""

import numpy as np
import osqp
import scipy.linalg
import scipy.sparse as sp

from stage1.footstep import plan_footsteps
from stage1.planners import get_planner
from stage1.world import WORLDS, SlipperyZone
from stage2.contact_schedule import build_contact_schedule
from stage2.lipm import LIPMParams, lipm_matrices
from stage2.preview_controller import compute_gains, run_preview_control
from stage2.traj_optimizer import build_propagation_matrix, free_response
from stage3.simulator import _slippery_zmp_bounds

# ------------------------------------------------------------------
# Build scenario
# ------------------------------------------------------------------
PARAMS = LIPMParams(h=0.80, g=9.81, dt=0.005)
world, start, goal = WORLDS["demo"]()
planner = get_planner("astar", inflation_margin=0.25)
path = planner.plan(world, start, goal)
footsteps = plan_footsteps(
    path,
    world,
    step_length=0.25,
    step_width=0.10,
    foot_length=0.16,
    foot_width=0.08,
    foot_clearance=0.05,
)
schedule = build_contact_schedule(footsteps, t_single=0.4, t_double=0.1, dt=PARAMS.dt)
gains = compute_gains(PARAMS, Q_e=1.0, R=1e-6, N_preview=200)
traj = run_preview_control(schedule, footsteps, gains)
zone = SlipperyZone(x=world.width / 3, y=0.0, w=world.width / 3, h=world.height, friction_scale=0.4)

# ------------------------------------------------------------------
# MPC internals
# ------------------------------------------------------------------
A, B, C = lipm_matrices(PARAMS)
N = 20
Q_e = 1.0
R = 1e-6
T = len(traj.t)

ref_x = np.column_stack([traj.x, traj.vx, traj.ax])

lb_x, ub_x, lb_y, ub_y = _slippery_zmp_bounds(schedule, footsteps, 0.16, 0.08, [zone])
_eps = 0.001
elb_x = np.minimum(lb_x, traj.zmp_x) - _eps
eub_x = np.maximum(ub_x, traj.zmp_x) + _eps

P_mat = build_propagation_matrix(A, B, C, N)
Q_state = Q_e * np.outer(C, C)
P_inf = scipy.linalg.solve_discrete_are(A, B.reshape(-1, 1), Q_state, np.array([[R]]))
Gamma = np.empty((3, N))
Ak = np.eye(3)
for j in range(N - 1, -1, -1):
    Gamma[:, j] = Ak @ B
    if j > 0:
        Ak = A @ Ak
A_N = np.linalg.matrix_power(A, N)
K_terminal = Gamma.T @ P_inf @ A_N
H_dense = 2.0 * (Q_e * P_mat.T @ P_mat + R * np.eye(N) + Gamma.T @ P_inf @ Gamma)
H_sp = sp.csc_matrix(np.triu(H_dense))
A_sp = sp.csc_matrix(P_mat)

print(f"P_mat shape: {P_mat.shape}")
print(f"P_mat rank: {np.linalg.matrix_rank(P_mat)} (expected {N - 1} since first row is 0)")
print(f"P_mat first row all-zero: {np.allclose(P_mat[0, :], 0)}")
print()

# ------------------------------------------------------------------
# Advance to k=2 with noise
# ------------------------------------------------------------------
rng = np.random.default_rng(0)
state_x = np.array([traj.x[0], traj.vx[0], traj.ax[0]])
for k in range(3):
    noise = rng.normal(scale=0.005)
    state_x += np.array([noise, noise, 0.0])

K_DIAG = 2  # examine this step
e = state_x - ref_x[K_DIAG]
zfx = free_response(A, C, e, N)
q_x = 2.0 * Q_e * (P_mat.T @ zfx) + 2.0 * (K_terminal @ e)

end = min(K_DIAG + N, T)


def win(arr):
    w = arr[K_DIAG:end]
    return np.concatenate([w, np.full(N - len(w), arr[-1])]) if len(w) < N else w


lx = win(elb_x) - win(traj.zmp_x) - zfx
ux = win(eub_x) - win(traj.zmp_x) - zfx

print(f"=== Step k={K_DIAG} ===")
print(f"e = {e}")
print(f"zfx[:5] = {zfx[:5]}")
print(f"l[:5] = {lx[:5]}")
print(f"u[:5] = {ux[:5]}")
print(f"bound width (u-l)[:5] = {(ux - lx)[:5]}")
print(f"All l <= u: {np.all(lx <= ux)}")
print()

# ------------------------------------------------------------------
# Attempt 1: Fresh solve (no warm-start)
# ------------------------------------------------------------------
print("--- Fresh solver ---")
solver_fresh = osqp.OSQP()
solver_fresh.setup(H_sp, q_x, A_sp, lx, ux, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000)
res = solver_fresh.solve()
print(f"status: {res.info.status!r}, x[0]: {res.x[0] if res.x is not None else None}")

# ------------------------------------------------------------------
# Attempt 2: Solve with relaxed (unconstrained) bounds
# ------------------------------------------------------------------
print("--- Unconstrained solve ---")
solver_uncons = osqp.OSQP()
solver_uncons.setup(
    H_sp, q_x, A_sp, np.full(N, -1e9), np.full(N, 1e9), verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000
)
res_unc = solver_uncons.solve()
print(f"status: {res_unc.info.status!r}, x[0]: {res_unc.x[0]:.6f}")
print(f"Unconstrained zmp_err = P@du + zfx (first 5): {(P_mat @ res_unc.x + zfx)[:5]}")

# ------------------------------------------------------------------
# Attempt 3: Warm-start from previous solve (as in actual MPC)
# ------------------------------------------------------------------
print()
print("--- Warm-start chain (simulating actual MPC) ---")
solver_ws = osqp.OSQP()
solver_ws.setup(
    H_sp, np.zeros(N), A_sp, np.full(N, -1e9), np.full(N, 1e9), verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000
)

state_x2 = np.array([traj.x[0], traj.vx[0], traj.ax[0]])
rng2 = np.random.default_rng(0)
for k in range(min(10, T)):
    noise = rng2.normal(scale=0.005)
    state_x2 += np.array([noise, noise, 0.0])
    e_k = state_x2 - ref_x[k]
    zfx_k = free_response(A, C, e_k, N)
    q_k = 2.0 * Q_e * (P_mat.T @ zfx_k) + 2.0 * (K_terminal @ e_k)
    end_k = min(k + N, T)

    def win_k(arr):
        w = arr[k:end_k]
        return np.concatenate([w, np.full(N - len(w), arr[-1])]) if len(w) < N else w

    l_k = win_k(elb_x) - win_k(traj.zmp_x) - zfx_k
    u_k = win_k(eub_x) - win_k(traj.zmp_x) - zfx_k
    solver_ws.update(q=q_k, l=l_k, u=u_k)
    res_k = solver_ws.solve()
    status = res_k.info.status
    du0 = res_k.x[0] if (res_k.x is not None and np.isfinite(res_k.x[0])) else float("nan")
    print(f"  k={k:2d}: status={status!r:30s} du={du0:10.4f}  |e|={np.linalg.norm(e_k):.5f}")
    u_ff_k = (traj.ax[k + 1] - traj.ax[k]) / PARAMS.dt if k < T - 1 else 0.0
    du_use = du0 if np.isfinite(du0) and abs(du0) < 1e6 else 0.0
    state_x2 = A @ state_x2 + B * (u_ff_k + du_use)
