"""Stage 3 closed-loop simulator.

Runs a discrete-time LIPM simulation under Gaussian state perturbations,
using a swappable Controller to compute jerk inputs at each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage1.footstep import Footstep
from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams, lipm_matrices
from stage2.preview_controller import CoMTrajectory
from stage3.controllers.base import Controller


@dataclass
class TrackingResult:
    t: np.ndarray        # (T,)
    x: np.ndarray        # (T,)  actual CoM x
    y: np.ndarray        # (T,)  actual CoM y
    vx: np.ndarray       # (T,)
    vy: np.ndarray       # (T,)
    ref_x: np.ndarray    # (T,)  Stage 2 reference CoM x
    ref_y: np.ndarray    # (T,)  Stage 2 reference CoM y
    err_x: np.ndarray    # (T,)  position error x  (actual - ref)
    err_y: np.ndarray    # (T,)  position error y
    u_x: np.ndarray      # (T,)  applied jerk x
    u_y: np.ndarray      # (T,)  applied jerk y
    grf_left: np.ndarray   # (T, 3)  [Fx, Fy, Fz] — zeros for LQR
    grf_right: np.ndarray  # (T, 3)


def run_simulation(
    traj: CoMTrajectory,
    schedule: ContactSchedule,
    footsteps: list[Footstep],
    params: LIPMParams,
    controller: Controller,
    noise_sigma: float = 0.001,
    rng_seed: int = 0,
) -> TrackingResult:
    """Simulate closed-loop CoM tracking with Gaussian state perturbations.

    At each timestep k:
      1. Perturb state with Gaussian noise (if noise_sigma > 0).
      2. Query controller for (jerk_x, jerk_y).
      3. Integrate state via LIPM dynamics.
      4. Record state, error, and control.

    Parameters
    ----------
    traj        : reference CoM trajectory from Stage 2
    schedule    : contact schedule (same object used to produce traj)
    footsteps   : ordered Footstep list from Stage 1
    params      : LIPM model parameters
    controller  : Controller instance (already constructed, not yet reset)
    noise_sigma : std-dev of zero-mean Gaussian added to each state component
    rng_seed    : seed for reproducibility

    Returns
    -------
    TrackingResult with all arrays of length T = len(traj.t)
    """
    A, B, _ = lipm_matrices(params)
    T = len(traj.t)
    rng = np.random.default_rng(rng_seed)

    controller.reset(traj, schedule, params)

    # Initialise actual state at the reference initial state
    state_x = np.array([traj.x[0], traj.vx[0], traj.ax[0]])
    state_y = np.array([traj.y[0], traj.vy[0], traj.ay[0]])

    out_x  = np.empty(T)
    out_y  = np.empty(T)
    out_vx = np.empty(T)
    out_vy = np.empty(T)
    out_ux = np.empty(T)
    out_uy = np.empty(T)
    grf_left  = np.zeros((T, 3))
    grf_right = np.zeros((T, 3))

    for k in range(T):
        if noise_sigma > 0.0:
            state_x = state_x + rng.normal(0.0, noise_sigma, 3)
            state_y = state_y + rng.normal(0.0, noise_sigma, 3)

        ux, uy = controller.step(k, state_x, state_y)

        out_x[k]  = state_x[0]
        out_y[k]  = state_y[0]
        out_vx[k] = state_x[1]
        out_vy[k] = state_y[1]
        out_ux[k] = ux
        out_uy[k] = uy

        state_x = A @ state_x + B * ux
        state_y = A @ state_y + B * uy

    return TrackingResult(
        t=traj.t,
        x=out_x,
        y=out_y,
        vx=out_vx,
        vy=out_vy,
        ref_x=traj.x,
        ref_y=traj.y,
        err_x=out_x - traj.x,
        err_y=out_y - traj.y,
        u_x=out_ux,
        u_y=out_uy,
        grf_left=grf_left,
        grf_right=grf_right,
    )
