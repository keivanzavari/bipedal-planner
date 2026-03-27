"""Stage 3 closed-loop simulator.

Runs a discrete-time LIPM simulation under Gaussian state perturbations,
using a swappable Controller to compute jerk inputs at each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage1.footstep import Footstep, _foot_corners
from stage1.world import SlipperyZone
from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams, lipm_matrices
from stage2.preview_controller import CoMTrajectory
from stage2.traj_optimizer import _compute_zmp_bounds
from stage3.controllers.base import Controller


@dataclass
class TrackingResult:
    t: np.ndarray  # (T,)
    x: np.ndarray  # (T,)  actual CoM x
    y: np.ndarray  # (T,)  actual CoM y
    vx: np.ndarray  # (T,)
    vy: np.ndarray  # (T,)
    ref_x: np.ndarray  # (T,)  Stage 2 reference CoM x
    ref_y: np.ndarray  # (T,)  Stage 2 reference CoM y
    err_x: np.ndarray  # (T,)  position error x  (actual - ref)
    err_y: np.ndarray  # (T,)  position error y
    u_x: np.ndarray  # (T,)  applied jerk x
    u_y: np.ndarray  # (T,)  applied jerk y
    grf_left: np.ndarray  # (T, 3)  [Fx, Fy, Fz] — zeros for LQR/MPC
    grf_right: np.ndarray  # (T, 3)
    zmp_x: np.ndarray  # (T,)  actual ZMP x  (= pos_x - h/g * acc_x)
    zmp_y: np.ndarray  # (T,)  actual ZMP y
    zmp_lb_x: np.ndarray  # (T,)  support-polygon lower bound x (friction-adjusted)
    zmp_ub_x: np.ndarray  # (T,)  support-polygon upper bound x
    zmp_lb_y: np.ndarray  # (T,)  support-polygon lower bound y
    zmp_ub_y: np.ndarray  # (T,)  support-polygon upper bound y
    friction: np.ndarray  # (T,)  effective friction coefficient at each step


def _friction_at(zones: list[SlipperyZone] | None, px: float, py: float) -> float:
    """Return minimum friction scale from all zones containing (px, py)."""
    if not zones:
        return 1.0
    scale = 1.0
    for z in zones:
        if z.contains(px, py):
            scale = min(scale, z.friction_scale)
    return scale


def _slippery_zmp_bounds(
    schedule: ContactSchedule,
    footsteps: list[Footstep],
    foot_length: float,
    foot_width: float,
    zones: list[SlipperyZone] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-timestep ZMP bounds, shrinking foot polygon in slippery zones."""
    lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(schedule, footsteps, foot_length, foot_width)
    if not zones:
        return lb_x, ub_x, lb_y, ub_y

    T = len(schedule.t)
    for k in range(T):
        phase_idx = int(schedule.phase[k])
        fs = footsteps[phase_idx]
        fscale = _friction_at(zones, fs.x, fs.y)

        if schedule.kind[k] == "double" and phase_idx > 0:
            prev = footsteps[phase_idx - 1]
            fscale = min(fscale, _friction_at(zones, prev.x, prev.y))

        if fscale < 1.0:
            fl_s = foot_length * fscale
            fw_s = foot_width * fscale
            corners = _foot_corners(fs.x, fs.y, fs.theta, fl_s, fw_s)
            if schedule.kind[k] == "double" and phase_idx > 0:
                prev = footsteps[phase_idx - 1]
                fscale_p = _friction_at(zones, prev.x, prev.y)
                corners_p = _foot_corners(
                    prev.x,
                    prev.y,
                    prev.theta,
                    foot_length * fscale_p,
                    foot_width * fscale_p,
                )
                corners = np.vstack([corners, corners_p])
            lb_x[k] = corners[:, 0].min()
            ub_x[k] = corners[:, 0].max()
            lb_y[k] = corners[:, 1].min()
            ub_y[k] = corners[:, 1].max()

    return lb_x, ub_x, lb_y, ub_y


def run_simulation(
    traj: CoMTrajectory,
    schedule: ContactSchedule,
    footsteps: list[Footstep],
    params: LIPMParams,
    controller: Controller,
    noise_sigma: float = 0.001,
    rng_seed: int = 0,
    slippery_zones: list[SlipperyZone] | None = None,
    foot_length: float = 0.16,
    foot_width: float = 0.08,
) -> TrackingResult:
    """Simulate closed-loop CoM tracking with Gaussian state perturbations.

    At each timestep k:
      1. Apply landing impulse if foot just landed on a slippery surface.
      2. Perturb state with Gaussian noise (if noise_sigma > 0).
      3. Query controller for (jerk_x, jerk_y).
      4. Integrate state via LIPM dynamics.
      5. Record state, error, ZMP, and control.

    Parameters
    ----------
    traj            : reference CoM trajectory from Stage 2
    schedule        : contact schedule (same object used to produce traj)
    footsteps       : ordered Footstep list from Stage 1
    params          : LIPM model parameters
    controller      : Controller instance (already constructed, not yet reset)
    noise_sigma     : std-dev of zero-mean Gaussian added to each state component
    rng_seed        : seed for reproducibility
    slippery_zones  : optional list of slippery floor regions
    foot_length     : foot rectangle length — used for support polygon bounds
    foot_width      : foot rectangle width
    """
    A, B, C = lipm_matrices(params)
    T = len(traj.t)
    rng = np.random.default_rng(rng_seed)

    # Precompute per-step friction and support-polygon bounds
    lb_x, ub_x, lb_y, ub_y = _slippery_zmp_bounds(schedule, footsteps, foot_length, foot_width, slippery_zones)
    friction = np.ones(T)
    for k in range(T):
        phase_idx = int(schedule.phase[k])
        fs = footsteps[phase_idx]
        friction[k] = _friction_at(slippery_zones, fs.x, fs.y)

    # Precompute per-step landing impulse sigma (fires once when foot lands in a zone)
    landing_sigma = np.zeros(T)
    if slippery_zones:
        for k in range(1, T):
            is_new_single = schedule.kind[k] == "single" and (
                schedule.kind[k - 1] != "single" or schedule.phase[k] != schedule.phase[k - 1]
            )
            if is_new_single:
                phase_idx = int(schedule.phase[k])
                fs = footsteps[phase_idx]
                fscale = _friction_at(slippery_zones, fs.x, fs.y)
                if fscale < 1.0:
                    landing_sigma[k] = noise_sigma * 10.0 * (1.0 - fscale)

    controller.reset(traj, schedule, params)

    state_x = np.array([traj.x[0], traj.vx[0], traj.ax[0]])
    state_y = np.array([traj.y[0], traj.vy[0], traj.ay[0]])

    out_x = np.empty(T)
    out_y = np.empty(T)
    out_vx = np.empty(T)
    out_vy = np.empty(T)
    out_ux = np.empty(T)
    out_uy = np.empty(T)
    out_zmp_x = np.empty(T)
    out_zmp_y = np.empty(T)
    grf_left = np.zeros((T, 3))
    grf_right = np.zeros((T, 3))

    for k in range(T):
        # Landing impulse (pos + vel, applied before regular noise)
        if landing_sigma[k] > 0.0:
            state_x[:2] += rng.normal(0.0, landing_sigma[k], 2)
            state_y[:2] += rng.normal(0.0, landing_sigma[k], 2)

        if noise_sigma > 0.0:
            state_x = state_x + rng.normal(0.0, noise_sigma, 3)
            state_y = state_y + rng.normal(0.0, noise_sigma, 3)

        ux, uy = controller.step(k, state_x, state_y)

        out_x[k] = state_x[0]
        out_y[k] = state_y[0]
        out_vx[k] = state_x[1]
        out_vy[k] = state_y[1]
        out_ux[k] = ux
        out_uy[k] = uy
        out_zmp_x[k] = float(C @ state_x)
        out_zmp_y[k] = float(C @ state_y)

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
        zmp_x=out_zmp_x,
        zmp_y=out_zmp_y,
        zmp_lb_x=lb_x,
        zmp_ub_x=ub_x,
        zmp_lb_y=lb_y,
        zmp_ub_y=ub_y,
        friction=friction,
    )
