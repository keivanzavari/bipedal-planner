"""
Controller comparison script.

Runs Stage 1 + Stage 2 once on a chosen world, then simulates each requested
controller on the shared reference trajectory and plots the results side-by-side.

Usage:
    python compare_controllers.py [world] [--planner NAME]
                                  [--controllers lqr mpc ...]
                                  [--noise FLOAT] [--seed INT]

Examples:
    python compare_controllers.py demo
    python compare_controllers.py demo --controllers lqr mpc --noise 0.005
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from stage1.footstep import plan_footsteps
from stage1.planners import PLANNERS, get_planner
from stage1.world import WORLDS, SlipperyZone
from stage2.contact_schedule import build_contact_schedule
from stage2.lipm import LIPMParams
from stage2.preview_controller import compute_gains, run_preview_control
from stage3.controllers import CONTROLLERS, get_controller
from stage3.controllers.mpc import MPCController
from stage3.simulator import TrackingResult, run_simulation

# ------------------------------------------------------------------
# Shared pipeline parameters (match stage3_main.py)
# ------------------------------------------------------------------
INFLATION_MARGIN = 0.25
FOOT_CLEARANCE = 0.05
STEP_LENGTH = 0.25
STEP_WIDTH = 0.10
FOOT_LENGTH = 0.16
FOOT_WIDTH = 0.08

LIPM_PARAMS = LIPMParams(h=0.80, g=9.81, dt=0.005)
T_SINGLE = 0.4
T_DOUBLE = 0.1
Q_E = 1.0
R_JERK = 1e-6
N_PREVIEW = 200

# One colour per controller (cycles if more than 8 are compared)
_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#34495e"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_controller(name: str, footsteps, foot_length: float, foot_width: float,
                       slippery_zones=None):
    if name == "mpc":
        return get_controller(name, footsteps=footsteps, foot_length=foot_length,
                              foot_width=foot_width, slippery_zones=slippery_zones)
    return get_controller(name)


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


# ------------------------------------------------------------------
# Main comparison routine
# ------------------------------------------------------------------

def compare(
    world,
    start,
    goal,
    planner_name: str = "astar",
    controller_names: list[str] | None = None,
    noise_sigma: float = 0.001,
    rng_seed: int = 0,
    slippery_zones=None,
) -> None:
    if controller_names is None:
        controller_names = list(CONTROLLERS)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    print(f"[Stage 1] Running {planner_name}...")
    planner = get_planner(planner_name, inflation_margin=INFLATION_MARGIN)
    path = planner.plan(world, start, goal)
    if path is None:
        print("  No path found — aborting.")
        return
    footsteps = plan_footsteps(
        path, world,
        step_length=STEP_LENGTH, step_width=STEP_WIDTH,
        foot_length=FOOT_LENGTH, foot_width=FOOT_WIDTH,
        foot_clearance=FOOT_CLEARANCE,
    )
    print(f"  Waypoints: {len(path)}  |  Footsteps: {len(footsteps)}")

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    print("[Stage 2] Building schedule and computing reference trajectory...")
    schedule = build_contact_schedule(
        footsteps, t_single=T_SINGLE, t_double=T_DOUBLE, dt=LIPM_PARAMS.dt,
    )
    gains = compute_gains(LIPM_PARAMS, Q_e=Q_E, R=R_JERK, N_preview=N_PREVIEW)
    traj = run_preview_control(schedule, footsteps, gains)
    print(f"  Duration: {schedule.t[-1]:.2f} s  |  Timesteps: {len(schedule.t)}")

    # ------------------------------------------------------------------
    # Stage 3 — run each controller
    # ------------------------------------------------------------------
    results: dict[str, TrackingResult] = {}
    timings: dict[str, float] = {}

    for name in controller_names:
        print(f"[Stage 3] Simulating '{name}'...")
        ctrl = _build_controller(name, footsteps, FOOT_LENGTH, FOOT_WIDTH, slippery_zones)
        t0 = time.perf_counter()
        result = run_simulation(
            traj, schedule, footsteps, LIPM_PARAMS, ctrl,
            noise_sigma=noise_sigma, rng_seed=rng_seed,
            slippery_zones=slippery_zones,
            foot_length=FOOT_LENGTH, foot_width=FOOT_WIDTH,
        )
        timings[name] = time.perf_counter() - t0
        results[name] = result

    # ------------------------------------------------------------------
    # Stats table
    # ------------------------------------------------------------------
    print()
    header = (
        f"{'Controller':<12}  {'Max |err| (cm)':>16}  {'RMS err (cm)':>13}"
        f"  {'ZMP viol-x':>11}  {'ZMP viol-y':>11}  {'Time (ms)':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        max_e = max(float(np.abs(res.err_x).max()), float(np.abs(res.err_y).max())) * 100
        rms_e = max(_rms(res.err_x), _rms(res.err_y)) * 100
        T = len(res.zmp_x)
        viol_x = int(np.sum((res.zmp_x < res.zmp_lb_x) | (res.zmp_x > res.zmp_ub_x)))
        viol_y = int(np.sum((res.zmp_y < res.zmp_lb_y) | (res.zmp_y > res.zmp_ub_y)))
        ms = timings[name] * 1000
        print(
            f"{name:<12}  {max_e:>16.2f}  {rms_e:>13.4f}"
            f"  {viol_x:>5}/{T:<5}  {viol_y:>5}/{T:<5}  {ms:>10.1f}"
        )
    print()

    # ------------------------------------------------------------------
    # Figure 1 — time-series comparison
    # ------------------------------------------------------------------
    t = traj.t
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Controller comparison — world={world.__class__.__name__}  "
        f"noise={noise_sigma}  seed={rng_seed}",
        fontsize=12,
    )

    ax_px, ax_py = axes[0]
    ax_ex, ax_ey = axes[1]
    ax_ux, ax_uy = axes[2]

    # Reference position
    ax_px.plot(t, traj.x, "k--", lw=1.2, label="reference", alpha=0.6)
    ax_py.plot(t, traj.y, "k--", lw=1.2, label="reference", alpha=0.6)

    for (name, res), color in zip(results.items(), _COLORS):
        ax_px.plot(t, res.x,    color=color, lw=1.0, label=name)
        ax_py.plot(t, res.y,    color=color, lw=1.0, label=name)
        ax_ex.plot(t, res.err_x * 100, color=color, lw=1.0, label=name)
        ax_ey.plot(t, res.err_y * 100, color=color, lw=1.0, label=name)
        ax_ux.plot(t, res.u_x,  color=color, lw=0.8, label=name)
        ax_uy.plot(t, res.u_y,  color=color, lw=0.8, label=name)

    ax_px.set_ylabel("CoM x (m)")
    ax_py.set_ylabel("CoM y (m)")
    ax_ex.set_ylabel("Error x (cm)")
    ax_ey.set_ylabel("Error y (cm)")
    ax_ux.set_ylabel("Jerk x (m/s³)")
    ax_uy.set_ylabel("Jerk y (m/s³)")
    ax_ux.set_xlabel("Time (s)")
    ax_uy.set_xlabel("Time (s)")

    ax_px.axhline(0, color="none")  # force y-axis to show zero area if needed
    ax_ex.axhline(0, color="gray", lw=0.5, ls=":")
    ax_ey.axhline(0, color="gray", lw=0.5, ls=":")

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2 — 2D spatial overview
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.set_title("2D spatial overview")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    # World boundary and obstacles
    W, H = world.width, world.height
    boundary_x = [0, W, W, 0, 0]
    boundary_y = [0, 0, H, H, 0]
    ax2.plot(boundary_x, boundary_y, "k-", lw=1.5)
    for obs in world.obstacles:
        rect = plt.Rectangle(
            (obs.x, obs.y), obs.w, obs.h,
            linewidth=1, edgecolor="#555", facecolor="#bbb", alpha=0.7,
        )
        ax2.add_patch(rect)

    # Slippery zone overlay
    if slippery_zones:
        for zone in slippery_zones:
            ax2.add_patch(plt.Rectangle(
                (zone.x, zone.y), zone.w, zone.h,
                linewidth=1.5, edgecolor="#64b4ff", facecolor="#b4dcff",
                alpha=0.45, zorder=2,
            ))
            ax2.text(
                zone.x + zone.w / 2, zone.y + zone.h / 2,
                f"μ={zone.friction_scale:.1f}",
                ha="center", va="center", fontsize=8, color="#0050a0",
            )

    # Reference CoM path
    ax2.plot(traj.x, traj.y, "k--", lw=1.5, label="reference", alpha=0.6, zorder=3)

    # Actual paths
    for (name, res), color in zip(results.items(), _COLORS):
        ax2.plot(res.x, res.y, color=color, lw=1.2, label=name, zorder=4)

    # Footstep markers
    for fs in footsteps:
        marker = "^" if fs.side == "L" else "v"
        fc = "#3498db" if fs.side == "L" else "#e74c3c"
        ax2.plot(fs.x, fs.y, marker=marker, color=fc, ms=5, zorder=5)

    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Stage 3 controllers on a shared trajectory.")
    parser.add_argument("world", nargs="?", default="demo", choices=list(WORLDS))
    parser.add_argument("--planner", default="astar", choices=list(PLANNERS))
    parser.add_argument(
        "--controllers", nargs="+", default=list(CONTROLLERS),
        choices=list(CONTROLLERS), metavar="CTRL",
        help=f"Controllers to compare (default: all). Available: {list(CONTROLLERS)}",
    )
    parser.add_argument("--noise", type=float, default=0.001, dest="noise_sigma")
    parser.add_argument("--seed", type=int, default=0, dest="rng_seed")
    parser.add_argument("--slippery", action="store_true",
                        help="Add a slippery zone across the middle third of the world")
    parser.add_argument("--friction-scale", type=float, default=0.4, dest="friction_scale")
    parser.add_argument("--zone", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                        help="Custom slippery zone geometry")
    args = parser.parse_args()

    world, start, goal = WORLDS[args.world]()

    slippery_zones = None
    if args.slippery:
        if args.zone:
            x, y, w, h = args.zone
        else:
            x = world.width / 3
            y = 0.0
            w = world.width / 3
            h = world.height
        slippery_zones = [SlipperyZone(x=x, y=y, w=w, h=h, friction_scale=args.friction_scale)]

    compare(
        world, start, goal,
        planner_name=args.planner,
        controller_names=args.controllers,
        noise_sigma=args.noise_sigma,
        rng_seed=args.rng_seed,
        slippery_zones=slippery_zones,
    )
