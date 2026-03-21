"""
Stage 2 — Trajectory Optimisation Pipeline.

Usage:
    python stage2_main.py [world_name]

Runs Stage 1 (footstep planning) then Stage 2 (ZMP preview control),
and shows both the 2D spatial plot and the time-series plot.
"""

import sys
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from world import WORLDS
from planner import astar, smooth_path
from footstep import plan_footsteps
from lipm import LIPMParams
from contact_schedule import build_contact_schedule
from preview_controller import compute_gains, run_preview_control, validate_zmp
from traj_visualizer import plot_trajectory_2d, plot_time_series

# ------------------------------------------------------------------
# Stage 1 parameters
# ------------------------------------------------------------------
INFLATION_MARGIN = 0.25
FOOT_CLEARANCE   = 0.05
STEP_LENGTH      = 0.25
STEP_WIDTH       = 0.10
FOOT_LENGTH      = 0.16
FOOT_WIDTH       = 0.08

# ------------------------------------------------------------------
# Stage 2 parameters
# ------------------------------------------------------------------
LIPM_PARAMS = LIPMParams(h=0.80, g=9.81, dt=0.005)
T_SINGLE    = 0.4     # single support duration (s)
T_DOUBLE    = 0.1     # double support duration (s)
Q_E         = 1.0     # ZMP tracking weight
R_JERK      = 1e-6    # jerk smoothness weight
N_PREVIEW   = 200     # preview horizon (steps = 1 s at dt=0.005)


def run(world, start, goal):

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    print("[Stage 1] Running A*...")
    raw_path = astar(world, start, goal, inflation_margin=INFLATION_MARGIN)
    if raw_path is None:
        print("  No path found.")
        return
    smooth = smooth_path(raw_path, world, inflation_margin=INFLATION_MARGIN)
    footsteps = plan_footsteps(
        smooth, world,
        step_length=STEP_LENGTH, step_width=STEP_WIDTH,
        foot_length=FOOT_LENGTH, foot_width=FOOT_WIDTH,
        foot_clearance=FOOT_CLEARANCE,
    )
    print(f"  Smoothed waypoints: {len(smooth)}  |  Footsteps: {len(footsteps)}")

    # ------------------------------------------------------------------
    # Stage 2 — contact schedule
    # ------------------------------------------------------------------
    print("[Stage 2] Building contact schedule...")
    schedule = build_contact_schedule(
        footsteps,
        t_single=T_SINGLE,
        t_double=T_DOUBLE,
        dt=LIPM_PARAMS.dt,
    )
    T_total = schedule.t[-1]
    print(f"  Trajectory duration: {T_total:.2f} s  |  Timesteps: {len(schedule.t)}")

    # ------------------------------------------------------------------
    # Stage 2 — compute gains (offline, once)
    # ------------------------------------------------------------------
    print("[Stage 2] Computing preview gains (offline LQR)...")
    t0 = time.perf_counter()
    gains = compute_gains(LIPM_PARAMS, Q_e=Q_E, R=R_JERK, N_preview=N_PREVIEW)
    print(f"  Gains computed in {(time.perf_counter() - t0)*1000:.1f} ms")

    # ------------------------------------------------------------------
    # Stage 2 — run preview controller
    # ------------------------------------------------------------------
    print("[Stage 2] Running preview controller...")
    t0 = time.perf_counter()
    traj = run_preview_control(schedule, footsteps, gains)
    print(f"  Trajectory computed in {(time.perf_counter() - t0)*1000:.1f} ms")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    print("[Stage 2] Validating ZMP constraints...")
    report = validate_zmp(traj, schedule, footsteps, FOOT_LENGTH, FOOT_WIDTH)
    total  = report["total_steps"]
    viols  = report["zmp_violations"]
    rate   = report["violation_rate"] * 100
    print(f"  ZMP violations: {viols} / {total}  ({rate:.1f}%)")
    if viols == 0:
        print("  ✓ ZMP fully inside support polygon at all timesteps")
    else:
        print(f"  First violation indices: {report['first_failures']}")

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    fig1, ax = plt.subplots(figsize=(14, 9))
    plot_trajectory_2d(traj, schedule, footsteps, world, ax=ax, show=False)
    plt.tight_layout()

    fig2 = plot_time_series(traj, schedule, footsteps,
                            FOOT_LENGTH, FOOT_WIDTH, show=False)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if name not in WORLDS:
        print(f"Unknown world '{name}'. Available: {list(WORLDS)}")
        sys.exit(1)
    world, start, goal = WORLDS[name]()
    run(world, start, goal)
