"""
Stage 2 — Trajectory Optimisation Pipeline.

Usage:
    python stage2_main.py [world_name] [--planner astar|theta_star|rrt] [--viz matplotlib|rerun]

Runs Stage 1 (footstep planning) then Stage 2 (ZMP preview control),
and shows both the 2D spatial plot and the time-series plot.
"""

import time

from stage1.footstep import plan_footsteps
from stage1.planners import PLANNERS, get_planner
from stage1.world import WORLDS
from stage2.contact_schedule import build_contact_schedule
from stage2.lipm import LIPMParams
from stage2.preview_controller import compute_gains, run_preview_control, validate_zmp
from stage2.traj_optimizer import run_trajectory_optimization

# ------------------------------------------------------------------
# Stage 1 parameters
# ------------------------------------------------------------------
INFLATION_MARGIN = 0.25
FOOT_CLEARANCE = 0.05
STEP_LENGTH = 0.25
STEP_WIDTH = 0.10
FOOT_LENGTH = 0.16
FOOT_WIDTH = 0.08

# ------------------------------------------------------------------
# Stage 2 parameters
# ------------------------------------------------------------------
LIPM_PARAMS = LIPMParams(h=0.80, g=9.81, dt=0.005)
T_SINGLE = 0.4  # single support duration (s)
T_DOUBLE = 0.1  # double support duration (s)
Q_E = 1.0  # ZMP tracking weight
R_JERK = 1e-6  # jerk smoothness weight
N_PREVIEW = 200  # preview horizon (steps = 1 s at dt=0.005)


def run(world, start, goal, planner_name: str = "astar", viz: str = "matplotlib", method: str = "preview"):
    planner = get_planner(planner_name, inflation_margin=INFLATION_MARGIN)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    print(f"[Stage 1] Running {planner_name}...")
    path = planner.plan(world, start, goal)
    if path is None:
        print("  No path found.")
        return
    footsteps = plan_footsteps(
        path,
        world,
        step_length=STEP_LENGTH,
        step_width=STEP_WIDTH,
        foot_length=FOOT_LENGTH,
        foot_width=FOOT_WIDTH,
        foot_clearance=FOOT_CLEARANCE,
    )
    print(f"  Waypoints: {len(path)}  |  Footsteps: {len(footsteps)}")

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
    # Stage 2 — compute trajectory
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    if method == "optimize":
        print("[Stage 2] Running QP trajectory optimizer (explicit polygon constraints)...")
        traj = run_trajectory_optimization(
            schedule,
            footsteps,
            LIPM_PARAMS,
            foot_length=FOOT_LENGTH,
            foot_width=FOOT_WIDTH,
            Q_e=Q_E,
            R_jerk=R_JERK,
        )
        print(f"  Trajectory optimized in {(time.perf_counter() - t0) * 1000:.1f} ms")
    else:
        print("[Stage 2] Computing preview gains (offline LQR)...")
        gains = compute_gains(LIPM_PARAMS, Q_e=Q_E, R=R_JERK, N_preview=N_PREVIEW)
        print(f"  Gains computed in {(time.perf_counter() - t0) * 1000:.1f} ms")

        print("[Stage 2] Running preview controller...")
        t0 = time.perf_counter()
        traj = run_preview_control(schedule, footsteps, gains)
        print(f"  Trajectory computed in {(time.perf_counter() - t0) * 1000:.1f} ms")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    print("[Stage 2] Validating ZMP constraints...")
    report = validate_zmp(traj, schedule, footsteps, FOOT_LENGTH, FOOT_WIDTH)
    total = report["total_steps"]
    viols = report["zmp_violations"]
    rate = report["violation_rate"] * 100
    print(f"  ZMP violations: {viols} / {total}  ({rate:.1f}%)")
    if viols == 0:
        print("  ✓ ZMP fully inside support polygon at all timesteps")
    else:
        print(f"  First violation indices: {report['first_failures']}")

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    print("\nRendering...")
    if viz == "rerun":
        from viz import visualize_stage2

        visualize_stage2(
            world,
            footsteps,
            schedule,
            traj,
            foot_length=FOOT_LENGTH,
            foot_width=FOOT_WIDTH,
            inflation_margin=INFLATION_MARGIN,
            com_height=LIPM_PARAMS.h,
        )
    else:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        from stage2.traj_visualizer import plot_time_series, plot_trajectory_2d

        fig1, ax = plt.subplots(figsize=(14, 9))
        plot_trajectory_2d(traj, schedule, footsteps, world, ax=ax, show=False)
        plt.tight_layout()

        plot_time_series(traj, schedule, footsteps, FOOT_LENGTH, FOOT_WIDTH, show=False)
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("world", nargs="?", default="demo", choices=list(WORLDS))
    parser.add_argument("--planner", default="astar", choices=list(PLANNERS))
    parser.add_argument("--viz", default="matplotlib", choices=["matplotlib", "rerun"])
    parser.add_argument("--method", default="preview", choices=["preview", "optimize"])
    args = parser.parse_args()

    world, start, goal = WORLDS[args.world]()
    run(world, start, goal, planner_name=args.planner, viz=args.viz, method=args.method)
