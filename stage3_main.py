"""
Stage 3 — Closed-Loop CoM Tracking.

Usage:
    python stage3_main.py [world_name] [--planner astar|theta_star|rrt]
                          [--controller lqr] [--noise FLOAT]

Runs Stage 1 (footstep planning) → Stage 2 (ZMP preview control) →
Stage 3 (closed-loop simulation) and visualises the result in Rerun.
"""

import time

from stage1.footstep import plan_footsteps
from stage1.planners import PLANNERS, get_planner
from stage1.world import WORLDS, World
from stage2.contact_schedule import build_contact_schedule
from stage2.lipm import LIPMParams
from stage2.preview_controller import CoMTrajectory, compute_gains, run_preview_control
from stage3.controllers import CONTROLLERS, get_controller
from stage3.simulator import run_simulation

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
T_SINGLE = 0.4
T_DOUBLE = 0.1
Q_E = 1.0
R_JERK = 1e-6
N_PREVIEW = 200


def run(
    world: World,
    start: tuple[float, float],
    goal: tuple[float, float],
    planner_name: str = "astar",
    controller_name: str = "lqr",
    noise_sigma: float = 0.001,
) -> None:
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
    # Stage 2
    # ------------------------------------------------------------------
    print("[Stage 2] Building contact schedule...")
    schedule = build_contact_schedule(
        footsteps,
        t_single=T_SINGLE,
        t_double=T_DOUBLE,
        dt=LIPM_PARAMS.dt,
    )
    print(f"  Duration: {schedule.t[-1]:.2f} s  |  Timesteps: {len(schedule.t)}")

    print("[Stage 2] Computing preview gains and trajectory...")
    t0 = time.perf_counter()
    gains = compute_gains(LIPM_PARAMS, Q_e=Q_E, R=R_JERK, N_preview=N_PREVIEW)
    traj: CoMTrajectory = run_preview_control(schedule, footsteps, gains)
    print(f"  Done in {(time.perf_counter() - t0) * 1000:.1f} ms")

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------
    print(f"[Stage 3] Running closed-loop simulation (controller={controller_name}, noise={noise_sigma})...")
    t0 = time.perf_counter()
    controller = get_controller(controller_name)
    result = run_simulation(
        traj,
        schedule,
        footsteps,
        LIPM_PARAMS,
        controller,
        noise_sigma=noise_sigma,
    )
    print(f"  Done in {(time.perf_counter() - t0) * 1000:.1f} ms")

    max_err = max(float(result.err_x.__abs__().max()), float(result.err_y.__abs__().max()))
    print(f"  Max position error: {max_err * 100:.2f} cm")

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    print("\nRendering in Rerun...")
    from viz import visualize_stage3

    visualize_stage3(
        world,
        footsteps,
        schedule,
        traj,
        result,
        foot_length=FOOT_LENGTH,
        foot_width=FOOT_WIDTH,
        inflation_margin=INFLATION_MARGIN,
        com_height=LIPM_PARAMS.h,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("world", nargs="?", default="demo", choices=list(WORLDS))
    parser.add_argument("--planner", default="astar", choices=list(PLANNERS))
    parser.add_argument("--controller", default="lqr", choices=list(CONTROLLERS))
    parser.add_argument("--noise", type=float, default=0.001, dest="noise_sigma")
    args = parser.parse_args()

    world, start, goal = WORLDS[args.world]()
    run(world, start, goal, planner_name=args.planner, controller_name=args.controller, noise_sigma=args.noise_sigma)
