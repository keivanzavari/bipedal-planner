"""
Stage 4 — MuJoCo Rigid-Body Simulation.

Usage:
    python stage4_main.py [world_name] [--planner astar|theta_star|rrt]
                          [--controller lqr|mpc]

Runs Stage 1 (footstep planning) → Stage 2 (ZMP preview control) →
Stage 4 (MuJoCo closed-loop simulation) and visualises the result in Rerun.

**Scope**: The MuJoCo simulation covers the sagittal (x, z) plane only.
The lateral (y) axis uses LIPM-analytical integration — it is not physically
simulated. Obstacles from Stage 1 are not added to the MuJoCo scene; the
simulation runs on flat ground.

Requires: uv sync --group mujoco
"""

import time

import numpy as np

from stage1.footstep import plan_footsteps
from stage1.planners import PLANNERS, get_planner
from stage1.world import WORLDS, World
from stage2.contact_schedule import build_contact_schedule
from stage2.lipm import LIPMParams
from stage2.preview_controller import compute_gains, run_preview_control
from stage3.controllers import CONTROLLERS, get_controller
from stage4.mujoco_sim import run_mujoco_simulation

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
    traj = run_preview_control(schedule, footsteps, gains)
    print(f"  Done in {(time.perf_counter() - t0) * 1000:.1f} ms")

    # ------------------------------------------------------------------
    # Stage 4 (MuJoCo)
    # ------------------------------------------------------------------
    print(f"[Stage 4] Running MuJoCo simulation (controller={controller_name})...")
    t0 = time.perf_counter()
    if controller_name == "mpc":
        controller = get_controller(
            controller_name,
            footsteps=footsteps,
            foot_length=FOOT_LENGTH,
            foot_width=FOOT_WIDTH,
        )
    else:
        controller = get_controller(controller_name)

    result = run_mujoco_simulation(
        traj,
        schedule,
        footsteps,
        LIPM_PARAMS,
        controller,
        foot_length=FOOT_LENGTH,
        foot_width=FOOT_WIDTH,
    )
    print(f"  Done in {(time.perf_counter() - t0) * 1000:.1f} ms")

    max_err = max(float(np.abs(result.err_x).max()), float(np.abs(result.err_y).max()))
    print(f"  Max position error: {max_err * 100:.2f} cm")
    height_drift = float(np.abs(result.com_height_actual - LIPM_PARAMS.h).max())
    print(f"  Max CoM height drift: {height_drift * 100:.2f} cm")

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    print("\nRendering in Rerun...")
    from viz.stage4_viz import visualize_stage4

    visualize_stage4(
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

    parser = argparse.ArgumentParser(description="Stage 4 — MuJoCo simulation")
    parser.add_argument("world", nargs="?", default="demo", choices=list(WORLDS))
    parser.add_argument("--planner", default="astar", choices=list(PLANNERS))
    parser.add_argument("--controller", default="lqr", choices=list(CONTROLLERS))
    args = parser.parse_args()

    world, start, goal = WORLDS[args.world]()

    run(
        world,
        start,
        goal,
        planner_name=args.planner,
        controller_name=args.controller,
    )
