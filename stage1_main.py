"""
Bipedal Robot Path Planner — Stage 1 pipeline.

Usage:
    python stage1_main.py [world] [--planner astar|theta_star|rrt] [--viz matplotlib|rerun]

Examples:
    python stage1_main.py
    python stage1_main.py warehouse --planner theta_star
    python stage1_main.py corridor  --planner rrt
    python stage1_main.py demo      --viz rerun
"""

import argparse

from stage1.footstep import plan_footsteps
from stage1.planners import PLANNERS, get_planner
from stage1.stability import check_stability, stability_summary
from stage1.world import WORLDS, World

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

INFLATION_MARGIN = 0.25
FOOT_CLEARANCE = 0.05
STEP_LENGTH = 0.25
STEP_WIDTH = 0.10
FOOT_LENGTH = 0.16
FOOT_WIDTH = 0.08

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def run(world: World, start: tuple, goal: tuple, planner_name: str = "astar", viz: str = "matplotlib"):
    planner = get_planner(planner_name, inflation_margin=INFLATION_MARGIN)

    print(f"World:   {world.width} x {world.height} m  ({world.rows} x {world.cols} cells @ {world.resolution} m/cell)")
    print(f"Start:   {start}")
    print(f"Goal:    {goal}")
    print(f"Planner: {planner_name}")

    # Layer 1 — Global path
    print(f"\n[1/3] Running {planner_name}...")
    path = planner.plan(world, start, goal)
    if path is None:
        print("  No path found. Check that start/goal are not inside obstacles.")
        return
    print(f"  Waypoints: {len(path)}")

    # Layer 2 — Footstep planning
    print("\n[2/3] Planning footsteps...")
    footsteps = plan_footsteps(
        path,
        world,
        step_length=STEP_LENGTH,
        step_width=STEP_WIDTH,
        foot_length=FOOT_LENGTH,
        foot_width=FOOT_WIDTH,
        foot_clearance=FOOT_CLEARANCE,
    )
    print(f"  Footsteps generated: {len(footsteps)}")

    # Layer 3 — Stability check
    print("\n[3/3] Checking stability...")
    phases = check_stability(footsteps, foot_length=FOOT_LENGTH, foot_width=FOOT_WIDTH)
    summary = stability_summary(phases)
    print(f"  Stable: {summary['stable']} / {summary['total_phases']} phases")

    # Visualise
    print("\nRendering...")
    if viz == "rerun":
        from viz import visualize_stage1

        visualize_stage1(
            world,
            start,
            goal,
            path,
            footsteps,
            phases,
            foot_length=FOOT_LENGTH,
            foot_width=FOOT_WIDTH,
            inflation_margin=INFLATION_MARGIN,
            planner_name=planner_name,
        )
    else:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        from stage1.visualizer import plot_footsteps, plot_path, plot_stability, plot_world

        inflated = world.inflated_grid(INFLATION_MARGIN)
        fig, ax = plt.subplots(figsize=(12, 9))
        plot_world(world, start=start, goal=goal, inflated_grid=inflated, ax=ax, show=False)
        plot_path(path, ax, color="#3498db", label=planner_name, show_waypoints=True)
        plot_footsteps(footsteps, ax, foot_length=FOOT_LENGTH, foot_width=FOOT_WIDTH)
        plot_stability(phases, ax)
        ax.set_title(f"Bipedal Path Planner — {planner_name}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("world", nargs="?", default="demo", choices=list(WORLDS))
    parser.add_argument("--planner", default="astar", choices=list(PLANNERS))
    parser.add_argument("--viz", default="matplotlib", choices=["matplotlib", "rerun"])
    args = parser.parse_args()

    world, start, goal = WORLDS[args.world]()
    run(world, start, goal, planner_name=args.planner, viz=args.viz)
