"""
Bipedal Robot Path Planner — end-to-end pipeline.

Usage:
    python main.py
"""

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from footstep import plan_footsteps
from planner import astar, smooth_path
from stability import check_stability, stability_summary
from visualizer import plot_footsteps, plot_path, plot_stability, plot_world
from world import WORLDS, World

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

INFLATION_MARGIN = 0.25  # CoM clearance from obstacles (m)
FOOT_CLEARANCE = 0.05  # extra margin for foot placement (m)
STEP_LENGTH = 0.25  # forward stride length (m)
STEP_WIDTH = 0.10  # lateral foot offset from CoM (m)
FOOT_LENGTH = 0.16  # foot rectangle length (m)
FOOT_WIDTH = 0.08  # foot rectangle width (m)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def run(world: World, start: tuple, goal: tuple):
    print(f"World:  {world.width} x {world.height} m  ({world.rows} x {world.cols} cells @ {world.resolution} m/cell)")
    print(f"Start:  {start}")
    print(f"Goal:   {goal}")

    # Layer 1 — Global path (A*)
    print("\n[1/3] Running A*...")
    raw_path = astar(world, start, goal, inflation_margin=INFLATION_MARGIN)
    if raw_path is None:
        print("  No path found. Check that start/goal are not inside obstacles.")
        return
    smooth = smooth_path(raw_path, world, inflation_margin=INFLATION_MARGIN)
    print(f"  Raw waypoints: {len(raw_path)}  →  Smoothed: {len(smooth)}")

    # Layer 2 — Footstep planning
    print("\n[2/3] Planning footsteps...")
    footsteps = plan_footsteps(
        smooth,
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
    print(f"  Phases total:    {summary['total_phases']}")
    print(f"  Stable:          {summary['stable']}")
    print(f"  Unstable:        {summary['unstable']}")
    if summary["unstable_indices"]:
        print(f"  Unstable steps:  {summary['unstable_indices']}")

    # Visualize
    print("\nRendering...")
    inflated = world.inflated_grid(INFLATION_MARGIN)
    fig, ax = plt.subplots(figsize=(12, 9))
    plot_world(world, start=start, goal=goal, inflated_grid=inflated, ax=ax, show=False)
    plot_path(raw_path, ax, color="#bbbbbb", label="Raw A*")
    plot_path(smooth, ax, color="#3498db", label="Smoothed path", show_waypoints=True)
    plot_footsteps(footsteps, ax, foot_length=FOOT_LENGTH, foot_width=FOOT_WIDTH)
    plot_stability(phases, ax)
    ax.set_title("Bipedal Path Planner — full pipeline")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if name not in WORLDS:
        print(f"Unknown world '{name}'. Available: {list(WORLDS)}")
        sys.exit(1)
    world, start, goal = WORLDS[name]()
    run(world, start, goal)
