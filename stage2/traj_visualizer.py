"""
Stage 2 visualisation:
  plot_trajectory_2d  — CoM + ZMP paths overlaid on the footstep map
  plot_time_series    — x(t) and y(t) subplots with support bounds
"""

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_trajectory_2d(
    traj,  # CoMTrajectory
    schedule,  # ContactSchedule
    footsteps: list,
    world,
    ax: Axes | None = None,
    show: bool = True,
) -> Axes:
    """
    Overlay CoM trajectory (solid) and ZMP trajectory (dashed) on the
    occupancy grid + footstep map.
    """
    from stage1.visualizer import plot_footsteps, plot_world

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 9))

    inflated = world.inflated_grid(0.25)
    plot_world(world, inflated_grid=inflated, ax=ax, show=False)
    plot_footsteps(footsteps, ax)

    # Downsample for plotting (every 10th point keeps it fast)
    s = 10
    ax.plot(traj.x[::s], traj.y[::s], "-", color="#e67e22", linewidth=2, zorder=8, label="CoM trajectory")
    ax.plot(traj.zmp_x[::s], traj.zmp_y[::s], "--", color="#9b59b6", linewidth=1.5, zorder=8, label="ZMP trajectory")

    # Mark start and end
    ax.plot(traj.x[0], traj.y[0], "o", color="#e67e22", markersize=8, zorder=9, markeredgecolor="black")
    ax.plot(traj.x[-1], traj.y[-1], "s", color="#e67e22", markersize=8, zorder=9, markeredgecolor="black")

    ax.set_title("Stage 2 — CoM & ZMP trajectories (2D)")
    ax.legend(loc="upper left")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_time_series(
    traj,
    schedule,
    footsteps: list,
    foot_length: float = 0.16,
    foot_width: float = 0.08,
    show: bool = True,
) -> Figure:
    """
    Time-series plot with four subplots:
      Row 1: x position (CoM + ZMP + ZMP ref) + phase shading
      Row 2: y position (CoM + ZMP + ZMP ref) + phase shading
      Row 3: velocities vx, vy
      Row 4: accelerations ax, ay
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    t = traj.t

    # --- Phase shading (alternating single/double support) ---
    def shade_phases(ax):
        in_double = False
        t_start = t[0]
        for k, kind in enumerate(schedule.kind):
            is_double = kind == "double"
            if is_double != in_double:
                if in_double:
                    ax.axvspan(
                        t_start, t[k], alpha=0.12, color="#3498db", label="Double support" if t_start == t[0] else ""
                    )
                in_double = is_double
                t_start = t[k]
        if in_double:
            ax.axvspan(t_start, t[-1], alpha=0.12, color="#3498db")

    # --- X position ---
    ax = axes[0]
    ax.plot(t, traj.x, color="#e67e22", linewidth=1.5, label="CoM x")
    ax.plot(t, traj.zmp_x, color="#9b59b6", linewidth=1, linestyle="--", label="ZMP x")
    ax.plot(t, schedule.zmp_x, color="#95a5a6", linewidth=0.8, linestyle=":", label="ZMP ref x")
    shade_phases(ax)
    ax.set_ylabel("x (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Y position ---
    ax = axes[1]
    ax.plot(t, traj.y, color="#27ae60", linewidth=1.5, label="CoM y")
    ax.plot(t, traj.zmp_y, color="#e74c3c", linewidth=1, linestyle="--", label="ZMP y")
    ax.plot(t, schedule.zmp_y, color="#95a5a6", linewidth=0.8, linestyle=":", label="ZMP ref y")
    shade_phases(ax)
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Velocities ---
    ax = axes[2]
    ax.plot(t, traj.vx, color="#e67e22", linewidth=1.2, label="vx")
    ax.plot(t, traj.vy, color="#27ae60", linewidth=1.2, label="vy")
    shade_phases(ax)
    ax.set_ylabel("velocity (m/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Accelerations ---
    ax = axes[3]
    ax.plot(t, traj.ax, color="#e67e22", linewidth=1.2, label="ax")
    ax.plot(t, traj.ay, color="#27ae60", linewidth=1.2, label="ay")
    shade_phases(ax)
    ax.set_ylabel("acceleration (m/s²)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Legend for phase shading
    axes[0].set_title("Stage 2 — CoM & ZMP time series  (shaded = double support)")
    ds_patch = mpatches.Patch(color="#3498db", alpha=0.3, label="Double support")
    axes[0].legend(
        handles=axes[0].get_legend_handles_labels()[0] + [ds_patch],
        loc="upper right",
        fontsize=8,
    )

    if show:
        plt.tight_layout()
        plt.show()

    return fig
