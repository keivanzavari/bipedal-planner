import matplotlib
import numpy as np

from stage1.world import World

matplotlib.use("TkAgg")  # Or 'QtAgg' if you installed PyQt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_world(
    world: World,
    start: tuple[float, float] | None = None,
    goal: tuple[float, float] | None = None,
    inflated_grid: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Render the occupancy grid.

    Parameters
    ----------
    world          : World instance
    start / goal   : (x, y) world coordinates, drawn as markers if provided
    inflated_grid  : optional pre-computed inflated grid to overlay
    ax             : existing Axes to draw on; a new figure is created if None
    show           : call plt.show() at the end
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    extent = [0, world.width, 0, world.height]

    # Base occupancy grid — white=free, dark=obstacle
    cmap_base = ListedColormap(["#f5f5f5", "#333333"])
    ax.imshow(
        world.grid,
        origin="lower",
        extent=extent,
        cmap=cmap_base,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=1,
    )

    # Inflated grid overlay (semi-transparent red)
    if inflated_grid is not None:
        overlay = np.zeros((*inflated_grid.shape, 4), dtype=float)
        mask = (inflated_grid == 1) & (world.grid == 0)  # inflation only
        overlay[mask] = [1.0, 0.2, 0.2, 0.35]
        ax.imshow(
            overlay,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            zorder=2,
        )

    # Start / goal markers
    if start is not None:
        ax.plot(
            *start,
            marker="o",
            markersize=10,
            color="#2ecc71",
            zorder=5,
            label="Start",
            markeredgecolor="black",
            markeredgewidth=1,
        )
    if goal is not None:
        ax.plot(
            *goal,
            marker="*",
            markersize=14,
            color="#e74c3c",
            zorder=5,
            label="Goal",
            markeredgecolor="black",
            markeredgewidth=1,
        )

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("World — occupancy grid")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(False)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_path(
    waypoints: list[tuple[float, float]],
    ax: plt.Axes,
    color: str = "#3498db",
    label: str = "CoM path",
    zorder: int = 4,
    show_waypoints: bool = False,
):
    """Overlay a list of (x, y) waypoints as a line on an existing Axes."""
    if not waypoints:
        return
    xs, ys = zip(*waypoints, strict=True)
    ax.plot(xs, ys, "-", color=color, linewidth=2, zorder=zorder, label=label)
    if show_waypoints:
        ax.plot(xs, ys, "o", color=color, markersize=6, zorder=zorder + 1, markeredgecolor="white", markeredgewidth=1)
    ax.legend(loc="upper left")


def plot_footsteps(
    footsteps,
    ax: plt.Axes,
    foot_length: float = 0.16,
    foot_width: float = 0.08,
    zorder: int = 6,
):
    """
    Draw each footstep as a rotated rectangle.
    Left foot = blue, Right foot = red.
    """
    colors = {"L": "#3498db", "R": "#e74c3c"}

    for i, fs in enumerate(footsteps):
        color = colors[fs.side]
        c, s = np.cos(fs.theta), np.sin(fs.theta)

        # Bottom-left corner of the rectangle (in world coords)
        # Centre is (fs.x, fs.y); rotate the local (-half_l, -half_w) corner
        half_l, half_w = foot_length / 2, foot_width / 2
        corner = np.array(
            [
                fs.x - c * half_l + s * half_w,
                fs.y - s * half_l - c * half_w,
            ]
        )
        angle_deg = np.degrees(fs.theta)

        rect = patches.Rectangle(
            corner,
            foot_length,
            foot_width,
            angle=angle_deg,
            linewidth=1,
            edgecolor="black",
            facecolor=color,
            alpha=0.7,
            zorder=zorder,
        )
        ax.add_patch(rect)

        # Step number label
        ax.text(fs.x, fs.y, str(i + 1), fontsize=5, ha="center", va="center", color="white", zorder=zorder + 1)

    # Legend proxies
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#3498db", edgecolor="black", label="Left foot"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Right foot"),
    ]
    existing = ax.get_legend_handles_labels()
    ax.legend(handles=list(existing[0]) + handles, loc="upper left")


def plot_stability(
    phases,
    ax: plt.Axes,
    show_com: bool = True,
    zorder: int = 7,
):
    """
    Overlay support polygons and CoM points for each stance phase.
    Green = stable, red = unstable.
    """
    from matplotlib.patches import Polygon as MplPolygon

    for phase in phases:
        color = "#2ecc71" if phase.stable else "#e74c3c"
        alpha = 0.25 if phase.kind == "double" else 0.15

        poly_patch = MplPolygon(
            phase.support_polygon,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(poly_patch)

        if show_com:
            marker = "x" if not phase.stable else "+"
            ax.plot(*phase.com, marker=marker, markersize=5, color=color, zorder=zorder + 1, markeredgewidth=1)

    # Legend proxies
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#2ecc71", alpha=0.5, label="Stable support polygon"),
        Patch(facecolor="#e74c3c", alpha=0.5, label="Unstable support polygon"),
    ]
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(handles=existing_handles + handles, loc="upper left")
