"""
Planner protocol and shared utilities (smooth_path, Bresenham line check).
"""

from typing import Protocol

import numpy as np

from stage1.world import World


class Planner(Protocol):
    def plan(
        self,
        world: World,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> list[tuple[float, float]] | None:
        """
        Return an ordered list of (x, y) world-coordinate waypoints from
        start to goal, or None if no path exists.

        The returned path must be collision-free in `world`.
        """
        ...


def smooth_path(
    path: list[tuple[float, float]],
    world: World,
    inflation_margin: float = 0.2,
    iterations: int = 3,
) -> list[tuple[float, float]]:
    """
    Path shortcutting: repeatedly skip intermediate waypoints when the
    straight line between two non-adjacent waypoints is collision-free.
    """
    grid = world.inflated_grid(inflation_margin)

    def line_free(p1, p2):
        r1, c1 = world.world_to_cell(*p1)
        r2, c2 = world.world_to_cell(*p2)
        for r, c in _bresenham(r1, c1, r2, c2):
            if not world.in_bounds(r, c) or grid[r, c] == 1:
                return False
        return True

    for _ in range(iterations):
        i = 0
        pruned = [path[0]]
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not line_free(path[i], path[j]):
                j -= 1
            pruned.append(path[j])
            i = j
        path = pruned

    return path


def _bresenham(r0, c0, r1, c1):
    """Yield all (row, col) cells on the line from (r0, c0) to (r1, c1)."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        yield r, c
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
