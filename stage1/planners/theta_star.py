"""
Theta* — any-angle path planning on a 2-D occupancy grid.

Like A*, but when relaxing a neighbour it checks whether the grandparent
has a direct line-of-sight to that neighbour. If so, it bypasses the
intermediate node entirely, producing true any-angle paths without a
separate smoothing pass.

Reference: Nash et al., "Theta*: Any-Angle Path Planning on Grids", 2007.
"""

import heapq

import numpy as np

from stage1.world import World
from stage1.planners.base import _bresenham, smooth_path


class ThetaStarPlanner:
    def __init__(self, inflation_margin: float = 0.25, smooth: bool = False):
        self.inflation_margin = inflation_margin
        # smooth=False by default: Theta* produces any-angle paths natively
        self.smooth = smooth

    def plan(
        self,
        world: World,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> list[tuple[float, float]] | None:
        grid = world.inflated_grid(self.inflation_margin)

        start_cell = world.world_to_cell(*start)
        goal_cell  = world.world_to_cell(*goal)

        if not world.in_bounds(*start_cell) or grid[start_cell] == 1:
            raise ValueError(f"Start {start} is inside an obstacle or out of bounds.")
        if not world.in_bounds(*goal_cell) or grid[goal_cell] == 1:
            raise ValueError(f"Goal {goal} is inside an obstacle or out of bounds.")

        def heuristic(r, c):
            return np.hypot(r - goal_cell[0], c - goal_cell[1])

        def line_of_sight(r0, c0, r1, c1) -> bool:
            for r, c in _bresenham(r0, c0, r1, c1):
                if not world.in_bounds(r, c) or grid[r, c] == 1:
                    return False
            return True

        open_heap: list = []
        heapq.heappush(open_heap, (heuristic(*start_cell), 0.0, *start_cell))
        came_from: dict = {}
        g_score = {start_cell: 0.0}

        while open_heap:
            f, g, row, col = heapq.heappop(open_heap)

            if (row, col) == goal_cell:
                path = _reconstruct(came_from, goal_cell, world)
                return smooth_path(path, world, self.inflation_margin) if self.smooth else path

            if g > g_score.get((row, col), float("inf")):
                continue

            # Theta* neighbour relaxation
            parent = came_from.get((row, col), (row, col))
            pr, pc = parent

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if not world.in_bounds(nr, nc) or grid[nr, nc] == 1:
                        continue

                    # Path 2: try to connect grandparent directly to neighbour
                    if line_of_sight(pr, pc, nr, nc):
                        pg = g_score.get((pr, pc), float("inf"))
                        new_g = pg + np.hypot(nr - pr, nc - pc)
                        if new_g < g_score.get((nr, nc), float("inf")):
                            g_score[(nr, nc)] = new_g
                            came_from[(nr, nc)] = (pr, pc)
                            heapq.heappush(
                                open_heap,
                                (new_g + heuristic(nr, nc), new_g, nr, nc),
                            )
                    else:
                        # Path 1: standard A* relaxation through current node
                        step = np.hypot(dr, dc)
                        new_g = g + step
                        if new_g < g_score.get((nr, nc), float("inf")):
                            g_score[(nr, nc)] = new_g
                            came_from[(nr, nc)] = (row, col)
                            heapq.heappush(
                                open_heap,
                                (new_g + heuristic(nr, nc), new_g, nr, nc),
                            )

        return None


def _reconstruct(came_from, goal_cell, world):
    path, cell = [], goal_cell
    while cell in came_from:
        path.append(world.cell_to_world(*cell))
        cell = came_from[cell]
    path.append(world.cell_to_world(*cell))
    path.reverse()
    return path
