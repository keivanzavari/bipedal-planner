import heapq

import numpy as np

from stage1.world import World
from stage1.planners.base import smooth_path

# 8-connected neighbours: (d_row, d_col, cost)
_NEIGHBOURS = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, 1.4142),
    (-1,  1, 1.4142),
    ( 1, -1, 1.4142),
    ( 1,  1, 1.4142),
]


class AStarPlanner:
    def __init__(self, inflation_margin: float = 0.25, smooth: bool = True):
        self.inflation_margin = inflation_margin
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

        open_heap = []
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

            for dr, dc, cost in _NEIGHBOURS:
                nr, nc = row + dr, col + dc
                if not world.in_bounds(nr, nc) or grid[nr, nc] == 1:
                    continue
                new_g = g + cost
                if new_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = new_g
                    came_from[(nr, nc)] = (row, col)
                    heapq.heappush(open_heap, (new_g + heuristic(nr, nc), new_g, nr, nc))

        return None


def _reconstruct(came_from, goal_cell, world):
    path, cell = [], goal_cell
    while cell in came_from:
        path.append(world.cell_to_world(*cell))
        cell = came_from[cell]
    path.append(world.cell_to_world(*cell))
    path.reverse()
    return path
