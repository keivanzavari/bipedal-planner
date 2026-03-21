import heapq

import numpy as np

from stage1.world import World

# 8-connected neighbours: (d_row, d_col, cost)
_NEIGHBOURS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 1.4142),
    (-1, 1, 1.4142),
    (1, -1, 1.4142),
    (1, 1, 1.4142),
]


def astar(
    world: World,
    start: tuple[float, float],
    goal: tuple[float, float],
    inflation_margin: float = 0.2,
) -> list[tuple[float, float]] | None:
    """
    Run A* on the inflated occupancy grid.

    Parameters
    ----------
    world            : World instance
    start / goal     : (x, y) in world coordinates (meters)
    inflation_margin : obstacle dilation applied before planning

    Returns
    -------
    List of (x, y) world-coordinate waypoints from start to goal,
    or None if no path exists.
    """
    grid = world.inflated_grid(inflation_margin)

    start_cell = world.world_to_cell(*start)
    goal_cell = world.world_to_cell(*goal)

    if not world.in_bounds(*start_cell) or grid[start_cell] == 1:
        raise ValueError(f"Start {start} is inside an obstacle or out of bounds.")
    if not world.in_bounds(*goal_cell) or grid[goal_cell] == 1:
        raise ValueError(f"Goal {goal} is inside an obstacle or out of bounds.")

    def heuristic(r, c):
        # Euclidean distance in cell units
        return np.hypot(r - goal_cell[0], c - goal_cell[1])

    # Priority queue entries: (f, g, row, col)
    open_heap = []
    heapq.heappush(open_heap, (heuristic(*start_cell), 0.0, *start_cell))

    came_from = {}  # cell → parent cell
    g_score = {start_cell: 0.0}

    while open_heap:
        f, g, row, col = heapq.heappop(open_heap)

        if (row, col) == goal_cell:
            return _reconstruct(came_from, goal_cell, world)

        # Skip stale entries
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

    return None  # no path found


def _reconstruct(
    came_from: dict,
    goal_cell: tuple[int, int],
    world: World,
) -> list[tuple[float, float]]:
    """Walk back through came_from and convert cells to world coordinates."""
    path = []
    cell = goal_cell
    while cell in came_from:
        path.append(world.cell_to_world(*cell))
        cell = came_from[cell]
    path.append(world.cell_to_world(*cell))  # start cell
    path.reverse()
    return path


def smooth_path(
    path: list[tuple[float, float]],
    world: World,
    inflation_margin: float = 0.2,
    iterations: int = 3,
) -> list[tuple[float, float]]:
    """
    Path shortcutting: repeatedly try to skip intermediate waypoints
    when the straight line between two non-adjacent waypoints is collision-free.
    """
    grid = world.inflated_grid(inflation_margin)

    def line_free(p1, p2):
        """Bresenham line check — all cells on segment must be free."""
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
            # Find the farthest point reachable in a straight line from path[i]
            j = len(path) - 1
            while j > i + 1 and not line_free(path[i], path[j]):
                j -= 1
            pruned.append(path[j])
            i = j
        path = pruned

    return path


def _bresenham(r0, c0, r1, c1):
    """Yield all (row, col) cells on the line from (r0,c0) to (r1,c1)."""
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

