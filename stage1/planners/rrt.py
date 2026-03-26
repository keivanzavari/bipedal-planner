"""
RRT — Rapidly-exploring Random Tree path planner.

Works directly in world coordinates without a grid. Collision checking is
done on line segments by sampling at `world.resolution` intervals against
the inflated occupancy grid.
"""

import numpy as np

from stage1.world import World
from stage1.planners.base import smooth_path


class RRTPlanner:
    def __init__(
        self,
        inflation_margin: float = 0.25,
        max_iterations: int = 5000,
        step_size: float = 0.3,
        goal_bias: float = 0.1,
        smooth: bool = True,
        seed: int | None = None,
    ):
        self.inflation_margin = inflation_margin
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.smooth = smooth
        self._rng = np.random.default_rng(seed)

    def plan(
        self,
        world: World,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> list[tuple[float, float]] | None:
        grid = world.inflated_grid(self.inflation_margin)

        def point_free(x, y) -> bool:
            r, c = world.world_to_cell(x, y)
            return world.in_bounds(r, c) and grid[r, c] == 0

        def edge_free(p1, p2) -> bool:
            dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps = max(2, int(dist / world.resolution))
            for t in np.linspace(0, 1, steps):
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                if not point_free(x, y):
                    return False
            return True

        if not point_free(*start):
            raise ValueError(f"Start {start} is inside an obstacle or out of bounds.")
        if not point_free(*goal):
            raise ValueError(f"Goal {goal} is inside an obstacle or out of bounds.")

        # Tree: list of nodes, parent index stored separately
        nodes = [np.array(start)]
        parents = [-1]  # root has no parent

        goal_arr = np.array(goal)

        for _ in range(self.max_iterations):
            # Sample: bias toward goal with probability goal_bias
            if self._rng.random() < self.goal_bias:
                sample = goal_arr.copy()
            else:
                sample = np.array(
                    [
                        self._rng.uniform(0, world.width),
                        self._rng.uniform(0, world.height),
                    ]
                )

            # Nearest node in the tree
            dists = [np.linalg.norm(n - sample) for n in nodes]
            nearest = int(np.argmin(dists))

            # Steer toward sample by at most step_size
            direction = sample - nodes[nearest]
            dist = np.linalg.norm(direction)
            if dist == 0:
                continue
            new_node = nodes[nearest] + direction / dist * min(dist, self.step_size)

            if not edge_free(nodes[nearest], new_node):
                continue

            nodes.append(new_node)
            parents.append(nearest)

            # Check if we can connect to the goal
            if np.linalg.norm(new_node - goal_arr) <= self.step_size:
                if edge_free(new_node, goal_arr):
                    nodes.append(goal_arr)
                    parents.append(len(nodes) - 2)
                    path = _reconstruct(nodes, parents)
                    return smooth_path(path, world, self.inflation_margin) if self.smooth else path

        return None  # no path found within max_iterations


def _reconstruct(nodes, parents):
    path = []
    idx = len(nodes) - 1
    while idx != -1:
        path.append(tuple(nodes[idx]))
        idx = parents[idx]
    path.reverse()
    return path
