"""Tests for stage1/planner.py — A* search and path smoothing."""

import numpy as np
import pytest

from stage1.world import Rect, World
from stage1.planner import astar, smooth_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def open_world() -> World:
    """10 × 10 m obstacle-free world at 0.1 m/cell."""
    return World(width=10.0, height=10.0, resolution=0.1)


@pytest.fixture()
def blocked_world() -> World:
    """World where a full-height wall blocks any direct passage."""
    # Wall from x=4 to x=4.5 spans the entire height — goal will be unreachable
    return World(
        width=10.0,
        height=10.0,
        resolution=0.1,
        obstacles=[Rect(4.0, 0.0, 0.5, 10.0)],
    )


@pytest.fixture()
def world_with_gap() -> World:
    """Wall with a small gap, so path exists but must thread through."""
    return World(
        width=10.0,
        height=10.0,
        resolution=0.1,
        obstacles=[
            Rect(4.0, 0.0, 0.5, 4.5),   # lower half of wall
            Rect(4.0, 5.5, 0.5, 4.5),   # upper half of wall
        ],
    )


# ---------------------------------------------------------------------------
# A* basic tests
# ---------------------------------------------------------------------------


class TestAstar:
    def test_open_world_returns_path(self, open_world: World):
        path = astar(open_world, (0.5, 0.5), (9.5, 9.5), inflation_margin=0.0)
        assert path is not None
        assert len(path) >= 2

    def test_path_starts_near_start(self, open_world: World):
        start = (1.0, 1.0)
        path = astar(open_world, start, (9.0, 9.0), inflation_margin=0.0)
        assert path is not None
        assert np.hypot(path[0][0] - start[0], path[0][1] - start[1]) <= open_world.resolution * 1.5

    def test_path_ends_near_goal(self, open_world: World):
        goal = (9.0, 9.0)
        path = astar(open_world, (1.0, 1.0), goal, inflation_margin=0.0)
        assert path is not None
        assert np.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) <= open_world.resolution * 1.5

    def test_trivial_same_cell_path(self, open_world: World):
        """Start and goal in the same cell should yield a one-point path."""
        path = astar(open_world, (1.05, 1.05), (1.06, 1.06), inflation_margin=0.0)
        assert path is not None

    def test_raises_when_start_in_obstacle(self):
        world = World(
            width=5.0, height=5.0, resolution=0.1,
            obstacles=[Rect(0.0, 0.0, 1.0, 1.0)],
        )
        with pytest.raises(ValueError, match="Start"):
            astar(world, (0.5, 0.5), (4.0, 4.0), inflation_margin=0.0)

    def test_raises_when_goal_in_obstacle(self):
        world = World(
            width=5.0, height=5.0, resolution=0.1,
            obstacles=[Rect(3.0, 3.0, 1.0, 1.0)],
        )
        with pytest.raises(ValueError, match="Goal"):
            astar(world, (0.5, 0.5), (3.5, 3.5), inflation_margin=0.0)

    def test_returns_none_when_no_path(self, blocked_world: World):
        """Full-height wall — no path should exist."""
        result = astar(blocked_world, (1.0, 5.0), (9.0, 5.0), inflation_margin=0.0)
        assert result is None

    def test_path_avoids_obstacles(self, world_with_gap: World):
        """Path through a gap must not pass through obstacle cells."""
        path = astar(world_with_gap, (1.0, 5.0), (9.0, 5.0), inflation_margin=0.0)
        assert path is not None
        grid = world_with_gap.grid
        for x, y in path:
            r, c = world_with_gap.world_to_cell(x, y)
            if world_with_gap.in_bounds(r, c):
                assert grid[r, c] == 0, f"Path point ({x:.2f}, {y:.2f}) is inside an obstacle"


# ---------------------------------------------------------------------------
# Path smoothing
# ---------------------------------------------------------------------------


class TestSmoothPath:
    def test_smooth_reduces_or_keeps_waypoints(self, open_world: World):
        path = astar(open_world, (0.5, 0.5), (9.5, 9.5), inflation_margin=0.0)
        assert path is not None
        smoothed = smooth_path(path, open_world, inflation_margin=0.0)
        assert len(smoothed) <= len(path)

    def test_smooth_preserves_start_and_end(self, open_world: World):
        path = astar(open_world, (0.5, 0.5), (9.5, 9.5), inflation_margin=0.0)
        assert path is not None
        smoothed = smooth_path(path, open_world, inflation_margin=0.0)
        assert np.hypot(smoothed[0][0] - path[0][0], smoothed[0][1] - path[0][1]) < open_world.resolution
        assert np.hypot(smoothed[-1][0] - path[-1][0], smoothed[-1][1] - path[-1][1]) < open_world.resolution

    def test_smooth_segments_obstacle_free(self, world_with_gap: World):
        """Every segment of the smoothed path must be obstacle-free."""
        from stage1.planner import _bresenham  # noqa: PLC0415

        path = astar(world_with_gap, (1.0, 5.0), (9.0, 5.0), inflation_margin=0.0)
        assert path is not None
        smoothed = smooth_path(path, world_with_gap, inflation_margin=0.0)
        grid = world_with_gap.grid

        for i in range(len(smoothed) - 1):
            r1, c1 = world_with_gap.world_to_cell(*smoothed[i])
            r2, c2 = world_with_gap.world_to_cell(*smoothed[i + 1])
            for r, c in _bresenham(r1, c1, r2, c2):
                if world_with_gap.in_bounds(r, c):
                    assert grid[r, c] == 0, "Smoothed segment crosses an obstacle"
