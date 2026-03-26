"""Tests for stage1/planners/rrt.py — RRTPlanner."""

import numpy as np
import pytest

from stage1.planners.rrt import RRTPlanner
from stage1.world import Rect, World

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def open_world() -> World:
    return World(width=10.0, height=10.0, resolution=0.1)


@pytest.fixture()
def blocked_world() -> World:
    """Full-width wall — goal is unreachable."""
    return World(
        width=10.0,
        height=10.0,
        resolution=0.1,
        obstacles=[Rect(4.0, 0.0, 0.5, 10.0)],
    )


@pytest.fixture()
def world_with_gap() -> World:
    return World(
        width=10.0,
        height=10.0,
        resolution=0.1,
        obstacles=[
            Rect(4.0, 0.0, 0.5, 4.5),
            Rect(4.0, 5.5, 0.5, 4.5),
        ],
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestRRTPlanner:
    def test_open_world_returns_path(self, open_world: World):
        planner = RRTPlanner(inflation_margin=0.0, seed=0)
        path = planner.plan(open_world, (0.5, 0.5), (9.5, 9.5))
        assert path is not None
        assert len(path) >= 2

    def test_path_starts_at_start(self, open_world: World):
        start = (1.0, 1.0)
        planner = RRTPlanner(inflation_margin=0.0, seed=1)
        path = planner.plan(open_world, start, (9.0, 9.0))
        assert path is not None
        assert np.hypot(path[0][0] - start[0], path[0][1] - start[1]) < 0.5

    def test_path_ends_at_goal(self, open_world: World):
        goal = (9.0, 9.0)
        planner = RRTPlanner(inflation_margin=0.0, seed=2)
        path = planner.plan(open_world, (1.0, 1.0), goal)
        assert path is not None
        assert np.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) < 0.5

    def test_raises_when_start_in_obstacle(self):
        world = World(width=5.0, height=5.0, resolution=0.1, obstacles=[Rect(0.0, 0.0, 1.0, 1.0)])
        with pytest.raises(ValueError, match="Start"):
            RRTPlanner(inflation_margin=0.0, seed=0).plan(world, (0.5, 0.5), (4.0, 4.0))

    def test_raises_when_goal_in_obstacle(self):
        world = World(width=5.0, height=5.0, resolution=0.1, obstacles=[Rect(3.0, 3.0, 1.0, 1.0)])
        with pytest.raises(ValueError, match="Goal"):
            RRTPlanner(inflation_margin=0.0, seed=0).plan(world, (0.5, 0.5), (3.5, 3.5))

    def test_returns_none_when_no_path(self, blocked_world: World):
        """Even with many iterations the blocked world should yield no path."""
        planner = RRTPlanner(inflation_margin=0.0, max_iterations=2000, seed=42)
        result = planner.plan(blocked_world, (1.0, 5.0), (9.0, 5.0))
        assert result is None

    def test_path_nodes_are_in_free_space(self, open_world: World):
        """All waypoints in the returned path should lie in obstacle-free cells."""
        planner = RRTPlanner(inflation_margin=0.0, seed=3, smooth=False)
        path = planner.plan(open_world, (0.5, 0.5), (9.5, 9.5))
        assert path is not None
        for x, y in path:
            r, c = open_world.world_to_cell(x, y)
            if open_world.in_bounds(r, c):
                assert open_world.grid[r, c] == 0

    def test_gap_world_finds_path(self, world_with_gap: World):
        """With enough iterations RRT must thread through the gap."""
        planner = RRTPlanner(
            inflation_margin=0.0,
            max_iterations=10_000,
            step_size=0.3,
            goal_bias=0.15,
            seed=7,
        )
        path = planner.plan(world_with_gap, (1.0, 5.0), (9.0, 5.0))
        assert path is not None


# ---------------------------------------------------------------------------
# Configuration / reproducibility
# ---------------------------------------------------------------------------


class TestRRTPlannerConfig:
    def test_seed_produces_reproducible_result(self, open_world: World):
        p1 = RRTPlanner(inflation_margin=0.0, seed=99, smooth=False)
        p2 = RRTPlanner(inflation_margin=0.0, seed=99, smooth=False)
        path1 = p1.plan(open_world, (1.0, 1.0), (9.0, 9.0))
        path2 = p2.plan(open_world, (1.0, 1.0), (9.0, 9.0))
        assert path1 is not None and path2 is not None
        assert len(path1) == len(path2)
        for (x1, y1), (x2, y2) in zip(path1, path2, strict=True):
            assert x1 == pytest.approx(x2) and y1 == pytest.approx(y2)

    def test_smooth_flag_reduces_waypoints(self, open_world: World):
        raw = RRTPlanner(inflation_margin=0.0, seed=5, smooth=False).plan(open_world, (0.5, 0.5), (9.5, 9.5))
        smoothed = RRTPlanner(inflation_margin=0.0, seed=5, smooth=True).plan(open_world, (0.5, 0.5), (9.5, 9.5))
        assert raw is not None and smoothed is not None
        assert len(smoothed) <= len(raw)

    def test_default_params_accessible(self):
        p = RRTPlanner()
        assert p.inflation_margin >= 0
        assert p.max_iterations > 0
        assert p.step_size > 0
        assert 0.0 <= p.goal_bias <= 1.0
