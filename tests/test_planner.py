"""Tests for stage1/planners — AStarPlanner, ThetaStarPlanner, smooth_path, and registry."""

import numpy as np
import pytest

from stage1.world import Rect, World
from stage1.planners import PLANNERS, get_planner
from stage1.planners.astar import AStarPlanner
from stage1.planners.base import _bresenham, smooth_path
from stage1.planners.theta_star import ThetaStarPlanner


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
            Rect(4.0, 0.0, 0.5, 4.5),  # lower half of wall
            Rect(4.0, 5.5, 0.5, 4.5),  # upper half of wall
        ],
    )


# ---------------------------------------------------------------------------
# Shared Planner contract — parametrized over A* and Theta*
# Both are deterministic grid-based planners implementing the same protocol.
# ---------------------------------------------------------------------------


GRID_PLANNERS = [
    pytest.param(AStarPlanner, id="astar"),
    pytest.param(ThetaStarPlanner, id="theta_star"),
]


@pytest.mark.parametrize("planner_cls", GRID_PLANNERS)
class TestPlannerContract:
    """Behaviour that every Planner implementation must satisfy."""

    def test_open_world_returns_path(self, planner_cls, open_world: World):
        path = planner_cls(inflation_margin=0.0).plan(open_world, (0.5, 0.5), (9.5, 9.5))
        assert path is not None
        assert len(path) >= 2

    def test_path_starts_near_start(self, planner_cls, open_world: World):
        start = (1.0, 1.0)
        path = planner_cls(inflation_margin=0.0).plan(open_world, start, (9.0, 9.0))
        assert path is not None
        assert np.hypot(path[0][0] - start[0], path[0][1] - start[1]) <= open_world.resolution * 1.5

    def test_path_ends_near_goal(self, planner_cls, open_world: World):
        goal = (9.0, 9.0)
        path = planner_cls(inflation_margin=0.0).plan(open_world, (1.0, 1.0), goal)
        assert path is not None
        assert np.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) <= open_world.resolution * 1.5

    def test_trivial_same_cell_returns_path(self, planner_cls, open_world: World):
        path = planner_cls(inflation_margin=0.0).plan(open_world, (1.05, 1.05), (1.06, 1.06))
        assert path is not None

    def test_raises_when_start_in_obstacle(self, planner_cls):
        world = World(width=5.0, height=5.0, resolution=0.1, obstacles=[Rect(0.0, 0.0, 1.0, 1.0)])
        with pytest.raises(ValueError, match="Start"):
            planner_cls(inflation_margin=0.0).plan(world, (0.5, 0.5), (4.0, 4.0))

    def test_raises_when_goal_in_obstacle(self, planner_cls):
        world = World(width=5.0, height=5.0, resolution=0.1, obstacles=[Rect(3.0, 3.0, 1.0, 1.0)])
        with pytest.raises(ValueError, match="Goal"):
            planner_cls(inflation_margin=0.0).plan(world, (0.5, 0.5), (3.5, 3.5))

    def test_returns_none_when_no_path(self, planner_cls, blocked_world: World):
        result = planner_cls(inflation_margin=0.0).plan(blocked_world, (1.0, 5.0), (9.0, 5.0))
        assert result is None

    def test_path_avoids_obstacles(self, planner_cls, world_with_gap: World):
        path = planner_cls(inflation_margin=0.0).plan(world_with_gap, (1.0, 5.0), (9.0, 5.0))
        assert path is not None
        grid = world_with_gap.grid
        for x, y in path:
            r, c = world_with_gap.world_to_cell(x, y)
            if world_with_gap.in_bounds(r, c):
                assert grid[r, c] == 0, f"Path point ({x:.2f}, {y:.2f}) is inside an obstacle"


# ---------------------------------------------------------------------------
# AStarPlanner-specific tests
# ---------------------------------------------------------------------------


class TestAStarPlanner:
    def test_smooth_flag_true_reduces_waypoints(self, open_world: World):
        """smooth=True should yield paths with fewer or equal waypoints than smooth=False."""
        raw = AStarPlanner(inflation_margin=0.0, smooth=False).plan(open_world, (0.5, 0.5), (9.5, 9.5))
        smoothed = AStarPlanner(inflation_margin=0.0, smooth=True).plan(open_world, (0.5, 0.5), (9.5, 9.5))
        assert raw is not None and smoothed is not None
        assert len(smoothed) <= len(raw)

    def test_inflation_blocks_path_near_obstacle(self):
        """With large inflation, a narrow-gap world should become impassable."""
        world = World(
            width=10.0,
            height=10.0,
            resolution=0.1,
            obstacles=[
                Rect(4.0, 0.0, 0.5, 4.8),
                Rect(4.0, 5.2, 0.5, 4.8),
            ],
        )
        # Gap is 0.4 m wide; inflation of 0.3 m should close it
        result = AStarPlanner(inflation_margin=0.3, smooth=False).plan(world, (1.0, 5.0), (9.0, 5.0))
        assert result is None


# ---------------------------------------------------------------------------
# ThetaStarPlanner-specific tests
# ---------------------------------------------------------------------------


class TestThetaStarPlanner:
    def test_produces_fewer_waypoints_than_astar(self, open_world: World):
        """Theta* produces any-angle paths, so waypoint count should be ≤ A*'s."""
        astar_path = AStarPlanner(inflation_margin=0.0, smooth=False).plan(open_world, (0.5, 0.5), (9.5, 9.5))
        theta_path = ThetaStarPlanner(inflation_margin=0.0, smooth=False).plan(open_world, (0.5, 0.5), (9.5, 9.5))
        assert astar_path is not None and theta_path is not None
        assert len(theta_path) <= len(astar_path)

    def test_path_segments_are_obstacle_free(self, world_with_gap: World):
        """Every straight segment of a Theta* path must be clear of obstacles."""
        path = ThetaStarPlanner(inflation_margin=0.0).plan(world_with_gap, (1.0, 5.0), (9.0, 5.0))
        assert path is not None
        grid = world_with_gap.grid
        for i in range(len(path) - 1):
            r1, c1 = world_with_gap.world_to_cell(*path[i])
            r2, c2 = world_with_gap.world_to_cell(*path[i + 1])
            for r, c in _bresenham(r1, c1, r2, c2):
                if world_with_gap.in_bounds(r, c):
                    assert grid[r, c] == 0, "Theta* segment crosses an obstacle"


# ---------------------------------------------------------------------------
# smooth_path (base utility)
# ---------------------------------------------------------------------------


class TestSmoothPath:
    def _raw_astar_path(self, world: World, start, goal):
        return AStarPlanner(inflation_margin=0.0, smooth=False).plan(world, start, goal)

    def test_reduces_or_keeps_waypoints(self, open_world: World):
        path = self._raw_astar_path(open_world, (0.5, 0.5), (9.5, 9.5))
        assert path is not None
        smoothed = smooth_path(path, open_world, inflation_margin=0.0)
        assert len(smoothed) <= len(path)

    def test_preserves_start_and_end(self, open_world: World):
        path = self._raw_astar_path(open_world, (0.5, 0.5), (9.5, 9.5))
        assert path is not None
        smoothed = smooth_path(path, open_world, inflation_margin=0.0)
        assert np.hypot(smoothed[0][0] - path[0][0], smoothed[0][1] - path[0][1]) < open_world.resolution
        assert np.hypot(smoothed[-1][0] - path[-1][0], smoothed[-1][1] - path[-1][1]) < open_world.resolution

    def test_segments_are_obstacle_free(self, world_with_gap: World):
        """Every segment of the smoothed path must clear all obstacle cells."""
        path = self._raw_astar_path(world_with_gap, (1.0, 5.0), (9.0, 5.0))
        assert path is not None
        smoothed = smooth_path(path, world_with_gap, inflation_margin=0.0)
        grid = world_with_gap.grid
        for i in range(len(smoothed) - 1):
            r1, c1 = world_with_gap.world_to_cell(*smoothed[i])
            r2, c2 = world_with_gap.world_to_cell(*smoothed[i + 1])
            for r, c in _bresenham(r1, c1, r2, c2):
                if world_with_gap.in_bounds(r, c):
                    assert grid[r, c] == 0, "Smoothed segment crosses an obstacle"

    def test_single_point_path_unchanged(self, open_world: World):
        path = [(3.0, 3.0)]
        smoothed = smooth_path(path, open_world, inflation_margin=0.0)
        assert smoothed == path

    def test_two_point_path_unchanged(self, open_world: World):
        path = [(1.0, 1.0), (8.0, 8.0)]
        smoothed = smooth_path(path, open_world, inflation_margin=0.0)
        assert len(smoothed) == 2


# ---------------------------------------------------------------------------
# get_planner factory and PLANNERS registry
# ---------------------------------------------------------------------------


class TestPlannerRegistry:
    def test_planners_dict_contains_expected_keys(self):
        assert "astar" in PLANNERS
        assert "theta_star" in PLANNERS
        assert "rrt" in PLANNERS

    def test_get_planner_returns_astar(self):
        p = get_planner("astar")
        assert isinstance(p, AStarPlanner)

    def test_get_planner_returns_theta_star(self):
        from stage1.planners.theta_star import ThetaStarPlanner as _T

        p = get_planner("theta_star")
        assert isinstance(p, _T)

    def test_get_planner_returns_rrt(self):
        from stage1.planners.rrt import RRTPlanner

        p = get_planner("rrt")
        assert isinstance(p, RRTPlanner)

    def test_get_planner_raises_on_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown planner"):
            get_planner("nonexistent_planner")

    def test_get_planner_passes_kwargs(self):
        p = get_planner("astar", inflation_margin=0.5)
        assert p.inflation_margin == pytest.approx(0.5)
