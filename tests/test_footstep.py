"""Tests for stage1/footstep.py — resampling, foot geometry, step planning."""

import math

import numpy as np
import pytest

from stage1.world import World
from stage1.footstep import (
    Footstep,
    _foot_corners,
    _foot_is_free,
    _resample_path,
    plan_footsteps,
)


# ---------------------------------------------------------------------------
# _resample_path
# ---------------------------------------------------------------------------


class TestResamplePath:
    def test_straight_segment_spacing(self):
        """Samples along a straight horizontal segment should be step_length apart."""
        waypoints = [(0.0, 0.0), (10.0, 0.0)]
        step = 0.5
        samples = _resample_path(waypoints, step_length=step)
        assert len(samples) > 0
        for i in range(len(samples) - 1):
            x0, y0, _ = samples[i]
            x1, y1, _ = samples[i + 1]
            dist = math.hypot(x1 - x0, y1 - y0)
            assert dist == pytest.approx(step, abs=1e-9)

    def test_heading_correct_for_horizontal(self):
        waypoints = [(0.0, 0.0), (5.0, 0.0)]
        samples = _resample_path(waypoints, step_length=1.0)
        for _, _, theta in samples:
            assert theta == pytest.approx(0.0)

    def test_heading_correct_for_vertical(self):
        waypoints = [(0.0, 0.0), (0.0, 5.0)]
        samples = _resample_path(waypoints, step_length=1.0)
        for _, _, theta in samples:
            assert theta == pytest.approx(math.pi / 2)

    def test_fewer_than_two_waypoints_empty(self):
        assert _resample_path([], step_length=0.25) == []
        assert _resample_path([(1.0, 1.0)], step_length=0.25) == []

    def test_zero_length_segment_skipped(self):
        """Duplicated waypoints (zero-length segment) should not produce samples."""
        waypoints = [(0.0, 0.0), (0.0, 0.0), (5.0, 0.0)]
        samples = _resample_path(waypoints, step_length=1.0)
        assert len(samples) > 0  # still processes valid segment

    def test_short_path_fewer_samples(self):
        """A path shorter than step_length yields at most one sample."""
        waypoints = [(0.0, 0.0), (0.1, 0.0)]
        samples = _resample_path(waypoints, step_length=1.0)
        assert len(samples) <= 1


# ---------------------------------------------------------------------------
# _foot_corners
# ---------------------------------------------------------------------------


class TestFootCorners:
    def test_shape(self):
        corners = _foot_corners(0.0, 0.0, 0.0, foot_length=0.16, foot_width=0.08)
        assert corners.shape == (4, 2)

    def test_theta_zero_axis_aligned(self):
        """At theta=0 the foot is aligned with the x-axis."""
        fl, fw = 0.2, 0.1
        corners = _foot_corners(0.0, 0.0, 0.0, fl, fw)
        xs = corners[:, 0]
        ys = corners[:, 1]
        assert xs.max() == pytest.approx(fl / 2, abs=1e-9)
        assert xs.min() == pytest.approx(-fl / 2, abs=1e-9)
        assert ys.max() == pytest.approx(fw / 2, abs=1e-9)
        assert ys.min() == pytest.approx(-fw / 2, abs=1e-9)

    def test_theta_90_rotates_long_axis(self):
        """At theta=π/2 the long axis should align with y."""
        fl, fw = 0.2, 0.1
        corners = _foot_corners(0.0, 0.0, math.pi / 2, fl, fw)
        ys = corners[:, 1]
        xs = corners[:, 0]
        assert ys.max() == pytest.approx(fl / 2, abs=1e-9)
        assert ys.min() == pytest.approx(-fl / 2, abs=1e-9)
        assert xs.max() == pytest.approx(fw / 2, abs=1e-9)
        assert xs.min() == pytest.approx(-fw / 2, abs=1e-9)

    def test_translation_applied(self):
        """Foot centre at (3, 4) should shift all corners accordingly."""
        corners_origin = _foot_corners(0.0, 0.0, 0.0, 0.2, 0.1)
        corners_moved = _foot_corners(3.0, 4.0, 0.0, 0.2, 0.1)
        np.testing.assert_allclose(corners_moved, corners_origin + np.array([3.0, 4.0]))


# ---------------------------------------------------------------------------
# _foot_is_free
# ---------------------------------------------------------------------------


class TestFootIsFree:
    @pytest.fixture()
    def free_world(self) -> World:
        return World(width=5.0, height=5.0, resolution=0.05)

    @pytest.fixture()
    def occupied_world(self) -> World:
        from stage1.world import Rect
        return World(width=5.0, height=5.0, resolution=0.05, obstacles=[Rect(2.0, 2.0, 1.0, 1.0)])

    def test_free_grid_returns_true(self, free_world: World):
        free_grid = free_world.grid.copy()  # all zeros
        result = _foot_is_free(2.5, 2.5, 0.0, 0.16, 0.08, free_world, free_grid)
        assert result is True

    def test_occupied_grid_returns_false(self, occupied_world: World):
        obst_grid = occupied_world.grid.copy()
        # Place foot directly on obstacle centre
        result = _foot_is_free(2.5, 2.5, 0.0, 0.16, 0.08, occupied_world, obst_grid)
        assert result is False

    def test_foot_far_from_obstacle_is_free(self, occupied_world: World):
        clear_grid = occupied_world.grid.copy()
        result = _foot_is_free(0.5, 0.5, 0.0, 0.16, 0.08, occupied_world, clear_grid)
        assert result is True


# ---------------------------------------------------------------------------
# plan_footsteps
# ---------------------------------------------------------------------------


class TestPlanFootsteps:
    @pytest.fixture()
    def corridor_world(self) -> World:
        """Simple wide corridor — enough room for feet on both sides."""
        return World(width=10.0, height=5.0, resolution=0.05)

    @pytest.fixture()
    def long_straight_path(self) -> list[tuple[float, float]]:
        return [(x, 2.5) for x in np.linspace(0.5, 9.5, 40)]

    def test_alternating_sides(self, corridor_world: World, long_straight_path):
        steps = plan_footsteps(long_straight_path, corridor_world, first_side="L")
        assert len(steps) > 0
        for i in range(len(steps) - 1):
            assert steps[i].side != steps[i + 1].side

    def test_first_side_respected(self, corridor_world: World, long_straight_path):
        steps_l = plan_footsteps(long_straight_path, corridor_world, first_side="L")
        steps_r = plan_footsteps(long_straight_path, corridor_world, first_side="R")
        assert steps_l[0].side == "L"
        assert steps_r[0].side == "R"

    def test_produces_steps_for_long_path(self, corridor_world: World, long_straight_path):
        steps = plan_footsteps(long_straight_path, corridor_world, step_length=0.25)
        assert len(steps) >= 2

    def test_footstep_has_valid_side(self, corridor_world: World, long_straight_path):
        steps = plan_footsteps(long_straight_path, corridor_world)
        for s in steps:
            assert s.side in ("L", "R")

    def test_empty_path_returns_no_steps(self, corridor_world: World):
        steps = plan_footsteps([], corridor_world)
        assert steps == []

    def test_short_path_returns_no_steps(self, corridor_world: World):
        """A path shorter than step_length should yield no footsteps."""
        short = [(0.0, 2.5), (0.1, 2.5)]
        steps = plan_footsteps(short, corridor_world, step_length=0.25)
        assert steps == []
