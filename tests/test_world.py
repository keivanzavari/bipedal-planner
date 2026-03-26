"""Tests for stage1/world.py — occupancy grid, coordinate transforms, inflation."""

import numpy as np
import pytest

from stage1.world import Rect, World, _circle_struct

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_world() -> World:
    """4 × 3 m world at 1 m/cell (4 cols × 3 rows) with no obstacles."""
    return World(width=4.0, height=3.0, resolution=1.0)


@pytest.fixture()
def world_with_obstacle() -> World:
    """4 × 4 m world at 0.5 m/cell with a 1 × 1 obstacle at (1, 1)."""
    return World(width=4.0, height=4.0, resolution=0.5, obstacles=[Rect(1.0, 1.0, 1.0, 1.0)])


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


class TestCoordinateHelpers:
    def test_world_to_cell_origin(self, simple_world: World):
        row, col = simple_world.world_to_cell(0.0, 0.0)
        assert row == 0 and col == 0

    def test_world_to_cell_roundtrip(self, simple_world: World):
        """world → cell → world lands within one resolution of the original point."""
        for x, y in [(0.5, 0.5), (1.7, 2.3), (3.9, 2.9)]:
            row, col = simple_world.world_to_cell(x, y)
            rx, ry = simple_world.cell_to_world(row, col)
            assert abs(rx - simple_world.resolution * (col + 0.5)) < 1e-9
            assert abs(ry - simple_world.resolution * (row + 0.5)) < 1e-9

    def test_cell_to_world_centre(self, simple_world: World):
        """cell_to_world returns the cell centre."""
        x, y = simple_world.cell_to_world(0, 0)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)

    def test_in_bounds_valid_corners(self, simple_world: World):
        assert simple_world.in_bounds(0, 0)
        assert simple_world.in_bounds(simple_world.rows - 1, simple_world.cols - 1)

    def test_in_bounds_negative(self, simple_world: World):
        assert not simple_world.in_bounds(-1, 0)
        assert not simple_world.in_bounds(0, -1)

    def test_in_bounds_too_large(self, simple_world: World):
        assert not simple_world.in_bounds(simple_world.rows, 0)
        assert not simple_world.in_bounds(0, simple_world.cols)


# ---------------------------------------------------------------------------
# Grid / obstacle marking
# ---------------------------------------------------------------------------


class TestObstacleMarking:
    def test_free_world_all_zeros(self, simple_world: World):
        assert simple_world.grid.sum() == 0

    def test_obstacle_cells_are_occupied(self, world_with_obstacle: World):
        w = world_with_obstacle
        # Obstacle covers (1,1)→(2,2) at res=0.5 → rows 2-3, cols 2-3
        for r in range(2, 4):
            for c in range(2, 4):
                assert w.grid[r, c] == 1, f"Expected cell ({r},{c}) to be occupied"

    def test_adjacent_cells_free(self, world_with_obstacle: World):
        w = world_with_obstacle
        assert w.grid[0, 0] == 0
        assert w.grid[1, 1] == 0

    def test_is_free_on_obstacle(self, world_with_obstacle: World):
        assert not world_with_obstacle.is_free(2, 2)

    def test_is_free_on_empty_cell(self, world_with_obstacle: World):
        assert world_with_obstacle.is_free(0, 0)

    def test_is_free_out_of_bounds(self, simple_world: World):
        assert not simple_world.is_free(-1, 0)
        assert not simple_world.is_free(0, simple_world.cols)


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------


class TestInflation:
    def test_inflated_has_more_occupied_cells(self, world_with_obstacle: World):
        raw_count = world_with_obstacle.grid.sum()
        inflated = world_with_obstacle.inflated_grid(margin=0.6)
        assert inflated.sum() >= raw_count

    def test_zero_margin_reproduces_raw(self, world_with_obstacle: World):
        inflated = world_with_obstacle.inflated_grid(margin=0.0)
        np.testing.assert_array_equal(inflated, world_with_obstacle.grid)

    def test_inflated_grid_is_binary(self, world_with_obstacle: World):
        inflated = world_with_obstacle.inflated_grid(margin=0.5)
        assert set(np.unique(inflated)).issubset({0, 1})


# ---------------------------------------------------------------------------
# _circle_struct
# ---------------------------------------------------------------------------


class TestCircleStruct:
    @pytest.mark.parametrize("radius", [1, 2, 3, 5])
    def test_shape(self, radius: int):
        s = _circle_struct(radius)
        assert s.shape == (2 * radius + 1, 2 * radius + 1)

    def test_centre_is_one(self):
        s = _circle_struct(3)
        assert s[3, 3] == 1

    def test_corner_outside_circle_for_radius_2(self):
        # Corner (0,0) of a 5×5 struct centred at (2,2): distance = sqrt(8) > 2
        s = _circle_struct(2)
        assert s[0, 0] == 0
