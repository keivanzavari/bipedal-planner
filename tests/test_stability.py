"""Tests for stage1/stability.py — support polygon and stability checks."""

import numpy as np

from stage1.footstep import Footstep
from stage1.stability import (
    StancePhase,
    _convex_hull_polygon,
    _point_in_polygon,
    check_stability,
)

# ---------------------------------------------------------------------------
# _point_in_polygon
# ---------------------------------------------------------------------------


class TestPointInPolygon:
    # Unit square vertices (counter-clockwise)
    UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    def test_centroid_inside(self):
        assert _point_in_polygon(np.array([0.5, 0.5]), self.UNIT_SQUARE) is True

    def test_near_edge_inside(self):
        assert _point_in_polygon(np.array([0.1, 0.1]), self.UNIT_SQUARE) is True

    def test_outside_positive_x(self):
        assert _point_in_polygon(np.array([2.0, 0.5]), self.UNIT_SQUARE) is False

    def test_outside_negative_x(self):
        assert _point_in_polygon(np.array([-0.5, 0.5]), self.UNIT_SQUARE) is False

    def test_outside_positive_y(self):
        assert _point_in_polygon(np.array([0.5, 2.0]), self.UNIT_SQUARE) is False

    def test_outside_negative_y(self):
        assert _point_in_polygon(np.array([0.5, -1.0]), self.UNIT_SQUARE) is False

    def test_far_from_polygon(self):
        assert _point_in_polygon(np.array([100.0, 100.0]), self.UNIT_SQUARE) is False


# ---------------------------------------------------------------------------
# _convex_hull_polygon
# ---------------------------------------------------------------------------


class TestConvexHullPolygon:
    def test_rectangle_has_four_vertices(self):
        rect = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]])
        hull = _convex_hull_polygon(rect)
        assert len(hull) == 4

    def test_hull_of_collinear_points(self):
        """For fewer than 3 points the function returns them as-is."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        hull = _convex_hull_polygon(pts)
        assert len(hull) == 2

    def test_interior_point_excluded(self):
        """A point strictly inside a square must not appear in its hull."""
        pts = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [1.0, 1.0],  # interior
            ]
        )
        hull = _convex_hull_polygon(pts)
        assert len(hull) == 4


# ---------------------------------------------------------------------------
# check_stability
# ---------------------------------------------------------------------------


def _make_footstep(x: float, y: float, side: str = "L", theta: float = 0.0) -> Footstep:
    return Footstep(side=side, x=x, y=y, theta=theta)


class TestCheckStability:
    def test_returns_list_of_stance_phases(self):
        steps = [_make_footstep(0.0, 0.1, "L"), _make_footstep(0.25, -0.1, "R")]
        phases = check_stability(steps)
        assert isinstance(phases, list)
        assert len(phases) > 0
        for p in phases:
            assert isinstance(p, StancePhase)

    def test_single_support_com_inside_foot_is_stable(self):
        """CoM at the foot centre — should be inside its rectangle → stable."""
        # Two widely spaced steps; during single support of first step the CoM
        # is the midpoint between step 0 and step 1. Place step 1 very close so
        # the midpoint stays inside the foot rectangle.
        foot_l, foot_w = 0.16, 0.08
        s0 = _make_footstep(0.0, 0.0, "L")
        # Next step is 0.05 m away — midpoint is 0.025 m from centre, inside foot
        s1 = _make_footstep(0.05, 0.0, "R")
        phases = check_stability([s0, s1], foot_length=foot_l, foot_width=foot_w)
        single_phases = [p for p in phases if p.kind == "single"]
        assert len(single_phases) > 0
        assert single_phases[0].stable is True

    def test_single_support_com_outside_foot_is_unstable(self):
        """CoM far from the planted foot → unstable single support."""
        foot_l, foot_w = 0.16, 0.08
        s0 = _make_footstep(0.0, 0.0, "L")
        # Huge stride: midpoint is 5 m away from s0, nowhere near the foot
        s1 = _make_footstep(10.0, 0.0, "R")
        phases = check_stability([s0, s1], foot_length=foot_l, foot_width=foot_w)
        single_phases = [p for p in phases if p.kind == "single"]
        assert len(single_phases) > 0
        assert single_phases[0].stable is False

    def test_double_support_com_between_feet_is_stable(self):
        """CoM midpoint between two nearby feet — well inside double-support polygon."""
        foot_l, foot_w = 0.16, 0.08
        s0 = _make_footstep(0.0, -0.1, "L")
        s1 = _make_footstep(0.0, 0.1, "R")
        phases = check_stability([s0, s1], foot_length=foot_l, foot_width=foot_w)
        double_phases = [p for p in phases if p.kind == "double"]
        assert len(double_phases) > 0
        assert double_phases[0].stable is True

    def test_single_footstep_yields_no_phases(self):
        """One footstep has neither a predecessor nor a successor — no phases."""
        phases = check_stability([_make_footstep(0.0, 0.0)])
        assert phases == []

    def test_kind_values(self):
        steps = [_make_footstep(i * 0.3, 0.0) for i in range(4)]
        phases = check_stability(steps)
        for p in phases:
            assert p.kind in ("single", "double")
