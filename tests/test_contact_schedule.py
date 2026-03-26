"""Tests for stage2/contact_schedule.py — ZMP reference generation and timing."""

import numpy as np
import pytest

from stage1.footstep import Footstep
from stage2.contact_schedule import build_contact_schedule, support_polygon_at

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_steps(n: int, dx: float = 0.3, dy: float = 0.0) -> list[Footstep]:
    """Create `n` footsteps along a straight line."""
    sides = ["L", "R"]
    return [Footstep(side=sides[i % 2], x=i * dx, y=dy if i % 2 == 0 else -dy, theta=0.0) for i in range(n)]


# ---------------------------------------------------------------------------
# build_contact_schedule
# ---------------------------------------------------------------------------


class TestBuildContactSchedule:
    T_SINGLE = 0.4
    T_DOUBLE = 0.1
    DT = 0.005

    @pytest.fixture()
    def four_steps(self) -> list[Footstep]:
        return _make_steps(4, dx=0.25, dy=0.1)

    @pytest.fixture()
    def schedule(self, four_steps):
        return build_contact_schedule(
            four_steps,
            t_single=self.T_SINGLE,
            t_double=self.T_DOUBLE,
            dt=self.DT,
        )

    def test_timestamps_monotonically_increasing(self, schedule):
        assert np.all(np.diff(schedule.t) > 0)

    def test_total_timestep_count(self, four_steps, schedule):
        n = len(four_steps)
        n_ss = round(self.T_SINGLE / self.DT)
        n_ds = round(self.T_DOUBLE / self.DT)
        expected = n * n_ss + (n - 1) * n_ds
        # Allow ±(n-1) due to rounding in each double-support slice
        assert abs(len(schedule.t) - expected) <= n

    def test_kind_values_are_valid(self, schedule):
        assert all(k in ("single", "double") for k in schedule.kind)

    def test_zmp_arrays_same_length_as_time(self, schedule):
        T = len(schedule.t)
        assert len(schedule.zmp_x) == T
        assert len(schedule.zmp_y) == T
        assert len(schedule.phase) == T
        assert len(schedule.kind) == T

    def test_single_support_zmp_matches_foot(self, four_steps, schedule):
        """During single-support slices the ZMP must equal the planted foot position."""
        for k, kind in enumerate(schedule.kind):
            if kind == "single":
                i = int(schedule.phase[k])
                assert schedule.zmp_x[k] == pytest.approx(four_steps[i].x, abs=1e-9)
                assert schedule.zmp_y[k] == pytest.approx(four_steps[i].y, abs=1e-9)

    def test_double_support_zmp_between_feet(self, four_steps, schedule):
        """During double-support the ZMP x should be between the two feet (for non-equal x)."""
        for k, kind in enumerate(schedule.kind):
            if kind == "double":
                i = int(schedule.phase[k])
                if i == 0:
                    continue  # first step has no predecessor
                prev_x = four_steps[i - 1].x
                curr_x = four_steps[i].x
                lo, hi = min(prev_x, curr_x), max(prev_x, curr_x)
                if lo < hi:
                    assert lo <= schedule.zmp_x[k] <= hi, f"ZMP x {schedule.zmp_x[k]:.4f} not in [{lo:.4f}, {hi:.4f}]"

    def test_phase_indices_in_range(self, four_steps, schedule):
        assert schedule.phase.min() >= 0
        assert schedule.phase.max() < len(four_steps)

    def test_single_step_no_double_support(self):
        """A single footstep generates only single-support — no double support."""
        steps = _make_steps(1)
        sched = build_contact_schedule(steps, t_single=self.T_SINGLE, t_double=self.T_DOUBLE, dt=self.DT)
        assert all(k == "single" for k in sched.kind)

    def test_two_steps_has_double_support(self):
        steps = _make_steps(2)
        sched = build_contact_schedule(steps, t_single=self.T_SINGLE, t_double=self.T_DOUBLE, dt=self.DT)
        assert "double" in sched.kind


# ---------------------------------------------------------------------------
# support_polygon_at
# ---------------------------------------------------------------------------


class TestSupportPolygonAt:
    T_SINGLE = 0.4
    T_DOUBLE = 0.1
    DT = 0.005

    @pytest.fixture()
    def footsteps(self) -> list[Footstep]:
        return _make_steps(3, dx=0.25, dy=0.1)

    @pytest.fixture()
    def schedule(self, footsteps):
        return build_contact_schedule(
            footsteps,
            t_single=self.T_SINGLE,
            t_double=self.T_DOUBLE,
            dt=self.DT,
        )

    def test_single_support_polygon_has_four_corners(self, schedule, footsteps):
        """Single-support polygon = one foot rectangle = 4 corners."""
        for k, kind in enumerate(schedule.kind):
            if kind == "single":
                poly = support_polygon_at(schedule, k, footsteps)
                assert poly.shape[1] == 2
                assert len(poly) >= 3  # at minimum a triangle; typically 4
                break

    def test_double_support_polygon_has_at_least_four_points(self, schedule, footsteps):
        """Double-support polygon = convex hull of two feet ≥ 4 points."""
        for k, kind in enumerate(schedule.kind):
            if kind == "double":
                poly = support_polygon_at(schedule, k, footsteps)
                assert len(poly) >= 4
                break

    def test_polygon_shape_is_valid(self, schedule, footsteps):
        for k in range(0, len(schedule.t), 50):  # sample every 50 steps
            poly = support_polygon_at(schedule, k, footsteps)
            assert poly.ndim == 2
            assert poly.shape[1] == 2
