"""Tests for stage2/traj_optimizer.py — QP-based CoM trajectory optimization."""

import numpy as np
import pytest

from stage1.footstep import Footstep
from stage2.contact_schedule import build_contact_schedule
from stage2.lipm import LIPMParams
from stage2.preview_controller import CoMTrajectory, validate_zmp
from stage2.traj_optimizer import (
    _compute_zmp_bounds,
    build_propagation_matrix,
    free_response,
    precompute_polygons,
    run_trajectory_optimization,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PARAMS = LIPMParams(h=0.80, g=9.81, dt=0.01)
FOOT_L = 0.16
FOOT_W = 0.08


def _make_steps(n: int, dx: float = 0.25, dy: float = 0.1) -> list[Footstep]:
    sides = ["L", "R"]
    return [Footstep(side=sides[i % 2], x=i * dx, y=dy if i % 2 == 0 else -dy, theta=0.0) for i in range(n)]


@pytest.fixture(scope="module")
def small_schedule_and_steps():
    # Small schedule for speed: 6 steps × (0.1+0.02)/0.01 ≈ 72 timesteps
    steps = _make_steps(6)
    schedule = build_contact_schedule(steps, t_single=0.1, t_double=0.02, dt=0.01)
    return schedule, steps


@pytest.fixture(scope="module")
def optimized_traj(small_schedule_and_steps):
    schedule, steps = small_schedule_and_steps
    return run_trajectory_optimization(schedule, steps, PARAMS, FOOT_L, FOOT_W)


# ---------------------------------------------------------------------------
# Unit tests — building blocks
# ---------------------------------------------------------------------------


class TestBuildPropagationMatrix:
    def test_shape(self):
        from stage2.lipm import lipm_matrices

        A, B, C = lipm_matrices(PARAMS)
        T = 50
        P = build_propagation_matrix(A, B, C, T)
        assert P.shape == (T, T)

    def test_strictly_lower_triangular(self):
        from stage2.lipm import lipm_matrices

        A, B, C = lipm_matrices(PARAMS)
        P = build_propagation_matrix(A, B, C, 30)
        # Upper triangle (including diagonal) must be zero
        assert np.allclose(np.triu(P), 0.0)

    def test_first_subdiagonal_equals_CB(self):
        """P[k, k-1] = C @ A^0 @ B = C @ B for all k≥1."""
        from stage2.lipm import lipm_matrices

        A, B, C = lipm_matrices(PARAMS)
        P = build_propagation_matrix(A, B, C, 20)
        CB = float(C @ B)
        assert np.allclose(np.diag(P, -1), CB)


class TestFreeResponse:
    def test_length(self):
        from stage2.lipm import lipm_matrices

        A, B, C = lipm_matrices(PARAMS)
        x0 = np.array([1.0, 0.0, 0.0])
        pf = free_response(A, C, x0, 100)
        assert len(pf) == 100

    def test_zero_initial_state(self):
        from stage2.lipm import lipm_matrices

        A, B, C = lipm_matrices(PARAMS)
        pf = free_response(A, C, np.zeros(3), 50)
        assert np.allclose(pf, 0.0)

    def test_first_value(self):
        """p_free[0] = C @ A^0 @ x0 = C @ x0."""
        from stage2.lipm import lipm_matrices

        A, B, C = lipm_matrices(PARAMS)
        x0 = np.array([2.0, 0.5, 0.1])
        pf = free_response(A, C, x0, 10)
        assert pf[0] == pytest.approx(float(C @ x0))


class TestComputeZmpBounds:
    def test_shape(self, small_schedule_and_steps):
        schedule, steps = small_schedule_and_steps
        lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(schedule, steps, FOOT_L, FOOT_W)
        T = len(schedule.t)
        for arr in (lb_x, ub_x, lb_y, ub_y):
            assert len(arr) == T

    def test_bounds_ordered(self, small_schedule_and_steps):
        schedule, steps = small_schedule_and_steps
        lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(schedule, steps, FOOT_L, FOOT_W)
        assert np.all(lb_x <= ub_x)
        assert np.all(lb_y <= ub_y)

    def test_foot_centre_inside_bounds(self, small_schedule_and_steps):
        """The foot centre must be within the ZMP bounds at every single-support timestep."""
        schedule, steps = small_schedule_and_steps
        lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(schedule, steps, FOOT_L, FOOT_W)
        for k in range(len(schedule.t)):
            if schedule.kind[k] != "single":
                continue
            i = int(schedule.phase[k])
            assert lb_x[k] <= steps[i].x <= ub_x[k]
            assert lb_y[k] <= steps[i].y <= ub_y[k]

    def test_bounds_width_matches_foot(self, small_schedule_and_steps):
        """For theta=0 feet, x-width = foot_length and y-width = foot_width."""
        schedule, steps = small_schedule_and_steps
        lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(schedule, steps, FOOT_L, FOOT_W)
        for k in range(len(schedule.t)):
            if schedule.kind[k] != "single":
                continue
            assert ub_x[k] - lb_x[k] == pytest.approx(FOOT_L, abs=1e-9)
            assert ub_y[k] - lb_y[k] == pytest.approx(FOOT_W, abs=1e-9)
            break


class TestPrecomputePolygons:
    def test_cache_keys(self, small_schedule_and_steps):
        schedule, steps = small_schedule_and_steps
        cache = precompute_polygons(schedule, steps, FOOT_L, FOOT_W)
        # Every (phase, kind) pair in the schedule must be in the cache
        for k in range(len(schedule.t)):
            key = (int(schedule.phase[k]), schedule.kind[k])
            assert key in cache

    def test_halfplane_shape(self, small_schedule_and_steps):
        schedule, steps = small_schedule_and_steps
        cache = precompute_polygons(schedule, steps, FOOT_L, FOOT_W)
        for A_k, b_k in cache.values():
            assert A_k.ndim == 2 and A_k.shape[1] == 2
            assert b_k.ndim == 1 and len(b_k) == A_k.shape[0]

    def test_foot_centre_inside_single_support(self, small_schedule_and_steps):
        """The foot centre must satisfy all half-plane inequalities strictly."""
        schedule, steps = small_schedule_and_steps
        cache = precompute_polygons(schedule, steps, FOOT_L, FOOT_W)
        for k in range(len(schedule.t)):
            if schedule.kind[k] != "single":
                continue
            i = int(schedule.phase[k])
            pt = np.array([steps[i].x, steps[i].y])
            A_k, b_k = cache[(i, "single")]
            assert np.all(A_k @ pt <= b_k + 1e-9), f"Foot centre outside its own polygon at k={k}"
            break


# ---------------------------------------------------------------------------
# Integration tests — run_trajectory_optimization
# ---------------------------------------------------------------------------


class TestRunTrajectoryOptimization:
    def test_output_lengths(self, small_schedule_and_steps, optimized_traj):
        schedule, _ = small_schedule_and_steps
        T = len(schedule.t)
        for arr in (
            optimized_traj.t,
            optimized_traj.x,
            optimized_traj.y,
            optimized_traj.vx,
            optimized_traj.vy,
            optimized_traj.ax,
            optimized_traj.ay,
            optimized_traj.zmp_x,
            optimized_traj.zmp_y,
        ):
            assert len(arr) == T

    def test_output_finite(self, optimized_traj):
        for arr in (
            optimized_traj.x,
            optimized_traj.y,
            optimized_traj.vx,
            optimized_traj.vy,
            optimized_traj.ax,
            optimized_traj.ay,
            optimized_traj.zmp_x,
            optimized_traj.zmp_y,
        ):
            assert np.all(np.isfinite(arr)), "Optimized trajectory contains non-finite values"

    def test_zmp_violations_zero(self, small_schedule_and_steps, optimized_traj):
        """The key guarantee: explicit polygon constraints → 0 ZMP violations."""
        schedule, steps = small_schedule_and_steps
        report = validate_zmp(optimized_traj, schedule, steps, FOOT_L, FOOT_W)
        assert report["zmp_violations"] == 0, (
            f"Expected 0 ZMP violations, got {report['zmp_violations']} ({report['violation_rate'] * 100:.1f}%)"
        )

    def test_zmp_tracking_bounded(self, small_schedule_and_steps, optimized_traj):
        """Mean absolute ZMP tracking error should be below 15 cm."""
        schedule, _ = small_schedule_and_steps
        mae_x = np.mean(np.abs(optimized_traj.zmp_x - schedule.zmp_x))
        mae_y = np.mean(np.abs(optimized_traj.zmp_y - schedule.zmp_y))
        assert mae_x < 0.15, f"ZMP x MAE too large: {mae_x:.4f} m"
        assert mae_y < 0.15, f"ZMP y MAE too large: {mae_y:.4f} m"

    def test_returns_com_trajectory_type(self, optimized_traj):
        assert isinstance(optimized_traj, CoMTrajectory)
