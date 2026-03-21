"""Tests for stage2/lipm.py — discrete LIPM matrices and ZMP formula."""

import numpy as np
import pytest

from stage2.lipm import LIPMParams, lipm_matrices, zmp_from_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_params() -> LIPMParams:
    return LIPMParams(h=0.80, g=9.81, dt=0.005)


# ---------------------------------------------------------------------------
# Matrix structure
# ---------------------------------------------------------------------------


class TestLIPMMatrices:
    def test_shapes(self, default_params: LIPMParams):
        A, B, C = lipm_matrices(default_params)
        assert A.shape == (3, 3)
        assert B.shape == (3,)
        assert C.shape == (3,)

    def test_A_top_left_identity_block(self, default_params: LIPMParams):
        """A[0,0] == 1 (position is integrated from velocity)."""
        A, _, _ = lipm_matrices(default_params)
        assert A[0, 0] == pytest.approx(1.0)
        assert A[1, 1] == pytest.approx(1.0)
        assert A[2, 2] == pytest.approx(1.0)

    def test_A_pos_row(self, default_params: LIPMParams):
        """Row 0 of A: [1, dt, 0.5*dt²]."""
        dt = default_params.dt
        A, _, _ = lipm_matrices(default_params)
        np.testing.assert_allclose(A[0], [1.0, dt, 0.5 * dt**2], rtol=1e-12)

    def test_A_vel_row(self, default_params: LIPMParams):
        """Row 1 of A: [0, 1, dt]."""
        dt = default_params.dt
        A, _, _ = lipm_matrices(default_params)
        np.testing.assert_allclose(A[1], [0.0, 1.0, dt], rtol=1e-12)

    def test_B_first_element(self, default_params: LIPMParams):
        """B[0] = dt³/6."""
        dt = default_params.dt
        _, B, _ = lipm_matrices(default_params)
        assert B[0] == pytest.approx(dt**3 / 6.0, rel=1e-12)

    def test_B_second_element(self, default_params: LIPMParams):
        """B[1] = dt²/2."""
        dt = default_params.dt
        _, B, _ = lipm_matrices(default_params)
        assert B[1] == pytest.approx(dt**2 / 2.0, rel=1e-12)

    def test_B_third_element(self, default_params: LIPMParams):
        """B[2] = dt."""
        dt = default_params.dt
        _, B, _ = lipm_matrices(default_params)
        assert B[2] == pytest.approx(dt, rel=1e-12)

    def test_C_zmp_formula(self, default_params: LIPMParams):
        """C = [1, 0, -h/g]."""
        h, g = default_params.h, default_params.g
        _, _, C = lipm_matrices(default_params)
        np.testing.assert_allclose(C, [1.0, 0.0, -h / g], rtol=1e-12)


# ---------------------------------------------------------------------------
# ZMP computations
# ---------------------------------------------------------------------------


class TestZMPFormula:
    def test_zero_velocity_and_acceleration(self, default_params: LIPMParams):
        """ZMP = CoM position when velocity and acceleration are zero."""
        _, _, C = lipm_matrices(default_params)
        state = np.array([3.0, 0.0, 0.0])
        assert zmp_from_state(state, C) == pytest.approx(3.0)

    def test_pure_acceleration(self, default_params: LIPMParams):
        """ZMP = -h/g * acc when position and velocity are zero."""
        h, g = default_params.h, default_params.g
        _, _, C = lipm_matrices(default_params)
        acc = 2.0
        state = np.array([0.0, 0.0, acc])
        assert zmp_from_state(state, C) == pytest.approx(-h / g * acc)

    def test_combined_state(self, default_params: LIPMParams):
        h, g = default_params.h, default_params.g
        _, _, C = lipm_matrices(default_params)
        pos, vel, acc = 1.0, 0.5, 1.0
        state = np.array([pos, vel, acc])
        expected = pos - (h / g) * acc
        assert zmp_from_state(state, C) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Discrete dynamics sanity check
# ---------------------------------------------------------------------------


class TestDiscreteDynamics:
    def test_zero_state_zero_input_stays_zero(self, default_params: LIPMParams):
        A, B, _ = lipm_matrices(default_params)
        state = np.zeros(3)
        next_state = A @ state + B * 0.0
        np.testing.assert_allclose(next_state, np.zeros(3), atol=1e-15)

    def test_constant_position_zero_vel_acc_zero_input(self, default_params: LIPMParams):
        """[pos, 0, 0] with u=0 → [pos, 0, 0] (no velocity or acceleration)."""
        A, B, _ = lipm_matrices(default_params)
        state = np.array([5.0, 0.0, 0.0])
        next_state = A @ state + B * 0.0
        np.testing.assert_allclose(next_state, state, atol=1e-12)

    def test_jerk_input_increases_acceleration(self, default_params: LIPMParams):
        """Positive jerk input must increase the acceleration state component."""
        A, B, _ = lipm_matrices(default_params)
        state = np.zeros(3)
        u = 1.0  # positive jerk
        next_state = A @ state + B * u
        assert next_state[2] > 0
