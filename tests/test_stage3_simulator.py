"""Tests for Stage 3 simulator."""

import numpy as np
import pytest

from stage1.footstep import Footstep
from stage2.contact_schedule import build_contact_schedule
from stage2.lipm import LIPMParams
from stage2.preview_controller import compute_gains, run_preview_control
from stage3.controllers.lqr import LQRController
from stage3.simulator import run_simulation

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulation_setup():
    """4-step walk with LQR controller, zero noise — module-scoped for speed."""
    footsteps = [
        Footstep(side="L", x=0.25, y=0.10, theta=0.0),
        Footstep(side="R", x=0.50, y=-0.10, theta=0.0),
        Footstep(side="L", x=0.75, y=0.10, theta=0.0),
        Footstep(side="R", x=1.00, y=-0.10, theta=0.0),
    ]
    params = LIPMParams(h=0.80, g=9.81, dt=0.005)
    schedule = build_contact_schedule(footsteps, t_single=0.4, t_double=0.1, dt=params.dt)
    gains = compute_gains(params, Q_e=1.0, R=1e-6, N_preview=200)
    traj = run_preview_control(schedule, footsteps, gains)
    result = run_simulation(traj, schedule, footsteps, params, LQRController(), noise_sigma=0.0)
    return traj, result


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestTrackingResultShape:
    def test_all_1d_arrays_length_T(self, simulation_setup):
        traj, result = simulation_setup
        T = len(traj.t)
        for field_name in ("t", "x", "y", "vx", "vy", "ref_x", "ref_y", "err_x", "err_y", "u_x", "u_y"):
            arr = getattr(result, field_name)
            assert arr.shape == (T,), f"{field_name}: expected ({T},), got {arr.shape}"

    def test_grf_arrays_shape(self, simulation_setup):
        traj, result = simulation_setup
        T = len(traj.t)
        assert result.grf_left.shape == (T, 3)
        assert result.grf_right.shape == (T, 3)

    def test_time_matches_reference(self, simulation_setup):
        traj, result = simulation_setup
        np.testing.assert_array_equal(result.t, traj.t)

    def test_ref_matches_traj(self, simulation_setup):
        traj, result = simulation_setup
        np.testing.assert_array_equal(result.ref_x, traj.x)
        np.testing.assert_array_equal(result.ref_y, traj.y)

    def test_error_is_actual_minus_ref(self, simulation_setup):
        _, result = simulation_setup
        np.testing.assert_allclose(result.err_x, result.x - result.ref_x)
        np.testing.assert_allclose(result.err_y, result.y - result.ref_y)


# ---------------------------------------------------------------------------
# LQR tracking accuracy under zero noise
# ---------------------------------------------------------------------------


class TestZeroNoiseLQR:
    def test_max_position_error_under_5cm(self, simulation_setup):
        """With zero noise the LQR should track the reference within 5 cm at all timesteps."""
        _, result = simulation_setup
        max_err_x = float(np.abs(result.err_x).max())
        max_err_y = float(np.abs(result.err_y).max())
        assert max_err_x < 0.05, f"Max x error {max_err_x:.4f} m exceeds 5 cm"
        assert max_err_y < 0.05, f"Max y error {max_err_y:.4f} m exceeds 5 cm"

    def test_all_states_finite(self, simulation_setup):
        _, result = simulation_setup
        for field_name in ("x", "y", "vx", "vy", "u_x", "u_y"):
            assert np.all(np.isfinite(getattr(result, field_name))), f"{field_name} contains non-finite values"


# ---------------------------------------------------------------------------
# Noise reproducibility
# ---------------------------------------------------------------------------


class TestNoise:
    def test_same_seed_same_result(self):
        footsteps = [
            Footstep(side="L", x=0.25, y=0.10, theta=0.0),
            Footstep(side="R", x=0.50, y=-0.10, theta=0.0),
            Footstep(side="L", x=0.75, y=0.10, theta=0.0),
            Footstep(side="R", x=1.00, y=-0.10, theta=0.0),
        ]
        params = LIPMParams(h=0.80, g=9.81, dt=0.005)
        schedule = build_contact_schedule(footsteps, t_single=0.4, t_double=0.1, dt=params.dt)
        gains = compute_gains(params, Q_e=1.0, R=1e-6, N_preview=200)
        traj = run_preview_control(schedule, footsteps, gains)

        r1 = run_simulation(traj, schedule, footsteps, params, LQRController(), noise_sigma=0.01, rng_seed=42)
        r2 = run_simulation(traj, schedule, footsteps, params, LQRController(), noise_sigma=0.01, rng_seed=42)
        np.testing.assert_array_equal(r1.x, r2.x)
        np.testing.assert_array_equal(r1.y, r2.y)

    def test_different_seed_different_result(self):
        footsteps = [
            Footstep(side="L", x=0.25, y=0.10, theta=0.0),
            Footstep(side="R", x=0.50, y=-0.10, theta=0.0),
            Footstep(side="L", x=0.75, y=0.10, theta=0.0),
            Footstep(side="R", x=1.00, y=-0.10, theta=0.0),
        ]
        params = LIPMParams(h=0.80, g=9.81, dt=0.005)
        schedule = build_contact_schedule(footsteps, t_single=0.4, t_double=0.1, dt=params.dt)
        gains = compute_gains(params, Q_e=1.0, R=1e-6, N_preview=200)
        traj = run_preview_control(schedule, footsteps, gains)

        r1 = run_simulation(traj, schedule, footsteps, params, LQRController(), noise_sigma=0.01, rng_seed=0)
        r2 = run_simulation(traj, schedule, footsteps, params, LQRController(), noise_sigma=0.01, rng_seed=1)
        assert not np.array_equal(r1.x, r2.x)
