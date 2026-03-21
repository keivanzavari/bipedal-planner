"""Tests for stage2/preview_controller.py — LQR gains and CoM trajectory generation."""

import numpy as np
import pytest

from stage1.footstep import Footstep
from stage2.lipm import LIPMParams
from stage2.contact_schedule import build_contact_schedule
from stage2.preview_controller import (
    CoMTrajectory,
    PreviewGains,
    _run_1d,
    compute_gains,
    run_preview_control,
    validate_zmp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_params() -> LIPMParams:
    return LIPMParams(h=0.80, g=9.81, dt=0.005)


@pytest.fixture(scope="module")
def gains(default_params: LIPMParams) -> PreviewGains:
    return compute_gains(default_params, N_preview=100)


def _make_steps(n: int, dx: float = 0.25, dy: float = 0.1) -> list[Footstep]:
    sides = ["L", "R"]
    return [
        Footstep(side=sides[i % 2], x=i * dx, y=dy if i % 2 == 0 else -dy, theta=0.0)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# compute_gains
# ---------------------------------------------------------------------------


class TestComputeGains:
    def test_K_length(self, gains: PreviewGains):
        """State-feedback gain K has length n+1 = 4 (augmented state)."""
        assert len(gains.K) == 4

    def test_Gp_length(self, gains: PreviewGains):
        assert len(gains.Gp) == 100

    def test_preview_gains_decay(self, gains: PreviewGains):
        """Preview gains should decay: |Gp[0]| ≥ |Gp[-1]|."""
        assert abs(gains.Gp[0]) >= abs(gains.Gp[-1])

    def test_preview_gains_positive(self, gains: PreviewGains):
        """All preview gains should be non-negative (controller attracts toward reference)."""
        assert np.all(gains.Gp >= 0)

    def test_stored_matrices_shapes(self, gains: PreviewGains):
        assert gains.A.shape == (3, 3)
        assert gains.B.shape == (3,)
        assert gains.C.shape == (3,)

    def test_custom_N_preview(self, default_params: LIPMParams):
        g = compute_gains(default_params, N_preview=50)
        assert len(g.Gp) == 50


# ---------------------------------------------------------------------------
# _run_1d
# ---------------------------------------------------------------------------


class TestRun1D:
    def test_output_lengths(self, gains: PreviewGains):
        T = 200
        zmp_ref = np.ones(T) * 1.0
        pos, vel, acc, zmp = _run_1d(zmp_ref, com_init=1.0, gains=gains)
        assert len(pos) == T
        assert len(vel) == T
        assert len(acc) == T
        assert len(zmp) == T

    def test_constant_reference_convergence(self, default_params: LIPMParams):
        """CoM should converge close to a constant ZMP reference over enough time."""
        gains_long = compute_gains(default_params, N_preview=200)
        T = 4000  # 20 s at dt=0.005
        ref = 2.0
        zmp_ref = np.full(T, ref)
        pos, _, _, _ = _run_1d(zmp_ref, com_init=ref, gains=gains_long)
        # In steady state the last 10 % of samples should be close to ref
        assert np.mean(np.abs(pos[-T // 10 :] - ref)) < 0.05

    def test_zero_reference_stays_near_zero(self, gains: PreviewGains):
        T = 500
        zmp_ref = np.zeros(T)
        pos, vel, acc, zmp = _run_1d(zmp_ref, com_init=0.0, gains=gains)
        assert np.all(np.abs(pos) < 1.0), "CoM drifted far from zero with zero reference"

    def test_output_finite(self, gains: PreviewGains):
        T = 300
        zmp_ref = np.ones(T) * 0.5
        pos, vel, acc, zmp = _run_1d(zmp_ref, com_init=0.5, gains=gains)
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(vel))
        assert np.all(np.isfinite(acc))
        assert np.all(np.isfinite(zmp))


# ---------------------------------------------------------------------------
# run_preview_control
# ---------------------------------------------------------------------------


class TestRunPreviewControl:
    @pytest.fixture(scope="class")
    def schedule_and_steps(self):
        steps = _make_steps(6)
        schedule = build_contact_schedule(steps, t_single=0.4, t_double=0.1, dt=0.005)
        return schedule, steps

    @pytest.fixture(scope="class")
    def trajectory(self, schedule_and_steps, gains):
        schedule, steps = schedule_and_steps
        return run_preview_control(schedule, steps, gains)

    def test_trajectory_length_matches_schedule(self, schedule_and_steps, trajectory: CoMTrajectory):
        schedule, _ = schedule_and_steps
        T = len(schedule.t)
        assert len(trajectory.t) == T
        assert len(trajectory.x) == T
        assert len(trajectory.y) == T
        assert len(trajectory.vx) == T
        assert len(trajectory.vy) == T
        assert len(trajectory.zmp_x) == T
        assert len(trajectory.zmp_y) == T

    def test_trajectory_finite(self, trajectory: CoMTrajectory):
        for arr in (trajectory.x, trajectory.y, trajectory.vx, trajectory.vy,
                    trajectory.ax, trajectory.ay, trajectory.zmp_x, trajectory.zmp_y):
            assert np.all(np.isfinite(arr)), "Trajectory contains non-finite values"

    def test_zmp_tracking_error_bounded(self, schedule_and_steps, trajectory: CoMTrajectory):
        """Mean absolute ZMP tracking error should be below 15 cm.

        The 6-step schedule is short, so startup transients dominate the mean.
        15 cm is a conservative bound that the preview controller easily meets
        in practice; steady-state error is an order of magnitude smaller.
        """
        schedule, _ = schedule_and_steps
        mae_x = np.mean(np.abs(trajectory.zmp_x - schedule.zmp_x))
        mae_y = np.mean(np.abs(trajectory.zmp_y - schedule.zmp_y))
        assert mae_x < 0.15, f"ZMP x MAE too large: {mae_x:.4f} m"
        assert mae_y < 0.15, f"ZMP y MAE too large: {mae_y:.4f} m"


# ---------------------------------------------------------------------------
# validate_zmp
# ---------------------------------------------------------------------------


class TestValidateZMP:
    def test_returns_dict_with_expected_keys(self):
        steps = _make_steps(4)
        schedule = build_contact_schedule(steps, t_single=0.4, t_double=0.1, dt=0.005)
        params = LIPMParams()
        g = compute_gains(params, N_preview=100)
        traj = run_preview_control(schedule, steps, g)
        result = validate_zmp(traj, schedule, steps)
        assert "total_steps" in result
        assert "zmp_violations" in result
        assert "violation_rate" in result
        assert "first_failures" in result

    def test_violation_rate_in_range(self):
        steps = _make_steps(4)
        schedule = build_contact_schedule(steps, t_single=0.4, t_double=0.1, dt=0.005)
        g = compute_gains(LIPMParams(), N_preview=100)
        traj = run_preview_control(schedule, steps, g)
        result = validate_zmp(traj, schedule, steps)
        assert 0.0 <= result["violation_rate"] <= 1.0
