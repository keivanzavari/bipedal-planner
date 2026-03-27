"""Tests for the slippery surface scenario."""

import numpy as np
import pytest

from stage1.footstep import Footstep
from stage1.world import SlipperyZone
from stage2.contact_schedule import build_contact_schedule
from stage2.lipm import LIPMParams
from stage2.preview_controller import compute_gains, run_preview_control
from stage3.controllers.lqr import LQRController
from stage3.controllers.mpc import MPCController
from stage3.simulator import _friction_at, run_simulation

_FOOT_LENGTH = 0.16
_FOOT_WIDTH = 0.08
_PARAMS = LIPMParams(h=0.80, g=9.81, dt=0.005)

_FOOTSTEPS = [
    Footstep(side="L", x=0.25, y=0.10, theta=0.0),
    Footstep(side="R", x=0.50, y=-0.10, theta=0.0),
    Footstep(side="L", x=0.75, y=0.10, theta=0.0),
    Footstep(side="R", x=1.00, y=-0.10, theta=0.0),
]


@pytest.fixture(scope="module")
def minimal_traj():
    schedule = build_contact_schedule(_FOOTSTEPS, t_single=0.4, t_double=0.1, dt=_PARAMS.dt)
    gains = compute_gains(_PARAMS, Q_e=1.0, R=1e-6, N_preview=200)
    traj = run_preview_control(schedule, _FOOTSTEPS, gains)
    return traj, schedule


# ---------------------------------------------------------------------------
# SlipperyZone unit tests
# ---------------------------------------------------------------------------


class TestSlipperyZone:
    def test_contains_interior(self):
        z = SlipperyZone(x=1.0, y=0.0, w=2.0, h=3.0)
        assert z.contains(2.0, 1.5)

    def test_contains_edge(self):
        z = SlipperyZone(x=1.0, y=0.0, w=2.0, h=3.0)
        assert z.contains(1.0, 0.0)
        assert z.contains(3.0, 3.0)

    def test_not_contains_outside(self):
        z = SlipperyZone(x=1.0, y=0.0, w=2.0, h=3.0)
        assert not z.contains(0.5, 1.5)
        assert not z.contains(4.0, 1.5)

    def test_friction_at_no_zones(self):
        assert _friction_at(None, 0.0, 0.0) == 1.0
        assert _friction_at([], 0.0, 0.0) == 1.0

    def test_friction_at_inside_zone(self):
        z = SlipperyZone(x=0.0, y=0.0, w=1.0, h=1.0, friction_scale=0.4)
        assert _friction_at([z], 0.5, 0.5) == pytest.approx(0.4)

    def test_friction_at_outside_zone(self):
        z = SlipperyZone(x=5.0, y=5.0, w=1.0, h=1.0, friction_scale=0.4)
        assert _friction_at([z], 0.5, 0.5) == pytest.approx(1.0)

    def test_friction_at_takes_minimum(self):
        z1 = SlipperyZone(x=0.0, y=0.0, w=2.0, h=2.0, friction_scale=0.6)
        z2 = SlipperyZone(x=0.0, y=0.0, w=2.0, h=2.0, friction_scale=0.3)
        assert _friction_at([z1, z2], 1.0, 1.0) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Simulator integration tests
# ---------------------------------------------------------------------------


class TestSimulatorSlippery:
    def test_result_has_friction_array(self, minimal_traj):
        traj, schedule = minimal_traj
        ctrl = LQRController()
        result = run_simulation(traj, schedule, _FOOTSTEPS, _PARAMS, ctrl, noise_sigma=0.0, rng_seed=0)
        assert result.friction.shape == (len(traj.t),)

    def test_friction_all_ones_no_zones(self, minimal_traj):
        traj, schedule = minimal_traj
        ctrl = LQRController()
        result = run_simulation(traj, schedule, _FOOTSTEPS, _PARAMS, ctrl, noise_sigma=0.0, rng_seed=0)
        assert np.all(result.friction == 1.0)

    def test_friction_drops_in_zone(self, minimal_traj):
        traj, schedule = minimal_traj
        zone = SlipperyZone(x=0.0, y=-0.5, w=2.0, h=1.0, friction_scale=0.5)
        ctrl = LQRController()
        result = run_simulation(
            traj, schedule, _FOOTSTEPS, _PARAMS, ctrl, noise_sigma=0.0, rng_seed=0, slippery_zones=[zone]
        )
        # All footsteps are inside the zone (x in [0.25, 1.0], y in [-0.1, 0.1])
        assert np.any(result.friction < 1.0)
        assert np.all(result.friction <= 1.0)

    def test_zmp_bounds_tighter_in_zone(self, minimal_traj):
        traj, schedule = minimal_traj
        zone = SlipperyZone(x=0.0, y=-0.5, w=2.0, h=1.0, friction_scale=0.5)
        ctrl = LQRController()

        result_normal = run_simulation(
            traj,
            schedule,
            _FOOTSTEPS,
            _PARAMS,
            ctrl,
            noise_sigma=0.0,
            foot_length=_FOOT_LENGTH,
            foot_width=_FOOT_WIDTH,
        )
        result_slippery = run_simulation(
            traj,
            schedule,
            _FOOTSTEPS,
            _PARAMS,
            ctrl,
            noise_sigma=0.0,
            slippery_zones=[zone],
            foot_length=_FOOT_LENGTH,
            foot_width=_FOOT_WIDTH,
        )
        # Bounds should be strictly narrower in the slippery region
        width_normal = result_normal.zmp_ub_x - result_normal.zmp_lb_x
        width_slippery = result_slippery.zmp_ub_x - result_slippery.zmp_lb_x
        assert np.all(width_slippery <= width_normal + 1e-9)
        assert np.any(width_slippery < width_normal - 1e-9)

    def test_result_has_zmp_arrays(self, minimal_traj):
        traj, schedule = minimal_traj
        ctrl = LQRController()
        result = run_simulation(traj, schedule, _FOOTSTEPS, _PARAMS, ctrl, noise_sigma=0.0)
        T = len(traj.t)
        for arr in (
            result.zmp_x,
            result.zmp_y,
            result.zmp_lb_x,
            result.zmp_ub_x,
            result.zmp_lb_y,
            result.zmp_ub_y,
            result.friction,
        ):
            assert arr.shape == (T,)

    def test_lqr_finite_with_slippery(self, minimal_traj):
        traj, schedule = minimal_traj
        zone = SlipperyZone(x=0.0, y=-0.5, w=2.0, h=1.0, friction_scale=0.4)
        ctrl = LQRController()
        result = run_simulation(
            traj, schedule, _FOOTSTEPS, _PARAMS, ctrl, noise_sigma=0.005, rng_seed=42, slippery_zones=[zone]
        )
        assert np.all(np.isfinite(result.x))
        assert np.all(np.isfinite(result.y))


# ---------------------------------------------------------------------------
# MPC slippery zone tests
# ---------------------------------------------------------------------------


class TestMPCSlippery:
    def test_mpc_reset_with_slippery_zone(self, minimal_traj):
        traj, schedule = minimal_traj
        zone = SlipperyZone(x=0.0, y=-0.5, w=2.0, h=1.0, friction_scale=0.5)
        ctrl = MPCController(
            footsteps=_FOOTSTEPS,
            foot_length=_FOOT_LENGTH,
            foot_width=_FOOT_WIDTH,
            slippery_zones=[zone],
        )
        ctrl.reset(traj, schedule, _PARAMS)  # should not raise

    def test_mpc_finite_with_slippery(self, minimal_traj):
        traj, schedule = minimal_traj
        zone = SlipperyZone(x=0.0, y=-0.5, w=2.0, h=1.0, friction_scale=0.4)
        ctrl = MPCController(
            footsteps=_FOOTSTEPS,
            foot_length=_FOOT_LENGTH,
            foot_width=_FOOT_WIDTH,
            slippery_zones=[zone],
        )
        result = run_simulation(
            traj, schedule, _FOOTSTEPS, _PARAMS, ctrl, noise_sigma=0.005, rng_seed=42, slippery_zones=[zone]
        )
        assert np.all(np.isfinite(result.x))
        assert np.all(np.isfinite(result.y))
