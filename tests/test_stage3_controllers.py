"""Tests for Stage 3 controller contract and registry."""

import numpy as np
import pytest

from stage1.footstep import Footstep
from stage2.contact_schedule import build_contact_schedule
from stage2.lipm import LIPMParams
from stage2.preview_controller import compute_gains, run_preview_control
from stage3.controllers import CONTROLLERS, get_controller
from stage3.controllers.lqr import LQRController
from stage3.controllers.mpc import MPCController

_FOOT_LENGTH = 0.16
_FOOT_WIDTH = 0.08

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minimal_setup():
    """Minimal 4-step walk used across all controller tests."""
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
    return traj, schedule, params, footsteps


def _make_controller(cls, footsteps):
    """Construct a controller instance with the appropriate arguments."""
    if cls is MPCController:
        return cls(footsteps=footsteps, foot_length=_FOOT_LENGTH, foot_width=_FOOT_WIDTH)
    return cls()


# ---------------------------------------------------------------------------
# Shared controller contract — parametrized over all implemented controllers
# ---------------------------------------------------------------------------

ALL_CONTROLLERS = [
    pytest.param(LQRController, id="lqr"),
    pytest.param(MPCController, id="mpc"),
]


@pytest.mark.parametrize("ctrl_cls", ALL_CONTROLLERS)
class TestControllerContract:
    """Behaviour that every Controller implementation must satisfy."""

    def test_reset_does_not_raise(self, ctrl_cls, minimal_setup):
        traj, schedule, params, footsteps = minimal_setup
        ctrl = _make_controller(ctrl_cls, footsteps)
        ctrl.reset(traj, schedule, params)

    def test_step_returns_two_floats(self, ctrl_cls, minimal_setup):
        traj, schedule, params, footsteps = minimal_setup
        ctrl = _make_controller(ctrl_cls, footsteps)
        ctrl.reset(traj, schedule, params)
        state_x = np.array([traj.x[0], traj.vx[0], traj.ax[0]])
        state_y = np.array([traj.y[0], traj.vy[0], traj.ay[0]])
        result = ctrl.step(0, state_x, state_y)
        assert isinstance(result, tuple)
        assert len(result) == 2
        ux, uy = result
        assert isinstance(ux, float)
        assert isinstance(uy, float)

    def test_step_returns_finite_values(self, ctrl_cls, minimal_setup):
        traj, schedule, params, footsteps = minimal_setup
        ctrl = _make_controller(ctrl_cls, footsteps)
        ctrl.reset(traj, schedule, params)
        state_x = np.array([traj.x[0], traj.vx[0], traj.ax[0]])
        state_y = np.array([traj.y[0], traj.vy[0], traj.ay[0]])
        ux, uy = ctrl.step(0, state_x, state_y)
        assert np.isfinite(ux)
        assert np.isfinite(uy)

    def test_step_last_index(self, ctrl_cls, minimal_setup):
        """step() must not raise at the last valid timestep index."""
        traj, schedule, params, footsteps = minimal_setup
        ctrl = _make_controller(ctrl_cls, footsteps)
        ctrl.reset(traj, schedule, params)
        T = len(traj.t)
        state_x = np.array([traj.x[-1], traj.vx[-1], traj.ax[-1]])
        state_y = np.array([traj.y[-1], traj.vy[-1], traj.ay[-1]])
        ux, uy = ctrl.step(T - 1, state_x, state_y)
        assert np.isfinite(ux) and np.isfinite(uy)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestControllerRegistry:
    def test_controllers_dict_contains_lqr(self):
        assert "lqr" in CONTROLLERS

    def test_controllers_dict_contains_mpc(self):
        assert "mpc" in CONTROLLERS

    def test_get_controller_returns_lqr(self):
        ctrl = get_controller("lqr")
        assert isinstance(ctrl, LQRController)

    def test_get_controller_raises_on_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown controller"):
            get_controller("nonexistent_controller")

    def test_get_controller_passes_kwargs_lqr(self):
        ctrl = get_controller("lqr", Q_e=2.0, R=1e-5)
        assert isinstance(ctrl, LQRController)
