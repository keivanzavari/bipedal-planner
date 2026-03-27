"""Tests for robot.config and robot.kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from robot.config import DEFAULT_ROBOT, RobotConfig
from robot.kinematics import (
    active_feet_at,
    compute_phase_progress,
    two_link_knee,
)
from stage1.footstep import Footstep
from stage2.contact_schedule import build_contact_schedule


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg() -> RobotConfig:
    return DEFAULT_ROBOT


@pytest.fixture()
def four_steps() -> list[Footstep]:
    """Four alternating footsteps walking in the +x direction."""
    return [
        Footstep(side="L", x=0.25, y=0.10, theta=0.0),
        Footstep(side="R", x=0.50, y=-0.10, theta=0.0),
        Footstep(side="L", x=0.75, y=0.10, theta=0.0),
        Footstep(side="R", x=1.00, y=-0.10, theta=0.0),
    ]


@pytest.fixture(scope="module")
def schedule_and_steps():
    """Module-scoped schedule + footstep list (avoids rebuilding in every test)."""
    footsteps = [
        Footstep(side="L", x=0.25, y=0.10, theta=0.0),
        Footstep(side="R", x=0.50, y=-0.10, theta=0.0),
        Footstep(side="L", x=0.75, y=0.10, theta=0.0),
        Footstep(side="R", x=1.00, y=-0.10, theta=0.0),
    ]
    schedule = build_contact_schedule(footsteps, t_single=0.4, t_double=0.1, dt=0.005)
    return schedule, footsteps


@pytest.fixture(scope="module")
def phase_alpha(schedule_and_steps):
    schedule, _ = schedule_and_steps
    return compute_phase_progress(schedule)


# ---------------------------------------------------------------------------
# RobotConfig
# ---------------------------------------------------------------------------


class TestRobotConfig:
    def test_default_robot_is_robot_config(self):
        assert isinstance(DEFAULT_ROBOT, RobotConfig)

    def test_leg_lengths_reach_ground(self, cfg):
        """upper + lower leg must exceed com_height - pelvis_offset so the leg
        can always reach the ground (legs are slightly bent at neutral stance)."""
        hip_z = 0.80 - cfg.pelvis_offset  # using default LIPM com_height
        max_reach = cfg.upper_leg + cfg.lower_leg
        assert max_reach > hip_z

    def test_custom_config_overrides(self):
        c = RobotConfig(upper_leg=0.35, lower_leg=0.35)
        assert c.upper_leg == pytest.approx(0.35)
        assert c.lower_leg == pytest.approx(0.35)
        # Other fields retain defaults
        assert c.hip_width == pytest.approx(DEFAULT_ROBOT.hip_width)


# ---------------------------------------------------------------------------
# two_link_knee
# ---------------------------------------------------------------------------


class TestTwoLinkKnee:
    def test_upper_leg_length_preserved(self, cfg):
        hip = np.array([0.0, 0.0, 0.75])
        foot = np.array([0.0, 0.0, 0.0])
        knee = two_link_knee(hip, foot, cfg.upper_leg, cfg.lower_leg)
        assert np.linalg.norm(knee - hip) == pytest.approx(cfg.upper_leg, abs=1e-6)

    def test_lower_leg_length_preserved(self, cfg):
        hip = np.array([0.0, 0.0, 0.75])
        foot = np.array([0.0, 0.0, 0.0])
        knee = two_link_knee(hip, foot, cfg.upper_leg, cfg.lower_leg)
        assert np.linalg.norm(knee - foot) == pytest.approx(cfg.lower_leg, abs=1e-6)

    def test_knee_above_foot(self, cfg):
        """Knee z-coordinate must be higher than the foot z-coordinate."""
        hip = np.array([0.0, 0.0, 0.75])
        foot = np.array([0.0, 0.0, 0.0])
        knee = two_link_knee(hip, foot, cfg.upper_leg, cfg.lower_leg)
        assert knee[2] > foot[2]

    def test_knee_below_hip(self, cfg):
        """Knee z-coordinate must be below the hip."""
        hip = np.array([0.0, 0.0, 0.75])
        foot = np.array([0.0, 0.0, 0.0])
        knee = two_link_knee(hip, foot, cfg.upper_leg, cfg.lower_leg)
        assert knee[2] < hip[2]

    def test_vertical_leg_knee_goes_forward(self, cfg):
        """With a vertical leg and forward=[1,0,0], knee x must be > hip x."""
        hip = np.array([0.0, 0.0, 0.75])
        foot = np.array([0.0, 0.0, 0.0])
        knee = two_link_knee(hip, foot, cfg.upper_leg, cfg.lower_leg,
                             forward=np.array([1.0, 0.0, 0.0]))
        assert knee[0] > hip[0]

    def test_tilted_leg_preserves_lengths(self, cfg):
        """Leg-length invariant holds when foot is offset horizontally."""
        hip = np.array([0.0, 0.1, 0.75])
        foot = np.array([0.15, -0.05, 0.0])
        knee = two_link_knee(hip, foot, cfg.upper_leg, cfg.lower_leg)
        assert np.linalg.norm(knee - hip) == pytest.approx(cfg.upper_leg, abs=1e-6)
        assert np.linalg.norm(knee - foot) == pytest.approx(cfg.lower_leg, abs=1e-6)

    def test_clamped_reach_does_not_raise(self, cfg):
        """When foot is farther than l1+l2, the function clamps and returns a point."""
        hip = np.array([0.0, 0.0, 0.0])
        foot = np.array([2.0, 0.0, 0.0])   # well beyond max reach of 0.8 m
        knee = two_link_knee(hip, foot, cfg.upper_leg, cfg.lower_leg)
        assert knee.shape == (3,)
        assert np.all(np.isfinite(knee))

    def test_symmetric_legs_mirror_in_y(self, cfg):
        """Left and right hips at ±y should produce mirrored knee positions."""
        hip_l = np.array([0.0,  0.10, 0.75])
        hip_r = np.array([0.0, -0.10, 0.75])
        foot_l = np.array([0.1,  0.10, 0.0])
        foot_r = np.array([0.1, -0.10, 0.0])
        fwd = np.array([1.0, 0.0, 0.0])
        knee_l = two_link_knee(hip_l, foot_l, cfg.upper_leg, cfg.lower_leg, fwd)
        knee_r = two_link_knee(hip_r, foot_r, cfg.upper_leg, cfg.lower_leg, fwd)
        assert knee_l[0] == pytest.approx(knee_r[0], abs=1e-6)   # same x
        assert knee_l[1] == pytest.approx(-knee_r[1], abs=1e-6)  # mirrored y
        assert knee_l[2] == pytest.approx(knee_r[2], abs=1e-6)   # same z


# ---------------------------------------------------------------------------
# compute_phase_progress
# ---------------------------------------------------------------------------


class TestComputePhaseProgress:
    def test_output_shape(self, schedule_and_steps, phase_alpha):
        schedule, _ = schedule_and_steps
        assert phase_alpha.shape == (len(schedule.t),)

    def test_values_in_unit_interval(self, phase_alpha):
        assert np.all(phase_alpha >= 0.0)
        assert np.all(phase_alpha < 1.0)

    def test_each_phase_starts_at_zero(self, schedule_and_steps, phase_alpha):
        """α must be 0 at the first timestep of every phase segment."""
        schedule, _ = schedule_and_steps
        T = len(schedule.t)
        for k in range(T):
            if k == 0:
                assert phase_alpha[k] == pytest.approx(0.0)
            else:
                prev_phase = int(schedule.phase[k - 1])
                prev_kind = schedule.kind[k - 1]
                curr_phase = int(schedule.phase[k])
                curr_kind = schedule.kind[k]
                if (curr_phase, curr_kind) != (prev_phase, prev_kind):
                    assert phase_alpha[k] == pytest.approx(0.0), (
                        f"Phase transition at k={k}: α should reset to 0"
                    )

    def test_alpha_monotone_within_phase(self, schedule_and_steps, phase_alpha):
        """α must be strictly increasing within each phase segment."""
        schedule, _ = schedule_and_steps
        T = len(schedule.t)
        for k in range(1, T):
            same_phase = (
                int(schedule.phase[k]) == int(schedule.phase[k - 1])
                and schedule.kind[k] == schedule.kind[k - 1]
            )
            if same_phase:
                assert phase_alpha[k] > phase_alpha[k - 1]


# ---------------------------------------------------------------------------
# active_feet_at
# ---------------------------------------------------------------------------


class TestActiveFeetAt:
    def test_output_shapes(self, schedule_and_steps, phase_alpha, cfg):
        schedule, footsteps = schedule_and_steps
        left, right = active_feet_at(0, footsteps, schedule, cfg, phase_alpha)
        assert left.shape == (3,)
        assert right.shape == (3,)

    def test_stance_foot_at_zero_z_during_single_support(
        self, schedule_and_steps, phase_alpha, cfg
    ):
        """The planted foot must be on the ground (z=0) during single support."""
        schedule, footsteps = schedule_and_steps
        T = len(schedule.t)
        for k in range(0, T, 20):
            if schedule.kind[k] != "single":
                continue
            left, right = active_feet_at(k, footsteps, schedule, cfg, phase_alpha)
            stance_side = footsteps[int(schedule.phase[k])].side
            planted = left if stance_side == "L" else right
            assert planted[2] == pytest.approx(0.0, abs=1e-9)

    def test_both_feet_grounded_in_double_support(
        self, schedule_and_steps, phase_alpha, cfg
    ):
        """During double support both feet must be at z=0."""
        schedule, footsteps = schedule_and_steps
        T = len(schedule.t)
        for k in range(0, T, 10):
            if schedule.kind[k] != "double":
                continue
            left, right = active_feet_at(k, footsteps, schedule, cfg, phase_alpha)
            assert left[2] == pytest.approx(0.0, abs=1e-9)
            assert right[2] == pytest.approx(0.0, abs=1e-9)

    def test_swing_foot_lifts_off_ground(self, schedule_and_steps, phase_alpha, cfg):
        """Swing foot z must be > 0 somewhere near mid-phase (parabolic arc)."""
        schedule, footsteps = schedule_and_steps
        T = len(schedule.t)
        found_lift = False
        for k in range(T):
            if schedule.kind[k] != "single":
                continue
            alpha = phase_alpha[k]
            if not (0.3 < alpha < 0.7):
                continue
            left, right = active_feet_at(k, footsteps, schedule, cfg, phase_alpha)
            stance_side = footsteps[int(schedule.phase[k])].side
            swing = right if stance_side == "L" else left
            if swing[2] > 1e-3:
                found_lift = True
                break
        assert found_lift, "Swing foot never lifted off the ground near mid-phase"

    def test_swing_foot_grounded_at_phase_start(
        self, schedule_and_steps, phase_alpha, cfg
    ):
        """At the very first timestep of single support, swing foot z ≈ 0."""
        schedule, footsteps = schedule_and_steps
        T = len(schedule.t)
        for k in range(T):
            if schedule.kind[k] != "single":
                continue
            if phase_alpha[k] != pytest.approx(0.0):
                continue
            left, right = active_feet_at(k, footsteps, schedule, cfg, phase_alpha)
            stance_side = footsteps[int(schedule.phase[k])].side
            swing = right if stance_side == "L" else left
            assert swing[2] == pytest.approx(0.0, abs=1e-6)
            break

    def test_stance_foot_xy_matches_footstep(self, schedule_and_steps, phase_alpha, cfg):
        """Planted foot (x, y) must match the corresponding Footstep exactly."""
        schedule, footsteps = schedule_and_steps
        T = len(schedule.t)
        for k in range(0, T, 20):
            if schedule.kind[k] != "single":
                continue
            i = int(schedule.phase[k])
            fs = footsteps[i]
            left, right = active_feet_at(k, footsteps, schedule, cfg, phase_alpha)
            planted = left if fs.side == "L" else right
            assert planted[0] == pytest.approx(fs.x, abs=1e-9)
            assert planted[1] == pytest.approx(fs.y, abs=1e-9)

    def test_left_right_assignment_consistent_with_side(
        self, schedule_and_steps, phase_alpha, cfg
    ):
        """The foot assigned to 'left' must always come from an 'L' footstep
        and vice versa — checked at double-support frames where both are planted."""
        schedule, footsteps = schedule_and_steps
        T = len(schedule.t)
        for k in range(0, T, 10):
            if schedule.kind[k] != "double":
                continue
            i = int(schedule.phase[k])
            fs_curr = footsteps[i]
            fs_prev = footsteps[i - 1] if i > 0 else fs_curr
            left, right = active_feet_at(k, footsteps, schedule, cfg, phase_alpha)
            for fs in (fs_curr, fs_prev):
                if fs.side == "L":
                    assert left[0] == pytest.approx(fs.x, abs=1e-9)
                else:
                    assert right[0] == pytest.approx(fs.x, abs=1e-9)
            break  # one double-support frame is sufficient
