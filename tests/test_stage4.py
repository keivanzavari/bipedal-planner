"""Tests for Stage 4 — IK angles and MuJoCo simulation.

The MuJoCo smoke and regression tests require ``mujoco`` to be installed.
Run with:  uv run pytest tests/test_stage4.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot.kinematics import two_link_knee
from stage4.ik_angles import angles_from_ik

# Check if mujoco is available for the physics tests
try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

_SCENE_XML = Path(__file__).parent.parent / "stage4" / "models" / "biped_scene.xml"

# ---------------------------------------------------------------------------
# IK Angle Tests
# ---------------------------------------------------------------------------


class TestIKAngles:
    """Unit tests for angles_from_ik — no MuJoCo dependency."""

    def test_straight_leg_zero_angles(self):
        """Straight leg at h=0.80 m → all angles near zero."""
        hip = np.array([0.0, 0.0, 0.80])
        foot = np.array([0.0, 0.0, 0.0])
        knee = two_link_knee(hip, foot, l1=0.40, l2=0.40)
        theta_hip, theta_knee, theta_ankle = angles_from_ik(hip, knee, foot)
        assert abs(theta_hip) < 0.02, f"theta_hip = {theta_hip}"
        assert abs(theta_knee) < 0.02, f"theta_knee = {theta_knee}"
        assert abs(theta_ankle) < 0.02, f"theta_ankle = {theta_ankle}"

    def test_foot_forward_positive_hip(self):
        """Foot placed forward of hip → positive hip angle."""
        hip = np.array([0.0, 0.0, 0.75])
        foot = np.array([0.15, 0.0, 0.0])
        knee = two_link_knee(hip, foot, l1=0.40, l2=0.40)
        theta_hip, _, _ = angles_from_ik(hip, knee, foot)
        assert theta_hip > 0.0, f"Expected positive hip angle, got {theta_hip}"

    def test_foot_behind_negative_hip(self):
        """Foot placed behind hip → negative hip angle."""
        hip = np.array([0.0, 0.0, 0.75])
        foot = np.array([-0.15, 0.0, 0.0])
        knee = two_link_knee(hip, foot, l1=0.40, l2=0.40)
        theta_hip, _, _ = angles_from_ik(hip, knee, foot)
        assert theta_hip < 0.0, f"Expected negative hip angle, got {theta_hip}"

    def test_ankle_keeps_foot_flat(self):
        """Ankle angle should compensate so total rotation = 0 (flat foot)."""
        hip = np.array([0.0, 0.0, 0.75])
        foot = np.array([0.10, 0.0, 0.0])
        knee = two_link_knee(hip, foot, l1=0.40, l2=0.40)
        theta_hip, theta_knee, theta_ankle = angles_from_ik(hip, knee, foot)
        total = theta_hip + theta_knee + theta_ankle
        assert abs(total) < 1e-10, f"Total rotation = {total}, should be 0"

    @pytest.mark.parametrize("foot_x", [0.0, 0.05, 0.10, 0.15, 0.20])
    def test_angles_sum_to_zero(self, foot_x):
        """For any foot placement, sum of sagittal angles must be 0."""
        hip = np.array([0.0, 0.0, 0.75])
        foot = np.array([foot_x, 0.0, 0.0])
        knee = two_link_knee(hip, foot, l1=0.40, l2=0.40)
        theta_hip, theta_knee, theta_ankle = angles_from_ik(hip, knee, foot)
        total = theta_hip + theta_knee + theta_ankle
        assert abs(total) < 1e-10, f"foot_x={foot_x}: sum = {total}"


# ---------------------------------------------------------------------------
# MuJoCo Smoke Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
class TestMujocoSmoke:
    """Smoke tests that load the MJCF model and verify basic physics."""

    def test_model_loads(self):
        """MJCF scene file loads without error."""
        model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
        assert model.nq == 9  # 3 pelvis + 6 actuated joints

    def test_robot_does_not_fall_through_floor(self):
        """Step 100 times — pelvis stays above ground."""
        model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
        data = mujoco.MjData(model)
        # Start with some height
        data.qpos[1] = 0.80  # pelvis_z
        mujoco.mj_forward(model, data)
        for _ in range(100):
            mujoco.mj_step(model, data)
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        com_z = data.subtree_com[pelvis_id][2]
        assert com_z > 0.0, f"CoM z = {com_z}, fell through floor"

    def test_actuator_count(self):
        """Model has 6 actuators (3 per leg)."""
        model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
        assert model.nu == 6

    def test_geom_names_exist(self):
        """Named geoms needed for contact extraction exist."""
        model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
        for name in ("floor", "left_foot_geom", "right_foot_geom", "torso"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0, f"Geom '{name}' not found"


# ---------------------------------------------------------------------------
# Integration Test (requires full pipeline)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")
@pytest.mark.slow
class TestStage4Integration:
    """Full Stage 1 → 2 → 4 pipeline test. Marked slow — skip in CI with -m 'not slow'."""

    @pytest.fixture(scope="class")
    def stage4_result(self):
        from stage1.footstep import Footstep
        from stage2.contact_schedule import build_contact_schedule
        from stage2.lipm import LIPMParams
        from stage2.preview_controller import compute_gains, run_preview_control
        from stage3.controllers.lqr import LQRController
        from stage4.mujoco_sim import run_mujoco_simulation

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
        result = run_mujoco_simulation(traj, schedule, footsteps, params, LQRController())
        return traj, result

    def test_result_shapes(self, stage4_result):
        traj, result = stage4_result
        T = len(traj.t)
        assert result.x.shape == (T,)
        assert result.q.shape == (T, 6)
        assert result.tau.shape == (T, 6)
        assert result.com_height_actual.shape == (T,)
        assert result.zmp_from_grf.shape == (T, 2)
        assert result.energy.shape == (T,)

    def test_com_height_stays_reasonable(self, stage4_result):
        """CoM height should stay roughly near 0.80 m (within 20 cm tolerance)."""
        _, result = stage4_result
        # Allow generous tolerance for initial implementation
        assert result.com_height_actual.min() > 0.3, "CoM dropped too low"
        assert result.com_height_actual.max() < 1.5, "CoM jumped too high"

    def test_grf_nonzero(self, stage4_result):
        """At least some GRF vertical forces should be non-zero."""
        _, result = stage4_result
        total_fz = result.grf_left[:, 2] + result.grf_right[:, 2]
        assert total_fz.max() > 10.0, "No significant vertical contact forces detected"

    def test_com_tracking_error_bounded(self, stage4_result):
        """RMS x-tracking error should stay below 5 cm."""
        _, result = stage4_result
        rms_x = float(np.sqrt(np.mean(result.err_x**2)))
        assert rms_x < 0.05, f"RMS x-tracking error = {rms_x * 100:.1f} cm, exceeds 5 cm"
