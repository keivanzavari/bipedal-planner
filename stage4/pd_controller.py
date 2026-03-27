"""Joint-level PD controller for the MuJoCo planar biped.

Computes torques τ = Kp * (q_des − q) + Kd * (qdot_des − qdot) for each actuated
joint, given desired CoM position and foot targets from the high-level planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mujoco
except ImportError as e:
    raise ImportError("MuJoCo is required for Stage 4. Install with: uv sync --group mujoco") from e

from robot.config import RobotConfig
from robot.kinematics import two_link_knee
from stage4.ik_angles import angles_from_ik, sagittal_ik

# Joint index mapping into qpos / qvel (after 3 pelvis DOFs)
# qpos: [pelvis_x, pelvis_z, pelvis_pitch,
#         hip_L, knee_L, ankle_L, hip_R, knee_R, ankle_R]
_JOINT_OFFSET = 3  # first 3 are pelvis DOFs
_HIP_L = 0
_KNEE_L = 1
_ANKLE_L = 2
_HIP_R = 3
_KNEE_R = 4
_ANKLE_R = 5
_N_ACTUATED = 6


@dataclass
class PDGains:
    """PD gains per joint group."""

    kp_hip: float = 2000.0
    kd_hip: float = 100.0
    kp_knee: float = 1500.0
    kd_knee: float = 80.0
    kp_ankle: float = 1000.0
    kd_ankle: float = 50.0


class PDController:
    """PD joint-torque controller for the planar biped."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cfg: RobotConfig,
        gains: PDGains | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._cfg = cfg
        self._gains = gains or PDGains()

        # Build Kp / Kd vectors aligned to actuator order
        g = self._gains
        self._kp = np.array([g.kp_hip, g.kp_knee, g.kp_ankle, g.kp_hip, g.kp_knee, g.kp_ankle])
        self._kd = np.array([g.kd_hip, g.kd_knee, g.kd_ankle, g.kd_hip, g.kd_knee, g.kd_ankle])

    def _compute_desired_angles(
        self,
        desired_com: np.ndarray,
        foot_L: np.ndarray,
        foot_R: np.ndarray,
    ) -> np.ndarray:
        """Compute desired joint angles for both legs via sagittal-plane IK.

        Uses ``sagittal_ik`` which always picks the knee-forward (flexion)
        solution, ensuring negative knee angles compatible with MuJoCo
        joint limits.

        Returns (6,) array: [hip_L, knee_L, ankle_L, hip_R, knee_R, ankle_R].
        """
        cfg = self._cfg
        # Hip positions in world frame
        hip_L = desired_com + np.array([0.0, cfg.hip_width, -cfg.pelvis_offset])
        hip_R = desired_com + np.array([0.0, -cfg.hip_width, -cfg.pelvis_offset])

        th_L = sagittal_ik(hip_L, foot_L, cfg.upper_leg, cfg.lower_leg)
        th_R = sagittal_ik(hip_R, foot_R, cfg.upper_leg, cfg.lower_leg)

        return np.array([*th_L, *th_R])

    def step(
        self,
        desired_com: np.ndarray,
        foot_L: np.ndarray,
        foot_R: np.ndarray,
        phase_kind: str,
        stance_side: str | None = None,  # kept for API compatibility, currently unused
    ) -> np.ndarray:
        """Compute desired joint angles via IK and send to MuJoCo position servos.

        Parameters
        ----------
        desired_com : (3,) desired whole-body CoM [x, y, z]
        foot_L      : (3,) left foot target position
        foot_R      : (3,) right foot target position
        phase_kind  : "single" or "double" (currently unused, kept for API compatibility)
        stance_side : unused

        Returns
        -------
        tau : (6,) actual actuator forces applied by MuJoCo servos
        """
        q_des = self._compute_desired_angles(desired_com, foot_L, foot_R)

        # All joints track IK targets regardless of phase. Locking the stance
        # knee to its current angle was intended to allow passive inverted-pendulum
        # pivot, but in practice creates a conflicting torque as the CoM advances,
        # destabilising single support. The IK already computes the correct knee
        # angle to maintain CoM height h as the robot advances.
        self._data.ctrl[:_N_ACTUATED] = q_des
        return self._data.actuator_force[:_N_ACTUATED].copy()
