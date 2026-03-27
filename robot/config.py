"""Robot geometry configuration for kinematic inference and visualization."""

from dataclasses import dataclass


@dataclass
class RobotConfig:
    # Leg geometry
    hip_width: float = 0.10       # lateral half-distance between left and right hip (m)
    upper_leg: float = 0.40       # femur length (m)
    lower_leg: float = 0.40       # tibia length (m)
    pelvis_offset: float = 0.05   # hip height below CoM (m)
    # Swing foot
    foot_clearance: float = 0.08  # peak foot lift during swing phase (m)
    # Torso box (visualization only)
    torso_width: float = 0.28
    torso_depth: float = 0.18
    torso_height: float = 0.45


DEFAULT_ROBOT = RobotConfig()
