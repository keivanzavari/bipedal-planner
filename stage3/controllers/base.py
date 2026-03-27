"""Controller Protocol for Stage 3 closed-loop CoM tracking."""

from typing import Protocol

import numpy as np

from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams
from stage2.preview_controller import CoMTrajectory


class Controller(Protocol):
    def reset(self, traj: CoMTrajectory, schedule: ContactSchedule, params: LIPMParams) -> None:
        """Offline setup: precompute gains / reference arrays from the Stage 2 trajectory."""
        ...

    def step(self, k: int, state_x: np.ndarray, state_y: np.ndarray) -> tuple[float, float]:
        """Online step: given current LIPM state, return (jerk_x, jerk_y)."""
        ...
