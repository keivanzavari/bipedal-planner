"""
Linear Inverted Pendulum Model (LIPM) — discrete-time state-space.

State:  x = [pos, vel, acc]
Input:  u = jerk
Output: p = pos - (h/g)*acc   (ZMP position)

Equations of motion:
    pos_ddot = (g/h) * (pos - zmp)
    =>  zmp  = pos - (h/g) * acc
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LIPMParams:
    h: float = 0.80  # CoM height (m)
    g: float = 9.81  # gravity (m/s²)
    dt: float = 0.005  # timestep (s)


def lipm_matrices(params: LIPMParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return discrete-time matrices (A, B, C) for the LIPM with jerk input.

    s[k+1] = A @ s[k] + B * u[k]
    p[k]   = C @ s[k]
    """
    dt = params.dt
    h = params.h
    g = params.g

    A = np.array(
        [
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )
    B = np.array([dt**3 / 6.0, dt**2 / 2.0, dt])
    C = np.array([1.0, 0.0, -h / g])

    return A, B, C


def zmp_from_state(state: np.ndarray, C: np.ndarray) -> float:
    """Compute ZMP from CoM state vector [pos, vel, acc]."""
    return float(C @ state)
