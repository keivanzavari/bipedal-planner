"""Stage 4 MuJoCo simulation loop.

Replaces the LIPM integrator in Stage 3 with MuJoCo rigid-body physics for the
sagittal (x, z) plane. The lateral (y) axis remains LIPM-analytical. Ground-reaction
forces, ZMP, joint angles, and CoM height are extracted from the physics engine.

Mirrors the structure of ``stage3/simulator.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as e:
    raise ImportError("MuJoCo is required for Stage 4. Install with: uv sync --group mujoco") from e

from robot.config import RobotConfig
from robot.kinematics import active_feet_at, compute_phase_progress
from stage1.footstep import Footstep
from stage2.contact_schedule import ContactSchedule
from stage2.lipm import LIPMParams, lipm_matrices
from stage2.preview_controller import CoMTrajectory
from stage2.traj_optimizer import _compute_zmp_bounds
from stage3.controllers.base import Controller
from stage4.mujoco_result import MujocoResult
from stage4.pd_controller import PDController, PDGains

# Path to the MJCF scene file
_SCENE_XML = Path(__file__).parent / "models" / "biped_scene.xml"

# Sub-steps per LIPM step: 5 × 1 ms = 5 ms
_N_SUBSTEPS = 5

# Pelvis DOF count (slide_x, slide_z, hinge_pitch)
_N_PELVIS_DOF = 3
_N_ACTUATED = 6


# --- Contact force extraction ---


def _extract_contact_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_geom_id: int,
    foot_L_geom_id: int,
    foot_R_geom_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum MuJoCo contact forces per foot, transformed to world frame.

    Returns (grf_L, grf_R) each (3,) in world frame [Fx, Fy, Fz].
    """
    grf_L = np.zeros(3)
    grf_R = np.zeros(3)
    for i in range(data.ncon):
        c = data.contact[i]
        geom_pair = {c.geom1, c.geom2}
        is_left = geom_pair == {floor_geom_id, foot_L_geom_id}
        is_right = geom_pair == {floor_geom_id, foot_R_geom_id}
        if not (is_left or is_right):
            continue

        raw = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, raw)
        # Contact frame is stored row-major in c.frame (9,) → (3, 3)
        # Columns of frame = axes of contact frame in world coords
        # raw[:3] = [normal_force, tangent1, tangent2] in contact frame
        frame = c.frame.reshape(3, 3)
        force_world = frame.T @ raw[:3]

        if is_left:
            grf_L += force_world
        else:
            grf_R += force_world

    return grf_L, grf_R


def _compute_zmp_from_grf(
    grf_L: np.ndarray,
    grf_R: np.ndarray,
    pos_L: np.ndarray,
    pos_R: np.ndarray,
) -> np.ndarray:
    """Compute ZMP from ground-reaction forces.

    ZMP_x = Σ(pos_i_x * Fz_i) / Σ(Fz_i), same for y.

    Returns (3,) array [zmp_x, zmp_y, 0].
    """
    Fz_total = grf_L[2] + grf_R[2]
    if Fz_total < 1.0:
        # Nearly zero vertical force — fall back to midpoint
        return np.array([(pos_L[0] + pos_R[0]) / 2.0, (pos_L[1] + pos_R[1]) / 2.0, 0.0])
    zmp_x = (grf_L[2] * pos_L[0] + grf_R[2] * pos_R[0]) / Fz_total
    zmp_y = (grf_L[2] * pos_L[1] + grf_R[2] * pos_R[1]) / Fz_total
    return np.array([zmp_x, zmp_y, 0.0])


# --- Main simulation ---


def run_mujoco_simulation(
    traj: CoMTrajectory,
    schedule: ContactSchedule,
    footsteps: list[Footstep],
    params: LIPMParams,
    controller: Controller,
    cfg: RobotConfig | None = None,
    pd_gains: PDGains | None = None,
    foot_length: float = 0.16,
    foot_width: float = 0.08,
) -> MujocoResult:
    """Run closed-loop CoM tracking using MuJoCo physics.

    Sagittal (x) axis: CoM position and velocity come from MuJoCo. Acceleration
    is maintained via shadow LIPM integration.

    Lateral (y) axis: fully LIPM-analytical (not simulated in MuJoCo).

    Parameters
    ----------
    traj        : reference CoM trajectory from Stage 2
    schedule    : contact schedule
    footsteps   : ordered Footstep list from Stage 1
    params      : LIPM parameters
    controller  : high-level controller (LQR or MPC)
    cfg         : robot geometry config
    pd_gains    : PD controller gains (default: PDGains())
    foot_length : for ZMP bounds computation
    foot_width  : for ZMP bounds computation
    """
    from robot.config import DEFAULT_ROBOT

    if cfg is None:
        cfg = DEFAULT_ROBOT

    A, B, C = lipm_matrices(params)
    T = len(traj.t)

    # --- Load MuJoCo model ---
    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)

    # Look up body/geom IDs
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_L_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
    foot_R_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")  # noqa: F841

    # --- PD controller ---
    pd = PDController(model, data, cfg, pd_gains)

    # --- Initialize robot pose ---
    data.qpos[0] = traj.x[0]  # pelvis_x
    data.qpos[1] = params.h  # pelvis_z
    data.qpos[2] = 0.0  # pelvis_pitch

    # Precompute phase alpha and initial foot targets
    phase_alpha = compute_phase_progress(schedule)
    foot_L_init, foot_R_init = active_feet_at(0, footsteps, schedule, cfg, phase_alpha)
    initial_com = np.array([traj.x[0], traj.y[0], params.h])

    # Set initial joint angles via IK (prevents bounce from default straight-leg pose)
    q_init = pd._compute_desired_angles(initial_com, foot_L_init, foot_R_init)
    data.qpos[_N_PELVIS_DOF : _N_PELVIS_DOF + _N_ACTUATED] = q_init

    # Warm-up: settle contacts with PD active for 0.2 s (200 steps).
    # IK is computed from the actual pelvis position each step (not the fixed
    # initial_com target) so the pelvis stays at its natural equilibrium and
    # does not drift backward.  Using initial_com as the IK x-target while the
    # pelvis is elsewhere creates a mis-aligned servo torque that pushes the
    # pelvis backward — a self-reinforcing loop that causes ~4 cm of drift over
    # 500 ms.  Tracking actual_x eliminates that force imbalance.
    mujoco.mj_forward(model, data)
    _WARMUP_STEPS = 200
    for _ in range(_WARMUP_STEPS):
        actual_warmup_com = np.array([float(data.qpos[0]), initial_com[1], params.h])
        pd.step(actual_warmup_com, foot_L_init, foot_R_init, "double")
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)  # ensure kinematics are current

    # --- Prepare controller ---
    controller.reset(traj, schedule, params)

    # Re-read actual position AND velocity after warmup to initialize shadow states.
    # Using traj.vx[0] here was wrong when the pelvis is still moving during settling.
    settled_x = float(data.qpos[0])
    settled_vx = float(data.qvel[0])  # pelvis_x slide velocity
    shadow_x = np.array([settled_x, settled_vx, traj.ax[0]])
    shadow_y = np.array([traj.y[0], traj.vy[0], traj.ay[0]])

    # Precompute ZMP bounds (reuse Stage 2 support polygon computation)
    lb_x, ub_x, lb_y, ub_y = _compute_zmp_bounds(schedule, footsteps, foot_length, foot_width)

    # --- Output arrays ---
    out_x = np.empty(T)
    out_y = np.empty(T)
    out_vx = np.empty(T)
    out_vy = np.empty(T)
    out_ux = np.empty(T)
    out_uy = np.empty(T)
    out_zmp_grf = np.empty((T, 2))
    out_zmp_lipm = np.empty((T, 2))
    out_grf_left = np.zeros((T, 3))
    out_grf_right = np.zeros((T, 3))
    out_com_z = np.empty(T)
    out_q = np.empty((T, _N_ACTUATED))
    out_qdot = np.empty((T, _N_ACTUATED))
    out_tau = np.empty((T, _N_ACTUATED))
    out_energy = np.empty(T)

    # For finite-difference pelvis velocity (used as LIPM CoM proxy)
    pelvis_prev_x = float(data.qpos[0])

    # --- Main loop ---
    for k in range(T):
        # 1. Use pelvis position as the LIPM CoM proxy (avoids oscillation from
        #    subtree_com which shifts as legs swing).
        pelvis_x = float(data.qpos[0])
        com_z = float(data.subtree_com[pelvis_id][2])
        pelvis_vx = (pelvis_x - pelvis_prev_x) / params.dt if k > 0 else traj.vx[0]

        # 2. Build LIPM states for the high-level controller
        mj_state_x = np.array([pelvis_x, pelvis_vx, shadow_x[2]])
        mj_state_y = shadow_y.copy()

        # 3. High-level controller produces (jerk_x, jerk_y)
        jerk_x, jerk_y = controller.step(k, mj_state_x, mj_state_y)

        # 4. Integrate shadow states
        shadow_x = A @ mj_state_x + B * jerk_x
        shadow_y = A @ mj_state_y + B * jerk_y

        # 5. Build desired_com for PD.
        # Clamp how far the IK target can lead the actual pelvis.  When the
        # shadow state races ahead of the actual position (e.g. after a large
        # initial tracking error), the IK places the hips far forward of the
        # actual pelvis.  The position servos then apply strong forward-lean
        # torques to reach those angles, but from the wrong geometry this
        # creates a backward reaction force on the pelvis — the opposite of
        # what is needed.  A ≤10 cm lead keeps the IK configuration physical
        # while still providing a forward-driving lean during normal walking.
        _MAX_IK_LEAD = 0.10  # m — maximum x-lead of IK target over actual pelvis
        # Only allow FORWARD lead (never let IK target go behind actual pelvis).
        # A negative lead means the reference has moved backward of the actual
        # position; commanding a backward lean then creates a strong backward
        # servo reaction force on the pelvis.  Clamping to zero allows the
        # passive inverted-pendulum dynamics to take over — the planted stance
        # foot naturally pulls the CoM forward under gravity.
        ik_x = pelvis_x + np.clip(shadow_x[0] - pelvis_x, 0.0, _MAX_IK_LEAD)
        desired_com = np.array([ik_x, shadow_y[0], params.h])

        # 6. Foot targets
        foot_L, foot_R = active_feet_at(k, footsteps, schedule, cfg, phase_alpha)

        # 6. Determine stance side for PD (no contact toggling — swing foot arc
        #    provides 8 cm clearance; toggling creates landing force spikes).
        stance_side = footsteps[int(schedule.phase[k])].side if schedule.kind[k] == "single" else None

        # 7. Sub-step MuJoCo
        pelvis_prev_x = pelvis_x
        for _ in range(_N_SUBSTEPS):
            tau = pd.step(desired_com, foot_L, foot_R, schedule.kind[k], stance_side)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)  # refresh kinematics (subtree_com)

        # 8. Extract contact forces and ZMP
        grf_L, grf_R = _extract_contact_forces(model, data, floor_geom_id, foot_L_geom_id, foot_R_geom_id)
        zmp_grf = _compute_zmp_from_grf(grf_L, grf_R, foot_L, foot_R)
        zmp_lipm_x = float(C @ shadow_x)
        zmp_lipm_y = float(C @ shadow_y)

        # 9. Record
        out_x[k] = pelvis_x
        out_y[k] = shadow_y[0]
        out_vx[k] = pelvis_vx
        out_vy[k] = shadow_y[1]
        out_ux[k] = jerk_x
        out_uy[k] = jerk_y
        out_zmp_grf[k] = [zmp_grf[0], zmp_grf[1]]
        out_zmp_lipm[k] = [zmp_lipm_x, zmp_lipm_y]
        out_grf_left[k] = grf_L
        out_grf_right[k] = grf_R
        out_com_z[k] = com_z
        out_q[k] = data.qpos[_N_PELVIS_DOF : _N_PELVIS_DOF + _N_ACTUATED]
        out_qdot[k] = data.qvel[_N_PELVIS_DOF : _N_PELVIS_DOF + _N_ACTUATED]
        out_tau[k] = tau
        out_energy[k] = data.energy[0] + data.energy[1]  # kinetic + potential

    return MujocoResult(
        t=traj.t,
        x=out_x,
        y=out_y,
        vx=out_vx,
        vy=out_vy,
        ref_x=traj.x,
        ref_y=traj.y,
        err_x=out_x - traj.x,
        err_y=out_y - traj.y,
        u_x=out_ux,
        u_y=out_uy,
        grf_left=out_grf_left,
        grf_right=out_grf_right,
        zmp_x=out_zmp_grf[:, 0],
        zmp_y=out_zmp_grf[:, 1],
        zmp_lb_x=lb_x,
        zmp_ub_x=ub_x,
        zmp_lb_y=lb_y,
        zmp_ub_y=ub_y,
        friction=np.ones(T),
        q=out_q,
        qdot=out_qdot,
        tau=out_tau,
        zmp_from_lipm=out_zmp_lipm,
        zmp_from_grf=out_zmp_grf,
        com_height_actual=out_com_z,
        energy=out_energy,
    )
