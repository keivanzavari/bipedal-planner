# Bipedal Planner

A 2-D bipedal robot path planning and trajectory optimisation demo.
Given an occupancy-grid world the system plans a collision-free walking path,
generates foot placements, computes a dynamically-balanced CoM trajectory, and
tracks it in closed-loop with LQR or MPC controllers.

![image](docs/image.png)

## Architecture

```text
stage1_main.py                    Stage 1 entry point
stage2_main.py                    Stage 1 + 2 entry point
stage3_main.py                    Stage 1 + 2 + 3 entry point (closed-loop tracking)
compare_controllers.py            Side-by-side LQR vs MPC comparison

stage1/
  world.py              2-D occupancy grid + obstacle inflation + SlipperyZone
  planners/
    base.py             Planner protocol + smooth_path utility
    astar.py            A* on inflated 8-connected grid
    theta_star.py       Theta* — any-angle A* variant (no post-smoothing needed)
    rrt.py              RRT — continuous-space rapidly-exploring random tree
  footstep.py           Alternating L/R foot placement along CoM waypoints
  stability.py          Support-polygon check

stage2/
  lipm.py               Discrete-time Linear Inverted Pendulum model
  contact_schedule.py   Assigns step timing, builds piecewise ZMP reference
  preview_controller.py ZMP Preview Control (Kajita 2003) — offline LQR + online loop
  traj_optimizer.py     QP-based trajectory optimizer (OSQP); propagation matrix utils
  traj_visualizer.py    2-D spatial overlay + 4-panel time-series plots (matplotlib)

stage3/
  simulator.py          Closed-loop simulation loop; TrackingResult; slippery-zone physics
  controllers/
    base.py             Controller protocol (reset / step)
    lqr.py              LQR controller — deadbeat + integrator on ZMP error
    mpc.py              MPC controller — receding-horizon QP with DARE terminal cost

robot/
  config.py             RobotConfig dataclass (leg geometry, torso dimensions)
  kinematics.py         Closed-form 2-link IK, swing-foot arc, phase progress

viz/
  __init__.py           Re-exports visualize_stage1/2/3
  primitives.py         Low-level rr.log helpers (world, feet, body, ZMP bounds, friction…)
  blueprint.py          Rerun blueprint layouts for stage 1, 2, and 3
  stage1_viz.py         Stage 1 Rerun entry point
  stage2_viz.py         Stage 2 Rerun entry point (rod or 2-link model)
  stage3_viz.py         Stage 3 Rerun entry point (tracking, ZMP bounds, support polygon)
```

The stages are intentionally separated:

- **Stage 1** outputs an ordered list of `Footstep` objects.
- **Stage 2** consumes that list and produces a `CoMTrajectory` (position, velocity,
  acceleration, ZMP at every timestep).
- **Stage 3** tracks the reference trajectory in closed-loop, logging actual CoM,
  ZMP vs safety bounds, tracking error, and ground-reaction forces.

The `robot/` module provides a lightweight kinematic model (hip/knee/ankle chain,
swing-foot parabolic arc) shared by the visualization and Stage 3 controllers.

## Worlds

| Name            | Size       | Description                                          |
|-----------------|------------|------------------------------------------------------|
| `demo`          | 8 × 6 m    | Simple abstract obstacles                            |
| `corridor`      | 12 × 6 m   | Factory hallway with machine rows and central door   |
| `assembly_line` | 14 × 8 m   | Parallel conveyor rows with freestanding pallets     |
| `warehouse`     | 16 × 12 m  | Grid of storage racks with picking aisles            |

## Prerequisites

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager

## Install

```bash
uv sync          # runtime deps (numpy, scipy, matplotlib, rerun-sdk, osqp)
uv sync --dev    # also installs ruff, pyright
```

## Run

**Stage 1** — footstep planner only:

```bash
uv run python stage1_main.py                                        # demo world, A*, matplotlib
uv run python stage1_main.py warehouse --planner theta_star
uv run python stage1_main.py corridor  --planner rrt
uv run python stage1_main.py demo      --viz rerun
```

**Stage 1 + 2** — footstep planner + CoM trajectory optimiser:

```bash
uv run python stage2_main.py                                        # demo world, A*, matplotlib
uv run python stage2_main.py warehouse --planner theta_star
uv run python stage2_main.py demo      --viz rerun
uv run python stage2_main.py demo      --viz rerun --body model     # 2-link stick figure
```

**Stage 1 + 2 + 3** — full closed-loop tracking (Rerun only):

```bash
uv run python stage3_main.py                                        # demo world, LQR
uv run python stage3_main.py demo --controller mpc
uv run python stage3_main.py demo --controller lqr --noise 0.005
uv run python stage3_main.py demo --controller mpc --slippery       # slippery zone, friction=0.4
uv run python stage3_main.py demo --controller mpc --slippery --friction-scale 0.3
uv run python stage3_main.py demo --controller lqr --slippery --zone 2.0 0.0 3.0 4.0
```

**Controller comparison** — LQR vs MPC side-by-side (matplotlib):

```bash
uv run python compare_controllers.py demo
uv run python compare_controllers.py demo --controllers lqr mpc --noise 0.005
uv run python compare_controllers.py demo --slippery --noise 0.005
uv run python compare_controllers.py warehouse --planner theta_star --slippery
```

### Stage 3 controllers

| Name  | Type                 | Notes                                                          |
|-------|----------------------|----------------------------------------------------------------|
| `lqr` | Infinite-horizon LQR | Deadbeat integrator on ZMP error; no constraint awareness      |
| `mpc` | Receding-horizon QP  | DARE terminal cost; enforces ZMP bounds over prediction window |

MPC uses two decoupled OSQP QPs (x and y axes) solved at each timestep.
The fixed Hessian is factorised once in `reset()`; only the linear cost and
bounds are updated per step, making each solve sub-millisecond.

### Slippery surface scenario

Pass `--slippery` to any Stage 3 command to add a reduced-friction zone:

```bash
uv run python stage3_main.py demo --controller mpc --slippery --noise 0.005
uv run python compare_controllers.py demo --slippery --noise 0.005 --friction-scale 0.4
```

The slippery zone shrinks the effective foot support polygon by `friction_scale`
and applies a stochastic landing impulse when a foot touches down inside it.
MPC's constraint-aware ZMP bounds result in roughly 2× fewer support-polygon
violations compared to LQR, at the cost of slightly higher tracking error.

```
Controller    Max |err| (cm)   RMS err (cm)   ZMP viol-x     ZMP viol-y   Time (ms)
lqr                    11.82         3.2016   2706/3780      2840/3780        269.1
mpc                    17.36         5.5719   1195/3780      1098/3780       2293.4
```

### Visualization backends

| Backend      | Stage 1                   | Stage 2                             | Stage 3                                    |
|--------------|---------------------------|-------------------------------------|--------------------------------------------|
| `matplotlib` | static figure             | two figures (spatial + time-series) | via `compare_controllers.py`               |
| `rerun`      | interactive Spatial3DView | Spatial3DView + TimeSeriesViews     | Spatial3DView + 3 stacked TimeSeriesViews  |

The Stage 3 Rerun layout is a 50/50 horizontal split:

- **Left** — 3-D spatial view: world obstacles, footsteps, slippery zone floor patch,
  reference + actual CoM paths, animated support polygon (shrinks in slippery zones),
  GRF arrows, body animation.
- **Right** — three stacked time-series panels:
  1. CoM position, ZMP x/y, ZMP reference
  2. Actual ZMP vs lower/upper bounds per axis, friction coefficient
  3. Tracking error x/y

### Body representation (Rerun, Stage 2 only)

| `--body` | Description |
|----------|-------------|
| `rod`    | Inverted-pendulum rod from ZMP to CoM (default, minimal) |
| `model`  | Animated 2-link stick figure: torso box + bending legs with parabolic swing-foot arc |

### Planners

| Name         | Type       | Notes                                                     |
|--------------|------------|-----------------------------------------------------------|
| `astar`      | Grid-based | 8-connected A* with post-smoothing (default)              |
| `theta_star` | Grid-based | Any-angle A* — naturally smooth, no post-processing       |
| `rrt`        | Sampling   | Continuous-space RRT with goal bias and post-smoothing    |

### Tests

```bash
uv run pytest tests
```

## Key parameters

| Parameter          | Default         | Meaning                                     |
|--------------------|-----------------|---------------------------------------------|
| `INFLATION_MARGIN` | 0.25 m          | CoM clearance from obstacles                |
| `FOOT_CLEARANCE`   | 0.05 m          | Extra margin for foot placement             |
| `STEP_LENGTH`      | 0.25 m          | Forward stride length                       |
| `STEP_WIDTH`       | 0.10 m          | Lateral foot offset from CoM               |
| `T_SINGLE`         | 0.4 s           | Single support duration                     |
| `T_DOUBLE`         | 0.1 s           | Double support duration                     |
| `N_PREVIEW`        | 200 steps (1 s) | ZMP preview horizon (Stage 2)               |
| `N_HORIZON`        | 20 steps (0.1 s)| MPC prediction horizon (Stage 3)            |
| `noise_sigma`      | 0.001 m         | Per-step Gaussian noise on CoM position     |
| `friction_scale`   | 0.4             | Support polygon scale factor in slippery zones |

## Lint & type check

```bash
uv run ruff check .
uv run ruff format .
uv run pyright .
```
