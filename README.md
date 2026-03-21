# Bipedal Planner

A 2-D bipedal robot path planning and trajectory optimisation demo.
Given an occupancy-grid world the system plans a collision-free walking path,
generates foot placements, and computes a dynamically-balanced CoM trajectory.

## Architecture

```
stage1_main.py                    Stage 1 entry point
stage2_main.py                    Stage 1 + 2 entry point

stage1/
  world.py              2-D occupancy grid + obstacle inflation
  planners/
    base.py             Planner protocol + smooth_path utility
    astar.py            A* on inflated 8-connected grid
    theta_star.py       Theta* — any-angle A* variant (no post-smoothing needed)
    rrt.py              RRT — continuous-space rapidly-exploring random tree
  footstep.py           Alternating L/R foot placement along CoM waypoints
  stability.py          Support-polygon check (placeholder for Stage 2 ZMP)
  visualizer.py         Stage 1 visualisation

stage2/
  lipm.py               Discrete-time Linear Inverted Pendulum model
  contact_schedule.py   Assigns step timing, builds piecewise ZMP reference
  preview_controller.py ZMP Preview Control (Kajita 2003) — offline LQR + online loop
  traj_visualizer.py    2-D spatial overlay + 4-panel time-series plots
```

The two stages are intentionally separated: Stage 1 outputs an ordered list of
`Footstep` objects; Stage 2 consumes that list and produces a `CoMTrajectory`
(position, velocity, acceleration, ZMP at every timestep) ready to hand off to
a real-time tracking controller (Stage 3 — not implemented).

## Worlds

| Name            | Size      | Description |
|-----------------|-----------|-------------|
| `demo`          | 8 × 6 m   | Simple abstract obstacles |
| `corridor`      | 12 × 6 m  | Factory hallway with machine rows and a central doorway |
| `assembly_line` | 14 × 8 m  | Parallel conveyor rows with freestanding pallets |
| `warehouse`     | 16 × 12 m | Grid of storage racks with picking aisles |

## Prerequisites

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager

## Install

```bash
uv sync          # runtime deps (numpy, scipy, matplotlib)
uv sync --dev    # also installs ruff, pyright
```

## Run

**Stage 1** — footstep planner only:
```bash
uv run python stage1_main.py                              # demo world, A*
uv run python stage1_main.py warehouse --planner theta_star
uv run python stage1_main.py corridor  --planner rrt
uv run python stage1_main.py assembly_line
```

**Stage 1 + 2** — footstep planner + CoM trajectory optimiser:
```bash
uv run python stage2_main.py                              # demo world, A*
uv run python stage2_main.py warehouse --planner theta_star
uv run python stage2_main.py corridor  --planner rrt
uv run python stage2_main.py assembly_line
```

`stage2_main.py` opens two windows: a 2-D spatial overlay (CoM + ZMP paths
over the footstep map) and a 4-panel time-series (position, velocity,
acceleration, with double-support phases shaded).

### Planners

| Name          | Type       | Notes |
|---------------|------------|-------|
| `astar`       | Grid-based | 8-connected A* with post-smoothing (default) |
| `theta_star`  | Grid-based | Any-angle A* — naturally smooth, no post-processing |
| `rrt`         | Sampling   | Continuous-space RRT with goal bias and post-smoothing |

### Tests

```bash
uv run pytest tests
```

## Key parameters

All tunable constants sit at the top of `stage1_main.py` and `stage2_main.py`:

| Parameter          | Default | Meaning |
|--------------------|---------|---------|
| `INFLATION_MARGIN` | 0.25 m  | CoM clearance from obstacles |
| `FOOT_CLEARANCE`   | 0.05 m  | Extra margin for foot placement |
| `STEP_LENGTH`      | 0.25 m  | Forward stride length |
| `STEP_WIDTH`       | 0.10 m  | Lateral foot offset from CoM |
| `T_SINGLE`         | 0.4 s   |  Single support duration |
| `T_DOUBLE`         | 0.1 s   | Double support duration |
| `N_PREVIEW`        | 200 steps (1 s) | ZMP preview horizon |

## Lint & type check

```bash
uv run ruff check .
uv run ruff format .
uv run pyright .
```
