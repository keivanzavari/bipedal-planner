# Bipedal Planner

A 2-D bipedal robot path and footstep planning demo.
Given an occupancy-grid world, the pipeline runs:

1. **A\*** path planning on an inflated grid
2. **Path smoothing** via line-of-sight shortcutting
3. **Footstep placement** along the CoM waypoints
4. **Static stability checking** using support-polygon tests
5. **Visualisation** with matplotlib

## Prerequisites

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package and project manager

## Install dependencies

```bash
uv sync          # installs runtime deps into .venv/
uv sync --dev    # also installs dev tools (ruff, pyright)
```

## Run

```bash
uv run python main.py            # default "demo" world
uv run python main.py warehouse  # warehouse world
uv run python main.py corridor   # corridor world
```

Available worlds: `demo`, `warehouse`, `corridor`, `assembly`.

## Lint & format

```bash
uv run ruff check .        # lint
uv run ruff format .       # auto-format
uv run ruff format --check .  # format check (CI)
```

## Type checking

```bash
uv run pyright .
```

Type checking is configured in `pyproject.toml` under `[tool.pyright]`
(mode: `basic`, target: Python 3.10).
