"""Rerun blueprint definitions for stage 1 and stage 2 visualizations."""

from __future__ import annotations

import rerun.blueprint as rrb


def build_stage1_blueprint() -> rrb.Blueprint:
    """Single Spatial3DView with world/planning/stability, time panel collapsed."""
    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            contents=["world/**", "planning/**", "stability/**"],
        ),
        collapse_panels=True,
    )


def build_stage2_blueprint() -> rrb.Blueprint:
    """Horizontal split: 60% 3D spatial view, 40% two stacked time-series views.

    Entity path layout:
      world/**                   — obstacles, boundary (static, z=0)
      planning/footsteps/**      — foot rectangles (static, z=0)
      spatial/com/**             — CoM overview strip (z=com_height) + animated marker
      spatial/zmp/**             — ZMP overview strip (z=0) + animated marker
      spatial/pendulum           — animated inverted-pendulum rod
      spatial/com/velocity       — animated velocity arrow at CoM
      trajectory/com/position/** — CoM x/y position scalars (time-indexed)
      trajectory/com/velocity/** — CoM x/y velocity scalars (time-indexed)
      trajectory/com/acceleration/** — CoM x/y acceleration (time-indexed)
      trajectory/zmp/**          — ZMP x/y scalars (time-indexed)
      trajectory/zmp_ref/**      — ZMP reference x/y scalars (time-indexed)
    """
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/",
                contents=["world/**", "planning/**", "spatial/**"],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/",
                    contents=[
                        "trajectory/com/position/**",
                        "trajectory/zmp/**",
                        "trajectory/zmp_ref/**",
                    ],
                ),
                rrb.TimeSeriesView(
                    origin="/",
                    contents=[
                        "trajectory/com/velocity/**",
                        "trajectory/com/acceleration/**",
                    ],
                ),
            ),
            column_shares=[3, 2],
        ),
    )
