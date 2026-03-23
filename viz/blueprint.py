"""Rerun blueprint definitions for stage 1 and stage 2 visualizations."""

from __future__ import annotations

import rerun.blueprint as rrb


def build_stage1_blueprint() -> rrb.Blueprint:
    """Single Spatial2DView with world/planning/stability, time panel collapsed."""
    return rrb.Blueprint(
        rrb.Spatial2DView(
            origin="/",
            contents=["world/**", "planning/**", "stability/**"],
        ),
        collapse_panels=True,
    )


def build_stage2_blueprint() -> rrb.Blueprint:
    """Horizontal split: 60% spatial view, 40% two stacked time-series views.

    Entity path layout:
      world/**                   — obstacles, boundary (static)
      planning/footsteps/**      — foot rectangles (static)
      spatial/com/**             — CoM overview strip + animated marker
      spatial/zmp/**             — ZMP overview strip + animated marker
      trajectory/com/position/** — CoM x/y position scalars (time-indexed)
      trajectory/com/velocity/** — CoM x/y velocity scalars (time-indexed)
      trajectory/com/acceleration/** — CoM x/y acceleration (time-indexed)
      trajectory/zmp/**          — ZMP x/y scalars (time-indexed)
      trajectory/zmp_ref/**      — ZMP reference x/y scalars (time-indexed)
    """
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
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
