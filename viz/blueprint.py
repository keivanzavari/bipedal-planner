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
    """Horizontal split: 60% spatial view, 40% two stacked time-series views."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin="/",
                contents=[
                    "world/**",
                    "planning/footsteps/**",
                    "trajectory/com/spatial",
                    "trajectory/zmp/spatial",
                ],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/trajectory",
                    contents=[
                        "com/position/**",
                        "zmp/**",
                        "zmp_ref/**",
                    ],
                ),
                rrb.TimeSeriesView(
                    origin="/trajectory",
                    contents=[
                        "com/velocity/**",
                        "com/acceleration/**",
                    ],
                ),
            ),
            column_shares=[3, 2],
        ),
    )
