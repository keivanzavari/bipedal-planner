from stage1.planners.astar import AStarPlanner
from stage1.planners.base import Planner
from stage1.planners.base import smooth_path as smooth_path
from stage1.planners.rrt import RRTPlanner
from stage1.planners.theta_star import ThetaStarPlanner

PLANNERS: dict[str, type] = {
    "astar": AStarPlanner,
    "theta_star": ThetaStarPlanner,
    "rrt": RRTPlanner,
}


def get_planner(name: str, **kwargs) -> Planner:
    if name not in PLANNERS:
        raise ValueError(f"Unknown planner '{name}'. Choose from: {list(PLANNERS)}")
    return PLANNERS[name](**kwargs)
