from stage3.controllers.base import Controller
from stage3.controllers.lqr import LQRController
from stage3.controllers.mpc import MPCController

CONTROLLERS: dict[str, type] = {
    "lqr": LQRController,
    "mpc": MPCController,
    # "wbc": WBCController,   ← deferred
}


def get_controller(name: str, **kwargs) -> Controller:
    if name not in CONTROLLERS:
        raise ValueError(f"Unknown controller '{name}'. Choose from: {list(CONTROLLERS)}")
    return CONTROLLERS[name](**kwargs)  # type: ignore[return-value]
