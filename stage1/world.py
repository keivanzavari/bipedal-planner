from dataclasses import dataclass, field

import numpy as np


@dataclass
class Rect:
    """Axis-aligned rectangular obstacle in world coordinates (meters)."""

    x: float  # left edge
    y: float  # bottom edge
    w: float  # width
    h: float  # height


@dataclass
class World:
    width: float  # world width in meters
    height: float  # world height in meters
    resolution: float  # meters per cell
    obstacles: list[Rect] = field(default_factory=list)

    def __post_init__(self):
        self.cols = int(np.ceil(self.width / self.resolution))
        self.rows = int(np.ceil(self.height / self.resolution))
        self._grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for obs in self.obstacles:
            self._mark_rect(obs, value=1)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """World (x, y) → grid (row, col). Origin is bottom-left."""
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Grid (row, col) → world (x, y) at cell centre."""
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    # ------------------------------------------------------------------
    # Grid access
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Raw occupancy grid (0 = free, 1 = obstacle)."""
        return self._grid

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self._grid[row, col] == 0

    # ------------------------------------------------------------------
    # Inflation
    # ------------------------------------------------------------------

    def inflated_grid(self, margin: float) -> np.ndarray:
        """Return a copy of the grid with obstacles dilated by `margin` meters."""
        from scipy.ndimage import binary_dilation

        radius_cells = int(np.ceil(margin / self.resolution))
        struct = _circle_struct(radius_cells)
        inflated = binary_dilation(self._grid, structure=struct).astype(np.uint8)
        return inflated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_rect(self, rect: Rect, value: int = 1):
        r0, c0 = self.world_to_cell(rect.x, rect.y)
        r1, c1 = self.world_to_cell(rect.x + rect.w, rect.y + rect.h)
        r0, r1 = max(0, r0), min(self.rows, r1 + 1)
        c0, c1 = max(0, c0), min(self.cols, c1 + 1)
        self._grid[r0:r1, c0:c1] = value


def _circle_struct(radius: int) -> np.ndarray:
    """Circular structuring element for dilation."""
    size = 2 * radius + 1
    cy, cx = radius, radius
    y, x = np.ogrid[:size, :size]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).astype(np.uint8)


# ------------------------------------------------------------------
# Worlds
# ------------------------------------------------------------------


def make_demo_world() -> tuple["World", tuple, tuple]:
    """Simple test world with a few abstract obstacles."""
    obstacles = [
        Rect(2.0, 0.5, 0.4, 2.0),  # vertical wall left
        Rect(3.5, 1.5, 0.4, 2.5),  # vertical wall right
        Rect(1.0, 3.5, 2.0, 0.4),  # horizontal bar top
        Rect(5.0, 1.0, 1.0, 1.0),  # box
    ]
    world = World(width=8.0, height=6.0, resolution=0.05, obstacles=obstacles)
    start = (0.5, 0.5)
    goal = (7.5, 5.5)
    return world, start, goal


def make_corridor_world() -> tuple["World", tuple, tuple]:
    """
    Factory corridor: a long main aisle with two side-branch corridors
    blocked by machinery, forcing the robot to navigate to a doorway at
    the far end.

    Layout (12 x 6 m):

      ┌──────────────────────────────────────────┐
      │  [machine row]   gap   [machine row]      │
      │                                           │
      │  ══════════════════════  (wall + doorway) │
      │                                           │
      │  [machine row]   gap   [machine row]      │
      └──────────────────────────────────────────┘
    """
    obstacles = [
        # Outer walls (thin perimeter strips)
        Rect(0.0, 0.0, 12.0, 0.2),  # bottom wall
        Rect(0.0, 5.8, 12.0, 0.2),  # top wall
        Rect(0.0, 0.0, 0.2, 6.0),  # left wall
        Rect(11.8, 0.0, 0.2, 6.0),  # right wall
        # Central dividing wall with a doorway gap (x=5.5–6.5)
        Rect(0.2, 2.8, 5.3, 0.4),  # left half of dividing wall
        Rect(6.5, 2.8, 5.3, 0.4),  # right half of dividing wall
        # Bottom corridor — machine blocks (left side)
        Rect(1.0, 0.3, 1.2, 0.8),
        Rect(2.8, 0.3, 1.2, 0.8),
        Rect(4.6, 0.3, 1.2, 0.8),
        Rect(7.0, 0.3, 1.2, 0.8),
        Rect(8.8, 0.3, 1.2, 0.8),
        # Top corridor — machine blocks (right side)
        Rect(1.0, 4.9, 1.2, 0.8),
        Rect(2.8, 4.9, 1.2, 0.8),
        Rect(4.6, 4.9, 1.2, 0.8),
        Rect(7.0, 4.9, 1.2, 0.8),
        Rect(8.8, 4.9, 1.2, 0.8),
        # Occasional support pillars in the aisle
        Rect(3.6, 1.4, 0.3, 0.3),
        Rect(7.8, 1.4, 0.3, 0.3),
        Rect(3.6, 4.3, 0.3, 0.3),
        Rect(7.8, 4.3, 0.3, 0.3),
    ]
    world = World(width=12.0, height=6.0, resolution=0.05, obstacles=obstacles)
    start = (0.5, 1.5)  # bottom-left aisle
    goal = (11.4, 4.5)  # top-right aisle
    return world, start, goal


def make_assembly_line_world() -> tuple["World", tuple, tuple]:
    """
    Assembly line: two parallel conveyor/machine rows running the length
    of the floor, with periodic cross-aisle gaps. The robot must cross
    from one end to the other while navigating the gaps.

    Layout (14 x 8 m):

      start →  [machine][gap][machine][gap][machine]  → goal
               [  aisle space between rows          ]
               [machine][gap][machine][gap][machine]
    """
    machine_w = 1.6  # machine block width
    machine_h = 1.2  # machine block depth
    gap = 1.0  # gap between machines (cross-aisle)
    top_y = 5.8  # y of top machine row (bottom edge)
    bot_y = 1.0  # y of bottom machine row (bottom edge)
    obstacles = []

    # Outer walls
    obstacles += [
        Rect(0.0, 0.0, 14.0, 0.2),
        Rect(0.0, 7.8, 14.0, 0.2),
        Rect(0.0, 0.0, 0.2, 8.0),
        Rect(13.8, 0.0, 0.2, 8.0),
    ]

    # Generate alternating machine blocks along both rows
    x = 0.5
    while x + machine_w <= 13.5:
        obstacles.append(Rect(x, bot_y, machine_w, machine_h))  # bottom row
        obstacles.append(Rect(x, top_y, machine_w, machine_h))  # top row
        x += machine_w + gap

    # A few freestanding pallets / crates in the central aisle
    obstacles += [
        Rect(3.2, 3.5, 0.6, 0.6),
        Rect(6.5, 3.2, 0.6, 0.6),
        Rect(9.8, 3.6, 0.6, 0.6),
    ]

    world = World(width=14.0, height=8.0, resolution=0.05, obstacles=obstacles)
    start = (0.5, 3.8)
    goal = (13.4, 3.8)
    return world, start, goal


def make_warehouse_world() -> tuple["World", tuple, tuple]:
    """
    Warehouse: a grid of storage rack rows with narrow picking aisles
    between them. The robot must navigate from a loading dock on one side
    to a dispatch area on the other.

    Layout (16 x 12 m):
      - 4 columns of racks, each rack is 0.6 m deep and 2.5 m long
      - 3 picking aisles (1.2 m wide) running left-right
      - 2 cross-aisles running top-bottom
    """
    rack_depth = 0.6
    rack_len = 2.5
    aisle_w = 1.2  # picking aisle width (between rack rows)
    cross_aisle = 1.5  # cross-aisle width (between rack columns)
    wall_t = 0.3
    world_w = 16.0
    world_h = 12.0
    obstacles = []

    rack_bay_h = 2 * rack_depth + 0.1  # total depth of a back-to-back rack pair
    col_step = rack_len + cross_aisle  # x-spacing between rack column starts
    row_step = rack_bay_h + aisle_w  # y-spacing between rack row starts

    # Outer walls
    obstacles += [
        Rect(0.0, 0.0, world_w, wall_t),
        Rect(0.0, world_h - wall_t, world_w, wall_t),
        Rect(0.0, 0.0, wall_t, world_h),
        Rect(world_w - wall_t, 0.0, wall_t, world_h),
    ]

    # Rack grid: 4 columns x 4 rows, positions derived from spacing variables
    col_xs = [wall_t + 0.5 + i * col_step for i in range(4)]
    row_ys = [wall_t + 0.5 + i * row_step for i in range(4)]

    for cx in col_xs:
        for ry in row_ys:
            # Each rack bay = two back-to-back racks with a small gap
            obstacles.append(Rect(cx, ry, rack_len, rack_depth))
            obstacles.append(Rect(cx, ry + rack_depth + 0.1, rack_len, rack_depth))

    # Partial blockages: one crate per cross-aisle/picking-aisle junction
    # placed at the midpoint of every other cross-aisle gap
    cross_aisle_xs = [col_xs[i] - cross_aisle / 2 - 0.2 for i in range(1, 4)]
    aisle_center_ys = [row_ys[i] + rack_bay_h + aisle_w / 2 for i in range(3)]
    obstacles += [
        Rect(cross_aisle_xs[0], aisle_center_ys[0] - 0.4, 0.4, 0.8),
        Rect(cross_aisle_xs[1], aisle_center_ys[1] - 0.4, 0.4, 0.8),
        Rect(cross_aisle_xs[2], aisle_center_ys[2] - 0.4, 0.4, 0.8),
    ]

    world = World(width=world_w, height=world_h, resolution=0.05, obstacles=obstacles)
    # Start in first picking aisle, goal in third — forces cross-aisle navigation
    start = (wall_t + 0.3, aisle_center_ys[0])
    goal = (world_w - wall_t - 0.3, aisle_center_ys[2])
    return world, start, goal


WORLDS = {
    "demo": make_demo_world,
    "corridor": make_corridor_world,
    "assembly_line": make_assembly_line_world,
    "warehouse": make_warehouse_world,
}
