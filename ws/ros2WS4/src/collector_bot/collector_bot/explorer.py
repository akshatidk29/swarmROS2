"""
Grid-based frontier exploration with multi-robot coordination.

The 12 × 12 m arena is divided into 2 m cells (6 × 6 = 36 zones).
Robots share visited cells on ``/visited_zones`` so they spread out.
"""

import math

GRID  = 2.0   # metres per cell
NX    = 6     # 12 m / 2 m
NY    = 6
ARENA = 6.0   # half-size of arena


class Explorer:
    """Per-robot exploration bookkeeper."""

    def __init__(self):
        self.visited = set()           # (i, j) tuples
        self.current_target = None     # (x, y) or None

    # ── grid helpers ──

    @staticmethod
    def pos_to_cell(x, y):
        i = int((x + ARENA) / GRID)
        j = int((y + ARENA) / GRID)
        return (max(0, min(NX - 1, i)),
                max(0, min(NY - 1, j)))

    @staticmethod
    def cell_to_pos(i, j):
        return ((i + 0.5) * GRID - ARENA,
                (j + 0.5) * GRID - ARENA)

    # ── visited bookkeeping ──

    def mark_visited(self, x, y):
        self.visited.add(self.pos_to_cell(x, y))

    def merge_remote(self, cells):
        """Merge cells received from ``/visited_zones``."""
        self.visited.update(cells)

    # ── serialise / deserialise for the topic ──

    def encode_visited(self):
        """→ ``'0:1,2:3,4:5'``"""
        return ','.join(f'{i}:{j}' for i, j in sorted(self.visited))

    @staticmethod
    def decode_visited(s):
        """→ set of (i,j)"""
        cells = set()
        for part in s.split(','):
            if ':' in part:
                a, b = part.split(':')
                cells.add((int(a), int(b)))
        return cells

    # ── target selection ──

    def get_target(self, robot_x, robot_y):
        """
        Return ``(x, y)`` of the nearest un-visited cell centre.
        Resets the grid when all cells have been visited.
        """
        self.mark_visited(robot_x, robot_y)

        # keep current target if still unvisited
        if self.current_target is not None:
            ci, cj = self.pos_to_cell(*self.current_target)
            if (ci, cj) not in self.visited:
                return self.current_target

        best, best_d = None, float('inf')
        for i in range(NX):
            for j in range(NY):
                if (i, j) in self.visited:
                    continue
                cx, cy = self.cell_to_pos(i, j)
                d = math.sqrt((robot_x - cx) ** 2 + (robot_y - cy) ** 2)
                if d < best_d:
                    best = (cx, cy)
                    best_d = d

        if best is None:
            # all visited → reset
            self.visited.clear()
            return self.get_target(robot_x, robot_y)

        self.current_target = best
        return best
