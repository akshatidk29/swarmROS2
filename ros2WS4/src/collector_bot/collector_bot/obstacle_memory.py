"""
Obstacle memory — fine-grained occupancy grid.

Records cells where the robot got stuck.  The explorer and navigator
consult this map to avoid revisiting blocked regions.

Resolution : 0.2 m per cell  →  60 × 60 = 3600 cells for a 12 m arena.
Decay      : blocked cells expire after BLOCK_TTL seconds.
"""

import math
import time

CELL   = 0.2       # metres per cell
HALF   = 6.0       # arena half-size
NX     = int(2 * HALF / CELL)   # 60
NY     = NX
BLOCK_TTL = 60.0   # seconds before a blocked mark expires
MARK_DEPTH = 1.0   # mark cells up to this far ahead when stuck


def _w2c(x, y):
    """World → cell index."""
    i = int((x + HALF) / CELL)
    j = int((y + HALF) / CELL)
    return max(0, min(NX - 1, i)), max(0, min(NY - 1, j))


def _c2w(i, j):
    """Cell centre → world."""
    return (i + 0.5) * CELL - HALF, (j + 0.5) * CELL - HALF


class ObstacleMemory:
    """Per-robot obstacle memory."""

    def __init__(self):
        self.blocked = {}      # (i, j) → expiry timestamp

    # ── mark ──

    def mark_stuck(self, robot_x, robot_y, heading_rad, scan_ranges):
        """Mark cells in the heading direction as blocked."""
        now = time.time()
        n = len(scan_ranges)
        # Sample a ±30° wedge in the heading direction
        for d_deg in range(-30, 31, 5):
            angle = heading_rad + math.radians(d_deg)
            # get LiDAR range at this angle
            r_est = MARK_DEPTH
            if n > 0:
                idx = int(round(((math.degrees(angle - heading_rad + math.pi) + 180) % 360)
                                * n / 360)) % n
                # We don't have the exact robot-frame mapping here,
                # so just use a fixed depth
                pass

            # Mark cells along this ray up to MARK_DEPTH
            for dist in [x * 0.2 for x in range(1, int(MARK_DEPTH / 0.2) + 1)]:
                wx = robot_x + dist * math.cos(angle)
                wy = robot_y + dist * math.sin(angle)
                if abs(wx) > HALF or abs(wy) > HALF:
                    break
                cell = _w2c(wx, wy)
                self.blocked[cell] = now + BLOCK_TTL

    def mark_point(self, x, y):
        """Directly mark a world point as blocked."""
        cell = _w2c(x, y)
        self.blocked[cell] = time.time() + BLOCK_TTL

    # ── query ──

    def is_blocked(self, x, y):
        """Check if a world coordinate is in a blocked cell."""
        self._expire()
        return _w2c(x, y) in self.blocked

    def is_path_blocked(self, x1, y1, x2, y2, step=0.3):
        """Check if the straight-line path has blocked cells."""
        dx, dy = x2 - x1, y2 - y1
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            return self.is_blocked(x1, y1)
        steps = int(dist / step) + 1
        for s in range(steps + 1):
            t = s / max(steps, 1)
            if self.is_blocked(x1 + t * dx, y1 + t * dy):
                return True
        return False

    def get_detour(self, robot_x, robot_y, goal_x, goal_y):
        """
        If direct path is blocked, return an intermediate waypoint
        that detours around the blocked region.  Returns None if clear.
        """
        if not self.is_path_blocked(robot_x, robot_y, goal_x, goal_y):
            return None

        # Try perpendicular offsets
        dx, dy = goal_x - robot_x, goal_y - robot_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.1:
            return None

        # Perpendicular unit vector
        px, py = -dy / dist, dx / dist

        best_wp = None
        best_d = float('inf')

        for sign in [1, -1]:
            for offset in [1.0, 1.5, 2.0, 2.5]:
                wp_x = robot_x + 0.5 * dx + sign * offset * px
                wp_y = robot_y + 0.5 * dy + sign * offset * py
                # Must be inside arena
                if abs(wp_x) > HALF - 0.5 or abs(wp_y) > HALF - 0.5:
                    continue
                # Must not be blocked itself
                if self.is_blocked(wp_x, wp_y):
                    continue
                # Path to waypoint must be clear
                if self.is_path_blocked(robot_x, robot_y, wp_x, wp_y):
                    continue
                d = math.sqrt((wp_x - goal_x) ** 2 + (wp_y - goal_y) ** 2)
                if d < best_d:
                    best_d = d
                    best_wp = (wp_x, wp_y)

        return best_wp

    # ── maintenance ──

    def _expire(self):
        now = time.time()
        expired = [k for k, v in self.blocked.items() if v < now]
        for k in expired:
            del self.blocked[k]

    def clear(self):
        self.blocked.clear()

    # ── serialisation for topic sharing ──

    def encode(self):
        """Encode blocked cells as string for ROS topic."""
        self._expire()
        return ','.join(f'{i}:{j}' for (i, j) in self.blocked.keys())

    @staticmethod
    def decode(s):
        """Decode string → set of (i,j) cells."""
        cells = set()
        for part in s.split(','):
            if ':' in part:
                a, b = part.split(':')
                try:
                    cells.add((int(a), int(b)))
                except ValueError:
                    pass
        return cells

    def merge_remote(self, cells):
        """Merge blocked cells from another robot."""
        now = time.time()
        for cell in cells:
            if cell not in self.blocked:
                self.blocked[cell] = now + BLOCK_TTL * 0.5  # shorter TTL for remote
