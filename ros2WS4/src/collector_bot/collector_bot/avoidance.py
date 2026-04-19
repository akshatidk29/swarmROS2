"""
Potential-field obstacle avoidance with robot-pose awareness.

Returns **repulsive** velocities (vx, vy) in the robot frame.
The caller adds these to the attractive (navigation) velocity.

IMPORTANT: LiDAR in Gazebo scans from -π to +π.
  index 0   = -π  (directly behind)
  index n/2 = 0   (directly in front)
  index n-1 = +π  (behind, other side)
"""

import math

# ── Tunables ──
INFLUENCE_DIST       = 1.2
REPULSION_GAIN       = 0.55
ROBOT_INFLUENCE_DIST = 1.8
ROBOT_REPULSION_GAIN = 0.65
MAX_REP_SPEED        = 0.50
SCAN_STEP_DEG        = 5


def _deg_to_idx(deg, n):
    """Convert robot-frame degrees (0=front, CCW+) to LiDAR array index.

    LiDAR covers [-π … +π],  so index 0 = -180°, index n/2 = 0° (front).
    """
    return int(round(((deg + 180) % 360) * n / 360)) % n


def compute_repulsion(scan_ranges, other_poses, robot_x, robot_y, robot_yaw):
    """
    Returns (rep_vx, rep_vy) in **robot frame**.
    """
    rx, ry = 0.0, 0.0
    n = len(scan_ranges)

    if n > 0:
        for deg in range(0, 360, SCAN_STEP_DEG):
            idx = _deg_to_idx(deg, n)
            r = scan_ranges[idx]
            if math.isinf(r) or math.isnan(r) or r < 0.05 or r > INFLUENCE_DIST:
                continue
            force = REPULSION_GAIN * (1.0 / r - 1.0 / INFLUENCE_DIST)
            angle = math.radians(deg)
            rx -= force * math.cos(angle)
            ry -= force * math.sin(angle)

    for ox, oy in other_poses:
        dx = ox - robot_x
        dy = oy - robot_y
        d = math.sqrt(dx * dx + dy * dy)
        if d < 0.01 or d > ROBOT_INFLUENCE_DIST:
            continue
        force = ROBOT_REPULSION_GAIN * (1.0 / d - 1.0 / ROBOT_INFLUENCE_DIST)
        global_angle = math.atan2(dy, dx)
        local_angle = global_angle - robot_yaw
        rx -= force * math.cos(local_angle)
        ry -= force * math.sin(local_angle)

    mag = math.sqrt(rx * rx + ry * ry)
    if mag > MAX_REP_SPEED:
        rx = rx / mag * MAX_REP_SPEED
        ry = ry / mag * MAX_REP_SPEED

    return rx, ry


def find_clearest_direction(scan_ranges, target_angle_rad=None):
    """
    Scan 8 compass directions, score by clearance + target alignment.
    Returns (angle_rad, score) — angle in robot frame.
    """
    n = len(scan_ranges)
    if n == 0:
        return math.pi, 0.0

    best_angle = math.pi
    best_score = -1.0

    for deg in range(0, 360, 45):
        total, count = 0.0, 0
        for d in range(deg - 20, deg + 21, 5):
            idx = _deg_to_idx(d, n)
            r = scan_ranges[idx]
            if not math.isinf(r) and not math.isnan(r) and r > 0.05:
                total += r
                count += 1
        avg = (total / count) if count else 0.0

        if avg < 0.25:
            continue

        score = avg
        if target_angle_rad is not None:
            diff = math.radians(deg) - target_angle_rad
            alignment = math.cos(diff)
            score += alignment * 0.5

        if score > best_score:
            best_score = score
            best_angle = math.radians(deg)

    return best_angle, max(best_score, 0.0)


def min_range(scan_ranges):
    """Return the smallest valid LiDAR reading (metres).  10.0 if empty."""
    best = 10.0
    for r in scan_ranges:
        if not math.isinf(r) and not math.isnan(r) and 0.05 < r < best:
            best = r
    return best
