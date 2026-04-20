"""
Pure geometry helpers for the 12×12 m arena.

Matches the Gazebo arena.world geometry exactly:
  - Walls at ±6.0m (0.3m thick, inner face at ±5.85m)
  - 17 static obstacles (pillars, boxes, cylinders, shelves, barrels, cones)
  - LiDAR convention: ray 0 = forward (heading), CCW positive, 360° coverage

No gymnasium or pygame dependency — importable from both training env
and ROS2 brain for identical sensor simulation.
"""

import math
from config import (
    ARENA_HALF, OBSTACLES, ROBOT_RADIUS,
    LIDAR_RAYS, LIDAR_MAX_DIST, LIDAR_MIN_DIST,
    CAMERA_FOV, CAMERA_RANGE, RAYCAST_STEP,
)


def norm_angle(a):
    """Normalize angle to [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def rect_contains(ox, oy, hw, hh, px, py, margin=0.0):
    """True if point (px,py) is inside axis-aligned rect with half-widths."""
    return abs(px - ox) < hw + margin and abs(py - oy) < hh + margin


def clamp_to_arena(x, y, margin=0.0):
    """Clamp (x,y) to arena bounds with margin."""
    x = max(-ARENA_HALF + margin, min(ARENA_HALF - margin, x))
    y = max(-ARENA_HALF + margin, min(ARENA_HALF - margin, y))
    return x, y


def check_wall_collision(new_x, new_y, margin=ROBOT_RADIUS):
    """Return (clamped_x, clamped_y, hit_wall).

    Wall inner face is at ±5.85m (6.0 - 0.15 wall half-thickness).
    Robot centre can't go past ±(5.85 - ROBOT_RADIUS).
    """
    wall_inner = ARENA_HALF - 0.15  # 5.85m — inner face of 0.3m-thick wall
    limit = wall_inner - margin
    hit = False
    if new_x < -limit:
        new_x = -limit
        hit = True
    if new_x > limit:
        new_x = limit
        hit = True
    if new_y < -limit:
        new_y = -limit
        hit = True
    if new_y > limit:
        new_y = limit
        hit = True
    return new_x, new_y, hit


def check_obstacle_collision(new_x, new_y, old_x, old_y, margin=ROBOT_RADIUS):
    """Return (final_x, final_y, hit_obstacle)."""
    for ox, oy, hw, hh in OBSTACLES:
        if rect_contains(ox, oy, hw, hh, new_x, new_y, margin):
            return old_x, old_y, True
    return new_x, new_y, False


def point_near_any_obstacle(px, py, margin):
    """True if point is within margin of any obstacle."""
    for ox, oy, hw, hh in OBSTACLES:
        if abs(px - ox) < hw + margin and abs(py - oy) < hh + margin:
            return True
    return False


def raycast(ox, oy, angle, max_dist, agents, objects, exclude_idx=-1,
            step=None):
    """
    Step-based raycasting for LiDAR simulation.

    Convention: angle is in world frame (absolute).
    Ray 0 of the LiDAR array corresponds to agent heading direction.

    Returns distance to nearest hit (wall, obstacle, robot, or object).
    Returns max_dist if nothing hit.
    Minimum return distance is LIDAR_MIN_DIST (matches Gazebo URDF).
    """
    if step is None:
        step = RAYCAST_STEP
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    wall_inner = ARENA_HALF - 0.15  # match Gazebo wall inner face

    for s in range(1, int(max_dist / step) + 1):
        d = s * step
        px = ox + d * cos_a
        py = oy + d * sin_a

        # Walls (inner face at ±5.85)
        if abs(px) > wall_inner or abs(py) > wall_inner:
            return max(d, LIDAR_MIN_DIST)

        # Obstacles
        for bx, by, hw, hh in OBSTACLES:
            if rect_contains(bx, by, hw, hh, px, py):
                return max(d, LIDAR_MIN_DIST)

        # Other robots
        for agent in agents:
            if agent.idx == exclude_idx:
                continue
            if (px - agent.x) ** 2 + (py - agent.y) ** 2 < ROBOT_RADIUS ** 2:
                return max(d, LIDAR_MIN_DIST)

        # Objects (small collision radius)
        for obj in objects:
            if not obj.get('alive', True):
                continue
            if (px - obj['x']) ** 2 + (py - obj['y']) ** 2 < 0.1 ** 2:
                return max(d, LIDAR_MIN_DIST)

    return max_dist


def detect_in_fov(agent_x, agent_y, agent_yaw, objects):
    """
    Detect alive objects within camera FOV.

    Returns list of (obj_name, obj_type, bearing, dist) sorted by distance.
    In the training env, positions are ground truth — no estimation error.
    Bearing is relative to agent heading.
    """
    half_fov = CAMERA_FOV / 2
    dets = []
    for obj in objects:
        if not obj.get('alive', True):
            continue
        dx = obj['x'] - agent_x
        dy = obj['y'] - agent_y
        d = math.sqrt(dx * dx + dy * dy)
        if d > CAMERA_RANGE or d < 0.1:
            continue
        angle = norm_angle(math.atan2(dy, dx) - agent_yaw)
        if abs(angle) < half_fov:
            dets.append((obj['name'], obj['type'], angle, d))
    dets.sort(key=lambda x: x[3])
    return dets


def compute_repulsion_from_lidar(lidar_norm, n_rays, max_dist):
    """
    Compute repulsive (vx, vy) in robot frame from normalized LiDAR.

    Ray 0 = forward (heading), rays go CCW.
    Used as deterministic safety override inside the env step.
    """
    rx, ry = 0.0, 0.0
    gain = 0.6
    influence = 1.0

    for i in range(n_rays):
        r = lidar_norm[i] * max_dist
        if r < LIDAR_MIN_DIST or r > influence:
            continue
        force = gain * (1.0 / max(r, 0.05) - 1.0 / influence)
        # Ray i angle in robot frame
        angle = (i / n_rays) * 2 * math.pi
        # Repulsion is AWAY from obstacle
        rx -= force * math.cos(angle)
        ry -= force * math.sin(angle)

    mag = math.sqrt(rx * rx + ry * ry)
    if mag > 0.4:
        rx = rx / mag * 0.4
        ry = ry / mag * 0.4
    return rx, ry
