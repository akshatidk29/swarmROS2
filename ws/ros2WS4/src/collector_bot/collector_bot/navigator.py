"""
Omni-drive go-to-point navigator.

Single reusable function for approaching objects, delivering to baskets,
and frontier exploration targets.
"""

import math

# ── Tunables ──
SPEED_GAIN   = 0.6       # proportional gain on distance
ANGULAR_GAIN = 2.5       # proportional gain on heading error
MAX_SPEED    = 0.50      # m/s  per axis
MAX_WZ       = 2.0       # rad/s


def _norm(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def go_to_point(target_x, target_y,
                robot_x, robot_y, robot_yaw,
                arrival_dist=0.3):
    """
    Compute omni-drive velocity to reach *target* from current pose.

    Returns
    -------
    (vx, vy, wz, arrived)
        vx, vy in **robot frame**; wz rotates to face target; arrived is bool.
    """
    dx = target_x - robot_x
    dy = target_y - robot_y
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < arrival_dist:
        return 0.0, 0.0, 0.0, True

    # global angle to target → robot-local angle
    angle_global = math.atan2(dy, dx)
    angle_local  = _norm(angle_global - robot_yaw)

    speed = min(MAX_SPEED, dist * SPEED_GAIN)
    vx = speed * math.cos(angle_local)
    vy = speed * math.sin(angle_local)

    # rotate to face target (smooth)
    heading_err = _norm(angle_global - robot_yaw)
    wz = max(-MAX_WZ, min(MAX_WZ, heading_err * ANGULAR_GAIN))

    return vx, vy, wz, False
