"""LiDAR-based obstacle avoidance — pure computation, no ROS deps."""

import math

# ── Tunables ──
SAFE_DIST = 1.0       # start reacting
CLOSE_DIST = 0.35     # sharp swerve
MAX_VX = 0.30
MIN_VX = 0.10
MAX_WZ = 1.5


def _region_min(ranges, start_deg, end_deg, n):
    """Minimum valid range in angular sector [start_deg, end_deg)."""
    if n == 0:
        return 10.0
    dps = 360.0 / n
    s = int(round(start_deg / dps)) % n
    e = int(round(end_deg / dps)) % n

    vals = []
    if s < e:
        indices = range(s, e)
    else:                              # wraps around 0°
        indices = list(range(s, n)) + list(range(0, e))

    for i in indices:
        r = ranges[i]
        if not math.isinf(r) and not math.isnan(r) and r > 0.05:
            vals.append(r)
    return min(vals) if vals else 10.0


def compute_avoidance(ranges):
    """Return dict  {vx, wz, front, f_left, f_right, left, right, rear}."""
    n = len(ranges)
    empty = dict(vx=MIN_VX, wz=0.0, front=10, f_left=10,
                 f_right=10, left=10, right=10, rear=10)
    if n == 0:
        return empty

    front   = _region_min(ranges, 345, 15, n)
    f_left  = _region_min(ranges, 15,  60, n)
    f_right = _region_min(ranges, 300, 345, n)
    left    = _region_min(ranges, 60, 100, n)
    right   = _region_min(ranges, 260, 300, n)
    rear    = _region_min(ranges, 160, 200, n)

    vx, wz = MAX_VX, 0.0
    clearer_right = (f_right >= f_left)

    # Boxed in: front + both sides close
    if front < CLOSE_DIST and f_left < CLOSE_DIST and f_right < CLOSE_DIST:
        vx = -MIN_VX if rear > CLOSE_DIST else MIN_VX
        wz = MAX_WZ

    # Close front obstacle
    elif front < CLOSE_DIST:
        vx = -MIN_VX if rear > CLOSE_DIST else MIN_VX
        wz = MAX_WZ if clearer_right else -MAX_WZ

    # Approaching obstacle
    elif front < SAFE_DIST:
        ratio = (front - CLOSE_DIST) / (SAFE_DIST - CLOSE_DIST)
        vx = MIN_VX + (MAX_VX - MIN_VX) * ratio
        turn = MAX_WZ * (1.0 - ratio)
        wz = turn if clearer_right else -turn

    # Side obstacle
    elif f_left < SAFE_DIST or f_right < SAFE_DIST:
        vx = MAX_VX * 0.8
        wz = -0.5 if f_left < f_right else 0.5

    return dict(vx=vx, wz=wz, front=front, f_left=f_left,
                f_right=f_right, left=left, right=right, rear=rear)
