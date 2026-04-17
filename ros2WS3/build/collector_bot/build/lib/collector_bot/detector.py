"""
Camera HSV detection + LiDAR fusion for world-frame localisation.
Multi-frame tracking avoids single-frame misses.
"""

import math
import time

import cv2
import numpy as np

# ── HSV thresholds ──
_RED_LO1 = np.array([0,   120, 80])
_RED_HI1 = np.array([10,  255, 255])
_RED_LO2 = np.array([165, 120, 80])
_RED_HI2 = np.array([180, 255, 255])

_BLUE_LO = np.array([95,  120, 80])
_BLUE_HI = np.array([135, 255, 255])

MIN_CONTOUR = 250            # px² noise gate
H_FOV       = 1.2            # camera horizontal FOV (rad)
IMG_W       = 640

# ── Tracking parameters ──
CONFIRM_FRAMES   = 3         # detections needed before "confirmed"
MATCH_RADIUS     = 0.5       # metres — same-object matching tolerance
TRACK_EXPIRY     = 2.0       # seconds without update → drop track


# ─────────────────────────────────────────────────────────────
class TrackedObject:
    """EMA-smoothed world-frame track of a detected object."""

    __slots__ = ('obj_type', 'wx', 'wy', 'confidence', 'last_seen')

    def __init__(self, obj_type, wx, wy):
        self.obj_type   = obj_type
        self.wx         = wx
        self.wy         = wy
        self.confidence = 1
        self.last_seen  = time.time()

    def update(self, wx, wy):
        a = 0.35
        self.wx = a * wx + (1 - a) * self.wx
        self.wy = a * wy + (1 - a) * self.wy
        self.confidence = min(self.confidence + 1, 15)
        self.last_seen = time.time()

    @property
    def confirmed(self):
        return self.confidence >= CONFIRM_FRAMES


# ─────────────────────────────────────────────────────────────
def detect_colors(cv_image):
    """
    HSV colour filter.

    Returns list[(obj_type, cx, cy, area)]  sorted by area desc.
    """
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    mask_red = (cv2.inRange(hsv, _RED_LO1, _RED_HI1) |
                cv2.inRange(hsv, _RED_LO2, _RED_HI2))
    mask_blue = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)

    results = []
    for mask, obj_type in [(mask_red, 'cube'), (mask_blue, 'sphere')]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            results.append((obj_type, cx, cy, area))

    results.sort(key=lambda r: r[3], reverse=True)
    return results


def fuse_with_lidar(detections, scan_ranges, robot_x, robot_y, robot_yaw):
    """
    Convert pixel detections → world-frame (x, y) using LiDAR range.

    Falls back to area-based distance estimate when LiDAR misses.

    Returns list[(obj_type, world_x, world_y)].
    """
    n = len(scan_ranges)
    results = []

    for obj_type, cx, cy, area in detections:
        # pixel → bearing  (centre of image = 0 rad)
        bearing = (0.5 - cx / IMG_W) * H_FOV

        # ── Try LiDAR ──
        # LiDAR covers [-π, +π]: index 0 = -π (behind), index n/2 = 0 (front)
        range_est = None
        if n > 0:
            centre_idx = int(round((bearing + math.pi) * n / (2 * math.pi))) % n
            candidates = []
            for off in range(-4, 5):
                idx = (centre_idx + off) % n
                r = scan_ranges[idx]
                if not math.isinf(r) and not math.isnan(r) and 0.12 < r < 4.0:
                    candidates.append(r)
            if candidates:
                range_est = min(candidates)

        # ── Fallback: area-based estimate ──
        if range_est is None:
            obj_size = 0.15 if obj_type == 'cube' else 0.20
            pix_w = math.sqrt(area)
            if pix_w < 1:
                continue
            range_est = obj_size * IMG_W / (2 * math.tan(H_FOV / 2) * pix_w)
            range_est = min(range_est, 4.0)

        world_angle = robot_yaw + bearing
        wx = robot_x + range_est * math.cos(world_angle)
        wy = robot_y + range_est * math.sin(world_angle)
        results.append((obj_type, wx, wy))

    return results


def update_tracks(tracks, fused, now=None):
    """
    Merge new fused detections into existing TrackedObject list **in-place**.

    * Matches by type + distance < MATCH_RADIUS.
    * Creates new tracks for unmatched detections.
    * Expires stale tracks.

    Returns the (possibly shorter) track list.
    """
    if now is None:
        now = time.time()

    for obj_type, wx, wy in fused:
        matched = False
        for t in tracks:
            if t.obj_type != obj_type:
                continue
            d = math.sqrt((t.wx - wx) ** 2 + (t.wy - wy) ** 2)
            if d < MATCH_RADIUS:
                t.update(wx, wy)
                matched = True
                break
        if not matched:
            tracks.append(TrackedObject(obj_type, wx, wy))

    # expire
    tracks[:] = [t for t in tracks if now - t.last_seen < TRACK_EXPIRY]
    return tracks
