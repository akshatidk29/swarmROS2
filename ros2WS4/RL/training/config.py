"""
Shared constants between RL training env and ROS2 brain_rl.

All arena geometry, object positions, sensor specs, and observation
dimensions are defined here so training ↔ deployment stay in sync.
"""

import numpy as np

# ═══════════════ ARENA ═══════════════
ARENA_HALF = 6.0          # half-side of the 12 × 12 m arena
ARENA_SIZE = 12.0

# ═══════════════ ROBOTS ═══════════════
N_ROBOTS   = 4
ROBOT_RADIUS = 0.20       # collision radius (m)
MAX_SPEED  = 0.50         # m/s per axis
MAX_WZ     = 2.0          # rad/s

ROBOT_STARTS = [
    ( 0.0, -2.0),
    ( 2.5, -4.0),
    (-2.5, -4.0),
    ( 0.0, -4.5),
]

# ═══════════════ OBJECTS ═══════════════
OBJECTS = [
    ('cube_1',   'cube',    1.5,   1.5),
    ('cube_2',   'cube',   -3.0,   0.5),
    ('cube_3',   'cube',    4.0,  -3.0),
    ('cube_4',   'cube',   -1.0,  -3.5),
    ('cube_5',   'cube',    2.0,   3.5),
    ('sphere_1', 'sphere', -2.0,   3.0),
    ('sphere_2', 'sphere',  1.0,  -1.5),
    ('sphere_3', 'sphere', -4.0,  -1.0),
    ('sphere_4', 'sphere',  3.5,   1.0),
    ('sphere_5', 'sphere', -0.5,   4.0),
]
N_OBJECTS = len(OBJECTS)

CUBE_BASKET   = np.array([ 5.0,  5.0])
SPHERE_BASKET = np.array([-5.0,  5.0])

PICK_DIST   = 0.60        # m — close enough to auto-pick
BASKET_DIST = 0.60        # m — close enough to auto-drop

# ═══════════════ OBSTACLES (x, y, half_w, half_h) ═══════════════
OBSTACLES = [
    ( 3.0,  0.0, 0.25, 1.0),
    (-2.0, -2.0, 0.25, 1.0),
    ( 0.0,  3.0, 1.0,  0.25),
    (-4.0,  2.0, 0.25, 1.0),
]

# ═══════════════ SENSORS ═══════════════
LIDAR_RAYS     = 36       # every 10°
LIDAR_MAX_DIST = 6.0      # m
CAMERA_FOV     = 1.2      # rad ≈ 69°
CAMERA_RANGE   = 4.0      # m
N_DETECTIONS   = 3        # top detections in obs

# ═══════════════ OBSERVATION / ACTION ═══════════════
OBS_DIM = 36 + 9 + 3 + 5 + 9 + 3   # = 65
ACT_DIM = 3                          # [vx, vy, wz]

# ═══════════════ REWARDS ═══════════════
R_TARGET_APPROACH  =  0.5   # per metre of distance closed
R_TARGET_REACHED   = 15.0
R_PICK             = 10.0
R_DROP_CORRECT     = 30.0
R_DROP_WRONG       = -15.0
R_WALL_COLLISION   = -3.0
R_ROBOT_COLLISION  = -5.0
R_ROBOT_PROXIMITY  = -0.5   # scaled by closeness
R_NEW_ZONE         =  2.0
R_IDLE             = -0.02
R_SMOOTH           =  0.05
R_ALL_DONE         = 100.0

# ═══════════════ EPISODE ═══════════════
MAX_STEPS  = 3000
DT         = 0.1          # seconds per step

# ═══════════════ GRID (for exploration) ═══════════════
GRID_RES = 2.0
GRID_NX  = int(ARENA_SIZE / GRID_RES)
GRID_NY  = int(ARENA_SIZE / GRID_RES)
