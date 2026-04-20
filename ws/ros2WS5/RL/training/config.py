"""
Shared constants — single source of truth for RL training and ROS2 deployment.

EVERY module in the system imports from this file. Do NOT duplicate constants.
ROS2 nodes access these via src/collector_bot/collector_bot/constants.py bridge.

Gazebo URDF LiDAR:  360 samples, [-π, +π], min 0.12m, max 5.0m
Training LiDAR:     18 down-sampled rays, ray 0 = forward (heading)
"""

import numpy as np

# ═══════════════ ARENA ═══════════════
# Gazebo walls are 0.3m thick at ±6.0m → inner face at ±5.85m.
# We use 6.0 as the nominal half-size; wall collision clamps with ROBOT_RADIUS.
ARENA_HALF = 6.0
ARENA_SIZE = 12.0

# ═══════════════ ROBOTS ═══════════════
N_ROBOTS     = 4
ROBOT_RADIUS = 0.20
MAX_SPEED    = 0.50    # m/s per axis
MAX_WZ       = 2.0     # rad/s

# Spawn: pick random center, place 4 robots in cross pattern at ±offset
SPAWN_CLUSTER_SPREAD = 1.0  # metres from center to each robot

# ═══════════════ OBJECTS ═══════════════
# Names and types only. Positions are randomized at reset.
# Kept for Gazebo model name references (delete/spawn services need names).
OBJECT_DEFS = [
    ('cube_1',   'cube'),
    ('cube_2',   'cube'),
    ('cube_3',   'cube'),
    ('cube_4',   'cube'),
    ('cube_5',   'cube'),
    ('sphere_1', 'sphere'),
    ('sphere_2', 'sphere'),
    ('sphere_3', 'sphere'),
    ('sphere_4', 'sphere'),
    ('sphere_5', 'sphere'),
]
N_OBJECTS = len(OBJECT_DEFS)

CUBE_BASKET   = np.array([ 5.0,  5.0])
SPHERE_BASKET = np.array([-5.0,  5.0])

PICK_DIST   = 0.30
BASKET_DIST = 0.60

# ═══════════════ OBSTACLES (x, y, half_w, half_h) ═══════════════
# Must match arena.world Gazebo geometry exactly.
# Derived from SDF <pose> and <size>/2.
OBSTACLES = [
    # Corner pillars (0.35×0.35 boxes)
    ( 5.9,   5.9,  0.175, 0.175),
    (-5.9,   5.9,  0.175, 0.175),
    ( 5.9,  -5.9,  0.175, 0.175),
    (-5.9,  -5.9,  0.175, 0.175),
    # obs_box_1: wooden crate 1.5×0.3 at (3, 0)
    ( 3.0,   0.0,  0.75,  0.15),
    # obs_cyl_1: barrel r=0.4 at (-2, -2)
    (-2.0,  -2.0,  0.40,  0.40),
    ( 2.0,  -2.0,  0.40,  0.40),
    # obs_box_2: shelving 0.3×2.0 at (0, 3)
    ( 0.0,   3.0,  0.15,  1.00),
    # obs_box_3: pallet crate 1.0×0.3 at (-4, 2)
    (-4.0,   2.0,  0.50,  0.15),
    # shelf_east_1: 0.5×2.5 at (5.5, -3)
    ( 5.5,  -3.0,  0.25,  1.25),
    # shelf_west_1: 0.5×2.5 at (-5.5, -3)
    (-5.5,  -3.0,  0.25,  1.25),
    # deco_barrel_1: r=0.25 at (5.3, -5.3)
    ( 5.3,  -5.3,  0.25,  0.25),
    # deco_barrel_2: r=0.25 at (4.7, -5.3)
    ( 4.7,  -5.3,  0.25,  0.25),
    # deco_barrel_3: r=0.25 at (-5.3, -5.3)
    (-5.3,  -5.3,  0.25,  0.25),
    # deco_crate_stack: 0.5×0.5 at (5.4, 5.4)
    ( 5.4,   5.4,  0.25,  0.25),
    # cone_1: r=0.12 at (2.0, -4.5)
    ( 2.0,  -4.5,  0.125, 0.125),
    # cone_2: r=0.12 at (-3.5, 4.5)
    (-3.5,   4.5,  0.125, 0.125),
    # cone_3: r=0.12 at (4.5, 4.5)
    ( 4.5,   4.5,  0.125, 0.125),
]

# ═══════════════ SENSORS ═══════════════
LIDAR_RAYS     = 18        # down-sampled from 360 Gazebo rays
LIDAR_MAX_DIST = 5.0       # metres — matches URDF <max>5.0</max>
LIDAR_MIN_DIST = 0.12      # metres — matches URDF <min>0.12</min>
RAYCAST_STEP   = 0.10      # metres — step size for PyGame ray marching

CAMERA_FOV     = 1.2       # rad ≈ 69°
CAMERA_RANGE   = 4.0       # metres
N_DETECTIONS   = 3         # camera detection slots in obs

# ═══════════════ SAFETY ═══════════════
# Hysteresis prevents oscillation between safety override and policy.
SAFETY_DIST_ENTER = 0.25   # activate safety override at this LiDAR distance
SAFETY_DIST_EXIT  = 0.40   # deactivate only after clearing this distance

# ═══════════════ ACTION SMOOTHING ═══════════════
ACTION_EMA = 0.3  # exponential moving average blend: cmd = (1-α)*new + α*prev

# ═══════════════ CARRY ═══════════════
CARRY_TIMEOUT_STEPS = 200

# ═══════════════ OBSERVATION / ACTION ═══════════════
# Lean 43-dim observation (no SLAM, no occ grid, no shared objects, no absolute yaw):
#   [0  : 18]  LiDAR (18 normalized distances, ray 0 = forward)
#   [18 : 27]  Other robots: 3 × (rel_x, rel_y, dist)
#   [27 : 30]  Target: (sin_angle, cos_angle, dist_norm)
#   [30 : 39]  Camera detections: 3 × (type_sign, bearing_norm, dist_norm)
#   [39 : 42]  Basket: (sin_angle, cos_angle, dist_norm)
#   [42]       Carry overtime fraction
OBS_DIM = LIDAR_RAYS + 9 + 3 + (N_DETECTIONS * 3) + 3 + 1   # 18 + 25 = 43
ACT_DIM = 3

# ═══════════════ REWARDS (defaults — overridden by curriculum) ═══════════════
R_STEP            = -0.01   # time pressure
R_NEW_CELL        =  0.0    # disabled (no SLAM)
R_REVISIT         =  0.0    # disabled (no SLAM)
R_APPROACH        =  0.5    # per metre closed toward target
R_PICK            = 15.0    # successful auto-pick
R_BROADCAST_BONUS =  1.0    # first detection of an object via camera
R_COLLAB_BONUS    =  3.0    # picking up object broadcast by another agent
R_DROP_CORRECT    = 40.0    # correct basket delivery
R_DROP_WRONG      = -20.0   # wrong basket
R_CARRY_OVERTIME  = -0.05   # per-step penalty beyond CARRY_TIMEOUT_STEPS
R_WALL_HIT        = -9.0    # wall / obstacle bounce
R_ROBOT_COLLISION = -12.0   # within 2×ROBOT_RADIUS
R_PROXIMITY_HARD  = -2.0    # < 0.5m between robots (scaled by closeness)
R_PROXIMITY_SOFT  = -0.5    # 0.5–1.5m between robots (scaled)
R_OBSTACLE_NEAR   = -0.3    # per step within 0.5m of any static obstacle
R_DANGLE          = -0.5    # stuck near obstacle > 3 steps
R_JERK            = -0.04   # penalize jerk (acceleration change)
R_ALL_DONE        = 100.0   # all objects placed

# ═══════════════ CURRICULUM PHASES ═══════════════
CURRICULUM = {
    'phase_1': {
        'name': 'Pick & Deliver',
        'timesteps': 200_000,
        'rewards': {
            'R_PICK':           15.0,
            'R_DROP_CORRECT':   40.0,
            'R_APPROACH':        1.0,
            'R_BROADCAST_BONUS': 0.5,
            'R_COLLAB_BONUS':    1.0,
            'R_WALL_HIT':      -2.0,
            'R_ROBOT_COLLISION':-2.0,
            'R_PROXIMITY_HARD': -0.5,
            'R_PROXIMITY_SOFT': -0.1,
            'R_OBSTACLE_NEAR':  -0.1,
            'R_DANGLE':         -0.1,
            'R_ALL_DONE':      100.0,
        },
    },
    'phase_2': {
        'name': 'Collision Avoidance',
        'timesteps': 200_000,
        'rewards': {
            'R_PICK':           15.0,
            'R_DROP_CORRECT':   40.0,
            'R_APPROACH':        0.5,
            'R_BROADCAST_BONUS': 1.0,
            'R_COLLAB_BONUS':    3.0,
            'R_WALL_HIT':      -9.0,
            'R_ROBOT_COLLISION':-12.0,
            'R_PROXIMITY_HARD': -2.0,
            'R_PROXIMITY_SOFT': -0.5,
            'R_OBSTACLE_NEAR':  -0.3,
            'R_DANGLE':         -0.5,
            'R_ALL_DONE':      100.0,
        },
    },
    'phase_3': {
        'name': 'Full Exploration',
        'timesteps': 400_000,
        'rewards': {
            'R_PICK':           15.0,
            'R_DROP_CORRECT':   40.0,
            'R_APPROACH':        0.5,
            'R_BROADCAST_BONUS': 1.0,
            'R_COLLAB_BONUS':    3.0,
            'R_WALL_HIT':      -9.0,
            'R_ROBOT_COLLISION':-12.0,
            'R_PROXIMITY_HARD': -2.0,
            'R_PROXIMITY_SOFT': -0.5,
            'R_OBSTACLE_NEAR':  -0.3,
            'R_DANGLE':         -0.5,
            'R_JERK':           -0.04,
            'R_ALL_DONE':      100.0,
        },
    },
}

# ═══════════════ EPISODE ═══════════════
MAX_STEPS      = 600    # per robot (600 × 4 = 2400 env steps/episode)
N_STEPS_PPO    = 2048   # PPO rollout length
DT             = 0.1    # seconds per step
N_ENVS_DEFAULT = 4      # SubprocVecEnv workers

# ═══════════════ EXPLORATION ═══════════════
# Random-walk exploration: assign random target, change every N steps
EXPLORE_RETARGET_STEPS = 50   # reassign random target after this many steps
