# ros2WS5 — Multi-Agent Swarm Collector

> Production-ready multi-agent reinforcement learning system for collaborative object collection and transport using 4 omni-drive robots in a 12×12m Gazebo arena.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Architecture Deep-Dive](#3-architecture-deep-dive)
4. [The RL Training Environment](#4-the-rl-training-environment)
5. [Observation Space (85-dim)](#5-observation-space-85-dim)
6. [Action Space](#6-action-space)
7. [Reward Function](#7-reward-function)
8. [Per-Agent SLAM Occupancy Grid](#8-per-agent-slam-occupancy-grid)
9. [The Gazebo Arena](#9-the-gazebo-arena)
10. [ROS2 Node Architecture](#10-ros2-node-architecture)
11. [ROS2 Topics Reference](#11-ros2-topics-reference)
12. [Safety System](#12-safety-system)
13. [Training Pipeline](#13-training-pipeline)
14. [Fine-Tuning Pipeline](#14-fine-tuning-pipeline)
15. [Configuration — Single Source of Truth](#15-configuration--single-source-of-truth)
16. [Installation & Setup](#16-installation--setup)
17. [Running the System](#17-running-the-system)
18. [File-by-File Reference](#18-file-by-file-reference)
19. [Key Design Decisions](#19-key-design-decisions)
20. [Known Limitations & Future Work](#20-known-limitations--future-work)

---

## 1. System Overview

Four omni-drive robots are placed in a warehouse-style 12×12m arena containing:
- **10 objects**: 5 cubes and 5 spheres, placed at random positions each episode
- **2 baskets**: a cube basket at `(+5, +5)` and a sphere basket at `(-5, +5)`
- **17 static obstacles**: pillars, shelves, barrels, crates, and cones (matching the Gazebo world exactly)
- **Walls** at ±6.0m (0.3m thick, inner face at ±5.85m)

The robots must **explore**, **find objects via camera**, **pick them up**, and **deliver them to the correct basket**. No robot is told where any object is — they must discover objects by looking with their cameras and share discoveries by broadcasting over ROS2.

The policy is learned using **IPPO (Independent Proximal Policy Optimization)** — a single shared MLP policy trained across all 4 agents simultaneously. The same weights control all 4 robots, giving emergent cooperative behaviour through independent decision-making.

```
                ┌─────────────────────────────────────────┐
                │              12m × 12m Arena             │
                │                                          │
                │  [sphere basket]        [cube basket]   │
                │      (-5,+5)               (+5,+5)      │
                │                                          │
                │  ┌──┐  ┌──┐           ┌──┐  ┌──┐      │
                │  │██│  │  │   ████    │  │  │  │      │ (obstacles)
                │  └──┘  └──┘           └──┘  └──┘      │
                │                                          │
                │   ● robot_1    ● robot_2                 │
                │   ● robot_3    ● robot_4                 │
                │                                          │
                │   ○ cubes/spheres (randomized)           │
                └─────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
ros2WS5/
├── RL/
│   ├── training/                   # PyGame training environment
│   │   ├── config.py               # ← SINGLE SOURCE OF TRUTH (all constants)
│   │   ├── train.py                # PPO training script
│   │   ├── paths.py                # Path helpers for models/logs
│   │   └── env/
│   │       ├── swarm_env.py        # Gymnasium multi-agent environment
│   │       ├── arena.py            # Geometry: walls, obstacles, raycasting
│   │       └── occupancy_grid.py   # Per-agent SLAM (48×48 grid)
│   ├── finetune/
│   │   ├── gazebo_env_wrapper.py   # Gymnasium wrapper over live Gazebo
│   │   └── finetune.py             # Fine-tuning script
│   ├── model/
│   │   └── policy/
│   │       ├── policy.zip          # Final trained model (after training)
│   │       └── best_model.zip      # Best checkpoint during training
│   └── requirements.txt
│
├── src/
│   └── collector_bot/              # ROS2 Python package
│       ├── collector_bot/          # Node source files
│       │   ├── constants.py        # Bridge → imports from RL/training/config.py
│       │   ├── swarm_brain.py      # Centralized RL inference node (main brain)
│       │   ├── safety_coordinator.py # Deterministic safety overrides
│       │   ├── sim_logger.py       # Event logging + HTML report generation
│       │   ├── detector.py         # Camera colour detection + LiDAR fusion
│       │   ├── explorer.py         # Coarse grid-based frontier explorer
│       │   ├── navigator.py        # Omni-drive go-to-point controller
│       │   ├── avoidance.py        # Potential-field repulsion (safety layer)
│       │   ├── gazebo_interface.py # Async Gazebo spawn/delete service wrappers
│       │   └── paths.py            # Path constants (model path, simulate dir)
│       ├── launch/
│       │   ├── simulation.launch.py # Main launch file
│       │   └── spawn_robot.launch.py # Single robot spawn helper
│       ├── worlds/
│       │   └── arena.world         # Gazebo SDF world (must match config.py)
│       ├── urdf/
│       │   └── robot.urdf.xacro    # Robot URDF (omni-drive + LiDAR + camera)
│       ├── setup.py
│       └── package.xml
│
└── simulate/
    └── logN/                       # Per-run simulation logs
        ├── events.jsonl            # Timestamped events (pick/drop/collision)
        ├── trajectory_map.png      # Top-down path visualization
        ├── report.html             # Full HTML report
        └── ...
```

---

## 3. Architecture Deep-Dive

### CTDE: Centralized Training, Decentralized Execution

- **Training**: All 4 agents share one policy. The Gymnasium env cycles through agents one at a time (current_agent index). From SB3's perspective this is a single-agent environment.
- **Deployment**: The `swarm_brain` node loads the policy once and runs batched inference for all 4 robots every 100ms.

### Data Flow (Training)

```
Episode Reset
  └─> _spawn_robots()      # cross-pattern, random centre
  └─> _randomize_objects() # 10 objects at random valid positions
  └─> _assign_targets()    # each agent gets frontier target (no objects known yet)

Step (agent i):
  1. Action from PPO → EMA smooth → scale to m/s
  2. Safety check: LiDAR min < SAFETY_DIST_ENTER → enable hysteresis override
  3. Physics: move robot, check wall/obstacle/robot collisions
  4. Occupancy grid update from LiDAR (filter out other robot positions)
  5. Camera detection in 69° FOV → update detected_objects cache → broadcast
  6. Merge broadcasts from other agents
  7. Auto-pick: robot within 0.6m of alive object → pick, publish, broadcast
  8. Auto-drop: robot within 0.6m of correct basket → drop
  9. Rewards: sum of all shaped components
 10. Return obs (85-dim) for agent (i+1)%4
```

### Data Flow (Deployment)

```
Gazebo running → 4 robots publishing /robot_N/{scan,odom,camera}
                                  ↓
                         swarm_brain._tick() @ 10Hz
                           ├─> per-robot: read LiDAR, odom, camera tracks
                           ├─> safety_coordinator override? → publish safety cmd
                           ├─> build 85-dim observation
                           ├─> PPO.predict(obs) → action
                           ├─> EMA smooth → pub /{robot_N}/cmd_vel
                           ├─> auto-pick via camera proximity
                           └─> auto-drop when near basket
```

---

## 4. The RL Training Environment

**File**: `RL/training/env/swarm_env.py`
**Class**: `SwarmCollectorEnv(gym.Env)`

### Key Design Points

#### Cross-Pattern Spawn
Each episode, a random centre point is picked in the lower half of the arena (`y < 0`). The 4 robots are spawned at:
```
robot_0: (cx + 1.0, cy + 0.0)   # right
robot_1: (cx - 1.0, cy + 0.0)   # left
robot_2: (cx + 0.0, cy + 1.0)   # top
robot_3: (cx + 0.0, cy - 1.0)   # bottom
```
Before accepting the configuration, all 4 positions are validated:
- Not outside arena walls
- Not within `ROBOT_RADIUS + 0.3m` of any static obstacle

If 200 random attempts fail, a hardcoded fallback is used.

#### Object Randomization
Objects are placed fully randomly (no hardcoded nominal positions). For each object, up to 300 placement attempts are made checking:
1. Not within 1.2m of either basket
2. Not within 0.4m of any static obstacle
3. Not within 0.6m of any other object
4. Not within 1.0m of any robot spawn position

#### No Oracle Knowledge
Agents start each episode with empty `detected_objects` dicts. They only learn about objects by:
1. **Camera detection**: Objects inside the 69° FOV and within 4m are detected
2. **Broadcasts**: When an agent detects an object, it's added to `shared.broadcasts` and other agents pick it up in their next step

#### Per-Agent State Isolation
```python
class AgentState:
    occ_grid: OccupancyGrid      # each agent's own SLAM map
    detected_objects: dict        # what THIS agent has seen (no oracle)
    safety_active: bool           # hysteresis safety state
    prev_action: np.array         # for EMA smoothing
    prev_prev_action: np.array    # for jerk penalty calculation
```

#### Action EMA Smoothing
```python
smoothed = (1 - 0.3) * action + 0.3 * prev_action
# blend: 70% new policy output, 30% previous action
```
This prevents high-frequency velocity oscillations that cause physical damage in real robots.

#### Safety Hysteresis
```python
# Activate safety only when very close
if not agent.safety_active and lidar_min < 0.25:
    agent.safety_active = True
# Deactivate only after clearing enough distance (prevents oscillation)
elif agent.safety_active and lidar_min > 0.40:
    agent.safety_active = False
```
When safety is active: potential-field repulsion replaces policy output.

---

## 5. Observation Space (85-dim)

All values clamped to **[-1, 1]**. Same layout in training env, `swarm_brain._build_obs()`, and `GazeboEnvWrapper._build_obs()`.

| Indices | Dim | Name | Description |
|---------|-----|------|-------------|
| `[0:18]` | 18 | **LiDAR** | Normalized distances. Ray 0 = forward (heading), CCW. `0=close, 1=far` |
| `[18:43]` | 25 | **Occupancy patch** | 5×5 ego-centric grid. `-1=unknown, 0=free, 1=occupied, 0.5=dynamic` |
| `[43:52]` | 9 | **Other robots** | 3 robots × (rel_x, rel_y, dist). Relative to this agent's frame |
| `[52:55]` | 3 | **Target** | (sin_angle, cos_angle, dist_norm) to current navigation target |
| `[55:60]` | 5 | **Self** | (carry/2, vx_norm, vy_norm, sin_yaw, cos_yaw) |
| `[60:69]` | 9 | **Camera detections** | 3 objects × (type_sign, bearing_norm, dist_norm). `+1=cube, -1=sphere` |
| `[69:72]` | 3 | **Basket** | (sin_angle, cos_angle, dist_norm) to current relevant basket |
| `[72:81]` | 9 | **Known objects** | 3 known objects (from cam/broadcasts) × (type_sign, rel_x, rel_y) |
| `[81:84]` | 3 | **Zone info** | (coverage_frac, nearby_unknown_frac, frontier_dist_norm) |
| `[84]` | 1 | **Carry overtime** | 0→1 fraction of carry timeout used (penalty signal) |

### LiDAR Convention
- **18 rays** downsampled from 360 Gazebo rays
- **Ray 0 = robot forward direction** (heading angle), then CCW
- Gazebo LiDAR publishes 360 samples from `-π` to `+π` (index 0 = behind)
- Mapping: `gazebo_idx = int(round(((deg + 180) % 360) * 360 / 360))`
- This ensures training env and Gazebo deployment use identical ray indexing

### Occupancy Patch
- 5×5 cells centred on the agent, axis-aligned (not rotated to agent frame)
- Each cell covers 0.25×0.25m → 5×5 patch covers 1.25×1.25m around agent
- `-1` = UNKNOWN (never seen), `0` = FREE, `1` = OCCUPIED (static obstacle), `0.5` = DYNAMIC (obstacle moved)
- Other robots' positions are **filtered out** — they do not appear as OCCUPIED

---

## 6. Action Space

3-dimensional continuous, values in **[-1, 1]**:

| Index | Scaled to | Description |
|-------|-----------|-------------|
| `action[0]` | `× 0.50 m/s` | `vx` in robot frame (forward/back) |
| `action[1]` | `× 0.50 m/s` | `vy` in robot frame (strafe left/right) |
| `action[2]` | `× 2.0 rad/s` | `wz` yaw rotation rate |

The robot is **omni-directional** (holonomic) — it can strafe sideways without rotating. This matches the Gazebo `libgazebo_ros_planar_move` plugin.

---

## 7. Reward Function

All rewards are additive within a step. Final episode reward is the sum.

| Reward | Value | Trigger |
|--------|-------|---------|
| `R_STEP` | −0.01 | Every step (time pressure) |
| `R_NEW_CELL` | +1.0 (scaled) | New occupancy cells revealed by LiDAR (max 5 cells credited per step) |
| `R_REVISIT` | −0.2 | Returning to already-explored areas |
| `R_APPROACH` | +0.5 × Δdist | Per metre closed toward current target |
| `R_PICK` | +15.0 | Successfully auto-picked an object |
| `R_BROADCAST_BONUS` | +1.0 | First camera detection of an object |
| `R_COLLAB_BONUS` | +3.0 | Picking an object that another agent detected and broadcast |
| `R_DROP_CORRECT` | +40.0 | Delivering to the correct basket |
| `R_DROP_WRONG` | −20.0 | Delivering to the wrong basket |
| `R_CARRY_OVERTIME` | −0.05 × overtime | Holding object beyond `CARRY_TIMEOUT_STEPS=200` |
| `R_WALL_HIT` | −5.0 | Hitting a wall or static obstacle |
| `R_ROBOT_COLLISION` | −8.0 | Within 2×ROBOT_RADIUS of another robot |
| `R_PROXIMITY_HARD` | −2.0 × closeness | Within 0.5m of another robot (continuous) |
| `R_PROXIMITY_SOFT` | −0.5 × closeness | Within 1.5m of another robot (continuous) |
| `R_OBSTACLE_NEAR` | −0.3 | Per step within 0.5m of any static obstacle |
| `R_DANGLE` | −0.5 | Stuck near an obstacle for >3 steps without moving |
| `R_JERK` | −0.04 × jerk | Jerk = mean of `|a_t - 2×a_{t-1} + a_{t-2}|` (penalizes oscillation) |
| `R_ALL_DONE` | +100.0 | All 10 objects correctly delivered |

### Design Philosophy
- **Collision penalties are high** (`R_WALL_HIT=-5`, `R_ROBOT_COLLISION=-8`) to teach the policy to navigate safely before it learns to pick/drop
- **Jerk penalty** (`R_JERK`) replaces the old "smooth bonus" (which penalized any motion). Jerk penalizes the second derivative of action — robots can move fast, they just can't change direction abruptly
- **R_OBSTACLE_NEAR** provides a continuous repulsion shaping signal even outside the binary safety override zone

---

## 8. Per-Agent SLAM Occupancy Grid

**File**: `RL/training/env/occupancy_grid.py`
**Class**: `OccupancyGrid`

### Grid Specifications
- **Size**: 48×48 cells
- **Cell size**: 0.25m × 0.25m
- **Coverage**: 12m × 12m (the full arena)
- **Origin**: cell `(0,0)` = world position `(-6.0, -6.0)` (bottom-left)

### Cell States
```
OCC_UNKNOWN  = -1   # never observed by LiDAR
OCC_FREE     =  0   # a LiDAR ray passed through this cell
OCC_OCCUPIED =  1   # a LiDAR ray terminated here (static obstacle)
OCC_DYNAMIC  =  2   # was OCCUPIED, but a ray now passes through (obstacle moved)
```

### Update Logic (`update_from_lidar`)
For each of the 18 LiDAR rays:
1. Compute ray end point in world coordinates
2. Find grid cell of end point
3. Use **Bresenham line algorithm** to step through all grid cells from agent to end
4. All intermediate cells → mark `FREE`
5. End cell:
   - If ray hit something AND end cell is NOT a known robot position → mark `OCCUPIED`
   - If ray hit something AND end cell IS a known robot position → mark `FREE` (robot is dynamic, not a static obstacle)
   - If ray reached max range → mark `UNKNOWN` (ceiling, no info)
6. Special case: if a previously `OCCUPIED` cell is now traversed by a ray → mark `DYNAMIC`

### Robot Filtering
The 3 other robot positions are passed to `update_from_lidar(other_robots=[(x,y),...])`. A 3×3 cell neighbourhood around each robot is excluded from OCCUPIED marking. This prevents other robots from appearing as permanent obstacles in the static map.

### Observation Extraction (`get_ego_patch`)
- Extracts a 5×5 patch of cells centered on the agent
- Axis-aligned (not rotated to agent heading) — keeps computation O(1)
- Covers 1.25×1.25m around the agent
- Encoded as float32: `UNKNOWN=-1, FREE=0, OCCUPIED=1, DYNAMIC=0.5`
- Flattened to 25 values for inclusion in observation vector

### Exploration Targeting (`get_exploration_target`)
- Finds the nearest **frontier cell**: a FREE cell with at least one UNKNOWN neighbour
- Returns world coordinates `(x, y)` of the frontier cell centre
- Returns `None` if the entire arena is explored
- Used to assign exploration targets when no objects are known

### Coverage Metric (`get_coverage_fraction`)
- Returns fraction of cells that are not UNKNOWN (free + occupied)
- Reported in episode info dict as `explore_frac`

---

## 9. The Gazebo Arena

**File**: `src/collector_bot/worlds/arena.world`

### Walls
- Outer shell: 12.3m × 12.3m × 1.2m (height)
- Wall thickness: 0.3m
- Inner face at ±5.85m → robots can navigate up to `±(5.85 - ROBOT_RADIUS)` = ±5.65m

### Static Obstacles (17 total)
All obstacles are defined in `config.py` as `(x, y, half_width, half_height)` tuples:

| Name | Position | Size (half) | Type |
|------|----------|-------------|------|
| Pillar NE | (+5.9, +5.9) | 0.175×0.175 | Corner pillar |
| Pillar NW | (-5.9, +5.9) | 0.175×0.175 | Corner pillar |
| Pillar SE | (+5.9, -5.9) | 0.175×0.175 | Corner pillar |
| Pillar SW | (-5.9, -5.9) | 0.175×0.175 | Corner pillar |
| obs_box_1 | (+3.0, 0.0) | 0.75×0.15 | Wooden crate |
| obs_cyl_1 | (-2.0, -2.0) | 0.40×0.40 | Barrel |
| obs_box_2 | (0.0, +3.0) | 0.15×1.00 | Shelving unit |
| obs_box_3 | (-4.0, +2.0) | 0.50×0.15 | Pallet crate |
| shelf_east_1 | (+5.5, -3.0) | 0.25×1.25 | East shelf |
| shelf_west_1 | (-5.5, -3.0) | 0.25×1.25 | West shelf |
| barrel_1 | (+5.3, -5.3) | 0.25×0.25 | Deco barrel |
| barrel_2 | (+4.7, -5.3) | 0.25×0.25 | Deco barrel |
| barrel_3 | (-5.3, -5.3) | 0.25×0.25 | Deco barrel |
| crate_stack | (+5.4, +5.4) | 0.25×0.25 | Crate stack |
| cone_1 | (+2.0, -4.5) | 0.125×0.125 | Traffic cone |
| cone_2 | (-3.5, +4.5) | 0.125×0.125 | Traffic cone |
| cone_3 | (+4.5, +4.5) | 0.125×0.125 | Traffic cone |

### Robot URDF (`robot.urdf.xacro`)
- **Drive**: `libgazebo_ros_planar_move` (holonomic/omni), publishes `/robot_N/odom`
- **LiDAR**: `libgazebo_ros_ray_sensor`, 360 samples, `-π` to `+π`, range 0.12–5.0m, 10Hz update
- **Camera**: `libgazebo_ros_camera`, 320×240 RGB, horizontal FOV 69°, 30Hz update
- **IMU**: `libgazebo_ros_imu_sensor`, 50Hz (not used by current policy)

---

## 10. ROS2 Node Architecture

Four nodes run simultaneously:

```
┌──────────────────────────────────────────────────────┐
│                   PROCESS MAP                         │
│                                                       │
│  [safety_coordinator]  ─── publishes ─────────────┐  │
│         ↑ subscribes                               │  │
│    /robot_N/scan                            /{N}/safety_override
│    /robot_N/odom                                   │  │
│    /collected, /dropped                            ↓  │
│                                              [swarm_brain]
│  [sim_logger]          ─── subscribes ────────────┤  │
│         subscribes                                 │  │
│    /robot_poses                             publishes  │
│    /collected, /dropped                   /{N}/cmd_vel │
│    /visited_zones                                  │  │
│    /collision_stats                                │  │
│                                         ┌──────────┘  │
│                                         ↓             │
│                                  [Gazebo + 4 robots]  │
│                                   publishing           │
│                              /{N}/scan, /odom, /camera │
└──────────────────────────────────────────────────────┘
```

### `swarm_brain.py` — The Main Brain
- Centralized RL inference at 10Hz
- Loads policy from `RL/model/policy/policy.zip`
- Per-robot state: position, LiDAR, camera tracks, detected objects, carry state
- Respects `safety_coordinator` overrides (published as `/robot_N/safety_override`)
- Auto-pick: objects detected via camera within 0.6m
- Auto-drop: near correct basket
- Publishes robot poses every 500ms for other agents

### `safety_coordinator.py` — Deterministic Safety Layer
Runs at **10Hz** independently of the RL policy:
1. **Inter-robot collision**: if two robots < 0.30m apart → emergency push apart; if < 0.50m and closing → slow down aggressor
2. **Obstacle proximity**: if LiDAR < 0.25m for >1s → force reverse; only deactivate when LiDAR > 0.40m (hysteresis)
3. **Deadlock recovery**: if robot moved <0.02m in 5s → random perturbation
- Publishes `Twist` on `/robot_N/safety_override`
- `swarm_brain` checks this before publishing cmd_vel — zero override = policy drives

### `sim_logger.py` — Event Logger
- Subscribes to all events: picks, drops, collisions, poses, zone visits
- Writes `events.jsonl` (one JSON line per event)
- On shutdown: generates `report.html` + `trajectory_map.png`
- Calculates: completion ratio, coverage %, collision counts, duration

### `detector.py` — Camera-LiDAR Fusion
- Detects red (cubes) and blue (spheres) blobs in camera image via HSV thresholding
- Estimates bearing from pixel position in 69° FOV image
- Fuses with LiDAR to estimate world coordinates:
  `wx = robot.x + dist × cos(robot.yaw + bearing)`
- Maintains a list of `TrackedObject` (confirmed after 3+ consistent detections)
- `confirmed=True` tracks are used for pick decisions and broadcasts

### `explorer.py` — Coarse Grid Explorer
- 6×6 grid (2m cells) over 12m arena — used in deployment by `swarm_brain`
- Marks cells as visited as robot passes through
- Returns nearest unvisited cell as exploration target
- Separate instance per robot (no sharing → independent exploration)

### `navigator.py` — Go-To-Point
- Simple omni-drive proportional controller
- Given target `(tx, ty)` and robot pose `(x, y, yaw)`:
  - Compute (dx, dy) in world frame
  - Rotate to robot frame using `(-yaw)` rotation matrix
  - Scale to max speed
- Returns `(vx, vy, wz, arrived)` where `arrived=True` when within `arrival_dist`

### `avoidance.py` — Potential Field Repulsion
- Reads LiDAR scan, computes repulsive force from all nearby readings
- Influence radius: 1.0m
- Force magnitude: `gain × (1/dist - 1/influence_radius)`
- Used by safety_coordinator as the repulsion vector when safety override is active

### `gazebo_interface.py` — Async Gazebo Services
- Wraps `spawn_entity` and `delete_entity` Gazebo services
- Returns Python `Future` objects for non-blocking pick/drop operations
- `delete_async(name)`: removes a model from Gazebo (simulates pick)
- `spawn_async(name, type, x, y)`: spawns a cube/sphere at basket (simulates drop)

---

## 11. ROS2 Topics Reference

### Per-Robot Topics (`/robot_N/` prefix)

| Topic | Type | Publisher | Subscriber | Description |
|-------|------|-----------|------------|-------------|
| `/robot_N/cmd_vel` | `Twist` | swarm_brain | Gazebo planar_move | Drive command |
| `/robot_N/odom` | `Odometry` | Gazebo | swarm_brain, safety | Robot pose + velocity |
| `/robot_N/scan` | `LaserScan` | Gazebo | swarm_brain, safety | 360-ray LiDAR |
| `/robot_N/camera/image_raw` | `Image` | Gazebo | swarm_brain | RGB camera 320×240 |
| `/robot_N/safety_override` | `Twist` | safety_coordinator | swarm_brain | Safety override cmd |

### Global Coordination Topics

| Topic | Type | Publisher | Subscriber | Description |
|-------|------|-----------|------------|-------------|
| `/collected` | `String` | swarm_brain | all nodes | Object name picked up |
| `/dropped` | `String` | swarm_brain | safety, logger | Object name delivered |
| `/claimed` | `String` | swarm_brain | swarm_brain | Object/position being approached |
| `/unclaimed` | `String` | swarm_brain | swarm_brain | Releasing a claim |
| `/robot_poses` | `String` | swarm_brain | wrapper, logger | `"robot_N:x.xx:y.yy"` |
| `/known_objects` | `String` | swarm_brain | swarm_brain, wrapper | `"ns:type:x.xx:y.yy,..."` broadcast |
| `/visited_zones` | `String` | swarm_brain | swarm_brain | Grid cells visited |
| `/collision_stats` | `String` | swarm_brain | logger | `"robot_N:count"` |

---

## 12. Safety System

The safety system has **two independent layers** that cannot conflict with each other because they operate at different levels:

### Layer 1: Training-Env Safety (RL side)
- Built into `swarm_env.step()`
- Activates when `lidar_min < SAFETY_DIST_ENTER = 0.25m`
- Deactivates when `lidar_min > SAFETY_DIST_EXIT = 0.40m`  
- Output: potential-field repulsion replaces policy action
- **Purpose**: The policy learns that getting close to obstacles is bad

### Layer 2: Safety Coordinator (ROS2 side)
- Independent ROS2 node `safety_coordinator.py`
- NEVER relies on the RL policy — pure deterministic logic
- Publishes on `/robot_N/safety_override`
- `swarm_brain` checks: if override has non-zero magnitude → publish override, skip RL
- **Purpose**: Absolute guarantee in hardware — prevents damage even if policy fails

### Why Two Layers?
- Layer 1 shapes the policy to avoid danger zones
- Layer 2 is a hardware-level failsafe for Gazebo/real robot deployment
- They cannot fight each other because Layer 2 completely replaces Layer 1's output

---

## 13. Training Pipeline

### Algorithm: PPO (Proximal Policy Optimization)
- Implementation: Stable-Baselines3
- Mode: IPPO (Independent PPO) — shared weights, independent agents
- Policy: `MlpPolicy` with `net_arch=[256, 256, 128]` for both actor and critic

### Hyperparameters (defaults)
```python
learning_rate = 3e-4
n_steps       = 2048      # rollout length per env
batch_size    = auto      # (n_steps × n_envs) / 16
n_epochs      = 10        # PPO update passes per rollout
gamma         = 0.99      # discount factor
gae_lambda    = 0.95      # GAE lambda
clip_range    = 0.2       # PPO clip
ent_coef      = 0.01      # entropy bonus (exploration)
```

### Training Command
```bash
conda activate rl
cd ~/Desktop/ros2WS5/RL/training
python train.py --timesteps 10000000 --n-envs 8 --device cpu
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--timesteps` | 500,000 | Total training steps |
| `--n-envs` | 4 | Number of parallel environments (SubprocVecEnv) |
| `--lr` | 3e-4 | Learning rate |
| `--device` | cpu | `cpu` or `cuda` |
| `--render` | False | Show PyGame window (forces n_envs=1) |
| `--resume` | None | Path to existing model to continue training |
| `--epoch-steps` | 50,000 | Steps between metric logging |
| `--max-steps` | 600 | Max steps per robot per episode |
| `--seed` | 0 | Random seed |

### Output
Training creates a timestamped run directory under `RL/training/runs/`:
```
RL/training/runs/run_YYYYMMDD_HHMMSS/
├── checkpoints/ppo_swarm_N_steps.zip   # periodic checkpoints
├── best_model.zip                       # best eval performance
├── output.txt                           # all stdout
├── epoch_log.jsonl                      # per-epoch metrics
└── training_summary.json               # final stats
```
Final model saved to `RL/model/policy/policy.zip`.

### Episode Metrics (logged per epoch)
- `mean_reward`: average total return per episode
- `mean_picks`: average objects picked per episode
- `mean_deliveries`: average correct deliveries per episode
- `mean_collisions`: average collision events per episode
- `mean_broadcasts`: average new object detections per episode
- `mean_collabs`: average times an agent picked an object found by another
- `mean_explore_frac`: average fraction of arena explored (by occupancy grid)

---

## 14. Fine-Tuning Pipeline

Bridges the **sim-to-real gap** between PyGame training and Gazebo deployment.

### When to Fine-Tune
After training the base policy for 10M steps in PyGame, fine-tune in Gazebo to adapt to:
- Real LiDAR noise (Gaussian stddev=0.008m)
- Real camera image processing delays
- Gazebo physics (friction, inertia not present in PyGame)
- Actual sensor timing (10Hz LiDAR, 30Hz camera)

### Fine-Tune Command
```bash
# Start Gazebo first:
ros2 launch collector_bot simulation.launch.py finetune_robot:=all

# Then in another terminal:
conda activate rl
cd ~/Desktop/ros2WS5/RL/finetune
python finetune.py --timesteps 100000 --robot-ns all --lr 1e-4
```

### Fine-Tune Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `RL/model/policy/policy.zip` | Base model to fine-tune |
| `--timesteps` | 50,000 | Fine-tuning steps |
| `--robot-ns` | `robot_1` | Which robot(s). `all` uses SubprocVecEnv for all 4 |
| `--lr` | 1e-4 | Learning rate (lower than pretraining) |
| `--step-duration` | 0.1s | Real time per step |

### `GazeboEnvWrapper`
- Wraps a live Gazebo simulation as a Gymnasium env
- Spins a ROS2 executor in a background thread
- Builds the same 85-dim observation as the training env (identical code path)
- Applies action EMA matching the training env
- Output saved to `RL/finetune/logs/ft_YYYYMMDD_HHMMSS/`

---

## 15. Configuration — Single Source of Truth

**File**: `RL/training/config.py`

This is the **only** place where constants are defined. All other files import from here.

ROS2 nodes access constants via the bridge:
```python
# src/collector_bot/collector_bot/constants.py
import sys, os
sys.path.insert(0, os.path.expanduser('~/Desktop/ros2WS5/RL/training'))
from config import *   # re-exports everything
```

ROS2 nodes then do:
```python
from collector_bot.constants import MAX_SPEED, LIDAR_RAYS, OBS_DIM, ...
```

### Critical Constants

```python
# Arena
ARENA_HALF = 6.0        # m half-size
ROBOT_RADIUS = 0.20     # m

# Robots
N_ROBOTS = 4
MAX_SPEED = 0.50        # m/s per axis
MAX_WZ = 2.0            # rad/s

# Sensors
LIDAR_RAYS = 18         # down-sampled from 360
LIDAR_MAX_DIST = 5.0    # m — matches URDF
CAMERA_FOV = 1.2        # rad (69°)
CAMERA_RANGE = 4.0      # m

# Occupancy grid
OCC_CELL = 0.25         # m per cell
OCC_NX = OCC_NY = 48   # cells per axis

# Observation
OBS_DIM = 85
ACT_DIM = 3

# Safety hysteresis
SAFETY_DIST_ENTER = 0.25  # m — activate override
SAFETY_DIST_EXIT  = 0.40  # m — deactivate

# Action smoothing
ACTION_EMA = 0.3  # blend: 70% new + 30% previous
```

---

## 16. Installation & Setup

### System Requirements
- Ubuntu 22.04
- ROS2 Humble
- Gazebo 11
- Python 3.10+
- Conda (miniconda recommended)

### 1. Create Conda Environment
```bash
conda create -n rl python=3.10
conda activate rl
pip install -r ~/Desktop/ros2WS5/RL/requirements.txt
```

### 2. Install ROS2 Dependencies
```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-cv-bridge \
     ros-humble-xacro ros-humble-robot-state-publisher
```

### 3. Build the ROS2 Package
```bash
cd ~/Desktop/ros2WS5
colcon build
source install/setup.bash
```

### 4. Verify Training Env
```bash
conda activate rl
cd ~/Desktop/ros2WS5/RL/training
python -c "from env.swarm_env import SwarmCollectorEnv; e=SwarmCollectorEnv(); o,_=e.reset(); print(o.shape)"
# Expected output: (85,)
```

---

## 17. Running the System

### Step 1: Train the RL Model (PyGame, ~6-8 hours for 10M steps)
```bash
conda activate rl
cd ~/Desktop/ros2WS5/RL/training
python train.py --timesteps 10000000 --n-envs 8 --device cpu
```

Watch training live:
```bash
python train.py --timesteps 100000 --render --n-envs 1
```

### Step 2: Launch Gazebo Simulation
```bash
source ~/Desktop/ros2WS5/install/setup.bash
ros2 launch collector_bot simulation.launch.py
```

Optional arguments:
```bash
# Specify trained model
ros2 launch collector_bot simulation.launch.py model_path:=/path/to/policy.zip

# Run headless (no GUI)
SWARM_HEADLESS=true ros2 launch collector_bot simulation.launch.py

# During fine-tuning (brain skips robot_1 so fine-tuner controls it)
ros2 launch collector_bot simulation.launch.py finetune_robot:=robot_1
```

### Step 3: Fine-Tune in Gazebo
```bash
# Terminal 1: Launch Gazebo (with fine-tune mode)
ros2 launch collector_bot simulation.launch.py finetune_robot:=all

# Terminal 2: Run fine-tuner
conda activate rl
cd ~/Desktop/ros2WS5/RL/finetune
python finetune.py --timesteps 100000 --robot-ns all --lr 1e-4
```

### Step 4: Run with Fine-Tuned Model
```bash
ros2 launch collector_bot simulation.launch.py \
    model_path:=~/Desktop/ros2WS5/RL/finetune/logs/ft_YYYYMMDD/policy_finetuned.zip
```

### Monitoring
```bash
# Watch robot velocities
ros2 topic echo /robot_1/cmd_vel

# Watch collision stats
ros2 topic echo /collision_stats

# Watch collected objects
ros2 topic echo /collected

# See all active topics
ros2 topic list

# TensorBoard training curves
tensorboard --logdir ~/Desktop/ros2WS5/RL/training/runs/
```

---

## 18. File-by-File Reference

### `RL/training/config.py`
**Role**: Single source of truth for every constant in the system.  
**Contents**: Arena dimensions, robot kinematics, sensor parameters, occupancy grid config, observation layout, reward weights, episode parameters.  
**Key rule**: NEVER copy a value from here into another file. Always import.

---

### `RL/training/env/occupancy_grid.py`
**Role**: Per-agent SLAM occupancy grid.  
**Class**: `OccupancyGrid`  
**Key methods**:
- `reset()` — clear to UNKNOWN at episode start
- `update_from_lidar(x, y, yaw, lidar_norm, other_robots=None)` — Bresenham update with robot filtering
- `get_ego_patch(x, y)` → `float32[25]` — 5×5 local view for observation
- `get_exploration_target(x, y)` → `(tx, ty)` or `None` — nearest frontier
- `get_coverage_fraction()` → `float` — fraction explored
- `get_nearby_unknown_fraction(x, y)` → `float` — local unexplored fraction
- `get_frontier_distance(x, y)` → `float` — distance to nearest frontier

---

### `RL/training/env/arena.py`
**Role**: Geometry primitives shared by training env and ROS2.  
**Key functions**:
- `check_wall_collision(x, y)` → `(x, y, hit)` — clamp to arena bounds
- `check_obstacle_collision(new_x, new_y, old_x, old_y)` → `(x, y, hit)` — AABB check against all 17 obstacles
- `point_near_any_obstacle(x, y, margin)` → `bool` — used for spawn validation
- `raycast(ox, oy, angle, max_dist, agents, objects, exclude_idx)` → `float` — step-based raycasting
- `detect_in_fov(x, y, yaw, objects)` → `list[(type, bearing, dist)]` — camera FOV detection
- `compute_repulsion_from_lidar(lidar_norm, n_rays, max_dist)` → `(vx, vy)` — safety repulsion

---

### `RL/training/env/swarm_env.py`
**Role**: Gymnasium multi-agent environment for PPO training.  
**Class**: `SwarmCollectorEnv(gym.Env)`  
**Key design**: CTDE, cycles through 4 agents per external step, full episode isolation, per-agent SLAM, no oracle.

---

### `RL/training/train.py`
**Role**: Training script.  
**Algorithm**: PPO from Stable-Baselines3.  
**Notable**: Epoch-based training loop, JSONL metric logging, eval callback, checkpoint saving, matplotlib plots on completion.

---

### `RL/training/paths.py`
**Role**: Path helpers.  
**Functions**: `next_run_dir()` → creates timestamped run dir; `model_path()` → RL/model/policy/

---

### `RL/finetune/gazebo_env_wrapper.py`
**Role**: Wraps live Gazebo as Gymnasium env.  
**Key**: Same 85-dim obs as training env, same action EMA smoothing. ROS2 executor spins in background thread.

---

### `RL/finetune/finetune.py`
**Role**: Continue PPO training with Gazebo as the data source.  
**Key**: Lower LR (1e-4 vs 3e-4 in pretraining), supports all-4-robot fine-tuning via SubprocVecEnv.

---

### `src/collector_bot/collector_bot/constants.py`
**Role**: Bridge — adds `RL/training/` to sys.path and re-exports all config values.  
**Usage**: `from collector_bot.constants import MAX_SPEED, OBS_DIM, ...`

---

### `src/collector_bot/collector_bot/swarm_brain.py`
**Role**: Main ROS2 node. Centralized RL inference + high-level task management.  
**Key behaviours**:  
1. Load policy at startup  
2. Per-robot state: LiDAR, odometry, camera tracks, carry state, detected objects  
3. 10Hz tick: build obs → predict → EMA smooth → publish cmd_vel (unless safety override)  
4. Object discovery via camera (no oracle)  
5. Broadcasting new detections on `/known_objects`  
6. Async Gazebo pick/drop via `GazeboInterface`

---

### `src/collector_bot/collector_bot/safety_coordinator.py`
**Role**: Deterministic safety layer. Runs fully independently from RL.  
**Checks**: Inter-robot distance, obstacle proximity (hysteresis), stuck detection.  
**Output**: `/robot_N/safety_override` Twist messages.

---

### `src/collector_bot/collector_bot/sim_logger.py`
**Role**: Event logger + report generator.  
**Subscribes**: All coordination topics.  
**On shutdown**: Writes `events.jsonl`, `report.html`, `trajectory_map.png`.

---

### `src/collector_bot/collector_bot/detector.py`
**Role**: Camera-based object detection.  
**Method**: HSV colour thresholding (red=cube, blue=sphere) → centroid bearing → LiDAR range fusion → tracked objects.  
**Output**: `r.tracks` list of `TrackedObject` with `confirmed=True` after 3 consistent detections.

---

### `src/collector_bot/collector_bot/explorer.py`
**Role**: Coarse grid-based frontier explorer (deployment only).  
**Grid**: 6×6 cells, 2m each.  
**Usage**: Each robot has its own `Explorer` instance in `swarm_brain._PerRobotState`.

---

### `src/collector_bot/collector_bot/navigator.py`
**Role**: Proportional omni-drive go-to-point controller.  
**Output**: `(vx, vy, wz, arrived)` — used as fallback when RL not active.

---

### `src/collector_bot/collector_bot/avoidance.py`
**Role**: Potential-field repulsion from LiDAR.  
**Used by**: `safety_coordinator` to generate the repulsion vector during emergency override.

---

### `src/collector_bot/collector_bot/gazebo_interface.py`
**Role**: Async wrappers for Gazebo `spawn_entity` / `delete_entity` services.  
**Why async**: Prevents the 10Hz control loop from blocking during slow Gazebo service calls.

---

### `src/collector_bot/collector_bot/paths.py`
**Role**: Path constants for the ROS2 package.  
**Contains**: `MODEL_PATH` (to RL model zip), `SIMULATE_DIR` (to simulate/ logs).

---

### `src/collector_bot/launch/simulation.launch.py`
**Role**: Main launch file.  
**Sequence**:
- t=0s: Gazebo + world  
- t=3s: sim_logger  
- t=5s: robot_1 spawn  
- t=9s: robot_2 spawn  
- t=13s: robot_3 spawn  
- t=17s: robot_4 spawn  
- t=18s: swarm_brain + safety_coordinator  

**Features**: Random spawn positions (cross pattern), headless mode (`SWARM_HEADLESS=true`), fine-tune mode (`finetune_robot:=robot_N`).

---

### `src/collector_bot/worlds/arena.world`
**Role**: Gazebo SDF world definition.  
**Must match**: All 17 obstacle positions/sizes in `config.py OBSTACLES`.

---

### `src/collector_bot/urdf/robot.urdf.xacro`
**Role**: Robot model. Instanced 4× with different namespaces.  
**Key sensors**: 360-ray LiDAR (10Hz), RGB camera (30Hz), IMU (50Hz).  
**Drive**: Planar move plugin (holonomic).

---

## 19. Key Design Decisions

### Why IPPO (shared weights)?
- Single model to train, store, and deploy
- Emergent cooperation without explicit communication protocol
- Well-studied: robots with identical weights still diverge in behaviour due to different observations

### Why 18 LiDAR rays instead of 360?
- Computation: 360 → 60ms per env step; 18 → ~3ms
- 18 rays gives 20° spacing — enough to detect doorways, obstacle gaps, other robots
- Downsampled in training; Gazebo publishes 360 and we index every 20th ray

### Why occupancy grid patch instead of frontier count?
- The 5×5 patch gives the policy **spatial structure** — it can "see" a wall to the left and a gap to the right
- Simple frontier count (scalar) loses all spatial information
- 25 extra dims is negligible vs. the spatial richness added

### Why axis-aligned patch instead of agent-frame patch?
- Rotating a 48×48 grid every step would require `scipy.ndimage.rotate` or OpenCV — ~5ms per call
- Axis-aligned: constant time, no interpolation artifacts
- The LiDAR rays (which are agent-frame) compensate for lack of rotation

### Why hysteresis in the safety system?
Without hysteresis: if threshold is 0.25m, robot at exactly 0.25m will oscillate between safety-override and policy 10× per second → velocity oscillations of ±0.5 m/s.  
With hysteresis: activate at 0.25m, only deactivate after clearing 0.40m → smooth transition, no oscillation.

### Why cross-pattern spawn?
- Fixed spawn → policy memorises positions, doesn't generalize
- Fully random spawn → robots can spawn overlapping or in obstacles
- Cross-pattern: random centre + deterministic offsets → always valid, always separated, widely varied

### Why jerk penalty instead of smooth bonus?
- Old `R_SMOOTH` penalized action magnitude → robots learned to move slowly (safe but slow)
- `R_JERK` penalizes `|a_t - 2a_{t-1} + a_{t-2}|` (second derivative) → robots can be fast but must be smooth
- This matches the physical reality: fast consistent motion is fine, sudden direction reversals are damaging

### Why no global oracle knowledge?
The old system had `WORLD_OBJECTS = {name: (x, y)}` hardcoded in `swarm_brain.py`. This caused:
1. All 4 robots immediately targeted the nearest object at launch → collision cascades
2. Policy trained without oracle failed at deployment with oracle → no behavioural sim-to-real gap
3. Removing oracle forces genuine exploration, which is the actual desirable behaviour

---

## 20. Known Limitations & Future Work

### Current Limitations

1. **Object positions not in observation**: Objects discovered via camera are stored in `detected_objects` but their estimated world positions are not updated after initial detection. If the robot's camera estimate is off by 0.5m, the pick may fail.

2. **Axis-aligned occupancy patch**: The 5×5 patch does not rotate with the robot. Two robots facing opposite directions at the same location will have different-looking patches but identical LiDAR. The policy must learn to use the LiDAR (agent-frame) and patch (world-frame) together.

3. **No map sharing in training**: Agents broadcast object positions but not occupancy grids. Two agents may explore the same area redundantly. Real map sharing would require `merge_remote()` which is implemented but not yet called.

4. **Static obstacles only**: The occupancy grid handles `OCC_DYNAMIC` state but training currently has no moving obstacles. The dynamic machinery is ready but untested.

5. **Gazebo model name resolution**: In deployment, picking requires knowing the Gazebo model name (e.g., `cube_3`). The current system uses a heuristic (first uncollected matching type). This can fail if two cubes are close together and one has already been collected.

### Planned Future Work

- **Dynamic obstacles**: Add moving carts/forklifts in Gazebo. The `OCC_DYNAMIC` state and `mark_dynamic()` method in `OccupancyGrid` are already designed for this.
- **Map sharing**: Call `agent.occ_grid.merge_remote(other_grid)` every N steps using the `/robot_poses` topic as a trigger. Requires serializing the 48×48 grid (2304 bytes) over ROS2.
- **Real robot deployment**: Bridge to physical omni-drive robot with ROS2 Nav2 and a real LiDAR (e.g., RPLidar A1 → 360 rays, 6m range).
- **Curriculum learning**: Start training with fewer obstacles and shorter episodes, gradually increase difficulty.
- **Heterogeneous agents**: Train separate specialist policies (explorer vs. transporter) and blend via meta-policy.
- **Visual occupancy**: Replace axis-aligned patch with a small CNN processing a top-down occupancy image (rotated to agent frame).
