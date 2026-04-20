"""
Centralized swarm brain — batched RL inference for all robots.

Architecture:
  - Single PPO model loaded once, batched inference for all N_ROBOTS
  - Per-robot camera-based object detection (no SLAM)
  - Random-walk exploration when no objects visible
  - Safety coordinator override respected

Observation: 43-dim (lean — no SLAM, no occ grid)
Action: 3-dim [vx, vy, wz] in [-1, 1]
"""

# ── Inject Conda environment first so ROS2 doesn't load outdated system packages ──
import sys
import os
conda_site = os.path.expanduser('~/miniconda3/envs/rl/lib/python3.10/site-packages')
if conda_site not in sys.path:
    sys.path.insert(0, conda_site)

import math
import time
import random

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from collector_bot.constants import (
    ARENA_HALF, ARENA_SIZE, N_ROBOTS, ROBOT_RADIUS,
    MAX_SPEED, MAX_WZ,
    OBJECT_DEFS, N_OBJECTS,
    CUBE_BASKET, SPHERE_BASKET, PICK_DIST, BASKET_DIST,
    LIDAR_RAYS, LIDAR_MAX_DIST,
    CAMERA_FOV, CAMERA_RANGE,
    N_DETECTIONS, CARRY_TIMEOUT_STEPS,
    OBS_DIM, ACT_DIM,
    SAFETY_DIST_ENTER, SAFETY_DIST_EXIT, ACTION_EMA,
    EXPLORE_RETARGET_STEPS,
)
from collector_bot.detector import detect_colors, fuse_with_lidar, update_tracks
from collector_bot.gazebo_interface import GazeboInterface
from collector_bot.avoidance import min_range
from collector_bot.navigator import go_to_point
from collector_bot.paths import MODEL_PATH

try:
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    HAS_SB3 = True
except ImportError as e2:
    import sys
    print(f"CRITICAL IMPORT ERROR (fallback): {e2}", file=sys.stderr)
    HAS_SB3 = False

# Object names only (no positions) — only used for Gazebo delete_entity calls
OBJECT_NAMES = {name: otype for name, otype in OBJECT_DEFS}

# Gazebo object positions (from arena.world <pose> tags)
# Used for matching camera tracks to Gazebo model names on pick
OBJECT_POSITIONS = {
    'cube_1':   ( 1.5,  1.5),
    'cube_2':   (-3.0,  0.5),
    'cube_3':   ( 4.0, -3.0),
    'cube_4':   (-1.0, -3.5),
    'cube_5':   ( 2.0,  3.5),
    'sphere_1': (-2.0,  3.0),
    'sphere_2': ( 1.0, -1.5),
    'sphere_3': (-4.0, -1.0),
    'sphere_4': ( 3.5,  1.0),
    'sphere_5': (-0.5,  4.0),
}

ROBOT_NAMES = [f'robot_{i+1}' for i in range(N_ROBOTS)]


class _PerRobotState:
    """Per-robot state container."""

    def __init__(self, ns):
        self.ns = ns
        self.x = self.y = self.yaw = 0.0
        self.vx = self.vy = 0.0
        self.scan_ranges = []
        self.tracks = []
        self.carrying = None
        self.carrying_name = None
        self.carry_steps = 0
        self.target = None
        self.approach_target = None
        self.approach_t0 = 0.0
        self.my_claim = None
        self._pick_future = None
        self._drop_future = None
        self._pick_name = None
        self._pick_type = None
        self.drop_count = 0
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)

        # Per-robot detection memory (no oracle)
        self.detected_objects = {}  # name → (x, y, type, source_ns)
        self._broadcast_sent = set()

        # Collision tracking
        self._collision_count = 0
        self._in_collision = False

        # Safety hysteresis
        self.safety_active = False

        # Random-walk exploration
        self._explore_target = None
        self._explore_steps = 0


class SwarmBrain(Node):
    """Centralized RL brain for all robots."""

    def __init__(self):
        super().__init__('swarm_brain')

        self.declare_parameter('model_path', '')
        self.declare_parameter('finetune_robot', '')
        model_p = self.get_parameter('model_path').value or MODEL_PATH
        self.finetune_ns = self.get_parameter('finetune_robot').value or ''

        # ── Per-robot state ──
        self.robots = {ns: _PerRobotState(ns) for ns in ROBOT_NAMES}

        # ── Shared coordination state ──
        self.claimed_set = set()
        self.collected_set = set()

        # ── ROS2 setup ──
        self.cmd_pubs = {}

        for ns in ROBOT_NAMES:
            self.create_subscription(
                LaserScan, f'/{ns}/scan',
                lambda msg, r=ns: self._scan_cb(r, msg), 10)
            self.create_subscription(
                Image, f'/{ns}/camera/image_raw',
                lambda msg, r=ns: self._image_cb(r, msg), 5)
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda msg, r=ns: self._odom_cb(r, msg), 10)
            self.cmd_pubs[ns] = self.create_publisher(
                Twist, f'/{ns}/cmd_vel', 10)

        # Global coordination topics
        self.create_subscription(String, '/claimed', self._claimed_cb, 10)
        self.create_subscription(String, '/unclaimed', self._unclaimed_cb, 10)
        self.create_subscription(String, '/collected', self._collected_cb, 10)
        self.create_subscription(
            String, '/known_objects', self._known_objects_cb, 10)

        # Safety override subscribers (per robot)
        self._safety_overrides = {ns: None for ns in ROBOT_NAMES}
        for ns in ROBOT_NAMES:
            self.create_subscription(
                Twist, f'/{ns}/safety_override',
                lambda msg, r=ns: self._safety_override_cb(r, msg), 10)

        # Publishers
        self.claim_pub = self.create_publisher(String, '/claimed', 10)
        self.unclaim_pub = self.create_publisher(String, '/unclaimed', 10)
        self.collect_pub = self.create_publisher(String, '/collected', 10)
        self.drop_pub = self.create_publisher(String, '/dropped', 10)
        self.pose_pub = self.create_publisher(String, '/robot_poses', 10)
        self.stats_pub = self.create_publisher(String, '/collision_stats', 10)
        self.known_pub = self.create_publisher(String, '/known_objects', 10)

        # Gazebo interface
        self.gz = GazeboInterface(self)

        # ── Load RL model ──
        self.policy = None
        self.get_logger().info(f'DEBUG LOAD: HAS_SB3={HAS_SB3}, model_p={model_p}')
        if model_p:
            self.get_logger().info(f'DEBUG LOAD: path.exists={os.path.exists(model_p)}')
        if HAS_SB3 and model_p and os.path.exists(model_p):
            try:
                self.policy = PPO.load(model_p)
                loaded_dim = self.policy.observation_space.shape[0]
                if loaded_dim != OBS_DIM:
                    self.get_logger().warn(
                        f'Model dim mismatch: expected {OBS_DIM}, got {loaded_dim}. '
                        'Rejecting model and using fallback navigator.'
                    )
                    self.policy = None
                else:
                    self.get_logger().info(f'RL model loaded successfully: {model_p}')
            except Exception as e:
                self.get_logger().error(f'Failed to load model: {e}')
                self.policy = None
        else:
            self.get_logger().warn('No valid RL model found — using fallback navigator')

        # ── Timers ──
        self.create_timer(0.1, self._tick)
        self.create_timer(0.5, self._publish_poses)
        self.create_timer(5.0, self._publish_stats)

        self.get_logger().info('SwarmBrain ready (43-dim obs, no SLAM)')

    # ── Callbacks ──

    def _scan_cb(self, ns, msg):
        self.robots[ns].scan_ranges = list(msg.ranges)

    def _image_cb(self, ns, msg):
        r = self.robots[ns]
        try:
            # We bypass cv_bridge entirely to avoid Numpy C-API ABI mismatch between ROS and Conda
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        except Exception:
            return
        raw = detect_colors(img)
        fused = fuse_with_lidar(raw, r.scan_ranges, r.x, r.y, r.yaw)
        update_tracks(r.tracks, fused)
        self._broadcast_new_objects(ns)

    def _odom_cb(self, ns, msg):
        r = self.robots[ns]
        r.x = msg.pose.pose.position.x
        r.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        r.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        r.vx = msg.twist.twist.linear.x
        r.vy = msg.twist.twist.linear.y

    def _claimed_cb(self, msg):
        self.claimed_set.add(msg.data)

    def _unclaimed_cb(self, msg):
        self.claimed_set.discard(msg.data)

    def _collected_cb(self, msg):
        self.collected_set.add(msg.data)
        self.claimed_set.discard(msg.data)

    def _known_objects_cb(self, msg):
        """Receive broadcasts from other robots or from self."""
        for item in msg.data.split(','):
            parts = item.split(':')
            if len(parts) == 4:
                src_ns, otype, sx, sy = parts
                try:
                    ox, oy = float(sx), float(sy)
                    # Distribute to robots that didn't send it
                    for ns, r in self.robots.items():
                        if ns == src_ns:
                            continue
                        key = f'{otype}_{ox:.1f}_{oy:.1f}'
                        if key not in r.detected_objects:
                            r.detected_objects[key] = (ox, oy, otype, src_ns)
                except ValueError:
                    pass

    def _safety_override_cb(self, ns, msg):
        """Store latest safety override for each robot."""
        magnitude = abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z)
        if magnitude > 0.01:
            self._safety_overrides[ns] = msg
        else:
            self._safety_overrides[ns] = None

    def _publish_poses(self):
        for ns, r in self.robots.items():
            m = String()
            m.data = f'{ns}:{r.x:.3f}:{r.y:.3f}'
            self.pose_pub.publish(m)

    def _publish_stats(self):
        for ns, r in self.robots.items():
            m = String()
            m.data = f'{ns}:{r._collision_count}'
            self.stats_pub.publish(m)

    def _broadcast_new_objects(self, ns):
        """Broadcast newly confirmed tracks from one robot."""
        r = self.robots[ns]
        items = []
        for t in r.tracks:
            if not t.confirmed:
                continue
            key = f'{t.obj_type}_{t.wx:.1f}_{t.wy:.1f}'
            if key in r._broadcast_sent:
                continue
            if key in r.detected_objects:
                continue
            r._broadcast_sent.add(key)
            r.detected_objects[key] = (t.wx, t.wy, t.obj_type, ns)
            items.append(f'{ns}:{t.obj_type}:{t.wx:.2f}:{t.wy:.2f}')
        if items:
            m = String()
            m.data = ','.join(items)
            self.known_pub.publish(m)

    # ── Main tick ──

    def _tick(self):
        for ns, r in self.robots.items():
            # Skip robots being fine-tuned
            if self.finetune_ns:
                if self.finetune_ns == 'all' or self.finetune_ns == ns:
                    continue

            self._check_collision(ns)

            # Handle async pick/drop
            if r._pick_future is not None:
                self._finish_pick(ns)
                continue
            if r._drop_future is not None:
                self._finish_drop(ns)
                continue

            if r.carrying is not None:
                r.carry_steps += 1

            self._update_target(ns)

            # ── Safety coordinator override takes priority ──
            safety_msg = self._safety_overrides.get(ns)
            if safety_msg is not None:
                self.cmd_pubs[ns].publish(safety_msg)
                continue

            # ── RL navigation ──
            if self.target_exists(ns) and self.policy is not None:
                obs = self._build_obs(ns)
                action, _ = self.policy.predict(obs, deterministic=True)

                # Action EMA smoothing
                smoothed = ((1.0 - ACTION_EMA) * action +
                            ACTION_EMA * r.prev_action)
                r.prev_action = action.copy()

                vx = float(smoothed[0]) * MAX_SPEED
                vy = float(smoothed[1]) * MAX_SPEED
                wz = float(smoothed[2]) * MAX_WZ
                self._pub(ns, vx, vy, wz)

            elif self.target_exists(ns):
                # Fallback navigator
                vx, vy, wz, _ = go_to_point(
                    r.target[0], r.target[1],
                    r.x, r.y, r.yaw, arrival_dist=0.20)
                self._pub(ns, vx, vy, wz)
            else:
                self._pub(ns, 0, 0, 0)

            # ── Auto-pick (via camera tracks, no oracle) ──
            if r.carrying is None:
                self._try_pick(ns)

            # ── Auto-drop ──
            if r.carrying is not None:
                basket = CUBE_BASKET if r.carrying == 'cube' else SPHERE_BASKET
                d = math.sqrt(
                    (r.x - basket[0])**2 + (r.y - basket[1])**2)
                if d < BASKET_DIST:
                    self._start_drop(ns)

    def target_exists(self, ns):
        return self.robots[ns].target is not None

    def _update_target(self, ns):
        """Assign target — no oracle, no SLAM. Camera detections + random walk."""
        r = self.robots[ns]

        # Prune ghost objects: remove collected objects from detection memory
        for key in list(r.detected_objects.keys()):
            for cname in self.collected_set:
                if cname in key or key == cname:
                    del r.detected_objects[key]
                    break

        if r.carrying is not None:
            basket = CUBE_BASKET if r.carrying == 'cube' else SPHERE_BASKET
            r.target = (float(basket[0]), float(basket[1]))
        elif r.approach_target is not None:
            r.target = (r.approach_target.wx, r.approach_target.wy)
            if time.time() - r.approach_t0 > 10.0:
                self._release_claim(ns)
                r.approach_target = None
        else:
            # Check camera tracks for approachable objects
            best = self._best_available_track(ns)
            if best is not None:
                self._claim_position(ns, best.wx, best.wy)
                r.approach_target = best
                r.approach_t0 = time.time()
                r.target = (best.wx, best.wy)
                return

            # Check known objects from broadcasts
            best_d, best_pos = float('inf'), None
            for key, (kx, ky, ktype, _) in r.detected_objects.items():
                d = math.sqrt((r.x - kx)**2 + (r.y - ky)**2)
                if d < best_d and d > PICK_DIST:
                    best_d = d
                    best_pos = (kx, ky)
            if best_pos is not None and best_d < 8.0:
                r.target = best_pos
                return

            # Random-walk exploration (no SLAM)
            r._explore_steps += 1
            if (r._explore_target is None or
                    r._explore_steps >= EXPLORE_RETARGET_STEPS):
                r._explore_target = (
                    random.uniform(-ARENA_HALF + 1.0, ARENA_HALF - 1.0),
                    random.uniform(-ARENA_HALF + 1.0, ARENA_HALF - 1.0),
                )
                r._explore_steps = 0
            # Check if we arrived
            if r._explore_target is not None:
                d = math.sqrt((r.x - r._explore_target[0])**2 +
                              (r.y - r._explore_target[1])**2)
                if d < 1.0:
                    r._explore_target = (
                        random.uniform(-ARENA_HALF + 1.0, ARENA_HALF - 1.0),
                        random.uniform(-ARENA_HALF + 1.0, ARENA_HALF - 1.0),
                    )
                    r._explore_steps = 0
            r.target = r._explore_target

    def _try_pick(self, ns):
        """Pick nearby objects using camera tracks — match by Gazebo position."""
        r = self.robots[ns]
        for t in r.tracks:
            if not t.confirmed:
                continue
            d = math.sqrt((r.x - t.wx)**2 + (r.y - t.wy)**2)
            if d < PICK_DIST:
                # Match by PROXIMITY to known Gazebo positions
                best_name, best_d = None, float('inf')
                for obj_name, obj_type in OBJECT_DEFS:
                    if obj_type != t.obj_type:
                        continue
                    if obj_name in self.collected_set:
                        continue
                    if obj_name in OBJECT_POSITIONS:
                        ox, oy = OBJECT_POSITIONS[obj_name]
                        od = math.sqrt((t.wx - ox)**2 + (t.wy - oy)**2)
                        if od < best_d:
                            best_d = od
                            best_name = obj_name
                if best_name and best_d < 2.0:
                    self._start_pick(ns, best_name, t.obj_type)
                    break

    # ── Observation builder (43-dim, lean) ──

    def _build_obs(self, ns):
        r = self.robots[ns]
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        # [0:18] LiDAR — ray 0 = forward (heading), CCW
        n = len(r.scan_ranges)
        for i_r in range(LIDAR_RAYS):
            if n > 0:
                deg = i_r * (360.0 / LIDAR_RAYS)
                li = int(round(((deg + 180) % 360) * n / 360)) % n
                val = r.scan_ranges[li]
                if math.isinf(val) or math.isnan(val):
                    val = LIDAR_MAX_DIST
                obs[idx] = min(val, LIDAR_MAX_DIST) / LIDAR_MAX_DIST
            else:
                obs[idx] = 1.0
            idx += 1

        # [18:27] Other robots: 3 × (rel_x, rel_y, dist)
        others = [s for rns, s in self.robots.items() if rns != ns][:3]
        for i in range(3):
            if i < len(others):
                o = others[i]
                dx, dy = o.x - r.x, o.y - r.y
                d = math.sqrt(dx*dx + dy*dy)
                cos_y = math.cos(-r.yaw)
                sin_y = math.sin(-r.yaw)
                rel_x = dx * cos_y - dy * sin_y
                rel_y = dx * sin_y + dy * cos_y
                obs[idx]     = np.clip(rel_x / ARENA_SIZE, -1, 1)
                obs[idx + 1] = np.clip(rel_y / ARENA_SIZE, -1, 1)
                obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
            idx += 3

        # [27:30] Target
        if r.target is not None:
            tx, ty = r.target
            dx, dy = tx - r.x, ty - r.y
            d = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx) - r.yaw
            obs[idx]     = math.sin(angle)
            obs[idx + 1] = math.cos(angle)
            obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
        idx += 3

        # [30:39] Camera detections: 3 × (type_sign, bearing_norm, dist_norm)
        confirmed = [t for t in r.tracks if t.confirmed]
        confirmed.sort(key=lambda t: math.sqrt(
            (t.wx - r.x)**2 + (t.wy - r.y)**2))
        for i in range(N_DETECTIONS):
            if i < len(confirmed):
                t = confirmed[i]
                dx, dy = t.wx - r.x, t.wy - r.y
                d = math.sqrt(dx*dx + dy*dy)
                angle = math.atan2(dy, dx) - r.yaw
                obs[idx]     = 1.0 if t.obj_type == 'cube' else -1.0
                obs[idx + 1] = angle / math.pi
                obs[idx + 2] = np.clip(d / CAMERA_RANGE, 0, 1)
            idx += 3

        # [39:42] Basket
        if r.carrying == 'cube':
            bx, by = float(CUBE_BASKET[0]), float(CUBE_BASKET[1])
        elif r.carrying == 'sphere':
            bx, by = float(SPHERE_BASKET[0]), float(SPHERE_BASKET[1])
        else:
            bx, by = float(CUBE_BASKET[0]), float(CUBE_BASKET[1])
        dx, dy = bx - r.x, by - r.y
        d = math.sqrt(dx*dx + dy*dy)
        angle = math.atan2(dy, dx) - r.yaw
        obs[idx]     = math.sin(angle)
        obs[idx + 1] = math.cos(angle)
        obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
        idx += 3

        # [42] Carry overtime
        if r.carrying is not None:
            obs[idx] = min(r.carry_steps / CARRY_TIMEOUT_STEPS, 1.0)
        else:
            obs[idx] = 0.0
        idx += 1

        return obs

    # ── PICK / DROP ──

    def _start_pick(self, ns, name, obj_type):
        r = self.robots[ns]
        self._claim_by_name(name)
        r._pick_name = name
        r._pick_type = obj_type
        r._pick_future = self.gz.delete_async(name)
        self._pub(ns, 0, 0, 0)

    def _finish_pick(self, ns):
        r = self.robots[ns]
        if not r._pick_future.done():
            self._pub(ns, 0, 0, 0)
            return
        res = r._pick_future.result()
        r._pick_future = None
        if res and res.success:
            r.carrying = r._pick_type
            r.carrying_name = r._pick_name
            r.carry_steps = 0
            self.collected_set.add(r._pick_name)
            m = String()
            m.data = r._pick_name
            self.collect_pub.publish(m)
            r.my_claim = None
            r.approach_target = None
            self.get_logger().info(f'[{ns}] ✓ Picked {r.carrying_name}')
        else:
            self._release_claim(ns)

    def _start_drop(self, ns):
        r = self.robots[ns]
        basket = CUBE_BASKET if r.carrying == 'cube' else SPHERE_BASKET
        r.drop_count += 1
        name = f'del_{ns}_{r.drop_count}'
        ox = float(basket[0]) + random.uniform(-0.3, 0.3)
        oy = float(basket[1]) + random.uniform(-0.3, 0.3)
        r._drop_future = self.gz.spawn_async(name, r.carrying, ox, oy)
        self._pub(ns, 0, 0, 0)

    def _finish_drop(self, ns):
        r = self.robots[ns]
        if not r._drop_future.done():
            self._pub(ns, 0, 0, 0)
            return
        self.get_logger().info(f'[{ns}] ✓ Dropped {r.carrying_name} at basket')
        if r.carrying_name:
            m = String()
            m.data = r.carrying_name
            self.drop_pub.publish(m)
        r.carrying = None
        r.carrying_name = None
        r.carry_steps = 0
        r._drop_future = None

    # ── Helpers ──

    def _best_available_track(self, ns):
        r = self.robots[ns]
        best = None
        for t in r.tracks:
            if not t.confirmed:
                continue
            d = math.sqrt((r.x - t.wx)**2 + (r.y - t.wy)**2)
            if d > CAMERA_RANGE:
                continue
            # Check if already claimed
            key = f'{t.wx:.1f}_{t.wy:.1f}'
            if key in self.claimed_set:
                continue
            if best is None or t.confidence > best.confidence:
                best = t
        return best

    def _claim_position(self, ns, wx, wy):
        """Claim by position (publish approach intent)."""
        r = self.robots[ns]
        key = f'{wx:.1f}_{wy:.1f}'
        r.my_claim = key
        self.claimed_set.add(key)
        m = String()
        m.data = key
        self.claim_pub.publish(m)

    def _claim_by_name(self, name):
        self.claimed_set.add(name)
        m = String()
        m.data = name
        self.claim_pub.publish(m)

    def _release_claim(self, ns):
        r = self.robots[ns]
        if r.my_claim:
            self.claimed_set.discard(r.my_claim)
            m = String()
            m.data = r.my_claim
            self.unclaim_pub.publish(m)
            r.my_claim = None
        r.approach_target = None

    def _check_collision(self, ns):
        r = self.robots[ns]
        closest = min_range(r.scan_ranges)
        if closest < 0.18 and not r._in_collision:
            r._collision_count += 1
            r._in_collision = True
        elif closest >= 0.25:
            r._in_collision = False

    def _pub(self, ns, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x = float(max(-MAX_SPEED, min(MAX_SPEED, vx)))
        cmd.linear.y = float(max(-MAX_SPEED, min(MAX_SPEED, vy)))
        cmd.angular.z = float(max(-MAX_WZ, min(MAX_WZ, wz)))
        self.cmd_pubs[ns].publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SwarmBrain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
