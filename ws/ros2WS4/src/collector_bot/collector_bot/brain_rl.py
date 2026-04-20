"""
Hybrid RL brain — deterministic target selection + RL navigation.

The deterministic layer (state machine) decides WHAT to do:
  EXPLORE  → target = nearest unvisited frontier cell
  APPROACH → target = detected object world position
  DELIVER  → target = correct basket
  PICK/DROP → Gazebo service calls (same as brain.py)

The RL layer decides HOW to get there:
  obs = [lidar_36, other_bots_9, target_3, self_5, detections_9, basket_3]
  action = policy.predict(obs) → [vx, vy, wz]
"""

import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge

from collector_bot.detector import detect_colors, fuse_with_lidar, update_tracks
from collector_bot.explorer import Explorer
from collector_bot.gazebo_interface import GazeboInterface
from collector_bot.avoidance import min_range

# Try importing SB3 for inference
try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

# ─── Constants (must match RL/training/config.py) ───
ARENA_HALF   = 6.0
ARENA_SIZE   = 12.0
MAX_SPEED    = 0.50
MAX_WZ       = 2.0
LIDAR_RAYS   = 36
LIDAR_MAX    = 6.0
CAMERA_FOV   = 1.2
CAMERA_RANGE = 4.0
N_DETECTIONS = 3
OBS_DIM      = 65
PICK_DIST    = 0.60
BASKET_DIST  = 0.60
SAFETY_DIST  = 0.20

WORLD_OBJECTS = [
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
CUBE_BASKET   = (5.0,  5.0)
SPHERE_BASKET = (-5.0, 5.0)

MATCH_TOLERANCE = 1.5


class CollectorBrainRL(Node):

    EXPLORE  = 'EXPLORE'
    APPROACH = 'APPROACH'
    PICK     = 'PICK'
    DELIVER  = 'DELIVER'
    DROP     = 'DROP'

    def __init__(self):
        super().__init__('collector_brain_rl')

        self.declare_parameter('robot_ns', '')
        self.declare_parameter('model_path', '')
        self.ns = self.get_parameter('robot_ns').value or ''
        model_path = self.get_parameter('model_path').value

        pfx = f'/{self.ns}' if self.ns else ''

        # ── Subscribers ──
        self.create_subscription(LaserScan, f'{pfx}/scan', self._scan_cb, 10)
        self.create_subscription(Image, f'{pfx}/camera/image_raw',
                                 self._image_cb, 5)
        self.create_subscription(Odometry, f'{pfx}/odom', self._odom_cb, 10)
        self.create_subscription(String, '/claimed', self._claimed_cb, 10)
        self.create_subscription(String, '/unclaimed', self._unclaimed_cb, 10)
        self.create_subscription(String, '/collected', self._collected_cb, 10)
        self.create_subscription(String, '/visited_zones', self._visited_cb, 10)
        self.create_subscription(String, '/robot_poses', self._poses_cb, 10)

        # ── Publishers ──
        self.cmd_pub     = self.create_publisher(Twist,  f'{pfx}/cmd_vel', 10)
        self.claim_pub   = self.create_publisher(String, '/claimed', 10)
        self.unclaim_pub = self.create_publisher(String, '/unclaimed', 10)
        self.collect_pub = self.create_publisher(String, '/collected', 10)
        self.drop_pub    = self.create_publisher(String, '/dropped', 10)
        self.visited_pub = self.create_publisher(String, '/visited_zones', 10)
        self.pose_pub    = self.create_publisher(String, '/robot_poses', 10)
        self.stats_pub   = self.create_publisher(String, '/collision_stats', 10)

        # ── Gazebo ──
        self.gz = GazeboInterface(self)

        # ── Modules ──
        self.explorer = Explorer()
        self.bridge   = CvBridge()

        # ── Load RL model ──
        self.policy = None
        if HAS_SB3 and model_path:
            try:
                self.policy = PPO.load(model_path)
                self.get_logger().info(f'[{self.ns}] RL model loaded: {model_path}')
            except Exception as e:
                self.get_logger().error(f'[{self.ns}] Failed to load model: {e}')
        else:
            self.get_logger().warn(
                f'[{self.ns}] No RL model — using fallback navigator')

        # ── State ──
        self.state = self.EXPLORE
        self.carrying = None
        self.carrying_name = None
        self.my_claim = None
        self.target = None    # (tx, ty) — the target the RL agent navigates to

        self.claimed_set   = set()
        self.collected_set = set()
        self.scan_ranges   = []
        self.tracks        = []
        self.x = self.y = self.yaw = 0.0
        self.vx = self.vy = 0.0
        self.other_poses   = {}

        self.approach_target = None
        self.approach_t0     = 0.0

        self._pick_future  = None
        self._drop_future  = None
        self._pick_name    = None
        self._pick_type    = None
        self.drop_count    = 0

        self._collision_count = 0
        self._in_collision    = False

        import random
        self._rng = random

        # ── Timers ──
        self.create_timer(0.1,  self._tick)
        self.create_timer(0.5,  self._publish_pose)
        self.create_timer(2.0,  self._publish_visited)
        self.create_timer(5.0,  self._publish_stats)

        self.get_logger().info(f'[{self.ns}] RL Brain ready')

    # ── Callbacks (same as brain.py) ──

    def _scan_cb(self, msg):
        self.scan_ranges = list(msg.ranges)

    def _image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return
        raw = detect_colors(img)
        fused = fuse_with_lidar(raw, self.scan_ranges, self.x, self.y, self.yaw)
        update_tracks(self.tracks, fused)

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y

    def _claimed_cb(self, msg): self.claimed_set.add(msg.data)
    def _unclaimed_cb(self, msg): self.claimed_set.discard(msg.data)
    def _collected_cb(self, msg):
        self.collected_set.add(msg.data)
        self.claimed_set.discard(msg.data)
    def _visited_cb(self, msg):
        self.explorer.merge_remote(Explorer.decode_visited(msg.data))
    def _poses_cb(self, msg):
        parts = msg.data.split(':')
        if len(parts) == 3 and parts[0] != self.ns:
            try:
                self.other_poses[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass

    def _publish_pose(self):
        m = String(); m.data = f'{self.ns}:{self.x:.3f}:{self.y:.3f}'
        self.pose_pub.publish(m)

    def _publish_visited(self):
        enc = self.explorer.encode_visited()
        if enc:
            m = String(); m.data = enc
            self.visited_pub.publish(m)

    def _publish_stats(self):
        m = String(); m.data = f'{self.ns}:{self._collision_count}'
        self.stats_pub.publish(m)
        self.get_logger().info(
            f'[{self.ns}] RL | state={self.state} carry={self.carrying} '
            f'coll={self._collision_count}')

    # ── Main tick ──

    def _tick(self):
        self._check_collision()

        if self.carrying is not None and self.state == self.EXPLORE:
            self.state = self.DELIVER

        if self.state == self.PICK:
            self._do_pick(); return
        if self.state == self.DROP:
            self._do_drop(); return

        # ── Deterministic: set target ──
        self._update_target()

        # ── RL: navigate to target ──
        if self.target is not None and self.policy is not None:
            obs = self._build_obs()
            action, _ = self.policy.predict(obs, deterministic=True)
            vx = float(action[0]) * MAX_SPEED
            vy = float(action[1]) * MAX_SPEED
            wz = float(action[2]) * MAX_WZ

            # Safety layer: override if pushing into obstacle
            closest = min_range(self.scan_ranges)
            if closest < SAFETY_DIST:
                from collector_bot.avoidance import compute_repulsion
                poses = list(self.other_poses.values())
                rx, ry = compute_repulsion(
                    self.scan_ranges, poses, self.x, self.y, self.yaw)
                mag = math.sqrt(rx*rx + ry*ry)
                if mag > 0.01:
                    vx = rx / mag * 0.3
                    vy = ry / mag * 0.3

            self._pub(vx, vy, wz)
        elif self.target is not None:
            # Fallback: simple proportional navigation
            from collector_bot.navigator import go_to_point
            vx, vy, wz, _ = go_to_point(
                self.target[0], self.target[1],
                self.x, self.y, self.yaw, arrival_dist=0.5)
            self._pub(vx, vy, wz)
        else:
            self._pub(0, 0, 0)

        # ── Auto-pick ──
        if self.carrying is None and self.state == self.APPROACH and self.my_claim:
            for name, otype, ox, oy in WORLD_OBJECTS:
                if name == self.my_claim:
                    d = math.sqrt((self.x - ox)**2 + (self.y - oy)**2)
                    if d < PICK_DIST:
                        self.state = self.PICK
                    break

        # ── Auto-drop ──
        if self.carrying is not None and self.state == self.DELIVER:
            basket = CUBE_BASKET if self.carrying == 'cube' else SPHERE_BASKET
            d = math.sqrt((self.x - basket[0])**2 + (self.y - basket[1])**2)
            if d < BASKET_DIST:
                self.state = self.DROP

    def _update_target(self):
        """Deterministic target selection based on state."""
        if self.state == self.DELIVER and self.carrying:
            basket = CUBE_BASKET if self.carrying == 'cube' else SPHERE_BASKET
            self.target = basket

        elif self.state == self.APPROACH and self.approach_target:
            self.target = (self.approach_target.wx, self.approach_target.wy)
            # Timeout
            if time.time() - self.approach_t0 > 10.0:
                self._release_claim()
                self.state = self.EXPLORE

        elif self.state == self.EXPLORE:
            # Check for detectable objects
            best = self._best_available_track()
            if best is not None:
                name = self._match_world(best.obj_type, best.wx, best.wy)
                if name:
                    self._claim(name)
                    self.approach_target = best
                    self.approach_t0 = time.time()
                    self.state = self.APPROACH
                    self.target = (best.wx, best.wy)
                    return

            # Frontier exploration
            t = self.explorer.get_target(self.x, self.y)
            self.target = t

    # ── Observation builder (must match training env) ──

    def _build_obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        # 1) LiDAR (36 rays) — subsample from 360
        n = len(self.scan_ranges)
        for r in range(LIDAR_RAYS):
            if n > 0:
                deg = r * (360 / LIDAR_RAYS)
                li = int(round(((deg + 180) % 360) * n / 360)) % n
                val = self.scan_ranges[li]
                if math.isinf(val) or math.isnan(val):
                    val = LIDAR_MAX
                obs[idx] = min(val, LIDAR_MAX) / LIDAR_MAX
            else:
                obs[idx] = 1.0
            idx += 1

        # 2) Other robots (3 × 3)
        others = list(self.other_poses.values())[:3]
        for i in range(3):
            if i < len(others):
                ox, oy = others[i]
                dx, dy = ox - self.x, oy - self.y
                d = math.sqrt(dx*dx + dy*dy)
                cos_y = math.cos(-self.yaw)
                sin_y = math.sin(-self.yaw)
                rel_x = dx * cos_y - dy * sin_y
                rel_y = dx * sin_y + dy * cos_y
                obs[idx]     = np.clip(rel_x / ARENA_SIZE, -1, 1)
                obs[idx + 1] = np.clip(rel_y / ARENA_SIZE, -1, 1)
                obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
            idx += 3

        # 3) Target relative (3)
        if self.target is not None:
            tx, ty = self.target
            dx, dy = tx - self.x, ty - self.y
            d = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx) - self.yaw
            obs[idx]     = math.sin(angle)
            obs[idx + 1] = math.cos(angle)
            obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
        idx += 3

        # 4) Self state (5)
        carry_val = self._carry_int()
        obs[idx] = carry_val / 2.0
        obs[idx + 1] = np.clip(self.vx / MAX_SPEED, -1, 1)
        obs[idx + 2] = np.clip(self.vy / MAX_SPEED, -1, 1)
        obs[idx + 3] = math.sin(self.yaw)
        obs[idx + 4] = math.cos(self.yaw)
        idx += 5

        # 5) Detections (3 × 3)
        confirmed = [t for t in self.tracks if t.confirmed]
        confirmed.sort(key=lambda t: math.sqrt(
            (t.wx - self.x)**2 + (t.wy - self.y)**2))
        for i in range(N_DETECTIONS):
            if i < len(confirmed):
                t = confirmed[i]
                dx, dy = t.wx - self.x, t.wy - self.y
                d = math.sqrt(dx*dx + dy*dy)
                angle = math.atan2(dy, dx) - self.yaw
                obs[idx]     = 1.0 if t.obj_type == 'cube' else -1.0
                obs[idx + 1] = angle / math.pi
                obs[idx + 2] = np.clip(d / CAMERA_RANGE, 0, 1)
            idx += 3

        # 6) Basket relative (3)
        if self.carrying == 'cube':
            bx, by = CUBE_BASKET
        elif self.carrying == 'sphere':
            bx, by = SPHERE_BASKET
        else:
            bx, by = CUBE_BASKET
        dx, dy = bx - self.x, by - self.y
        d = math.sqrt(dx*dx + dy*dy)
        angle = math.atan2(dy, dx) - self.yaw
        obs[idx]     = math.sin(angle)
        obs[idx + 1] = math.cos(angle)
        obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)

        return obs

    # ── PICK / DROP (same as brain.py) ──

    def _do_pick(self):
        if self._pick_future is None:
            name = self.my_claim
            if not name:
                self.state = self.EXPLORE; return
            obj_type = None
            for n, t, _, _ in WORLD_OBJECTS:
                if n == name: obj_type = t; break
            if not obj_type:
                self._release_claim(); self.state = self.EXPLORE; return
            self._pick_name = name
            self._pick_type = obj_type
            self._pick_future = self.gz.delete_async(name)
            return
        if self._pick_future.done():
            res = self._pick_future.result()
            self._pick_future = None
            if res and res.success:
                self.carrying = self._pick_type
                self.carrying_name = self._pick_name
                self.collected_set.add(self._pick_name)
                m = String(); m.data = self._pick_name
                self.collect_pub.publish(m)
                self.my_claim = None
                self.state = self.DELIVER
                self.get_logger().info(f'[{self.ns}] ✓ Picked {self.carrying_name}')
            else:
                self._release_claim(); self.state = self.EXPLORE
        self._pub(0, 0, 0)

    def _do_drop(self):
        if self._drop_future is None:
            basket = CUBE_BASKET if self.carrying == 'cube' else SPHERE_BASKET
            self.drop_count += 1
            name = f'del_{self.ns}_{self.drop_count}'
            ox = basket[0] + self._rng.uniform(-0.3, 0.3)
            oy = basket[1] + self._rng.uniform(-0.3, 0.3)
            self._drop_future = self.gz.spawn_async(name, self.carrying, ox, oy)
            return
        if self._drop_future.done():
            self.get_logger().info(
                f'[{self.ns}] ✓ Dropped {self.carrying_name} at basket')
            if self.carrying_name:
                m = String(); m.data = self.carrying_name
                self.drop_pub.publish(m)
            self.carrying = None
            self.carrying_name = None
            self._drop_future = None
            self.state = self.EXPLORE
        self._pub(0, 0, 0)

    # ── Helpers ──

    def _best_available_track(self):
        best = None
        for t in self.tracks:
            if not t.confirmed: continue
            name = self._match_world(t.obj_type, t.wx, t.wy)
            if not name: continue
            if name in self.collected_set or name in self.claimed_set: continue
            if best is None or t.confidence > best.confidence:
                best = t
        return best

    def _match_world(self, obj_type, wx, wy):
        best_name, best_d = None, float('inf')
        for name, otype, ox, oy in WORLD_OBJECTS:
            if otype != obj_type or name in self.collected_set: continue
            d = math.sqrt((wx - ox)**2 + (wy - oy)**2)
            if d < best_d: best_name, best_d = name, d
        return best_name if best_d < MATCH_TOLERANCE else None

    def _claim(self, name):
        self.my_claim = name
        self.claimed_set.add(name)
        m = String(); m.data = name
        self.claim_pub.publish(m)

    def _release_claim(self):
        if self.my_claim:
            self.claimed_set.discard(self.my_claim)
            m = String(); m.data = self.my_claim
            self.unclaim_pub.publish(m)
            self.my_claim = None
        self.approach_target = None

    def _check_collision(self):
        closest = min_range(self.scan_ranges)
        if closest < 0.18 and not self._in_collision:
            self._collision_count += 1
            self._in_collision = True
        elif closest >= 0.25:
            self._in_collision = False

    def _carry_int(self):
        if self.carrying == 'cube':
            return 1
        if self.carrying == 'sphere':
            return 2
        return 0

    def _pub(self, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x  = float(max(-MAX_SPEED, min(MAX_SPEED, vx)))
        cmd.linear.y  = float(max(-MAX_SPEED, min(MAX_SPEED, vy)))
        cmd.angular.z = float(max(-MAX_WZ, min(MAX_WZ, wz)))
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CollectorBrainRL()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
