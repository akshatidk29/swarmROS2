"""
Production-level collector brain — omni-drive state machine.

States
------
EXPLORE  → roam to unvisited frontier cells, scan for objects
APPROACH → navigate to a confirmed tracked object
PICK     → delete object from Gazebo (async)
DELIVER  → navigate to the correct basket
DROP     → spawn object at basket (async)
ESCAPE   → systematic stuck recovery (move toward clearest direction)

Coordination topics (global, not namespaced)
--------------------------------------------
/claimed        object being targeted by a robot
/unclaimed      object released (approach failed)
/collected      object permanently removed
/visited_zones  exploration grid sync
/robot_poses    pose sharing for avoidance
"""

import math
import random
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge

from collector_bot.avoidance import compute_repulsion, find_clearest_direction, min_range
from collector_bot.detector import (
    detect_colors, fuse_with_lidar, update_tracks, TrackedObject,
)
from collector_bot.navigator import go_to_point
from collector_bot.explorer import Explorer
from collector_bot.gazebo_interface import GazeboInterface

# ═══════════════ WORLD KNOWLEDGE (must match arena.world) ═══════════════
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

# ═══════════════ CONSTANTS ═══════════════
PICK_DIST        = 0.60   # m — close enough to delete (generous)
BASKET_DIST      = 0.6    # m — close enough to basket
APPROACH_TIMEOUT = 10.0   # s — give up on approach
MATCH_TOLERANCE  = 1.5    # m — world-object matching radius

MAX_LIN   = 0.50          # m/s per axis  (fast)
MAX_ANG   = 2.0           # rad/s

COLLISION_DIST = 0.18     # m — LiDAR reading below this = collision
COLLISION_CLR  = 0.25     # m — hysteresis clear threshold

STUCK_DIST    = 0.05      # m moved in STUCK_WINDOW → stuck
STUCK_WINDOW  = 2.5       # s
ESCAPE_TIME   = 2.5       # s
EMERGENCY_DIST = 0.30     # m — suppress navigation below this
BLEND_DIST     = 0.60     # m — start blending nav down


# ═══════════════ NODE ═══════════════
class CollectorBrain(Node):

    EXPLORE  = 'EXPLORE'
    APPROACH = 'APPROACH'
    PICK     = 'PICK'
    DELIVER  = 'DELIVER'
    DROP     = 'DROP'
    ESCAPE   = 'ESCAPE'

    def __init__(self):
        super().__init__('collector_brain')

        self.declare_parameter('robot_ns', '')
        self.ns = self.get_parameter('robot_ns').value or ''
        pfx = f'/{self.ns}' if self.ns else ''

        # ── Subscribers ──
        self.create_subscription(LaserScan, f'{pfx}/scan', self._scan_cb, 10)
        self.create_subscription(Image, f'{pfx}/camera/image_raw',
                                 self._image_cb, 5)
        self.create_subscription(Odometry, f'{pfx}/odom', self._odom_cb, 10)
        self.create_subscription(String, '/claimed',       self._claimed_cb, 10)
        self.create_subscription(String, '/unclaimed',     self._unclaimed_cb, 10)
        self.create_subscription(String, '/collected',     self._collected_cb, 10)
        self.create_subscription(String, '/visited_zones', self._visited_cb, 10)
        self.create_subscription(String, '/robot_poses',   self._poses_cb, 10)

        # ── Publishers ──
        self.cmd_pub      = self.create_publisher(Twist,  f'{pfx}/cmd_vel', 10)
        self.claim_pub    = self.create_publisher(String, '/claimed', 10)
        self.unclaim_pub  = self.create_publisher(String, '/unclaimed', 10)
        self.collect_pub  = self.create_publisher(String, '/collected', 10)
        self.visited_pub  = self.create_publisher(String, '/visited_zones', 10)
        self.pose_pub     = self.create_publisher(String, '/robot_poses', 10)

        # ── Gazebo ──
        self.gz = GazeboInterface(self)

        # ── Modules ──
        self.explorer = Explorer()
        self.bridge   = CvBridge()

        # ── State ──
        self.state    = self.EXPLORE
        self.carrying = None          # 'cube' | 'sphere' | None
        self.carrying_name = None
        self.my_claim = None          # object name currently claimed

        self.claimed_set   = set()
        self.collected_set = set()

        self.scan_ranges   = []
        self.tracks        = []       # list[TrackedObject]
        self.x = self.y = self.yaw = 0.0
        self.other_poses   = {}       # {robot_name: (x, y)}

        self.approach_target = None   # TrackedObject
        self.approach_t0     = 0.0

        self._pick_future  = None
        self._drop_future  = None
        self._pick_name    = None
        self._pick_type    = None
        self.drop_count    = 0

        # stuck / escape
        self._pos_hist     = []
        self._esc_t0       = 0.0

        # collision tracking
        self._collision_count = 0
        self._in_collision    = False
        self._esc_target   = None   # (x, y) world target during escape or None

        # ── Publishers (extra) ──
        self.stats_pub = self.create_publisher(String, '/collision_stats', 10)

        # ── Timers ──
        self.create_timer(0.1,  self._tick)
        self.create_timer(0.5,  self._publish_pose)
        self.create_timer(2.0,  self._publish_visited)
        self.create_timer(5.0,  self._publish_stats)

        self.get_logger().info(f'[{self.ns}] Brain ready (omni-drive)')

    # ──────────────── callbacks ────────────────

    def _scan_cb(self, msg):
        self.scan_ranges = list(msg.ranges)

    def _image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return
        raw = detect_colors(img)
        fused = fuse_with_lidar(
            raw, self.scan_ranges, self.x, self.y, self.yaw)
        update_tracks(self.tracks, fused)

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _claimed_cb(self, msg):
        self.claimed_set.add(msg.data)

    def _unclaimed_cb(self, msg):
        self.claimed_set.discard(msg.data)

    def _collected_cb(self, msg):
        self.collected_set.add(msg.data)
        self.claimed_set.discard(msg.data)

    def _visited_cb(self, msg):
        cells = Explorer.decode_visited(msg.data)
        self.explorer.merge_remote(cells)

    def _poses_cb(self, msg):
        parts = msg.data.split(':')
        if len(parts) == 3 and parts[0] != self.ns:
            try:
                self.other_poses[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass

    # ── periodic publishers ──

    def _publish_pose(self):
        m = String()
        m.data = f'{self.ns}:{self.x:.3f}:{self.y:.3f}'
        self.pose_pub.publish(m)

    def _publish_visited(self):
        enc = self.explorer.encode_visited()
        if enc:
            m = String()
            m.data = enc
            self.visited_pub.publish(m)

    def _publish_stats(self):
        m = String()
        m.data = f'{self.ns}:{self._collision_count}'
        self.stats_pub.publish(m)
        self.get_logger().info(
            f'[{self.ns}] collisions={self._collision_count}  '
            f'state={self.state}  carrying={self.carrying}')

    def _check_collision(self):
        closest = min_range(self.scan_ranges)
        if closest < COLLISION_DIST and not self._in_collision:
            self._collision_count += 1
            self._in_collision = True
        elif closest >= COLLISION_CLR:
            self._in_collision = False

    # ──────────────── main tick ────────────────

    def _tick(self):
        now = time.time()
        self._check_stuck(now)
        self._check_collision()

        handler = {
            self.EXPLORE:  self._do_explore,
            self.APPROACH: self._do_approach,
            self.PICK:     self._do_pick,
            self.DELIVER:  self._do_deliver,
            self.DROP:     self._do_drop,
            self.ESCAPE:   self._do_escape,
        }
        handler[self.state]()

    # ──────────────── EXPLORE ────────────────

    def _do_explore(self):
        # Safety: if we're carrying something, go deliver it first!
        if self.carrying is not None:
            self.state = self.DELIVER
            return

        target = self.explorer.get_target(self.x, self.y)
        nav_vx, nav_vy, nav_wz, _ = go_to_point(
            target[0], target[1], self.x, self.y, self.yaw, arrival_dist=0.8)

        vx, vy, wz = self._blend(nav_vx, nav_vy, nav_wz)
        self._pub(vx, vy, wz)

        # look for confirmed, available targets — only if not carrying
        best = self._best_available_track()
        if best is not None:
            name = self._match_world(best.obj_type, best.wx, best.wy)
            if name:
                self._claim(name)
                self.approach_target = best
                self.approach_t0 = time.time()
                self.state = self.APPROACH
                self.get_logger().info(
                    f'[{self.ns}] → APPROACH {name}')

    # ──────────────── APPROACH ────────────────

    def _do_approach(self):
        t = self.approach_target
        if t is None or self.my_claim is None:
            self._release_claim()
            self.state = self.EXPLORE
            return

        # abort on timeout
        if time.time() - self.approach_t0 > APPROACH_TIMEOUT:
            self._release_claim()
            self.state = self.EXPLORE
            self.get_logger().info(f'[{self.ns}] Approach timeout')
            return

        # if another robot collected our target
        if self.my_claim in self.collected_set:
            self._release_claim()
            self.state = self.EXPLORE
            return

        nav_vx, nav_vy, nav_wz, arrived = go_to_point(
            t.wx, t.wy, self.x, self.y, self.yaw, arrival_dist=PICK_DIST)

        if arrived:
            self.state = self.PICK
            self._pub(0, 0, 0)
            return

        # Use LIGHT repulsion during approach — nav must dominate
        # so the robot can actually reach the object (the object itself
        # creates LiDAR readings that would otherwise kill navigation)
        rep_vx, rep_vy = self._repulsion()
        vx = nav_vx + rep_vx * 0.25
        vy = nav_vy + rep_vy * 0.25
        self._pub(vx, vy, nav_wz)

    # ──────────────── PICK ────────────────

    def _do_pick(self):
        if self._pick_future is None:
            name = self.my_claim
            if name is None:
                self.state = self.EXPLORE
                return
            obj_type = None
            for n, t, _, _ in WORLD_OBJECTS:
                if n == name:
                    obj_type = t
                    break
            if obj_type is None:
                self._release_claim()
                self.state = self.EXPLORE
                return
            self._pick_name = name
            self._pick_type = obj_type
            self._pick_future = self.gz.delete_async(name)
            self.get_logger().info(f'[{self.ns}] Picking {name}…')
            return

        if self._pick_future.done():
            res = self._pick_future.result()
            self._pick_future = None
            if res and res.success:
                self.carrying = self._pick_type
                self.carrying_name = self._pick_name
                self.collected_set.add(self._pick_name)
                self._broadcast('/collected', self._pick_name)
                self.claimed_set.discard(self._pick_name)
                self.my_claim = None
                self.state = self.DELIVER
                self.get_logger().info(
                    f'[{self.ns}] ✓ Picked {self.carrying_name}')
            else:
                self.get_logger().warn(f'[{self.ns}] Pick failed')
                self._release_claim()
                self.state = self.EXPLORE

        self._pub(0, 0, 0)

    # ──────────────── DELIVER ────────────────

    def _do_deliver(self):
        basket = CUBE_BASKET if self.carrying == 'cube' else SPHERE_BASKET
        nav_vx, nav_vy, nav_wz, arrived = go_to_point(
            basket[0], basket[1], self.x, self.y, self.yaw,
            arrival_dist=BASKET_DIST)

        if arrived:
            self.state = self.DROP
            self._pub(0, 0, 0)
            return

        vx, vy, wz = self._blend(nav_vx, nav_vy, nav_wz)
        self._pub(vx, vy, wz)

    # ──────────────── DROP ────────────────

    def _do_drop(self):
        if self._drop_future is None:
            basket = CUBE_BASKET if self.carrying == 'cube' else SPHERE_BASKET
            self.drop_count += 1
            name = f'del_{self.ns}_{self.drop_count}'
            ox = basket[0] + random.uniform(-0.3, 0.3)
            oy = basket[1] + random.uniform(-0.3, 0.3)
            self._drop_future = self.gz.spawn_async(
                name, self.carrying, ox, oy)
            return

        if self._drop_future.done():
            self.get_logger().info(
                f'[{self.ns}] ✓ Dropped {self.carrying_name} at basket')
            self.carrying = None
            self.carrying_name = None
            self._drop_future = None
            self.state = self.EXPLORE

        self._pub(0, 0, 0)

    # ──────────────── ESCAPE ────────────────

    def _do_escape(self):
        if time.time() - self._esc_t0 > ESCAPE_TIME:
            # Return to DELIVER if carrying, otherwise EXPLORE
            if self.carrying is not None:
                self.state = self.DELIVER
            else:
                self.state = self.EXPLORE
            return

        # Re-scan every tick for the best direction to move
        target_local = None
        if self._esc_target is not None:
            dx = self._esc_target[0] - self.x
            dy = self._esc_target[1] - self.y
            global_a = math.atan2(dy, dx)
            target_local = global_a - self.yaw

        angle, _ = find_clearest_direction(
            self.scan_ranges, target_angle_rad=target_local)
        vx = MAX_LIN * math.cos(angle)
        vy = MAX_LIN * math.sin(angle)
        self._pub(vx, vy, 0.4)

    # ──────────────── helpers ────────────────

    def _repulsion(self):
        """Raw repulsive velocity (robot frame)."""
        poses = list(self.other_poses.values())
        return compute_repulsion(
            self.scan_ranges, poses, self.x, self.y, self.yaw)

    def _blend(self, nav_vx, nav_vy, nav_wz):
        """
        Blend navigation with repulsion.

        - Far from obstacles: nav + rep (normal)
        - Inside BLEND_DIST:  nav fades out, rep amplifies
        - Inside EMERGENCY_DIST: nav suppressed, pure repulsion
        """
        rep_vx, rep_vy = self._repulsion()
        closest = min_range(self.scan_ranges)

        if closest < EMERGENCY_DIST:
            # Navigation is the PROBLEM — kill it, amplify repulsion
            alpha = 0.0
            rep_scale = 3.0
        elif closest < BLEND_DIST:
            # Linearly fade navigation, boost repulsion
            alpha = (closest - EMERGENCY_DIST) / (BLEND_DIST - EMERGENCY_DIST)
            rep_scale = 1.0 + (1.0 - alpha) * 2.0   # 1× … 3×
        else:
            alpha = 1.0
            rep_scale = 1.0

        vx = alpha * nav_vx + rep_scale * rep_vx
        vy = alpha * nav_vy + rep_scale * rep_vy
        wz = alpha * nav_wz

        # Guarantee minimum escape speed when very close
        if closest < EMERGENCY_DIST:
            mag = math.sqrt(vx * vx + vy * vy)
            if mag < 0.15:
                # Force movement away (repulsion direction)
                rmag = math.sqrt(rep_vx * rep_vx + rep_vy * rep_vy)
                if rmag > 0.01:
                    vx = rep_vx / rmag * 0.25
                    vy = rep_vy / rmag * 0.25
                else:
                    vx = -0.25   # fallback: reverse

        return vx, vy, wz

    def _best_available_track(self):
        """Return highest-confidence confirmed track not claimed/collected."""
        best = None
        for t in self.tracks:
            if not t.confirmed:
                continue
            name = self._match_world(t.obj_type, t.wx, t.wy)
            if name is None:
                continue
            if name in self.collected_set or name in self.claimed_set:
                continue
            if best is None or t.confidence > best.confidence:
                best = t
        return best

    def _match_world(self, obj_type, wx, wy):
        """Find closest uncollected world object matching type + position."""
        best_name, best_d = None, float('inf')
        for name, otype, ox, oy in WORLD_OBJECTS:
            if otype != obj_type or name in self.collected_set:
                continue
            d = math.sqrt((wx - ox) ** 2 + (wy - oy) ** 2)
            if d < best_d:
                best_name, best_d = name, d
        return best_name if best_d < MATCH_TOLERANCE else None

    def _claim(self, name):
        self.my_claim = name
        self.claimed_set.add(name)
        self._broadcast('/claimed', name)

    def _release_claim(self):
        if self.my_claim:
            self.claimed_set.discard(self.my_claim)
            self._broadcast('/unclaimed', self.my_claim)
            self.my_claim = None
        self.approach_target = None

    def _broadcast(self, topic, data):
        m = String()
        m.data = data
        if topic == '/claimed':
            self.claim_pub.publish(m)
        elif topic == '/unclaimed':
            self.unclaim_pub.publish(m)
        elif topic == '/collected':
            self.collect_pub.publish(m)

    def _check_stuck(self, now):
        if self.state in (self.PICK, self.DROP, self.ESCAPE):
            self._pos_hist.clear()
            return
        self._pos_hist.append((now, self.x, self.y))
        self._pos_hist = [p for p in self._pos_hist
                          if now - p[0] < STUCK_WINDOW]
        if len(self._pos_hist) < int(STUCK_WINDOW / 0.1):
            return
        dx = self.x - self._pos_hist[0][1]
        dy = self.y - self._pos_hist[0][2]
        if math.sqrt(dx * dx + dy * dy) < STUCK_DIST:
            # Determine what target the robot was heading toward
            self._esc_target = None
            if self.state == self.DELIVER and self.carrying:
                self._esc_target = CUBE_BASKET if self.carrying == 'cube' else SPHERE_BASKET
            elif self.state == self.APPROACH and self.approach_target:
                self._esc_target = (self.approach_target.wx, self.approach_target.wy)
            elif self.state == self.EXPLORE:
                t = self.explorer.get_target(self.x, self.y)
                self._esc_target = t

            self._esc_t0 = now
            self._pos_hist.clear()
            if self.state == self.APPROACH:
                self._release_claim()
            self.state = self.ESCAPE
            self.get_logger().warn(f'[{self.ns}] STUCK → escape')

    def _pub(self, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x  = float(max(-MAX_LIN, min(MAX_LIN, vx)))
        cmd.linear.y  = float(max(-MAX_LIN, min(MAX_LIN, vy)))
        cmd.angular.z = float(max(-MAX_ANG, min(MAX_ANG, wz)))
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CollectorBrain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
