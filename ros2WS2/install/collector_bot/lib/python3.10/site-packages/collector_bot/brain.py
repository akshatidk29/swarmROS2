"""
Collector brain — state-machine node that coordinates roaming, detection,
pick-up and delivery for a single namespaced robot.

States:  ROAM → APPROACH → PICK → DELIVER → DROP → ROAM
         any  → UNSTUCK  → ROAM   (stuck recovery)
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

from collector_bot.avoidance import compute_avoidance
from collector_bot.detector import detect_objects
from collector_bot.gazebo_interface import GazeboInterface

# ═══════════════════════ WORLD KNOWLEDGE ═══════════════════════
# Must match arena.world
WORLD_OBJECTS = [
    ('cube_1',   'cube',   1.5,  1.5),
    ('cube_2',   'cube',  -3.0,  0.5),
    ('cube_3',   'cube',   4.0, -3.0),
    ('cube_4',   'cube',  -1.0, -3.5),
    ('cube_5',   'cube',   2.0,  3.5),
    ('sphere_1', 'sphere', -2.0,  3.0),
    ('sphere_2', 'sphere',  1.0, -1.5),
    ('sphere_3', 'sphere', -4.0, -1.0),
    ('sphere_4', 'sphere',  3.5,  1.0),
    ('sphere_5', 'sphere', -0.5,  4.0),
]
CUBE_BASKET   = (5.0,  5.0)
SPHERE_BASKET = (-5.0, 5.0)

# ═══════════════════════ THRESHOLDS ════════════════════════════
MIN_DETECT_AREA  = 400      # ignore smaller blobs
PICK_AREA        = 12000    # camera blob area → close enough to pick
PICK_RADIUS      = 0.7      # max odom distance (m) to match object
APPROACH_TIMEOUT = 4.0      # seconds before losing target

MAX_VX  = 0.30
MIN_VX  = 0.10
MAX_WZ  = 1.5

STUCK_DIST       = 0.08     # m in STUCK_WINDOW → stuck
STUCK_WINDOW     = 3.0      # seconds
UNSTUCK_DURATION = 2.0      # reverse time
BASKET_RADIUS    = 0.6      # m, "close enough" to basket

IMG_WIDTH = 640


# ═══════════════════════ NODE ═════════════════════════════════
class CollectorBrain(Node):

    # States
    ROAM     = 'ROAM'
    APPROACH = 'APPROACH'
    PICK     = 'PICK'
    DELIVER  = 'DELIVER'
    DROP     = 'DROP'
    UNSTUCK  = 'UNSTUCK'

    def __init__(self):
        super().__init__('collector_brain')

        # ── params ──
        self.declare_parameter('robot_ns', '')
        self.ns = self.get_parameter('robot_ns').value or ''
        pfx = f'/{self.ns}' if self.ns else ''

        # ── subscribers ──
        self.create_subscription(LaserScan, f'{pfx}/scan',
                                 self._scan_cb, 10)
        self.create_subscription(Image, f'{pfx}/camera/image_raw',
                                 self._image_cb, 5)
        self.create_subscription(Odometry, f'{pfx}/odom',
                                 self._odom_cb, 10)
        self.create_subscription(String, '/collected',
                                 self._collected_cb, 10)

        # ── publishers ──
        self.cmd_pub       = self.create_publisher(Twist, f'{pfx}/cmd_vel', 10)
        self.collected_pub = self.create_publisher(String, '/collected', 10)

        # ── gazebo ──
        self.gz = GazeboInterface(self)

        # ── state ──
        self.state          = self.ROAM
        self.carrying       = None      # 'cube' | 'sphere' | None
        self.carrying_name  = None
        self.collected_set  = set()

        self.scan_ranges    = []
        self.detections     = []        # [(type, cx, cy, area)]
        self.x = self.y = self.yaw = 0.0

        self.wander_bias       = 0.0
        self.approach_lost_t   = 0.0

        # async futures for pick / drop
        self._pick_future      = None
        self._pick_target_name = None
        self._pick_target_type = None
        self._drop_future      = None
        self.deliver_count     = 0

        # stuck detection
        self._pos_hist    = []          # [(t, x, y)]
        self._unstuck_t0  = 0.0
        self._unstuck_wz  = 0.0

        self.bridge = CvBridge()

        # ── timers ──
        self.create_timer(0.1, self._tick)
        self.create_timer(2.5, self._new_wander)

        self.get_logger().info(f'[{self.ns}] Brain node started')

    # ─────────────── callbacks ───────────────

    def _scan_cb(self, msg):
        self.scan_ranges = list(msg.ranges)

    def _image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.detections = detect_objects(img)
        except Exception:
            self.detections = []

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _collected_cb(self, msg):
        self.collected_set.add(msg.data)

    def _new_wander(self):
        self.wander_bias = random.uniform(-0.35, 0.35)

    # ─────────────── main tick ───────────────

    def _tick(self):
        now = time.time()
        self._check_stuck(now)

        if   self.state == self.ROAM:     self._do_roam()
        elif self.state == self.APPROACH: self._do_approach()
        elif self.state == self.PICK:     self._do_pick()
        elif self.state == self.DELIVER:  self._do_deliver()
        elif self.state == self.DROP:     self._do_drop()
        elif self.state == self.UNSTUCK:  self._do_unstuck(now)

    # ─────────────── states ───────────────

    def _do_roam(self):
        avd = compute_avoidance(self.scan_ranges)
        vx = max(avd['vx'], MIN_VX)
        wz = avd['wz'] + self.wander_bias

        # look for objects
        if self.carrying is None and self.detections:
            best = self.detections[0]
            if best[3] > MIN_DETECT_AREA:
                self.state = self.APPROACH
                self.approach_lost_t = time.time()
                self.get_logger().info(
                    f'[{self.ns}] See {best[0]} (area={best[3]}), approaching')

        self._pub(vx, wz)

    def _do_approach(self):
        now = time.time()
        avd = compute_avoidance(self.scan_ranges)

        if not self.detections:
            if now - self.approach_lost_t > APPROACH_TIMEOUT:
                self._to_roam('Lost target')
                return
            self._pub(max(avd['vx'], MIN_VX), avd['wz'])
            return

        self.approach_lost_t = now
        obj_type, cx, cy, area = self.detections[0]

        # close enough?
        if area > PICK_AREA:
            self.state = self.PICK
            self._pub(0.0, 0.0)
            return

        # steer toward blob centre
        err = (cx - IMG_WIDTH / 2) / (IMG_WIDTH / 2)   # −1 … +1
        task_wz = -err * 2.0
        ratio = min(area / PICK_AREA, 1.0)
        task_vx = MAX_VX * (1.0 - 0.5 * ratio)

        vx, wz = self._blend(task_vx, task_wz, avd, relax_front=True)
        self._pub(max(vx, MIN_VX), wz)

    def _do_pick(self):
        # phase 1: start async delete
        if self._pick_future is None:
            det_type = self.detections[0][0] if self.detections else 'cube'
            name, otype, dist = self._closest_object(det_type)

            if name and dist < PICK_RADIUS:
                self._pick_target_name = name
                self._pick_target_type = otype
                self._pick_future = self.gz.delete_async(name)
                self.get_logger().info(
                    f'[{self.ns}] Picking {name} (d={dist:.2f}m)…')
            else:
                self._to_roam(f'Nothing within {PICK_RADIUS}m')
            return

        # phase 2: wait for result
        if self._pick_future.done():
            res = self._pick_future.result()
            if res and res.success:
                self.carrying = self._pick_target_type
                self.carrying_name = self._pick_target_name
                self.collected_set.add(self._pick_target_name)
                self._broadcast_collected(self._pick_target_name)
                basket = 'cube' if self.carrying == 'cube' else 'sphere'
                self.get_logger().info(
                    f'[{self.ns}] ✓ Picked {self.carrying_name}, '
                    f'delivering to {basket} basket')
                self.state = self.DELIVER
            else:
                self._to_roam('Delete failed')
            self._pick_future = None

        self._pub(0.0, 0.0)     # stay still while picking

    def _do_deliver(self):
        target = CUBE_BASKET if self.carrying == 'cube' else SPHERE_BASKET
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < BASKET_RADIUS:
            self.state = self.DROP
            return

        desired = math.atan2(dy, dx)
        herr = self._norm_angle(desired - self.yaw)

        task_vx = min(MAX_VX, dist * 0.4)
        task_wz = max(-MAX_WZ, min(MAX_WZ, herr * 2.5))

        avd = compute_avoidance(self.scan_ranges)
        vx, wz = self._blend(task_vx, task_wz, avd)
        self._pub(max(vx, MIN_VX), wz)

    def _do_drop(self):
        if self._drop_future is None:
            target = CUBE_BASKET if self.carrying == 'cube' else SPHERE_BASKET
            self.deliver_count += 1
            name = f'del_{self.ns}_{self.deliver_count}'
            ox = target[0] + random.uniform(-0.3, 0.3)
            oy = target[1] + random.uniform(-0.3, 0.3)
            self._drop_future = self.gz.spawn_async(
                name, self.carrying, ox, oy)
            return

        if self._drop_future.done():
            self.get_logger().info(
                f'[{self.ns}] ✓ Dropped {self.carrying_name} at basket!')
            self.carrying = None
            self.carrying_name = None
            self._drop_future = None
            self.state = self.ROAM

        self._pub(0.0, 0.0)

    def _do_unstuck(self, now):
        if now - self._unstuck_t0 > UNSTUCK_DURATION:
            self.state = self.ROAM
            return
        self._pub(-MIN_VX, self._unstuck_wz)

    # ─────────────── helpers ───────────────

    def _to_roam(self, reason=''):
        self.state = self.ROAM
        self._pick_future = None
        if reason:
            self.get_logger().info(f'[{self.ns}] → ROAM ({reason})')

    def _closest_object(self, desired_type):
        """Return (name, type, distance) of nearest un-collected object."""
        best_name, best_type, best_dist = None, None, float('inf')
        for name, otype, ox, oy in WORLD_OBJECTS:
            if name in self.collected_set:
                continue
            if otype != desired_type:
                continue
            d = math.sqrt((self.x - ox) ** 2 + (self.y - oy) ** 2)
            if d < best_dist:
                best_name, best_type, best_dist = name, otype, d
        return best_name, best_type, best_dist

    def _broadcast_collected(self, name):
        msg = String()
        msg.data = name
        self.collected_pub.publish(msg)

    def _blend(self, task_vx, task_wz, avd, relax_front=False):
        """Merge task velocity with avoidance.  Avoidance wins when close."""
        f = avd['front']
        if not relax_front:
            if f < 0.20:
                return avd['vx'], avd['wz']
            if f < 0.50:
                a = (f - 0.20) / 0.30
                return (a * task_vx + (1 - a) * avd['vx'],
                        a * task_wz + (1 - a) * avd['wz'])
        else:
            # relax front during approach but still dodge side obstacles
            if avd['f_left'] < 0.35 or avd['f_right'] < 0.35:
                return task_vx, 0.5 * task_wz + 0.5 * avd['wz']
        return task_vx, task_wz

    def _check_stuck(self, now):
        """If robot barely moved in STUCK_WINDOW seconds, enter UNSTUCK."""
        if self.state in (self.PICK, self.DROP, self.UNSTUCK):
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
            self.state = self.UNSTUCK
            self._unstuck_t0 = now
            self._unstuck_wz = random.choice([-MAX_WZ, MAX_WZ])
            self._pos_hist.clear()
            self.get_logger().warn(f'[{self.ns}] STUCK → reversing')

    def _pub(self, vx, wz):
        cmd = Twist()
        cmd.linear.x = float(max(-MAX_VX, min(MAX_VX, vx)))
        cmd.angular.z = float(max(-MAX_WZ, min(MAX_WZ, wz)))
        self.cmd_pub.publish(cmd)

    @staticmethod
    def _norm_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a


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
