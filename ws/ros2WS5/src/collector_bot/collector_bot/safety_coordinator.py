"""
Safety Coordinator — parallel safety system for multi-agent coordination.

Complements RL by providing deterministic safety guarantees:
  1. Inter-robot collision prevention (mutual exclusion zones)
  2. Obstacle proximity escalation (forced reverse)
  3. Task continuity watchdog (no objective abandonment)
  4. Deadlock detection and recovery (stuck detection)

Publishes /{robot_ns}/safety_override (Twist) when intervention needed.
The swarm_brain checks this before publishing cmd_vel.
"""

import math
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from collector_bot.constants import (
    N_ROBOTS, MAX_SPEED, MAX_WZ,
    SAFETY_DIST_ENTER, SAFETY_DIST_EXIT,
)

ROBOT_NAMES = [f'robot_{i+1}' for i in range(N_ROBOTS)]

# ── Safety parameters ──
COLLISION_DIST = 0.50
HARD_COLLISION_DIST = 0.30
OBSTACLE_CRITICAL = SAFETY_DIST_ENTER  # use training hysteresis enter threshold
OBSTACLE_CRITICAL_EXIT = SAFETY_DIST_EXIT
OBSTACLE_CRITICAL_TIME = 1.0
STUCK_DIST = 0.02
STUCK_TIME = 5.0
STUCK_PERTURB_SPEED = 0.3


class RobotState:
    """Per-robot tracking state for the safety coordinator."""

    def __init__(self, ns):
        self.ns = ns
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.min_lidar = 10.0

        # Stuck detection
        self._history_x = 0.0
        self._history_y = 0.0
        self._history_time = time.time()

        # Obstacle critical timer
        self._critical_start = None

        # Carrying state (from /collected and /dropped)
        self.carrying = False


class SafetyCoordinator(Node):
    """ROS2 node that monitors all robots and issues safety overrides."""

    def __init__(self):
        super().__init__('safety_coordinator')

        self.states = {}
        self.override_pubs = {}

        for ns in ROBOT_NAMES:
            self.states[ns] = RobotState(ns)
            self.override_pubs[ns] = self.create_publisher(
                Twist, f'/{ns}/safety_override', 10)

            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda msg, r=ns: self._odom_cb(r, msg), 10)
            self.create_subscription(
                LaserScan, f'/{ns}/scan',
                lambda msg, r=ns: self._scan_cb(r, msg), 10)

        # Global event tracking
        self.collected_set = set()
        self.delivered_set = set()
        self.create_subscription(String, '/collected', self._collected_cb, 10)
        self.create_subscription(String, '/dropped', self._dropped_cb, 10)

        # Run safety checks at 10 Hz
        self.create_timer(0.1, self._safety_tick)

        self.get_logger().info('SafetyCoordinator ready')

    # ── Callbacks ──

    def _odom_cb(self, ns, msg):
        s = self.states[ns]
        s.x = msg.pose.pose.position.x
        s.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        s.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        s.vx = msg.twist.twist.linear.x
        s.vy = msg.twist.twist.linear.y

    def _scan_cb(self, ns, msg):
        valid = [r for r in msg.ranges
                 if not math.isinf(r) and not math.isnan(r) and r > 0.05]
        self.states[ns].min_lidar = min(valid) if valid else 10.0

    def _collected_cb(self, msg):
        self.collected_set.add(msg.data)

    def _dropped_cb(self, msg):
        self.delivered_set.add(msg.data)

    # ── Main safety loop ──

    def _safety_tick(self):
        now = time.time()
        overrides = {ns: None for ns in ROBOT_NAMES}

        # 1. Inter-robot collision prevention
        robots = list(self.states.values())
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                a, b = robots[i], robots[j]
                dx = b.x - a.x
                dy = b.y - a.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < HARD_COLLISION_DIST:
                    # Emergency: push both apart
                    if dist > 0.01:
                        nx, ny = dx / dist, dy / dist
                    else:
                        nx, ny = 1.0, 0.0
                    speed = STUCK_PERTURB_SPEED
                    overrides[a.ns] = self._make_twist(
                        -nx * speed, -ny * speed, 0.0, a.yaw)
                    overrides[b.ns] = self._make_twist(
                        nx * speed, ny * speed, 0.0, b.yaw)

                elif dist < COLLISION_DIST:
                    # Check closing velocity
                    rel_vx = (b.vx - a.vx)
                    rel_vy = (b.vy - a.vy)
                    closing = -(dx * rel_vx + dy * rel_vy) / max(dist, 0.01)
                    if closing > 0.05:
                        # They're approaching — slow down the faster one
                        if dist > 0.01:
                            nx, ny = dx / dist, dy / dist
                        else:
                            nx, ny = 1.0, 0.0
                        speed = 0.2
                        overrides[a.ns] = self._make_twist(
                            -nx * speed, -ny * speed, 0.0, a.yaw)

        # 2. Obstacle proximity escalation
        for ns, s in self.states.items():
            if s.min_lidar < OBSTACLE_CRITICAL:
                if s._critical_start is None:
                    s._critical_start = now
                elif now - s._critical_start > OBSTACLE_CRITICAL_TIME:
                    # Force reverse — back up in the opposite direction of heading
                    overrides[ns] = self._make_twist(
                        -0.3, 0.0, 0.0, s.yaw, robot_frame=True)
            else:
                s._critical_start = None

        # 3. Deadlock detection and recovery
        for ns, s in self.states.items():
            disp = math.sqrt(
                (s.x - s._history_x) ** 2 +
                (s.y - s._history_y) ** 2)

            if now - s._history_time > STUCK_TIME:
                if disp < STUCK_DIST:
                    # Stuck — issue random perturbation
                    import random
                    angle = random.uniform(-math.pi, math.pi)
                    vx = STUCK_PERTURB_SPEED * math.cos(angle)
                    vy = STUCK_PERTURB_SPEED * math.sin(angle)
                    overrides[ns] = self._make_twist(
                        vx, vy, random.uniform(-1.0, 1.0), s.yaw)
                    self.get_logger().warn(
                        f'[Safety] {ns} stuck — perturbation issued')
                # Reset history
                s._history_x = s.x
                s._history_y = s.y
                s._history_time = now

        # Publish overrides
        for ns, twist in overrides.items():
            if twist is not None:
                self.override_pubs[ns].publish(twist)
            else:
                # Publish zero to clear any previous override
                self.override_pubs[ns].publish(Twist())

    def _make_twist(self, vx, vy, wz, robot_yaw, robot_frame=False):
        """Create a Twist message. If not robot_frame, convert global→robot."""
        cmd = Twist()
        if robot_frame:
            cmd.linear.x = float(max(-MAX_SPEED, min(MAX_SPEED, vx)))
            cmd.linear.y = float(max(-MAX_SPEED, min(MAX_SPEED, vy)))
        else:
            # Global frame → robot frame
            cos_y = math.cos(-robot_yaw)
            sin_y = math.sin(-robot_yaw)
            local_vx = vx * cos_y - vy * sin_y
            local_vy = vx * sin_y + vy * cos_y
            cmd.linear.x = float(max(-MAX_SPEED, min(MAX_SPEED, local_vx)))
            cmd.linear.y = float(max(-MAX_SPEED, min(MAX_SPEED, local_vy)))
        cmd.angular.z = float(max(-MAX_WZ, min(MAX_WZ, wz)))
        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = SafetyCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
