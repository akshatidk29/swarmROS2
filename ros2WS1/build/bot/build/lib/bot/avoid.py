import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import random


class ObstacleAvoidance(Node):
    """
    Obstacle avoidance node using 5-region LiDAR scan analysis.

    Divides the 360° LiDAR scan into five regions:
      - front:       ±15° ahead
      - front_left:  15°–60° left
      - front_right: 15°–60° right (i.e. 300°–345°)
      - left:        60°–90° left
      - right:       60°–90° right (270°–300°)

    Behaviour:
      1. If front is clear  → drive forward with slight random wander
      2. If front blocked   → turn toward the clearer side
      3. Speed scales down proportionally as obstacles get closer
    """

    # Tunables
    SAFE_DIST = 1.0          # metres – start reacting early
    CLOSE_DIST = 0.4         # metres – sharp swerve (but still moving!)
    MAX_LINEAR = 0.30        # m/s – cruising speed
    MIN_LINEAR = 0.12        # m/s – minimum forward speed (never stop!)
    MAX_ANGULAR = 1.5        # rad/s – sharp turn speed
    WANDER_AMPLITUDE = 0.2   # rad/s – gentle wander when path clear

    def __init__(self):
        super().__init__('obstacle_avoidance')

        # Declare & read the namespace parameter
        self.declare_parameter('robot_ns', '')
        ns = self.get_parameter('robot_ns').get_parameter_value().string_value

        # Build topic names
        scan_topic = f'/{ns}/scan' if ns else '/scan'
        cmd_topic = f'/{ns}/cmd_vel' if ns else '/cmd_vel'

        self.get_logger().info(
            f'Avoidance node started  scan={scan_topic}  cmd={cmd_topic}')

        self.sub = self.create_subscription(
            LaserScan, scan_topic, self._scan_cb, 10)
        self.pub = self.create_publisher(Twist, cmd_topic, 10)

        # Internal state
        self._wander_timer = self.create_timer(2.0, self._randomise_wander)
        self._wander_bias = 0.0

    # ------------------------------------------------------------------
    def _randomise_wander(self):
        """Periodically pick a small random angular bias for exploration."""
        self._wander_bias = random.uniform(
            -self.WANDER_AMPLITUDE, self.WANDER_AMPLITUDE)

    # ------------------------------------------------------------------
    @staticmethod
    def _region_min(ranges, start_idx, end_idx, num_samples):
        """Return minimum valid range in [start_idx, end_idx) wrapping."""
        vals = []
        for i in range(start_idx, end_idx):
            idx = i % num_samples
            r = ranges[idx]
            if not math.isinf(r) and not math.isnan(r) and r > 0.05:
                vals.append(r)
        return min(vals) if vals else 10.0   # 10 m if nothing valid

    # ------------------------------------------------------------------
    def _scan_cb(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return

        # Angular width per sample
        # Typical 360-sample scan → 1°/sample
        deg_per_sample = 360.0 / n

        def deg_to_idx(deg):
            return int(round(deg / deg_per_sample)) % n

        # Regions (degrees, measured from front=0, CCW positive)
        # Front wraps around 0°: use ranges from 345° to 360° + 0° to 15°
        front_samples = (
            [msg.ranges[i % n] for i in range(deg_to_idx(345), n)] +
            [msg.ranges[i] for i in range(0, deg_to_idx(15) + 1)]
        )
        front_valid = [r for r in front_samples
                       if not math.isinf(r) and not math.isnan(r) and r > 0.05]
        front = min(front_valid) if front_valid else 10.0
        f_left = self._region_min(msg.ranges, deg_to_idx(15), deg_to_idx(60), n)
        f_right = self._region_min(msg.ranges, deg_to_idx(300), deg_to_idx(345), n)
        left = self._region_min(msg.ranges, deg_to_idx(60), deg_to_idx(90), n)
        right = self._region_min(msg.ranges, deg_to_idx(270), deg_to_idx(300), n)

        cmd = Twist()

        # Check if rear is clear (for reverse manoeuvres)
        rear = self._region_min(msg.ranges, deg_to_idx(160), deg_to_idx(200), n)

        # ---- Decision logic (always keep moving!) ----

        # PRIORITY 1: Boxed in — front AND both sides blocked → REVERSE
        if (front < self.CLOSE_DIST and
                f_left < self.CLOSE_DIST and f_right < self.CLOSE_DIST):
            # Cornered! Back up while spinning to find an opening
            cmd.linear.x = -self.MIN_LINEAR if rear > self.CLOSE_DIST else 0.0
            cmd.angular.z = self.MAX_ANGULAR  # spin to find space
            state = 'REVERSE_SPIN'

        # PRIORITY 2: Very close in front → reverse-swerve or sharp swerve
        elif front < self.CLOSE_DIST:
            if rear > self.CLOSE_DIST:
                # Rear is clear → back up while turning toward open side
                cmd.linear.x = -self.MIN_LINEAR
                if f_left < f_right or (f_left == f_right and left < right):
                    cmd.angular.z = -self.MAX_ANGULAR   # turn right
                else:
                    cmd.angular.z = self.MAX_ANGULAR    # turn left
                state = 'REVERSE_SWERVE'
            else:
                # Rear also blocked → just swerve forward slowly
                cmd.linear.x = self.MIN_LINEAR
                if f_left < f_right or (f_left == f_right and left < right):
                    cmd.angular.z = -self.MAX_ANGULAR
                else:
                    cmd.angular.z = self.MAX_ANGULAR
                state = 'SHARP_SWERVE'

        # PRIORITY 3: Obstacle approaching → smooth curve away
        elif front < self.SAFE_DIST:
            ratio = (front - self.CLOSE_DIST) / (self.SAFE_DIST - self.CLOSE_DIST)
            cmd.linear.x = self.MIN_LINEAR + (self.MAX_LINEAR - self.MIN_LINEAR) * ratio

            turn_strength = self.MAX_ANGULAR * (1.0 - ratio)
            if f_left < f_right:
                cmd.angular.z = -turn_strength    # curve right
            else:
                cmd.angular.z = turn_strength     # curve left
            state = 'CURVE_AWAY'

        # PRIORITY 4: Side obstacles → nudge away
        elif f_left < self.SAFE_DIST or f_right < self.SAFE_DIST:
            cmd.linear.x = self.MAX_LINEAR * 0.8
            if f_left < f_right:
                cmd.angular.z = -0.5   # nudge right
            else:
                cmd.angular.z = 0.5    # nudge left
            state = 'SIDE_NUDGE'

        # PRIORITY 5: Path clear → cruise with wander
        else:
            cmd.linear.x = self.MAX_LINEAR
            cmd.angular.z = self._wander_bias
            state = 'CRUISE'

        self.pub.publish(cmd)
        self.get_logger().info(
            f'{state}  front={front:.2f} fl={f_left:.2f} fr={f_right:.2f} '
            f'rear={rear:.2f} v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down obstacle avoidance…')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()