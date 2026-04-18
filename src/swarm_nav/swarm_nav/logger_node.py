#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import json
import math
import os
import time

class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')
        
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        # Log structure
        self.log_data = {
            "start_time": self.start_time,
            "end_time": 0.0,
            "total_time": 0.0,
            "trajectories": {"robot_1": [], "robot_2": [], "robot_3": []}, # list of [time, x, y]
            "events": [], # list of {type, time, robot, color, x, y}
            "visited": [] # list of [time, x, y]
        }
        
        self.poses = {}
        
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        # Subs for each robot
        for bot in ['robot_1', 'robot_2', 'robot_3']:
            self.create_subscription(Odometry, f'/{bot}/odom', lambda msg, b=bot: self._cb_odom(msg, b), qos)
            self.create_subscription(LaserScan, f'/{bot}/scan', lambda msg, b=bot: self._cb_scan(msg, b), qos)
            
        # Subs for swarm events
        self.create_subscription(String, '/swarm/visited', self._cb_visited, 10)
        self.create_subscription(String, '/swarm/obj_locations', self._cb_discovered, 10)
        self.create_subscription(String, '/swarm/placed', self._cb_placed, 10)
        self.create_subscription(String, '/swarm/picked', self._cb_picked, 10)

        self.get_logger().info("LoggerNode started. Will save swarm_log.json on shutdown.")

    def _cb_odom(self, msg, bot):
        now = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        self.poses[bot] = (x, y, yaw)
        
        # Downsample trajectory logging to ~5Hz (every 0.2s)
        if len(self.log_data["trajectories"][bot]) == 0 or (now - self.log_data["trajectories"][bot][-1][0]) > 0.2:
            self.log_data["trajectories"][bot].append([round(now, 2), round(x, 2), round(y, 2)])

    def _cb_scan(self, msg, bot):
        now = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        if bot not in self.poses: return
        rx, ry, ryaw = self.poses[bot]
        
        # Chassis dimensions (0.3m x 0.2m)
        half_l = 0.151 # Small buffer to avoid self-hits
        half_w = 0.101
        
        for i, d in enumerate(msg.ranges):
            if math.isnan(d) or math.isinf(d): continue
            
            # Angle of beam relative to robot front
            a = msg.angle_min + i * msg.angle_increment
            cos_a = abs(math.cos(a))
            sin_a = abs(math.sin(a))
            
            # Distance to robot's own rectangular boundary at this angle
            if cos_a > 1e-3 and sin_a > 1e-3:
                r_self = min(half_l / cos_a, half_w / sin_a)
            elif cos_a <= 1e-3:
                r_self = half_w
            else:
                r_self = half_l
            
            # Skip if distance is within robot's own footprint (+ small margin)
            if d < (r_self + 0.01): continue

    def _cb_visited(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        parts = msg.data.split(',')
        if len(parts) == 2:
            self.log_data["visited"].append([round(now, 2), float(parts[0]), float(parts[1])])

    def _cb_picked(self, msg):
        self._log_event("picked", msg.data)

    def _cb_placed(self, msg):
        self._log_event("placed", msg.data)

    def _log_event(self, event_type, data):
        now = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        parts = data.split(',')
        if len(parts) == 4:
            color, bot, x, y = parts
            self.log_data["events"].append({
                "type": event_type,
                "time": round(now, 2),
                "robot": bot,
                "color": color,
                "x": float(x),
                "y": float(y)
            })
            self.get_logger().info(f"Logged {event_type} event: {bot} -> {color}")

    def _cb_discovered(self, msg):
        parts = msg.data.split(',')
        if len(parts) == 4:
            cid, x, y, bot = parts[0], parts[1], parts[2], parts[3]
            colors = {"1": "red", "2": "green", "3": "blue"}
            color = colors.get(cid, "unknown")
            already_discovered = any(e["type"] == "discovered" and e["color"] == color for e in self.log_data["events"])
            if not already_discovered:
                now = self.get_clock().now().nanoseconds / 1e9
                self.log_data["events"].append({
                    "type": "discovered", "time": round(now - self.start_time, 2),
                    "robot": bot, "color": color, "x": float(x), "y": float(y)
                })
                self.get_logger().info(f"Collaboration! {bot} discovered {color} object at ({x}, {y})")

    def save_log(self):
        end_time = self.get_clock().now().nanoseconds / 1e9
        self.log_data["end_time"] = end_time
        self.log_data["total_time"] = round(end_time - self.start_time, 2)
        
        # Save to swarm_ws root
        ws_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'silver_quick', 'cs671_2026_hack', 'swarm_ws')
        
        # Add timestamp to filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(ws_dir, f'swarm_log_{timestamp}.json')
        
        with open(log_path, 'w') as f:
            json.dump(self.log_data, f, indent=2)
            
        # Use standard print to avoid ROS logging context issues during Ctrl+C teardown
        print(f"\n[LoggerNode] Successfully saved simulation log to {log_path} (Time: {self.log_data['total_time']}s)\n")

def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_log()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
