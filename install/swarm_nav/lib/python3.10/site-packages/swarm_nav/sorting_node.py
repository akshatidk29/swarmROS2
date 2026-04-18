#!/usr/bin/env python3
import os
import math
import pickle
import random
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import DeleteEntity
from std_msgs.msg import String

from swarm_nav.camera_processor import CameraProcessor

def norm_angle(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

COLOR_TO_OBJ = {1: 'sort_obj_red', 2: 'sort_obj_green', 3: 'sort_obj_blue'}

class SortingNode(Node):
    def __init__(self):
        super().__init__('sorting_node')
        self.declare_parameter('robot_name', 'robot_1')
        self.name = self.get_parameter('robot_name').value
        self.priority = int(self.name.split('_')[-1]) if '_' in self.name else 99
        self.other_robots = {}

        # --- Q-Table Load ---
        self.q_table = {}
        q_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q_table.pkl')
        if os.path.exists(q_path):
            with open(q_path, 'rb') as f:
                self.q_table = pickle.load(f)
            self.get_logger().info(f"[{self.name}] Loaded Q-table with {len(self.q_table)} states.")
        else:
            self.get_logger().error(f"[{self.name}] Q-table not found at {q_path}!")

        # --- Subsystems ---
        self.camera = CameraProcessor(self.get_logger().info)
        self.obj_detections = {}
        self.bin_detections = {}

        # --- State ---
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.carrying = 0
        self.picked_obj_name = ''
        self.last_scan = None
        self.task_done = False
        
        # Shared Map
        self.visited_grid = set()
        self.global_placed = {'red': False, 'green': False, 'blue': False}
        self.global_picked = {'red': False, 'green': False, 'blue': False}

        # Dwell State
        self.dwell_active = False
        self.dwell_start_time = 0.0

        self.nav_mode = "NORMAL"
        self.nav_mode_start = 0.0
        self.wf_dir = 2

        self.last_turn_dir = 1
        self.last_action = 0
        self.last_action_reason = ""

        # ROS Setup
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(LaserScan, 'scan', self._cb_scan, qos)
        self.create_subscription(Odometry, 'odom', self._cb_odom, qos)
        self.create_subscription(Image, 'camera/image_raw', self._cb_cam, qos)
        self.create_subscription(Image, 'camera/depth/image_raw', self._cb_depth, qos)
        
        # Shared communication
        self.create_subscription(String, '/swarm/visited', self._cb_visited, 10)
        self.create_subscription(String, '/swarm/placed', self._cb_placed, 10)
        self.create_subscription(String, '/swarm/picked', self._cb_picked, 10)
        self.create_subscription(String, '/swarm/bin_locations', self._cb_bin_loc, 10)
        
        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub_visited = self.create_publisher(String, '/swarm/visited', 10)
        self.pub_placed = self.create_publisher(String, '/swarm/placed', 10)
        self.pub_picked = self.create_publisher(String, '/swarm/picked', 10)
        self.pub_bin_loc = self.create_publisher(String, '/swarm/bin_locations', 10)
        
        self.shared_bins = {}
        
        self.create_subscription(PoseStamped, '/swarm/poses', self._cb_swarm, 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/swarm/poses', 10)

        # Service clients
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')
        
        self.last_depth_msg = None
        self.dt = 0.1
        self.create_timer(self.dt, self._loop)
        self.create_timer(1.0, self._status_log)

    def _cb_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        gx = round(self.x * 2) / 2
        gy = round(self.y * 2) / 2
        if (gx, gy) not in self.visited_grid:
            self.visited_grid.add((gx, gy))
            msg = String()
            msg.data = f"{gx},{gy}"
            self.pub_visited.publish(msg)
        self._broadcast_pose()

    def _cb_scan(self, msg):
        self.last_scan = msg
        
    def _cb_cam(self, msg):
        res = self.camera.process_image(msg, self.last_depth_msg)
        if len(res) == 2:
            self.obj_detections, self.bin_detections = res
        
    def _cb_depth(self, msg):
        self.last_depth_msg = msg

    def _cb_visited(self, msg):
        parts = msg.data.split(',')
        if len(parts) == 2:
            self.visited_grid.add((float(parts[0]), float(parts[1])))

    def _cb_placed(self, msg):
        color = msg.data
        if color in self.global_placed:
            self.global_placed[color] = True

    def _cb_picked(self, msg):
        color = msg.data
        if color in self.global_picked:
            self.global_picked[color] = True

    def _cb_bin_loc(self, msg):
        parts = msg.data.split(',')
        if len(parts) == 3:
            cid = int(parts[0])
            self.shared_bins[cid] = (float(parts[1]), float(parts[2]))

    def _cb_swarm(self, msg):
        if msg.header.frame_id != self.name:
            self.other_robots[msg.header.frame_id] = (msg.pose.position.x, msg.pose.position.y)

    def _broadcast_pose(self):
        m = PoseStamped()
        m.header.frame_id = self.name
        m.pose.position.x = self.x
        m.pose.position.y = self.y
        self.pub_pose.publish(m)

    def _status_log(self):
        if self.task_done: return
        self.get_logger().info(f"[{self.name}] pos=({self.x:.1f},{self.y:.1f}) carry={self.carrying} placed={list(self.global_placed.values())}")

    def _delete_obj(self, name):
        req = DeleteEntity.Request()
        req.name = name
        self.delete_cli.call_async(req)

    def _lidar_wall_front(self):
        if not self.last_scan: return 0
        cone = math.radians(40)
        left_points = 0
        right_points = 0
        for i, d in enumerate(self.last_scan.ranges):
            if d < 0.12 or math.isnan(d) or math.isinf(d): continue
            a = norm_angle(self.last_scan.angle_min + i*self.last_scan.angle_increment)
            if -cone <= a <= cone and d < 0.45:
                if a > 0: left_points += 1
                else: right_points += 1
        if left_points == 0 and right_points == 0: return 0
        
        if "Obstacle" in self.last_action_reason and self.last_action in [1, 2]:
            return self.last_action
            
        return 1 if right_points >= left_points else 2

    def _lidar_wall_side(self, check_left=True):
        if not self.last_scan: return False
        min_angle = math.radians(40)
        max_angle = math.radians(110)
        count = 0
        for i, d in enumerate(self.last_scan.ranges):
            if d < 0.12 or math.isnan(d) or math.isinf(d): continue
            a = norm_angle(self.last_scan.angle_min + i*self.last_scan.angle_increment)
            if check_left and min_angle <= a <= max_angle and d < 0.6: count += 1
            elif not check_left and -max_angle <= a <= -min_angle and d < 0.6: count += 1
        return count > 1

    def _visited_ahead(self):
        lookahead = 0.6
        tx = self.x + math.cos(self.yaw) * lookahead
        ty = self.y + math.sin(self.yaw) * lookahead
        gx = round(tx * 2) / 2
        gy = round(ty * 2) / 2
        return 1 if (gx, gy) in self.visited_grid else 0

    def _get_camera_state(self):
        # target_type: 0=none, 1=want_obj, 2=want_bin
        tt = 0; td = 0; is_close = False; best_cid = 0
        
        colors = {1: 'red', 2: 'green', 3: 'blue'}
        
        if self.carrying == 0:
            best_area = 0
            for cid in [1, 2, 3]:
                if not self.global_picked[colors[cid]] and not self.global_placed[colors[cid]] and cid in self.obj_detections:
                    cur_td, cur_is_close, cur_area, cur_dist = self.obj_detections[cid]
                    if cur_area > best_area:
                        tt = 1
                        td = cur_td
                        is_close = cur_is_close
                        best_cid = cid
                        best_area = cur_area
        else:
            cid = self.carrying
            if cid in self.bin_detections:
                tt = 2
                td, is_close, _, _ = self.bin_detections[cid]
                best_cid = cid
            elif cid in self.shared_bins:
                tt = 3
                gx, gy = self.shared_bins[cid]
                angle_to_bin = math.atan2(gy - self.y, gx - self.x)
                angle_diff = norm_angle(angle_to_bin - self.yaw)
                if angle_diff > 0.3: td = 1
                elif angle_diff < -0.3: td = 2
                else: td = 0
                is_close = False
                best_cid = cid
                
        return tt, td, is_close, best_cid

    def _loop(self):
        cmd = Twist()
        now = self.get_clock().now().nanoseconds / 1e9
        

        if all(self.global_placed.values()):
            if not self.task_done:
                self.get_logger().info(f"[{self.name}] All items sorted! Stopping.")
                self.task_done = True
            self.pub_cmd.publish(cmd)
            return

        # DWELL Logic
        if self.dwell_active:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.dwell_start_time >= 5.0:
                colors = {1: 'red', 2: 'green', 3: 'blue'}
                color_name = colors[self.carrying]
                self.get_logger().info(f"[{self.name}] Dropped {color_name} object into bin.")
                
                # Signal global placed
                msg = String()
                msg.data = color_name
                self.pub_placed.publish(msg)
                self.global_placed[color_name] = True
                
                self.carrying = 0
                self.dwell_active = False
            self.pub_cmd.publish(cmd)
            return

        # Broadcast bins we currently see
        for cid, (td, is_close, area, dist) in self.bin_detections.items():
            if dist < 9.0:
                angle_offset = 0.0
                if td == 1: angle_offset = 0.4
                elif td == 2: angle_offset = -0.4
                bin_x = self.x + dist * math.cos(self.yaw + angle_offset)
                bin_y = self.y + dist * math.sin(self.yaw + angle_offset)
                if cid not in self.shared_bins:
                    self.shared_bins[cid] = (bin_x, bin_y)
                    msg = String()
                    msg.data = f"{cid},{bin_x:.2f},{bin_y:.2f}"
                    self.pub_bin_loc.publish(msg)

        tt, td, is_close, cid = self._get_camera_state()
        wf_raw = self._lidar_wall_front()
        
        # State transitions
        if self.nav_mode == "EXPLORE":
            if now - self.nav_mode_start > 30.0:
                self.get_logger().info(f"[{self.name}] Finished EXPLORE override. Resuming NORMAL.")
                self.nav_mode = "NORMAL"
            else:
                if tt == 3: tt = 0 # Ignore shared map target during exploration
                
        if self.nav_mode == "NORMAL" and tt == 3 and wf_raw > 0:
            self.get_logger().info(f"[{self.name}] Obstacle blocks shared target. Starting WALL_FOLLOW.")
            self.nav_mode = "WALL_FOLLOW"
            self.nav_mode_start = now
            self.wf_dir = 2 # Initial turn right (wall on left)
            
        if self.nav_mode == "WALL_FOLLOW":
            if tt != 3:
                self.nav_mode = "NORMAL"
            elif now - self.nav_mode_start > 15.0:
                self.get_logger().info(f"[{self.name}] Stuck in Wall Follow. Switching to EXPLORE for 30s.")
                self.nav_mode = "EXPLORE"
                self.nav_mode_start = now
                tt = 0
            else:
                side_blocked = self._lidar_wall_side(check_left=(self.wf_dir == 2))
                if not side_blocked and wf_raw == 0:
                    self.get_logger().info(f"[{self.name}] Obstacle cleared. Resuming DIRECT to target.")
                    self.nav_mode = "NORMAL"

        # Pickup / Dropoff Trigger
        if is_close:
            colors = {1: 'red', 2: 'green', 3: 'blue'}
            if self.carrying == 0 and tt == 1:
                self.get_logger().info(f"[{self.name}] robot has picked the object and the colour of that object is {colors[cid]}")
                self.carrying = cid
                
                # Signal global picked
                msg = String()
                msg.data = colors[cid]
                self.pub_picked.publish(msg)
                self.global_picked[colors[cid]] = True
                
                self._delete_obj(COLOR_TO_OBJ[cid])
                self.pub_cmd.publish(cmd)
                return
            elif self.carrying != 0 and tt == 2:
                self.get_logger().info(f"[{self.name}] Reached {colors[cid]} bin! Dwelling for 5s...")
                self.dwell_active = True
                self.dwell_start_time = self.get_clock().now().nanoseconds / 1e9
                self.pub_cmd.publish(cmd)
                return

        # RL Execution
        wf = 1 if wf_raw > 0 else 0
        va = self._visited_ahead()
        
        state = (self.carrying, tt, td, wf, va)
        action = 0 # 0=FWD, 1=LEFT, 2=RIGHT
        reason = ""
        
        if state in self.q_table:
            import numpy as np
            action = int(np.argmax(self.q_table[state]))
            reason = "Q-table"
        else:
            # Fallback robust rules if RL state not seen
            if tt in [1, 2]: # Camera overrides
                if td == 1: action = 1; reason = "Camera fallback (turn left)"
                elif td == 2: action = 2; reason = "Camera fallback (turn right)"
                else: action = 0; reason = "Camera fallback (forward)"
            elif wf_raw > 0: 
                action = wf_raw # Obstacle avoidance
                reason = "Obstacle avoidance (turn " + ("left" if action==1 else "right") + ")"
            elif tt == 3:
                if td == 1: action = 1; reason = "Shared map fallback (turn left)"
                elif td == 2: action = 2; reason = "Shared map fallback (turn right)"
                else: action = 0; reason = "Shared map fallback (forward)"
            else: 
                action = 0; reason = "Forward (no obstacle)"

        # Enforce camera precedence constraint explicitly to ensure safety
        if tt in [1, 2]:
            if td == 1: action = 1; reason = "Camera precedence (turn left)"
            elif td == 2: action = 2; reason = "Camera precedence (turn right)"
            else: action = 0; reason = "Camera precedence (forward)"
        elif self.nav_mode == "WALL_FOLLOW":
            if wf_raw > 0:
                other_blocked = self._lidar_wall_side(check_left=(self.wf_dir != 2))
                if other_blocked:
                    self.get_logger().info(f"[{self.name}] Cornered! Reversing wall follow direction.")
                    self.wf_dir = 1 if self.wf_dir == 2 else 2
                action = self.wf_dir
                reason = "Wall Follow (turning " + ("left" if action==1 else "right") + ")"
            else:
                action = 0
                reason = "Wall Follow (forward)"
        elif wf_raw > 0:
            action = wf_raw # Override Q-table to prevent oscillation near walls
            reason = "Obstacle precedence (turn " + ("left" if action==1 else "right") + ")"
        elif tt == 3:
            if td == 1: action = 1; reason = "Shared map precedence (turn left)"
            elif td == 2: action = 2; reason = "Shared map precedence (turn right)"
            else: action = 0; reason = "Shared map precedence (forward)"

        if action in [1, 2]:
            self.last_turn_dir = action
            
        if action != self.last_action or reason != self.last_action_reason:
            if action == 1:
                self.get_logger().info(f"[{self.name}] Turning LEFT. Reason: {reason}")
            elif action == 2:
                self.get_logger().info(f"[{self.name}] Turning RIGHT. Reason: {reason}")
            self.last_action = action
            self.last_action_reason = reason

        # Execute action
        if action == 0:
            cmd.linear.x = 0.3
        elif action == 1:
            cmd.angular.z = 0.8
        elif action == 2:
            cmd.angular.z = -0.8

        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SortingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
