"""
Gazebo environment wrapper — wraps a running Gazebo sim as a Gymnasium env.

Reads from ROS2 topics (LiDAR, odom, camera) and writes to cmd_vel.
Builds the same 85-dim observation vector as the training env and swarm_brain.

Usage:
    # Start Gazebo sim first, then:
    env = GazeboEnvWrapper(robot_ns='robot_1')
    obs, info = env.reset()
    action = policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
"""

import math
import time
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from config import (
    ARENA_HALF, ARENA_SIZE, N_ROBOTS,
    MAX_SPEED, MAX_WZ,
    LIDAR_RAYS, LIDAR_MAX_DIST,
    CAMERA_FOV, CAMERA_RANGE,
    N_DETECTIONS, N_SHARED_OBJ, CARRY_TIMEOUT_STEPS,
    OCC_PATCH_SIZE, OBS_DIM, ACT_DIM, ACTION_EMA,
    CUBE_BASKET, SPHERE_BASKET,
    GRID_RES, GRID_NX, GRID_NY, MAX_STEPS,
)

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

try:
    from cv_bridge import CvBridge
    HAS_CV = True
except ImportError:
    HAS_CV = False


class _ROSBridge(Node):
    """Internal ROS2 node that subscribes/publishes for a single robot."""

    def __init__(self, robot_ns):
        super().__init__(f'gazebo_env_{robot_ns}')
        self.ns = robot_ns
        pfx = f'/{robot_ns}'

        self.x = self.y = self.yaw = 0.0
        self.vx = self.vy = 0.0
        self.scan_ranges = []
        self.carrying = None
        self._carry_steps = 0
        self.other_poses = {}
        self.known_objects = {}
        self.collected_set = set()
        self.visited = set()
        self.target = None
        self.tracks = []
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)

        if HAS_CV:
            self.bridge = CvBridge()

        # Subs
        self.create_subscription(LaserScan, f'{pfx}/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, f'{pfx}/odom', self._odom_cb, 10)
        self.create_subscription(String, '/robot_poses', self._poses_cb, 10)
        self.create_subscription(String, '/known_objects', self._known_cb, 10)
        self.create_subscription(String, '/collected', self._collected_cb, 10)
        self.create_subscription(String, '/dropped', self._dropped_cb, 10)
        self.create_subscription(String, '/visited_zones', self._visited_cb, 10)

        # Pub
        self.cmd_pub = self.create_publisher(Twist, f'{pfx}/cmd_vel', 10)

    def _scan_cb(self, msg):
        self.scan_ranges = list(msg.ranges)

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        # Mark visited grid cell
        i = max(0, min(GRID_NX - 1, int((self.x + ARENA_HALF) / GRID_RES)))
        j = max(0, min(GRID_NY - 1, int((self.y + ARENA_HALF) / GRID_RES)))
        self.visited.add((i, j))

    def _poses_cb(self, msg):
        parts = msg.data.split(':')
        if len(parts) == 3 and parts[0] != self.ns:
            try:
                self.other_poses[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass

    def _known_cb(self, msg):
        for item in msg.data.split(','):
            parts = item.split(':')
            if len(parts) == 4:
                try:
                    key = f'{parts[1]}_{parts[2][:4]}'
                    self.known_objects[key] = (
                        float(parts[2]), float(parts[3]), parts[1], parts[0])
                except ValueError:
                    pass

    def _collected_cb(self, msg):
        self.collected_set.add(msg.data)

    def _dropped_cb(self, msg):
        pass

    def _visited_cb(self, msg):
        for part in msg.data.split(','):
            if ':' in part:
                try:
                    a, b = part.split(':')
                    self.visited.add((int(a), int(b)))
                except ValueError:
                    pass

    def pub_cmd(self, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x = float(max(-MAX_SPEED, min(MAX_SPEED, vx)))
        cmd.linear.y = float(max(-MAX_SPEED, min(MAX_SPEED, vy)))
        cmd.angular.z = float(max(-MAX_WZ, min(MAX_WZ, wz)))
        self.cmd_pub.publish(cmd)


class GazeboEnvWrapper(gym.Env):
    """Gymnasium wrapper for a single robot in a running Gazebo simulation.

    Parameters
    ----------
    robot_ns : str
        ROS2 namespace of the robot (e.g. 'robot_1').
    step_duration : float
        Seconds per step (controls action duration).
    max_steps : int
        Maximum steps per episode.
    """

    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, robot_ns='robot_1', step_duration=0.1,
                 max_steps=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(-1, 1, shape=(OBS_DIM,),
                                           dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(ACT_DIM,),
                                       dtype=np.float32)
        self.step_duration = step_duration
        self.max_steps = max_steps or MAX_STEPS
        self._step_count = 0

        # Init ROS2
        if not rclpy.ok():
            rclpy.init()

        self._node = _ROSBridge(robot_ns)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        # Spin in background thread
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Wait for data
        time.sleep(1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._node.prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._node.pub_cmd(0, 0, 0)
        time.sleep(0.5)
        return self._build_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)

        # Action EMA smoothing (matching training env)
        smoothed = ((1.0 - ACTION_EMA) * action +
                    ACTION_EMA * self._node.prev_action)
        self._node.prev_action = action.copy()

        vx = float(smoothed[0]) * MAX_SPEED
        vy = float(smoothed[1]) * MAX_SPEED
        wz = float(smoothed[2]) * MAX_WZ

        self._node.pub_cmd(vx, vy, wz)
        time.sleep(self.step_duration)

        # Spin to get latest data
        self._executor.spin_once(timeout_sec=0.01)

        obs = self._build_obs()
        self._step_count += 1

        reward = -0.01  # step cost
        terminated = False
        truncated = self._step_count >= self.max_steps
        info = {'step': self._step_count}

        return obs, reward, terminated, truncated, info

    def _build_obs(self):
        """Build 85-dim obs vector matching training env exactly."""
        r = self._node
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        # [0:18] LiDAR — ray 0 = forward, CCW
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

        # [18:43] Occupancy grid ego-patch from LiDAR (same as swarm_brain)
        patch = np.full(OCC_PATCH_SIZE ** 2, -1.0, dtype=np.float32)
        if n > 0:
            half = OCC_PATCH_SIZE // 2
            for i_r in range(LIDAR_RAYS):
                deg = i_r * (360.0 / LIDAR_RAYS)
                li = int(round(((deg + 180) % 360) * n / 360)) % n
                val = r.scan_ranges[li]
                if math.isinf(val) or math.isnan(val):
                    val = LIDAR_MAX_DIST
                dist = min(val, LIDAR_MAX_DIST)
                angle = (i_r / LIDAR_RAYS) * 2 * math.pi
                for step_d in np.arange(0.25, dist, 0.25):
                    ci = half + int(step_d * math.cos(angle) / 0.25)
                    cj = half + int(step_d * math.sin(angle) / 0.25)
                    if 0 <= ci < OCC_PATCH_SIZE and 0 <= cj < OCC_PATCH_SIZE:
                        patch[ci * OCC_PATCH_SIZE + cj] = 0.0
                if dist < LIDAR_MAX_DIST - 0.2:
                    ci = half + int(dist * math.cos(angle) / 0.25)
                    cj = half + int(dist * math.sin(angle) / 0.25)
                    if 0 <= ci < OCC_PATCH_SIZE and 0 <= cj < OCC_PATCH_SIZE:
                        patch[ci * OCC_PATCH_SIZE + cj] = 1.0
        obs[idx:idx + OCC_PATCH_SIZE**2] = patch
        idx += OCC_PATCH_SIZE ** 2

        # [43:52] Other robots
        others = list(r.other_poses.values())[:3]
        for i in range(3):
            if i < len(others):
                ox, oy = others[i]
                dx, dy = ox - r.x, oy - r.y
                d = math.sqrt(dx * dx + dy * dy)
                cos_y = math.cos(-r.yaw)
                sin_y = math.sin(-r.yaw)
                rel_x = dx * cos_y - dy * sin_y
                rel_y = dx * sin_y + dy * cos_y
                obs[idx]     = np.clip(rel_x / ARENA_SIZE, -1, 1)
                obs[idx + 1] = np.clip(rel_y / ARENA_SIZE, -1, 1)
                obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
            idx += 3

        # [52:55] Target
        if r.target is not None:
            tx, ty = r.target
            dx, dy = tx - r.x, ty - r.y
            d = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx) - r.yaw
            obs[idx]     = math.sin(angle)
            obs[idx + 1] = math.cos(angle)
            obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
        idx += 3

        # [55:60] Self
        c_val = 0
        if r.carrying == 'cube':
            c_val = 1
        elif r.carrying == 'sphere':
            c_val = 2
        obs[idx]     = c_val / 2.0
        obs[idx + 1] = np.clip(r.vx / MAX_SPEED, -1, 1)
        obs[idx + 2] = np.clip(r.vy / MAX_SPEED, -1, 1)
        obs[idx + 3] = math.sin(r.yaw)
        obs[idx + 4] = math.cos(r.yaw)
        idx += 5

        # [60:69] Camera detections (zeros — camera not processed here)
        idx += N_DETECTIONS * 3

        # [69:72] Basket
        if r.carrying == 'cube':
            bx, by = float(CUBE_BASKET[0]), float(CUBE_BASKET[1])
        elif r.carrying == 'sphere':
            bx, by = float(SPHERE_BASKET[0]), float(SPHERE_BASKET[1])
        else:
            bx, by = float(CUBE_BASKET[0]), float(CUBE_BASKET[1])
        dx, dy = bx - r.x, by - r.y
        d = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx) - r.yaw
        obs[idx]     = math.sin(angle)
        obs[idx + 1] = math.cos(angle)
        obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
        idx += 3

        # [72:81] Known objects
        known = []
        for key, (kx, ky, ktype, _) in r.known_objects.items():
            kd = math.sqrt((r.x - kx)**2 + (r.y - ky)**2)
            known.append((ktype, kx, ky, kd))
        known.sort(key=lambda x: x[3])
        for i in range(N_SHARED_OBJ):
            if i < len(known):
                ktype, kx, ky, _ = known[i]
                obs[idx]     = 1.0 if ktype == 'cube' else -1.0
                obs[idx + 1] = np.clip((kx - r.x) / ARENA_SIZE, -1, 1)
                obs[idx + 2] = np.clip((ky - r.y) / ARENA_SIZE, -1, 1)
            idx += 3

        # [81:84] Zone info
        total_cells = GRID_NX * GRID_NY
        obs[idx] = len(r.visited) / max(total_cells, 1)

        nearby_unvisited = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ci = max(0, min(GRID_NX - 1,
                         int((r.x + ARENA_HALF) / GRID_RES) + di))
                cj = max(0, min(GRID_NY - 1,
                         int((r.y + ARENA_HALF) / GRID_RES) + dj))
                if (ci, cj) not in r.visited:
                    nearby_unvisited += 1
        obs[idx + 1] = nearby_unvisited / 9.0

        nearest_d = ARENA_SIZE
        for ic in range(GRID_NX):
            for jc in range(GRID_NY):
                if (ic, jc) in r.visited:
                    continue
                cx = (ic + 0.5) * GRID_RES - ARENA_HALF
                cy = (jc + 0.5) * GRID_RES - ARENA_HALF
                dd = math.sqrt((r.x - cx)**2 + (r.y - cy)**2)
                if dd < nearest_d:
                    nearest_d = dd
        obs[idx + 2] = np.clip(nearest_d / ARENA_SIZE, 0, 1)
        idx += 3

        # [84] Carry overtime
        if r.carrying is not None:
            obs[idx] = min(r._carry_steps / CARRY_TIMEOUT_STEPS, 1.0)
        else:
            obs[idx] = 0.0
        idx += 1

        return obs

    def close(self):
        self._node.pub_cmd(0, 0, 0)
        self._executor.shutdown()
        self._node.destroy_node()

    def render(self):
        pass  # Gazebo handles rendering
