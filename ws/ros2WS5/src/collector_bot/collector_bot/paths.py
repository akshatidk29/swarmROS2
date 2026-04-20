"""Centralized paths for the ROS2 collector_bot package."""
import os

_ROOT = os.path.expanduser('~/Desktop/ros2WS5')

MODEL_PATH   = os.path.join(_ROOT, 'RL', 'model', 'policy', 'policy.zip')
SIMULATE_DIR = os.path.join(_ROOT, 'simulate')