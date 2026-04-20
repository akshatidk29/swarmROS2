"""
Constants bridge — imports everything from RL/training/config.py.

This is the ONLY source of constants for all ROS2 nodes.
Never duplicate values in ROS2 code — always import from here.
"""

import sys
import os

# Add RL training path so we can import config.py directly
_rl_training = os.path.join(
    os.path.expanduser('~/Desktop/ros2WS5'), 'RL', 'training')
if _rl_training not in sys.path:
    sys.path.insert(0, _rl_training)

# Re-export everything
from config import *  # noqa: F401,F403
