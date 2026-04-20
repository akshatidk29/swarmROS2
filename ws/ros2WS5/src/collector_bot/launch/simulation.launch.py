"""Launch: Gazebo arena + 4 RL-controlled robots + safety + sim_logger.

Supports randomized spawn positions via 'randomize:=true' launch arg.
Supports headless mode via 'headless:=true' launch arg.
"""

import os
import sys
import random
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Import constants from config.py via path manipulation
_rl_training = os.path.join(
    os.path.expanduser('~/Desktop/ros2WS5'), 'RL', 'training')
if _rl_training not in sys.path:
    sys.path.insert(0, _rl_training)
from config import (
    ARENA_HALF, ROBOT_RADIUS, N_ROBOTS, OBSTACLES,
    SPAWN_CLUSTER_SPREAD,
)

# Default (deterministic) positions — fallback only
DEFAULT_ROBOTS = [
    ('robot_1',  0.0, -2.0),
    ('robot_2',  2.0, -2.0),
    ('robot_3', -2.0, -2.0),
    ('robot_4',  0.0, -4.0),
]


def _point_near_obstacle(px, py, margin):
    """Check if point is within margin of any obstacle."""
    for ox, oy, hw, hh in OBSTACLES:
        if abs(px - ox) < hw + margin and abs(py - oy) < hh + margin:
            return True
    return False


def _randomize_positions(n=4, margin=1.5):
    """Generate cross-pattern spawn positions around a random centre."""
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    spread = SPAWN_CLUSTER_SPREAD
    edge = spread + ROBOT_RADIUS + 0.5

    for _ in range(200):
        cx = random.uniform(-ARENA_HALF + edge + 0.15,
                            ARENA_HALF - edge - 0.15)
        cy = random.uniform(-ARENA_HALF + edge + 0.15, 0.0)

        positions = []
        all_ok = True
        for dx, dy in offsets:
            px = cx + dx * spread
            py = cy + dy * spread
            if abs(px) > ARENA_HALF - 0.5 or abs(py) > ARENA_HALF - 0.5:
                all_ok = False
                break
            if _point_near_obstacle(px, py, ROBOT_RADIUS + 0.3):
                all_ok = False
                break
            positions.append((px, py))

        if all_ok and len(positions) == n:
            return positions

    # Fallback
    return [(x, y) for _, x, y in DEFAULT_ROBOTS[:n]]


def generate_launch_description():
    pkg = get_package_share_directory('collector_bot')
    world = os.path.join(pkg, 'worlds', 'arena.world')
    spawn_launch = os.path.join(pkg, 'launch', 'spawn_robot.launch.py')

    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.expanduser(
            '~/Desktop/ros2WS5/RL/model/policy/policy.zip'),
        description='Path to trained RL model (.zip)')

    randomize_arg = DeclareLaunchArgument(
        'randomize', default_value='true',
        description='Randomize robot spawn positions')

    finetune_arg = DeclareLaunchArgument(
        'finetune_robot', default_value='',
        description='Robot namespace being fine-tuned (or "all")')

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run Gazebo without GUI')

    # Gazebo launch arguments
    gazebo_args = {'world': world}
    if os.environ.get('SWARM_HEADLESS', 'false').lower() == 'true':
        gazebo_args['gui'] = 'false'

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')),
        launch_arguments=gazebo_args.items(),
    )

    # Sim logger
    logger = TimerAction(
        period=3.0,
        actions=[Node(
            package='collector_bot',
            executable='sim_logger',
            name='sim_logger',
            output='screen',
        )],
    )

    # Safety coordinator
    safety = TimerAction(
        period=18.0,
        actions=[Node(
            package='collector_bot',
            executable='safety_coordinator',
            name='safety_coordinator',
            output='screen',
        )],
    )

    # Spawn positions
    use_random = os.environ.get('SWARM_RANDOMIZE', 'true').lower() == 'true'
    if use_random:
        rand_pos = _randomize_positions(N_ROBOTS)
        robots = [
            (f'robot_{i+1}', round(rand_pos[i][0], 2), round(rand_pos[i][1], 2))
            for i in range(N_ROBOTS)
        ]
    else:
        robots = DEFAULT_ROBOTS

    spawns = []
    for idx, (name, x, y) in enumerate(robots):
        spawn_robot = TimerAction(
            period=float(5.0 + idx * 4.0),
            actions=[IncludeLaunchDescription(
                PythonLaunchDescriptionSource(spawn_launch),
                launch_arguments={
                    'robot_name': name,
                    'x': str(x), 'y': str(y), 'z': '0.1',
                    'use_brain': 'false',
                }.items(),
            )],
        )
        spawns.append(spawn_robot)

    # Centralized Swarm Brain
    swarm_brain = TimerAction(
        period=18.0,
        actions=[Node(
            package='collector_bot',
            executable='swarm_brain',
            name='swarm_brain',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'finetune_robot': LaunchConfiguration('finetune_robot'),
            }],
            output='screen',
        )],
    )

    return LaunchDescription([
        model_arg, randomize_arg, finetune_arg, headless_arg,
        gazebo, logger, safety, swarm_brain, *spawns,
    ])
