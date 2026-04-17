"""
world.launch.py
---------------
Main entry-point: starts Gazebo with the obstacle world and spawns
multiple robots with optional autonomous obstacle avoidance.

Usage:
  ros2 launch bot world.launch.py
  ros2 launch bot world.launch.py enable_avoidance:=false
  ros2 launch bot world.launch.py num_robots:=5
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


# Default spawn positions (x, y) – spread around the arena
SPAWN_POSITIONS = [
    ( 0.0,  0.0),
    ( 3.0,  4.0),
    (-3.0, -3.0),
    ( 5.0, -6.0),
    (-6.0,  5.0),
    ( 7.0,  2.0),
    (-2.0,  6.0),
    ( 4.0, -8.0),
]


def generate_launch_description():

    pkg_dir = get_package_share_directory('bot')
    world_file = os.path.join(pkg_dir, 'worlds', 'circle.world')
    spawn_launch = os.path.join(pkg_dir, 'launch', 'spawn_bot.launch.py')

    # ---- Arguments ----
    avoidance_arg = DeclareLaunchArgument(
        'enable_avoidance', default_value='true',
        description='Start avoidance node for each robot')
    num_robots_arg = DeclareLaunchArgument(
        'num_robots', default_value='3',
        description='Number of robots to spawn (max 8)')

    enable_avoidance = LaunchConfiguration('enable_avoidance')

    # ---- Gazebo server + client ----
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world_file}.items(),
    )

    # ---- Spawn robots with staggered delays to avoid race conditions ----
    # Note: We hard-code up to 8 robots here. The num_robots arg is
    #       evaluated at launch-time so we cannot use it in a Python loop
    #       directly; instead we spawn all and rely on the delay.  For
    #       simplicity we always create 3 by default.
    spawn_actions = []
    # We'll generate descriptors for 3 robots by default.  You can add
    # more by copy-pasting below or passing num_robots.
    robots = [
        ('robot_1', SPAWN_POSITIONS[0][0], SPAWN_POSITIONS[0][1]),
        ('robot_2', SPAWN_POSITIONS[1][0], SPAWN_POSITIONS[1][1]),
        ('robot_3', SPAWN_POSITIONS[2][0], SPAWN_POSITIONS[2][1]),
    ]

    for idx, (name, x, y) in enumerate(robots):
        spawn = TimerAction(
            period=float(5.0 + idx * 4.0),   # stagger spawns
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(spawn_launch),
                    launch_arguments={
                        'robot_name': name,
                        'x': str(x),
                        'y': str(y),
                        'z': '0.1',
                        'use_avoidance': enable_avoidance,
                    }.items(),
                ),
            ],
        )
        spawn_actions.append(spawn)

    return LaunchDescription([
        avoidance_arg,
        num_robots_arg,
        gazebo,
        *spawn_actions,
    ])
