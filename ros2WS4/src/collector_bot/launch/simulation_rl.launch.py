"""RL launch: Gazebo arena + 4 RL-controlled robots + sim_logger."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROBOTS = [
    ('robot_1',  0.0, -2.0),
    ('robot_2',  2.5, -4.0),
    ('robot_3', -2.5, -4.0),
    ('robot_4',  0.0, -4.5),
]


def generate_launch_description():
    pkg = get_package_share_directory('collector_bot')
    world = os.path.join(pkg, 'worlds', 'arena.world')
    spawn_launch = os.path.join(pkg, 'launch', 'spawn_robot.launch.py')

    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.expanduser(
            '~/Desktop/ros2WS4/RL/model/policy'),
        description='Path to trained RL model (.zip)')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world}.items(),
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

    spawns = []
    for idx, (name, x, y) in enumerate(ROBOTS):
        # Spawn robot (no deterministic brain)
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

        # RL brain node
        rl_brain = TimerAction(
            period=float(6.0 + idx * 4.0),
            actions=[Node(
                package='collector_bot',
                executable='brain_rl',
                name='collector_brain_rl',
                namespace=name,
                parameters=[{
                    'robot_ns': name,
                    'model_path': LaunchConfiguration('model_path'),
                }],
                output='screen',
            )],
        )
        spawns.append(rl_brain)

    return LaunchDescription([model_arg, gazebo, logger, *spawns])
