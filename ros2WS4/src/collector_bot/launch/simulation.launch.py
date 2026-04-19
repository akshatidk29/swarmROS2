"""Main launch: Gazebo arena + 4 collector robots (omni-drive) + sim_logger."""

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
    spawn = os.path.join(pkg, 'launch', 'spawn_robot.launch.py')

    brain_arg = DeclareLaunchArgument(
        'use_brain', default_value='true',
        description='Run collector brain on each robot')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world}.items(),
    )

    # Sim logger node
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
        spawns.append(TimerAction(
            period=float(5.0 + idx * 4.0),
            actions=[IncludeLaunchDescription(
                PythonLaunchDescriptionSource(spawn),
                launch_arguments={
                    'robot_name': name,
                    'x': str(x), 'y': str(y), 'z': '0.1',
                    'use_brain': LaunchConfiguration('use_brain'),
                }.items(),
            )],
        ))

    return LaunchDescription([brain_arg, gazebo, logger, *spawns])
