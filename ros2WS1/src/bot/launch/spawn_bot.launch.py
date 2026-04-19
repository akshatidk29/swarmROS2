"""
spawn_bot.launch.py
-------------------
Reusable launch file that spawns ONE robot into an already-running Gazebo.

Arguments (set via LaunchConfiguration):
  robot_name    – unique name / namespace  (default: robot_1)
  x, y, z       – spawn position           (defaults: 0, 0, 0.1)
  use_avoidance – start avoidance node     (default: true)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg_dir = get_package_share_directory('bot')
    xacro_file = os.path.join(pkg_dir, 'urdf', 'bot.urdf.xacro')

    # ---- Declare arguments ----
    robot_name_arg = DeclareLaunchArgument(
        'robot_name', default_value='robot_1',
        description='Unique robot name / namespace')
    x_arg = DeclareLaunchArgument('x', default_value='0.0')
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    z_arg = DeclareLaunchArgument('z', default_value='0.1')
    avoidance_arg = DeclareLaunchArgument(
        'use_avoidance', default_value='true',
        description='Launch autonomous obstacle-avoidance node')

    robot_name = LaunchConfiguration('robot_name')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')
    use_avoidance = LaunchConfiguration('use_avoidance')

    # ---- Process xacro → URDF string ----
    robot_desc = Command([
        'xacro ', xacro_file, ' robot_ns:=', robot_name
    ])

    # ---- robot_state_publisher (each robot needs its own) ----
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=robot_name,
        parameters=[{'robot_description': ParameterValue(robot_desc, value_type=str),
                      'frame_prefix': [robot_name, '/']}],
        output='screen',
    )

    # ---- Spawn into Gazebo ----
    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', robot_name,
            '-topic', ['/robot_description'],
            '-robot_namespace', robot_name,
            '-x', x, '-y', y, '-z', z,
        ],
        remappings=[('/robot_description',
                     ['/', robot_name, '/robot_description'])],
        output='screen',
    )

    # ---- Obstacle avoidance (conditional) ----
    avoidance = Node(
        package='bot',
        executable='avoid',
        name='obstacle_avoidance',
        namespace=robot_name,
        parameters=[{'robot_ns': robot_name}],
        output='screen',
        condition=IfCondition(use_avoidance),
    )

    return LaunchDescription([
        robot_name_arg, x_arg, y_arg, z_arg, avoidance_arg,
        rsp,
        spawn,
        avoidance,
    ])
