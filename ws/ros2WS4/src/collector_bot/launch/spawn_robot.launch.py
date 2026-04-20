"""Spawn a single namespaced omni-drive robot into a running Gazebo."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('collector_bot')
    xacro_file = os.path.join(pkg, 'urdf', 'robot.urdf.xacro')

    name_arg  = DeclareLaunchArgument('robot_name', default_value='robot_1')
    x_arg     = DeclareLaunchArgument('x', default_value='0.0')
    y_arg     = DeclareLaunchArgument('y', default_value='0.0')
    z_arg     = DeclareLaunchArgument('z', default_value='0.1')
    brain_arg = DeclareLaunchArgument('use_brain', default_value='true')

    rn = LaunchConfiguration('robot_name')

    robot_desc = Command(['xacro ', xacro_file, ' robot_ns:=', rn])

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=rn,
        parameters=[{
            'robot_description': ParameterValue(robot_desc, value_type=str),
            'frame_prefix': [rn, '/'],
        }],
        output='screen',
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', rn,
            '-topic', ['/robot_description'],
            '-robot_namespace', rn,
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
        ],
        remappings=[('/robot_description', ['/', rn, '/robot_description'])],
        output='screen',
    )

    brain = Node(
        package='collector_bot',
        executable='brain',
        name='collector_brain',
        namespace=rn,
        parameters=[{'robot_ns': rn}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_brain')),
    )

    return LaunchDescription([
        name_arg, x_arg, y_arg, z_arg, brain_arg,
        rsp, spawn, brain,
    ])
