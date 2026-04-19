"""
teleop.launch.py
----------------
Convenience launcher for teleoperating a single robot.

Usage:
  ros2 launch bot teleop.launch.py                       # controls robot_1
  ros2 launch bot teleop.launch.py robot_ns:=robot_2     # controls robot_2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    ns_arg = DeclareLaunchArgument(
        'robot_ns', default_value='robot_1',
        description='Namespace of the robot to drive')

    robot_ns = LaunchConfiguration('robot_ns')

    teleop = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'teleop_twist_keyboard',
            'teleop_twist_keyboard',
            '--ros-args',
            '-r', ['/cmd_vel:=/', robot_ns, '/cmd_vel'],
        ],
        output='screen',
        prefix='xterm -e',      # opens in a new terminal window
    )

    return LaunchDescription([
        ns_arg,
        teleop,
    ])
