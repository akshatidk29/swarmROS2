import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_dir = get_package_share_directory('swarm_description')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'swarm_bot.urdf.xacro')
    world_file = os.path.join(pkg_dir, 'worlds', 'warehouse.world')

    robots = [
        {'name': 'robot_1', 'x': '-4.0', 'y': '3.0', 'yaw': '0.0'},
        {'name': 'robot_2', 'x': '-4.0', 'y': '1.0', 'yaw': '0.0'},
        {'name': 'robot_3', 'x': '-4.0', 'y': '-1.0', 'yaw': '0.0'},
    ]

    ld = LaunchDescription()

    gazebo_server = ExecuteProcess(
        cmd=['gazebo', '--verbose',
             '-s', 'libgazebo_ros_factory.so',
             world_file],
        output='screen',
    )
    ld.add_action(gazebo_server)

    for i, robot in enumerate(robots):
        ns = robot['name']

        robot_description = xacro.process_file(
            urdf_file,
            mappings={'robot_ns': ns}
        ).toxml()

        robot_state_pub = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=ns,
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'frame_prefix': ns + '/',
                'use_sim_time': True
            }],
        )

        spawn_robot = TimerAction(
            period=float(i * 5 + 3),
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    name=f'spawn_{ns}',
                    output='screen',
                    arguments=[
                        '-entity', ns,
                        '-topic', f'/{ns}/robot_description',
                        '-x', robot['x'],
                        '-y', robot['y'],
                        '-z', '0.05',
                        '-Y', robot['yaw'],
                        '-robot_namespace', ns,
                    ],
                )
            ]
        )

        sorting_node = TimerAction(
            period=float(i * 5 + 6),
            actions=[
                Node(
                    package='swarm_nav',
                    executable='sorting_node',
                    namespace=ns,
                    name='sorting_node',
                    output='screen',
                    parameters=[{'robot_name': ns, 'use_sim_time': True}]
                )
            ]
        )

        ld.add_action(robot_state_pub)
        ld.add_action(spawn_robot)
        ld.add_action(sorting_node)

        static_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'static_tf_map_to_{ns}_odom',
            arguments=[
                robot['x'], robot['y'], '0',
                robot['yaw'], '0', '0',
                'map',
                f'{ns}/odom'
            ],
            parameters=[{'use_sim_time': True}]
        )
        ld.add_action(static_tf)

    randomizer_node = TimerAction(
        period=18.0,
        actions=[
            Node(
                package='swarm_nav',
                executable='randomizer_node',
                name='randomizer_node',
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )
    ld.add_action(randomizer_node)

    logger_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='swarm_nav',
                executable='logger_node',
                name='logger_node',
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        ]
    )
    ld.add_action(logger_node)

    return ld
