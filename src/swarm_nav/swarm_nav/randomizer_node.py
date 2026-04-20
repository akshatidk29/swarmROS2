#!/usr/bin/env python3
"""Randomize object and bin positions at startup."""
import time
import random
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState


class RandomizerNode(Node):
    def __init__(self):
        super().__init__('randomizer_node')
        self.cli = self.create_client(SetEntityState, '/set_entity_state')

        max_retries = 12
        retries = 0
        while not self.cli.wait_for_service(timeout_sec=5.0):
            retries += 1
            self.get_logger().info(
                f'Waiting for /set_entity_state service... ({retries}/{max_retries})')
            if retries >= max_retries:
                self.get_logger().error(
                    'Could not reach /set_entity_state! '
                    'Make sure the gazebo_ros_state plugin is loaded in your world file.')
                return

        self.get_logger().info('Connected to /set_entity_state. Randomizing...')

        objects = ['sort_obj_red', 'sort_obj_green', 'sort_obj_blue']
        bins = ['bin_red', 'bin_green', 'bin_blue']

        locations = [
            (-3.5, 3.0),
            (-1.0, 3.0),
            (1.0, 3.0),
            (2.0, 3.0),
            (-4.0, 1.0),
            (4.5, 1.0),
            (2.0, 1.0),
            (-1.0, 1.0),
            (4.0, -1.0),
            (2.0, -1.0),
            (-1.0, -1.0),
            (-4.0, -3.0),
            (-2.0, -3.0),
            (2.0, -3.0),
            (4.0, -3.0),
        ]

        random.shuffle(locations)

        for i, name in enumerate(objects):
            self.set_pose(name, locations[i][0], locations[i][1], 0.075)

        for i, name in enumerate(bins):
            self.set_pose(name, locations[i + 3][0], locations[i + 3][1], 0.005)

        self.get_logger().info('Randomization complete.')

    def set_pose(self, name, x, y, z):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.w = 1.0
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            self.get_logger().info(f'  Moved {name} to ({x:.1f}, {y:.1f})')
        else:
            self.get_logger().warn(f'  Failed to move {name}')


def main(args=None):
    rclpy.init(args=args)
    node = RandomizerNode()
    time.sleep(2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
