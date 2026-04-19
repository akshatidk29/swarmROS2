#!/usr/bin/env python3
"""
RandomizerNode – randomize object/bin positions at simulation start.

Requires the gazebo_ros_state plugin in the world file to provide
the /set_entity_state service.

All candidate locations have been verified to NOT overlap with any shelf
in the 10m x 8m warehouse.
"""
import time
import random
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState


class RandomizerNode(Node):
    def __init__(self):
        super().__init__('randomizer_node')
        self.cli = self.create_client(SetEntityState, '/set_entity_state')

        # Wait up to 60 seconds for the service (Gazebo needs time to load)
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

        # ---- SAFE locations verified to be clear of all shelves ----
        # Warehouse: x ∈ [-5, 5], y ∈ [-4, 4]
        # Shelves occupy:
        #   shelf_1a/1b/1c: x ∈ [-3.5, -1.5], y ∈ {2.0, 0.0, -2.0} ±0.2
        #   shelf_2a/2b/2c: x ∈ [-0.5, 1.5],  y ∈ {2.0, 0.0, -2.0} ±0.2
        #   shelf_3a/3b:    x ∈ [2.75, 4.25],  y ∈ {2.0, -2.0} ±0.2
        # Robot spawns:     x = -4.0, y ∈ {3.0, 1.0, -1.0}
        #
        # All locations below are at least 0.5m from any shelf edge.
        locations = [
            # Upper open areas (y=3.0, above all shelves)
            (-3.5, 3.0),   # far left top
            (-1.0, 3.0),   # left-center top
            (1.0, 3.0),    # center-right top
            (2.0, 3.0),    # right top (below shelf_3a x-range but y is clear)
            # Middle aisle (y=1.0, between shelf rows at y=2.0 and y=0.0)
            (-4.0, 1.0),   # far left mid - NOTE: may conflict with robot_2 spawn
            (4.5, 1.0),    # far right mid
            (2.0, 1.0),    # center right mid
            (-1.0, 1.0),   # center left mid
            # Middle aisle (y=-1.0, between shelf rows at y=0.0 and y=-2.0)
            (4.0, -1.0),   # far right
            (2.0, -1.0),   # center right
            (-1.0, -1.0),  # center left
            # Lower open areas (y=-3.0, below all shelves)
            (-4.0, -3.0),  # far left bottom
            (-2.0, -3.0),  # left bottom
            (2.0, -3.0),   # center right bottom
            (4.0, -3.0),   # far right bottom
        ]

        random.shuffle(locations)

        # Place objects (first 3 shuffled locations)
        for i, name in enumerate(objects):
            self.set_pose(name, locations[i][0], locations[i][1], 0.075)

        # Place bins (next 3 shuffled locations, guaranteed different from objects)
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
    # Give it a moment to finish any async work
    time.sleep(2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
