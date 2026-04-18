#!/usr/bin/env python3
import time
import random
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState

class RandomizerNode(Node):
    def __init__(self):
        super().__init__('randomizer_node')
        self.cli = self.create_client(SetEntityState, '/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('service not available, waiting again...')
            
        self.get_logger().info('Randomizing objects and bins...')
        
        objects = ['sort_obj_red', 'sort_obj_green', 'sort_obj_blue']
        bins = ['bin_red', 'bin_green', 'bin_blue']
        
        # We define a list of possible discrete free locations
        locations = [
            (-3.5, 3.0), (3.5, 2.0), (-2.5, -3.0), (2.0, -3.0),
            (0.0, -3.0), (4.0, 1.0), (4.0, -1.0), (-3.0, 0.0),
            (-1.0, 3.0), (1.0, 3.0), (-4.0, -3.0), (0.0, 2.0)
        ]
        
        random.shuffle(locations)
        
        for i, name in enumerate(objects):
            self.set_pose(name, locations[i][0], locations[i][1], 0.075)
            
        for i, name in enumerate(bins):
            self.set_pose(name, locations[i+3][0], locations[i+3][1], 0.005)
            
        self.get_logger().info('Randomization complete.')

    def set_pose(self, name, x, y, z):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.w = 1.0
        self.cli.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = RandomizerNode()
    # give it a second to send requests
    time.sleep(2.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
