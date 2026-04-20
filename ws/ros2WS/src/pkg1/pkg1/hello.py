import rclpy
from rclpy.node import Node

class Hello(Node):
    def __init__(self):
        super().__init__('hello')
        self.get_logger().info("Hello ROS2!")

def main(args=None):
    rclpy.init(args=args)
    node = Hello()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()