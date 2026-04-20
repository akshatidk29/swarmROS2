#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        
        self.visited_grid = set()
        self.bins = {}
        
        self.create_subscription(String, '/swarm/visited', self._cb_visited, 10)
        self.create_subscription(String, '/swarm/bin_locations', self._cb_bin_loc, 10)
        
        self.marker_pub = self.create_publisher(MarkerArray, '/swarm/visualization', 10)
        
        self.create_timer(1.0, self._publish_markers)
        self.get_logger().info('Swarm Visualization Node started.')

    def _cb_visited(self, msg):
        parts = msg.data.split(',')
        if len(parts) == 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                self.visited_grid.add((x, y))
            except ValueError:
                pass

    def _cb_bin_loc(self, msg):
        parts = msg.data.split(',')
        if len(parts) == 3:
            try:
                cid = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                self.bins[cid] = (x, y)
            except ValueError:
                pass

    def _publish_markers(self):
        marker_array = MarkerArray()
        
        grid_marker = Marker()
        grid_marker.header.frame_id = "map"
        grid_marker.header.stamp = self.get_clock().now().to_msg()
        grid_marker.ns = "visited_grid"
        grid_marker.id = 0
        grid_marker.type = Marker.CUBE_LIST
        grid_marker.action = Marker.ADD
        grid_marker.pose.orientation.w = 1.0
        grid_marker.scale.x = 0.5
        grid_marker.scale.y = 0.5
        grid_marker.scale.z = 0.05
        grid_marker.color.r = 0.2
        grid_marker.color.g = 0.8
        grid_marker.color.b = 0.2
        grid_marker.color.a = 0.4
        
        for (gx, gy) in self.visited_grid:
            from geometry_msgs.msg import Point
            p = Point()
            p.x = float(gx)
            p.y = float(gy)
            p.z = 0.025
            grid_marker.points.append(p)
            
        if len(grid_marker.points) > 0:
            marker_array.markers.append(grid_marker)

        colors = {
            1: (1.0, 0.2, 0.2),
            2: (0.2, 0.9, 0.2),
            3: (0.2, 0.2, 1.0)
        }
        
        for cid, (x, y) in self.bins.items():
            bin_marker = Marker()
            bin_marker.header.frame_id = "map"
            bin_marker.header.stamp = self.get_clock().now().to_msg()
            bin_marker.ns = "shared_bins"
            bin_marker.id = cid
            bin_marker.type = Marker.CYLINDER
            bin_marker.action = Marker.ADD
            bin_marker.pose.position.x = float(x)
            bin_marker.pose.position.y = float(y)
            bin_marker.pose.position.z = 0.5
            bin_marker.pose.orientation.w = 1.0
            bin_marker.scale.x = 1.0
            bin_marker.scale.y = 1.0
            bin_marker.scale.z = 1.0
            
            r, g, b = colors.get(cid, (1.0, 1.0, 1.0))
            bin_marker.color.r = float(r)
            bin_marker.color.g = float(g)
            bin_marker.color.b = float(b)
            bin_marker.color.a = 0.8
            
            marker_array.markers.append(bin_marker)
            
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
