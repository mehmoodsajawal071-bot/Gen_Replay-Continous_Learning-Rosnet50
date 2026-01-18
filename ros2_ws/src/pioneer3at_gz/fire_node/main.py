# main.py
import rclpy
from fire_detection_node import FireDetectionNode

def main(args=None):
    rclpy.init(args=args)
    node = FireDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
