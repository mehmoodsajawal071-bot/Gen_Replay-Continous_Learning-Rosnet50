import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix


class PioneerController(Node):
    def __init__(self):
        super().__init__('pioneer_controller')

        # Publisher to /cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber to GPS data
        self.create_subscription(NavSatFix, '/gps/gps/data', self.gps_callback, 10)

        # Timer to move robot forward
        self.timer = self.create_timer(0.1, self.move_robot)

    def move_robot(self):
        msg = Twist()
        msg.linear.x = 0.5   # forward speed
        msg.angular.z = 0.0  # no rotation
        self.cmd_pub.publish(msg)

    def gps_callback(self, msg: NavSatFix):
        self.get_logger().info(
            f"GPS -> Lat: {msg.latitude:.6f}, Lon: {msg.longitude:.6f}, Alt: {msg.altitude:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PioneerController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


