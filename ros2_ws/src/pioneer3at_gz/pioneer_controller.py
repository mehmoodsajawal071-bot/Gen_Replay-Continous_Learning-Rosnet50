import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class PioneerController(Node):
    def __init__(self):
        super().__init__('pioneer_controller')
        # Create a publisher to send commands to /cmd_vel
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # Create a timer to send commands periodically
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1 second timer

    def timer_callback(self):
        # Create a Twist message for velocity
        msg = Twist()
        msg.linear.x = 0.5  # Move forward at 0.5 m/s
        msg.angular.z = 0.1  # Rotate at 0.1 rad/s
        # Publish the message to /cmd_vel
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = PioneerController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

