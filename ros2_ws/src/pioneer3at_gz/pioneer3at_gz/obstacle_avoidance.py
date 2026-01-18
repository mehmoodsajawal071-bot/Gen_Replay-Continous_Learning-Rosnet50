import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import threading
import sys
import termios
import tty

class TeleopObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('teleop_obstacle_avoidance')

        # Publisher to /cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber to LiDAR /scan
        self.create_subscription(LaserScan, '/pioneer3at/lidar_plugin/out', self.lidar_callback, 10)

        # Desired velocity from keyboard
        self.desired_cmd = Twist()

        # Obstacle avoidance parameters
        self.SAFE_DISTANCE = 1.5       # meters
        self.EMERGENCY_STOP = 0.3      # meters
        self.MAX_FORWARD_SPEED = 0.3   # m/s
        self.TURN_SPEED = 0.8          # rad/s

        # Start keyboard thread
        threading.Thread(target=self.keyboard_loop, daemon=True).start()

    def lidar_callback(self, msg: LaserScan):
        ranges = [r if r > 0 else float('inf') for r in msg.ranges]
        num_points = len(ranges)

        # Indices for front ±30°, left 30–90°, right -90° to -30°
        front_indices = list(range(0, 60)) + list(range(num_points-60, num_points))
        left_indices = list(range(60, 120))
        right_indices = list(range(num_points-120, num_points-60))

        front = min([ranges[i] for i in front_indices])
        left = min([ranges[i] for i in left_indices])
        right = min([ranges[i] for i in right_indices])

        # Start with desired keyboard command
        cmd = Twist()
        cmd.linear.x = self.desired_cmd.linear.x
        cmd.angular.z = self.desired_cmd.angular.z

        # Obstacle avoidance logic
        if front < self.EMERGENCY_STOP:
            # Very close → stop and turn
            cmd.linear.x = 0.0
            cmd.angular.z = self.TURN_SPEED if left > right else -self.TURN_SPEED
        elif front < self.SAFE_DISTANCE:
            # Slow down proportionally as obstacle approaches
            scale = (front - self.EMERGENCY_STOP) / (self.SAFE_DISTANCE - self.EMERGENCY_STOP)
            cmd.linear.x = self.MAX_FORWARD_SPEED * max(scale, 0.0)
            # Optional small turn to avoid sides
            if left < right:
                cmd.angular.z += -0.3
            else:
                cmd.angular.z += 0.3

        self.cmd_pub.publish(cmd)

        # Optional logging
        self.get_logger().info(f"Front: {front:.2f}, Left: {left:.2f}, Right: {right:.2f}, Linear: {cmd.linear.x:.2f}")

    def keyboard_loop(self):
        print("Use WASD keys to move the robot. Press Q to quit.")
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while True:
                key = sys.stdin.read(1)
                cmd = Twist()

                if key == 'w':
                    cmd.linear.x = self.MAX_FORWARD_SPEED
                elif key == 's':
                    cmd.linear.x = -0.2
                elif key == 'a':
                    cmd.angular.z = 0.5
                elif key == 'd':
                    cmd.angular.z = -0.5
                elif key == 'q':
                    break

                self.desired_cmd = cmd

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopObstacleAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

