# fire_detection_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import torch
from .fire_model_loader import FireClassifier  # Relative import

class FireDetectionNode(Node):
    def __init__(self):
        super().__init__('fire_detection_node')

        # Parameters
        self.model_path = "/home/ubuntu/ros2_ws/src/pioneer3at_gz/gr_model.pth"
        self.image_size = 224
        self.fire_threshold = 0.7

        # Load Generative Replay model
        self.classifier = FireClassifier(self.model_path)
        self.get_logger().info("🔥 FireClassifier loaded successfully")

        # ROS interfaces
        self.bridge = CvBridge()

        # Subscribe to Pioneer3AT camera topic
        self.subscription = self.create_subscription(
            Image,
            '/pioneer3at/camera/image_raw',  # ✅ correct topic
            self.image_callback,
            10
        )

        # Publisher for fire detection
        self.publisher = self.create_publisher(Bool, '/fire_detected', 10)

        self.get_logger().info("🔥 Fire Detection Node started (GPU enabled if available)")

    def preprocess(self, cv_image):
        """Resize, normalize, convert to tensor"""
        image = cv2.resize(cv_image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return tensor.unsqueeze(0)

    def image_callback(self, msg):
        """Process each incoming camera frame"""
        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Optional: visualize camera feed
        cv2.imshow("Pioneer3AT Camera", cv_image)
        cv2.waitKey(1)

        # Preprocess and predict
        input_tensor = self.preprocess(cv_image)
        probs = self.classifier.predict(input_tensor)
        
        # Fire probability (class 1 = fire)
        fire_prob = probs[0, 1].item()
        fire_msg = Bool()
        fire_msg.data = fire_prob > self.fire_threshold
        self.publisher.publish(fire_msg)

        self.get_logger().info(f"Fire probability: {fire_prob:.2f}")

# -----------------------
# Main function required by ROS 2 entry point
# -----------------------
def main(args=None):
    rclpy.init(args=args)
    node = FireDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()  # Close OpenCV windows
        rclpy.shutdown()

# Allows running the script directly
if __name__ == '__main__':
    main()

