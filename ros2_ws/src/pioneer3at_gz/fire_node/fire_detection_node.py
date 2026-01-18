#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import torch
from .fire_model_loader import FireClassifier  # Your wrapper class

class VideoFirePublisher(Node):
    def __init__(self, video_path, model_path, topic_name='pioneer3at/camera/image_raw', device='cuda'):
        super().__init__('video_fire_publisher')

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open video: {video_path}")
            raise RuntimeError(f"Cannot open video: {video_path}")

        # ROS publisher
        self.publisher = self.create_publisher(Image, topic_name, 10)
        self.fire_pub = self.create_publisher(Bool, '/fire_detected', 10)
        self.bridge = CvBridge()

        # Fire detection model
        self.device = device
        self.classifier = FireClassifier(model_path, device=device)  # internally moves model to device

        # Parameters
        self.image_size = 224
        self.fire_threshold = 0.7

        # OpenCV window setup (resizable)
        self.window_name = "Fire Detection Video"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # make window resizable
        cv2.resizeWindow(self.window_name, 960, 540)           # initial size (optional)

        # Timer for publishing frames (30 FPS)
        self.timer = self.create_timer(1/30, self.timer_callback)

        self.get_logger().info("VideoFirePublisher node started")

    def preprocess(self, cv_image):
        """Convert OpenCV image to PyTorch tensor"""
        image = cv2.resize(cv_image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return tensor.unsqueeze(0).to(self.device)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # Publish frame to ROS topic
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)

        # Predict fire probability
        input_tensor = self.preprocess(frame)
        probs = self.classifier.predict(input_tensor)  # shape [1, num_classes]
        fire_prob = probs[0, 1].item()  # class 1 = fire

        # Publish fire detection
        fire_msg = Bool()
        fire_msg.data = fire_prob > self.fire_threshold
        self.fire_pub.publish(fire_msg)

        # Overlay probability on frame
        label = f"Fire Probability: {fire_prob:.2f}"
        color = (0, 0, 255) if fire_msg.data else (0, 255, 0)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show frame
        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    video_path = ''   # Change to your video path
    model_path = 'pioneer3at_gz/gr_model.pth'
    node = VideoFirePublisher(video_path, model_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

