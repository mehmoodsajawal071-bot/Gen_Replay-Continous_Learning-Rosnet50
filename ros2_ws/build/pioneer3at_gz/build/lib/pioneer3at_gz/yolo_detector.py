import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import json

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Load YOLO model (downloads if not cached yet)
        self.model = YOLO("yolov8n.pt")

        # ROS utils
        self.bridge = CvBridge()
        self.create_subscription(Image, '/pioneer3at/camera/image_raw', self.image_callback, 10)
        self.detection_pub = self.create_publisher(String, '/yolo/detections', 10)

        self.get_logger().info("YOLOv8 Detector Node started, waiting for images...")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Run YOLO inference
        results = self.model(frame)

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                detection = {
                    "class": self.model.names[cls],
                    "confidence": round(conf, 2),
                    "bbox": [round(x, 2) for x in xyxy]
                }
                detections.append(detection)

                # Draw on frame
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{self.model.names[cls]} {conf:.2f}",
                            (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Publish detections as JSON string
        if detections:
            self.detection_pub.publish(String(data=json.dumps(detections)))

        # Show in OpenCV
        cv2.imshow("YOLOv8 Detections", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

