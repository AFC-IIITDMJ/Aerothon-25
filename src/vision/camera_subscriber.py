from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from src.vision.yolo_detector import YOLODetector

class CameraSubscriber(Node):
    
    def __init__(self, shared_data, model_path: str):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image, '/camera', self.image_callback, 10)
        self.shared_data = shared_data
        self.br = CvBridge()
        self.detector = YOLODetector(model_path)
        self.window_name = "YOLOv8 Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)
        self.get_logger().info("Camera subscriber started")
    
    def image_callback(self, msg) -> None:
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            bbox = self.detector.detect(cv_image)
            with self.shared_data.lock:
                self.shared_data.coords = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if bbox else None
                if bbox:
                    print(self.shared_data.coords)
            cv2.imshow(self.window_name, cv_image)
            if cv2.waitKey(1) == ord('q'):
                self.get_logger().info("Exiting...")
                self.destroy_node()
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            cv2.destroyAllWindows()
    
    def destroy_node(self) -> None:
        cv2.destroyAllWindows()
        super().destroy_node()