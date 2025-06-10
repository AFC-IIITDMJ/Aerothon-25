import threading
import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pid_controller import StablePID
from coordinates import pixel_to_gps


class YoloDetector(Node):
    """
    ROS 2 Node that performs YOLOv4-tiny inference on incoming camera frames.
    Publishes pixel and GPS coordinates for detected "targets" (class 0) and records "hotspots" (class 1).
    """

    INPUT_RESOLUTION = (640, 480)
    FOV = (60, 45)  # (horizontal_FOV, vertical_FOV) in degrees
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    DUP_DISTANCE_SQ = 307200
    COOLDOWN_FRAMES = 60

    def __init__(self,
                 config_path: str,
                 weights_path: str,
                 classes_path: str,
                 shared_state):
        """
        Initialize the YOLODetector node.

        Args:
            config_path (str): Path to YOLOv4-tiny .cfg file.
            weights_path (str): Path to YOLOv4-tiny .weights file.
            classes_path (str): Path to class names (.data or .txt).
            shared_state (dict): A dictionary containing thread-safe shared variables and locks.
        """
        super().__init__('yolo_detector')

        self.shared = shared_state
        self.bridge = CvBridge()

        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Initialize YOLOv4-tiny model
        self.net = cv2.dnn.readNet(weights_path, config_path)
        # Uncomment the following lines to enable CUDA acceleration if available
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        # Frame center
        self.frame_cx = self.INPUT_RESOLUTION[0] // 2
        self.frame_cy = self.INPUT_RESOLUTION[1] // 2

        # Cooldown histories
        self.published_targets = {}  # { ((u,v),(cx,cy)) : cooldown_counter }
        self.published_hotspots = {}

        # ROS 2 Image subscription
        self.subscription = self.create_subscription(
            Image, '/camera', self.image_callback, 10
        )

    def image_callback(self, msg: Image) -> None:
        """
        Callback for each incoming camera frame. Runs YOLO inference, updates shared state.
        """
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, self.INPUT_RESOLUTION)

        classes, scores, boxes = self.model.detect(
            frame, self.CONF_THRESHOLD, self.NMS_THRESHOLD
        )

        seen_targets = []
        seen_hotspots = []

        # Retrieve drone GPS state
        with self.shared['vehicle_lock']:
            veh = self.shared['vehicle']
            if veh is None:
                return
            lat0 = veh.location.global_frame.lat
            lon0 = veh.location.global_frame.lon
            heading_deg = veh.heading

        for cls_id, score, box in zip(classes, scores, boxes):
            xmin, ymin, width, height = box
            xmax = xmin + width
            ymax = ymin + height
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)

            # TARGET CLASS (class 0)
            if int(cls_id) == 0:
                with self.shared['target_lock']:
                    if self.shared['target_serviced']:
                        self._draw_box(frame, xmin, ymin, xmax, ymax, (0, 255, 0), (0, 0, 255), cx, cy)
                        continue

                is_dup = False
                for prev_key in list(self.published_targets):
                    prev_coord = prev_key[0]
                    dist_sq = (cx - prev_coord[0])**2 + (cy - prev_coord[1])**2
                    if dist_sq < self.DUP_DISTANCE_SQ:
                        is_dup = True
                        self.published_targets[prev_key] = 0
                        break

                if not is_dup:
                    # Update shared raw pixel list
                    with self.shared['detected_targets_lock']:
                        self.shared['detected_targets'].append((cx, cy, self.frame_cx, self.frame_cy, lat0, lon0))

                    # Convert to GPS and update shared GPS list
                    lat_t, lon_t = pixel_to_gps(cx, cy, lat0, lon0,
                                                self.shared['ALTITUDE'], heading_deg,
                                                resolution=self.INPUT_RESOLUTION,
                                                fov=self.FOV)
                    with self.shared['gps_lock']:
                        self.shared['detected_targets_gps'].append((lat_t, lon_t))

                    self.published_targets[((cx, cy), (self.frame_cx, self.frame_cy))] = 0
                    print(f"[TARGET] pixel=({cx},{cy}), gps=({lat_t:.7f},{lon_t:.7f})")
                else:
                    print(f"[TARGET-dup] Updated pixel for centering: ({cx},{cy})")
                    self.published_targets[((cx, cy), (self.frame_cx, self.frame_cy))] = 0

                # Always update latest pixel for centering
                with self.shared['pixel_lock']:
                    self.shared['latest_target_pixel'] = (cx, cy)

                self._draw_box(frame, xmin, ymin, xmax, ymax, (0, 255, 0), (0, 0, 255), cx, cy)

            # HOTSPOT CLASS (class 1)
            elif int(cls_id) == 1:
                is_dup_h = False
                for prev_key in list(self.published_hotspots):
                    prev_coord = prev_key[0]
                    dist_sq_h = (cx - prev_coord[0])**2 + (cy - prev_coord[1])**2
                    if dist_sq_h < self.DUP_DISTANCE_SQ:
                        is_dup_h = True
                        self.published_hotspots[prev_key] = 0
                        break

                if not is_dup_h:
                    lat_h, lon_h = pixel_to_gps(cx, cy, lat0, lon0,
                                                self.shared['ALTITUDE'], heading_deg,
                                                resolution=self.INPUT_RESOLUTION,
                                                fov=self.FOV)
                    # Check against existing hotspots
                    with self.shared['hotspot_lock']:
                        found = False
                        for hid, (lat_e, lon_e) in self.shared['hotspot_dict'].items():
                            dist_m = haversine_distance(lat_h, lon_h, lat_e, lon_e)
                            if dist_m <= 0.8:
                                found = True
                                break

                        if not found:
                            hid = self.shared['hotspot_id_counter']
                            self.shared['hotspot_dict'][hid] = (lat_h, lon_h)
                            self.shared['hotspot_id_counter'] += 1
                            print(f"[HOTSPOT-new] ID={hid}, gps=({lat_h:.7f},{lon_h:.7f})")
                        else:
                            print(f"[HOTSPOT-dup] gps=({lat_h:.7f},{lon_h:.7f}) within 0.8 m of existing")

                    self.published_hotspots[((cx, cy), (self.frame_cx, self.frame_cy))] = 0

                self._draw_box(frame, xmin, ymin, xmax, ymax, (0, 0, 255), (255, 255, 0), cx, cy)

        # Draw center reference
        cv2.circle(frame, (self.frame_cx, self.frame_cy), 5, (255, 0, 0), -1)
        cv2.imshow("YOLO Detection", frame)
        cv2.waitKey(1)

        # Cooldown cleanup for hotspots
        to_remove = []
        for key in list(self.published_hotspots):
            if key[0] not in [(c[0], c[1]) for c in boxes]:
                self.published_hotspots[key] += 1
                if self.published_hotspots[key] >= self.COOLDOWN_FRAMES:
                    to_remove.append(key)
        for key in to_remove:
            del self.published_hotspots[key]

    @staticmethod
    def _draw_box(frame, xmin, ymin, xmax, ymax, box_color, centroid_color, cx, cy):
        """
        Utility function to draw a bounding box and centroid circle on a frame.
        """
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.circle(frame, (cx, cy), 5, centroid_color, -1)
