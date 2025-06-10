import threading
import time
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from shapely.geometry import Polygon, LineString
from dronekit import connect, VehicleMode, Command, LocationGlobalRelative
from pymavlink import mavutil


class StablePID:
    """
    Enhanced PID controller with filtering and smoothing for stable output.

    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        max_output (float): Maximum absolute output.
        tau (float): Time constant for derivative filter.
        derivative_filter_tau (float): Time constant for derivative smoothing.
        error_ema_alpha (float): Alpha for exponential moving average on error.
        output_smoothing_alpha (float): Alpha for smoothing the final output.
        integral_limit (float): Clamps for integral anti-windup.
    """

    def __init__(self, Kp, Ki, Kd, max_output,
                 tau=0.1, derivative_filter_tau=0.05,
                 error_ema_alpha=0.2, output_smoothing_alpha=0.1,
                 integral_limit=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.tau = tau
        self.derivative_filter_tau = derivative_filter_tau
        self.error_ema_alpha = error_ema_alpha
        self.output_smoothing_alpha = output_smoothing_alpha
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        """Reset internal states for a fresh PID cycle."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_error = 0.0
        self.prev_derivative = 0.0
        self.filtered_derivative = 0.0
        self.prev_output = 0.0
        self.prev_time = time.time()

    def update(self, error):
        """
        Compute the PID output given a new error measurement.

        Args:
            error (float): Current error value.
        Returns:
            float: Smoothed, clipped PID output.
        """
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0:
            dt = 1e-16
        self.prev_time = now

        # Exponential moving average on error
        self.filtered_error = (
            self.error_ema_alpha * error +
            (1 - self.error_ema_alpha) * self.filtered_error
        )

        # Integral with anti-windup
        prev_integral = self.integral
        self.integral += self.filtered_error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        # Raw derivative on filtered error
        raw_derivative = (self.filtered_error - self.prev_error) / dt
        derivative = (
            (self.tau * self.prev_derivative + dt * raw_derivative) /
            (self.tau + dt)
        )
        self.prev_derivative = derivative

        # Smooth derivative
        self.filtered_derivative = (
            (self.derivative_filter_tau * self.filtered_derivative + dt * derivative) /
            (self.derivative_filter_tau + dt)
        )

        # PID components
        p_term = self.Kp * self.filtered_error
        i_term = self.Ki * self.integral
        d_term = self.Kd * self.filtered_derivative
        raw_output = p_term + i_term + d_term

        # Clip output and restore integral if windup occurred
        clipped_output = np.clip(raw_output, -self.max_output, self.max_output)
        if clipped_output != raw_output:
            self.integral = prev_integral

        # Smooth final output
        smoothed_output = (
            self.output_smoothing_alpha * clipped_output +
            (1 - self.output_smoothing_alpha) * self.prev_output
        )
        self.prev_output = smoothed_output
        self.prev_error = self.filtered_error

        return smoothed_output * 0.9


def pixel_to_gps(u, v, drone_lat, drone_lon, altitude, heading_deg,
                 resolution=(640, 480), fov=(60, 45)):
    """
    Convert image pixel (u, v) to GPS coordinates (lat, lon), given drone state.

    Args:
        u, v (int): Pixel coordinates.
        drone_lat, drone_lon (float): Drone latitude/longitude.
        altitude (float): Altitude above ground (meters).
        heading_deg (float): Drone heading in degrees (0 = north).
        resolution (tuple): (width, height) in pixels.
        fov (tuple): (h_FOV, v_FOV) in degrees.
    Returns:
        (float, float): (target_lat, target_lon) in degrees.
    """
    fx = resolution[0] / (2 * math.tan(math.radians(fov[0] / 2)))
    fy = resolution[1] / (2 * math.tan(math.radians(fov[1] / 2)))
    cx, cy = resolution[0] / 2, resolution[1] / 2

    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy

    ground_x_cam = altitude * x_norm
    ground_y_cam = -altitude * y_norm

    heading_rad = math.radians(heading_deg)
    delta_north = (ground_y_cam * math.cos(heading_rad) -
                   ground_x_cam * math.sin(heading_rad))
    delta_east = (ground_y_cam * math.sin(heading_rad) +
                  ground_x_cam * math.cos(heading_rad))

    earth_radius = 6378137.0
    meters_per_deg_lat = earth_radius * math.pi / 180.0
    delta_lat = delta_north / meters_per_deg_lat

    lat_rad = math.radians(drone_lat)
    meters_per_deg_lon = meters_per_deg_lat * math.cos(lat_rad)
    delta_lon = delta_east / meters_per_deg_lon

    return drone_lat + delta_lat, drone_lon + delta_lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute Haversine distance (meters) between two GPS coordinates.
    """
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class YoloDetector(Node):
    """
    ROS 2 Node that performs YOLOv4-tiny inference on incoming camera frames.
    Updates shared_state with detected targets and hotspots.
    """

    INPUT_RESOLUTION = (640, 480)
    FOV = (60, 45)
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    DUP_DISTANCE_SQ = 307200
    COOLDOWN_FRAMES = 60

    def __init__(self, config_path, weights_path, classes_path, shared_state):
        super().__init__('yolo_detector')
        self.shared = shared_state
        self.bridge = CvBridge()

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.net = cv2.dnn.readNet(weights_path, config_path)
        # Optionally enable CUDA here
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        self.frame_cx = self.INPUT_RESOLUTION[0] // 2
        self.frame_cy = self.INPUT_RESOLUTION[1] // 2

        # Cooldown histories
        self.published_targets = {}
        self.published_hotspots = {}

        self.subscription = self.create_subscription(
            Image, '/camera', self.image_callback, 10
        )

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, self.INPUT_RESOLUTION)
        classes, scores, boxes = self.model.detect(
            frame, self.CONF_THRESHOLD, self.NMS_THRESHOLD
        )

        # Get drone GPS state
        with self.shared['vehicle_lock']:
            veh = self.shared['vehicle']
            if veh is None:
                return
            lat0 = veh.location.global_frame.lat
            lon0 = veh.location.global_frame.lon
            heading_deg = veh.heading

        for classId, score, box in zip(classes, scores, boxes):
            xmin, ymin, w, h = box
            xmax, ymax = xmin + w, ymin + h
            cx = int((xmin + xmax)/2)
            cy = int((ymin + ymax)/2)

            # TARGET (class 0)
            if int(classId) == 0:
                with self.shared['target_lock']:
                    if self.shared['target_serviced']:
                        self._draw_box(frame, xmin, ymin, xmax, ymax, (0,255,0), (0,0,255), cx, cy)
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
                    with self.shared['detected_targets_lock']:
                        self.shared['detected_targets'].append((cx, cy, self.frame_cx, self.frame_cy, lat0, lon0))
                    lat_t, lon_t = pixel_to_gps(
                        cx, cy, lat0, lon0,
                        self.shared['ALTITUDE'], heading_deg,
                        resolution=self.INPUT_RESOLUTION, fov=self.FOV
                    )
                    with self.shared['gps_lock']:
                        self.shared['detected_targets_gps'].append((lat_t, lon_t))
                    self.published_targets[((cx, cy), (self.frame_cx, self.frame_cy))] = 0
                    print(f"[TARGET] pixel=({cx},{cy}), gps=({lat_t:.7f},{lon_t:.7f})")
                else:
                    print(f"[TARGET-dup] Updated pixel for centering: ({cx},{cy})")
                    self.published_targets[((cx, cy), (self.frame_cx, self.frame_cy))] = 0

                with self.shared['pixel_lock']:
                    self.shared['latest_target_pixel'] = (cx, cy)

                self._draw_box(frame, xmin, ymin, xmax, ymax, (0,255,0), (0,0,255), cx, cy)

            # HOTSPOT (class 1)
            elif int(classId) == 1:
                is_dup_h = False
                for prev_key in list(self.published_hotspots):
                    prev_coord = prev_key[0]
                    dist_sq_h = (cx - prev_coord[0])**2 + (cy - prev_coord[1])**2
                    if dist_sq_h < self.DUP_DISTANCE_SQ:
                        is_dup_h = True
                        self.published_hotspots[prev_key] = 0
                        break

                if not is_dup_h:
                    lat_h, lon_h = pixel_to_gps(
                        cx, cy, lat0, lon0,
                        self.shared['ALTITUDE'], heading_deg,
                        resolution=self.INPUT_RESOLUTION, fov=self.FOV
                    )
                    with self.shared['hotspot_lock']:
                        found = False
                        for hid, (lat_e, lon_e) in self.shared['hotspot_dict'].items():
                            if haversine_distance(lat_h, lon_h, lat_e, lon_e) <= 0.8:
                                found = True
                                break
                        if not found:
                            hid = self.shared['hotspot_id_counter']
                            self.shared['hotspot_dict'][hid] = (lat_h, lon_h)
                            self.shared['hotspot_id_counter'] += 1
                            print(f"[HOTSPOT-new] ID={hid}, gps=({lat_h:.7f},{lon_h:.7f})")
                        else:
                            print(f"[HOTSPOT-dup] gps=({lat_h:.7f},{lon_h:.7f}) within 0.8m")
                    self.published_hotspots[((cx, cy), (self.frame_cx, self.frame_cy))] = 0

                self._draw_box(frame, xmin, ymin, xmax, ymax, (0,0,255), (255,255,0), cx, cy)

        # Draw frame center reference and display
        cv2.circle(frame, (self.frame_cx, self.frame_cy), 5, (255,0,0), -1)
        cv2.imshow("YOLO Detection", frame)
        cv2.waitKey(1)

        # Cleanup cooldown for hotspots
        to_remove = []
        for key in list(self.published_hotspots):
            if key[0] not in [(b[0]+b[2]//2, b[1]+b[3]//2) for b in boxes]:
                self.published_hotspots[key] += 1
                if self.published_hotspots[key] >= self.COOLDOWN_FRAMES:
                    to_remove.append(key)
        for key in to_remove:
            del self.published_hotspots[key]

    @staticmethod
    def _draw_box(frame, xmin, ymin, xmax, ymax, box_color, centroid_color, cx, cy):
        """
        Draw bounding box and centroid circle on frame.
        """
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.circle(frame, (cx, cy), 5, centroid_color, -1)


class DroneMission:
    """
    Manages drone connection, mission upload, survey, and dynamic target diversion.
    """

    def __init__(self, shared_state):
        self.shared = shared_state
        self.ALTITUDE = self.shared['ALTITUDE']
        self.LOW_ALTITUDE = self.shared['LOW_ALTITUDE']
        self.ZIGZAG_SPACING = self.shared['ZIGZAG_SPACING']
        self.POLYGON = self.shared['GEOFENCE_POLYGON']
        self.CONN_STR = self.shared['CONNECTION_STRING']

    def connect_vehicle(self):
        print("Connecting to vehicle...")
        conn = connect(self.CONN_STR, wait_ready=True)
        with self.shared['vehicle_lock']:
            self.shared['vehicle'] = conn
        print("Connected to vehicle.")

    def upload_geofence(self):
        veh = self.shared['vehicle']
        veh._master.mav.param_set_send(
            veh._master.target_system,
            veh._master.target_component,
            b'FENCE_ACTION', 1, mavutil.mavlink.MAV_PARAM_TYPE_INT32
        )
        print("Geofence enabled.")

    def shrink_polygon(self, inset_m=1.0):
        poly = Polygon([(lon, lat) for lat, lon in self.POLYGON])
        inset_deg = inset_m / 111139.0
        inner_poly = poly.buffer(-inset_deg)
        if inner_poly.is_empty:
            raise ValueError("Inset too large, no area left.")
        if inner_poly.geom_type == 'MultiPolygon':
            inner_poly = max(inner_poly.geoms, key=lambda p: p.area)
        return [(lat, lon) for lon, lat in inner_poly.exterior.coords]

    def generate_zigzag_waypoints(self, polygon):
        poly = Polygon([(lon, lat) for lat, lon in polygon])
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        lat_step = self.ZIGZAG_SPACING / 111139.0
        waypoints = []
        current_lat = min_lat
        toggle = False
        while current_lat <= max_lat:
            line = LineString([(min_lon, current_lat), (max_lon, current_lat)])
            inter = poly.intersection(line)
            if not inter.is_empty:
                segments = inter.geoms if inter.geom_type == 'MultiLineString' else [inter]
                for seg in segments:
                    coords = list(seg.coords)
                    if toggle:
                        coords.reverse()
                    for lon_pt, lat_pt in coords:
                        waypoints.append((lat_pt, lon_pt))
                    toggle = not toggle
            current_lat += lat_step
        return waypoints

    def upload_mission(self, waypoints):
        veh = self.shared['vehicle']
        cmds = veh.commands
        cmds.clear()
        time.sleep(1)
        home = veh.location.global_frame
        cmds.add(
            Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0,
                home.lat, home.lon, self.ALTITUDE
            )
        )
        for lat, lon in waypoints:
            cmds.add(
                Command(
                    0, 0, 0,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0, 0, 0, 0, 0, 0,
                    lat, lon, self.ALTITUDE
                )
            )
        cmds.upload()
        print("Mission uploaded.")

    def arm_and_takeoff(self):
        veh = self.shared['vehicle']
        while not veh.is_armable:
            print("Waiting for vehicle to become armable...")
            time.sleep(1)
        print("Arming...")
        veh.mode = VehicleMode("GUIDED")
        veh.armed = True
        while not veh.armed:
            time.sleep(1)
        print("Taking off...")
        veh.simple_takeoff(self.ALTITUDE)
        while veh.location.global_relative_frame.alt < self.ALTITUDE * 0.95:
            print(f"Altitude: {veh.location.global_relative_frame.alt:.2f} m")
            time.sleep(1)
        print("Reached survey altitude.")

    def send_body_velocity(self, vx, vy, vz=0.0):
        with self.shared['vehicle_lock']:
            veh = self.shared['vehicle']
            if veh is None:
                return
            msg = veh.message_factory.set_position_target_local_ned_encode(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000111111000111,
                0, 0, 0,
                vx, vy, vz,
                0, 0, 0,
                0, 0
            )
            veh.send_mavlink(msg)
            veh.flush()

    def execute_mission(self):
        self.connect_vehicle()
        self.upload_geofence()
        inner_poly = self.shrink_polygon(1.0)
        survey_wps = self.generate_zigzag_waypoints(inner_poly)
        self.upload_mission(survey_wps)
        self.arm_and_takeoff()
        with self.shared['vehicle_lock']:
            veh = self.shared['vehicle']
            try:
                veh.parameters['WPNAV_SPEED'] = 500
            except Exception as e:
                print(f"WARNING: could not set WPNAV_SPEED: {e}")
            veh.airspeed = 5.0
        print("Switching to AUTO for survey...")
        veh.mode = VehicleMode("AUTO")

        while True:
            with self.shared['gps_lock']:
                if self.shared['detected_targets_gps']:
                    lat_t, lon_t = self.shared['detected_targets_gps'].pop(-1)
                    print(f"Diverting to target: ({lat_t:.7f}, {lon_t:.7f})")
                    veh.mode = VehicleMode("GUIDED")
                    time.sleep(1)
                    veh.simple_goto(LocationGlobalRelative(lat_t, lon_t, self.ALTITUDE))
                    while True:
                        cur = veh.location.global_relative_frame
                        horiz_dist = math.sqrt((cur.lat - lat_t)**2 + (cur.lon - lon_t)**2) * 111139.0
                        print(f"Distance to target: {horiz_dist:.2f} m")
                        if horiz_dist < 1.5:
                            print("Within 1.5 m horizontally.")
                            break
                        time.sleep(1)
                    print(f"Descending to low altitude: {self.LOW_ALTITUDE} m.")
                    veh.simple_goto(LocationGlobalRelative(lat_t, lon_t, self.LOW_ALTITUDE))
                    while True:
                        if abs(veh.location.global_relative_frame.alt - self.LOW_ALTITUDE) < 0.3:
                            print("Reached low altitude.")
                            break
                        time.sleep(0.5)
                    pid_x = StablePID(0.15, 0.002, 0.02, 0.2)
                    pid_y = StablePID(0.15, 0.002, 0.02, 0.2)
                    pid_x.reset()
                    pid_y.reset()
                    frame_cx, frame_cy = 640//2, 480//2
                    threshold = 10
                    last_detect = time.time()
                    timeout = 1.0
                    print("Starting centering loop...")
                    while True:
                        with self.shared['pixel_lock']:
                            pix = self.shared['latest_target_pixel']
                        if pix is None or (time.time() - last_detect) > timeout:
                            print("Lost target or timed out.")
                            self.send_body_velocity(0,0,0)
                            break
                        last_detect = time.time()
                        ex = pix[0] - frame_cx
                        ey = pix[1] - frame_cy
                        print(f"Pixel error: ex={ex}, ey={ey}")
                        if abs(ex) <= threshold and abs(ey) <= threshold:
                            print("Within deadband. Stopping.")
                            pid_x.reset()
                            pid_y.reset()
                            for _ in range(5):
                                self.send_body_velocity(0,0,0)
                                time.sleep(0.02)
                            with self.shared['pixel_lock']:
                                self.shared['latest_target_pixel'] = None
                            break
                        vx = pid_y.update(ey)
                        vy = -pid_x.update(ex)
                        print(f"Velocity cmd: vx={vx:.3f}, vy={vy:.3f}")
                        if abs(vx) < 0.04 and abs(vy) < 0.04:
                            print("Velocities near zero. Finishing.")
                            pid_x.reset()
                            pid_y.reset()
                            for _ in range(5):
                                self.send_body_velocity(0,0,0)
                                time.sleep(0.02)
                            with self.shared['pixel_lock']:
                                self.shared['latest_target_pixel'] = None
                            break
                        self.send_body_velocity(-vx, -vy, 0)
                        time.sleep(0.1)
                    print("Hovering at low altitude (10 s)...")
                    self.send_body_velocity(0,0,0)
                    time.sleep(10)
                    print(f"Ascending to survey altitude: {self.ALTITUDE} m.")
                    veh.simple_goto(LocationGlobalRelative(lat_t, lon_t, self.ALTITUDE))
                    while True:
                        if abs(veh.location.global_relative_frame.alt - self.ALTITUDE) < 0.5:
                            print("Reached survey altitude.")
                            break
                        time.sleep(0.5)
                    with self.shared['target_lock']:
                        self.shared['target_serviced'] = True
                    print("Resuming survey in AUTO mode.")
                    try:
                        veh.parameters['WPNAV_SPEED'] = 500
                    except Exception as e:
                        print(f"WARNING: could not set speed: {e}")
                    veh.airspeed = 5.0
                    veh.mode = VehicleMode("AUTO")
                    continue
            if self.shared['vehicle'].commands.next >= self.shared['vehicle'].commands.count:
                break
            time.sleep(1)

        print("Survey complete. Returning to launch...")
        self.shared['vehicle'].mode = VehicleMode("RTL")
        while self.shared['vehicle'].armed:
            time.sleep(1)
        with self.shared['hotspot_lock']:
            print("Collected hotspots:")
            for hid, (lat_h, lon_h) in self.shared['hotspot_dict'].items():
                print(f"  ID {hid}: ({lat_h:.7f}, {lon_h:.7f})")
            print(f"Total hotspots: {len(self.shared['hotspot_dict'])}")


def main():
    """
    Entry point: sets up shared state and threads for detector and mission.
    """
    shared_state = {
        'ALTITUDE': 15.0,
        'LOW_ALTITUDE': 10.0,
        'ZIGZAG_SPACING': 18.0,
        'GEOFENCE_POLYGON': [
            (-35.36375764, 149.16612950),
            (-35.36374710, 149.16495300),
            (-35.36235539, 149.16494007),
            (-35.36239757, 149.16592264),
            (-35.36375764, 149.16612950)
        ],
        'CONNECTION_STRING': '127.0.0.1:14550',
        'detected_targets': [],
        'detected_targets_gps': [],
        'hotspot_dict': {},
        'hotspot_id_counter': 1,
        'published_hotspot_history': {},
        'latest_target_pixel': None,
        'target_serviced': False,
        'vehicle': None,
        'detected_targets_lock': threading.Lock(),
        'gps_lock': threading.Lock(),
        'hotspot_lock': threading.Lock(),
        'pixel_lock': threading.Lock(),
        'target_lock': threading.Lock(),
        'vehicle_lock': threading.Lock()
    }

    rclpy.init()
    detector_node = YoloDetector(
        config_path='path/to/yolov4-tiny.cfg',
        weights_path='path/to/yolov4-tiny.weights',
        classes_path='path/to/classes.txt',
        shared_state=shared_state
    )
    detector_thread = threading.Thread(target=rclpy.spin, args=(detector_node,), daemon=True)
    detector_thread.start()

    mission = DroneMission(shared_state)
    mission_thread = threading.Thread(target=mission.execute_mission)
    mission_thread.start()

    mission_thread.join()
    detector_node.destroy_node()
    rclpy.shutdown()
    detector_thread.join()
    print("All threads completed.")


if __name__ == '__main__':
    main()
