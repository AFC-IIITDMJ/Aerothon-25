import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
from dronekit import connect, VehicleMode, Command, LocationGlobalRelative
from pymavlink import mavutil
from shapely.geometry import Polygon, LineString

# === GLOBAL CONSTANTS ===
ALTITUDE = 15
LOW_ALTITUDE = 10
ZIGZAG_SPACING = 18
CONNECTION_STRING = '127.0.0.1:14550'

# Your mission polygon
polygon_coords = [
    (-35.36375764, 149.16612950),
    (-35.36374710, 149.16495300),
    (-35.36235539, 149.16494007),
    (-35.36239757, 149.16592264),
    (-35.36375764, 149.16612950)
]

# === SHARED STATE ===
detected_targets = []           # (u, v, cx, cy, lat0, lon0)
detected_targets_gps = []       # (lat_target, lon_target)
detected_targets_lock = threading.Lock()

# For hotspots:
hotspot_dict = {}               # {hotspot_id: (lat, lon)}
hotspot_id_counter = 1          # next integer ID to assign
published_hotspot_history = {}  # { ((u, v), (frame_cx, frame_cy)) : cooldown_counter }
hotspot_lock = threading.Lock()

# Latest pixel-of-interest (target centering)
latest_target_pixel = None      # (u, v)
pixel_lock = threading.Lock()

# Standard locks for vehicle & GPS list
vehicle = None
vehicle_lock = threading.Lock()
gps_lock = threading.Lock()

# New: once a target has been fully serviced, disable further “target” handling
target_serviced = False
target_lock = threading.Lock()

# === STABLE PID CLASS (unchanged) ===
class StablePID:
    """
    EliteStablePID - Enhanced PID controller with multiple filters for superior stability
    """
    def __init__(self, Kp, Ki, Kd, max_output, tau=0.1, derivative_filter_tau=0.05,
                 error_ema_alpha=0.2, output_smoothing_alpha=0.1, integral_limit=1.0):
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
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_error = 0.0
        self.prev_derivative = 0.0
        self.filtered_derivative = 0.0
        self.prev_output = 0.0
        self.prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0:
            dt = 1e-16
        self.prev_time = now

        # EMA on error
        self.filtered_error = self.error_ema_alpha * error + (1 - self.error_ema_alpha) * self.filtered_error

        # Integral with anti-windup
        prev_integral = self.integral
        self.integral += self.filtered_error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        # Raw derivative on filtered error
        raw_derivative = (self.filtered_error - self.prev_error) / dt
        derivative = (self.tau * self.prev_derivative + dt * raw_derivative) / (self.tau + dt)
        self.prev_derivative = derivative

        # Extra smoothing
        self.filtered_derivative = (self.derivative_filter_tau * self.filtered_derivative +
                                   dt * derivative) / (self.derivative_filter_tau + dt)

        # PID terms
        p_term = self.Kp * self.filtered_error
        i_term = self.Ki * self.integral
        d_term = self.Kd * self.filtered_derivative
        raw_output = p_term + i_term + d_term

        # Clip & anti-windup
        clipped_output = np.clip(raw_output, -self.max_output, self.max_output)
        if clipped_output != raw_output:
            self.integral = prev_integral

        # Output smoothing
        smoothed_output = self.output_smoothing_alpha * clipped_output + (1 - self.output_smoothing_alpha) * self.prev_output
        self.prev_output = smoothed_output

        self.prev_error = self.filtered_error
        return smoothed_output * 0.9

# === PIXEL→GPS CONVERSION (unchanged) ===

def pixel_to_gps_improved(u, v, drone_lat, drone_lon, altitude, heading_deg,
                              resolution=(640, 480), fov=(60, 45)):
    """
    Convert image pixel (u, v) to a GPS coordinate, accounting for the drone's heading.

    Args:
        u, v: Pixel coordinates in the image frame.
        drone_lat, drone_lon: Current latitude and longitude of the drone (in degrees).
        altitude: Drone's altitude above ground in meters.
        heading_deg: Drone's heading in degrees (0 = North, increasing clockwise).
        resolution: Tuple (width, height) of the camera image in pixels.
        fov: Tuple (horizontal_fov, vertical_fov) in degrees.

    Returns:
        (target_lat, target_lon): Converted GPS latitude and longitude.
    """
    # 1) Compute focal lengths in pixels
    fx = resolution[0] / (2 * math.tan(math.radians(fov[0] / 2)))
    fy = resolution[1] / (2 * math.tan(math.radians(fov[1] / 2)))
    cx, cy = resolution[0] / 2, resolution[1] / 2

    # 2) Normalized image coordinates
    x_norm = (u - cx) / fx   # Positive → to the right
    y_norm = (v - cy) / fy   # Positive → downward

    # 3) Project onto ground plane at given altitude
    ground_x_cam = altitude * x_norm       # In camera frame: +x_cam = right
    ground_y_cam = -altitude * y_norm      # In camera frame: +y_cam = forward (north when heading = 0)

    # 4) Rotate by heading to get North/East offsets
    heading_rad = math.radians(heading_deg)
    delta_north =  ground_y_cam * math.cos(heading_rad) - ground_x_cam * math.sin(heading_rad)
    delta_east  =  ground_y_cam * math.sin(heading_rad) + ground_x_cam * math.cos(heading_rad)

    # 5) Convert from meters to degrees
    earth_radius = 6378137.0  # meters
    meters_per_deg_lat = earth_radius * math.pi / 180.0
    delta_lat = delta_north / meters_per_deg_lat

    lat_rad = math.radians(drone_lat)
    meters_per_deg_lon = meters_per_deg_lat * math.cos(lat_rad)
    delta_lon = delta_east / meters_per_deg_lon

    target_lat = drone_lat + delta_lat
    target_lon = drone_lon + delta_lon

    return target_lat, target_lon


# === SEND BODY VELOCITY (unchanged) ===
def send_body_velocity(vx, vy, vz=0.0):
    with vehicle_lock:
        if vehicle is None:
            return
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )
        vehicle.send_mavlink(msg)
        vehicle.flush()

# ===  DETECTOR NODE ===
class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        # YOLOv4-tiny model files - update these paths to your model files
        config_path = r'/home/wolf/Downloads/TargetDetection.v1i.darknet-20250529T134516Z-1-001/TargetDetection.v1i.darknet/yolov4-tiny.cfg'
        weights_path = r'/home/wolf/Downloads/TargetDetection.v1i.darknet-20250529T134516Z-1-001/TargetDetection.v1i.darknet/yolov4-tiny_last.weights'
        classes_path = r'/home/wolf/Downloads/TargetDetection.v1i.darknet-20250529T134516Z-1-001/TargetDetection.v1i.darknet/obj.data'  # or classes.txt

        # Load class names
        self.class_names = []
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Initialize YOLOv4-tiny with OpenCV DNN
        self.net = cv2.dnn.readNet(weights_path, config_path)

        # Optional: Use CUDA acceleration if available
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        # Preserve original initialization parameters
        self.resW, self.resH = 640, 480
        self.frame_cx, self.frame_cy = self.resW // 2, self.resH // 2
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4  # Non-maximum suppression threshold
        self.duplicate_distance_sq = 307200
        self.cooldown_frames = 60
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera', self.image_callback, 10
        )
        self.published_coords_history = {}  # {( (u,v), (cx,cy) ): cooldown_counter}

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, (self.resW, self.resH))

        # Perform detection using YOLOv4-tiny with OpenCV DNN
        classes, scores, boxes = self.model.detect(
            frame, self.conf_threshold, self.nms_threshold
        )

        seen_coords_targets = []
        seen_coords_hotspots = []

        # Get current drone GPS
        with vehicle_lock:
            if vehicle is None:
                return
            lat0 = vehicle.location.global_frame.lat
            lon0 = vehicle.location.global_frame.lon
            heading_deg=vehicle.heading

        # Process detections
        for classId, score, box in zip(classes, scores, boxes):
            # OpenCV DNN returns boxes as (x, y, width, height)
            xmin, ymin, width, height = box
            xmax = xmin + width
            ymax = ymin + height

            object_cx = int((xmin + xmax) / 2)
            object_cy = int((ymin + ymax) / 2)

            cls_id = int(classId)

            # === TARGET LOGIC ===
            if cls_id == 0:
                # If we've already serviced a target, skip any further target processing:
                with target_lock:
                    if target_serviced:
                        # Just draw the box (optional) and continue
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.circle(frame, (object_cx, object_cy), 5, (0, 0, 255), -1)
                        continue

                seen_coords_targets.append((object_cx, object_cy))

                # Duplicate check for target
                is_duplicate_t = False
                for prev_key in self.published_coords_history:
                    prev_coord = prev_key[0]
                    dist_sq = (object_cx - prev_coord[0])**2 + (object_cy - prev_coord[1])**2
                    if dist_sq < self.duplicate_distance_sq:
                        is_duplicate_t = True
                        self.published_coords_history[prev_key] = 0
                        break

                if not is_duplicate_t:
                    # 1) Record raw pixel target
                    with detected_targets_lock:
                        detected_targets.append((object_cx, object_cy, self.frame_cx, self.frame_cy, lat0, lon0))

                    # 2) Convert to GPS for the target
                    lat_target, lon_target = pixel_to_gps_improved(
                        object_cx, object_cy, lat0, lon0, ALTITUDE,heading_deg,
                        resolution=(self.resW, self.resH), fov=(60, 45)
                    )
                    with gps_lock:
                        detected_targets_gps.append((lat_target, lon_target))

                    self.published_coords_history[((object_cx, object_cy), (self.frame_cx, self.frame_cy))] = 0

                # Always update latest pixel for centering loop
                with pixel_lock:
                    global latest_target_pixel
                    latest_target_pixel = (object_cx, object_cy)

                if not is_duplicate_t:
                    print(
                        f"[TARGET] pixel=({object_cx},{object_cy}), drone_gps=({lat0:.7f},{lon0:.7f}), "
                        f"target_gps=({lat_target:.7f},{lon_target:.7f})"
                    )
                else:
                    print(f"[TARGET-dup] Updated pixel for centering: ({object_cx},{object_cy})")

                # Draw bounding-box & centroid
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.circle(frame, (object_cx, object_cy), 5, (0, 0, 255), -1)

            # === HOTSPOT LOGIC ===
            elif cls_id == 1:
                seen_coords_hotspots.append((object_cx, object_cy))

                # Duplicate‐within‐frame check for hotspot
                is_dup_h = False
                for prev_key in published_hotspot_history:
                    prev_coord = prev_key[0]
                    dist_sq_h = (object_cx - prev_coord[0])**2 + (object_cy - prev_coord[1])**2
                    if dist_sq_h < self.duplicate_distance_sq:
                        is_dup_h = True
                        published_hotspot_history[prev_key] = 0
                        break

                if not is_dup_h:
                    # Convert this hotspot pixel to GPS
                    lat_hs, lon_hs = pixel_to_gps_improved(
                        object_cx, object_cy, lat0, lon0, ALTITUDE,
                        resolution=(self.resW, self.resH), fov=(60, 45)
                    )

                    # ---- NEW: Use Haversine instead of per‐deg approximation ----
                    def haversine_dist(lat1, lon1, lat2, lon2):
                        R = 6371000.0  # meters
                        phi1, phi2 = math.radians(lat1), math.radians(lat2)
                        dphi = math.radians(lat2 - lat1)
                        dlambda = math.radians(lon2 - lon1)
                        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                        return R * c

                    # Check against existing hotspot GPS entries (within 0.8 m)
                    is_new_hs = True
                    with hotspot_lock:
                        for existing_id, (lat_exist, lon_exist) in hotspot_dict.items():
                            horiz_dist = haversine_dist(lat_hs, lon_hs, lat_exist, lon_exist)
                            if horiz_dist <= 0.8:
                                is_new_hs = False
                                break

                        if is_new_hs:
                            this_id = hotspot_id_counter
                            hotspot_dict[this_id] = (lat_hs, lon_hs)
                            hotspot_id_counter += 1
                            print(f"[HOTSPOT‐new] ID={this_id}, gps=({lat_hs:.7f},{lon_hs:.7f})")
                        else:
                            print(f"[HOTSPOT‐dup] gps=({lat_hs:.7f},{lon_hs:.7f}) within 0.8 m of existing")

                    published_hotspot_history[((object_cx, object_cy), (self.frame_cx, self.frame_cy))] = 0
        
                # Always draw bounding box & centroid
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.circle(frame, (object_cx, object_cy), 5, (255, 255, 0), -1)

        # Draw image center for reference
        cv2.circle(frame, (self.frame_cx, self.frame_cy), 5, (255, 0, 0), -1)
        cv2.imshow("YOLO Detection", frame)
        cv2.waitKey(1)

        # === COOLDOWN CLEANUP: HOTSPOTS ===
        to_delete_h = []
        for key in list(published_hotspot_history):
            coord = key[0]
            if coord not in seen_coords_hotspots:
                published_hotspot_history[key] += 1
                if published_hotspot_history[key] >= self.cooldown_frames:
                    to_delete_h.append(key)
        for key in to_delete_h:
            del published_hotspot_history[key]

# === VEHICLE CONNECTION & GEofence / MISSION SETUP (unchanged except for final print) ===
def connect_vehicle():
    print("Connecting to vehicle...")
    conn = connect(CONNECTION_STRING, wait_ready=True)
    with vehicle_lock:
        global vehicle
        vehicle = conn
    print("Connected.")

def upload_geofence(vehicle, fence_coords):
    print("Uploading geofence…")
    vehicle._master.mav.param_set_send(
        vehicle._master.target_system,
        vehicle._master.target_component,
        b'FENCE_ACTION',
        1,
        mavutil.mavlink.MAV_PARAM_TYPE_INT32
    )

def shrink_polygon(polygon_coords, inset_m=1):
    poly = Polygon([(lon, lat) for lat, lon in polygon_coords])
    inset_deg = inset_m / 111139
    inner_poly = poly.buffer(-inset_deg)
    if inner_poly.is_empty:
        raise ValueError("Inset too large, no area left.")
    if inner_poly.geom_type == "MultiPolygon":
        inner_poly = max(inner_poly.geoms, key=lambda p: p.area)
    inner_coords = [(lat, lon) for lon, lat in inner_poly.exterior.coords]
    return inner_coords

def generate_zigzag_waypoints(polygon, spacing_m):
    poly = Polygon([(lon, lat) for lat, lon in polygon])
    min_lon, min_lat, max_lon, max_lat = poly.bounds
    lat_spacing = spacing_m / 111139
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
        current_lat += lat_spacing
    return waypoints

def upload_mission(veh, waypoints, altitude):
    cmds = veh.commands
    cmds.clear()
    time.sleep(1)
    home = veh.location.global_frame
    # Takeoff
    cmds.add(
        Command(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0,
            home.lat, home.lon, altitude
        )
    )
    # Survey waypoints
    for lat, lon in waypoints:
        cmds.add(
            Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 0, 0, 0, 0, 0,
                lat, lon, altitude
            )
        )
    cmds.upload()
    print("Mission uploaded.")

def arm_and_takeoff(veh, target_alt):
    while not veh.is_armable:
        print("Waiting for vehicle to become armable...")
        time.sleep(1)
    print("Arming...")
    veh.mode = VehicleMode("GUIDED")
    veh.armed = True
    while not veh.armed:
        time.sleep(1)
    print("Taking off...")
    veh.simple_takeoff(target_alt)
    while veh.location.global_relative_frame.alt < target_alt * 0.95:
        print(f"Altitude: {veh.location.global_relative_frame.alt:.2f}")
        time.sleep(1)
    print("Reached target altitude.")

def drone_mission_thread():
    global latest_target_pixel
    global vehicle

    # 1) Connect, geofence, shrink polygon
    connect_vehicle()
    upload_geofence(vehicle, polygon_coords)
    inner_polygon = shrink_polygon(polygon_coords, inset_m=1)

    # 2) Upload survey mission
    waypoints = generate_zigzag_waypoints(inner_polygon, ZIGZAG_SPACING)
    upload_mission(vehicle, waypoints, ALTITUDE)

    # 3) Arm & take off
    arm_and_takeoff(vehicle, ALTITUDE)
    
    with vehicle_lock:
        try:
            vehicle.parameters['WPNAV_SPEED'] = 500  # cm/s → 5.0 m/s
        except Exception as e:
            print(f"WARNING: could not set WPNAV_SPEED: {e}")
        vehicle.airspeed = 5.0  # m/s

    print("Switching to AUTO (survey) at 5.0 m/s…")
    
    vehicle.mode = VehicleMode("AUTO")

    # 4) Monitor for detected target GPS (same as before)
    while True:
        with gps_lock:
            if detected_targets_gps:
                lat_target, lon_target = detected_targets_gps.pop(-1)
                print(f"Diverting to target: ({lat_target:.7f}, {lon_target:.7f})")

                # 4a) GUIDED → move to above target
                vehicle.mode = VehicleMode("GUIDED")
                time.sleep(1)
                vehicle.simple_goto(LocationGlobalRelative(lat_target, lon_target, ALTITUDE))
                # Wait within 1.0 m horizontally
                while True:
                    cur = vehicle.location.global_relative_frame
                    horiz_dist = np.sqrt((cur.lat - lat_target)**2 + (cur.lon - lon_target)**2) * 111139
                    print(f"Distance to target: {horiz_dist:.2f} m")
                    if horiz_dist < 1.5:
                        print("Within 1.0 m horizontal of target.")
                        break
                    time.sleep(1)

                # 5) Descend to LOW_ALTITUDE
                print(f"Descending to {LOW_ALTITUDE} m for precise alignment.")
                vehicle.simple_goto(LocationGlobalRelative(lat_target, lon_target, LOW_ALTITUDE))
                while True:
                    cur_alt = vehicle.location.global_relative_frame.alt
                    if abs(cur_alt - LOW_ALTITUDE) < 0.3:
                        print(f"Reached {LOW_ALTITUDE} m altitude.")
                        break
                    time.sleep(0.5)

                # 6) Precise centering using PID
                pid_x = StablePID(Kp=0.15, Ki=0.002, Kd=0.02, max_output=0.2)
                pid_y = StablePID(Kp=0.15, Ki=0.002, Kd=0.02, max_output=0.2)
                pid_x.reset()
                pid_y.reset()
                frame_cx, frame_cy = 640 // 2, 480 // 2
                alignment_threshold_px = 10
                last_detection_time = time.time()
                detection_timeout = 1

                print("Starting centering loop at 10 m altitude.")
                while True:
                    with pixel_lock:
                        pixel = latest_target_pixel

                    if pixel is None or (time.time() - last_detection_time > detection_timeout):
                        print("Lost target or timed out during centering.")
                        send_body_velocity(0, 0, 0)
                        break

                    last_detection_time = time.time()
                    object_cx, object_cy = pixel
                    error_x = object_cx - frame_cx
                    error_y = object_cy - frame_cy
                    print(f"Pixel error: ex={error_x}, ey={error_y}")

                    x = abs(error_x)
                    y = abs(error_y)

                    if x <= alignment_threshold_px and y <= alignment_threshold_px:
                        print("Within deadband → zeroing PID and stopping.")
                        pid_x.reset()
                        pid_y.reset()
                        for _ in range(5):
                            send_body_velocity(0, 0, 0)
                            time.sleep(0.02)
                        with pixel_lock:
                            latest_target_pixel = None
                        print("Target centered and stopped.")
                        break

                    vx_cmd = pid_y.update(error_y)
                    vy_cmd = -pid_x.update(error_x)
                    print(f"Sending body-frame velocity vx={vx_cmd:.3f}, vy={vy_cmd:.3f}")

                    small_vel_thresh = 0.04
                    small_vel_count = 0
                    small_vel_max = 5

                    if abs(vx_cmd) < small_vel_thresh and abs(vy_cmd) < small_vel_thresh:
                        small_vel_count += 1
                    else:
                        small_vel_count = 0
                    if small_vel_count >= small_vel_max:
                        print(f"Velocities ≈0 for {small_vel_max} loops → ending centering.")
                        pid_x.reset()
                        pid_y.reset()
                        for _ in range(5):
                            send_body_velocity(0, 0, 0)
                            time.sleep(0.02)
                        with pixel_lock:
                            latest_target_pixel = None
                        break

                    send_body_velocity(-vx_cmd, -vy_cmd, 0)
                    time.sleep(0.1)

                # 7) Hover 10 seconds
                print("Hovering above target for 10 seconds.")
                send_body_velocity(0, 0, 0)
                time.sleep(10)

                # 8) Ascend back to ALTITUDE
                print(f"Ascending back to {ALTITUDE} m.")
                vehicle.simple_goto(LocationGlobalRelative(lat_target, lon_target, ALTITUDE))
                while True:
                    cur_alt = vehicle.location.global_relative_frame.alt
                    if abs(cur_alt - ALTITUDE) < 0.5:
                        print(f"Reached {ALTITUDE} m altitude.")
                        break
                    time.sleep(0.5)

                # Mark this target as serviced so the detector ignores it from now on
                with target_lock:
                    target_serviced = True

                # Switch back to AUTO to resume survey
                print("Target service complete. Resuming survey in AUTO mode.")
                with vehicle_lock:
                    try:
                        vehicle.parameters['WPNAV_SPEED'] = 500  # cm/s → 5.0 m/s
                    except Exception as e:
                        print(f"WARNING: could not set WPNAV_SPEED: {e}")
                    vehicle.airspeed = 5.0  # m/s

                print("Switching to AUTO and resuming (survey) at 5.0 m/s…")
                vehicle.mode = VehicleMode("AUTO")
                continue  # resume monitoring loop

        # If survey mission is done (no more waypoints), break
        if vehicle.commands.next >= vehicle.commands.count:
            break
        time.sleep(1)

    # If no more waypoints or no target found → RTL
    print("Survey complete. Returning to launch…")
    vehicle.mode = VehicleMode("RTL")
    cv2.destroyAllWindows()  # close OpenCV window to stop further detection
    while vehicle.armed:
        time.sleep(1)
    print("Mission complete (RTL).")

    # === AT THE END: print hotspot dictionary and count ===
    with hotspot_lock:
        print("Hotspot dictionary collected during mission:")
        for hs_id, (lat_h, lon_h) in hotspot_dict.items():
            print(f"  ID {hs_id}: ({lat_h:.7f}, {lon_h:.7f})")
        print(f"Total unique hotspots detected: {len(hotspot_dict)}")

# === MAIN ENTRYPOINT ===
def yolo_thread():
    rclpy.init()
    detector = YOLODetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    thread1 = threading.Thread(target=yolo_thread, daemon=True)
    thread2 = threading.Thread(target=drone_mission_thread)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    cv2.destroyAllWindows()
    print("All threads completed.")
