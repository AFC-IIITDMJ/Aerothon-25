import threading
import time
import numpy as np
from shapely.geometry import Polygon, LineString
from dronekit import connect, VehicleMode, Command, LocationGlobalRelative
from pymavlink import mavutil
from pid_controller import StablePID
from coordinates import pixel_to_gps, haversine_distance


class DroneMission:
    """
    Manages the drone mission: connection, geofence, waypoint generation, mission execution, and dynamic target diversion.
    """

    def __init__(self, shared_state):
        """
        Initialize DroneMission with shared state and constants.

        Args:
            shared_state (dict): Shared variables and locks used by both detector and mission.
        """
        self.shared = shared_state
        self.connected = False

        # Retrieve constants from shared state
        self.ALTITUDE = self.shared['ALTITUDE']
        self.LOW_ALTITUDE = self.shared['LOW_ALTITUDE']
        self.ZIGZAG_SPACING = self.shared['ZIGZAG_SPACING']
        self.POLYGON = self.shared['GEOFENCE_POLYGON']
        self.CONN_STR = self.shared['CONNECTION_STRING']

    def connect_vehicle(self):
        """
        Establish a DroneKit connection to the vehicle.
        """
        print("Connecting to vehicle...")
        conn = connect(self.CONN_STR, wait_ready=True)
        with self.shared['vehicle_lock']:
            self.shared['vehicle'] = conn
        self.connected = True
        print("Connected to vehicle.")

    def upload_geofence(self):
        """
        Enable geofence on the vehicle (simple parameter set).
        """
        veh = self.shared['vehicle']
        veh._master.mav.param_set_send(
            veh._master.target_system,
            veh._master.target_component,
            b'FENCE_ACTION',
            1,
            mavutil.mavlink.MAV_PARAM_TYPE_INT32
        )
        print("Geofence enabled.")

    def shrink_polygon(self, inset_m: float = 1.0) -> list:
        """
        Shrink the survey polygon by inset_m meters using Shapely.

        Args:
            inset_m (float): Buffer distance in meters to shrink.

        Returns:
            list: List of (lat, lon) tuples for the shrunken polygon.
        """
        poly = Polygon([(lon, lat) for lat, lon in self.POLYGON])
        inset_deg = inset_m / 111139.0
        inner_poly = poly.buffer(-inset_deg)
        if inner_poly.is_empty:
            raise ValueError("Inset too large, no area left.")
        if inner_poly.geom_type == 'MultiPolygon':
            inner_poly = max(inner_poly.geoms, key=lambda p: p.area)
        return [(lat, lon) for lon, lat in inner_poly.exterior.coords]

    def generate_zigzag_waypoints(self, polygon: list) -> list:
        """
        Generate a zigzag (boustrophedon) survey pattern within a polygon.

        Args:
            polygon (list): List of (lat, lon) tuples defining the area.

        Returns:
            list: List of (lat, lon) waypoints.
        """
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

    def upload_mission(self, waypoints: list):
        """
        Upload a mission: takeoff to ALTITUDE, then visit each waypoint at ALTITUDE.

        Args:
            waypoints (list): List of (lat, lon) waypoints.
        """
        veh = self.shared['vehicle']
        cmds = veh.commands
        cmds.clear()
        time.sleep(1)

        home = veh.location.global_frame
        # Takeoff command
        cmds.add(
            Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0,
                home.lat, home.lon, self.ALTITUDE
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
                    lat, lon, self.ALTITUDE
                )
            )
        cmds.upload()
        print("Mission uploaded.")

    def arm_and_takeoff(self):
        """
        Arms the vehicle and takes off to ALTITUDE.
        """
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
        print("Reached target altitude.")

    def send_body_velocity(self, vx: float, vy: float, vz: float = 0.0) -> None:
        """
        Send a MAVLink SET_POSITION_TARGET_LOCAL_NED message in the body frame.
        """
        with self.shared['vehicle_lock']:
            veh = self.shared['vehicle']
            if veh is None:
                return
            msg = veh.message_factory.set_position_target_local_ned_encode(
                0,
                0, 0,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000111111000111,
                0, 0, 0,
                vx, vy, vz,
                0, 0, 0,
                0, 0
            )
            veh.send_mavlink(msg)
            veh.flush()

    def execute_mission(self) -> None:
        """
        Orchestrates the full mission, including survey and dynamic target diversion.
        """
        # 1) Connect, geofence, shrink polygon
        self.connect_vehicle()
        self.upload_geofence()
        inner_poly = self.shrink_polygon(inset_m=1.0)

        # 2) Generate survey waypoints & upload mission
        survey_wps = self.generate_zigzag_waypoints(inner_poly)
        self.upload_mission(survey_wps)

        # 3) Arm and takeoff
        self.arm_and_takeoff()
        with self.shared['vehicle_lock']:
            veh = self.shared['vehicle']
            try:
                veh.parameters['WPNAV_SPEED'] = 500  # cm/s → 5 m/s
            except Exception as e:
                print(f"WARNING: could not set WPNAV_SPEED: {e}")
            veh.airspeed = 5.0
        print("Switching to AUTO mode for survey...")
        veh.mode = VehicleMode("AUTO")

        # 4) Survey loop with dynamic diversion
        while True:
            with self.shared['gps_lock']:
                if self.shared['detected_targets_gps']:
                    lat_t, lon_t = self.shared['detected_targets_gps'].pop(-1)
                    print(f"Diverting to target: ({lat_t:.7f}, {lon_t:.7f})")

                    # 4a) Switch to GUIDED and go above target
                    veh.mode = VehicleMode("GUIDED")
                    time.sleep(1)
                    veh.simple_goto(LocationGlobalRelative(lat_t, lon_t, self.ALTITUDE))
                    while True:
                        cur = veh.location.global_relative_frame
                        horiz_dist = np.sqrt((cur.lat - lat_t)**2 + (cur.lon - lon_t)**2) * 111139.0
                        print(f"Distance to target: {horiz_dist:.2f} m")
                        if horiz_dist < 1.5:
                            print("Within 1.5 m horizontally.")
                            break
                        time.sleep(1)

                    # 4b) Descend to LOW_ALTITUDE
                    print(f"Descending to low altitude: {self.LOW_ALTITUDE} m.")
                    veh.simple_goto(LocationGlobalRelative(lat_t, lon_t, self.LOW_ALTITUDE))
                    while True:
                        if abs(veh.location.global_relative_frame.alt - self.LOW_ALTITUDE) < 0.3:
                            print(f"Reached {self.LOW_ALTITUDE} m.")
                            break
                        time.sleep(0.5)

                    # 5) PID-based centering
                    pid_x = StablePID(0.15, 0.002, 0.02, max_output=0.2)
                    pid_y = StablePID(0.15, 0.002, 0.02, max_output=0.2)
                    pid_x.reset()
                    pid_y.reset()
                    frame_cx, frame_cy = 640 // 2, 480 // 2
                    threshold = 10
                    last_detect = time.time()
                    timeout = 1.0

                    print("Starting centering loop...")
                    while True:
                        with self.shared['pixel_lock']:
                            pix = self.shared['latest_target_pixel']
                        if pix is None or (time.time() - last_detect) > timeout:
                            print("Lost or timed out on target.")
                            self.send_body_velocity(0, 0, 0)
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
                                self.send_body_velocity(0, 0, 0)
                                time.sleep(0.02)
                            with self.shared['pixel_lock']:
                                self.shared['latest_target_pixel'] = None
                            break
                        vx = pid_y.update(ey)
                        vy = -pid_x.update(ex)
                        print(f"Velocity cmd: vx={vx:.3f}, vy={vy:.3f}")

                        if abs(vx) < 0.04 and abs(vy) < 0.04:
                            print("Velocities near zero. Finishing centering.")
                            pid_x.reset()
                            pid_y.reset()
                            for _ in range(5):
                                self.send_body_velocity(0, 0, 0)
                                time.sleep(0.02)
                            with self.shared['pixel_lock']:
                                self.shared['latest_target_pixel'] = None
                            break

                        self.send_body_velocity(-vx, -vy, 0)
                        time.sleep(0.1)

                    # 6) Hover and ascend
                    print("Hovering at low altitude (10 s)...")
                    self.send_body_velocity(0, 0, 0)
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

            # Check for mission completion
            if self.shared['vehicle'].commands.next >= self.shared['vehicle'].commands.count:
                break
            time.sleep(1)

        # Survey complete → RTL
        print("Survey complete. RTL.")
        self.shared['vehicle'].mode = VehicleMode("RTL")
        while self.shared['vehicle'].armed:
            time.sleep(1)

        # Print hotspot summary
        with self.shared['hotspot_lock']:
            print("Hotspot dictionary:")
            for hid, (lat_h, lon_h) in self.shared['hotspot_dict'].items():
                print(f"  ID {hid}: ({lat_h:.7f}, {lon_h:.7f})")
            print(f"Total hotspots: {len(self.shared['hotspot_dict'])}")
