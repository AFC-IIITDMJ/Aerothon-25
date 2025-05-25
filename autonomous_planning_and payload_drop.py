import threading
import time
import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
from pymavlink import mavutil
import cv2
from ultralytics import YOLO
model = YOLO(r"/home/wolf/runs/detect/train/weights/best.pt")


class SharedData:
    def __init__(self):
        self.lock=threading.Lock()
        self.coords=None
        self.frame_center=(320,240)


class CameraSubscriber(Node):
    def __init__(self,shared_data):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera',
            self.image_callback,
            10)
        self.shared_data=shared_data
        self.br=CvBridge()
        self.get_logger().info("Camera subscriber started")
        self.window_name="Gazebo Camera Feed"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)
    def image_callback(self,msg):
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            results = model(cv_image)[0]
            if len(results.boxes)==0:
                with self.shared_data.lock:
                    self.shared_data.coords=None
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if conf>=0.7:
                    cv2.rectangle(cv_image,(x1,y1),(x2,y2),(0,0,255),2)
                    label_text = f'{label} {conf:.2f}'
                    cv2.putText(cv_image, label_text, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(25,255,25),2)
                    bbox_msg = String()
                    bbox_msg.data = f'{x1},{y1},{x2},{y2}'
                    with self.shared_data.lock:
                        self.shared_data.coords=bbox_msg.data
                        print(self.shared_data.coords)
            cv2.imshow("YOLOv8 Detection", cv_image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.get_logger().info("Exiting...")
                self.destroy_node()
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            cv2.destroyAllWindows()
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

class StablePID:
    def __init__(self, Kp,Ki,Kd,max_output,tau=0.1):
        self.Kp=Kp
        self.Ki=Ki
        self.Kd=Kd
        self.max_output=max_output
        self.tau=tau
        self.reset()
    def reset(self):
        self.integral=0.0
        self.prev_error=0.0
        self.prev_derivative=0.0
        self.prev_time=time.time()
    def update(self,error):
        now=time.time()
        dt=now-self.prev_time
        if dt <= 0:
            dt = 1e-16
        self.prev_time=now
        self.integral+=error * dt
        self.integral=np.clip(self.integral,-1.0,1.0)
        raw_derivative=(error - self.prev_error) / dt
        derivative=(self.tau * self.prev_derivative + dt * raw_derivative) / (self.tau + dt)
        self.prev_derivative = derivative
        output=self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output=np.clip(output,-self.max_output, self.max_output)
        return output

def set_geofence(vehicle, coordinates, enable=True):
    master=vehicle._master
    target_system=master.target_system
    target_component=master.target_component
    master.mav.command_long_send(
        target_system,
        target_component,
        mavutil.mavlink.MAV_CMD_DO_FENCE_ENABLE,
        0,
        0,0,0,0,0,0,0)
    print("[*] Geofence disabled.")
    time.sleep(1)
    vehicle.parameters['FENCE_TOTAL']=len(coordinates)
    print(f"[*] Set FENCE_TOTAL to {len(coordinates)}")

    time.sleep(1)
    master.mav.mission_count_send(
        target_system,
        target_component,
        len(coordinates),
        mavutil.mavlink.MAV_MISSION_TYPE_FENCE
    )
    for idx, (lat, lon) in enumerate(coordinates):
        master.mav.mission_item_int_send(
            target_system,
            target_component,
            idx,
            mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
            mavutil.mavlink.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION,
            0,1,
            len(coordinates),
            0,0,0,
            int(lat * 1e7),
            int(lon * 1e7),
            0,
            mavutil.mavlink.MAV_MISSION_TYPE_FENCE)
        print(f"[*] Sent vertex {idx}: {lat}, {lon}")
        time.sleep(0.1)

    if enable:
        master.mav.command_long_send(
            target_system,
            target_component,
            mavutil.mavlink.MAV_CMD_DO_FENCE_ENABLE,
            0,
            1,0,0,0,0,0,0
        )
        print("[✔] Geofence enabled.")

def create_inner_polygon(outer_polygon, buffer_distance=-0.000045):
    inner = outer_polygon.buffer(buffer_distance)
    if inner.is_empty:
        print(f"Warning: Buffer distance {buffer_distance} resulted in empty geometry.")
        return None
    if not inner.is_valid:
        print("Attempting to fix invalid geometry...")
        inner = inner.buffer(0)
    return inner

def generate_lawnmower_path(polygon, line_spacing=0.0001):
    if polygon is None or polygon.is_empty:
        return []
    minx,miny,maxx,maxy=polygon.bounds
    lines=[]
    x= minx
    direction=True
    while x<=maxx:
        scan_line=LineString([(x, miny),(x, maxy)])
        clipped=scan_line.intersection(polygon)
        if not clipped.is_empty:
            if clipped.geom_type=="LineString":
                coords=list(clipped.coords)
                if not direction:
                    coords.reverse()
                lines.append(coords)
            elif clipped.geom_type=="MultiLineString":
                for line in clipped.geoms:
                    coords=list(line.coords)
                    if not direction:
                        coords.reverse()
                    lines.append(coords)
        x += line_spacing
        direction = not direction
    return lines

def plot_mission(geofence,inner_poly,waypoints):
    """Visualize geofence, buffer, and waypoints"""
    plt.figure(figsize=(12, 10))
    x, y = geofence.exterior.xy
    plt.plot(x, y, 'r-', label='Original Geofence')
    if inner_poly and not inner_poly.is_empty:
        if inner_poly.geom_type == 'Polygon':
            xi, yi = inner_poly.exterior.xy
            plt.plot(xi, yi, 'b--', label='Inner Buffer')
        elif inner_poly.geom_type == 'MultiPolygon':
            for poly in inner_poly.geoms:
                xi, yi = poly.exterior.xy
                plt.plot(xi, yi, 'b--')
    if waypoints:
        lats = [wp[0] for wp in waypoints]
        lons = [wp[1] for wp in waypoints]
        plt.plot(lons, lats, 'g-', linewidth=1.5, label='Flight Path')
        plt.plot(lons[0], lats[0], 'mo', markersize=10, label='Start')
        plt.plot(lons[-1], lats[-1], 'ko', markersize=10, label='End')
    plt.title('Drone Mission Planning')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def upload_mission(vehicle, waypoints, altitude=15):
    """Upload mission to vehicle"""
    cmds = vehicle.commands
    cmds.clear()
    cmds.wait_ready()
    print("Uploading mission...")
    cmds.add(Command(
        0,0,0,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,0,0,0,0,0,
        waypoints[0][0],waypoints[0][1],altitude))
    for lat, lon in waypoints:
        cmds.add(Command(
            0,0,0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0,0,0,0,0,0,
            lat,lon,altitude))
    cmds.add(Command(
        0,0,0,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,0,0,0,0,0,0,0,0))
    cmds.upload()
    print(f"Uploaded {len(waypoints)+2} mission items")

class AlignmentController:
    def __init__(self, shared_data,vehicle):
        self.vehicle =vehicle
        self.shared_data = shared_data
        self.y_pid = StablePID(0.12, 0.002, 0.02, 0.2)
        self.x_pid = StablePID(0.12, 0.002, 0.02, 0.2)
        self.aligned_count = 0
        self.last_bbox = None
        self.smooth_factor = 0.3
        self.required_aligned_cycles = 20
        self.last_vx = 0
        self.last_vy = 0
        self.max_slew_rate = 0.6
        self.last_update = time.time()
        self.landed=False
    def alignment_loop(self):
        while not self.landed:
            self.update()
            print("loop_started")

    def update(self):
        with self.shared_data.lock:
            bbox = self.shared_data.coords
            frame_center = self.shared_data.frame_center
        if not self.vehicle.armed or bbox is None or self.landed:
            print("**************Returning***************")
            if self.vehicle.mode==VehicleMode("GUIDED"):
                self.vehicle.mode=VehicleMode("AUTO")
            return
        print(bbox)

        if self.vehicle.mode==VehicleMode("AUTO"):
            self.send_velocity(0,0)
            self.vehicle.mode=VehicleMode("GUIDED")
        # msg = self.vehicle.message_factory.command_long_encode(
        #     0,0,
        #     mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE,
        #     0,
        #     0,0,0,0,0,0,0)
        # self.vehicle.send_mavlink(msg)
        #
        # # msg = vehicle.message_factory.command_long_encode(
        # #     0,0,
        # #     mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE,
        # #     0,
        # #     1,0,0,0,0,0,0)
        # # vehicle.send_mavlink(msg)

        self.last_bbox=list(map(int,bbox.split(',')))
        x1,y1,x2,y2=self.last_bbox
        bbox_cx=(x1 + x2) // 2
        bbox_cy=(y1 + y2) // 2
        coords=[bbox_cx,bbox_cy]
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        x_error=coords[0]-frame_center[0]
        y_error=coords[1]-frame_center[1]
        vy=self.y_pid.update(y_error)
        vx=self.x_pid.update(x_error)
        # dvx=self.smooth_factor*vx + (1-self.smooth_factor)*self.last_vx
        # dvy=self.smooth_factor*vy +(1-self.smooth_factor)*self.last_vy
        dvx=vx-self.last_vx
        dvy=vy-self.last_vy
        dvx=np.clip(dvx,-self.max_slew_rate*dt,self.max_slew_rate*dt)
        dvy=np.clip(dvy,-self.max_slew_rate*dt,self.max_slew_rate*dt)
        self.last_vx += dvx
        self.last_vy += dvy
        print(self.last_bbox)
        print(f"x_error={x_error},y_error={y_error},Vy={self.last_vy},vx={self.last_vx}")
        if y_error<0 and x_error<0:
            self.send_velocity(self.last_vx,self.last_vy )
        else:
            self.send_velocity(self.last_vx,self.last_vy)
        # elif y_error<0 and x_error>0:
        #     self.send_velocity(-self.last_vx,self.last_vy)
        # elif y_error>0 and x_error<0:
        #     self.send_velocity(self.last_vx,-self.last_vy)
        # else:
        #     self.send_velocity(self.last_vx,self.last_vy)
        # Check alignment
        if abs(x_error)<10 and abs(y_error)<10:
            self.aligned_count+=1
            if self.aligned_count>=self.required_aligned_cycles:
                print(f"Descending to {10}m...")
                self.vehicle.simple_goto(LocationGlobalRelative(
                    self.vehicle.location.global_relative_frame.lat,
                    self.vehicle.location.global_relative_frame.lon,
                    10))
                while abs(self.vehicle.location.global_relative_frame.alt - 10) > 1:
                    time.sleep(1)

                print(f"Hovering for {5}s...")
                time.sleep(5)

                print(f"Ascending back to mission altitude...")
                self.vehicle.simple_goto(LocationGlobalRelative(
                    self.vehicle.location.global_relative_frame.lat,
                    self.vehicle.location.global_relative_frame.lon,
                    15))
                while abs(self.vehicle.location.global_relative_frame.alt - 15)>1:
                    time.sleep(1)
                self.vehicle.mode=VehicleMode("RTL")
        else:
            self.aligned_count = 0

    def send_velocity(self,vn,ve):
            msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,
                0, 0, 0,
                vn, ve, 0,
                0, 0, 0,
                0, 0
            )
            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()

def spin_camera(shared_data):
    rclpy.init()
    node=CameraSubscriber(shared_data)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
print("Connecting to vehicle...")
vehicle = connect("127.0.0.1:14550", wait_ready=True)
geofence_coords = [
    (-35.36324697,149.16615638),
    (-35.36340159,149.16514499),
    (-35.36253012,149.16494645),
    (-35.36237550,149.16595783),
    (-35.36324697,149.16615638)]
geofence_poly=Polygon([(lon, lat) for lat, lon in geofence_coords])
inner_poly = create_inner_polygon(geofence_poly)
if not inner_poly or inner_poly.is_empty:
    print("Using fallback buffer distance")
    inner_poly = create_inner_polygon(geofence_poly, -0.00002)
path = generate_lawnmower_path(inner_poly)
waypoints = []
for line in path:
    for lon, lat in line:
        waypoints.append((lat, lon))
plot_mission(geofence_poly, inner_poly, waypoints)
set_geofence(vehicle, geofence_coords)
if waypoints:
    upload_mission(vehicle, waypoints)
print("Arming motors...")
vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True
while not vehicle.armed:
    print("- Waiting for arming...")
    time.sleep(1)
print(f"Taking off to {15} meters...")
vehicle.simple_takeoff(15)
timeout = 30
start_time = time.time()
while True:
    current_alt = vehicle.location.global_relative_frame.alt
    print(f"Current altitude: {current_alt:.1f} m")
    if current_alt >= 15 * 0.95:
        print("Reached target altitude!")
        break
    if time.time() - start_time > timeout:
        print("Takeoff timed out!")
        break

time.sleep(1)
print("Hovering for 5 seconds...")
time.sleep(5)
print("Starting mission...")
vehicle.mode = VehicleMode("AUTO")
# vehicle.close()
shared_data = SharedData()
landing_controller = AlignmentController(shared_data,vehicle)

print("procedding")
t2 = threading.Thread(target=landing_controller.alignment_loop)
t1=threading.Thread(target=spin_camera,args=(shared_data,))
t1.start()
t2.start()
t1.join()
t2.join()
vehicle.close()
