import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import numpy as np
import asyncio
from collections import deque
from mavsdk import System, action
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from mavsdk.mission import MissionItem, MissionPlan
import wp_cmd

connection_string = "udp://:14540"
# connection_string = "serial:///dev/ttyACM0:9600"
altitude = 10.0  # meters
speed = 10.0  # m/s
gripper_servo_channel = 1  # Adjust as needed

geofence_coords = []

survey_coords = [
    (23.177053285530338, 80.02196321569755),
    (23.176676027386808, 80.02133826101728),
    (23.175734110255565, 80.02166280829759),
    (23.175891918823606, 80.02249429306535),
    (23.176944793101267, 80.02219120345646)
]

sweep_spacing = 10.0
survey_inset = 5.0
target_class = "hotspot"  # Change as needed

weights = "v4 tiny custom/yolov4-tiny-custom_best.weights"
config = "v4 tiny custom/yolov4-tiny-custom.cfg"
names = "v4 tiny custom/obj.names"

# -----------------------------Dont edit below this line-----------------------------

# Global parameters

payload = False

image_width = 0
image_height = 0
center_x = 0
center_y = 0
dead_zone_x = 0
dead_zone_y = 0

# Moving average filter buffers
x_buffer = deque(maxlen=5)
y_buffer = deque(maxlen=5)

# YOLO model loading
net = cv2.dnn.readNet(weights, config)
with open(names, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def init_camera(camera_id=0):
    """Initialize camera with proper error checking"""
    global image_width, image_height, center_x, center_y, dead_zone_x, dead_zone_y
    
    print(f"📸 Attempting to open camera {camera_id}...")
    camera = cv2.VideoCapture(camera_id)

    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    # Test read a frame
    print("📸 Testing camera frame capture...")
    ret, test_frame = camera.read()
    
    if not ret or test_frame is None:
        print("❌ Camera opened but cannot read frames")
        camera.release()
        return None
    
    image_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = image_width // 2
    center_y = image_height // 2
    dead_zone_x = int(image_width * 0.1)
    dead_zone_y = int(image_height * 0.1)
    
    print(f"✅ Camera initialized: {image_width}x{image_height}")
    print(f"   Center: ({center_x}, {center_y})")
    print(f"   Dead zone: ±{dead_zone_x}x{dead_zone_y} pixels")
    
    return camera

camera = init_camera(camera_id=0)

async def connect_drone():
    """Connect to drone and setup"""
    drone = System()
    await drone.connect(system_address=connection_string)
    print("🔗 Connecting to drone...")
    
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Drone connected!")
            break
    
    print("⏳ Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ Position ready.")
            break
    
    return drone


async def create_mission(drone, waypoints):
    """Create and upload mission from waypoint list"""
    mission_items = []
    
    for i, (lat, lon, alt) in enumerate(waypoints):
        mission_items.append(MissionItem(
            lat,
            lon,
            alt,
            speed,  # Speed in m/s
            True,  # is_fly_through (don't stop at waypoint)
            float('nan'),  # gimbal_pitch_deg
            float('nan'),  # gimbal_yaw_deg
            MissionItem.CameraAction.NONE,
            float('nan'),  # loiter_time_s
            float('nan'),  # camera_photo_interval_s
            float('nan'),  # acceptance_radius_m
            float('nan'),  # yaw_deg
            float('nan'),   # camera_photo_distance_m
            MissionItem.VehicleAction.NONE
        ))
    
    mission_plan = MissionPlan(mission_items)
    
    print(f"📤 Uploading mission with {len(waypoints)} waypoints...")
    await drone.mission.upload_mission(mission_plan)
    print("✅ Mission uploaded successfully")
    await asyncio.sleep(0.5)  # Give time for mission to be processed


def detect_target(frame):
    """
    Detect the specified target class using YOLOv4.
    Returns (target_x, target_y, found)
    where target_x, target_y are offsets from image center in pixels.
    """

    if frame is None:
        return None, None, False

    height, width, _ = frame.shape

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Run detection
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the boxes and labels on the frame
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Sending target info
            if label == target_class:
                x, y, w, h = boxes[i]
                target_x = x + w // 2
                target_y = y + h // 2

                return target_x, target_y, True

    return None, None, False


def apply_moving_average(target_x, target_y):
    """Apply moving average filter to smooth coordinates"""
    x_buffer.append(target_x)
    y_buffer.append(target_y)
    
    filtered_x = int(np.mean(x_buffer))
    filtered_y = int(np.mean(y_buffer))
    
    return filtered_x, filtered_y


def draw_overlay(frame, target_x, target_y, detected, mode="MISSION", extra_text=""):
    """Draw center crosshair, dead zone, and status"""
    if frame is None:
        return None
    
    # Draw center crosshair
    cv2.line(frame, (center_x - 20, center_y), 
             (center_x + 20, center_y), (0, 255, 255), 2)
    cv2.line(frame, (center_x, center_y - 20), 
             (center_x, center_y + 20), (0, 255, 255), 2)
    
    # Draw dead zone rectangle
    cv2.rectangle(frame, 
                  (center_x - dead_zone_x, center_y - dead_zone_y),
                  (center_x + dead_zone_x, center_y + dead_zone_y),
                  (255, 178, 0), 2)
    
    # Display mode
    cv2.putText(frame, f"MODE: {mode}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display status
    status = "TARGET DETECTED" if detected else "SEARCHING..."
    color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, status, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Extra text
    if extra_text:
        cv2.putText(frame, extra_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return frame


async def center_on_target(drone, camera):
    """
    Detect and center on target in offboard mode.
    Once centered:
        1️⃣ Hover for 5 seconds
        2️⃣ Descend by 5 m (using feedback)
        3️⃣ Flip servo
        4️⃣ Ascend back to original altitude
    """

    print("🎯 Target detected! Switching to offboard mode...")

    global payload

    # Set initial setpoint before starting offboard
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
        print("🟢 Offboard mode started")
    except OffboardError as error:
        print(f"❌ Offboard start failed: {error._result.result}")
        return False

    centered_count = 0
    required_centered_frames = 10
    max_centering_attempts = 200
    attempt = 0

    print("🔄 Centering on target...")

    while centered_count < required_centered_frames and attempt < max_centering_attempts:
        attempt += 1
        ret, frame = camera.read()
        if not ret or frame is None:
            await asyncio.sleep(0.05)
            continue

        target_x, target_y, detected = detect_target(frame)

        if detected:
            filtered_x, filtered_y = apply_moving_average(target_x, target_y)
            offset_x = filtered_x - center_x
            offset_y = filtered_y - center_y

            if abs(offset_x) <= dead_zone_x and abs(offset_y) <= dead_zone_y:
                centered_count += 1
                print(f"✅ Centered ({centered_count}/{required_centered_frames})")
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            else:
                centered_count = 0
                yaw_gain = 0.1
                forward_gain = 0.001

                yaw_rate = -offset_x * yaw_gain
                forward_speed = -offset_y * forward_gain

                yaw_rate = np.clip(yaw_rate, -30.0, 30.0)
                forward_speed = np.clip(forward_speed, -1.0, 1.0)

                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(forward_speed, 0.0, 0.0, yaw_rate)
                )
        else:
            centered_count = 0
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

        frame = draw_overlay(frame, target_x, target_y, detected, "OFFBOARD-CENTERING")
        cv2.imshow("Drone Mission with Target Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        await asyncio.sleep(0.05)

    if centered_count < required_centered_frames:
        print("⚠️ Centering timeout - continuing anyway")

    print("✅ Target centered! Hovering for 5 seconds...")
    hover_duration = 5.0
    hover_start = asyncio.get_event_loop().time()

    while (asyncio.get_event_loop().time() - hover_start) < hover_duration:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(0.05)

    # Get current altitude
    async for pos in drone.telemetry.position():
        current_alt = pos.relative_altitude_m
        break

    target_alt_down = current_alt - 5.0  # descend target
    print(f"⬇️ Descending from {current_alt:.2f}m to {target_alt_down:.2f}m...")

    # Descend using feedback
    while True:
        async for pos in drone.telemetry.position():
            alt_now = pos.relative_altitude_m
            break

        if alt_now <= target_alt_down + 0.2:
            print(f"✅ Reached target descent altitude: {alt_now:.2f}m")
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            break

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.5, 0.0))
        await asyncio.sleep(0.1)

    # --- Trigger servo ---
    print("🔧 Triggering servo action...")
    await drone.action.set_actuator(gripper_servo_channel, 1.0)
    await asyncio.sleep(1.0)
    await drone.action.set_actuator(gripper_servo_channel, 0.0)
    print("✅ Servo action complete")

    hover_start = asyncio.get_event_loop().time()

    while (asyncio.get_event_loop().time() - hover_start) < hover_duration:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(0.05)

    # Ascend back
    print(f"⬆️ Ascending back to {current_alt:.2f}m...")
    while True:
        async for pos in drone.telemetry.position():
            alt_now = pos.relative_altitude_m
            break

        if alt_now >= current_alt - 0.2:
            print(f"✅ Reached ascent target: {alt_now:.2f}m")
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            break

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, -0.5, 0.0))
        await asyncio.sleep(0.1)

    # Stop offboard and continue mission
    print("🟡 Exiting offboard mode and resuming mission...")
    await drone.offboard.stop()
    payload = True
    await asyncio.sleep(0.5)
    print("✅ Returning to mission mode")

    return True

async def monitor_mission_with_detection(drone, camera):
    """Monitor mission and look for targets - FIXED VERSION"""
    print("👀 Monitoring mission for targets...")
    
    global payload
    target_found = False
    mission_complete = False
    last_progress = None
    
    # Create a background task for mission monitoring
    async def track_mission_progress():
        nonlocal mission_complete, last_progress
        async for progress in drone.mission.mission_progress():
            last_progress = progress
            if progress.current >= progress.total and progress.total > 0:
                mission_complete = True
                print(f"✅ Mission completed! ({progress.current}/{progress.total})")
                break
    
    # Start mission progress tracker
    progress_task = asyncio.create_task(track_mission_progress())
    
    try:
        # Vision processing loop - runs independently
        while not mission_complete:
            # Read frame (non-blocking)
            ret, frame = camera.read()
            
            if not ret or frame is None:
                await asyncio.sleep(0.05)
                continue
            
            # Detect target
            target_x, target_y, detected = detect_target(frame)
            
            # Draw overlay with progress info
            progress_text = ""
            if last_progress:
                progress_text = f"WP: {last_progress.current}/{last_progress.total}"
            
            frame = draw_overlay(frame, target_x, target_y, detected, "MISSION", progress_text)
            cv2.imshow("Drone Mission with Target Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 User quit")
                break
            
            # If target detected and not yet processed
            if detected and not target_found and not payload:
                target_found = True
                print("🎯 TARGET FOUND! Pausing mission...")
                
                # Cancel progress monitoring temporarily
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
                
                # Pause mission
                await drone.mission.pause_mission()
                await asyncio.sleep(1.0)  # Wait for pause to take effect
                
                # Center on target and hover
                success = await center_on_target(drone, camera)
                
                if success:
                    print("🚀 Resuming mission...")
                    
                    # Clear buffers
                    x_buffer.clear()
                    y_buffer.clear()
                    
                    # Resume mission
                    await drone.mission.start_mission()
                    await asyncio.sleep(0.5)
                    
                    # Restart progress monitoring
                    progress_task = asyncio.create_task(track_mission_progress())
                    
                    # Reset flag for next target
                    target_found = False
                else:
                    print("❌ Centering failed or interrupted")
                    break
            
            # Yield control to other async tasks
            await asyncio.sleep(0.02)  # 50 Hz loop - faster response
    
    finally:
        # Cleanup progress task
        if not progress_task.done():
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass


async def run_mission(waypoints):
    """Main function to run mission with target detection"""

    if camera is None:
        print("❌ Cannot proceed without camera")
        return
    
    # Connect to drone
    drone = await connect_drone()
    
    try:
        await drone.mission.clear_mission()
        print("Mission cleared successfully.")
    except Exception as e:
        print(f"Failed to clear mission: {e}")

    # Create and upload mission
    await create_mission(drone, waypoints)
    
    # Arm drone
    print("-- Arming")
    await drone.action.arm()
    await asyncio.sleep(1.0)
    
    # Takeoff
    print("🛫 Taking off...")
    await drone.action.set_takeoff_altitude(altitude=altitude)
    await drone.action.takeoff()
    
    await asyncio.sleep(10)
    
    # Start mission
    print("🚀 Starting mission...")
    await drone.mission.start_mission()
    await asyncio.sleep(1.0)
    
    # Monitor mission with target detection
    await monitor_mission_with_detection(drone, camera)
    
    # Return to launch
    print("🏠 Returning to launch...")
    await drone.action.return_to_launch()
    
    await asyncio.sleep(15)  # Wait for RTL to complete
    
    try:
        await drone.mission.clear_mission()
        print("Mission cleared successfully.")
    except Exception as e:
        print(f"Failed to clear mission: {e}")
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()


async def main():
    """Entry point"""
    print("🗺️ Generating survey waypoints...")
    boundary = wp_cmd.create_inset_boundary(survey_coords, survey_inset)
    waypoints = wp_cmd.generate_survey_waypoints(boundary, sweep_spacing, altitude)
    wp_cmd.plot_mission(survey_coords, boundary, waypoints)

    await run_mission(waypoints)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        if camera:
            camera.release()
        cv2.destroyAllWindows()