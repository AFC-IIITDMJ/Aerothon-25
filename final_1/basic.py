import cv2
import numpy as np
import asyncio
from collections import deque
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from mavsdk.mission import MissionItem, MissionPlan

# Global parameters
image_width = 0
image_height = 0
center_x = 0
center_y = 0
dead_zone_x = 0
dead_zone_y = 0

# Detection Parameters
weights_path = "v4 tiny custom/yolov4-tiny-custom_best.weights"
config_path = "v4 tiny custom/yolov4-tiny-custom.cfg"
names_path = "v4 tiny custom/obj.names"
confidence_threshold = 0.3
nms_threshold = 0.4
input_size = (320, 320)
classes_of_interest = ['hotspot']

# Moving average filter buffers
x_buffer = deque(maxlen=5)
y_buffer = deque(maxlen=5)


def init_camera(camera_id=0):
    """Initialize camera with proper error checking"""
    global image_width, image_height, center_x, center_y, dead_zone_x, dead_zone_y
    
    print(f"📸 Attempting to open camera {camera_id}...")
    camera = cv2.VideoCapture(camera_id)
    
    if not camera.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
    
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


async def connect_drone():
    """Connect to drone and setup"""
    drone = System()
    await drone.connect(system_address="udp://:14540")
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


async def create_mission(drone,speed, waypoints):
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


def detect_target(frame):
    """Detect red object and return its center coordinates"""
    if frame is None:
        return None, None, False
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color detection (adjust for your target)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        
        # Only consider if area is significant
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            target_x = x + w // 2
            target_y = y + h // 2
            
            # Draw detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)
            
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
    """Enter offboard mode and center on target, then hover for 10 seconds"""
    print("🎯 Target detected! Switching to offboard mode...")
    
    # Set initial velocity setpoint
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    
    # Start offboard mode
    try:
        await drone.offboard.start()
        print("🟢 Offboard mode started")
    except OffboardError as error:
        print(f"❌ Offboard start failed: {error._result.result}")
        return False
    
    # Centering loop
    centered_count = 0
    required_centered_frames = 10
    
    print("🔄 Centering on target...")
    
    max_centering_attempts = 200  # Timeout after 10 seconds (200 * 0.05)
    attempt = 0
    
    while centered_count < required_centered_frames and attempt < max_centering_attempts:
        attempt += 1
        
        ret, frame = camera.read()
        
        if not ret or frame is None:
            print("⚠️ Failed to read frame during centering")
            await asyncio.sleep(0.05)
            continue
        
        # Detect target
        target_x, target_y, detected = detect_target(frame)
        
        if detected:
            # Apply moving average filter
            filtered_x, filtered_y = apply_moving_average(target_x, target_y)
            
            # Calculate offset from center
            offset_x = filtered_x - center_x
            offset_y = filtered_y - center_y
            
            # Check if target is centered
            if abs(offset_x) <= dead_zone_x and abs(offset_y) <= dead_zone_y:
                centered_count += 1
                print(f"✅ Centered ({centered_count}/{required_centered_frames})")
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
            else:
                centered_count = 0
                
                # Calculate velocities
                yaw_gain = 0.1
                forward_gain = 0.001
                
                yaw_rate = -offset_x * yaw_gain
                forward_speed = -offset_y * forward_gain
                
                # Clip velocities
                yaw_rate = np.clip(yaw_rate, -30.0, 30.0)
                forward_speed = np.clip(forward_speed, -1.0, 1.0)
                
                print(f"🎯 Centering... Offset: ({offset_x:4d}, {offset_y:4d})")
                
                # Send velocity command
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(forward_speed, 0.0, 0.0, yaw_rate)
                )
        else:
            print("⚠️ Target lost during centering")
            centered_count = 0
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
        
        # Draw overlay
        frame = draw_overlay(frame, target_x, target_y, detected, "OFFBOARD-CENTERING")
        cv2.imshow("Drone Mission with Target Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        
        await asyncio.sleep(0.05)
    
    if centered_count < required_centered_frames:
        print("⚠️ Centering timeout - continuing anyway")
    
    # Hover for 10 seconds
    print("✅ Target centered! Hovering for 10 seconds...")
    
    hover_start = asyncio.get_event_loop().time()
    hover_duration = 10.0
    
    while (asyncio.get_event_loop().time() - hover_start) < hover_duration:
        ret, frame = camera.read()
        
        if ret and frame is not None:
            remaining = hover_duration - (asyncio.get_event_loop().time() - hover_start)
            
            target_x, target_y, detected = detect_target(frame)
            frame = draw_overlay(frame, target_x, target_y, detected, "HOVERING", 
                                f"Time: {remaining:.1f}s")
            
            cv2.imshow("Drone Mission with Target Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        
        # Maintain hover
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )
        
        await asyncio.sleep(0.05)
    
    print("✅ Hover complete! Stopping offboard mode...")
    
    # Stop offboard mode
    await drone.offboard.stop()
    
    print("✅ Returning to mission mode")
    
    return True


async def monitor_mission_with_detection(drone, camera):
    """Monitor mission and look for targets"""
    print("👀 Monitoring mission for targets...")
    
    target_found = False
    frame_count = 0
    
    # Create a task to read mission progress
    async def mission_progress_monitor():
        async for progress in drone.mission.mission_progress():
            return progress
    
    # Monitor loop
    while True:
        frame_count += 1
        
        # Read frame
        ret, frame = camera.read()
        
        if not ret or frame is None:
            print(f"⚠️ Frame {frame_count}: Failed to read")
            await asyncio.sleep(0.1)
            continue
        
        # Get mission progress (non-blocking check)
        try:
            progress_task = asyncio.create_task(mission_progress_monitor())
            progress = await asyncio.wait_for(progress_task, timeout=0.01)
            
            print(f"📍 Mission progress: {progress.current}/{progress.total}")
            
            if progress.current == progress.total:
                print("✅ Mission completed!")
                break
                
        except asyncio.TimeoutError:
            pass  # No new progress update yet
        
        # Detect target
        target_x, target_y, detected = detect_target(frame)
        
        # Draw overlay
        frame = draw_overlay(frame, target_x, target_y, detected, "MISSION")
        cv2.imshow("Drone Mission with Target Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 User quit")
            break
        
        # If target detected and not yet processed
        if detected and not target_found:
            target_found = True
            print("🎯 TARGET FOUND! Pausing mission...")
            
            # Pause mission
            await drone.mission.pause_mission()
            await asyncio.sleep(1)
            
            # Center on target and hover
            success = await center_on_target(drone, camera)
            
            if success:
                print("🚁 Resuming mission...")
                
                # Clear buffers
                x_buffer.clear()
                y_buffer.clear()
                
                # Resume mission
                await drone.mission.start_mission()
                
                # Reset flag for next target
                target_found = False
            else:
                print("❌ Centering failed or interrupted")
                break
        
        await asyncio.sleep(0.05)  # 20 Hz loop


async def run_mission(waypoints, camera_id=0):
    """Main function to run mission with target detection"""
    # Initialize camera
    camera = init_camera(camera_id)
    if camera is None:
        print("❌ Cannot proceed without camera")
        return
    
    # Test camera continuously for 3 seconds
    print("📸 Testing camera for 3 seconds...")
    test_start = asyncio.get_event_loop().time()
    frame_test_count = 0
    
    while (asyncio.get_event_loop().time() - test_start) < 3.0:
        ret, frame = camera.read()
        if ret and frame is not None:
            frame_test_count += 1
            cv2.imshow("Camera Test", frame)
            cv2.waitKey(1)
        await asyncio.sleep(0.033)  # ~30 fps
    
    cv2.destroyWindow("Camera Test")
    print(f"✅ Camera test: {frame_test_count} frames captured in 3 seconds")
    
    if frame_test_count < 10:
        print("❌ Camera not working properly")
        camera.release()
        return
    
    # Connect to drone
    drone = await connect_drone()
    
    # Create and upload mission
    await create_mission(drone,5.0, waypoints)
    
    # Arm drone
    print("-- Arming")
    await drone.action.arm()
    
    # Takeoff
    print("🛫 Taking off...")
    await drone.action.set_takeoff_altitude(10.0)
    await drone.action.takeoff()
    
    await asyncio.sleep(10)
    
    # Start mission
    print("🚁 Starting mission...")
    await drone.mission.start_mission()
    
    # Monitor mission with target detection
    await monitor_mission_with_detection(drone, camera)
    
    # Return to launch
    print("🏠 Returning to launch...")
    await drone.action.return_to_launch()
    
    await asyncio.sleep(10)
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()


async def main():
    """Entry point"""
    # Define waypoints: (latitude, longitude, altitude_in_meters)
    waypoints = [
            (23.176746009833696, 80.02200315931218,10.0),  # Waypoint 1
            (23.17625728678608, 80.02204303133941,10.0),  # Waypoint 2
            (23.176599393106795, 80.02156678212542,10.0),  # Waypoint 3
        ]
    
    await run_mission(waypoints, camera_id=0)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        cv2.destroyAllWindows()