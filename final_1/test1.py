import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import numpy as np
import asyncio
from collections import deque
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from mavsdk.mission import MissionItem, MissionPlan

payload = False

# Global parameters
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
net = cv2.dnn.readNet("v4 tiny custom/yolov4-tiny-custom_best.weights", "v4 tiny custom/yolov4-tiny-custom.cfg")
with open("v4 tiny custom/obj.names", "r") as f:
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


async def create_mission(drone, speed, waypoints):
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
    """Detect red object and return its center coordinates"""
    if frame is None:
        return None, None, False
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color detection (adjust for your target)
    lower_red = np.array([100, 150, 50])
    upper_red = np.array([130, 255, 255])
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
    
    global payload

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
    payload = True
    await asyncio.sleep(0.5)  # Give time for mode switch
    
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
    
    # Create and upload mission
    await create_mission(drone, 5.0, waypoints)
    
    # Arm drone
    print("-- Arming")
    await drone.action.arm()
    await asyncio.sleep(1.0)
    
    # Takeoff
    print("🛫 Taking off...")
    await drone.action.set_takeoff_altitude(10.0)
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
    # Define waypoints: (latitude, longitude, altitude_in_meters)
    waypoints = [
        (23.176895691294206, 80.02217055884137, 7.0),
        (23.176983850254707, 80.02192271709133, 7.0)
    ]
    
    await run_mission(waypoints)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        if camera:
            camera.release()
        cv2.destroyAllWindows()