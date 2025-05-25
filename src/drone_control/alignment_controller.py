from dronekit import Vehicle, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import numpy as np
from typing import Tuple, Optional
from src.utilities.pid_controller import StablePID

class AlignmentController:
    
    def __init__(self, shared_data, vehicle: Vehicle, pid_params: dict):
        self.vehicle = vehicle
        self.shared_data = shared_data
        self.y_pid = StablePID(**pid_params['y_pid'])
        self.x_pid = StablePID(**pid_params['x_pid'])
        self.aligned_count = 0
        self.last_bbox = None
        self.smooth_factor = 0.3
        self.required_aligned_cycles = 20
        self.last_vx = 0
        self.last_vy = 0
        self.max_slew_rate = 0.6
        self.last_update = time.time()
        self.landed = False
        self.frame_center = (320, 240)
    
    def alignment_loop(self) -> None:
        while not self.landed:
            self.update()
            print("Alignment loop running")
    
    def update(self) -> None:
        with self.shared_data.lock:
            bbox = self.shared_data.coords
        if not self.vehicle.armed or bbox is None or self.landed:
            print("Returning to AUTO mode")
            if self.vehicle.mode == VehicleMode("GUIDED"):
                self.vehicle.mode = VehicleMode("AUTO")
            return
        
        if self.vehicle.mode == VehicleMode("AUTO"):
            self.send_velocity(0, 0)
            self.vehicle.mode = VehicleMode("GUIDED")
        
        self.last_bbox = list(map(int, bbox.split(',')))
        x1, y1, x2, y2 = self.last_bbox
        bbox_cx = (x1 + x2) // 2
        bbox_cy = (y1 + y2) // 2
        coords = [bbox_cx, bbox_cy]
        
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        x_error = coords[0] - self.frame_center[0]
        y_error = coords[1] - self.frame_center[1]
        vy = self.y_pid.update(y_error)
        vx = self.x_pid.update(x_error)
        
        dvx = vx - self.last_vx
        dvy = vy - self.last_vy
        dvx = np.clip(dvx, -self.max_slew_rate * dt, self.max_slew_rate * dt)
        dvy = np.clip(dvy, -self.max_slew_rate * dt, self.max_slew_rate * dt)
        self.last_vx += dvx
        self.last_vy += dvy
        
        print(f"BBox: {self.last_bbox}, x_error={x_error}, y_error={y_error}, vx={self.last_vx}, vy={self.last_vy}")
        self.send_velocity(self.last_vx, self.last_vy)
        
        if abs(x_error) < 10 and abs(y_error) < 10:
            self.aligned_count += 1
            if self.aligned_count >= self.required_aligned_cycles:
                print(f"Descending to 10m...")
                self.vehicle.simple_goto(LocationGlobalRelative(
                    self.vehicle.location.global_relative_frame.lat,
                    self.vehicle.location.global_relative_frame.lon, 10))
                while abs(self.vehicle.location.global_relative_frame.alt - 10) > 1:
                    time.sleep(1)
                
                print(f"Hovering for 5s...")
                time.sleep(5)
                
                print(f"Ascending back to mission altitude...")
                self.vehicle.simple_goto(LocationGlobalRelative(
                    self.vehicle.location.global_relative_frame.lat,
                    self.vehicle.location.global_relative_frame.lon, 15))
                while abs(self.vehicle.location.global_relative_frame.alt - 15) > 1:
                    time.sleep(1)
                self.vehicle.mode = VehicleMode("RTL")
        else:
            self.aligned_count = 0
    
    def send_velocity(self, vn: float, ve: float) -> None:
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111, 0, 0, 0, vn, ve, 0, 0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()