from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
import time

class VehicleManager:
    
    def __init__(self, connection_string: str = "127.0.0.1:14550", wait_ready: bool = True):
        self.connection_string = connection_string
        self.wait_ready = wait_ready
        self.vehicle = None
    
    def connect_vehicle(self) -> None:
        print("Connecting to vehicle...")
        self.vehicle = connect(self.connection_string, wait_ready=self.wait_ready)
        print("Connected to vehicle.")
    
    def arm_and_takeoff(self, altitude: float, timeout: float = 30) -> bool:
        if not self.vehicle:
            print("No vehicle connected.")
            return False
        
        print("Arming motors...")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print("- Waiting for arming...")
            time.sleep(1)
        
        print(f"Taking off to {altitude} meters...")
        self.vehicle.simple_takeoff(altitude)
        
        start_time = time.time()
        while True:
            current_alt = self.vehicle.location.global_relative_frame.alt
            print(f"Current altitude: {current_alt:.1f} m")
            if current_alt >= altitude * 0.95:
                print("Reached target altitude!")
                return True
            if time.time() - start_time > timeout:
                print("Takeoff timed out!")
                return False
            time.sleep(1)
    
    def set_mode(self, mode: str) -> None:
        if not self.vehicle:
            print("No vehicle connected.")
            return
        self.vehicle.mode = VehicleMode(mode)
        print(f"Set mode to {mode}")
    
    def close(self) -> None:
        if self.vehicle:
            self.vehicle.close()
            print("Vehicle connection closed.")
    
    def get_vehicle(self):
        return self.vehicle