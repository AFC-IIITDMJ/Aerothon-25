from dronekit import Vehicle
from pymavlink import mavutil
import time
from typing import List, Tuple

class GeofenceManager:
    
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
    
    def set_geofence(self, coordinates: List[Tuple[float, float]], enable: bool = True) -> None:
        master = self.vehicle._master
        target_system = master.target_system
        target_component = master.target_component
        
        # Disable geofence first
        master.mav.command_long_send(
            target_system, target_component,
            mavutil.mavlink.MAV_CMD_DO_FENCE_ENABLE,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        print("[*] Geofence disabled.")
        time.sleep(1)
        
        # Set total number of fence points
        self.vehicle.parameters['FENCE_TOTAL'] = len(coordinates)
        print(f"[*] Set FENCE_TOTAL to {len(coordinates)}")
        time.sleep(1)
        
        # Send mission count
        master.mav.mission_count_send(
            target_system, target_component,
            len(coordinates),
            mavutil.mavlink.MAV_MISSION_TYPE_FENCE
        )
        
        # Send each vertex
        for idx, (lat, lon) in enumerate(coordinates):
            master.mav.mission_item_int_send(
                target_system, target_component, idx,
                mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
                mavutil.mavlink.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION,
                0, 1, len(coordinates), 0, 0, 0,
                int(lat * 1e7), int(lon * 1e7), 0,
                mavutil.mavlink.MAV_MISSION_TYPE_FENCE
            )
            print(f"[*] Sent vertex {idx}: {lat}, {lon}")
            time.sleep(0.1)
        
        if enable:
            master.mav.command_long_send(
                target_system, target_component,
                mavutil.mavlink.MAV_CMD_DO_FENCE_ENABLE,
                0, 1, 0, 0, 0, 0, 0, 0
            )
            print("[✔] Geofence enabled.")