from dronekit import Vehicle, Command
from pymavlink import mavutil
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.utilities.geometry_utils import create_inner_polygon, generate_lawnmower_path

class MissionPlanner:
    
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
    
    def plan_mission(self, geofence_coords: List[Tuple[float, float]], buffer_distance: float = -0.000045,
                     line_spacing: float = 0.0001, altitude: float = 15) -> List[Tuple[float, float]]:
        geofence_poly = Polygon([(lon, lat) for lat, lon in geofence_coords])
        inner_poly = create_inner_polygon(geofence_poly, buffer_distance)
        if not inner_poly or inner_poly.is_empty:
            print("Using fallback buffer distance")
            inner_poly = create_inner_polygon(geofence_poly, -0.00002)
        
        path = generate_lawnmower_path(inner_poly, line_spacing)
        waypoints = [(lat, lon) for line in path for lon, lat in line]
        
        self.plot_mission(geofence_poly, inner_poly, waypoints)
        if waypoints:
            self.upload_mission(waypoints, altitude)
        return waypoints
    
    def upload_mission(self, waypoints: List[Tuple[float, float]], altitude: float) -> None:
        cmds = self.vehicle.commands
        cmds.clear()
        cmds.wait_ready()
        print("Uploading mission...")
        
        cmds.add(Command(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, waypoints[0][0], waypoints[0][1], altitude
        ))
        
        for lat, lon in waypoints:
            cmds.add(Command(
                0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 0, 0, 0, 0, 0, lat, lon, altitude
            ))
        
        cmds.add(Command(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ))
        
        cmds.upload()
        print(f"Uploaded {len(waypoints)+2} mission items")
    
    def plot_mission(self, geofence: Polygon, inner_poly: Polygon, waypoints: List[Tuple[float, float]]) -> None:
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
        plt.savefig('mission_plan.png')