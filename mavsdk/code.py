"""
Autonomous Drone Survey Mission Planner with MAVSDK
This modular program creates survey missions with geofencing, waypoint generation,
path visualization, and mission upload capabilities.
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from typing import List, Tuple, Dict
from dataclasses import dataclass
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
from mavsdk.geofence import Point, Polygon, FenceType, GeofenceData, Circle
import folium
from folium import plugins


@dataclass
class MissionConfig:
    """Configuration parameters for the drone mission"""
    geofence_coords: List[Tuple[float, float]]  # [(lat, lon), ...]
    survey_altitude: float  # meters
    payload_drop_altitude: float  # meters
    sweep_spacing: float  # meters between parallel sweep lines
    geofence_threshold: float  # meters inset from geofence boundary
    takeoff_altitude: float = 10.0  # meters
    speed: float = 10.0  # m/s
    
    
class GeofenceManager:
    """Handles geofence operations and boundary calculations"""
    
    @staticmethod
    def create_inset_boundary(coords: List[Tuple[float, float]], 
                             threshold_m: float) -> List[Tuple[float, float]]:
        """
        Create an inset boundary from geofence coordinates
        
        Args:
            coords: Original geofence coordinates [(lat, lon), ...]
            threshold_m: Inset distance in meters
            
        Returns:
            Inset boundary coordinates
        """
        # Convert lat/lon to approximate meters (simplified)
        center_lat = np.mean([c[0] for c in coords])
        meters_per_degree_lat = 111320
        meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
        
        # Convert to cartesian
        points = np.array([
            [(lon - coords[0][1]) * meters_per_degree_lon,
             (lat - coords[0][0]) * meters_per_degree_lat]
            for lat, lon in coords
        ])
        
        # Calculate polygon centroid
        centroid = np.mean(points, axis=0)
        
        # Move each point toward centroid by threshold distance
        inset_points = []
        for point in points:
            direction = centroid - point
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                new_point = point + direction * threshold_m
                inset_points.append(new_point)
            else:
                inset_points.append(point)
        
        # Convert back to lat/lon
        inset_coords = [
            (coords[0][0] + p[1] / meters_per_degree_lat,
             coords[0][1] + p[0] / meters_per_degree_lon)
            for p in inset_points
        ]
        
        return inset_coords
    
    @staticmethod
    def get_bounding_box(coords: List[Tuple[float, float]]) -> Dict[str, float]:
        """Get bounding box of coordinates"""
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }


class SurveyPathGenerator:
    """Generates survey waypoints using lawnmower pattern"""
    
    @staticmethod
    def generate_lawnmower_pattern(boundary: List[Tuple[float, float]],
                                   spacing_m: float,
                                   altitude: float,
                                   threshold_m: float = 0.0) -> List[MissionItem]:
        """
        Generate lawnmower survey pattern waypoints
        
        Args:
            boundary: Boundary coordinates [(lat, lon), ...]
            spacing_m: Spacing between sweep lines in meters
            altitude: Survey altitude in meters
            
        Returns:
            List of MissionItem waypoints
        """
        # If a threshold (inset) is provided, create an inset boundary first so
        # generated waypoints remain at least `threshold_m` inside the geofence.
        if threshold_m and threshold_m > 0.0:
            boundary = GeofenceManager.create_inset_boundary(boundary, threshold_m)

        # Convert boundary to local meters using centroid reference
        center_lat = np.mean([c[0] for c in boundary])
        center_lon = np.mean([c[1] for c in boundary])

        meters_per_degree_lat = 111320
        meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))

        pts_m = []
        for lat, lon in boundary:
            x = (lon - center_lon) * meters_per_degree_lon
            y = (lat - center_lat) * meters_per_degree_lat
            pts_m.append((x, y))

        # Find the longest polygon edge and its angle
        longest = 0.0
        longest_angle = 0.0
        n = len(pts_m)
        for i in range(n):
            x1, y1 = pts_m[i]
            x2, y2 = pts_m[(i + 1) % n]
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if length > longest:
                longest = length
                longest_angle = np.arctan2(dy, dx)

        # Rotate points so the longest edge aligns with the x-axis
        theta = -longest_angle
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        def rotate(pt: Tuple[float, float]) -> Tuple[float, float]:
            x, y = pt
            xr = x * cos_t - y * sin_t
            yr = x * sin_t + y * cos_t
            return (xr, yr)

        def inv_rotate(xr: float, yr: float) -> Tuple[float, float]:
            # inverse rotation by -theta (i.e., rotate by +longest_angle)
            cos_i = np.cos(-theta)
            sin_i = np.sin(-theta)
            x = xr * cos_i - yr * sin_i
            y = xr * sin_i + yr * cos_i
            return (x, y)

        pts_rot = [rotate(p) for p in pts_m]

        # Bounding box in rotated coordinates
        xs = [p[0] for p in pts_rot]
        ys = [p[1] for p in pts_rot]
        y_min = min(ys)
        y_max = max(ys)

        # Number of sweep lines in rotated meters
        num_lines = int((y_max - y_min) / spacing_m) + 1

        waypoints: List[MissionItem] = []

        # Helper to build mission item from rotated meters back to lat/lon
        def _mk_wp_from_m(x_m: float, y_m: float) -> MissionItem:
            x_orig, y_orig = inv_rotate(x_m, y_m)
            lon = center_lon + x_orig / meters_per_degree_lon
            lat = center_lat + y_orig / meters_per_degree_lat
            return MissionItem(
                lat,
                lon,
                altitude,
                10.0,
                True,
                float('nan'),
                float('nan'),
                MissionItem.CameraAction.NONE,
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
                MissionItem.VehicleAction.NONE
            )

        # For each sweep line (horizontal in rotated frame), compute intersections
        for i in range(num_lines):
            y_line = y_min + i * spacing_m

            # Find intersections with polygon edges in rotated frame
            intersections: List[float] = []  # x positions where edges cross y_line
            for j in range(n):
                x1, y1 = pts_rot[j]
                x2, y2 = pts_rot[(j + 1) % n]

                if (y1 > y_line) != (y2 > y_line):
                    t = (y_line - y1) / (y2 - y1)
                    x_inter = x1 + t * (x2 - x1)
                    intersections.append(x_inter)

            if not intersections:
                continue

            intersections.sort()

            # Pair into inside intervals
            intervals: List[Tuple[float, float]] = []
            for k in range(0, len(intersections) - 1, 2):
                x_start = intersections[k]
                x_end = intersections[k + 1]
                if abs(x_end - x_start) < 1e-6:
                    continue
                intervals.append((x_start, x_end))

            if not intervals:
                continue

            forward = (i % 2 == 0)
            if not forward:
                intervals = list(reversed(intervals))

            for (x_start, x_end) in intervals:
                if forward:
                    waypoints.append(_mk_wp_from_m(x_start, y_line))
                    waypoints.append(_mk_wp_from_m(x_end, y_line))
                else:
                    waypoints.append(_mk_wp_from_m(x_end, y_line))
                    waypoints.append(_mk_wp_from_m(x_start, y_line))

        return waypoints


class PathVisualizer:
    """Visualizes drone path using matplotlib and folium"""
    
    @staticmethod
    def plot_path_matplotlib(geofence: List[Tuple[float, float]],
                            inset_boundary: List[Tuple[float, float]],
                            waypoints: List[MissionItem],
                            payload_location: Tuple[float, float] = None):
        """
        Plot drone path using matplotlib
        
        Args:
            geofence: Original geofence coordinates
            inset_boundary: Inset boundary coordinates
            waypoints: Survey waypoints
            payload_location: Optional payload drop location
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot geofence
        geofence_array = np.array(geofence + [geofence[0]])
        ax.plot(geofence_array[:, 1], geofence_array[:, 0], 
               'r-', linewidth=2, label='Geofence')
        ax.fill(geofence_array[:, 1], geofence_array[:, 0], 
               'r', alpha=0.1)
        
        # Plot inset boundary
        inset_array = np.array(inset_boundary + [inset_boundary[0]])
        ax.plot(inset_array[:, 1], inset_array[:, 0], 
               'g--', linewidth=2, label='Survey Boundary')
        
        # Plot waypoints
        wp_lats = [wp.latitude_deg for wp in waypoints]
        wp_lons = [wp.longitude_deg for wp in waypoints]
        ax.plot(wp_lons, wp_lats, 'b-', linewidth=1, label='Survey Path', alpha=0.7)
        ax.plot(wp_lons, wp_lats, 'bo', markersize=4)
        
        # Mark start and end
        ax.plot(wp_lons[0], wp_lats[0], 'go', markersize=10, label='Start')
        ax.plot(wp_lons[-1], wp_lats[-1], 'ro', markersize=10, label='End')
        
        # Plot payload drop if provided
        if payload_location:
            ax.plot(payload_location[1], payload_location[0], 
                   'y*', markersize=15, label='Payload Drop')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Drone Survey Mission Path')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('mission_path_matplotlib.png', dpi=300, bbox_inches='tight')
        print("✓ Matplotlib visualization saved as 'mission_path_matplotlib.png'")
        plt.show()
    
    @staticmethod
    def plot_path_folium(geofence: List[Tuple[float, float]],
                        inset_boundary: List[Tuple[float, float]],
                        waypoints: List[MissionItem],
                        payload_location: Tuple[float, float] = None) -> folium.Map:
        """
        Create interactive map using folium
        
        Args:
            geofence: Original geofence coordinates
            inset_boundary: Inset boundary coordinates
            waypoints: Survey waypoints
            payload_location: Optional payload drop location
            
        Returns:
            Folium map object
        """
        # Calculate center
        center_lat = np.mean([c[0] for c in geofence])
        center_lon = np.mean([c[1] for c in geofence])
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='OpenStreetMap'
        )
        
        # Add geofence
        folium.Polygon(
            locations=geofence,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.1,
            weight=3,
            popup='Geofence Boundary'
        ).add_to(m)
        
        # Add inset boundary
        folium.Polygon(
            locations=inset_boundary,
            color='green',
            fill=False,
            weight=2,
            dashArray='10, 5',
            popup='Survey Boundary'
        ).add_to(m)
        
        # Add survey path
        survey_coords = [(wp.latitude_deg, wp.longitude_deg) for wp in waypoints]
        folium.PolyLine(
            locations=survey_coords,
            color='blue',
            weight=2,
            opacity=0.7,
            popup='Survey Path'
        ).add_to(m)
        
        # Add waypoint markers
        for i, wp in enumerate(waypoints):
            folium.CircleMarker(
                location=[wp.latitude_deg, wp.longitude_deg],
                radius=3,
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.6,
                popup=f'WP{i+1}: Alt {wp.relative_altitude_m}m'
            ).add_to(m)
        
        # Mark start
        folium.Marker(
            location=[waypoints[0].latitude_deg, waypoints[0].longitude_deg],
            popup='Start',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # Mark end
        folium.Marker(
            location=[waypoints[-1].latitude_deg, waypoints[-1].longitude_deg],
            popup='End',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
        
        # Add payload drop
        if payload_location:
            folium.Marker(
                location=payload_location,
                popup='Payload Drop',
                icon=folium.Icon(color='orange', icon='download')
            ).add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Save map
        m.save('mission_path_folium.html')
        print("✓ Folium map saved as 'mission_path_folium.html'")
        
        return m


class MissionPlanner:
    """Main mission planning and execution coordinator"""
    
    def __init__(self, config: MissionConfig):
        self.config = config
        self.inset_boundary = None
        self.survey_waypoints = []
        self.full_mission_items = []
        
    def plan_mission(self) -> List[MissionItem]:
        """
        Plan complete mission from takeoff to RTL
        
        Returns:
            Complete list of mission items
        """
        print("\n📋 Planning Mission...")
        
        # Create inset boundary
        print(f"  → Creating {self.config.geofence_threshold}m inset boundary...")
        self.inset_boundary = GeofenceManager.create_inset_boundary(
            self.config.geofence_coords,
            self.config.geofence_threshold
        )
        
        # Generate survey waypoints (generator will apply the inset using threshold)
        print(f"  → Generating survey pattern (spacing: {self.config.sweep_spacing}m, threshold: {self.config.geofence_threshold}m)...")
        self.survey_waypoints = SurveyPathGenerator.generate_lawnmower_pattern(
            self.config.geofence_coords,
            self.config.sweep_spacing,
            self.config.survey_altitude,
            threshold_m=self.config.geofence_threshold
        )
        
        print(f"  ✓ Generated {len(self.survey_waypoints)} survey waypoints")
        
        # Build complete mission
        self.full_mission_items = self.survey_waypoints
        
        return self.full_mission_items
    
    def visualize_mission(self, payload_location: Tuple[float, float] = None):
        """Visualize the planned mission"""
        print("\n🗺️  Generating Visualizations...")
        
        # Matplotlib visualization
        PathVisualizer.plot_path_matplotlib(
            self.config.geofence_coords,
            self.inset_boundary,
            self.survey_waypoints,
            payload_location
        )
        
        # Folium visualization
        PathVisualizer.plot_path_folium(
            self.config.geofence_coords,
            self.inset_boundary,
            self.survey_waypoints,
            payload_location
        )
        
        print("✓ Visualizations complete")
    
    async def upload_mission(self, drone: System, rtl: bool = True):
        """
        Upload mission to drone
        
        Args:
            drone: MAVSDK System instance
            rtl: Whether to add RTL at the end
        """
        print("\n📤 Uploading Mission to Drone...")
        
        try:
            # Create mission plan
            mission_plan = MissionPlan(self.full_mission_items)
            
            # Upload mission
            await drone.mission.upload_mission(mission_plan)
            print(f"  ✓ Uploaded {len(self.full_mission_items)} waypoints")
            
            # Set RTL after mission if requested
            if rtl:
                await drone.mission.set_return_to_launch_after_mission(True)
                print("  ✓ RTL enabled after mission completion")
            
            print("✓ Mission upload successful")
            
        except Exception as e:
            print(f"✗ Mission upload failed: {e}")
            raise
    
    async def upload_geofence(self, drone: System):
        """Upload geofence to drone"""
        print("\n🚧 Uploading Geofence...")
        
        try:
            # Create geofence points
            points = [Point(lat, lon) for lat, lon in self.config.geofence_coords]
            
            # Create polygon
            polygon = Polygon(points, FenceType.INCLUSION)
            
            circle = Circle(Point(0, 0), 0, FenceType.INCLUSION)
            # Upload geofence
            geofenceData = GeofenceData([polygon], [circle])
            await drone.geofence.upload_geofence(geofenceData)
            print("  ✓ Geofence uploaded successfully")
            
        except Exception as e:
            print(f"✗ Geofence upload failed: {e}")
            raise
    
    async def ret(self, drone: System):
        await drone.mission.set_return_to_launch_after_mission(True)

    async def flush_mission(self, drone: System):
        """Flush existing mission from drone"""
        print("\n🧹 Flushing Existing Mission...")
        
        try:
            await drone.mission.clear_mission()
            print("  ✓ Existing mission cleared")
        except Exception as e:
            print(f"✗ Failed to clear mission: {e}")
            raise



class DroneController:
    """Handles drone connection and mission execution"""
    
    def __init__(self, connection_url: str = "udp://:14540"):
        self.connection_url = connection_url
        self.drone = System()
        
    async def connect(self):
        """Connect to drone"""
        print(f"\n🔌 Connecting to drone at {self.connection_url}...")
        await self.drone.connect(system_address=self.connection_url)
        
        print("  → Waiting for drone connection...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("  ✓ Drone connected")
                break
        
        # Wait for global position estimate
        print("  → Waiting for GPS lock...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("  ✓ GPS lock acquired")
                break
    
    async def execute_mission(self, mission_planner: MissionPlanner, 
                             upload_geofence: bool = True):
        """
        Execute the complete mission
        
        Args:
            mission_planner: MissionPlanner instance with planned mission
            upload_geofence: Whether to upload geofence
        """
        print("\n🚁 Preparing for Mission Execution...")
        await mission_planner.flush_mission(self.drone)
        await mission_planner.ret(self.drone)

        # Upload geofence if requested
        if upload_geofence:
            await mission_planner.upload_geofence(self.drone)
        
        # Upload mission
        await mission_planner.upload_mission(self.drone, rtl=True)
        
        # Arm drone
        print("\n⚡ Arming drone...")
        await self.drone.action.arm()
        print("  ✓ Drone armed")
        
        # Start mission
        print("\n🚀 Starting mission...")
        await self.drone.mission.start_mission()
        print("  ✓ Mission started")
        
        # Monitor mission progress
        print("\n📊 Mission Progress:")
        async for progress in self.drone.mission.mission_progress():
            print(f"  → Waypoint {progress.current}/{progress.total}")
            
            if progress.current == progress.total:
                print("\n✓ Mission completed!")
                break
        
        # Wait for RTL completion
        print("\n🏠 Returning to launch...")
        async for flight_mode in self.drone.telemetry.flight_mode():
            if flight_mode == "HOLD":
                print("  ✓ Landed safely")
                break


# Main execution function
async def main():
    """Main execution function"""
    
    print("=" * 60)
    print("  AUTONOMOUS DRONE SURVEY MISSION PLANNER")
    print("  Powered by MAVSDK")
    print("=" * 60)
    
    # ============================================
    # USER INPUT SECTION
    # ============================================
    
    # Example geofence coordinates (replace with user input)
    geofence_coords = [
        (23.177053285530338, 80.02196321569755),
        (23.176676027386808, 80.02133826101728),
        (23.175734110255565, 80.02166280829759),
        (23.175891918823606, 80.02249429306535),
        (23.176944793101267, 80.02219120345646)
    ]
    
    # Mission parameters (replace with user input)
    survey_altitude = 10.0  # meters
    payload_drop_altitude = 5.0  # meters
    sweep_spacing = 25.0  # meters between lines
    geofence_threshold = 10.0  # meters inset from boundary
    
    # Optional payload drop location
    payload_location = (47.3972, 8.5466)
    
    # Create mission configuration
    config = MissionConfig(
        geofence_coords=geofence_coords,
        survey_altitude=survey_altitude,
        payload_drop_altitude=payload_drop_altitude,
        sweep_spacing=sweep_spacing,
        geofence_threshold=geofence_threshold
    )
    
    # ============================================
    # MISSION PLANNING
    # ============================================
    
    # Create mission planner
    planner = MissionPlanner(config)
    
    # Plan mission
    planner.plan_mission()
    
    # Visualize mission
    planner.visualize_mission(payload_location)
    
    # ============================================
    # USER CONFIRMATION
    # ============================================
    
    print("\n" + "=" * 60)
    confirmation = input("📋 Review the mission visualizations. Proceed with upload? (yes/no): ")
    
    if confirmation.lower() not in ['yes', 'y']:
        print("❌ Mission cancelled by user")
        return
    
    # ============================================
    # MISSION EXECUTION
    # ============================================
    
    # Create drone controller
    controller = DroneController(connection_url="udpin://0.0.0.0:14540")
    
    # Connect to drone
    await controller.connect()
    
    # Execute mission
    await controller.execute_mission(planner, upload_geofence=True)
    
    print("\n" + "=" * 60)
    print("  ✓ MISSION COMPLETE")
    print("=" * 60)


# Entry point
if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
