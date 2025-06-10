import math


def pixel_to_gps(u: int, v: int,
                 drone_lat: float, drone_lon: float,
                 altitude: float, heading_deg: float,
                 resolution=(640, 480), fov=(60, 45)) -> (float, float):
    """
    Convert image pixel (u, v) to GPS coordinates (lat, lon) given drone state.

    Args:
        u, v (int): Pixel coordinates in the image frame.
        drone_lat, drone_lon (float): Drone's current latitude and longitude (degrees).
        altitude (float): Drone altitude above ground (meters).
        heading_deg (float): Drone heading angle (degrees, 0 = North).
        resolution (tuple): (width, height) in pixels.
        fov (tuple): (horizontal_FOV, vertical_FOV) in degrees.

    Returns:
        (float, float): (target_lat, target_lon) computed from pixel offset.
    """
    # 1) Compute focal lengths in pixels
    fx = resolution[0] / (2 * math.tan(math.radians(fov[0] / 2)))
    fy = resolution[1] / (2 * math.tan(math.radians(fov[1] / 2)))
    cx, cy = resolution[0] / 2, resolution[1] / 2

    # 2) Normalized image coordinates
    x_norm = (u - cx) / fx   # Positive → to the right
    y_norm = (v - cy) / fy   # Positive → downward

    # 3) Project onto ground plane
    ground_x_cam = altitude * x_norm  # Camera frame: +x_cam = right
    ground_y_cam = -altitude * y_norm  # Camera frame: +y_cam = forward (north when heading = 0)

    # 4) Rotate by heading to get N/E offsets
    heading_rad = math.radians(heading_deg)
    delta_north = (ground_y_cam * math.cos(heading_rad)
                   - ground_x_cam * math.sin(heading_rad))
    delta_east = (ground_y_cam * math.sin(heading_rad)
                  + ground_x_cam * math.cos(heading_rad))

    # 5) Convert from meters to degrees
    earth_radius = 6378137.0  # meters
    meters_per_deg_lat = earth_radius * math.pi / 180.0
    delta_lat = delta_north / meters_per_deg_lat

    lat_rad = math.radians(drone_lat)
    meters_per_deg_lon = meters_per_deg_lat * math.cos(lat_rad)
    delta_lon = delta_east / meters_per_deg_lon

    return drone_lat + delta_lat, drone_lon + delta_lon


def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """
    Compute the Haversine distance (meters) between two GPS coordinates.

    Args:
        lat1, lon1, lat2, lon2 (float): Coordinates in degrees.

    Returns:
        float: Distance in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
