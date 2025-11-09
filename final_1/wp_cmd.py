import numpy as np
import matplotlib.pyplot as plt

def create_inset_boundary(coords, inset_meters):
    """Create boundary inset by specified meters"""
    center_lat = np.mean([c[0] for c in coords])
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(center_lat))
    
    # Convert to meters
    points = np.array([
        [(lon - coords[0][1]) * meters_per_deg_lon,
         (lat - coords[0][0]) * meters_per_deg_lat]
        for lat, lon in coords
    ])
    
    # Move points toward centroid
    centroid = np.mean(points, axis=0)
    inset_points = []
    
    for point in points:
        direction = centroid - point
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
            new_point = point + direction * inset_meters
            inset_points.append(new_point)
        else:
            inset_points.append(point)
    
    # Convert back to lat/lon
    inset_coords = [
        (coords[0][0] + p[1] / meters_per_deg_lat,
         coords[0][1] + p[0] / meters_per_deg_lon)
        for p in inset_points
    ]
    
    return inset_coords

def generate_survey_waypoints(boundary, spacing, altitude):
    """Generate lawnmower pattern waypoints"""
    center_lat = np.mean([c[0] for c in boundary])
    center_lon = np.mean([c[1] for c in boundary])
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(center_lat))
    
    # Convert to meters
    pts_m = [
        ((lon - center_lon) * meters_per_deg_lon,
         (lat - center_lat) * meters_per_deg_lat)
        for lat, lon in boundary
    ]
    
    # Find longest edge angle
    longest_length = 0.0
    longest_angle = 0.0
    n = len(pts_m)
    
    for i in range(n):
        x1, y1 = pts_m[i]
        x2, y2 = pts_m[(i + 1) % n]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > longest_length:
            longest_length = length
            longest_angle = np.arctan2(y2 - y1, x2 - x1)
    
    # Rotate to align with longest edge
    theta = -longest_angle + + np.pi / 2
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    pts_rot = [(x * cos_t - y * sin_t, x * sin_t + y * cos_t) for x, y in pts_m]
    
    # Get Y bounds
    ys = [p[1] for p in pts_rot]
    y_min, y_max = min(ys), max(ys)
    num_lines = int((y_max - y_min) / spacing) + 1
    
    waypoints = []
    
    # Generate sweep lines
    for i in range(num_lines):
        y_line = y_min + i * spacing
        intersections = []
        
        # Find intersections with polygon edges
        for j in range(n):
            x1, y1 = pts_rot[j]
            x2, y2 = pts_rot[(j + 1) % n]
            
            if (y1 > y_line) != (y2 > y_line):
                t = (y_line - y1) / (y2 - y1)
                x_inter = x1 + t * (x2 - x1)
                intersections.append(x_inter)
        
        if len(intersections) < 2:
            continue
        
        intersections.sort()
        
        # Create waypoint pairs
        forward = (i % 2 == 0)
        for k in range(0, len(intersections) - 1, 2):
            x_start, x_end = intersections[k], intersections[k + 1]
            
            # Rotate back and convert to lat/lon
            for x_val in ([x_start, x_end] if forward else [x_end, x_start]):
                cos_i, sin_i = np.cos(-theta), np.sin(-theta)
                x_orig = x_val * cos_i - y_line * sin_i
                y_orig = x_val * sin_i + y_line * cos_i
                
                lon = center_lon + x_orig / meters_per_deg_lon
                lat = center_lat + y_orig / meters_per_deg_lat
                
                waypoints.append([lat,lon,altitude])
    
    return waypoints

def plot_mission(geofence, boundary, waypoints, output='mission_map.png'):
    """Simple matplotlib visualization"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Survey Area
    gf = np.array(geofence + [geofence[0]])
    ax.plot(gf[:, 1], gf[:, 0], 'r-', linewidth=2, label='Survey Area')
    ax.fill(gf[:, 1], gf[:, 0], 'r', alpha=0.1)
    
    # Boundary
    bd = np.array(boundary + [boundary[0]])
    ax.plot(bd[:, 1], bd[:, 0], 'g--', linewidth=2, label='Survey Boundary')
    
    # Waypoints
    lats = [w[0] for w in waypoints]
    lons = [w[1] for w in waypoints]
    ax.plot(lons, lats, 'b-', linewidth=1, alpha=0.7, label='Path')
    ax.plot(lons, lats, 'bo', markersize=3)
    ax.plot(lons[0], lats[0], 'go', markersize=10, label='Start')
    ax.plot(lons[-1], lats[-1], 'ro', markersize=10, label='End')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Survey Mission')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.savefig(output, dpi=200, bbox_inches='tight')  
    plt.show()  