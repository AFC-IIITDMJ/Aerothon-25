from shapely.geometry import Polygon, LineString
from typing import List, Tuple, Optional

def create_inner_polygon(outer_polygon: Polygon, buffer_distance: float = -0.000045) -> Optional[Polygon]:
    inner = outer_polygon.buffer(buffer_distance)
    if inner.is_empty:
        print(f"Warning: Buffer distance {buffer_distance} resulted in empty geometry.")
        return None
    if not inner.is_valid:
        print("Attempting to fix invalid geometry...")
        inner = inner.buffer(0)
    return inner

def generate_lawnmower_path(polygon: Polygon, line_spacing: float = 0.0001) -> List[List[Tuple[float, float]]]:
    if polygon is None or polygon.is_empty:
        return []
    
    minx, miny, maxx, maxy = polygon.bounds
    lines = []
    x = minx
    direction = True
    
    while x <= maxx:
        scan_line = LineString([(x, miny), (x, maxy)])
        clipped = scan_line.intersection(polygon)
        if not clipped.is_empty:
            if clipped.geom_type == "LineString":
                coords = list(clipped.coords)
                if not direction:
                    coords.reverse()
                lines.append(coords)
            elif clipped.geom_type == "MultiLineString":
                for line in clipped.geoms:
                    coords = list(line.coords)
                    if not direction:
                        coords.reverse()
                    lines.append(coords)
        x += line_spacing
        direction = not direction
    return lines