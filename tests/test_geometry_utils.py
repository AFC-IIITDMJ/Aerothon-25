import unittest
from shapely.geometry import Polygon
from src.utilities.geometry_utils import create_inner_polygon, generate_lawnmower_path

class TestGeometryUtils(unittest.TestCase):
    def test_create_inner_polygon(self):
        coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        poly = Polygon(coords)
        inner = create_inner_polygon(poly, -0.1)
        self.assertFalse(inner.is_empty, "Inner polygon is empty")

    def test_generate_lawnmower_path(self):
        coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        poly = Polygon(coords)
        path = generate_lawnmower_path(poly, 0.1)
        self.assertTrue(len(path) > 0, "No path generated")

if __name__ == '__main__':
    unittest.main()