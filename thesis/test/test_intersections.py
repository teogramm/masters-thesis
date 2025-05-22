import unittest

from thesis.filtering.area_intersection import coords_intersect_polygon, points_are_counterclockwise
from thesis.processing.crossing_times import line_intersection_index


class TestAreaIntersection(unittest.TestCase):

    def test_inside(self):
        p1 = [-24.54, -17.07]
        p2 = [-8.52, -15.98]
        p3 = [-16.77, -2.95]
        p4 = [-29.53, -9.1]
        p = [-21.91, -11.46]
        self.assertTrue(coords_intersect_polygon(p[0], p[1], p1, p2, p3, p4))

    def test_outside(self):
        p1 = [8.22, -5.02]
        p2 = [22.02, -5.71]
        p3 = [27.34, 3.05]
        p4 = [3.03, 2.33]
        p = [6.57, 3.11]
        self.assertFalse(coords_intersect_polygon(p[0], p[1], p1, p2, p3, p4))

    def test_edge(self):
        p1 = [8.22, -5.02]
        p2 = [22.02, -5.71]
        p3 = [27.34, 3.05]
        p4 = [3.03, 2.33]
        p = [4.72, 2.38]
        self.assertTrue(coords_intersect_polygon(p[0], p[1], p1, p2, p3, p4))


class TestCrossing(unittest.TestCase):

    def test_empty_x_y(self):
        l0 = [1, 1]
        l1 = [0, 0]
        x = []
        y = []
        self.assertEqual(line_intersection_index(x, y, l0, l1), -1)

    def test_intersection_first(self):
        l0 = [-6, -1]
        l1 = [-4, 1]
        x = [-5.22, -4.79, -3.98, -3.19]
        y = [0.35, -0.22, -0.58, -0.54]
        self.assertEqual(line_intersection_index(x, y, l0, l1), 0)

    def test_intersection_last(self):
        l0 = [-6, -1]
        l1 = [-4, 1]
        x = [-3.19, -3.98, -4.79, -5.22]
        y = [-0.54, -0.58, -0.22, 0.35]
        self.assertEqual(line_intersection_index(x, y, l0, l1), 2)
        
    def test_intersection_middle(self):
        l0 = [2.66, 7.04]
        l1 = [-2.09, 4.96]
        x = [-3.47, -3, -2.33, -1.59, -1.16, -0.95, 1.86, 3.84, 6.28]
        y = [10.96, 9.95, 8.41, 7.07, 5.93, 5.03, 4.73, 4.79, 4.83]
        self.assertEqual(line_intersection_index(x, y, l0, l1), 4)

class TestCounterclockwise(unittest.TestCase):
    
    def test_clockwise(self):
        points = [
            (-10.52, -3.25),
            (-8.48, 1.43),
            (-1.28, 1.53),
            (-6.44, -3.43)
        ]
        self.assertFalse(points_are_counterclockwise(points))
        
    def test_counterclockwise(self):
        points = [
            (6.86, 3.56),
            (2.26, 1.92),
            (3.7, -4.5),
            (15.56, 4.24)
        ]
        self.assertTrue(points_are_counterclockwise(points))
    
    # def test_equal_min_y(self):
    #     self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
