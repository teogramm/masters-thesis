import unittest
from thesis.util.image_coord_transforms import transform_riddarhuskajen, transform_riddarholmsbron_n

class TestImageCoordTransforms(unittest.TestCase):
    
    def test_riddarhuskajen(self):
        image_points = [(565, 1266), (662, 1192)]
        transformed_points = transform_riddarhuskajen(image_points)
        expected_result = [(-350.0/477, 7610.0/477), (620.0/477, 2290.0/159)]
        for result, expected in zip(transformed_points, expected_result):
            self.assertAlmostEqual(result[0], expected[0])
            self.assertAlmostEqual(result[1], expected[1])
    
    def test_riddarholmsbron_n(self):
        image_points = [(565, 954), (960, 634)]
        transformed_points = transform_riddarholmsbron_n(image_points)
        expected_result = [(-5900.0/941, -530.0/941), (-1950.0/941, -3730.0/941)]
        for result, expected in zip(transformed_points, expected_result):
            self.assertAlmostEqual(result[0], expected[0])
            self.assertAlmostEqual(result[1], expected[1])