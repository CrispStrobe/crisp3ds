import unittest

from scripts.pipes_sparse.contacts import source_xy


class ContactsTests(unittest.TestCase):
    def test_resized_pixel_edge_coordinates_return_to_source_frame(self):
        scales = [1024 / 6220, 682 / 4141]
        source = [3040.25, 2220.75]
        resized = [source[0] * scales[0], source[1] * scales[1]]
        restored = source_xy(resized, scales)
        self.assertAlmostEqual(restored[0], source[0], places=10)
        self.assertAlmostEqual(restored[1], source[1], places=10)
        with self.assertRaises(ValueError):
            source_xy(resized, [0, scales[1]])


if __name__ == "__main__":
    unittest.main()
