from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.object_dataset import evaluate, preview
from scripts.object_dataset.test_evaluate_surface import write_triangle


class PreviewTests(unittest.TestCase):
    def test_same_bounds_for_reference_and_output_panels(self):
        reference = np.array([[0., 0., 0.], [1., 2., 3.]])
        output = np.array([[10., 20., 30.], [11., 21., 31.]])
        center, radius = preview.panel_bounds(reference, output, (0, 1))
        np.testing.assert_allclose(center, [5.5, 10.5])
        self.assertGreater(radius, 10)

    def test_bounded_deterministic_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            mesh = write_triangle(Path(temporary) / "triangle.ply")
            _, vertices, faces = evaluate.inspect_ply(mesh, geometry=True)
            image_a = preview.make_preview((vertices, faces), (vertices, faces),
                                           np.eye(4), count=128, seed=30)
            image_b = preview.make_preview((vertices, faces), (vertices, faces),
                                           np.eye(4), count=128, seed=30)
            self.assertEqual(image_a.size, (2 * preview.PANEL, 3 * preview.PANEL))
            self.assertEqual(image_a.tobytes(), image_b.tobytes())
            with self.assertRaisesRegex(ValueError, "20,000"):
                preview.make_preview((vertices, faces), (vertices, faces),
                                     np.eye(4), count=20_001)


if __name__ == "__main__":
    unittest.main()
