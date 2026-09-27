import unittest

from PIL import Image

from scripts.object_motion import sam_point_review48 as review


class SAMPointReview48Tests(unittest.TestCase):
    def test_exact_common_prompt_records_and_original_crop(self):
        rows = [{"path": f"photos/NP3_{angle:03}.jpg", "sha256": f"{i:064x}"}
                for i, angle in enumerate(a for a in range(0, 360, 6) if (a // 6) % 5 != 4)]
        prompts = review.prompt_rows(rows)
        self.assertEqual(len(prompts), 48)
        self.assertEqual(prompts[0]["points_xy_label"],
                         [[600, 525, 1], [720, 620, 0], [520, 630, 0]])
        self.assertEqual(prompts[-1]["name"], "NP3_348.jpg")
        crop = review.annotated_crop(Image.new("RGB", (1280, 1024), "white"))
        self.assertEqual(crop.size, (280, 350))
        self.assertNotEqual(crop.getpixel((125, 225)), (255, 255, 255))
        with self.assertRaisesRegex(ValueError, "48"):
            review.prompt_rows(rows[:-1])


if __name__ == "__main__":
    unittest.main()
