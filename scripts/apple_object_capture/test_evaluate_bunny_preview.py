from pathlib import Path
import unittest

from scripts.apple_object_capture import evaluate_bunny_preview as target


class FrozenAppleBunnyProtocolTest(unittest.TestCase):
    def test_same_full_and_roi_threshold_panels_as_prior_bunny_protocol(self):
        output = Path("/tmp/fresh-apple-evaluation")
        stages = target.commands(output, Path("/tmp/project-python"))
        self.assertEqual([stage[0] for stage in stages],
                         ["whole_fit", "whole_score", "roi_fit", "roi_score"])
        for name, command, artifact, _ in stages:
            self.assertEqual(command[command.index("--output") + 1], str(target.MESH))
            if name.endswith("_fit"):
                self.assertEqual(command[command.index("--samples") + 1], "1024")
                self.assertEqual(artifact, output / f"{name.split('_')[0]}-reference-fit.json")
            else:
                self.assertEqual(command[command.index("--samples") + 1], "4096")
                self.assertEqual(command[command.index("--seed") + 1], "2027")
        def thresholds(command):
            return tuple(float(command[index + 1]) for index, word in enumerate(command)
                         if word == "--threshold")
        self.assertEqual(thresholds(stages[1][1]), target.FULL_THRESHOLDS)
        self.assertEqual(thresholds(stages[3][1]),
                         target.ROI_THRESHOLDS + target.FULL_THRESHOLDS)


if __name__ == "__main__":
    unittest.main()
