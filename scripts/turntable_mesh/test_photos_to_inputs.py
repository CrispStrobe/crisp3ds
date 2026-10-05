"""Pure parts of the photo front stage: no AliceVision, SAM or Torch needed."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.turntable_mesh import photos_to_inputs as front

EXAMPLE = Path(front.__file__).parent / "calibrations" / "3dlf-pro.json"
LIMITS = {"minimum_registered_fraction": 0.8, "minimum_observations_per_view": 20,
          "maximum_view_reprojection_p95_pixels": 4.0, "minimum_positive_depth_fraction": 1.0,
          "maximum_radius_spread_percent": 5.0,
          "maximum_out_of_plane_percent": 5.0, "maximum_angular_gap_deg": 30.0, "maximum_reversed_steps": 0,
          "maximum_optical_axis_miss_percent": 25.0, "duplicate_step_deg": 0.5}
GOOD_AUDIT = {"passed": True, "reasons": [], "coverage": 1.0, "registered_cameras": 36, "input_images": 36}


def orbit(angles_deg, radius=2.0, height=0.3, noise=None):
    """Cameras on a circle around the y axis looking at the origin: (indices, centres, world-to-camera rotations)."""
    centres, rotations = [], []
    for n, angle in enumerate(np.deg2rad(angles_deg)):
        centre = np.array([radius * math.sin(angle), height, radius * math.cos(angle)])
        if noise is not None:
            centre = centre + noise[n]
        forward = -centre / np.linalg.norm(centre)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        rotations.append(np.stack([right, np.cross(forward, right), forward]))
        centres.append(centre)
    return list(range(len(centres))), centres, rotations


class CalibrationTests(unittest.TestCase):
    def test_example_file_scales_to_the_3dlf_photo_size(self):
        calibration = front.load_calibration(EXAMPLE)
        self.assertEqual((calibration["width"], calibration["height"]), (583, 385))
        scaled = front.scale_calibration(calibration, 1749, 1155)
        # The values the accepted Dragon, Armadillo and Bunny cameras were recovered with.
        self.assertAlmostEqual(scaled["fx"], 2328.2847290039062, places=9)
        self.assertAlmostEqual(scaled["fy"], 2329.8724365234375, places=9)
        self.assertAlmostEqual(scaled["cx"], 874.1592102050781, places=9)
        self.assertAlmostEqual(scaled["cy"], 555.7409973144531, places=9)
        self.assertEqual(scaled["k"], calibration["k"])

    def test_same_resolution_is_identity_and_crops_are_refused(self):
        calibration = front.load_calibration(EXAMPLE)
        same = front.scale_calibration(calibration, 583, 385)
        self.assertAlmostEqual(same["cx"], calibration["cx"], places=12)
        self.assertAlmostEqual(same["fy"], calibration["fy"], places=12)
        with self.assertRaises(ValueError):
            front.scale_calibration(calibration, 1749, 1000)

    def test_raw_dataset_spelling_and_rejections(self):
        ours = json.loads(EXAMPLE.read_text())
        raw = {"model": "brown5", "width": 583, "height": 385, "fx": ours["fx"], "fy": ours["fy"], "cx": ours["cx"],
               "cy": ours["cy"], "distortion": {"k1": ours["k1"], "k2": ours["k2"], "p1": 0.0, "p2": 0.0, "k3": ours["k3"]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lens.json"
            path.write_text(json.dumps(raw))
            self.assertEqual(front.load_calibration(path), front.load_calibration(EXAMPLE))
            raw["distortion"]["p1"] = 1e-4
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                front.load_calibration(path)
            path.write_text(json.dumps({**ours, "fx": float("nan")}))
            with self.assertRaises(ValueError):
                front.load_calibration(path)
            path.write_text(json.dumps({"fx": 1}))
            with self.assertRaises(ValueError):
                front.load_calibration(path)

    def test_alicevision_fields_round_trip_through_the_audit_convention(self):
        scaled = front.scale_calibration(front.load_calibration(EXAMPLE), 1749, 1155)
        scene = {"views": [], "intrinsics": [{"type": "pinhole", "distortionType": "radialk3", "width": "1749",
                                              "height": "1155", "sensorWidth": "36", "focalLength": "40"}]}
        intrinsic = front.calibrated_scene(scene, scaled)["intrinsics"][0]
        self.assertEqual(scene["intrinsics"][0]["focalLength"], "40")  # input untouched
        self.assertEqual(intrinsic["locked"], "true")
        fy = float(intrinsic["focalLength"]) * 1749 / 36
        self.assertAlmostEqual(fy, scaled["fy"], places=9)
        self.assertAlmostEqual(fy / float(intrinsic["pixelRatio"]), scaled["fx"], places=9)
        self.assertAlmostEqual(float(intrinsic["principalPoint"][0]) + 1749 / 2, scaled["cx"], places=9)
        self.assertAlmostEqual(float(intrinsic["principalPoint"][1]) + 1155 / 2, scaled["cy"], places=9)
        self.assertEqual([float(v) for v in intrinsic["distortionParams"]], scaled["k"])
        with self.assertRaises(ValueError):
            front.calibrated_scene({**scene, "poses": [{}]}, scaled)


class MaskTests(unittest.TestCase):
    def test_envelope_forms(self):
        self.assertEqual(front.resolve_envelope("auto", 1749, 1155), (0, 0, 1749, 1155))
        self.assertEqual(front.resolve_envelope("200,250,1520,1150", 1749, 1155), (200, 250, 1520, 1150))
        self.assertEqual(front.resolve_envelope("0.1,0.2,0.9,1", 1000, 500), (100, 100, 900, 500))
        self.assertEqual(front.resolve_envelope("10,10,5000,5000", 100, 80), (10, 10, 100, 80))
        for bad in ("1,2,3", "a,b,c,d", "50,50,40,90", "-1,0,5,5"):
            with self.assertRaises(ValueError):
                front.resolve_envelope(bad, 100, 80)

    def test_coarse_mask_keeps_largest_dark_component_inside_envelope(self):
        gray = np.full((60, 80), 200, np.uint8)
        gray[20:40, 30:50] = 30  # object
        gray[5:8, 5:8] = 10  # speck inside the envelope
        gray[50:60, 0:80] = 0  # dark border, larger than the object
        mask, info = front.coarse_mask(gray, 70, (0, 0, 80, 48))
        self.assertEqual(int(mask.sum()), 400)
        self.assertTrue(mask[25, 35] and not mask[6, 6] and not mask[55, 10])
        self.assertEqual(info["bbox_xyxy"], [30, 20, 50, 40])
        self.assertEqual(info["other_dark_pixels_in_envelope"], 9)
        self.assertEqual(info["dark_pixels_outside_envelope"], 800)
        self.assertFalse(info["touches_envelope"])
        # Whole frame: the border wins and touches the frame, which the caller reports.
        mask, info = front.coarse_mask(gray, 70, front.resolve_envelope("auto", 80, 60))
        self.assertTrue(mask[55, 10] and info["touches_envelope"])
        self.assertTrue(front.coarse_mask(gray, 70, (0, 0, 80, 38))[1]["touches_envelope"])
        with self.assertRaises(ValueError):
            front.coarse_mask(np.full((10, 10), 200, np.uint8), 70, (0, 0, 10, 10))

    def test_threshold_is_strict_and_otsu_separates_two_levels(self):
        gray = np.full((20, 20), 180, np.uint8)
        gray[5:10, 5:10] = 70
        with self.assertRaises(ValueError):
            front.coarse_mask(gray, 70, (0, 0, 20, 20))
        self.assertEqual(int(front.coarse_mask(gray, 71, (0, 0, 20, 20))[0].sum()), 25)
        level = front.otsu_threshold(gray)
        self.assertTrue(70 < level <= 180)
        mask, info = front.coarse_mask(gray, "otsu", (0, 0, 20, 20))
        self.assertEqual((int(mask.sum()), info["threshold"]), (25, level))

    def test_contrast_matches_the_profile_used_for_the_dragon(self):
        self.assertEqual(front.gamma_table(0.5), [round(255 * math.sqrt(v / 255)) for v in range(256)])
        self.assertEqual(front.gamma_table(1.0), list(range(256)))
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 120, (64, 96, 3), dtype=np.uint8), "RGB")
        untouched = np.asarray(front.contrast_image(image, 1.0, 0, 8))
        np.testing.assert_array_equal(untouched, np.asarray(image))
        try:
            import cv2  # noqa: F401
            from scripts.mve_full.prepare import transform
        except ImportError:
            self.skipTest("OpenCV or the reference profile is not available")
        np.testing.assert_array_equal(np.asarray(front.contrast_image(image, 0.5, 2.0, 8)),
                                      np.asarray(transform(image, "gamma05_clahe2")))


class GateTests(unittest.TestCase):
    def test_clean_full_turn_passes(self):
        ring = front.ring_statistics(*orbit(np.arange(0, 360, 10)))
        self.assertFalse(ring["degenerate"])
        self.assertAlmostEqual(ring["fitted_radius"], 2.0, places=9)
        self.assertLess(ring["radius_spread_percent"], 1e-6)
        self.assertAlmostEqual(ring["largest_angular_gap_deg"], 10, places=6)
        self.assertAlmostEqual(ring["step_deg_per_capture"]["median"], 10, places=6)
        self.assertLess(ring["optical_axis_miss_percent"]["max"], 1e-6)  # elevated cameras looking down still hit the axis
        self.assertEqual(ring["cameras_looking_outward"], 0)
        self.assertEqual((ring["reversed_steps"], ring["duplicate_pose_pairs"]), (0, []))
        self.assertEqual(front.decide_gates(GOOD_AUDIT, ring, LIMITS), (True, []))

    def test_turning_direction_does_not_matter(self):
        ring = front.ring_statistics(*orbit(-np.arange(0, 360, 10)))
        self.assertAlmostEqual(ring["step_deg_per_capture"]["median"], 10, places=6)
        self.assertTrue(front.decide_gates(GOOD_AUDIT, ring, LIMITS)[0])

    def test_duplicate_frame_is_reported_but_not_a_failure(self):
        angles = np.concatenate([np.arange(0, 130, 10), np.arange(120, 360, 10)])  # 120 twice
        ring = front.ring_statistics(*orbit(angles))
        self.assertEqual(ring["duplicate_pose_pairs"], [[12, 13]])
        self.assertEqual(ring["reversed_steps"], 0)
        self.assertTrue(front.decide_gates(GOOD_AUDIT, ring, LIMITS)[0])

    def test_missing_views_use_capture_positions(self):
        indices, centres, rotations = orbit(np.arange(0, 360, 10))
        keep = [i for i in indices if i not in (3, 4)]
        ring = front.ring_statistics(keep, [centres[i] for i in keep], [rotations[i] for i in keep])
        self.assertAlmostEqual(ring["step_deg_per_capture"]["max"], 10, places=6)
        self.assertAlmostEqual(ring["largest_angular_gap_deg"], 30, places=6)

    def test_half_turn_and_reversal_and_scatter_fail_with_reasons(self):
        half = front.ring_statistics(*orbit(np.arange(0, 180, 5)))
        passed, reasons = front.decide_gates(GOOD_AUDIT, half, LIMITS)
        self.assertFalse(passed)
        self.assertTrue(any("largest gap" in r for r in reasons))
        self.assertTrue(front.decide_gates(GOOD_AUDIT, half, {**LIMITS, "maximum_angular_gap_deg": 360})[0])
        angles = np.arange(0, 360, 10).astype(float)
        angles[[20, 21]] = angles[[21, 20]]  # two photos swapped
        swapped = front.ring_statistics(*orbit(angles))
        self.assertEqual(swapped["reversed_steps"], 1)
        self.assertTrue(any("against the turning direction" in r for r in front.decide_gates(GOOD_AUDIT, swapped, LIMITS)[1]))
        noise = np.random.default_rng(1).normal(0, 0.15, (36, 3))
        scattered = front.ring_statistics(*orbit(np.arange(0, 360, 10), noise=noise))
        passed, reasons = front.decide_gates(GOOD_AUDIT, scattered, LIMITS)
        self.assertFalse(passed)
        self.assertTrue(any("radius spread" in r or "ring plane" in r for r in reasons))

    def test_outward_looking_cameras_fail(self):
        indices, centres, rotations = orbit(np.arange(0, 360, 10))
        flipped = [np.diag([-1.0, 1.0, -1.0]) @ r for r in rotations]
        ring = front.ring_statistics(indices, centres, flipped)
        self.assertEqual(ring["cameras_looking_outward"], 36)
        self.assertEqual(front.decide_gates(GOOD_AUDIT, ring, LIMITS), (False, ["36 cameras look away from the ring axis"]))
        yaw = np.deg2rad(20)  # every camera turned 20 degrees sideways: the axes miss the turntable axis
        turn = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
        ring = front.ring_statistics(indices, centres, [turn @ r for r in rotations])
        self.assertGreater(ring["optical_axis_miss_percent"]["min"], 25)
        self.assertTrue(any("misses the ring axis" in r for r in front.decide_gates(GOOD_AUDIT, ring, LIMITS)[1]))

    def test_audit_failures_and_degenerate_rings_fail(self):
        ring = front.ring_statistics(*orbit(np.arange(0, 360, 10)))
        audit = {**GOOD_AUDIT, "passed": False, "reasons": ["per-view reprojection exceeds declared bound"]}
        self.assertEqual(front.decide_gates(audit, ring, LIMITS),
                         (False, ["camera audit: per-view reprojection exceeds declared bound"]))
        low = {**GOOD_AUDIT, "coverage": 0.5, "registered_cameras": 18}
        self.assertTrue(any("registered 18 of 36" in r for r in front.decide_gates(low, ring, LIMITS)[1]))
        self.assertFalse(front.decide_gates({**GOOD_AUDIT, "passed": False}, ring, LIMITS)[0])
        stray = {**GOOD_AUDIT, "passed": False, "reasons": ["landmark behind camera"], "positive_depth_fraction": 0.99997}
        passed, reasons = front.decide_gates(stray, ring, LIMITS)  # a limit of 1.0 rejects one stray point
        self.assertFalse(passed)
        self.assertTrue(reasons[0].startswith("only 0.999970 of the sparse observations"))
        self.assertEqual(front.decide_gates(stray, ring, {**LIMITS, "minimum_positive_depth_fraction": 0.999}), (True, []))
        self.assertFalse(front.decide_gates({**stray, "positive_depth_fraction": 0.9}, ring,
                                            {**LIMITS, "minimum_positive_depth_fraction": 0.999})[0])
        few = front.ring_statistics(*orbit([0, 10, 20]))
        self.assertTrue(few["degenerate"])
        self.assertEqual(front.decide_gates(GOOD_AUDIT, few, LIMITS), (False, ["camera centres do not define a ring"]))
        line = front.ring_statistics(list(range(6)), [[i, 0, 0] for i in range(6)], [np.eye(3)] * 6)
        self.assertTrue(line["degenerate"])


class CommandTests(unittest.TestCase):
    def configuration(self, folder, **changes):
        folder = Path(folder)
        (folder / "photos").mkdir(exist_ok=True)
        for n in (1, 2, 10):
            (folder / "photos" / f"thing_{n}_rgb.png").write_bytes(b"")
        arguments = ["--photos", str(folder / "photos"), "--calibration", str(EXAMPLE), "--output", str(folder / "out"),
                     "--alicevision", str(folder / "av.py"), "--sam-python", "/sam/python", "--sam-source", "/sam/source",
                     "--sam-checkpoint", "/sam/tiny.pt", "--python", "/sci/python"]
        for name, value in changes.items():
            arguments += ["--" + name.replace("_", "-"), *([] if value is True else [str(value)])]
        return front.resolve(front.parser().parse_args(arguments))

    def test_natural_photo_order(self):
        names = ["x_10_rgb.png", "x_2_rgb.png", "x_1_rgb.png", "X_11_rgb.png"]
        self.assertEqual(sorted(names, key=front.natural_key), ["x_1_rgb.png", "x_2_rgb.png", "x_10_rgb.png", "X_11_rgb.png"])
        self.assertEqual(front.capture_name(7), "capture_0007.png")

    def test_wrapper_and_prefix_forms(self):
        with tempfile.TemporaryDirectory() as folder:
            command, environment = front.alicevision_command(Path(folder) / "av.py", "globalSfM", ["--input", 3], "/sci/python")
            self.assertEqual(command, ["/sci/python", str(Path(folder) / "av.py"), "globalSfM", "--input", "3"])
            self.assertEqual(environment, {})
            command, _ = front.alicevision_command(Path(folder) / "av", "cameraInit", [], "/sci/python")
            self.assertEqual(command, [str(Path(folder) / "av"), "cameraInit"])
            command, environment = front.alicevision_command(folder, "cameraInit", ["-x"], "/sci/python")
            self.assertEqual(command, [str(Path(folder) / "bin" / "aliceVision_cameraInit"), "-x"])
            self.assertEqual(environment["ALICEVISION_ROOT"], folder)
            self.assertTrue(environment["DYLD_LIBRARY_PATH"].startswith(str(Path(folder) / "lib")))
            self.assertEqual(front.sensor_database(folder), Path(folder) / "share/aliceVision/cameraSensors.db")
            self.assertEqual(front.sensor_database(Path(folder) / "av.py"), Path(folder) / "prefix/share/aliceVision/cameraSensors.db")
            self.assertEqual(front.sensor_database(folder, "/else/db"), Path("/else/db"))

    def test_defaults_reproduce_the_hand_driven_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self.configuration(folder)
            out = Path(cfg["output"])
            self.assertEqual((cfg["photo_count"], cfg["dark_threshold"], cfg["envelope"]), (3, 70, "auto"))
            commands = front.build_commands(cfg)
            self.assertEqual(list(commands), ["coarse", "sam", "cleanup", "publish-masks", "contrast",
                                              "cameraInit-uncalibrated", "featureExtraction", "cameraInit", "imageMatching",
                                              "featureMatching", "globalSfM", "audit", "prepareDenseScene",
                                              "verify-prepared", "inputs", "overlay"])
            sam = commands["sam"]["command"]
            self.assertEqual(sam[:3], ["/sam/python", "-m", "scripts.turntable_mesh.segment"])
            self.assertEqual(sam[sam.index("--device") + 1], "mps")
            self.assertEqual(sam[sam.index("--views") + 1], "3")
            self.assertEqual(sam[-3:], ["--multimask", "--preserve-holes", "--automatic-cues"])
            self.assertNotIn("--model-config", sam)
            self.assertEqual(commands["sam"]["environment"]["PYTHONPATH"], str(front.REPOSITORY))
            cleanup = commands["cleanup"]["command"]
            self.assertEqual(cleanup[-3:], ["--dark-object-bright-background", "--maximum-total-filled-foreground-fraction", "0.02"])
            features = commands["featureExtraction"]["command"]
            self.assertEqual(features[:3], ["/sci/python", str(Path(folder).absolute() / "av.py"), "featureExtraction"])
            for flag, value in (("--describerTypes", "sift"), ("--describerPreset", "normal"), ("--forceCpuExtraction", "true"),
                                ("--masksFolder", str(out / "masks")), ("--maxThreads", "2"), ("--maxCoresAvailable", "2"),
                                ("--maxMemoryAvailable", str(4 << 30))):
                self.assertEqual(features[features.index(flag) + 1], value)
            matching = commands["imageMatching"]["command"]
            self.assertEqual(matching[matching.index("--method") + 1], "Exhaustive")
            sfm = commands["globalSfM"]["command"]
            self.assertEqual(sfm[sfm.index("--lockAllIntrinsics") + 1], "true")
            self.assertEqual(sfm[sfm.index("--randomSeed") + 1], "0")
            self.assertEqual(sfm[sfm.index("--output") + 1], str(out / "sfm/final.sfm"))
            self.assertEqual(commands["globalSfM"]["environment"]["OPENBLAS_NUM_THREADS"], "1")
            prepare = commands["prepareDenseScene"]["command"]
            self.assertEqual(prepare[prepare.index("--outputFileType") + 1], "png")
            self.assertEqual(prepare[1], str(Path(folder).absolute() / "av.py"))  # falls back to --alicevision
            inputs = commands["inputs"]["command"]
            self.assertEqual(inputs[inputs.index("--raw-masks") + 1], str(out / "masks"))
            self.assertEqual(commands["audit"]["command"][-4:], ["--internal-step", "audit", "--output", str(out)])
            json.dumps(cfg)  # the configuration is what the internal steps read back

    def test_options_reach_the_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self.configuration(folder, device="cpu", threads=3, no_sam_multimask=True, sam_config="configs/x.yaml",
                                     sam_pythonpath="/a:/b", hole_cleanup_budget=0.05, matching_method="Sequential",
                                     random_seed=7, alicevision_dense=str(Path(folder) / "dense"), dark_threshold="otsu",
                                     envelope="0.1,0.1,0.9,0.9")
            commands = front.build_commands(cfg)
            sam = commands["sam"]["command"]
            self.assertEqual(sam[sam.index("--device") + 1], "cpu")
            self.assertEqual(sam[sam.index("--model-config") + 1], "configs/x.yaml")
            self.assertNotIn("--multimask", sam)
            self.assertTrue(commands["sam"]["environment"]["PYTHONPATH"].endswith("/a:/b"))
            self.assertEqual(commands["cleanup"]["command"][-1], "0.05")
            self.assertEqual(commands["imageMatching"]["command"][commands["imageMatching"]["command"].index("--method") + 1], "Sequential")
            self.assertEqual(commands["featureMatching"]["command"][commands["featureMatching"]["command"].index("--randomSeed") + 1], "7")
            self.assertIn("3", commands["globalSfM"]["command"][commands["globalSfM"]["command"].index("--maxCoresAvailable") + 1])
            self.assertEqual(commands["prepareDenseScene"]["command"][1], "prepareDenseScene")  # a non-.py wrapper runs directly
            self.assertEqual((cfg["dark_threshold"], cfg["envelope"]), ("otsu", "0.1,0.1,0.9,0.9"))

    def test_invalid_requests_are_refused_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                self.configuration(folder, dark_threshold="300")
            (Path(folder) / "bad.json").write_text('{"fx": 1}')
            with self.assertRaises(ValueError):
                self.configuration(folder, calibration=str(Path(folder) / "bad.json"))
            (Path(folder) / "two").mkdir()
            for name in ("a.png", "b.png", "notes.txt"):
                (Path(folder) / "two" / name).write_bytes(b"")
            with self.assertRaises(ValueError):
                self.configuration(folder, photos=str(Path(folder) / "two"))


if __name__ == "__main__":
    unittest.main()
