import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_registry import validate


REGISTRY = json.loads(validate.DEFAULT.read_text())


class RegistryTests(unittest.TestCase):
    def test_checked_in_registry_has_honest_states_and_bounded_oracles(self):
        result = validate.validate_data(REGISTRY)
        self.assertEqual(result["pipelines"], 11)
        self.assertEqual(result["executed"], 3)
        self.assertEqual({p["execution_state"] for p in REGISTRY["pipelines"]},
                         {"planned", "audited", "executed"})
        metal = next(p for p in REGISTRY["pipelines"] if p["id"] == "mtl_alicevision")
        self.assertEqual(metal["execution_state"], "audited")
        self.assertEqual(metal["runs"], [])
        seedexr = next(p for p in REGISTRY["pipelines"] if p["id"] == "seedexr_alicevision_mac")
        self.assertEqual(seedexr["execution_state"], "audited")
        self.assertEqual(seedexr["runs"], [])
        self.assertIn("docs/ALICEVISION-MAC-ORACLE.md", seedexr["doc_refs"])
        classical = next(p for p in REGISTRY["pipelines"] if p["id"] == "colmap_openmvs")
        fresh = next(run for run in classical["runs"] if run["id"] == "ycb_turntable_fresh_005")
        summary = json.loads((validate.ROOT / fresh["result_ref"]).read_text())
        self.assertEqual(fresh["composition"], "composed")
        self.assertEqual(summary["artifacts"]["result_sha256"],
                         "dafc6ccb2a90f0c5a45748291571455237acd8ae08da01dd80adc10222a3625e")
        self.assertEqual(summary["artifacts"]["score_sha256"],
                         "165b8d5169e8bdb917f008864b01c434f4e03455b6ffe688d8ae8a824b2847c9")
        self.assertEqual(summary["artifacts"]["sparse_report_sha256"],
                         "d302084d806989d86666030c060f64380952f5503162a4b4aad295d04b846571")
        self.assertEqual(summary["artifacts"]["camera_gate_sha256"],
                         "1656ac8aea32650953f4ec83413cd9022f28ad85ef69cf91af0c46026f868214")
        self.assertAlmostEqual(summary["score"]["reference_fitted_sim3"], 0.4223070276497696)
        self.assertAlmostEqual(summary["score"]["transported_006_gauge"], 0.3091683783255086)
        self.assertFalse(summary["result"]["quality_accepted"])
        self.assertFalse(summary["result"]["shipping_approved"])
        brush = next(p for p in REGISTRY["pipelines"] if p["id"] == "brush")
        self.assertEqual(brush["evidence"], [])
        self.assertEqual(brush["runs"][0]["reference_scope"], "none")
        self.assertEqual(brush["runs"][0]["artifact_type"], "splats")

    def test_executed_requires_hash_bound_result_and_comparability(self):
        data = copy.deepcopy(REGISTRY)
        del data["pipelines"][0]["runs"][0]["result_sha256"]
        with self.assertRaisesRegex(ValueError, "result_sha256"):
            validate.validate_data(data)
        data = copy.deepcopy(REGISTRY)
        del data["pipelines"][0]["runs"][0]["camera_lane"]
        with self.assertRaisesRegex(ValueError, "camera_lane"):
            validate.validate_data(data)

    def test_missing_ignored_result_is_allowed_but_hash_is_still_required(self):
        data = copy.deepcopy(REGISTRY)
        data["pipelines"][0]["runs"][0]["result_ref"] = "build-opencv/not-checked-in/result.json"
        self.assertEqual(validate.validate_data(data)["executed"], 3)
        data["pipelines"][0]["runs"][0]["result_sha256"] = "unknown"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate.validate_data(data)

    def test_local_result_hash_mismatch_rejected(self):
        data = copy.deepcopy(REGISTRY)
        data["pipelines"][0]["runs"][0]["result_sha256"] = "0" * 64
        if not (validate.ROOT / data["pipelines"][0]["runs"][0]["result_ref"]).exists():
            self.skipTest("ignored local result unavailable on CI")
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            validate.validate_data(data)

    def test_physical_evidence_scope_and_splat_mesh_extraction(self):
        data = copy.deepcopy(REGISTRY)
        data["pipelines"][0]["runs"][0]["reference_scope"] = "software_oracle"
        with self.assertRaisesRegex(ValueError, "kind disagrees"):
            validate.validate_data(data)
        data = copy.deepcopy(REGISTRY)
        pipeline = data["pipelines"][0]
        pipeline["artifact_type"] = "splats"
        for run in pipeline["runs"]:
            run["artifact_type"] = "splats"
            run["stage"] = "splats"
        with self.assertRaisesRegex(ValueError, "mesh metrics require"):
            validate.validate_data(data)

    def test_no_aggregate_rank_or_results_in_planned_lane(self):
        data = copy.deepcopy(REGISTRY)
        data["aggregate_rank"] = ["mve"]
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate.validate_data(data)
        data = copy.deepcopy(REGISTRY)
        data["pipelines"][-1]["runs"] = [data["pipelines"][0]["runs"][0]]
        with self.assertRaisesRegex(ValueError, "unrun pipeline"):
            validate.validate_data(data)

    def test_parent_symlink_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, outside = base / "repo", base / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "claim.md").write_text("external")
            try:
                (root / "docs").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks unavailable")
            with self.assertRaisesRegex(ValueError, "outside repository"):
                validate.reference("docs/claim.md", root, "escape", ".md", True)


if __name__ == "__main__":
    unittest.main()
