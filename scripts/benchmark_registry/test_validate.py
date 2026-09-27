import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_registry import validate


REGISTRY = json.loads(validate.DEFAULT.read_text())


class RegistryTests(unittest.TestCase):
    def test_checked_in_registry_has_eight_honest_states(self):
        result = validate.validate_data(REGISTRY)
        self.assertEqual(result["pipelines"], 8)
        self.assertEqual(result["executed"], 2)
        self.assertEqual({p["execution_state"] for p in REGISTRY["pipelines"]},
                         {"planned", "audited", "executed"})
        metal = next(p for p in REGISTRY["pipelines"] if p["id"] == "mtl_alicevision")
        self.assertEqual(metal["execution_state"], "audited")
        self.assertEqual(metal["runs"], [])

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
        self.assertEqual(validate.validate_data(data)["executed"], 2)
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
