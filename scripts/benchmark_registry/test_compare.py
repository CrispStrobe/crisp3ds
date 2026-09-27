import copy
import json
import unittest

from scripts.benchmark_registry import compare, validate


REGISTRY = json.loads(validate.DEFAULT.read_text())
HASH = "a" * 64
PROTOCOL = "b" * 64
REFERENCE = "c" * 64


def contract(run):
    return {"result_sha256": run["result_sha256"],
            "provenance_sha256": run["provenance_sha256"],
            "image_set_sha256": HASH, "mask_set_sha256": "none",
            "intrinsics_policy": "estimated_from_images",
            "input_policy": "original_rgb_same_decode",
            "reference_sha256": REFERENCE,
            "metric_protocol_sha256": PROTOCOL,
            "outcome": "success", "score": 0.4}


def matching_pair(scope="independent_geometry"):
    registry = copy.deepcopy(REGISTRY)
    pipeline = registry["pipelines"][0]
    first = pipeline["runs"][0]
    second = copy.deepcopy(first)
    second["id"] = "matched_control"
    second["result_sha256"] = "d" * 64
    pipeline["runs"].append(second)
    first["reference_scope"] = second["reference_scope"] = scope
    kind = compare.SCOPE_KIND[scope]
    pipeline["evidence"] = [{"kind": kind, "metric_target": "mesh",
                             "doc_ref": "docs/YCB-COMPARISON.md", "scope": "test",
                             "run_ids": [first["id"], second["id"]]}]
    left = f"{pipeline['id']}/{first['id']}"
    right = f"{pipeline['id']}/{second['id']}"
    return registry, left, right, {left: contract(first), right: contract(second)}


class ComparisonTests(unittest.TestCase):
    def test_current_registered_runs_cannot_claim_head_to_head(self):
        left = "colmap_openmvs/ycb_native_masked_008"
        right = "mve/bunny_contrast_finalized_v2"
        report = compare.assess_pair(REGISTRY, left, right)
        self.assertFalse(report["comparable"])
        self.assertEqual(report["lane"], "not_comparable")
        self.assertIn("dataset_id", report["differences"])
        self.assertIn("input_profile", report["differences"])
        self.assertTrue(any("contract unavailable" in item for item in report["blockers"]))

    def test_supplied_camera_and_image_estimated_are_separate(self):
        registry = copy.deepcopy(REGISTRY)
        colmap = registry["pipelines"][0]
        mve = registry["pipelines"][1]
        left_run, right_run = colmap["runs"][0], mve["runs"][0]
        for key in ("dataset_id", "input_profile", "artifact_type", "stage",
                    "composition", "reference_scope"):
            right_run[key] = left_run[key]
        right_run["camera_lane"] = "supplied_camera"
        mve["evidence"][0]["run_ids"] = [right_run["id"]]
        left = f"{colmap['id']}/{left_run['id']}"
        right = f"{mve['id']}/{right_run['id']}"
        contracts = {left: contract(left_run), right: contract(right_run)}
        report = compare.assess_pair(registry, left, right, contracts)
        self.assertFalse(report["comparable"])
        self.assertEqual(report["differences"]["camera_lane"][right], "supplied_camera")

    def test_exact_inputs_and_metric_protocol_must_match(self):
        registry, left, right, contracts = matching_pair()
        contracts[right]["image_set_sha256"] = "e" * 64
        contracts[right]["metric_protocol_sha256"] = "f" * 64
        report = compare.assess_pair(registry, left, right, contracts)
        self.assertFalse(report["comparable"])
        self.assertIn("image_set_sha256", report["differences"])
        self.assertIn("metric_protocol_sha256", report["differences"])

    def test_contract_must_bind_registered_artifacts(self):
        registry, left, right, contracts = matching_pair()
        contracts[right]["provenance_sha256"] = "f" * 64
        report = compare.assess_pair(registry, left, right, contracts)
        self.assertFalse(report["comparable"])
        self.assertTrue(any("does not bind" in item for item in report["blockers"]))

    def test_failure_cannot_be_successful_zero_quality(self):
        registry, left, right, contracts = matching_pair()
        contracts[right]["outcome"] = "failure"
        contracts[right]["score"] = 0
        report = compare.assess_pair(registry, left, right, contracts)
        self.assertFalse(report["comparable"])
        self.assertTrue(any("must have null score" in item for item in report["blockers"]))
        self.assertIsNone(report["outcomes"][right]["quality_score"])

    def test_matched_independent_shape_still_not_certified_accuracy(self):
        registry, left, right, contracts = matching_pair()
        report = compare.assess_pair(registry, left, right, contracts)
        self.assertTrue(report["comparable"])
        self.assertEqual(report["lane"], "conditional_shape_diagnostic")
        self.assertFalse(report["supports_physical_accuracy_claim"])
        for selector in (left, right):
            compare.find_run(registry, selector)[1]["composition"] = "fresh"
        report = compare.assess_pair(registry, left, right, contracts)
        self.assertEqual(report["lane"], "end_to_end_shape_diagnostic")
        self.assertFalse(report["supports_physical_accuracy_claim"])

    def test_software_agreement_is_comparable_but_not_physical_truth(self):
        registry, left, right, contracts = matching_pair("software_oracle")
        report = compare.assess_pair(registry, left, right, contracts)
        self.assertTrue(report["comparable"])
        self.assertEqual(report["lane"], "software_agreement")
        self.assertFalse(report["supports_physical_accuracy_claim"])
        contracts[left]["supports_physical_accuracy_claim"] = True
        self.assertFalse(compare.assess_pair(registry, left, right, contracts)
                         ["supports_physical_accuracy_claim"])


if __name__ == "__main__":
    unittest.main()
