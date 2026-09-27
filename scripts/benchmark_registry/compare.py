"""Check whether two registered runs support a like-for-like comparison.

Optional contracts describe exact inputs and metric protocol. They must name the
registered result/provenance hashes; missing contracts never imply equivalence.
This is a readiness check, not a scorer or a physical-accuracy certificate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.benchmark_registry import validate


MATCH_FIELDS = ("image_set_sha256", "mask_set_sha256", "intrinsics_policy",
                "input_policy", "reference_sha256", "metric_protocol_sha256")
HASH_FIELDS = {"image_set_sha256", "reference_sha256", "metric_protocol_sha256"}
SCOPE_KIND = {"independent_geometry": "measured_geometry",
              "sensor_depth": "sensor_diagnostic",
              "software_oracle": "software_oracle"}


def find_run(registry: dict, selector: str) -> tuple[dict, dict]:
    """Resolve pipeline/run without making run IDs globally unique."""
    if selector.count("/") != 1:
        raise ValueError("run selector must be pipeline_id/run_id")
    pipeline_id, run_id = selector.split("/")
    for pipeline in registry["pipelines"]:
        if pipeline["id"] == pipeline_id:
            for run in pipeline["runs"]:
                if run["id"] == run_id:
                    return pipeline, run
    raise ValueError(f"registered run not found: {selector}")


def evidence_kind(pipeline: dict, run: dict) -> str | None:
    expected = SCOPE_KIND.get(run["reference_scope"])
    if expected is None:
        return None
    if any(entry["kind"] == expected and entry["metric_target"] == run["artifact_type"]
           and run["id"] in entry["run_ids"]
           for entry in pipeline["evidence"]):
        return expected
    return None


def contract_issues(run: dict, contract: object, selector: str) -> list[str]:
    if not isinstance(contract, dict):
        return [f"{selector}: exact comparison contract unavailable"]
    issues = []
    for key in ("result_sha256", "provenance_sha256"):
        if contract.get(key) != run[key]:
            issues.append(f"{selector}: {key} does not bind to registered run")
    for key in MATCH_FIELDS:
        value = contract.get(key)
        if key == "mask_set_sha256" and value == "none":
            continue
        if key in HASH_FIELDS or key == "mask_set_sha256":
            if not isinstance(value, str) or validate.HASH.fullmatch(value) is None:
                issues.append(f"{selector}: {key} requires exact SHA-256")
        elif not isinstance(value, str) or not value.strip() or value != value.strip():
            issues.append(f"{selector}: {key} is missing")
    outcome = contract.get("outcome")
    score = contract.get("score")
    if not isinstance(outcome, str) or outcome not in {"success", "failure", "unavailable"}:
        issues.append(f"{selector}: outcome must be success, failure or unavailable")
    elif outcome != "success":
        issues.append(f"{selector}: outcome is {outcome}; no quality score is available")
        if score is not None:
            issues.append(f"{selector}: {outcome} must have null score")
    if outcome == "success" and (isinstance(score, bool) or
                                  not isinstance(score, (int, float)) or
                                  not math.isfinite(score)):
        issues.append(f"{selector}: successful comparison requires a finite score")
    return issues


def assess_pair(registry: dict, left: str, right: str,
                contracts: dict | None = None) -> dict:
    """Return explicit blockers and differences; registry must be validated first."""
    if left == right:
        raise ValueError("comparison requires two distinct registered runs")
    contracts = contracts or {}
    if not isinstance(contracts, dict):
        raise ValueError("contracts must be an object keyed by pipeline_id/run_id")
    left_pipeline, left_run = find_run(registry, left)
    right_pipeline, right_run = find_run(registry, right)
    left_contract, right_contract = contracts.get(left), contracts.get(right)
    blockers = contract_issues(left_run, left_contract, left)
    blockers += contract_issues(right_run, right_contract, right)
    differences = {}
    for key in ("dataset_id", "input_profile", "camera_lane", "artifact_type",
                "stage", "composition", "reference_scope"):
        if left_run[key] != right_run[key]:
            differences[key] = {left: left_run[key], right: right_run[key]}
    if isinstance(left_contract, dict) and isinstance(right_contract, dict):
        for key in MATCH_FIELDS:
            if left_contract.get(key) != right_contract.get(key):
                differences[key] = {left: left_contract.get(key),
                                    right: right_contract.get(key)}
    matched_fields = ("dataset_id", "input_profile", "camera_lane", "artifact_type",
                      "stage", "reference_scope", *MATCH_FIELDS)
    for key in matched_fields:
        if key in differences:
            blockers.append(f"different {key}")
    kinds = {left: evidence_kind(left_pipeline, left_run),
             right: evidence_kind(right_pipeline, right_run)}
    for selector, kind in kinds.items():
        if kind is None:
            blockers.append(f"{selector}: no run-bound evidence for its reference scope")
    if kinds[left] != kinds[right]:
        blockers.append("different evidence kinds")
    if left_run["reference_scope"] == "none" or right_run["reference_scope"] == "none":
        blockers.append("no reference for a quality comparison")
    comparable = not blockers
    if not comparable:
        lane = "not_comparable"
    elif kinds[left] == "software_oracle":
        lane = "software_agreement"
    elif kinds[left] == "sensor_diagnostic":
        lane = "sensor_depth_diagnostic"
    elif (left_run["camera_lane"] == "image_estimated" and
          left_run["composition"] == right_run["composition"] == "fresh"):
        lane = "end_to_end_shape_diagnostic"
    else:
        lane = "conditional_shape_diagnostic"
    return {"left": left, "right": right, "comparable": comparable,
            "lane": lane, "supports_physical_accuracy_claim": False,
            "evidence_kind": kinds, "differences": differences, "blockers": blockers,
            "outcomes": {selector: {"status": contract.get("outcome"),
                                    "quality_score": (contract.get("score") if
                                                      contract.get("outcome") == "success" and
                                                      isinstance(contract.get("score"), (int, float)) and
                                                      not isinstance(contract.get("score"), bool) and
                                                      math.isfinite(contract["score"]) else None)}
                         if isinstance(contract, dict) else {"status": "unavailable",
                                                           "quality_score": None}
                         for selector, contract in ((left, left_contract), (right, right_contract))},
            "comparison_basis": "When present, exact input and metric fields are run-hash-bound contract assertions.",
            "qualification": ("Independent shape evidence is a diagnostic until registration "
                              "and physical scale are independently validated.")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="pipeline_id/run_id")
    parser.add_argument("right", help="pipeline_id/run_id")
    parser.add_argument("--registry", type=Path, default=validate.DEFAULT)
    parser.add_argument("--contracts", type=Path, help="JSON object keyed by pipeline_id/run_id")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    validate.validate_data(registry, validate.ROOT)
    contracts = json.loads(args.contracts.read_text()) if args.contracts else None
    report = assess_pair(registry, args.left, args.right, contracts)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["comparable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
