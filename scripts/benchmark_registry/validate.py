"""Validate claims and references in benchmarks/pipeline-evidence.json.

Ignored local result files need not exist on CI, but their recorded SHA-256 is
mandatory and is checked whenever the file is available. Tracked documentation
references must exist. This registry stores no numerical score or rank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "benchmarks" / "pipeline-evidence.json"
KINDS = ("measured_geometry", "sensor_diagnostic", "software_oracle",
         "analytic_golden", "rendering")
STATES = {"planned", "audited", "executed"}
ARTIFACTS = {"mesh", "splats"}
LANES = {"image_estimated", "calibrated_intrinsics", "supplied_camera", "unverified"}
COMPOSITIONS = {"fresh", "composed", "partial"}
STAGES = {"rough_mesh", "refined_mesh", "cleaned_mesh", "splats"}
REFERENCES = {"independent_geometry", "sensor_depth", "software_oracle", "none"}
TARGETS = {"mesh", "splats", "none"}
HASH = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*\Z")


def fail(where: str, message: str) -> None:
    raise ValueError(f"{where}: {message}")


def keys(value: object, required: set[str], optional: set[str], where: str) -> dict:
    if not isinstance(value, dict):
        fail(where, "expected object")
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        fail(where, f"missing {sorted(missing)}; unsupported {sorted(extra)}")
    return value


def identifier(value: object, where: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        fail(where, "expected lowercase snake_case identifier")
    return value


def nonempty(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "\n" in value:
        fail(where, "expected nonempty single-line text")
    return value


def sha256(value: object, where: str) -> str:
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        fail(where, "expected lowercase SHA-256")
    return value


def reference(value: object, root: Path, where: str, suffix: str, required_on_disk: bool,
              expected_sha256: str | None = None) -> Path:
    if not isinstance(value, str) or "\\" in value or "//" in value:
        fail(where, "expected portable repository-relative path")
    relative = PurePosixPath(value)
    if (relative.is_absolute() or not relative.parts or
            any(part in (".", "..") for part in value.split("/")) or
            relative.as_posix() != value or relative.suffix != suffix):
        fail(where, "unsafe path or wrong suffix")
    path = root / Path(*relative.parts)
    if not path.resolve(strict=False).is_relative_to(root.resolve()):
        fail(where, "reference resolves outside repository")
    if any((root / Path(*relative.parts[:depth])).is_symlink()
           for depth in range(1, len(relative.parts) + 1)):
        fail(where, "symlink reference is unsupported")
    if required_on_disk and not path.is_file():
        fail(where, "tracked documentation does not exist")
    if path.exists():
        if not path.is_file():
            fail(where, "reference is not a regular file")
        if expected_sha256 is not None:
            if path.stat().st_size > (32 << 20):
                fail(where, "referenced JSON exceeds 32 MiB hash bound")
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    hasher.update(chunk)
            actual = hasher.hexdigest()
            if actual != expected_sha256:
                fail(where, "local artifact SHA-256 differs from registry")
    return path


def validate_data(data: object, root: Path = ROOT) -> dict:
    root = root.resolve()
    doc = keys(data, {"schema", "evidence_types", "pipelines"}, set(), "registry")
    if doc["schema"] != "pipeline_evidence_v1" or doc["evidence_types"] != list(KINDS):
        fail("registry", "schema or evidence taxonomy differs")
    if not isinstance(doc["pipelines"], list) or not doc["pipelines"]:
        fail("registry.pipelines", "expected nonempty list")
    seen_pipelines: set[str] = set()
    for index, raw in enumerate(doc["pipelines"]):
        where = f"pipelines[{index}]"
        pipeline = keys(raw, {"id", "name", "execution_state", "artifact_type",
                              "camera_lanes", "doc_refs", "runs", "evidence"},
                        {"audit_scope"}, where)
        pipeline_id = identifier(pipeline["id"], f"{where}.id")
        if pipeline_id in seen_pipelines:
            fail(where, "duplicate pipeline id")
        seen_pipelines.add(pipeline_id)
        nonempty(pipeline["name"], f"{where}.name")
        state, artifact = pipeline["execution_state"], pipeline["artifact_type"]
        if state not in STATES or artifact not in ARTIFACTS:
            fail(where, "unknown execution state or artifact type")
        if (not isinstance(pipeline["camera_lanes"], list) or
                not pipeline["camera_lanes"] or
                len(set(pipeline["camera_lanes"])) != len(pipeline["camera_lanes"]) or
                any(lane not in LANES for lane in pipeline["camera_lanes"])):
            fail(where, "invalid camera lanes")
        if not isinstance(pipeline["doc_refs"], list) or not pipeline["doc_refs"]:
            fail(where, "status needs tracked documentation")
        for j, path in enumerate(pipeline["doc_refs"]):
            reference(path, root, f"{where}.doc_refs[{j}]", ".md", True)
        if not isinstance(pipeline["runs"], list) or not isinstance(pipeline["evidence"], list):
            fail(where, "runs and evidence must be lists")
        if state == "audited":
            if pipeline.get("audit_scope") not in ("source_build_graph", "documentation_only"):
                fail(where, "audited pipeline needs a precise audit scope")
        elif "audit_scope" in pipeline:
            fail(where, "audit scope is reserved for audited pipelines")
        if state != "executed" and (pipeline["runs"] or pipeline["evidence"] or
                                     pipeline["camera_lanes"] != ["unverified"]):
            fail(where, "unrun pipeline cannot claim runs, evidence or verified camera lanes")
        if state == "executed" and not pipeline["runs"]:
            fail(where, "executed pipeline needs at least one referenced result")
        run_ids: set[str] = set()
        run_lanes: set[str] = set()
        run_scopes: dict[str, str] = {}
        for j, raw_run in enumerate(pipeline["runs"]):
            run_where = f"{where}.runs[{j}]"
            run = keys(raw_run, {"id", "dataset_id", "input_profile", "camera_lane",
                                 "artifact_type", "stage", "composition", "reference_scope",
                                 "result_ref", "result_sha256", "provenance_ref",
                                 "provenance_sha256"}, set(), run_where)
            run_id = identifier(run["id"], f"{run_where}.id")
            if run_id in run_ids:
                fail(run_where, "duplicate run id")
            run_ids.add(run_id)
            for field in ("dataset_id", "input_profile"):
                identifier(run[field], f"{run_where}.{field}")
            lane = run["camera_lane"]
            if (lane not in LANES - {"unverified"} or run["artifact_type"] != artifact or
                    run["stage"] not in STAGES or run["composition"] not in COMPOSITIONS or
                    run["reference_scope"] not in REFERENCES):
                fail(run_where, "invalid comparability field")
            if (artifact == "splats" and run["stage"] != "splats") or (artifact == "mesh" and run["stage"] == "splats"):
                fail(run_where, "artifact and stage disagree")
            run_lanes.add(lane)
            run_scopes[run_id] = run["reference_scope"]
            for role in ("result", "provenance"):
                sha = sha256(run[f"{role}_sha256"], f"{run_where}.{role}_sha256")
                reference(run[f"{role}_ref"], root, f"{run_where}.{role}_ref", ".json", False, sha)
        if state == "executed" and set(pipeline["camera_lanes"]) != run_lanes:
            fail(where, "camera lanes must match referenced executions exactly")
        for j, raw_evidence in enumerate(pipeline["evidence"]):
            evidence_where = f"{where}.evidence[{j}]"
            entry = keys(raw_evidence, {"kind", "metric_target", "doc_ref", "scope", "run_ids"},
                         {"mesh_extraction_ref", "mesh_extraction_sha256"}, evidence_where)
            if entry["kind"] not in KINDS or entry["metric_target"] not in TARGETS:
                fail(evidence_where, "unknown evidence kind or target")
            reference(entry["doc_ref"], root, f"{evidence_where}.doc_ref", ".md", True)
            nonempty(entry["scope"], f"{evidence_where}.scope")
            if (not isinstance(entry["run_ids"], list) or
                    any(run_id not in run_ids for run_id in entry["run_ids"]) or
                    len(entry["run_ids"]) != len(set(entry["run_ids"]))):
                fail(evidence_where, "evidence references unknown or duplicate run IDs")
            if entry["kind"] in ("measured_geometry", "sensor_diagnostic") and not entry["run_ids"]:
                fail(evidence_where, "physical evidence must identify a run")
            if entry["kind"] == "measured_geometry" and entry["metric_target"] != "mesh":
                fail(evidence_where, "measured geometry requires mesh target")
            required_scope = {"measured_geometry": "independent_geometry",
                              "sensor_diagnostic": "sensor_depth",
                              "software_oracle": "software_oracle"}.get(entry["kind"])
            if required_scope and any(run_scopes[run_id] != required_scope for run_id in entry["run_ids"]):
                fail(evidence_where, "evidence kind disagrees with referenced run scope")
            extraction = {"mesh_extraction_ref", "mesh_extraction_sha256"} & entry.keys()
            if extraction and extraction != {"mesh_extraction_ref", "mesh_extraction_sha256"}:
                fail(evidence_where, "mesh extraction needs both path and SHA-256")
            if artifact == "splats" and entry["metric_target"] == "mesh" and not extraction:
                fail(evidence_where, "splat mesh metrics require a hash-bound extraction")
            if extraction:
                sha = sha256(entry["mesh_extraction_sha256"], f"{evidence_where}.mesh_extraction_sha256")
                reference(entry["mesh_extraction_ref"], root, f"{evidence_where}.mesh_extraction_ref",
                          ".json", False, sha)
    return {"schema": doc["schema"], "pipelines": len(seen_pipelines),
            "executed": sum(item["execution_state"] == "executed" for item in doc["pipelines"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = validate_data(json.loads(args.registry.read_text()), ROOT)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
