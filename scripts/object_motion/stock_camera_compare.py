#!/usr/bin/env python3
"""Post hoc stock COLMAP camera comparison against sealed Berkeley rig metadata.

This evaluation reads an image-only model; it never supplies poses to reconstruction.
The rig convention remains a diagnostic, not certified physical ground truth.
"""

import argparse
import json
from pathlib import Path

from scripts.object_motion import ycb_camera_reference as rig
from scripts.upstream_control import colmap_camera_compare as colmap

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_REPORT = ROOT / "build-opencv/object-motion/ycb-camera-reference-003/report.json"
REFERENCE_SHA256 = "744b50dfcbf66c0801fa9f36c3a7eac124ba09f3053909c10e7c647d585a723e"
EXPECTED_NAMES = {f"NP3_{angle:03}.jpg" for angle in rig.ANGLES}


def verified_metadata(reference_report, metadata_dir, expected_report_sha=REFERENCE_SHA256):
    reference_report, metadata_dir = Path(reference_report), Path(metadata_dir)
    if reference_report.is_symlink() or colmap.sha256(reference_report) != expected_report_sha:
        raise ValueError("sealed Berkeley reference report hash mismatch")
    report = json.loads(reference_report.read_text())
    if (report.get("schema") != "ycb_berkeley_camera_oracle_v1" or
            report.get("source_archive_sha256") != rig.ARCHIVE_SHA256 or
            report.get("google_mesh_cross_frame_transform_known") is not False):
        raise ValueError("unexpected Berkeley reference report provenance")
    members = report.get("metadata", {}).get("members", {})
    if set(members) != rig.MEMBERS or metadata_dir.is_symlink() or not metadata_dir.is_dir():
        raise ValueError("incomplete Berkeley rig metadata")
    total = 0
    for member in sorted(rig.MEMBERS):
        path = metadata_dir / member.removeprefix(rig.PREFIX)
        record = members[member]
        if (path.is_symlink() or not path.is_file() or
                path.stat().st_size != record.get("bytes") or
                colmap.sha256(path) != record.get("sha256")):
            raise ValueError(f"Berkeley rig metadata hash mismatch: {member}")
        total += path.stat().st_size
    if total != report["metadata"].get("total_uncompressed_bytes"):
        raise ValueError("Berkeley rig metadata byte count mismatch")
    return {"report_sha256": expected_report_sha,
            "metadata_sha256": {name: members[name]["sha256"] for name in sorted(members)},
            "total_uncompressed_bytes": total}


def evaluate(model, output, *, reference_report=REFERENCE_REPORT, metadata_dir=None,
             expected_report_sha=REFERENCE_SHA256):
    model, output, reference_report = map(Path, (model, output, reference_report))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if metadata_dir is None:
        metadata_dir = reference_report.parent / "metadata"
    evidence = verified_metadata(reference_report, metadata_dir, expected_report_sha)
    source, model_hashes, software = colmap.read_candidate(model)
    if not 4 <= len(source) <= 60 or not set(source) <= EXPECTED_NAMES:
        raise ValueError("candidate needs 4–60 distinct selected NP3 cameras")
    centers, rotations, _, _ = rig.reference_cameras(metadata_dir)
    if set(centers) != EXPECTED_NAMES or set(rotations) != EXPECTED_NAMES:
        raise ValueError("rig camera names differ from the 60-photo selection")
    target = {name: (centers[name], rotations[name]) for name in EXPECTED_NAMES}
    comparison = colmap.compare(source, target)
    if model_hashes != {name: colmap.sha256(model / name) for name in model_hashes}:
        raise ValueError("candidate model changed during comparison")
    if verified_metadata(reference_report, metadata_dir, expected_report_sha) != evidence:
        raise ValueError("Berkeley rig metadata changed during comparison")
    software_paths = {"adapter": Path(__file__), "colmap_compare": Path(colmap.__file__),
                      "rig_reader": Path(rig.__file__)}
    result = {
        "schema": "ycb_stock_berkeley_camera_diagnostic_v1",
        "interpretation": "evaluation-only named-camera and trajectory agreement after fitted Sim(3); Berkeley pose convention is not independently reprojected ground truth; no Google mesh transform or metric mesh accuracy claim",
        "candidate_model_dir": str(model.resolve()),
        "candidate_model_sha256": model_hashes,
        "reference_evidence": evidence,
        "software": {**software, **{name + "_sha256": colmap.sha256(path) for name, path in software_paths.items()}},
        "coverage": {"registered": len(source), "selected": 60,
                     "fraction": len(source) / 60,
                     "missing_names": comparison["missing_reference_names"]},
        "comparison": comparison,
        "google_mesh_cross_frame_transform_known": False,
        "metric_mesh_accuracy_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.model, args.report)
    print(json.dumps({"report": str(args.report), "coverage": result["coverage"],
                      "center_rms_over_reference_radius": result["comparison"]["center_rms_over_reference_radius"],
                      "orientation_p95_degrees": result["comparison"]["orientation_p95_degrees"]}))


if __name__ == "__main__":
    main()
