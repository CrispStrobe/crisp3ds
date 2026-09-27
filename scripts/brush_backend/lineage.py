"""Exact sealed lineage for the existing YCB-008 undistorted Brush input."""
from __future__ import annotations

import json
from pathlib import Path

from .preflight import sha256

ROOT = Path(__file__).resolve().parents[2]
RUN_008 = ROOT / "build-opencv/classical-ycb-native-masked-008"
RUN_002 = ROOT / "build-opencv/classical-ycb-foreground-002"
PRODUCER = ROOT / "build-opencv/object-motion/foreground"
MANIFEST = ROOT / "build-opencv/object-motion/prepare-001/manifest.json"
EXPECTED = {
    "masked_result": "8f27196add8d006691103dc3ddff33336c4a62bb86cf08b46d0c52e74614c3d9",
    "mask_report": "07764a781446a91978ba7fa00e6fb5f0b5e64b5e4d0ca87f67cfa3d5770c60d3",
    "source_result": "223c38ed4655178cb38f119b5645d1fd13b5754ec50f638a4ba5c397c2613102",
    "producer_summary": "cba4dc6c6acb311538718b7d39220495f0ee3988b1de3ff8c9b9e90a9a21c12d",
    "producer_provenance": "7bcabaf83904f35d89109dd0bea86b442a89a467a1293cbbdec6274792787050",
    "input_manifest": "0b78469039d11516d0ac96b642553099267cd97c9a6439a1ec4b2fcb133615bd",
}


def _must(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _model_hashes(path: Path) -> dict[str, str]:
    return {name: sha256(path / name) for name in ("cameras.bin", "images.bin", "points3D.bin")}


def validate_008(dataset: Path, image_hashes: dict[str, str], model_hashes: dict[str, str],
                 *, run_008: Path = RUN_008, run_002: Path = RUN_002,
                 producer: Path = PRODUCER, manifest_path: Path = MANIFEST,
                 expected: dict[str, str] = EXPECTED) -> dict:
    _must(dataset.resolve() == (run_008 / "dense").resolve(), "dataset is not sealed YCB-008")
    files = {
        "masked_result": run_008 / "result.json",
        "mask_report": run_008 / "masks/report.json",
        "source_result": run_002 / "result.json",
        "producer_summary": producer / "summary.json",
        "producer_provenance": producer / "provenance.json",
        "input_manifest": manifest_path,
    }
    for key, path in files.items():
        _must(path.is_file() and not path.is_symlink() and sha256(path) == expected[key],
              f"sealed evidence mismatch: {key}")
    current = json.loads(files["masked_result"].read_text())
    masks = json.loads(files["mask_report"].read_text())
    source = json.loads(files["source_result"].read_text())
    summary = json.loads(files["producer_summary"].read_text())
    _must(current.get("schema") == "classical_masked_dense_v1" and current.get("status") == "complete",
          "YCB-008 result not completed")
    _must(current.get("mask_report_sha256") == expected["mask_report"] and
          current.get("source_result_sha256") == expected["source_result"], "YCB-008 report links mismatch")
    _must(masks.get("schema") == "classical_dense_masks_v1" and masks.get("status") == "complete" and
          masks.get("max_pose_matrix_difference", 1) <= 1e-8, "mask/camera audit mismatch")
    _must(masks.get("manifest_sha256") == expected["input_manifest"], "mask manifest link mismatch")
    _must(masks.get("undistorted_model_sha256") == model_hashes and
          _model_hashes(run_002 / "dense/sparse") == model_hashes,
          "undistorted model differs from sealed mask/source evidence")
    _must(masks.get("source_model_sha256") == source.get("sfm_source", {}).get("files_sha256") and
          _model_hashes(run_002 / "sparse/0") == masks["source_model_sha256"],
          "source model differs from SfM producer")
    _must(source.get("sfm_source", {}).get("producer_result_sha256") == expected["producer_summary"] and
          source.get("sfm_source", {}).get("provenance_sha256") == expected["producer_provenance"] and
          source.get("sfm_source", {}).get("manifest_sha256") == expected["input_manifest"],
          "SfM producer links mismatch")
    _must(summary.get("registered") == 60 and
          summary.get("model_files_sha256") == masks["source_model_sha256"] and
          summary.get("producer_provenance_sha256") == expected["producer_provenance"],
          "producer model evidence mismatch")
    _must(next((x.get("status") for x in source.get("stages", [])
                if x.get("name") == "undistort"), None) == "complete",
          "source undistort stage not complete")
    actual_undistorted = {row["name"]: row["undistorted_image_sha256"] for row in masks.get("images", [])}
    _must(len(actual_undistorted) == 60 and actual_undistorted == image_hashes,
          "undistorted image hashes differ from sealed mask report")
    _must(len(source.get("inputs", [])) == 60 and
          {row["name"] for row in source["inputs"]} == set(image_hashes),
          "SfM source image names mismatch")
    for row in source["inputs"]:
        _must(sha256(Path(row["source"])) == row["sha256"],
              f"original SfM photo changed: {row['name']}")
    return {"schema": "brush_ycb_008_lineage_v1", "evidence_sha256": expected.copy(),
            "registered_images": 60, "camera_lane": "image-only poses; photo-derived pose masks",
            "source_result_status": source.get("status"),
            "source_undistort_stage": next((x.get("status") for x in source.get("stages", [])
                                            if x.get("name") == "undistort"), None)}
