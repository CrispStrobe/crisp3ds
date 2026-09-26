#!/usr/bin/env python3
"""Compare a COLMAP model with an OpenMVS InterfaceCOLMAP reverse export.

The command only reads two already-produced sparse models and writes a report.
It checks camera projection consistency, never object shape or point identities.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare(source_dir, export_dir):
    import numpy as np
    import pycolmap
    source = pycolmap.Reconstruction(str(source_dir))
    exported = pycolmap.Reconstruction(str(export_dir))
    a = {Path(im.name).name: im for im in source.images.values()}
    b = {Path(im.name).name: im for im in exported.images.values()}
    if len(a) != len(source.images) or len(b) != len(exported.images) or set(a) != set(b):
        raise ValueError("camera names do not form identical unique basename sets")
    if len(a) < 3 or source.num_points3D() < 1:
        raise ValueError("need a multi-view sparse source")
    camera_diffs = []
    center_diffs = []
    rotation_diffs = []
    translation_diffs = []
    projection_diffs = []
    source_points = [p.xyz for _, p in sorted(source.points3D.items())[:50]]
    for name in sorted(a):
        x, y = a[name], b[name]
        ca, cb = source.cameras[x.camera_id], exported.cameras[y.camera_id]
        if ca.model.name != "PINHOLE" or cb.model.name != "PINHOLE" or (ca.width, ca.height) != (cb.width, cb.height):
            raise ValueError(f"nonmatching PINHOLE camera dimensions: {name}")
        camera_diffs.append(float(np.max(np.abs(ca.params - cb.params))))
        rotation_diffs.append(float(np.linalg.norm(x.cam_from_world.rotation.matrix() - y.cam_from_world.rotation.matrix())))
        translation_diffs.append(float(np.linalg.norm(x.cam_from_world.translation - y.cam_from_world.translation)))
        center_diffs.append(float(np.linalg.norm(x.cam_from_world.inverse().translation - y.cam_from_world.inverse().translation)))
        for point in source_points:
            ua, ub = x.project_point(point), y.project_point(point)
            if ua is not None and ub is not None:
                projection_diffs.append(float(np.linalg.norm(ua - ub)))
    if not projection_diffs:
        raise ValueError("no finite common-world-point projections")
    # InterfaceCOLMAP writes decimal TXT; this bound admits its rounding, not a
    # meaningful camera-model change. Translation/rotation are arbitrary scale.
    tolerances = {"intrinsics_px": 0.01, "center_arbitrary_units": 0.0001,
                  "rotation_matrix_frobenius": 0.0001, "projection_px": 0.01}
    metrics = {"max_intrinsics_abs_px": max(camera_diffs),
               "max_center_distance_arbitrary_units": max(center_diffs),
               "max_translation_distance_arbitrary_units": max(translation_diffs),
               "max_rotation_matrix_frobenius": max(rotation_diffs),
               "max_projection_difference_px": max(projection_diffs),
               "tested_projections": len(projection_diffs)}
    passed = (metrics["max_intrinsics_abs_px"] <= tolerances["intrinsics_px"] and
              metrics["max_center_distance_arbitrary_units"] <= tolerances["center_arbitrary_units"] and
              metrics["max_rotation_matrix_frobenius"] <= tolerances["rotation_matrix_frobenius"] and
              metrics["max_projection_difference_px"] <= tolerances["projection_px"])
    def input_hashes(directory):
        names = ("cameras.bin", "images.bin", "points3D.bin") if (directory / "cameras.bin").exists() else (
            "cameras.txt", "images.txt", "points3D.txt")
        return {name: digest(directory / name) for name in names}
    return {"schema": "object_motion_interface_roundtrip_v1", "status": "pass" if passed else "fail",
            "source_model": str(source_dir.resolve()), "exported_model": str(export_dir.resolve()),
            "source_files_sha256": input_hashes(source_dir), "exported_files_sha256": input_hashes(export_dir),
            "source_registered": source.num_reg_images(), "exported_registered": exported.num_reg_images(),
            "source_points3D": source.num_points3D(), "exported_points3D": exported.num_points3D(),
            "same_image_basenames": True, "tolerances": tolerances, "metrics": metrics,
            "scope": "Camera/intrinsic/projection roundtrip after OpenMVS import; no shape, point-ID, mesh, metric-scale or supplied-pose validation"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-model", required=True, type=Path)
    p.add_argument("--exported-model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = compare(a.source_model, a.exported_model)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], **report["metrics"]}))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
