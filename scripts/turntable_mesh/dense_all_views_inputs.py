"""Prepare every registered AliceVision view for the multiscale dense stage.

The earlier stereo adapter stops at 32 views. This writes one camera table for
all registered views, pointing at the native undistorted PNGs in place, and
remaps the raw photo masks through the same radialk3 undistortion (nearest).
No reference geometry, supplied poses or depth are read.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def run(scene, prepared, raw_masks, output):
    scene, prepared, raw_masks, output = map(Path, (scene, prepared, raw_masks, output))
    data = json.loads(scene.read_text())
    intrinsic = data["intrinsics"][0]
    if len(data["intrinsics"]) != 1 or intrinsic["distortionType"] != "radialk3":
        raise ValueError("expected one shared radialk3 intrinsic")
    poses = {str(p["poseId"]): p["pose"]["transform"] for p in data["poses"]}
    width, height = int(intrinsic["width"]), int(intrinsic["height"])
    fy = float(intrinsic["focalLength"]) * width / float(intrinsic["sensorWidth"])
    fx = fy / float(intrinsic["pixelRatio"])
    cx, cy = (float(v) for v in intrinsic["principalPoint"])
    # Integer-pixel principal point for OpenCV; half-centre K for our cameras.
    cx0, cy0 = width / 2 + cx, height / 2 + cy
    k1, k2, k3 = (float(v) for v in intrinsic["distortionParams"])
    matrix = np.array([[fx, 0, cx0], [0, fy, cy0], [0, 0, 1.0]])
    mapx, mapy = cv2.initUndistortRectifyMap(
        matrix, np.array([k1, k2, 0, 0, k3]), None, matrix, (width, height), cv2.CV_32FC1
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "masks").mkdir()
    rows = []
    for view in sorted(data["views"], key=lambda v: Path(v["path"]).name):
        if str(view["poseId"]) not in poses:
            continue
        vid = str(view["viewId"])
        pose = poses[str(view["poseId"])]
        rotation = np.asarray(pose["rotation"], float).reshape(3, 3, order="F")
        center = np.asarray(pose["center"], float)
        image = prepared / f"{vid}.png"
        raw = cv2.imread(str(raw_masks / Path(view["path"]).name), cv2.IMREAD_GRAYSCALE)
        if not image.is_file() or raw is None or raw.shape != (height, width):
            raise ValueError("missing prepared image or raw mask for " + vid)
        mask = cv2.remap(raw, mapx, mapy, cv2.INTER_NEAREST, borderValue=0)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        name = f"view_{vid}"
        cv2.imwrite(str(output / "masks" / f"{name}.png"), mask)
        rows.append(
            {
                "name": name,
                "source": Path(view["path"]).name,
                "image": str(image),
                "mask": str(output / "masks" / f"{name}.png"),
                "width": width,
                "height": height,
                "k": [fx, fy, cx0 + 0.5, cy0 + 0.5],
                "rotation": rotation.tolist(),
                "translation": (-rotation @ center).tolist(),
            }
        )
    points = np.asarray([p["X"] for p in data["structure"]], float)
    np.save(output / "sparse_points.npy", points)
    (output / "cameras.json").write_text(json.dumps({"views": rows}, indent=1) + "\n")
    return {"views": len(rows), "sparse_points": len(points)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "prepared", "raw-masks", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.scene, args.prepared, args.raw_masks, args.output)))


if __name__ == "__main__":
    main()
