"""Regularise recovered cameras with the turntable constraint.

On a turntable the camera is fixed and the object turns about one axis, so in
the object's frame every camera is the same camera rotated about that axis.
Free structure-from-motion does not know this: each view gets six free
parameters, and on weakly textured objects the recovered orbit comes out
slightly non-circular with uneven steps, which distorts the reconstruction.

This fits the single-axis model to the recovered poses (axis, one canonical
camera pose, one angle per view) and writes a new inputs directory with the
model's cameras. ``--steps measured`` keeps each view's fitted angle;
``--steps uniform`` additionally assumes equal steps between consecutive
capture indices, which is true for a stepper-driven turntable and wrong for a
hand-turned one. Only photo-derived poses are read.
"""

import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np


def rotation(axis, angle):
    """Rodrigues rotation matrices for unit axis and an array of angles (radians)."""
    angle = np.atleast_1d(angle)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def project_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def fit(rotations, centers):
    """Axis, point on axis, per-view angle and canonical pose from world-to-camera R and centres C."""
    rotations, centers = np.asarray(rotations, float), np.asarray(centers, float)
    # Camera-to-world rotations of all views differ by rotations about the axis:
    # Q_i = Rot(axis, angle_i) Q_ref. The axis is the direction they all preserve.
    Q = np.transpose(rotations, (0, 2, 1))
    relative = Q @ Q[0].T
    vectors = np.stack((relative[:, 2, 1] - relative[:, 1, 2], relative[:, 0, 2] - relative[:, 2, 0],
                        relative[:, 1, 0] - relative[:, 0, 1]), 1)
    normal = np.linalg.svd(centers - centers.mean(0))[2][2]
    vectors = vectors * np.sign(vectors @ normal)[:, None]
    axis = vectors.sum(0) + 1e-9 * normal
    axis /= np.linalg.norm(axis)
    # Circle through the centres in the plane normal to the axis.
    e1 = np.linalg.svd(np.eye(3) - np.outer(axis, axis))[0][:, 0]
    e2 = np.cross(axis, e1)
    mean = centers.mean(0)
    xy = np.stack(((centers - mean) @ e1, (centers - mean) @ e2), 1)
    A = np.c_[2 * xy, np.ones(len(xy))]
    solution = np.linalg.lstsq(A, (xy * xy).sum(1), rcond=None)[0]
    point = mean + solution[0] * e1 + solution[1] * e2
    planar = centers - point
    from_position = np.arctan2(planar @ e2, planar @ e1)
    # Angle from orientation as well: twist of Q_i Q_0^T about the axis.
    from_rotation = np.arctan2(np.einsum("i,nij,j->n", e2, relative, e1), np.einsum("i,nij,j->n", e1, relative, e1))
    offset = np.angle(np.exp(1j * (from_position - from_rotation)).mean())
    angles = np.unwrap(np.angle(np.exp(1j * from_position) + np.exp(1j * (from_rotation + offset))))
    # Canonical pose: every view turned back by its angle, then averaged.
    back = rotation(axis, -angles)
    canonical_Q = project_rotation((back @ Q).mean(0))
    canonical_C = (np.einsum("nij,nj->ni", back, planar)).mean(0) + point
    return {"axis": axis, "point": point, "angles": angles, "canonical_Q": canonical_Q, "canonical_C": canonical_C}


def model_poses(model, angles):
    turn = rotation(model["axis"], angles)
    Q = turn @ model["canonical_Q"]
    C = np.einsum("nij,j->ni", turn, model["canonical_C"] - model["point"]) + model["point"]
    R = np.transpose(Q, (0, 2, 1))
    return R, C


def capture_index(row):
    numbers = re.findall(r"\d+", row.get("source", row["name"]))
    if not numbers:
        raise ValueError("no capture index in " + row["name"])
    return int(numbers[-1])


def run(inputs, output, steps="measured", step_degrees=None):
    inputs, output = Path(inputs), Path(output)
    data = json.loads((inputs / "cameras.json").read_text())
    rows = data["views"]
    R = np.array([r["rotation"] for r in rows])
    C = np.array([-np.array(r["rotation"]).T @ np.array(r["translation"]) for r in rows])
    model = fit(R, C)
    angles = model["angles"]
    index = np.array([capture_index(r) for r in rows], float)
    order = np.argsort(index)
    local = np.diff(angles[order]) / np.diff(index[order])
    # The median local step ignores slow drift of the recovered angles; the
    # end-to-end sweep and a line fit both absorb it.
    slope = float(np.median(local)) if step_degrees is None else float(np.radians(step_degrees) * np.sign(np.median(local)))
    intercept = float(np.mean(angles - slope * index))
    report = {"steps": steps, "views": len(rows), "step_degrees": float(np.degrees(slope)),
              "step_degrees_given": step_degrees,
              "sweep_first_to_last_degrees": float(np.degrees(angles[order][-1] - angles[order][0])),
              "axial_spread_fraction_of_radius": None,
              "angle_deviation_from_uniform_degrees": {"rms": float(np.degrees(np.sqrt(np.mean((angles - slope * index - intercept) ** 2)))),
                                                       "maximum": float(np.degrees(np.abs(angles - slope * index - intercept).max()))}}
    if steps == "uniform":
        angles = slope * index + intercept
    elif steps != "measured":
        raise ValueError("steps must be measured or uniform")
    new_R, new_C = model_poses(model, angles)
    radius = np.linalg.norm(model["canonical_C"] - model["point"] - model["axis"] * ((model["canonical_C"] - model["point"]) @ model["axis"]))
    along = (C - model["point"]) @ model["axis"]
    report["axial_spread_fraction_of_radius"] = float(np.ptp(along) / radius)
    turn = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", new_R, R) - 1) / 2, -1, 1)))
    report["change_from_input"] = {"rotation_degrees_median": float(np.median(turn)), "rotation_degrees_maximum": float(turn.max()),
                                   "centre_fraction_of_radius_median": float(np.median(np.linalg.norm(new_C - C, axis=1)) / radius),
                                   "centre_fraction_of_radius_maximum": float(np.linalg.norm(new_C - C, axis=1).max() / radius)}
    output.mkdir(parents=True, exist_ok=False)
    for row, r, c in zip(rows, new_R, new_C):
        row["rotation"], row["translation"] = r.tolist(), (-r @ c).tolist()
        for key in ("image", "mask"):
            path = Path(row[key])
            row[key] = str((path if path.is_absolute() else inputs / path).resolve())
    (output / "cameras.json").write_text(json.dumps(data, indent=1) + "\n")
    shutil.copyfile(inputs / "sparse_points.npy", output / "sparse_points.npy")
    (output / "turntable_rig.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", choices=("measured", "uniform"), default="measured")
    parser.add_argument("--step-degrees", type=float, help="known turntable step per capture index (uniform only)")
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.output, args.steps, args.step_degrees), indent=2))


if __name__ == "__main__":
    main()
