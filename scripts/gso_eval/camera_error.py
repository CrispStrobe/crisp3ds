"""Solved cameras against the renderer's truth, for one case.

    python camera_error.py work/CASE

Fits a proper similarity from solved camera centres to true centres, then reports the centre error in % of the
orbit radius and the orientation error per view in degrees after the fit (a small orientation
error under a proper fit means the solution is not a mirror image).
"""
import json
import sys
from pathlib import Path

import numpy as np


def umeyama(src, dst, mirror=False):
    src = src * ([-1, 1, 1] if mirror else 1)
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, s, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    sign = np.diag([1, 1, d])
    r = u @ sign @ vt
    scale = (s * np.diag(sign)).sum() / (a ** 2).sum()
    return scale, r, md - scale * r @ ms


def main():
    case = Path(sys.argv[1])
    truth = json.loads((case / "capture/truth.json").read_text())
    tv = {v["photo"]: v for v in truth["views"]}
    rows = json.loads((case / "run/frontend/inputs/cameras.json").read_text())["views"]
    true_c, solved_c, true_r, solved_r = [], [], [], []
    for row in rows:
        n = int(row["source"].split("_")[-1].split(".")[0])
        t = tv[f"shot_{n:03d}.png"]
        r = np.array(row["rotation"])
        solved_c.append(-r.T @ np.array(row["translation"]))
        solved_r.append(r)
        true_c.append(np.array(t["centre"]))
        true_r.append(np.array(t["rotation"]))
    true_c, solved_c = np.array(true_c), np.array(solved_c)
    radius = truth["orbit_radius"]
    out = {}
    for mirror in (False,):  # a mirrored fit of centres on one ring is degenerate; orientations decide
        s, r, t = umeyama(solved_c, true_c, mirror)
        moved = s * (solved_c * ([-1, 1, 1] if mirror else 1)) @ r.T + t
        rms = float(np.sqrt(((moved - true_c) ** 2).sum(1).mean()) / radius * 100)
        out["mirrored" if mirror else "proper"] = rms
        if not mirror:
            # Orientation: solved world->camera R_s; in true world: R_s @ r.T.
            ang = []
            for rs, rt in zip(solved_r, true_r):
                d = (rs @ r.T) @ rt.T
                ang.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
            ang = np.array(ang)
            out["orientation_deg_median"] = float(np.median(ang))
            out["orientation_deg_max"] = float(ang.max())
            out["centre_err_pct_max"] = float((np.linalg.norm(moved - true_c, axis=1) / radius * 100).max())
    print(json.dumps({"views": len(rows), "centre_rms_pct_of_radius": out}, indent=1))
    (case / "camera_error.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
