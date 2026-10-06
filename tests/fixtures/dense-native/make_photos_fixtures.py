"""Reference values for the native photo front stage (crates/dense/src/photos).

Writes opencv-contrast.json next to this file: values produced by OpenCV
(cv2.cvtColor RGB2LAB / LAB2RGB on 8-bit data, cv2.createCLAHE) and by the
reference contrast_image, which the Rust tests reproduce byte for byte.

    PYTHONPATH=<repository> python tests/fixtures/dense-native/make_photos_fixtures.py [--exhaustive]

Needs NumPy, OpenCV and Pillow. The committed file was written with OpenCV
4.10.0, NumPy 1.26. ``--exhaustive`` additionally compares the integer
reconstruction of the two colour conversions below (the algorithm the Rust code
implements, table for table) with cv2 for all 2^24 input triples; that takes a
few minutes and about 2 GB of memory.
"""

import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from scripts.turntable_mesh import photos_to_inputs as front

f32 = np.float32
BASE = 1 << 14
MINIMUM_AB = -8145
TO_XYZ = np.array([[1777, 1541, 778], [871, 2929, 296], [73, 448, 3575]], np.int64)
FROM_XYZ = np.array([[12615, -6296, -2223], [-3773, 7684, 185], [217, -836, 4715]], np.int64)


def opencv_cbrt(x):
    """cv::cbrt for float: rational approximation on the mantissa, evaluated in double."""
    bits = int(f32(x).view(np.int32))
    ix = bits & 0x7FFFFFFF
    if ix == 0:
        return f32(0)
    exponent = ((bits >> 23) & 255) - 127
    shift = int(math.fmod(exponent, 3))
    shift -= 3 if shift >= 0 else 0
    exponent = (exponent - shift) // 3
    fr = float(np.array((ix & ((1 << 23) - 1)) | ((shift + 127) << 23), np.int32).view(np.float32))
    fr = ((((45.2548339756803022511987494 * fr + 192.2798368355061050458134625) * fr + 119.1654824285581628956914143) * fr
           + 13.43250139086239872172837314) * fr + 0.1636161226585754240958355063) / \
         ((((14.80884093219134573786480845 * fr + 151.9714051044435648658557668) * fr + 168.5254414101568283957668343) * fr
           + 33.9905941350215598754191872) * fr + 1.0)
    return np.array(int(f32(fr).view(np.int32)) + exponent * 8388608, np.int32).view(np.float32)


def tables():
    x = np.arange(256) / 255.0
    gamma = np.rint(2040 * np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)).astype(np.int64)
    cbrt = np.zeros(2041, np.int64)
    for i in range(2041):
        t = f32(i) / f32(2040)
        value = t * (f32(841) / f32(108)) + f32(16) / f32(116) if t < f32(216) / f32(24389) else opencv_cbrt(t)
        cbrt[i] = int(np.rint(f32(1 << 15) * f32(value)))
    cbrt[324] = 17745  # a tie in this evaluation that OpenCV resolves downwards
    lightness = np.zeros((256, 2), np.int64)
    for i in range(256):
        if i <= 20:
            y = int(np.rint(f32(i * BASE * 20 * 9) / f32(17 * 29 * 29 * 29)))
            fy = int(np.rint(f32(BASE) * (f32(16) / f32(116) + f32(i * 5) / f32(3 * 17 * 29))))
        else:
            value = f32(i * 100 * BASE) / f32(255 * 116) + f32(16 * BASE) / f32(116)
            fy = int(np.rint(value))
            y = int(np.rint(value * value * value / f32(BASE * BASE)))
        lightness[i] = (y, fy)
    i = np.arange(MINIMUM_AB, BASE * 9 // 4 + MINIMUM_AB, dtype=np.int64)

    def cdiv(a, b):  # C integer division
        return np.sign(a) * (np.abs(a) // b)

    ab = np.where(i <= 3390, cdiv(i * 108, 841) - BASE * 16 // 116 * 108 // 841, cdiv(cdiv(i * i, BASE) * i, BASE))
    x = np.arange(4096) / 4096.0
    inverse = 255 * np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    margins = {"gamma": float(np.abs((2040 * np.where(np.arange(256) / 255.0 <= 0.04045, np.arange(256) / 255.0 / 12.92,
                                                      ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4)) % 1 - 0.5).min()),
               "inverse_gamma": float(np.abs(inverse % 1 - 0.5).min())}
    return gamma, cbrt, lightness, ab, np.rint(inverse).astype(np.int64), margins


def descale(v, n):
    return (v + (1 << (n - 1))) >> n


def rgb_to_lab(rgb, gamma, cbrt):
    r, g, b = (gamma[rgb[..., k]] for k in range(3))
    fx, fy, fz = (cbrt[descale(r * row[0] + g * row[1] + b * row[2], 12)] for row in TO_XYZ)
    lab = [descale(296 * fy - 1336934, 15), descale(500 * (fx - fy) + (128 << 15), 15), descale(200 * (fy - fz) + (128 << 15), 15)]
    return np.clip(np.stack(lab, -1), 0, 255).astype(np.uint8)


def lab_to_rgb(lab, lightness, ab, inverse):
    L, a, b = (lab[..., k].astype(np.int64) for k in range(3))
    y, fy = lightness[L, 0], lightness[L, 1]
    x = ab[fy + ((5 * a * 53687 + (1 << 7)) >> 13) - 128 * BASE // 500 - MINIMUM_AB]
    z = ab[fy - (((b * 41943 + (1 << 4)) >> 9) - 128 * BASE // 200 + 1) - MINIMUM_AB]
    return np.stack([inverse[np.clip(descale(row[0] * x + row[1] * y + row[2] * z, 14), 0, 4095)] for row in FROM_XYZ], -1).astype(np.uint8)


def main():
    gamma, cbrt, lightness, ab, inverse, margins = tables()
    rng = np.random.default_rng(227)
    edge = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [1, 1, 1], [20, 128, 128], [21, 128, 128],
                     [128, 0, 255], [128, 255, 0], [254, 1, 127]], np.uint8)
    colours = np.concatenate([edge, rng.integers(0, 256, (1100, 3), dtype=np.uint8)]).reshape(-1, 1, 3)
    lab = cv2.cvtColor(colours, cv2.COLOR_RGB2LAB)
    back = cv2.cvtColor(colours, cv2.COLOR_LAB2RGB)
    assert (rgb_to_lab(colours.astype(np.int64), gamma, cbrt) == lab).all()
    assert (lab_to_rgb(colours, lightness, ab, inverse) == back).all()
    if "--exhaustive" in sys.argv:
        v = np.arange(256, dtype=np.uint8)
        for first in range(0, 256, 32):
            a, b, c = np.meshgrid(v[first:first + 32], v, v, indexing="ij")
            block = np.stack([a, b, c], -1).reshape(-1, 1, 3)
            assert (rgb_to_lab(block.astype(np.int64), gamma, cbrt) == cv2.cvtColor(block, cv2.COLOR_RGB2LAB)).all(), first
            assert (lab_to_rgb(block, lightness, ab, inverse) == cv2.cvtColor(block, cv2.COLOR_LAB2RGB)).all(), first
            print("exhaustive block", first, "identical in both directions", flush=True)
    clahe = []
    for width, height, clip, grid, kind in ((16, 16, 2.0, 2, "noise"), (40, 24, 2.0, 8, "noise"), (37, 29, 2.0, 8, "noise"),
                                            (40, 29, 2.0, 8, "ramp"), (64, 48, 40.0, 4, "noise"), (48, 32, 0.01, 4, "dark"),
                                            (24, 24, 3.0, 3, "flat"), (9, 7, 2.0, 8, "noise"), (50, 36, 1.5, 5, "dark")):
        if kind == "noise":
            plane = rng.integers(0, 256, (height, width), dtype=np.uint8)
        elif kind == "dark":
            plane = np.minimum(rng.gamma(1.2, 18.0, (height, width)), 255).astype(np.uint8)
        elif kind == "ramp":
            plane = ((np.arange(width)[None, :] * 5 + np.arange(height)[:, None] * 3) % 256).astype(np.uint8)
        else:
            plane = np.full((height, width), 77, np.uint8)
        out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(plane)
        clahe.append({"width": width, "height": height, "clip": clip, "grid": grid, "input": plane.ravel().tolist(),
                      "output": out.ravel().tolist()})
    contrast = []
    for width, height, top, gamma_value, clip, grid in ((48, 40, 120, 0.5, 2.0, 8), (37, 26, 256, 0.5, 2.0, 8), (32, 24, 256, 1.0, 3.0, 4),
                                                         (30, 20, 200, 0.7, 0.0, 8)):
        rgb = rng.integers(0, top, (height, width, 3), dtype=np.uint8)
        out = np.asarray(front.contrast_image(Image.fromarray(rgb, "RGB"), gamma_value, clip, grid))
        contrast.append({"width": width, "height": height, "gamma": gamma_value, "clip": clip, "grid": grid,
                         "input": rgb.ravel().tolist(), "output": out.ravel().tolist()})
    data = {
        "produced_by": "tests/fixtures/dense-native/make_photos_fixtures.py",
        "opencv": cv2.__version__, "numpy": np.__version__,
        "gamma_half": front.gamma_table(0.5),
        "table_sums": {"gamma": int(gamma.sum()), "cbrt": int(cbrt.sum()), "lightness": int((lightness[:, 0] + 3 * lightness[:, 1]).sum()),
                       "ab": int(ab.sum()), "inverse_gamma": int(inverse.sum())},
        "table_tie_margins": margins,
        "colours": {"rgb": colours.ravel().tolist(), "lab": lab.ravel().tolist(), "rgb_from_lab": back.ravel().tolist()},
        "clahe": clahe, "contrast": contrast,
    }
    path = Path(__file__).with_name("opencv-contrast.json")
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(path, path.stat().st_size, "bytes; tie margins", margins)


if __name__ == "__main__":
    main()
