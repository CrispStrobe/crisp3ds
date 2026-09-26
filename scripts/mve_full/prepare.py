"""Fixed, image-only photometric profiles for difficult real-photo SfM tests.

No reference geometry, supplied camera poses, depth, mask or labels are read.
Outputs are new PNGs and a provenance manifest; source images remain untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time

try:
    import resource
except ImportError:  # Windows
    resource = None

from PIL import Image, ImageOps


PROFILES = ("gamma05", "clahe2", "gamma05_clahe2")
RESERVE = 10 << 30
SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def transform(image: Image.Image, profile: str) -> Image.Image:
    if profile not in PROFILES:
        raise ValueError(f"unknown fixed profile: {profile}")
    photo = ImageOps.exif_transpose(image).convert("RGB")
    if profile.startswith("gamma05"):
        lut = [round(255 * math.sqrt(value / 255)) for value in range(256)]
        photo = photo.point(lut * 3)
    if "clahe2" in profile:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("CLAHE profile needs a local Python with cv2 and NumPy") from exc
        rgb = np.asarray(photo)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        photo = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), "RGB")
    return photo


def prepare(input_dir: Path, output_dir: Path, profile: str, max_images: int,
            max_gib: float = 2.0) -> dict:
    started = time.monotonic()
    if profile not in PROFILES or max_images < 3 or not math.isfinite(max_gib) or max_gib <= 0:
        raise ValueError("invalid profile or resource setting")
    if not input_dir.is_dir() or input_dir.is_symlink():
        raise ValueError("input must be a real directory")
    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in SUFFIXES)
    files = files[:max_images]
    if len(files) < 3 or any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("at least three regular source images are required")
    if output_dir.exists() or output_dir.is_symlink() or not output_dir.parent.is_dir():
        raise ValueError("output must be a fresh directory with an existing parent")
    cap = int(max_gib * (1 << 30))
    if shutil.disk_usage(output_dir.parent).free < RESERVE + cap:
        raise ValueError("insufficient output allowance plus 10 GiB reserve")
    output_dir.mkdir()
    report = {"schema": "mve_image_only_preprocess_v2", "profile": profile,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": digest(Path(__file__)), "cv2_version": None,
              "numpy_version": None,
              "pillow_version": Image.__version__, "settings": {"gamma": 0.5 if "gamma05" in profile else None,
              "clahe_clip_limit": 2.0 if "clahe2" in profile else None,
              "clahe_tile_grid": [8, 8] if "clahe2" in profile else None}, "images": []}
    if "clahe2" in profile:
        import cv2
        import numpy as np
        report["cv2_version"] = cv2.__version__
        report["numpy_version"] = np.__version__
    try:
        for index, source in enumerate(files):
            before = digest(source)
            with Image.open(source) as input_image:
                output_image = transform(input_image, profile)
            target = output_dir / f"frame_{index:04d}.png"
            output_image.save(target, format="PNG", compress_level=3)
            after = digest(source)
            if before != after:
                raise ValueError(f"source image changed during processing: {source}")
            used = sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file())
            if used > cap or shutil.disk_usage(output_dir).free < RESERVE:
                raise ValueError("preprocessing exceeded output allowance or disk reserve")
            report["images"].append({"source": source.name, "source_sha256": before,
                                     "output": target.name, "output_sha256": digest(target),
                                     "width": output_image.width, "height": output_image.height})
        report["status"] = "succeeded"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["duration_seconds"] = round(time.monotonic() - started, 2)
        if resource is None:
            report["peak_self_rss_mib"] = None
        else:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["peak_self_rss_mib"] = round(rss / (1 << 20) if platform.system() == "Darwin" else rss / 1024, 2)
        (output_dir / "prepare-manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--max-gib", type=float, default=2.0)
    args = parser.parse_args()
    report = prepare(args.input, args.output, args.profile, args.max_images, args.max_gib)
    print(json.dumps({"status": report["status"], "images": len(report["images"]),
                      "manifest": str(args.output / "prepare-manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
