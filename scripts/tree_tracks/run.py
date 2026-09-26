"""Bounded live runner with input-integrity record for the fixed tree scene."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sift", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output_root = root / "build-opencv/tree-tracks"
    report = args.report.resolve()
    manifest = report.with_name(report.stem + "-integrity.json")
    if not report.is_relative_to(output_root) or report.exists() or manifest.exists():
        parser.error("report must be a new file within build-opencv/tree-tracks")
    if shutil.disk_usage(root).free < 10 * 1024**3:
        parser.error("less than 10 GiB free")
    views = args.views.resolve()
    rows = [line.split() for line in views.read_text().splitlines() if line.strip()]
    if len(rows) != 10 or any(len(row) != 20 for row in rows):
        parser.error("expected exactly ten complete views")
    inputs = [views] + [Path(row[3]).resolve() for row in rows]
    inputs += [args.binary.resolve(), Path(__file__).resolve(),
               Path(__file__).with_name("tracks.cc")]
    before = {str(path): digest(path) for path in inputs}
    output_root.mkdir(parents=True, exist_ok=True)
    temp = root / ".local-tools/tmp"
    temp.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TMPDIR"] = str(temp)
    command = [str(args.binary.resolve()), str(views), str(report)]
    if args.sift:
        command.append("--sift")
    subprocess.run(command,
                   check=True, cwd=root, env=environment, timeout=120)
    after = {str(path): digest(path) for path in inputs}
    integrity = {"inputs_unchanged": before == after, "sha256_before": before,
                 "sha256_after": after}
    manifest.write_text(json.dumps(integrity, indent=2) + "\n")
    if not integrity["inputs_unchanged"]:
        raise RuntimeError("input hash changed during run")
    if report.stat().st_size + manifest.stat().st_size > 100 * 1024**2:
        raise RuntimeError("tree-tracks artifacts exceed 100 MiB")
    print(f"report: {report}\nintegrity: {manifest}")


if __name__ == "__main__":
    main()
