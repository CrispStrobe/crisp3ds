"""Bounded diagnostic mapper replay from a sealed COLMAP feature/match database.

The database is read, not regenerated; this probes mapper initialization and
multi-model retry separately from feature and matching nondeterminism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import time

import pycolmap

from scripts.classical_backend.run import digest, init_pair_ids


def options_for(retry_models: bool, pair_ids: tuple[int, int] | None) -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 2
    options.mapper.num_threads = 2
    options.multiple_models = retry_models
    options.max_num_models = 5 if retry_models else 1
    options.min_model_size = 10 if retry_models else 2
    if pair_ids:
        options.init_image_id1, options.init_image_id2 = pair_ids
    return options


def probe(database: Path, images: Path, output: Path, seed: int, retry_models: bool,
          pair: list[str] | None) -> dict:
    if output.exists():
        raise ValueError(f"output must be fresh: {output}")
    if database.is_symlink() or not database.is_file() or images.is_symlink() or not images.is_dir():
        raise ValueError("database and image directory must be real inputs")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        names = [row[0] for row in connection.execute("SELECT name FROM images ORDER BY name")]
    ids = init_pair_ids(database, names, pair)
    options = options_for(retry_models, ids)
    report = {"schema": "classical_sfm_probe_v1", "status": "running",
              "database": str(database.resolve()), "database_sha256": digest(database),
              "images": str(images.resolve()), "image_names": names,
              "image_sha256": {name: digest(images / name) for name in names},
              "pycolmap_version": pycolmap.__version__,
              "pycolmap_binary_sha256": digest(Path(pycolmap._core.__file__)),
              "seed": seed, "retry_models": retry_models, "init_image_pair_names": pair,
              "effective_options": options.todict()}
    output.mkdir(parents=True)
    local_database = output / "database.db"
    shutil.copyfile(database, local_database)
    if digest(local_database) != report["database_sha256"]:
        raise ValueError("diagnostic database copy differs from source")
    report["mapper_database"] = str(local_database)
    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    save()
    pycolmap.set_random_seed(seed)
    started = time.monotonic()
    try:
        models = pycolmap.incremental_mapping(str(local_database), str(images), str(output / "models"),
                                                options=options)
        report["models"] = [{"index": index, "registered": model.num_reg_images(),
                             "points3D": model.num_points3D(),
                             "registered_names": sorted(model.images[i].name for i in model.reg_image_ids())}
                            for index, model in models.items()]
        report["disk_models"] = []
        for directory in sorted((output / "models").iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                continue
            files = {name: directory / name for name in ("cameras.bin", "images.bin", "points3D.bin")}
            if any(not path.is_file() or path.is_symlink() for path in files.values()):
                raise ValueError(f"mapper disk model incomplete: {directory}")
            disk_model = pycolmap.Reconstruction(str(directory))
            report["disk_models"].append({"directory": str(directory),
                                          "registered": disk_model.num_reg_images(),
                                          "points3D": disk_model.num_points3D(),
                                          "files_sha256": {name: digest(path) for name, path in files.items()}})
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", failure=str(error))
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["database_unchanged"] = digest(database) == report["database_sha256"]
    report["image_changes"] = [name for name in names
                               if digest(images / name) != report["image_sha256"][name]]
    if not report["database_unchanged"] or report["image_changes"]:
        report.update(status="failed", failure="mapper input bytes changed")
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--retry-models", action="store_true")
    parser.add_argument("--init-image-pair", nargs=2, metavar=("IMAGE1", "IMAGE2"))
    args = parser.parse_args()
    result = probe(args.database, args.images, args.output, args.seed,
                   args.retry_models, args.init_image_pair)
    print(json.dumps({"status": result["status"], "model_count": len(result.get("models", [])),
                      "registered": max((model["registered"] for model in result.get("models", [])), default=0),
                      "result": str((args.output / "result.json").resolve())}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
