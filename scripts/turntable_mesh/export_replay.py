"""Copy a run's event log and the artifacts it names into a small replay bundle.

The bundle is what a front end needs to show a finished run without an engine:
``events.jsonl``, ``config.json`` and every file an artifact event points to
(sheets, preview meshes, the final mesh, reports). Depth arrays, volumes, masks
and logs are left out. Standard library only.
"""

import argparse
import json
from pathlib import Path
import shutil

from .dense_events import read


def run(source, output, *, skip_final_mesh=False):
    source, output = Path(source), Path(output)
    events = read(source / "events.jsonl")
    if not events or events[-1]["type"] != "run_finished":
        raise ValueError("run has no finished event log")
    output.mkdir(parents=True, exist_ok=False)
    kept, copied = [], 0
    for event in events:
        event = {k: v for k, v in event.items() if k != "seq"}
        if event["type"] == "artifact":
            if event["kind"] == "preview_volume" or (skip_final_mesh and event["kind"] == "final_mesh"):
                continue
            target = source / event["path"]
            if not target.is_file():
                continue
            (output / event["path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, output / event["path"])
            copied += target.stat().st_size
        kept.append(event)
    (output / "events.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in kept))
    if (source / "config.json").is_file():
        shutil.copyfile(source / "config.json", output / "config.json")
    return {"events": len(kept), "bytes": copied}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-final-mesh", action="store_true", help="leave out the full-resolution STL")
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.output, skip_final_mesh=args.skip_final_mesh)))


if __name__ == "__main__":
    main()
