#!/usr/bin/env python3
"""Dense replay of the saved, rejected BA candidate; never marks it accepted."""
import json
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts/mve_spike"))
from convert_scene import convert
from scripts.bundle_quality.run import replace_scene,digest,camera_error,score_depth,RESERVE,PIN


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"build-opencv/bundle-quality")
    args=parser.parse_args()
    output=args.output.resolve()
    fixture=ROOT/"build-opencv/synthetic-sparse-fixture"
    optimized=output/"optimized.json"
    data=json.loads(optimized.read_text())
    if data["final"]["marker_rms_px"]<=data["initial"]["marker_rms_px"]:
        raise ValueError("candidate is not rejected by marker safeguard")
    scene=output/"scene"
    if scene.exists():raise FileExistsError(scene)
    for name in ("rejected-mve.log","rejected-diagnostic.json","rejected-status.json","rejected-manifest.json"):
        if (output/name).exists():raise FileExistsError(output/name)
    original_manifest=json.loads((output/"manifest.json").read_text())
    pinned={name:checksum for name,checksum in original_manifest["sha256"].items()
            if name.startswith("build-opencv/synthetic-sparse-fixture/") or
               name.endswith("/apps/dmrecon/dmrecon")}
    def verify_pinned():
        for name,checksum in pinned.items():
            if digest(ROOT/name)!=checksum:raise RuntimeError(f"original BA input changed: {name}")
    verify_pinned()
    baseline_path=ROOT/"build-opencv/quality-ablation-final/results.json"
    new_manifest={"original_ba_manifest_sha256":digest(output/"manifest.json"),
                  "verified_original_input_sha256":pinned,
                  "optimized_sha256":digest(optimized),
                  "scorer_sha256":digest(ROOT/"scripts/quality_ablation/score.py"),
                  "replay_runner_sha256":digest(Path(__file__)),
                  "baseline_results_sha256":digest(baseline_path)}
    if shutil.disk_usage(output).free<RESERVE+512*1024**2:raise RuntimeError("10 GiB reserve")
    convert(fixture,scene)
    replace_scene(scene,data["cameras"],data["points"])
    binary=ROOT/".local-tools/mve-spike"/f"mve-{PIN}"/"apps/dmrecon/dmrecon"
    command=["/usr/bin/time","-l",str(binary),"--master-view=1","--scale=1","--neighbors=2",
             "--local-neighbors=2","--progress=simple",str(scene)]
    start=time.monotonic()
    with (output/"rejected-mve.log").open("w") as stream:
        run=subprocess.run(command,stdout=stream,stderr=subprocess.PIPE,text=True,timeout=120,
                           env=dict(os.environ,TMPDIR=str(ROOT/".local-tools/tmp")))
        stream.write("\n--- time -l / stderr ---\n"+run.stderr)
    if run.returncode:raise RuntimeError(f"MVE failed {run.returncode}")
    verify_pinned()
    truth_path=fixture/"truth.json"
    truth=json.loads(truth_path.read_text())
    project=json.loads((fixture/"project.json").read_text())
    depth=scene/"views/view_0001.mve/depth-L1.mvei"
    score,_=score_depth(depth,data["cameras"][1],truth["views"][1],project["calibration"],truth["objectPlanes"])
    if digest(baseline_path)!=new_manifest["baseline_results_sha256"]:
        raise RuntimeError("baseline changed during replay")
    baseline=json.loads(baseline_path.read_text())["cases"]["estimated"]["L1"]["score"]
    report={"status":"REJECTED: marker RMS increased under frozen equal weights",
            "marker_rms_before_px":data["initial"]["marker_rms_px"],
            "marker_rms_after_px":data["final"]["marker_rms_px"],
            "optimizer_output_sha256":digest(optimized),"truth_sha256":digest(truth_path),
            "depth_sha256":digest(depth),"run":{"command":command,"wall_seconds":time.monotonic()-start,
                "peak_rss_bytes":int(m.group(1)) if (m:=re.search(r"(\d+)\s+maximum resident set size",run.stderr)) else None},
            "score":score,"estimated_baseline_score":baseline,
            "camera_errors_vs_truth":camera_error({f"view-{i+1}":dict(c) for i,c in enumerate(data["cameras"])},
                                                   {v["imageId"]:v for v in truth["views"]})}
    (output/"rejected-diagnostic.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (output/"rejected-manifest.json").write_text(json.dumps(new_manifest,indent=2,sort_keys=True)+"\n")
    (output/"rejected-status.json").write_text(json.dumps({"state":"rejected_scored","reason":report["status"]})+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":main()
