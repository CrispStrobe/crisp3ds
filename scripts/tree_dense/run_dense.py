#!/usr/bin/env python3
"""Bounded live selected-MVE dense run on two predeclared real-tree reference views."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from scripts.mve_spike.evaluate_depth import read_mvei_float

RESERVE=10*1024**3
MAX_BATCH=1024**3
REFERENCES=(1,4)


def size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def run(binary,scene,output):
    binary=binary.resolve(strict=True)
    scene=scene.resolve(strict=True)
    output.mkdir(parents=True,exist_ok=True)
    for index in REFERENCES:
        if (output/f"dmrecon-{index}.log").exists() or (scene/"views"/f"view_{index:04d}.mve"/"depth-L0.mvei").exists():
            raise FileExistsError("dense output already exists; use a fresh scene/output")
    if (output/"dense-status.json").exists():
        raise FileExistsError("dense status already exists; use a fresh output")
    if shutil.disk_usage(output).free<RESERVE+MAX_BATCH:
        raise RuntimeError("10 GiB disk reserve plus 1 GiB batch capacity required")
    conversion=json.loads((scene/"conversion.json").read_text())
    input_files=[binary,Path(__file__).resolve(),scene/"conversion.json",
                 scene/"synth_0.out",scene/"measured-seeds.txt"]
    for index in range(len(conversion["views"])):
        folder=scene/"views"/f"view_{index:04d}.mve"
        input_files.extend((folder/"meta.ini",folder/"undistorted.png"))
    input_files.extend(Path(view["source"]) for view in conversion["views"])
    input_hashes={str(path):sha256(path) for path in input_files}
    source_bytes=sum(Path(view["source"]).stat().st_size for view in conversion["views"])
    if size(scene)+source_bytes>MAX_BATCH:
        raise RuntimeError("source photos plus scene exceed 1 GiB cap")
    tmp=output/"tmp"
    tmp.mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update(TMPDIR=str(tmp.resolve()),OMP_NUM_THREADS="2")
    status={"binary":str(binary),"scene":str(scene),"references":list(REFERENCES),
            "settings":"--scale=0 --neighbors=3 --local-neighbors=3 --progress=simple",
            "per_view_timeout_seconds":120,"input_hashes_sha256":input_hashes,"results":[]}
    for index in REFERENCES:
        if not (scene/"views"/f"view_{index:04d}.mve").is_dir():
            raise ValueError("missing reference view")
        command=[str(binary),f"--master-view={index}","--scale=0","--neighbors=3",
                 "--local-neighbors=3","--progress=simple",str(scene)]
        start=time.monotonic()
        result={"view":index,"command":command}
        try:
            with (output/f"dmrecon-{index}.log").open("w") as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                         env=env,timeout=120,check=False)
            result["exit_code"]=completed.returncode
            result["seconds"]=round(time.monotonic()-start,3)
            # Darwin reports ru_maxrss in bytes. Linux reports KiB.
            peak=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            result["child_peak_rss_bytes_upper_bound"]=peak*(1024 if os.uname().sysname=="Linux" else 1)
            depth=scene/"views"/f"view_{index:04d}.mve"/"depth-L0.mvei"
            result["depth_exists"]=depth.is_file()
            result["depth_bytes"]=depth.stat().st_size if depth.is_file() else 0
            if depth.is_file():
                width,height,values=read_mvei_float(depth)
                result["depth_dimensions"]=[width,height]
                result["valid_depth_pixels"]=sum(math.isfinite(v) and v>0 for v in values)
                result["nonfinite_depth_pixels"]=sum(not math.isfinite(v) for v in values)
                result["depth_sha256"]=sha256(depth)
            result["scene_bytes_after"]=size(scene)
            result["free_bytes_after"]=shutil.disk_usage(output).free
            status["results"].append(result)
            if completed.returncode or not depth.is_file() or result.get("nonfinite_depth_pixels",0) or size(scene)+source_bytes>MAX_BATCH or result["free_bytes_after"]<RESERVE:
                raise RuntimeError(f"dense view {index} failed resource/output check")
        except Exception as error:
            result["error"]=str(error)
            if result not in status["results"]:
                status["results"].append(result)
            (output/"dense-status.json").write_text(json.dumps(status,indent=2)+"\n")
            raise
        (output/"dense-status.json").write_text(json.dumps(status,indent=2)+"\n")
    status["inputs_unchanged"]=all(sha256(Path(name))==expected for name,expected in input_hashes.items())
    status["source_plus_scene_bytes"]=source_bytes+size(scene)
    status["nonempty_both_references"]=all(result.get("valid_depth_pixels",0)>0 for result in status["results"])
    status["status"]="completed_nonempty" if status["nonempty_both_references"] else "dense_failed_empty_reference"
    if not status["inputs_unchanged"]:
        status["status"]="dense_failed_input_changed"
    (output/"dense-status.json").write_text(json.dumps(status,indent=2)+"\n")
    if status["status"]!="completed_nonempty":
        raise RuntimeError(status["status"])
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.binary,a.scene,a.output),indent=2))


if __name__=="__main__":
    main()
