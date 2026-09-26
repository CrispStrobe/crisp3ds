#!/usr/bin/env python3
"""Fixed real-tree pair comparison; no depth ground truth is used."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pfm(path):
    with path.open("rb") as stream:
        if stream.readline() != b"Pf\n":
            raise ValueError(f"invalid PFM {path}")
        w, h = map(int, stream.readline().split())
        if stream.readline() != b"-1.0\n":
            raise ValueError("expected little-endian PFM")
        raw = stream.read()
    if len(raw) != 4*w*h:
        raise ValueError("truncated PFM")
    values = struct.unpack(f"<{w*h}f", raw)
    return w, h, [v for y in range(h-1, -1, -1) for v in values[y*w:(y+1)*w]]


def percentile(data, fraction):
    values = sorted(data)
    return values[round((len(values)-1)*fraction)] if values else None


def sample(image, w, h, x, y):
    ix, iy = round(x), round(y)
    return image[iy*w+ix] if 0 <= ix < w and 0 <= iy < h else math.inf


def score(path, observations, ndisp, right=None):
    w, h, values = pfm(path)
    valid = sum(math.isfinite(d) and 0 <= d < ndisp for d in values)
    positives=[d for d in values if math.isfinite(d) and 0 < d < ndisp]
    usable, errors = 0, []
    for row in observations:
        x, y = float(row["left_x"]), float(row["left_y"])
        d = sample(values,w,h,x,y)
        if math.isfinite(d) and 0 <= d < ndisp:
            usable += 1
            errors.append(abs(d-float(row["observed_disparity"])))
    result = {"valid_pixels":valid,"positive_pixels":len(positives),"total_pixels":w*h,"full_image_coverage":valid/(w*h),
              "positive_disparity_p10_px":percentile(positives,.1),
              "positive_disparity_median_px":percentile(positives,.5),
              "positive_disparity_p90_px":percentile(positives,.9),
              "heldout_valid":usable,"heldout_total":len(observations),
              "heldout_median_absolute_disparity_error_px":statistics.median(errors) if errors else None,
              "heldout_within_2px":sum(e<=2 for e in errors),
              "heldout_within_5px":sum(e<=5 for e in errors)}
    if right is not None:
        rw,rh,rvalues=pfm(right)
        if (rw,rh)!=(w,h): raise ValueError("right disparity shape mismatch")
        consistent=0
        for y in range(h):
            for x in range(w):
                d=values[y*w+x]
                if math.isfinite(d) and 0 <= d < ndisp:
                    dr=sample(rvalues,w,h,x-d,y)
                    if math.isfinite(dr) and 0 <= dr < ndisp and abs(d-dr)<=1: consistent+=1
        result["left_right_consistent_1px"]=consistent
        result["left_right_consistent_fraction_of_valid"]=consistent/valid if valid else None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--tool",type=Path,required=True)
    p.add_argument("--elas",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    scene=args.scene.resolve(); output=args.output.resolve()
    if output.exists(): p.error("output exists; use a fresh path")
    source=json.loads((scene/"conversion.json").read_text())
    inputs=[scene/"conversion.json",scene/"measured-seeds.txt",
            Path(source["views"][1]["output_image"]),Path(source["views"][4]["output_image"]),
            args.tool.resolve(),args.elas.resolve()]
    before={str(path):sha(path) for path in inputs}
    subprocess.run([str(args.tool.resolve()),"selftest"],check=True,timeout=10)
    subprocess.run([str(args.tool.resolve()),"prepare",str(scene),str(output)],check=True,timeout=30)
    geometry=json.loads((output/"geometry.json").read_text())
    rows=list(csv.DictReader((output/"correspondences.csv").open()))
    hold=[row for row in rows if row["split"]=="holdout"]
    residuals=[float(row["epipolar_residual"]) for row in hold]
    rect={"heldout_median_vertical_residual_px":statistics.median(residuals),
          "heldout_p90_vertical_residual_px":percentile(residuals,.9),
          "heldout_max_vertical_residual_px":max(residuals)}
    report={"scene":str(scene),"pair_original_view_ids":[1,4],"geometry":geometry,"rectification":rect}
    if rect["heldout_median_vertical_residual_px"]>2 or rect["heldout_p90_vertical_residual_px"]>4:
        report["status"]="failed_rectification_gate"
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        raise SystemExit("rectification residual gate failed; dense comparison stopped")
    ndisp=geometry["ndisp"]
    subprocess.run([str(args.tool.resolve()),"sgbm",str(output),str(ndisp)],check=True,timeout=120)
    subprocess.run([str(args.elas.resolve()),"--left",str(output/"left.pgm"),
                    "--right",str(output/"right.pgm"),"--ndisp",str(ndisp),
                    "--output",str(output/"elas.pfm")],check=True,timeout=120)
    report["status"]="completed"
    report["matchers"]={"OpenCV StereoSGBM":score(output/"sgbm.pfm",hold,ndisp,output/"sgbm-right.pfm"),
                        "libELAS":score(output/"elas.pfm",hold,ndisp)}
    report["settings"]={"shared_disparity_min":0,"shared_disparity_max_inclusive":ndisp-1,
                        "sgbm":{"block_size":5,"P1":200,"P2":800,"disp12MaxDiff":1,"preFilterCap":31,
                                "uniquenessRatio":10,"speckleWindowSize":100,"speckleRange":2,"mode":"SGBM"},
                        "elas":"MIDDLEBURY preset from separate local GPL oracle"}
    after={str(path):sha(path) for path in inputs}
    if before!=after:
        report["status"]="inputs_changed_during_run"
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        raise SystemExit("scene or binary input changed during run")
    report["inputs_unchanged"]=True
    files=inputs+[output/"left.pgm",output/"right.pgm",output/"sgbm.pfm",output/"sgbm-right.pfm",output/"elas.pfm"]
    report["sha256"]={str(path):sha(path) for path in files}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["matchers"],indent=2))


if __name__=="__main__":main()
