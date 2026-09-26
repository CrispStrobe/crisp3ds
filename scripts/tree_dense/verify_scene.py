#!/usr/bin/env python3
"""Check actual selected MVE loader K/poses and measured seed observations."""

import argparse
import json
import math
from pathlib import Path
import subprocess
try:
    from scripts.tree_dense.import_scene import project
except ModuleNotFoundError:
    from import_scene import project


def verify(inspector, scene):
    conversion=json.loads((scene/"conversion.json").read_text())
    lines=subprocess.check_output([str(inspector.resolve(strict=True)),str(scene.resolve())],text=True,timeout=30).splitlines()
    bundle=[line for line in lines if line.startswith("BUNDLE ")]
    expected=[len(conversion["views"]),conversion["measured_seed_tracks"]]
    if len(bundle)!=1 or list(map(int,bundle[0].split()[1:]))!=expected:
        raise ValueError("MVE loaded bundle counts differ")
    loaded={}
    for line in lines:
        if not line.startswith("VIEW "):
            continue
        numbers=list(map(float,line.split()[1:]))
        if len(numbers)!=24 or not all(math.isfinite(x) for x in numbers):
            raise ValueError("unexpected MVE loaded view record")
        index=int(numbers[0])
        if index in loaded or not 0<=index<len(conversion["views"]):
            raise ValueError("duplicate/out of range loaded view")
        view=conversion["views"][index]
        camera=view["camera"]
        if tuple(numbers[1:3])!=(camera["width"],camera["height"]):
            raise ValueError("MVE loaded image size differs")
        k=numbers[3:12]
        expected_k=[camera["fx"],0,camera["cx"]+.5,0,camera["fy"],camera["cy"]+.5,0,0,1]
        if max(abs(a-b) for a,b in zip(k,expected_k))>.002:
            raise ValueError("MVE loaded K differs from pixel-center convention")
        rotation,translation=numbers[12:21],numbers[21:24]
        if max(abs(a-b) for a,b in zip(rotation,view["rotation"]))>.0001 or max(
                abs(a-b) for a,b in zip(translation,view["translation"]))>.0001:
            raise ValueError("MVE loaded pose differs")
        loaded[index]=(camera,rotation,translation)
    if len(loaded)!=len(conversion["views"]):
        raise ValueError("MVE loaded view count differs")
    errors=[]
    for line in (scene/"measured-seeds.txt").read_text().splitlines():
        data=line.split()
        if len(data)!=11:
            raise ValueError("malformed measured seed row")
        point=list(map(float,data[:3]))
        for index,pixel in ((int(data[3]),(float(data[4]),float(data[5]))),
                            (int(data[6]),(float(data[7]),float(data[8])))):
            camera,rotation,translation=loaded[index]
            error=math.dist(project(camera,rotation,translation,point),pixel)
            if not math.isfinite(error):
                raise ValueError("nonfinite loaded reprojection error")
            errors.append(error)
    if len(errors)!=2*expected[1] or not errors:
        raise ValueError("measured seed observation count differs")
    result=dict(loaded_views=len(loaded),loaded_seeds=expected[1],observations=len(errors),
                loaded_reprojection_rms_px=math.sqrt(sum(e*e for e in errors)/len(errors)),
                loaded_reprojection_max_px=max(errors))
    if result["loaded_reprojection_max_px"]>2.05:
        raise ValueError("MVE loaded seed reprojection exceeds 2.05 px")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector",type=Path,required=True)
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=verify(args.inspector,args.scene)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
