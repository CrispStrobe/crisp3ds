#!/usr/bin/env python3
"""Summarize full-image real-photo depth coverage and two-view overlap."""

import argparse
import json
import math
from pathlib import Path
from PIL import Image
from scripts.mve_spike.evaluate_depth import read_mvei_float


def evaluate(scene,output,refs=(1,4)):
    output.mkdir(parents=True,exist_ok=True)
    for name in ("depth-evaluation.json",*(f"depth-preview-{index}.png" for index in refs)):
        if (output/name).exists():
            raise FileExistsError(output/name)
    conversion=json.loads((scene/"conversion.json").read_text())
    maps={}
    summary={"reference_views":list(refs),"views":{},"cross_view":{}}
    for index in refs:
        path=scene/"views"/f"view_{index:04d}.mve"/"depth-L0.mvei"
        width,height,depths=read_mvei_float(path)
        if (width,height)!=(conversion["views"][index]["camera"]["width"],
                            conversion["views"][index]["camera"]["height"]):
            raise ValueError("depth/image dimensions differ")
        valid=[(i,value) for i,value in enumerate(depths) if math.isfinite(value) and value>0]
        invalid_numeric=sum(not math.isfinite(v) for v in depths)
        if invalid_numeric:
            raise ValueError("nonfinite depth output")
        maps[index]=(width,height,depths)
        summary["views"][str(index)]=dict(width=width,height=height,valid_depth_pixels=len(valid),
            total_pixels=width*height,full_image_coverage=len(valid)/(width*height),
            positive_depth_min=min((v for _,v in valid),default=None),
            positive_depth_max=max((v for _,v in valid),default=None),
            source_image=conversion["views"][index]["name"])
        preview=Image.new("RGB",(width,height))
        if valid:
            lo=min(v for _,v in valid); hi=max(v for _,v in valid)
            colors=[(0,0,0)]*(width*height)
            for pos,value in valid:
                shade=round(255*(value-lo)/(hi-lo)) if hi>lo else 192
                colors[pos]=(255-shade,shade,80)
            preview.putdata(colors)
        preview.save(output/f"depth-preview-{index}.png")
    a,b=refs
    # A consistency value needs actual positive depth maps in both views.
    summary["cross_view"]={"tested_correspondences":0,"consistent_correspondences":0,
        "status":"unavailable: one or both maps lack projected overlapping valid depths"}
    if summary["views"][str(a)]["valid_depth_pixels"] and summary["views"][str(b)]["valid_depth_pixels"]:
        # Project each radial depth through reference camera and compare to nearest
        # pixel in the second map. This is only a consistency diagnostic.
        views=conversion["views"]
        wa,ha,da=maps[a]; wb,hb,db=maps[b]
        ca,cb=views[a]["camera"],views[b]["camera"]
        ra,rb=views[a]["rotation"],views[b]["rotation"]
        ta,tb=views[a]["translation"],views[b]["translation"]
        tested=consistent=0
        for pos,depth in enumerate(da):
            if depth<=0: continue
            x,y=pos%wa,pos//wa
            ray=[(x-ca["cx"])/ca["fx"],(y-ca["cy"])/ca["fy"],1]
            norm=math.sqrt(sum(q*q for q in ray))
            cam=[depth*q/norm for q in ray]
            world=[sum(ra[3*j+i]*(cam[j]-ta[j]) for j in range(3)) for i in range(3)]
            target=[sum(rb[3*i+j]*world[j] for j in range(3))+tb[i] for i in range(3)]
            if target[2]<=0: continue
            u=round(cb["fx"]*target[0]/target[2]+cb["cx"])
            v=round(cb["fy"]*target[1]/target[2]+cb["cy"])
            if not (0<=u<wb and 0<=v<hb): continue
            observed=db[v*wb+u]
            if observed<=0: continue
            tested+=1
            expected=math.sqrt(sum(q*q for q in target))
            if abs(expected-observed)/expected<=.05: consistent+=1
        summary["cross_view"]={"tested_correspondences":tested,
            "consistent_correspondences":consistent,
            "fraction_within_5pct_radial_depth":consistent/tested if tested else None,
            "status":"measured" if tested else "unavailable: no projected overlapping valid depths"}
    (output/"depth-evaluation.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(evaluate(a.scene,a.output),indent=2))


if __name__=="__main__":
    main()
