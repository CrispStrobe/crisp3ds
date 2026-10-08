"""Photo-only camera diagnostic: remove complete match components before recovery.

Run partition, recover cameras with its training match cache, then evaluate the
reserved tracks. This avoids reserving individual edges of a training track.
No dataset poses are consumed. Incorrect matches remain in the diagnostic;
report counts and percentiles, do not treat this as a replacement quality gate.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def partition(matches, every=10):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for pair in matches['pairs']:
        for a,b in pair['matches']:
            x,y=find((pair['first'],a)),find((pair['second'],b));parent[x]=y
    groups={}
    for x in sorted(parent):groups.setdefault(find(x),[]).append(x)
    groups=sorted(groups.values())
    # Split entire components regardless of ambiguity; report ambiguous ones.
    reserved=[g for i,g in enumerate(groups) if i%every==0]
    nodes={tuple(x) for g in reserved for x in g}
    train={**matches,'pairs':[]}
    for p in matches['pairs']:
        train['pairs'].append({**p,'matches':[[a,b] for a,b in p['matches'] if (p['first'],a) not in nodes]})
    tracks=[g for g in reserved if len(g)>=4 and len({v for v,f in g})==len(g)]
    return train,{'tracks':tracks,'reserved_components':len(reserved),'reserved_nodes':len(nodes),
                  'ambiguous_or_short_components':len(reserved)-len(tracks),'split':'every tenth complete connected component, sorted by feature IDs'}


def evaluate(matches, held, rows, lens):
    import cv2
    k=np.array([[lens['fx'],0,lens['cx']],[0,lens['fy'],lens['cy']],[0,0,1]])
    distortion=np.array([lens['k1'],lens['k2'],0,0,lens['k3']])
    points=[cv2.undistortPoints(np.array(p,float).reshape(-1,1,2),k,distortion).reshape(-1,2) for p in matches['keypoints']]
    errors=[];per_view=[[] for r in rows];behind=0;failed=0;fit_verified=[];fit_verified_tracks=0
    for track in held['tracks']:
        fit,test=track[::2],track[1::2]
        design=[]
        for v,f in fit:
            p=np.column_stack((rows[v]['rotation'],rows[v]['translation']));x,y=points[v][f]
            design.extend([x*p[2]-p[0],y*p[2]-p[1]])
        _,_,vt=np.linalg.svd(design);h=vt[-1]
        if abs(h[3])<1e-12:failed+=1;continue
        world=h[:3]/h[3]
        fit_errors=[]
        for v,f in fit:
            q=np.array(rows[v]['rotation'])@world+rows[v]['translation']
            fit_errors.append(float(np.linalg.norm((q[:2]/q[2]-points[v][f])*[lens['fx'],lens['fy']])) if q[2]>0 else float('inf'))
        # Selection uses only the fitting observations, never held-out errors.
        verified=len(fit)>=3 and max(fit_errors)<1.0
        fit_verified_tracks+=int(verified)
        for v,f in test:
            q=np.array(rows[v]['rotation'])@world+rows[v]['translation']
            if q[2]<=0:behind+=1;continue
            residual=(q[:2]/q[2]-points[v][f])*[lens['fx'],lens['fy']]
            e=float(np.linalg.norm(residual));errors.append(e);per_view[v].append(e)
            if verified:fit_verified.append(e)
    def summary(e):
        return {'count':len(e),'median':float(np.median(e)) if e else None,'p95':float(np.percentile(e,95)) if e else None,'fraction_below_1px':float(np.mean(np.array(e)<1)) if e else None}
    return {'schema':'crisp3ds_camera_holdout_v1','reference_used':False,'training_tracks_must_exclude_reserved_components':True,
            'reserved_tracks':len(held['tracks']),'failed_triangulations':failed,'behind_camera_observations':behind,
            'held_out_reprojection_pixels':summary(errors),'per_view':[{'name':r['name'],**summary(e)} for r,e in zip(rows,per_view)],
            'fit_verified_tracks':fit_verified_tracks,'fit_verified_held_out_pixels':summary(fit_verified),
            'limits':['Report is meaningful only when cameras were recovered using the partitioned training cache.',
                      'False feature matches are included; this diagnostic does not establish dense or local facial accuracy.',
                      'Reserved tracks are triangulated using alternating observations; remaining observations score the cameras.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['partition','evaluate']:
        q=sub.add_parser(name);q.add_argument('--matches',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        if name=='evaluate':
            q.add_argument('--reserved',type=Path,required=True);q.add_argument('--cameras',type=Path,required=True);q.add_argument('--calibration',type=Path,required=True)
    a=p.parse_args();matches=json.loads(a.matches.read_text())
    if a.output.exists():raise ValueError('output exists')
    a.output.mkdir(parents=True)
    if a.command=='partition':
        train,held=partition(matches)
        (a.output/'training-matches.json').write_text(json.dumps(train,separators=(',',':')))
        (a.output/'reserved-tracks.json').write_text(json.dumps(held,indent=2))
        print({k:v for k,v in held.items() if k!='tracks'})
    else:
        result=evaluate(matches,json.loads(a.reserved.read_text()),json.loads(a.cameras.read_text())['views'],json.loads(a.calibration.read_text()))
        (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['held_out_reprojection_pixels']))

if __name__=='__main__':main()
