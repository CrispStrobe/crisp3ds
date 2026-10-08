"""Evaluation-only ray-depth/relief diagnostics in fixed, recovered cameras.

An already aligned independent STL may be supplied ONLY for evaluation. No
output of this script is a reconstruction input. Plane removal measures local
relief, not absolute accuracy; report both metrics and coverage on common rays.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np


def read_stl(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError('expected binary STL')
    return np.frombuffer(raw, dtype=[('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')], offset=84)['vertices'].astype(float)


def ray_depth(triangles, row, crop):
    """Perspective-correct nearest surface on original photo pixel centres."""
    left, top, width, height = crop
    camera = triangles @ np.array(row['rotation']).T + row['translation']
    fx, fy, cx, cy = row['k']
    pixels = camera[:, :, :2] / camera[:, :, 2:] * [fx, fy] + [cx - .5 - left, cy - .5 - top]
    usable = ((camera[:, :, 2] > 0).all(1) & (pixels.max(1) >= 0).all(1) & (pixels.min(1) < [width, height]).all(1))
    pixels, camera = pixels[usable], camera[usable]
    result = np.full((height, width), np.inf)
    # Crop bounds keep this loop small; process a triangle's pixels in NumPy.
    for p, q in zip(pixels, camera):
        a, b, c = p
        area = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        if abs(area) < 1e-12:
            continue
        lo = np.maximum(np.floor(p.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(p.max(0)).astype(int), [width-1, height-1])
        y, x = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        u = ((b[0]-x)*(c[1]-y)-(b[1]-y)*(c[0]-x))/area
        v = ((c[0]-x)*(a[1]-y)-(c[1]-y)*(a[0]-x))/area
        w = 1-u-v
        valid = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
        inv = u/q[0,2]+v/q[1,2]+w/q[2,2]
        depth = np.divide(1, inv, out=np.full(inv.shape, np.inf), where=valid & (inv > 0))
        dest = result[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        np.minimum(dest, depth, out=dest)
    result[~np.isfinite(result)] = np.nan
    return result


def depth_crop(depth, original_k, stage_k, crop):
    left, top, width, height = crop
    y, x = np.mgrid[top:top+height, left:left+width]
    # Nearest sample: do not invent a smooth surface across missing pixels.
    sx = np.rint((x+.5-original_k[2])/original_k[0]*stage_k[0]+stage_k[2]-.5).astype(int)
    sy = np.rint((y+.5-original_k[3])/original_k[1]*stage_k[1]+stage_k[3]-.5).astype(int)
    inside = (sx >= 0) & (sy >= 0) & (sx < depth.shape[1]) & (sy < depth.shape[0])
    out = np.full((height, width), np.nan)
    out[inside] = depth[sy[inside], sx[inside]]
    out[out <= 0] = np.nan
    return out


def relief_metrics(reference, candidate, common_mask=None):
    common = np.isfinite(reference) & np.isfinite(candidate)
    if common_mask is not None:
        common &= common_mask
    if common.sum() < 16:
        return {'common_pixels': int(common.sum()), 'coverage': float(np.isfinite(candidate).mean())}, None
    y, x = np.indices(reference.shape)
    basis = np.column_stack((np.ones(common.sum()), x[common], y[common]))
    # Inverse depth of a plane is affine; remove its fitted trend separately.
    inv_ref, inv_test = 1/reference[common], 1/candidate[common]
    ref_relief = inv_ref-basis@np.linalg.lstsq(basis, inv_ref, rcond=None)[0]
    test_relief = inv_test-basis@np.linalg.lstsq(basis, inv_test, rcond=None)[0]
    denominator = float(ref_relief@ref_relief)
    metrics = {
        'common_pixels': int(common.sum()), 'coverage': float(np.isfinite(candidate).mean()),
        'ray_depth_rmse': float(np.sqrt(np.mean((candidate[common]-reference[common])**2))),
        'ray_depth_median_bias': float(np.median(candidate[common]-reference[common])),
        'inverse_depth_relief_gain': float(ref_relief@test_relief/denominator) if denominator > 1e-20 else None,
        'inverse_depth_relief_correlation': float(np.corrcoef(ref_relief,test_relief)[0,1]) if np.std(ref_relief)>1e-12 and np.std(test_relief)>1e-12 else None,
        'inverse_depth_relief_rmse': float(np.sqrt(np.mean((ref_relief-test_relief)**2))),
    }
    # Broad head curvature can conceal a missing eye in the plane residual.
    # Measure several finer scales too, on identical valid support in both maps.
    from scipy.ndimage import gaussian_filter
    fine = {}
    for sigma in [2, 4, 8]:
        weight = gaussian_filter(common.astype(float), sigma, mode='nearest')
        residuals = []
        for depth in [reference, candidate]:
            inv = np.zeros(depth.shape); inv[common] = 1/depth[common]
            trend = gaussian_filter(inv, sigma, mode='nearest')/np.maximum(weight, 1e-20)
            residuals.append((inv-trend)[common])
        r, t = residuals; den = float(r@r)
        fine[str(sigma)] = {
            'gain': float(r@t/den) if den>1e-20 else None,
            'correlation': float(np.corrcoef(r,t)[0,1]) if np.std(r)>1e-12 and np.std(t)>1e-12 else None,
            'rmse': float(np.sqrt(np.mean((r-t)**2))),
        }
    metrics['fine_inverse_depth_relief_by_sigma_pixels'] = fine
    maps=[]
    for values in residuals:
        m=np.full(reference.shape,np.nan);m[common]=values;maps.append(m)
    return metrics,maps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-plots',action='store_true',help='Write measurements only; needs NumPy/SciPy but not Matplotlib')
    a=p.parse_args();spec=json.loads(a.spec.read_text())
    rows=json.loads(Path(spec['cameras']).read_text())['views']
    ref=read_stl(spec['evaluation_reference_stl'])
    meshes={name:read_stl(path) for name,path in spec.get('meshes',{}).items()}
    a.output.mkdir(parents=True,exist_ok=False)
    report={'evaluation_only':True,'reference_used_for_reconstruction':False,
            'region_definitions':spec['regions'],
            'limits':['Reference must already be aligned independently; this script does not align it.',
                      'Relief uses separately fitted inverse-depth planes on common rays; also inspect absolute error and coverage.',
                      'Pointwise metrics can include scan alignment error. Correlation alone is not feature recovery.'], 'regions':{}}
    plots=[]
    for region in spec['regions']:
        i=next(i for i,r in enumerate(rows) if r['name']==region['view']);row=rows[i];crop=region['crop']
        reference=ray_depth(ref,row,crop);stages={}
        for name,s in spec.get('depths',{}).items():
            meta=json.loads(Path(s['metadata']).read_text())
            with np.load(s['archive']) as archive:
                stages[name]=depth_crop(archive[f'depth_{i:03d}'],row['k'],meta['views'][i]['k'],crop)
        stages.update({name:ray_depth(t,row,crop) for name,t in meshes.items()})
        fixed_common=np.isfinite(reference)
        for depth in stages.values():fixed_common &= np.isfinite(depth)
        rr={}
        for name,depth in stages.items():
            metrics,_=relief_metrics(reference,depth)
            fixed,maps=relief_metrics(reference,depth,fixed_common)
            metrics['fixed_common_ray_metrics']=fixed;rr[name]=metrics
            if maps is not None: plots.append((region['name']+' / '+region['view']+' / '+name,maps,fixed))
        report['regions'][region['name']+'/'+region['view']]=rr
    (a.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if a.no_plots:
        print(json.dumps(report,indent=2));return
    # Standalone exportable diagnostic, not a reconstruction or a beauty render.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for start in range(0,len(plots),6):
        batch=plots[start:start+6];fig,axes=plt.subplots(len(batch),3,figsize=(12,3*len(batch)),squeeze=False)
        for axes_row,(title,maps,metrics) in zip(axes,batch):
            scale=max(np.nanpercentile(np.abs(maps[0]),95),1e-12)
            for ax,data,label in zip(axes_row[:2],maps,['Scanner relief','Recovered relief']):
                ax.imshow(data,cmap='coolwarm',vmin=-scale,vmax=scale);ax.set_title(title+'\n'+label+' (sigma 8 px high-pass)',fontsize=8);ax.axis('off')
            ax=axes_row[2]
            for data,label in zip(maps,['Scanner','Recovered']):
                count=np.isfinite(data).sum(0);values=np.nansum(data,axis=0)/np.maximum(count,1);values[count==0]=np.nan
                ax.plot(values,label=label)
            gain=metrics['fine_inverse_depth_relief_by_sigma_pixels']['8']['gain']
            ax.legend();ax.set_title('Mean column relief; gain '+(f'{gain:.2f}' if gain is not None else 'undefined on flat reference'),fontsize=8)
        fig.tight_layout();fig.savefig(a.output/f'relief-{start//6:02d}.png',dpi=130);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
