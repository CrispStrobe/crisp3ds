"""Independent two-view OpenCV SGBM control from own recovered pinhole inputs.

Use original lens-undistorted RGB and own masks/poses/sparse points, never
supplied dataset poses, masks or depth. This is a diagnostic, not a COLMAP MVS
benchmark or production backend. OpenCV is an optional tool dependency only.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np


def intrinsic(row):
    fx, fy, cx, cy = row['k']
    return np.array([[fx, 0, cx-.5], [0, fy, cy-.5], [0, 0, 1.]])


def match_rectified(images, masks, low, count):
    if count <= 0 or count % 16 or count > 512:
        raise ValueError('disparity count must be a positive multiple of 16, at most 512')
    common = dict(numDisparities=count, blockSize=5, P1=8*3*25, P2=32*3*25,
                  disp12MaxDiff=1, preFilterCap=31, uniquenessRatio=10,
                  speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    dl = cv2.StereoSGBM_create(minDisparity=low, **common).compute(*images).astype(np.float32)/16
    rightlow = -low-count+1
    dr = cv2.StereoSGBM_create(minDisparity=rightlow, **common).compute(images[1], images[0]).astype(np.float32)/16
    yy, xx = np.indices(dl.shape);xr = np.rint(xx-dl).astype(np.int32)
    inside = (xr >= 0) & (xr < dl.shape[1]);xr = np.clip(xr, 0, dl.shape[1]-1)
    valid = (dl > low-1) & (dl < low+count-1) & inside & (masks[0] > 0) & (masks[1][yy, xr] > 0)
    valid &= (dr[yy, xr] > rightlow-1) & (np.abs(dl+dr[yy, xr]) <= 1)
    return dl, valid


def original_depth(disparity, valid, q, rotation, k, size):
    """Rectified camera Z -> original camera Z, nearest Z-buffer reprojection."""
    points = cv2.reprojectImageTo3D(disparity, q)@rotation
    good = valid & np.isfinite(points).all(2) & (points[:, :, 2] > 0)
    pc = points[good]
    uv = pc[:, :2]/pc[:, 2:]*[k[0, 0], k[1, 1]]+[k[0, 2], k[1, 2]]
    ix, iy = np.rint(uv).astype(np.int32).T
    inside = (ix >= 0) & (ix < size[0]) & (iy >= 0) & (iy < size[1])
    plane = np.full(size[0]*size[1], np.inf, dtype=np.float32)
    np.minimum.at(plane, iy[inside]*size[0]+ix[inside], pc[inside, 2])
    plane = plane.reshape(size[1], size[0]);plane[~np.isfinite(plane)] = 0
    return plane


def pair_depth(a, b, images, masks, sparse, padding):
    size = (a['width'], a['height'])
    if size != (b['width'], b['height']) or any(im.shape[:2] != size[::-1] for im in images+masks):
        raise ValueError('both photos/masks must match the common original camera canvas')
    ra, rb = np.array(a['rotation']), np.array(b['rotation']);ta, tb = np.array(a['translation']), np.array(b['translation'])
    ka, kb = intrinsic(a), intrinsic(b);relative = rb@ra.T;translation = tb-relative@ta
    if np.linalg.norm(translation) < 1e-9:
        raise ValueError('stereo requires a nonzero baseline')
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(ka, None, kb, None, size, relative, translation,
                                              flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    if abs(p2[0, 3]) <= abs(p2[1, 3]):
        raise ValueError('vertical rectification is unsupported by this horizontal control')
    rectified_sparse = (sparse@ra.T+ta)@r1.T
    z = rectified_sparse[:, 2];z = z[np.isfinite(z) & (z > 0)]
    if len(z) < 2:
        raise ValueError('own sparse points must provide positive rectified depth bounds')
    disparity = -p2[0, 3]/z+p1[0, 2]-p2[0, 2]
    low = int(np.floor(np.percentile(disparity, 1)-padding))
    high = int(np.ceil(np.percentile(disparity, 99)+padding));count = int(np.ceil((high-low)/16)*16)
    warped_images, warped_masks = [], []
    for im, mask, k, r, p in zip(images, masks, [ka, kb], [r1, r2], [p1, p2]):
        mx, my = cv2.initUndistortRectifyMap(k, None, r, p[:, :3], size, cv2.CV_32FC1)
        warped_images.append(cv2.remap(im, mx, my, cv2.INTER_LINEAR))
        warped_masks.append(cv2.remap(mask, mx, my, cv2.INTER_NEAREST))
    disparity, valid = match_rectified(warped_images, warped_masks, low, count)
    return original_depth(disparity, valid, q, r1, ka, size), {'minimum_disparity': low, 'num_disparities': count, 'valid_rectified_pixels': int(valid.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='cameras.json with original lens-undistorted RGB, masks, and own sparse_points.npy')
    parser.add_argument('--stage-cameras', type=Path, required=True, help='own native-level crop metadata; no depth archive is read')
    parser.add_argument('--output', type=Path, required=True, help='fresh output directory')
    parser.add_argument('--pair-gap', type=int, default=4)
    parser.add_argument('--padding', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    try:
        if args.threads < 1 or args.padding < 1:
            raise ValueError('threads and disparity padding must be positive')
        cv2.setNumThreads(args.threads)
        rows = json.loads((args.inputs/'cameras.json').read_text())['views']
        meta = json.loads(args.stage_cameras.read_text());stage = meta['views'];boxes = meta['boxes']
        if len(rows) != len(stage) or len(rows) != len(boxes) or not 0 < args.pair_gap < len(rows):
            raise ValueError('matching camera/crop counts and 0 < pair gap < views required')
        sparse = np.load(args.inputs/'sparse_points.npy', allow_pickle=False)
        if sparse.ndim != 2 or sparse.shape[1] != 3 or not np.isfinite(sparse).all():
            raise ValueError('own sparse points must be finite N x 3')
        for row, crop, st in zip(rows, boxes, stage):
            x0, y0, x1, y1 = crop
            expected = np.array(row['k'])-[0, 0, x0, y0]
            if not np.allclose(expected, st['k'], rtol=1e-6, atol=1e-4) or (y1-y0, x1-x0) != (st['height'], st['width']):
                raise ValueError('this control requires native-resolution crop metadata')
        args.output.mkdir(parents=True, exist_ok=False)
        depths, records = {}, [];started = time.monotonic()
        for i, a in enumerate(rows):
            j = (i+args.pair_gap) % len(rows);b = rows[j]
            images, masks = [], []
            for row in [a, b]:
                images.append(cv2.imread(str(args.inputs/row['image'])))
                masks.append(cv2.imread(str(args.inputs/row['mask']), cv2.IMREAD_GRAYSCALE))
            if any(im is None for im in images+masks):
                raise ValueError('unable to read source RGB or own mask')
            depth, record = pair_depth(a, b, images, masks, sparse, args.padding)
            x0, y0, x1, y1 = boxes[i];crop = depth[y0:y1, x0:x1]
            depths[f'depth_{i:03}'] = crop
            records.append(dict(record, view=i, source_view=j, valid_crop_pixels=int((crop > 0).sum())))
        np.savez_compressed(args.output/'depths.npz', **depths)
        report = {'reference_used': False, 'opencv_version': cv2.__version__, 'pair_gap': args.pair_gap,
                  'padding': args.padding, 'threads': args.threads, 'views': records, 'seconds': time.monotonic()-started,
                  'limits': ['Two-view SGBM, not COLMAP MVS', 'Reprojection sparsifies depths', 'Same masks/cameras do not validate their correctness', 'No production adoption or patent-clearance claim']}
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    except (ValueError, OSError, KeyError, cv2.error) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
