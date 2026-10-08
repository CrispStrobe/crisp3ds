// Truncated signed distance votes per hull voxel (port of the inner loop of
// `Stereo.tsdf`). One invocation handles one voxel and adds the votes of the
// views of one batch, in view order, to `total` and `weight`. Depth maps are
// looked up at the nearest pixel (round-half-to-even), zero outside the map.
// `ROW` is prepended by the dispatch layer.

struct Params {
    count: u32,
    views: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    interpolate: u32,
    pad1: u32,
    pad2: u32,
    truncation: f32,
    behind: f32,
    behind_weight: f32,
    free_weight: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Four rows per view: rotation rows with the translation in w, then fx, fy, cx, cy of the level.
@group(0) @binding(1) var<storage, read> cameras: array<vec4<f32>>;
// Per view: width, height, offset of its depth map.
@group(0) @binding(2) var<storage, read> frames: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> depths: array<f32>;
@group(0) @binding(4) var<storage, read> axes: array<f32>;
@group(0) @binding(5) var<storage, read> voxels: array<u32>;
@group(0) @binding(6) var<storage, read_write> totals: array<f32>;
@group(0) @binding(7) var<storage, read_write> weights: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x + id.y * ROW;
    if (n >= params.count) {
        return;
    }
    let linear = voxels[n];
    let k = linear % params.nz;
    let ij = linear / params.nz;
    let j = ij % params.ny;
    let i = ij / params.ny;
    let point = vec3<f32>(axes[i], axes[params.nx + j], axes[params.nx + params.ny + k]);
    var total = totals[n];
    var weight = weights[n];
    for (var v = 0u; v < params.views; v++) {
        let r0 = cameras[4u * v];
        let r1 = cameras[4u * v + 1u];
        let r2 = cameras[4u * v + 2u];
        let kk = cameras[4u * v + 3u];
        let px = point.x * r0.x + point.y * r0.y + point.z * r0.z + r0.w;
        let py = point.x * r1.x + point.y * r1.y + point.z * r1.z + r1.w;
        let z = point.x * r2.x + point.y * r2.y + point.z * r2.z + r2.w;
        let zc = max(z, 1e-6);
        let x = round(px / zc * kk.x + kk.z - 0.5);
        let y = round(py / zc * kk.y + kk.w - 0.5);
        let frame = frames[v];
        var measured = 0.0;
        if (x >= 0.0 && x < f32(frame.x) && y >= 0.0 && y < f32(frame.y)) {
            measured = depths[frame.z + u32(y) * frame.x + u32(x)];
        }
        if (params.interpolate != 0u) {
            let fx = px / zc * kk.x + kk.z - 0.5;
            let fy = py / zc * kk.y + kk.w - 0.5;
            let x0 = floor(fx);
            let y0 = floor(fy);
            if (x0 >= 0.0 && y0 >= 0.0 && x0 + 1.0 < f32(frame.x) && y0 + 1.0 < f32(frame.y)) {
                let at = frame.z + u32(y0) * frame.x + u32(x0);
                let a = depths[at];
                let b = depths[at + 1u];
                let c = depths[at + frame.x];
                let d = depths[at + frame.x + 1u];
                let low = min(min(a,b),min(c,d));
                let high = max(max(a,b),max(c,d));
                if (low > 0.0 && high-low <= params.truncation) {
                    let u = fx-x0;
                    let v = fy-y0;
                    measured = 1.0 / ((1.0-u)*(1.0-v)/a + u*(1.0-v)/b + (1.0-u)*v/c + u*v/d);
                }
            }
        }
        let sdf = measured - z;
        let seen = measured > 0.0 && z > 0.0;
        let near = seen && sdf > -params.truncation;
        let behind = seen && !near && sdf > -params.behind;
        var vote = 0.0;
        if (near) {
            vote = select(1.0, params.free_weight, sdf >= params.truncation);
        }
        let inside = select(0.0, params.behind_weight, behind);
        total += vote * min(sdf / params.truncation, 1.0) - inside;
        weight += vote + inside;
    }
    totals[n] = total;
    weights[n] = weight;
}
