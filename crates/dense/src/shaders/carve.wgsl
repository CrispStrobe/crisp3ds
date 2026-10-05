// Silhouette violations per voxel centre (port of `carve` in multiscale_stereo.py).
//
// One invocation handles one voxel of a voxel list and loops over the views of
// one mask batch; batches accumulate into `counts`. Masks are bit-packed, one
// row per `ceil(width / 32)` words. `ROW` is prepended by the dispatch layer.

struct Params {
    count: u32,
    views: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Four rows per view: rotation rows with the translation in w, then fx, fy, cx, cy.
@group(0) @binding(1) var<storage, read> cameras: array<vec4<f32>>;
// Per view: width, height, first mask word, 1 if the mask stays clear of the image border.
@group(0) @binding(2) var<storage, read> frames: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> masks: array<u32>;
// Voxel centre coordinates along x, then y, then z.
@group(0) @binding(4) var<storage, read> axes: array<f32>;
@group(0) @binding(5) var<storage, read> voxels: array<u32>;
@group(0) @binding(6) var<storage, read_write> counts: array<u32>;

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
    var violations = 0u;
    for (var v = 0u; v < params.views; v++) {
        let r0 = cameras[4u * v];
        let r1 = cameras[4u * v + 1u];
        let r2 = cameras[4u * v + 2u];
        let kk = cameras[4u * v + 3u];
        let px = point.x * r0.x + point.y * r0.y + point.z * r0.z + r0.w;
        let py = point.x * r1.x + point.y * r1.y + point.z * r1.z + r1.w;
        let pz = point.x * r2.x + point.y * r2.y + point.z * r2.z + r2.w;
        let z = max(pz, 1e-6);
        // round() is round-half-to-even, like torch.round.
        let u = round(px / z * kk.x + kk.z - 0.5);
        let w = round(py / z * kk.y + kk.w - 0.5);
        let frame = frames[v];
        let inside = u >= 0.0 && u < f32(frame.x) && w >= 0.0 && w < f32(frame.y) && pz > 0.0;
        var hit = false;
        if (inside) {
            let ui = u32(u);
            let stride = (frame.x + 31u) / 32u;
            let word = masks[frame.z + u32(w) * stride + ui / 32u];
            hit = ((word >> (ui % 32u)) & 1u) == 1u;
        }
        if (frame.w == 1u) {
            violations += u32(!(inside && hit));
        } else {
            violations += u32(inside && !hit);
        }
    }
    counts[n] += violations;
}
