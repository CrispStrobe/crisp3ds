// Pixels of one view that a list of voxel centres projects onto, as a 2x2 splat
// (port of `Stereo.cover`). The cover image is bit-packed, one row per `stride`
// words; bits are set with atomicOr, so the result does not depend on the order
// of invocations. `ROW` is prepended by the dispatch layer.

struct Params {
    count: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    width: u32,
    height: u32,
    stride: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Rotation rows with the translation in w, then fx, fy, cx, cy.
@group(0) @binding(1) var<storage, read> camera: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> axes: array<f32>;
@group(0) @binding(3) var<storage, read> voxels: array<u32>;
@group(0) @binding(4) var<storage, read_write> cover: array<atomic<u32>>;

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
    let r0 = camera[0];
    let r1 = camera[1];
    let r2 = camera[2];
    let kk = camera[3];
    let px = point.x * r0.x + point.y * r0.y + point.z * r0.z + r0.w;
    let py = point.x * r1.x + point.y * r1.y + point.z * r1.z + r1.w;
    let pz = point.x * r2.x + point.y * r2.y + point.z * r2.z + r2.w;
    if (!(pz > 0.0)) {
        return;
    }
    let u = px / pz * kk.x + kk.z - 0.5;
    let v = py / pz * kk.y + kk.w - 0.5;
    if (!(u >= 0.0 && u < f32(params.width - 1u) && v >= 0.0 && v < f32(params.height - 1u))) {
        return;
    }
    let ui = u32(floor(u));
    let vi = u32(floor(v));
    for (var dv = 0u; dv < 2u; dv++) {
        for (var du = 0u; du < 2u; du++) {
            let x = ui + du;
            atomicOr(&cover[(vi + dv) * params.stride + x / 32u], 1u << (x % 32u));
        }
    }
}
