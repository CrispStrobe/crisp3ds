// Warp of one neighbour image onto the hypotheses of a reference view (the
// sampling half of `Stereo.score`).
//
// One invocation handles one (hypothesis, reference pixel). The pixel is lifted
// to the hypothesis depth, tested against the reference mask and the grown
// hull, projected into the neighbour and sampled bilinearly with the
// conventions of `grid_sample(align_corners=True, padding_mode="zeros")`. The
// sample is valid when the hull and the mask allow it, the bilinear mask sample
// is at least 0.999 and the point is in front of the neighbour. Invalid samples
// are written as INVALID. `ROW` is prepended by the dispatch layer.

const INVALID: f32 = -1e30;

struct Params {
    // Rotation rows with the translation in w, then fx, fy, cx, cy.
    ref_r0: vec4<f32>,
    ref_r1: vec4<f32>,
    ref_r2: vec4<f32>,
    ref_k: vec4<f32>,
    nbr_r0: vec4<f32>,
    nbr_r1: vec4<f32>,
    nbr_r2: vec4<f32>,
    nbr_k: vec4<f32>,
    // Hull origin and voxel size.
    hull: vec4<f32>,
    // 2 / (width - 1), 2 / (height - 1) of the neighbour, as the reference normalises its grid.
    grid: vec4<f32>,
    // Hull nx, ny, nz; mode (0: offsets are inverse depths, 1: offsets are added to 1 / init).
    shape: vec4<u32>,
    // Reference width, height; neighbour width, height.
    size: vec4<u32>,
    // Hypotheses in this chunk; index of the chunk's first offset.
    chunk: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ref_mask: array<u32>;
@group(0) @binding(2) var<storage, read> hull: array<u32>;
@group(0) @binding(3) var<storage, read> init: array<f32>;
@group(0) @binding(4) var<storage, read> nbr_gray: array<f32>;
@group(0) @binding(5) var<storage, read> nbr_mask: array<u32>;
@group(0) @binding(6) var<storage, read> offsets: array<f32>;
@group(0) @binding(7) var<storage, read_write> warped: array<f32>;

fn mask_at(x: i32, y: i32) -> f32 {
    let w = i32(params.size.z);
    if (x < 0 || y < 0 || x >= w || y >= i32(params.size.w)) {
        return 0.0;
    }
    let stride = (params.size.z + 31u) / 32u;
    let word = nbr_mask[u32(y) * stride + u32(x) / 32u];
    return f32((word >> (u32(x) % 32u)) & 1u);
}

fn gray_at(x: i32, y: i32) -> f32 {
    let w = i32(params.size.z);
    if (x < 0 || y < 0 || x >= w || y >= i32(params.size.w)) {
        return 0.0;
    }
    return nbr_gray[u32(y) * params.size.z + u32(x)];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x + id.y * ROW;
    let pixels = params.size.x * params.size.y;
    let hypothesis = n / pixels;
    if (hypothesis >= params.chunk.x) {
        return;
    }
    let p = n % pixels;
    let x = p % params.size.x;
    let y = p / params.size.x;
    let stride = (params.size.x + 31u) / 32u;
    let in_mask = ((ref_mask[y * stride + x / 32u] >> (x % 32u)) & 1u) == 1u;
    if (!in_mask) {
        warped[n] = INVALID;
        return;
    }
    var inverse = offsets[params.chunk.y + hypothesis];
    if (params.shape.w == 1u) {
        inverse = 1.0 / max(init[p], 1e-6) + inverse;
    }
    let depth = 1.0 / max(inverse, 1e-6);
    // world = (ray * depth - t) @ R
    let k = params.ref_k;
    let cx = (f32(x) + 0.5 - k.z) / k.x * depth - params.ref_r0.w;
    let cy = (f32(y) + 0.5 - k.w) / k.y * depth - params.ref_r1.w;
    let cz = depth - params.ref_r2.w;
    let world = vec3<f32>(
        cx * params.ref_r0.x + cy * params.ref_r1.x + cz * params.ref_r2.x,
        cx * params.ref_r0.y + cy * params.ref_r1.y + cz * params.ref_r2.y,
        cx * params.ref_r0.z + cy * params.ref_r1.z + cz * params.ref_r2.z,
    );
    let cell = floor((world - params.hull.xyz) / params.hull.w);
    let limit = vec3<f32>(params.shape.xyz);
    if (!(all(cell >= vec3<f32>(0.0)) && all(cell < limit))) {
        warped[n] = INVALID;
        return;
    }
    let linear = (u32(cell.x) * params.shape.y + u32(cell.y)) * params.shape.z + u32(cell.z);
    if (((hull[linear / 32u] >> (linear % 32u)) & 1u) == 0u) {
        warped[n] = INVALID;
        return;
    }
    let qx = world.x * params.nbr_r0.x + world.y * params.nbr_r0.y + world.z * params.nbr_r0.z + params.nbr_r0.w;
    let qy = world.x * params.nbr_r1.x + world.y * params.nbr_r1.y + world.z * params.nbr_r1.z + params.nbr_r1.w;
    let qz = world.x * params.nbr_r2.x + world.y * params.nbr_r2.y + world.z * params.nbr_r2.z + params.nbr_r2.w;
    if (!(qz > 0.0)) {
        warped[n] = INVALID;
        return;
    }
    let zc = max(qz, 1e-6);
    let u = qx / zc * params.nbr_k.x + params.nbr_k.z - 0.5;
    let v = qy / zc * params.nbr_k.y + params.nbr_k.w - 0.5;
    // Through the normalised grid and back, as grid_sample does with align_corners.
    let gx = u * params.grid.x - 1.0;
    let gy = v * params.grid.y - 1.0;
    let ix = (gx + 1.0) / 2.0 * f32(params.size.z - 1u);
    let iy = (gy + 1.0) / 2.0 * f32(params.size.w - 1u);
    // Far outside the image nothing is sampled; this also keeps the integer conversion in range.
    if (!(ix > -2.0 && iy > -2.0 && ix < f32(params.size.z) + 1.0 && iy < f32(params.size.w) + 1.0)) {
        warped[n] = INVALID;
        return;
    }
    let x0f = floor(ix);
    let y0f = floor(iy);
    let x0 = i32(x0f);
    let y0 = i32(y0f);
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let nw = (x0f + 1.0 - ix) * (y0f + 1.0 - iy);
    let ne = (ix - x0f) * (y0f + 1.0 - iy);
    let sw = (x0f + 1.0 - ix) * (iy - y0f);
    let se = (ix - x0f) * (iy - y0f);
    let coverage = mask_at(x0, y0) * nw + mask_at(x1, y0) * ne + mask_at(x0, y1) * sw + mask_at(x1, y1) * se;
    if (!(coverage >= 0.999)) {
        warped[n] = INVALID;
        return;
    }
    warped[n] = gray_at(x0, y0) * nw + gray_at(x1, y0) * ne + gray_at(x0, y1) * sw + gray_at(x1, y1) * se;
}
