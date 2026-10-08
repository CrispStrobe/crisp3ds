// Independent per-pixel slanted-plane refinement.
// Each invocation reads and updates only planes[p] and scores[p].
// Neighboring photographs supply brightness samples, never plane candidates.
// No propagation, random search, or geometric reprojection cost is used.
// ROW is prepended by the dispatch layer.

const INVALID_SCORE: f32 = -2.0;
// Percentile stretching keeps valid negative brightness values.
const INVALID_SAMPLE: f32 = -1e30;
const NEIGHBOURS: u32 = 8u;  // neighbour views a plane is scored against, at most

struct Neighbour {
    // Reference camera frame to neighbour camera frame: rows of R with t in w.
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    // fx, fy, cx, cy of the neighbour.
    k: vec4<f32>,
    // Width, height, first grey value, first mask word.
    extent: vec4<u32>,
}

struct Params {
    // fx, fy, cx, cy of the reference.
    ref_k: vec4<f32>,
    // Reference camera to world: rows of R^T with -R^T t in w (world = R^T x - R^T t).
    world_r0: vec4<f32>,
    world_r1: vec4<f32>,
    world_r2: vec4<f32>,
    // Search hull origin and voxel size; its shape.
    hull: vec4<f32>,
    shape: vec4<u32>,
    // Reference width, height; neighbour count; window radius.
    size: vec4<u32>,
    // Window stride; best_of; two reserved components.
    pass_: vec4<u32>,
    // min_variance, window_fill, inverse-depth perturbation, normal perturbation.
    gates: vec4<f32>,
    // Smallest and largest inverse depth at its pixel; two reserved components.
    range: vec4<f32>,
    nbr: array<Neighbour, 8>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ref_gray: array<f32>;
@group(0) @binding(2) var<storage, read> ref_mask: array<u32>;
@group(0) @binding(3) var<storage, read> nbr_gray: array<f32>;
@group(0) @binding(4) var<storage, read> nbr_mask: array<u32>;
@group(0) @binding(5) var<storage, read> hull: array<u32>;
@group(0) @binding(6) var<storage, read_write> planes: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> scores: array<f32>;

fn in_ref_mask(x: i32, y: i32) -> bool {
    if (x < 0 || y < 0 || x >= i32(params.size.x) || y >= i32(params.size.y)) {
        return false;
    }
    let stride = (params.size.x + 31u) / 32u;
    return ((ref_mask[u32(y) * stride + u32(x) / 32u] >> (u32(x) % 32u)) & 1u) == 1u;
}

fn ray(x: f32, y: f32) -> vec3<f32> {
    let k = params.ref_k;
    return vec3<f32>((x + 0.5 - k.z) / k.x, (y + 0.5 - k.w) / k.y, 1.0);
}

fn nbr_mask_at(j: u32, x: i32, y: i32) -> f32 {
    let extent = params.nbr[j].extent;
    if (x < 0 || y < 0 || x >= i32(extent.x) || y >= i32(extent.y)) {
        return 0.0;
    }
    let stride = (extent.x + 31u) / 32u;
    let word = nbr_mask[extent.w + u32(y) * stride + u32(x) / 32u];
    return f32((word >> (u32(x) % 32u)) & 1u);
}

fn nbr_gray_at(j: u32, x: i32, y: i32) -> f32 {
    let extent = params.nbr[j].extent;
    if (x < 0 || y < 0 || x >= i32(extent.x) || y >= i32(extent.y)) {
        return 0.0;
    }
    return nbr_gray[extent.z + u32(y) * extent.x + u32(x)];
}

// Bilinear sample of neighbour j at the reference-frame point `point`; negative when invalid.
fn sample(j: u32, point: vec3<f32>) -> f32 {
    let n = params.nbr[j];
    let qx = dot(n.r0.xyz, point) + n.r0.w;
    let qy = dot(n.r1.xyz, point) + n.r1.w;
    let qz = dot(n.r2.xyz, point) + n.r2.w;
    if (!(qz > 0.0)) {
        return INVALID_SAMPLE;
    }
    let ix = qx / qz * n.k.x + n.k.z - 0.5;
    let iy = qy / qz * n.k.y + n.k.w - 0.5;
    if (!(ix > -2.0 && iy > -2.0 && ix < f32(n.extent.x) + 1.0 && iy < f32(n.extent.y) + 1.0)) {
        return INVALID_SAMPLE;
    }
    let x0f = floor(ix);
    let y0f = floor(iy);
    let x0 = i32(x0f);
    let y0 = i32(y0f);
    let nw = (x0f + 1.0 - ix) * (y0f + 1.0 - iy);
    let ne = (ix - x0f) * (y0f + 1.0 - iy);
    let sw = (x0f + 1.0 - ix) * (iy - y0f);
    let se = (ix - x0f) * (iy - y0f);
    let coverage = nbr_mask_at(j, x0, y0) * nw + nbr_mask_at(j, x0 + 1, y0) * ne + nbr_mask_at(j, x0, y0 + 1) * sw
        + nbr_mask_at(j, x0 + 1, y0 + 1) * se;
    if (!(coverage >= 0.999)) {
        return INVALID_SAMPLE;
    }
    return nbr_gray_at(j, x0, y0) * nw + nbr_gray_at(j, x0 + 1, y0) * ne + nbr_gray_at(j, x0, y0 + 1) * sw
        + nbr_gray_at(j, x0 + 1, y0 + 1) * se;
}

fn in_hull(point: vec3<f32>) -> bool {
    let world = vec3<f32>(
        dot(params.world_r0.xyz, point) + params.world_r0.w,
        dot(params.world_r1.xyz, point) + params.world_r1.w,
        dot(params.world_r2.xyz, point) + params.world_r2.w,
    );
    let cell = floor((world - params.hull.xyz) / params.hull.w);
    if (!(all(cell >= vec3<f32>(0.0)) && all(cell < vec3<f32>(params.shape.xyz)))) {
        return false;
    }
    let linear = (u32(cell.x) * params.shape.y + u32(cell.y)) * params.shape.z + u32(cell.z);
    return ((hull[linear / 32u] >> (linear % 32u)) & 1u) == 1u;
}

// Score of `plane` at reference pixel (x, y).
fn score(x: i32, y: i32, plane: vec4<f32>) -> f32 {
    let depth = plane.x;
    let normal = plane.yzw;
    let r = ray(f32(x), f32(y));
    if (!(depth > 0.0) || !(dot(normal, r) < 0.0)) {
        return INVALID_SCORE;
    }
    let inverse = 1.0 / depth;
    if (inverse < params.range.x || inverse > params.range.y) {
        return INVALID_SCORE;
    }
    let centre = r * depth;
    if (!in_hull(centre)) {
        return INVALID_SCORE;
    }
    let offset = dot(normal, centre);
    let radius = i32(params.size.w);
    let stride = i32(params.pass_.x);
    let side = f32((2u * params.size.w) / params.pass_.x + 1u);
    let scale = 1.0 / (side * side);
    // Window samples outside, neighbours inside: the lifted point and the reference value are
    // computed once per sample. Six sums per neighbour.
    var sums: array<array<f32, 6>, NEIGHBOURS>;
    for (var j = 0u; j < NEIGHBOURS; j++) {
        sums[j] = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    for (var dy = -radius; dy <= radius; dy += stride) {
        for (var dx = -radius; dx <= radius; dx += stride) {
            let qx = x + dx;
            let qy = y + dy;
            if (!in_ref_mask(qx, qy)) {
                continue;
            }
            let rq = ray(f32(qx), f32(qy));
            let along = dot(normal, rq);
            if (!(along < -1e-6)) {
                continue;
            }
            let point = rq * (offset / along);
            let a = ref_gray[u32(qy) * params.size.x + u32(qx)] - 0.5;
            for (var j = 0u; j < params.size.z; j++) {
                let value = sample(j, point);
                if (value == INVALID_SAMPLE) {
                    continue;
                }
                let b = value - 0.5;
                sums[j][0] += 1.0;
                sums[j][1] += a;
                sums[j][2] += b;
                sums[j][3] += a * a;
                sums[j][4] += b * b;
                sums[j][5] += a * b;
            }
        }
    }
    var best = array<f32, 4>(INVALID_SCORE, INVALID_SCORE, INVALID_SCORE, INVALID_SCORE);
    for (var j = 0u; j < params.size.z; j++) {
        let fill = sums[j][0] * scale;
        let d = max(fill, 1e-6);
        let mr = sums[j][1] * scale / d;
        let mt = sums[j][2] * scale / d;
        let vr = max(sums[j][3] * scale / d - mr * mr, 0.0);
        let vt = max(sums[j][4] * scale / d - mt * mt, 0.0);
        var s = INVALID_SCORE;
        if (fill >= params.gates.y && vr >= params.gates.x && vt >= params.gates.x) {
            s = clamp((sums[j][5] * scale / d - mr * mt) / sqrt(vr * vt + 1e-12), -1.0, 1.0);
        }
        // Insert into the running best list (descending).
        for (var rank = 0u; rank < 4u; rank++) {
            let lower = min(best[rank], s);
            best[rank] = max(best[rank], s);
            s = lower;
        }
    }
    var total = 0.0;
    var valid = 0u;
    for (var rank = 0u; rank < params.pass_.y; rank++) {
        if (best[rank] > -1.5) {
            total += best[rank];
            valid += 1u;
        }
    }
    if (valid < 2u) {
        return INVALID_SCORE;
    }
    return total / f32(valid);
}


@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x + id.y * ROW;
    if (p >= params.size.x * params.size.y) { return; }
    let x = i32(p % params.size.x);
    let y = i32(p / params.size.x);
    if (!in_ref_mask(x, y)) { scores[p] = INVALID_SCORE; return; }
    scores[p] = score(x, y, planes[p]);
}

@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x + id.y * ROW;
    if (p >= params.size.x * params.size.y) { return; }
    let x = i32(p % params.size.x);
    let y = i32(p / params.size.x);
    if (!in_ref_mask(x, y)) { return; }
    var best = planes[p];
    var best_score = scores[p];
    if (!(best.x > 0.0)) { return; }
    for (var axis = 0u; axis < 3u; axis++) {
        let centre = best;
        for (var side = 0u; side < 2u; side++) {
            let direction = select(-1.0, 1.0, side == 1u);
            var candidate = centre;
            if (axis == 0u) {
                let inverse = 1.0 / centre.x + direction * params.gates.z;
                if (!(inverse > 0.0)) { continue; }
                candidate.x = 1.0 / inverse;
            } else {
                var normal = centre.yzw;
                if (axis == 1u) { normal.x += direction * params.gates.w; }
                else { normal.y += direction * params.gates.w; }
                candidate = vec4<f32>(centre.x, normalize(normal));
            }
            let value = score(x, y, candidate);
            if (value > best_score) { best = candidate; best_score = value; }
        }
    }
    planes[p] = best;
    scores[p] = best_score;
}
