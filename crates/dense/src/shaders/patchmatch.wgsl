// Per-pixel slanted planes (PatchMatch stereo) for one reference view.
//
// A plane is stored per reference pixel as (depth, nx, ny, nz): the camera depth at the pixel
// centre and the unit normal in the reference camera's frame (facing the camera). Its score is
// the matching score of `Stereo.score` with the window sampled through the plane instead of at
// one depth: every window pixel q is lifted to the plane along its own ray, projected into each
// neighbour and sampled bilinearly (the conventions of `warp.wgsl`); the normalised
// cross-correlation of the window against the reference is taken per neighbour, and the score is
// the mean of the best `best_of` neighbour correlations (at least two valid), -2 when unsupported.
//
// Entry points:
// - `evaluate`: scores the plane of every masked pixel (`planes` -> `scores`).
// - `sweep`: one half of a red-black PatchMatch iteration. Pixels of one colour take the best of
//   their own plane, four planes of opposite-colour pixels around them (per direction the
//   adjacent or the farther pixel, whichever scored better; each re-expressed at this pixel) and
//   three random perturbations of the best. Only pixels of the other colour are read, so
//   the update has no races.
// `ROW` is prepended by the dispatch layer.

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
    // Window stride; best_of; colour of this sweep; random seed.
    pass_: vec4<u32>,
    // min_variance, window_fill, inverse-depth perturbation, normal perturbation.
    gates: vec4<f32>,
    // Smallest and largest inverse depth a plane may take at its pixel; propagation distance (far ring).
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

// The plane of pixel (sx, sy) re-expressed at pixel (x, y): same normal, depth where (x, y)'s ray meets it.
fn transfer(plane: vec4<f32>, sx: i32, sy: i32, x: i32, y: i32) -> vec4<f32> {
    let normal = plane.yzw;
    let along = dot(normal, ray(f32(x), f32(y)));
    if (!(plane.x > 0.0) || !(along < -1e-6)) {
        return vec4<f32>(0.0);
    }
    let depth = dot(normal, ray(f32(sx), f32(sy)) * plane.x) / along;
    return vec4<f32>(depth, normal);
}

fn hash(value: u32) -> u32 {
    var state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn uniform01(seed: ptr<function, u32>) -> f32 {
    *seed = hash(*seed);
    return f32(*seed >> 8u) / 16777216.0;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x + id.y * ROW;
    if (p >= params.size.x * params.size.y) {
        return;
    }
    let x = i32(p % params.size.x);
    let y = i32(p / params.size.x);
    if (!in_ref_mask(x, y)) {
        scores[p] = INVALID_SCORE;
        return;
    }
    scores[p] = score(x, y, planes[p]);
}

@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) id: vec3<u32>) {
    // One invocation per pixel of this colour: half of each row, rounded up.
    let n = id.x + id.y * ROW;
    let half_width = (params.size.x + 1u) / 2u;
    let y = n / half_width;
    if (y >= params.size.y) {
        return;
    }
    let x = 2u * (n % half_width) + ((y + params.pass_.z) & 1u);
    if (x >= params.size.x) {
        return;
    }
    let ix = i32(x);
    let iy = i32(y);
    if (!in_ref_mask(ix, iy)) {
        return;
    }
    let p = y * params.size.x + x;
    var best_plane = planes[p];
    var best_score = scores[p];
    // Propagation: per direction, the adjacent pixel of the other colour or the one `far` away,
    // whichever holds the better score (four candidates).
    let far = i32(params.range.z);
    let directions = array<vec2<i32>, 4>(vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1));
    for (var c = 0u; c < 4u; c++) {
        var chosen = -1;
        var chosen_score = -1.5;
        for (var reach = 0u; reach < 2u; reach++) {
            let step = select(far, 1, reach == 0u);
            let sx = ix + directions[c].x * step;
            let sy = iy + directions[c].y * step;
            if (!in_ref_mask(sx, sy)) {
                continue;
            }
            let q = sy * i32(params.size.x) + sx;
            if (scores[q] > chosen_score) {
                chosen_score = scores[q];
                chosen = q;
            }
        }
        if (chosen < 0) {
            continue;
        }
        let sx = chosen % i32(params.size.x);
        let sy = chosen / i32(params.size.x);
        let candidate = transfer(planes[chosen], sx, sy, ix, iy);
        let s = score(ix, iy, candidate);
        if (s > best_score) {
            best_score = s;
            best_plane = candidate;
        }
    }
    // Refinement: random perturbations of depth (in inverse depth) and normal around the best.
    var seed = hash(p ^ hash(params.pass_.w));
    let r = ray(f32(ix), f32(iy));
    for (var t = 0u; t < 3u; t++) {
        if (!(best_plane.x > 0.0)) {
            break;
        }
        var inverse = 1.0 / best_plane.x;
        var normal = best_plane.yzw;
        if (t != 1u) {
            inverse += (2.0 * uniform01(&seed) - 1.0) * params.gates.z;
        }
        if (t != 0u) {
            let jitter = vec3<f32>(uniform01(&seed), uniform01(&seed), uniform01(&seed)) * 2.0 - 1.0;
            normal = normalize(normal + jitter * params.gates.w);
            if (dot(normal, r) > -0.05) {
                continue;
            }
        }
        if (!(inverse > 0.0)) {
            continue;
        }
        let candidate = vec4<f32>(1.0 / inverse, normal);
        let s = score(ix, iy, candidate);
        if (s > best_score) {
            best_score = s;
            best_plane = candidate;
        }
    }
    planes[p] = best_plane;
    scores[p] = best_score;
}
