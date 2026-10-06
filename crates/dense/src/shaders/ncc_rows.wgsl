// First half of the jointly masked ZNCC (`ncc` in multiscale_stereo.py): the six
// window sums along rows, for one warped neighbour. The window sum of the
// reference's pooling is separable, so rows are summed here and columns in
// `ncc_columns.wgsl`. Per (hypothesis, pixel) the kernel writes the count of
// jointly valid pixels and the sums of a, b, a*a, b*b, a*b over the row
// segment, with a = reference grey - 0.5 and b = warped grey - 0.5. Pixels
// outside the image contribute nothing (zero padding).
// `ROW` is prepended by the dispatch layer.

struct Params {
    // Width, height, hypotheses in this chunk, window.
    size: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ref_gray: array<f32>;
@group(0) @binding(2) var<storage, read> warped: array<f32>;
@group(0) @binding(3) var<storage, read_write> sums: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x + id.y * ROW;
    let width = params.size.x;
    let pixels = width * params.size.y;
    if (n >= pixels * params.size.z) {
        return;
    }
    let p = n % pixels;
    let x = i32(p % width);
    let row = p - u32(x);
    let plane = n - p;
    let half = i32(params.size.w) / 2;
    var count = 0.0;
    var sa = 0.0;
    var sb = 0.0;
    var saa = 0.0;
    var sbb = 0.0;
    var sab = 0.0;
    for (var qx = max(x - half, 0); qx <= min(x + half, i32(width) - 1); qx++) {
        let q = row + u32(qx);
        let sample = warped[plane + q];
        if (sample > -1e29) {
            let a = ref_gray[q] - 0.5;
            let b = sample - 0.5;
            count += 1.0;
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
        }
    }
    let slot = n * 6u;
    sums[slot] = count;
    sums[slot + 1u] = sa;
    sums[slot + 2u] = sb;
    sums[slot + 3u] = saa;
    sums[slot + 4u] = sbb;
    sums[slot + 5u] = sab;
}
