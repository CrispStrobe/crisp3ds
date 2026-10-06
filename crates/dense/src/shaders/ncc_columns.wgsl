// Second half of the jointly masked ZNCC and the running best-N over
// neighbours (the scoring half of `Stereo.score`, with `ncc`).
//
// One invocation handles one (hypothesis, pixel): the row sums of
// `ncc_rows.wgsl` are added along the column, divided by window^2 as
// `avg_pool2d` with zero padding does, and turned into a score. The score is
// -2 where unsupported: centre invalid, window fill below `window_fill`, or
// either variance below `min_variance`. Scores are inserted into the per-pixel
// best-N list; after the last neighbour the mean of the valid entries is
// written to the cost volume (-2 with fewer than two).
// Sums are accumulated row by row in float32, which is not the order of the
// reference's pooling; scores differ in the last bits.
// `ROW` is prepended by the dispatch layer.

struct Params {
    // Width, height, hypotheses in this chunk, window.
    size: vec4<u32>,
    // 1 for the first neighbour, 1 for the last, best_of, first hypothesis of the chunk in the volume.
    flags: vec4<u32>,
    // min_variance, window_fill.
    gates: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> warped: array<f32>;
@group(0) @binding(2) var<storage, read> sums: array<f32>;
@group(0) @binding(3) var<storage, read_write> best: array<f32>;
@group(0) @binding(4) var<storage, read_write> volume: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x + id.y * ROW;
    let width = params.size.x;
    let height = params.size.y;
    let pixels = width * height;
    let hypothesis = n / pixels;
    if (hypothesis >= params.size.z) {
        return;
    }
    let p = n % pixels;
    let x = p % width;
    let y = i32(p / width);
    let plane = n - p;
    let window = i32(params.size.w);
    let half = window / 2;
    var score = -2.0;
    if (warped[n] > -1e29) {
        var count = 0.0;
        var sr = 0.0;
        var st = 0.0;
        var srr = 0.0;
        var stt = 0.0;
        var srt = 0.0;
        for (var qy = max(y - half, 0); qy <= min(y + half, i32(height) - 1); qy++) {
            let slot = (plane + u32(qy) * width + x) * 6u;
            count += sums[slot];
            sr += sums[slot + 1u];
            st += sums[slot + 2u];
            srr += sums[slot + 3u];
            stt += sums[slot + 4u];
            srt += sums[slot + 5u];
        }
        let scale = 1.0 / f32(window * window);
        let fill = count * scale;
        let d = max(fill, 1e-6);
        let mr = sr * scale / d;
        let mt = st * scale / d;
        let vr = max(srr * scale / d - mr * mr, 0.0);
        let vt = max(stt * scale / d - mt * mt, 0.0);
        if (fill >= params.gates.y && vr >= params.gates.x && vt >= params.gates.x) {
            score = clamp((srt * scale / d - mr * mt) / sqrt(vr * vt + 1e-12), -1.0, 1.0);
        }
    }
    // Best-N by min/max insertion, as the reference does.
    let ranks = params.flags.z;
    var total = 0.0;
    var valid = 0.0;
    var carry = score;
    for (var rank = 0u; rank < ranks; rank++) {
        let slot = (rank * params.size.z + hypothesis) * pixels + p;
        var current = -2.0;
        if (params.flags.x == 0u) {
            current = best[slot];
        }
        let lower = min(current, carry);
        current = max(current, carry);
        carry = lower;
        best[slot] = current;
        if (current > -1.5) {
            total += current;
            valid += 1.0;
        }
    }
    if (params.flags.y == 1u) {
        var mean = -2.0;
        if (valid >= 2.0) {
            mean = total / valid;
        }
        volume[(params.flags.w + hypothesis) * pixels + p] = mean;
    }
}
