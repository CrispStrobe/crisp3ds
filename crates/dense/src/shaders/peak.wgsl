// Best hypothesis per pixel with a parabolic sub-step offset, and the depth it
// stands for (port of `Stereo._peak` and the tails of `sweep` and `refine`).
//
// depth = 1 / (base + (index - shift + offset) * step), where base is a
// constant inverse depth for a sweep and 1 / init for a refinement. A pixel
// keeps its depth when the best score reaches `min_score`, the peak is not at
// either end of the search, both neighbours of the peak are supported, and
// (refinement) the initial depth exists. Otherwise the depth is 0.
// `ROW` is prepended by the dispatch layer.

struct Params {
    // Width, height, hypotheses, mode (0 sweep, 1 refine).
    size: vec4<u32>,
    // Step of inverse depth, shift (half-width of a refinement, 0 for a sweep), min_score, sweep base.
    values: vec4<f32>,
    // Minimum margin over a distinct competing mode; zero preserves legacy selection.
    confidence: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<storage, read> init: array<f32>;
@group(0) @binding(3) var<storage, read_write> depth: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x + id.y * ROW;
    let pixels = params.size.x * params.size.y;
    if (p >= pixels) {
        return;
    }
    let count = params.size.z;
    var best = volume[p];
    var index = 0u;
    for (var h = 1u; h < count; h++) {
        let value = volume[h * pixels + p];
        if (value > best) {
            best = value;
            index = h;
        }
    }
    var result = 0.0;
    if (index > 0u && index < count - 1u) {
        let left = volume[(index - 1u) * pixels + p];
        let right = volume[(index + 1u) * pixels + p];
        var usable = left > -1.5 && right > -1.5 && best >= params.values.z;
        var base = params.values.w;
        if (params.size.w == 1u) {
            usable = usable && init[p] > 0.0;
            base = 1.0 / max(init[p], 1e-6);
        }
        if (usable && params.confidence.x > 0.0) {
            var competitor = -2.0;
            for (var h = 0u; h < count; h++) {
                if (abs(i32(h) - i32(index)) > 2) {
                    let score = volume[h * pixels + p];
                    let before = volume[u32(max(i32(h) - 1, 0)) * pixels + p];
                    let after = volume[min(h + 1u, count - 1u) * pixels + p];
                    if (score > -1.5 && score >= before && score >= after) {
                        var valley = score;
                        for (var k = min(h, index); k <= max(h, index); k++) {
                            valley = min(valley, volume[k * pixels + p]);
                        }
                        // Broad unimodal shoulders are not separate depth explanations.
                        if (valley < min(score, best) - 0.02) {
                            competitor = max(competitor, score);
                        }
                    }
                }
            }
            usable = best - competitor >= params.confidence.x;
        }
        if (usable) {
            let curve = left - 2.0 * best + right;
            var offset = 0.0;
            if (curve < -1e-6) {
                offset = clamp(0.5 * (left - right) / curve, -0.5, 0.5);
            }
            result = 1.0 / max(base + (f32(index) - params.values.y + offset) * params.values.x, 1e-6);
        }
    }
    depth[p] = result;
}
