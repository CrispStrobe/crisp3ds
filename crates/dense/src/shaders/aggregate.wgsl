// Validity-weighted Gaussian blur of each hypothesis slice of the cost volume
// (port of `Stereo._aggregate` with `_gauss`): separable, replicate padding.
// `horizontal` writes the row-blurred value and validity; `vertical` blurs
// those along columns and replaces the volume: unsupported centres stay
// unsupported, as do centres with less than 0.3 of valid weight around them.
// `ROW` is prepended by the dispatch layer.

struct Params {
    // Width, height, hypotheses, radius.
    size: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> volume: array<f32>;
@group(0) @binding(3) var<storage, read_write> numerator: array<f32>;
@group(0) @binding(4) var<storage, read_write> denominator: array<f32>;

@compute @workgroup_size(64)
fn horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x + id.y * ROW;
    let width = params.size.x;
    let pixels = width * params.size.y;
    if (n >= pixels * params.size.z) {
        return;
    }
    let x = i32((n % pixels) % width);
    let row = n - u32(x);
    let radius = i32(params.size.w);
    var num = 0.0;
    var den = 0.0;
    for (var i = -radius; i <= radius; i++) {
        let qx = u32(clamp(x + i, 0, i32(width) - 1));
        let value = volume[row + qx];
        if (value > -1.5) {
            let w = weights[u32(i + radius)];
            num += w * value;
            den += w;
        }
    }
    numerator[n] = num;
    denominator[n] = den;
}

@compute @workgroup_size(64)
fn vertical(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x + id.y * ROW;
    let width = params.size.x;
    let height = params.size.y;
    let pixels = width * height;
    if (n >= pixels * params.size.z) {
        return;
    }
    let p = n % pixels;
    let x = p % width;
    let y = i32(p / width);
    let plane = n - p;
    let radius = i32(params.size.w);
    var num = 0.0;
    var den = 0.0;
    for (var i = -radius; i <= radius; i++) {
        let q = plane + u32(clamp(y + i, 0, i32(height) - 1)) * width + x;
        let w = weights[u32(i + radius)];
        num += w * numerator[q];
        den += w * denominator[q];
    }
    if (volume[n] > -1.5 && den > 0.3) {
        volume[n] = num / max(den, 1e-6);
    } else {
        volume[n] = -2.0;
    }
}
