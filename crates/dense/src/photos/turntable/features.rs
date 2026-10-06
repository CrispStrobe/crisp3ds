//! Scale-invariant features after Lowe's SIFT (the patent expired in 2020),
//! written for this crate: a Gaussian scale space of the doubled image,
//! extrema of the differences of Gaussians located to sub-pixel accuracy,
//! rejection of low contrast and of edges, orientations from gradient
//! histograms, and 4 x 4 x 8 gradient descriptors.
//!
//! The parameters follow the common choices (three layers per octave, sigma
//! 1.6, edge ratio 10, orientation peaks from 80 %). Only the part of the
//! photo inside the mask's bounding box is processed.

use crate::inputs::Plane;
use crate::photos::markers::linalg::solve;

pub const DESCRIPTOR: usize = 128;

/// Feature positions in pixels of the photo ((0, 0) is the centre of the top-left pixel) and their descriptors.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Features {
    pub points: Vec<[f64; 2]>,
    /// `DESCRIPTOR` values per feature, unit length.
    pub descriptors: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// Keep at most this many features, the strongest first.
    pub maximum: usize,
    /// Smallest difference-of-Gaussians response of a feature, for an image scaled to 0..1.
    pub contrast: f64,
    pub edge_ratio: f64,
    /// Pixels the mask is shrunk by before features are accepted.
    pub mask_margin: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { maximum: 6000, contrast: 0.01, edge_ratio: 10.0, mask_margin: 2 }
    }
}

const LAYERS: usize = 3;
const SIGMA: f64 = 1.6;

#[derive(Clone)]
struct Image {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl Image {
    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
}

fn blur(source: &Image, sigma: f64) -> Image {
    let radius = (3.5 * sigma).ceil().max(1.0) as i64;
    let kernel: Vec<f32> = (-radius..=radius).map(|d| (-(d * d) as f64 / (2.0 * sigma * sigma)).exp() as f32).collect();
    let total: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|k| k / total).collect();
    let (w, h) = (source.width, source.height);
    let mut rows = vec![0f32; w * h];
    let mut padded = vec![0f32; w + 2 * radius as usize];
    for y in 0..h {
        let line = &source.data[y * w..(y + 1) * w];
        for (k, value) in padded.iter_mut().enumerate() {
            *value = line[(k as i64 - radius).clamp(0, w as i64 - 1) as usize];
        }
        for x in 0..w {
            rows[y * w + x] = kernel.iter().zip(&padded[x..]).map(|(k, v)| k * v).sum();
        }
    }
    let mut out = vec![0f32; w * h];
    for (k, weight) in kernel.iter().enumerate() {
        for y in 0..h {
            let from = (y as i64 + k as i64 - radius).clamp(0, h as i64 - 1) as usize;
            let (target, origin) = (&mut out[y * w..(y + 1) * w], &rows[from * w..(from + 1) * w]);
            for (t, o) in target.iter_mut().zip(origin) {
                *t += weight * o;
            }
        }
    }
    Image { width: w, height: h, data: out }
}

/// Twice the size by bilinear interpolation: pixel `x` of the result is at `x / 2 - 0.25` of the source.
fn double(source: &Image) -> Image {
    let (w, h) = (source.width, source.height);
    let mut out = Image { width: 2 * w, height: 2 * h, data: vec![0.0; 4 * w * h] };
    for y in 0..2 * h {
        let sy = (y as f32 / 2.0 - 0.25).clamp(0.0, h as f32 - 1.0);
        let (y0, ty) = (sy.floor() as usize, sy - sy.floor());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..2 * w {
            let sx = (x as f32 / 2.0 - 0.25).clamp(0.0, w as f32 - 1.0);
            let (x0, tx) = (sx.floor() as usize, sx - sx.floor());
            let x1 = (x0 + 1).min(w - 1);
            out.data[y * 2 * w + x] = (1.0 - ty) * ((1.0 - tx) * source.at(x0, y0) + tx * source.at(x1, y0))
                + ty * ((1.0 - tx) * source.at(x0, y1) + tx * source.at(x1, y1));
        }
    }
    out
}

fn halve(source: &Image) -> Image {
    let (w, h) = (source.width / 2, source.height / 2);
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            data.push(source.at(2 * x, 2 * y));
        }
    }
    Image { width: w, height: h, data }
}

struct Candidate {
    /// Position in the doubled image, scale in the octave, orientation in radians.
    x: f64,
    y: f64,
    octave: usize,
    layer: usize,
    scale: f64,
    response: f64,
}

/// Refines an extremum of the difference of Gaussians by a quadratic fit; `None` when it is weak, on an edge or drifts away.
fn refine(dog: &[Image], octave: usize, mut layer: usize, mut x: usize, mut y: usize, settings: &Settings) -> Option<Candidate> {
    let (w, h) = (dog[0].width, dog[0].height);
    let mut offset = [0.0f64; 3];
    let mut gradient = [0.0f64; 3];
    let mut converged = false;
    for _ in 0..5 {
        let d = |l: usize, xx: usize, yy: usize| dog[l].at(xx, yy) as f64;
        gradient = [
            (d(layer, x + 1, y) - d(layer, x - 1, y)) * 0.5,
            (d(layer, x, y + 1) - d(layer, x, y - 1)) * 0.5,
            (d(layer + 1, x, y) - d(layer - 1, x, y)) * 0.5,
        ];
        let centre = d(layer, x, y);
        let dxx = d(layer, x + 1, y) + d(layer, x - 1, y) - 2.0 * centre;
        let dyy = d(layer, x, y + 1) + d(layer, x, y - 1) - 2.0 * centre;
        let dss = d(layer + 1, x, y) + d(layer - 1, x, y) - 2.0 * centre;
        let dxy = (d(layer, x + 1, y + 1) - d(layer, x - 1, y + 1) - d(layer, x + 1, y - 1) + d(layer, x - 1, y - 1)) * 0.25;
        let dxs = (d(layer + 1, x + 1, y) - d(layer + 1, x - 1, y) - d(layer - 1, x + 1, y) + d(layer - 1, x - 1, y)) * 0.25;
        let dys = (d(layer + 1, x, y + 1) - d(layer + 1, x, y - 1) - d(layer - 1, x, y + 1) + d(layer - 1, x, y - 1)) * 0.25;
        let hessian = vec![vec![dxx, dxy, dxs], vec![dxy, dyy, dys], vec![dxs, dys, dss]];
        let step = solve(hessian, gradient.iter().map(|g| -g).collect())?;
        offset = [step[0], step[1], step[2]];
        if offset.iter().all(|o| o.abs() < 0.5) {
            // Edge response: ratio of the principal curvatures in the image plane.
            let (trace, determinant) = (dxx + dyy, dxx * dyy - dxy * dxy);
            let r = settings.edge_ratio;
            if determinant <= 0.0 || trace * trace * r >= (r + 1.0) * (r + 1.0) * determinant {
                return None;
            }
            converged = true;
            break;
        }
        if offset.iter().any(|o| !o.is_finite() || o.abs() > 1e6) {
            return None;
        }
        let (nx, ny, nl) =
            (x as i64 + offset[0].round() as i64, y as i64 + offset[1].round() as i64, layer as i64 + offset[2].round() as i64);
        if nl < 1 || nl > LAYERS as i64 || nx < 5 || ny < 5 || nx >= w as i64 - 5 || ny >= h as i64 - 5 {
            return None;
        }
        (x, y, layer) = (nx as usize, ny as usize, nl as usize);
    }
    if !converged {
        return None;
    }
    let response = dog[layer].at(x, y) as f64 + 0.5 * (gradient[0] * offset[0] + gradient[1] * offset[1] + gradient[2] * offset[2]);
    if response.abs() * (LAYERS as f64) < settings.contrast {
        return None;
    }
    let factor = (1usize << octave) as f64;
    Some(Candidate {
        x: (x as f64 + offset[0]) * factor,
        y: (y as f64 + offset[1]) * factor,
        octave,
        layer,
        scale: SIGMA * 2f64.powf((layer as f64 + offset[2]) / LAYERS as f64),
        response: response.abs(),
    })
}

/// Dominant gradient directions around a point of a Gaussian image (radians), up to a few per point.
fn orientations(image: &Image, x: f64, y: f64, scale: f64) -> Vec<f64> {
    const BINS: usize = 36;
    let radius = (3.0 * 1.5 * scale).round() as i64;
    let (cx, cy) = (x.round() as i64, y.round() as i64);
    let weight_scale = -1.0 / (2.0 * (1.5 * scale).powi(2));
    let mut histogram = [0f64; BINS];
    for dy in -radius..=radius {
        let yy = cy + dy;
        if yy <= 0 || yy >= image.height as i64 - 1 {
            continue;
        }
        for dx in -radius..=radius {
            let xx = cx + dx;
            if xx <= 0 || xx >= image.width as i64 - 1 {
                continue;
            }
            let (xx, yy) = (xx as usize, yy as usize);
            let gx = (image.at(xx + 1, yy) - image.at(xx - 1, yy)) as f64;
            let gy = (image.at(xx, yy + 1) - image.at(xx, yy - 1)) as f64;
            let bin =
                (gy.atan2(gx).rem_euclid(2.0 * std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * BINS as f64).round() as usize % BINS;
            histogram[bin] += gx.hypot(gy) * ((dx * dx + dy * dy) as f64 * weight_scale).exp();
        }
    }
    let raw = histogram;
    for (bin, value) in histogram.iter_mut().enumerate() {
        let at = |offset: i64| raw[(bin as i64 + offset).rem_euclid(BINS as i64) as usize];
        *value = (at(-2) + at(2)) / 16.0 + (at(-1) + at(1)) * 4.0 / 16.0 + at(0) * 6.0 / 16.0;
    }
    let highest = histogram.iter().cloned().fold(0.0, f64::max);
    let mut out = Vec::new();
    for bin in 0..BINS {
        let (left, right) = (histogram[(bin + BINS - 1) % BINS], histogram[(bin + 1) % BINS]);
        let value = histogram[bin];
        if value > left && value > right && value >= 0.8 * highest && value > 0.0 {
            let shift = 0.5 * (left - right) / (left - 2.0 * value + right);
            out.push((bin as f64 + shift).rem_euclid(BINS as f64) * 2.0 * std::f64::consts::PI / BINS as f64);
        }
    }
    out
}

/// The 4 x 4 x 8 gradient histogram around a point, in the frame turned by `angle`.
fn describe(image: &Image, x: f64, y: f64, scale: f64, angle: f64) -> [f32; DESCRIPTOR] {
    const D: usize = 4;
    const N: usize = 8;
    let width = 3.0 * scale;
    let radius = ((width * std::f64::consts::SQRT_2 * (D as f64 + 1.0) * 0.5).round() as i64)
        .min(((image.width * image.width + image.height * image.height) as f64).sqrt() as i64);
    let (cx, cy) = (x.round() as i64, y.round() as i64);
    let (sine, cosine) = angle.sin_cos();
    let mut histogram = vec![0f64; (D + 2) * (D + 2) * (N + 2)];
    for dy in -radius..=radius {
        let yy = cy + dy;
        if yy <= 0 || yy >= image.height as i64 - 1 {
            continue;
        }
        for dx in -radius..=radius {
            let xx = cx + dx;
            if xx <= 0 || xx >= image.width as i64 - 1 {
                continue;
            }
            // Offsets in the turned frame, in units of one histogram cell.
            let (u, v) = ((dx as f64 * cosine + dy as f64 * sine) / width, (-(dx as f64) * sine + dy as f64 * cosine) / width);
            let (column, row) = (u + D as f64 / 2.0 - 0.5, v + D as f64 / 2.0 - 0.5);
            if column <= -1.0 || column >= D as f64 || row <= -1.0 || row >= D as f64 {
                continue;
            }
            let (xx, yy) = (xx as usize, yy as usize);
            let gx = (image.at(xx + 1, yy) - image.at(xx - 1, yy)) as f64;
            let gy = (image.at(xx, yy + 1) - image.at(xx, yy - 1)) as f64;
            let magnitude = gx.hypot(gy) * (-(u * u + v * v) / (0.5 * (D * D) as f64)).exp();
            let direction = (gy.atan2(gx) - angle).rem_euclid(2.0 * std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * N as f64;
            let (r0, c0, o0) = (row.floor(), column.floor(), direction.floor());
            let (tr, tc, to) = (row - r0, column - c0, direction - o0);
            for (ri, wr) in [(0usize, 1.0 - tr), (1, tr)] {
                for (ci, wc) in [(0usize, 1.0 - tc), (1, tc)] {
                    for (oi, wo) in [(0usize, 1.0 - to), (1, to)] {
                        let index = ((r0 as i64 + 1) as usize + ri) * (D + 2) * (N + 2)
                            + ((c0 as i64 + 1) as usize + ci) * (N + 2)
                            + (o0 as usize + oi) % N;
                        histogram[index] += magnitude * wr * wc * wo;
                    }
                }
            }
        }
    }
    let mut out = [0f32; DESCRIPTOR];
    for row in 0..D {
        for column in 0..D {
            for bin in 0..N {
                out[(row * D + column) * N + bin] = histogram[(row + 1) * (D + 2) * (N + 2) + (column + 1) * (N + 2) + bin] as f32;
            }
        }
    }
    let normalise = |values: &mut [f32; DESCRIPTOR]| {
        let length = values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        values.iter_mut().for_each(|v| *v /= length);
    };
    normalise(&mut out);
    out.iter_mut().for_each(|v| *v = v.min(0.2));
    normalise(&mut out);
    out
}

/// Features of a grey photo, inside `mask` (non-zero) when one is given.
pub fn detect(gray: &Plane<u8>, mask: Option<&Plane<u8>>, settings: &Settings) -> Features {
    let (w, h) = (gray.width, gray.height);
    // The region to process: the mask's bounding box with room for the descriptors.
    let (mut x0, mut y0, mut x1, mut y1) = (0usize, 0usize, w, h);
    if let Some(mask) = mask {
        let (mut left, mut top, mut right, mut bottom) = (w, h, 0usize, 0usize);
        for y in 0..h.min(mask.height) {
            for x in 0..w.min(mask.width) {
                if mask.data[y * mask.width + x] > 127 {
                    (left, top, right, bottom) = (left.min(x), top.min(y), right.max(x + 1), bottom.max(y + 1));
                }
            }
        }
        if right <= left {
            return Features::default();
        }
        const ROOM: usize = 24;
        (x0, y0, x1, y1) = (left.saturating_sub(ROOM), top.saturating_sub(ROOM), (right + ROOM).min(w), (bottom + ROOM).min(h));
    }
    let inside = |x: f64, y: f64| -> bool {
        let Some(mask) = mask else { return true };
        let (cx, cy, m) = (x.round() as i64, y.round() as i64, settings.mask_margin as i64);
        // All of the square of side 2 margin + 1 around the point belongs to the object.
        (-m..=m).all(|dy| {
            (-m..=m).all(|dx| {
                let (u, v) = (cx + dx, cy + dy);
                u >= 0
                    && v >= 0
                    && (u as usize) < mask.width
                    && (v as usize) < mask.height
                    && mask.data[v as usize * mask.width + u as usize] > 127
            })
        })
    };
    let (cw, ch) = (x1 - x0, y1 - y0);
    if cw < 16 || ch < 16 {
        return Features::default();
    }
    let crop = Image {
        width: cw,
        height: ch,
        data: (0..ch).flat_map(|y| gray.data[(y0 + y) * w + x0..(y0 + y) * w + x1].iter().map(|&v| v as f32 / 255.0)).collect(),
    };
    // The doubled image has a blur of 1 (twice the 0.5 assumed of a photo); bring it to sigma.
    let mut base = blur(&double(&crop), (SIGMA * SIGMA - 1.0).sqrt());
    let octaves = (((cw.min(ch) * 2) as f64).log2() as usize).saturating_sub(3).max(1);
    let k = 2f64.powf(1.0 / LAYERS as f64);
    let mut candidates: Vec<(Candidate, f64, [f32; DESCRIPTOR])> = Vec::new();
    for octave in 0..octaves {
        let mut gaussians = vec![base];
        for layer in 1..LAYERS + 3 {
            let previous = SIGMA * k.powi(layer as i32 - 1);
            let sigma = ((previous * k).powi(2) - previous * previous).sqrt();
            let next = blur(&gaussians[layer - 1], sigma);
            gaussians.push(next);
        }
        let dog: Vec<Image> = gaussians
            .windows(2)
            .map(|pair| Image {
                width: pair[0].width,
                height: pair[0].height,
                data: pair[1].data.iter().zip(&pair[0].data).map(|(a, b)| a - b).collect(),
            })
            .collect();
        let (ow, oh) = (dog[0].width, dog[0].height);
        let threshold = (0.5 * settings.contrast / LAYERS as f64) as f32;
        if ow > 12 && oh > 12 {
            for layer in 1..=LAYERS {
                for y in 5..oh - 5 {
                    for x in 5..ow - 5 {
                        let value = dog[layer].at(x, y);
                        if value.abs() <= threshold {
                            continue;
                        }
                        let mut extreme = true;
                        'around: for l in layer - 1..=layer + 1 {
                            for yy in y - 1..=y + 1 {
                                for xx in x - 1..=x + 1 {
                                    let other = dog[l].at(xx, yy);
                                    if (value > 0.0 && other > value) || (value < 0.0 && other < value) {
                                        extreme = false;
                                        break 'around;
                                    }
                                }
                            }
                        }
                        if !extreme {
                            continue;
                        }
                        let Some(candidate) = refine(&dog, octave, layer, x, y, settings) else { continue };
                        // Back to the photo: the doubled crop's pixel p is at p / 2 - 0.25 of the crop.
                        let (px, py) = (candidate.x / 2.0 - 0.25 + x0 as f64, candidate.y / 2.0 - 0.25 + y0 as f64);
                        if !inside(px, py) {
                            continue;
                        }
                        let factor = (1usize << octave) as f64;
                        let (lx, ly) = (candidate.x / factor, candidate.y / factor);
                        let image = &gaussians[candidate.layer];
                        for angle in orientations(image, lx, ly, candidate.scale) {
                            let descriptor = describe(image, lx, ly, candidate.scale, angle);
                            candidates.push((Candidate { x: px, y: py, ..candidate_copy(&candidate) }, angle, descriptor));
                        }
                    }
                }
            }
        }
        base = halve(&gaussians[LAYERS]);
        if base.width < 16 || base.height < 16 {
            break;
        }
    }
    candidates.sort_by(|a, b| b.0.response.total_cmp(&a.0.response));
    candidates.truncate(settings.maximum);
    let mut features = Features::default();
    for (candidate, _, descriptor) in candidates {
        features.points.push([candidate.x, candidate.y]);
        features.descriptors.extend(descriptor);
    }
    features
}

fn candidate_copy(c: &Candidate) -> Candidate {
    Candidate { x: c.x, y: c.y, octave: c.octave, layer: c.layer, scale: c.scale, response: c.response }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photos::turntable::matching::match_pair;

    /// A smooth random texture, optionally shifted and turned about the image centre.
    fn texture(width: usize, height: usize, shift: [f64; 2], turn: f64) -> Plane<u8> {
        let (s, c) = turn.sin_cos();
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let (dx, dy) = (x as f64 - width as f64 / 2.0, y as f64 - height as f64 / 2.0);
                let (u, v) = (c * dx - s * dy + shift[0], s * dx + c * dy + shift[1]);
                let mut value = 0.0;
                for k in 1..40 {
                    let (a, b, p) = ((k as f64 * 12.9898).sin() * 0.35, (k as f64 * 78.233).cos() * 0.35, (k as f64 * 3.7).sin() * 6.0);
                    value += (a * u + b * v + p).sin() / (1.0 + 0.1 * k as f64);
                }
                data.push((128.0 + 22.0 * value).clamp(0.0, 255.0) as u8);
            }
        }
        Plane { width, height, data }
    }

    #[test]
    fn features_follow_a_shift_and_a_turn() {
        let settings = Settings::default();
        let first = detect(&texture(200, 160, [0.0, 0.0], 0.0), None, &settings);
        assert!(first.points.len() > 80, "{} features", first.points.len());
        assert_eq!(first.descriptors.len(), first.points.len() * DESCRIPTOR);
        for (shift, turn) in [([7.5, -4.25], 0.0), ([0.0, 0.0], 0.35)] {
            let second = detect(&texture(200, 160, shift, turn), None, &settings);
            let matches = match_pair(&first, &second, 0.8);
            assert!(matches.len() > 40, "{} matches", matches.len());
            // A point p of the first image appears at R^-1 (p - centre - shift) + centre in the second.
            let (s, c) = turn.sin_cos();
            let mut errors: Vec<f64> = matches
                .iter()
                .map(|&(a, b)| {
                    let (p, q) = (first.points[a as usize], second.points[b as usize]);
                    let (u, v) = (p[0] - 100.0 - shift[0], p[1] - 80.0 - shift[1]);
                    let expected = [c * u + s * v + 100.0, -s * u + c * v + 80.0];
                    (expected[0] - q[0]).hypot(expected[1] - q[1])
                })
                .collect();
            errors.sort_by(f64::total_cmp);
            let good = errors.iter().filter(|e| **e < 1.0).count();
            assert!(
                good * 10 >= errors.len() * 9 && errors[errors.len() / 2] < 0.3,
                "median {} px, {good} of {} within a pixel",
                errors[errors.len() / 2],
                errors.len()
            );
        }
    }

    #[test]
    fn features_stay_inside_the_mask() {
        let gray = texture(200, 160, [0.0, 0.0], 0.0);
        let mut mask = Plane::<u8>::new(200, 160);
        for y in 40..120 {
            for x in 60..150 {
                mask.data[y * 200 + x] = 255;
            }
        }
        let found = detect(&gray, Some(&mask), &Settings::default());
        assert!(found.points.len() > 15);
        assert!(found.points.iter().all(|p| p[0] >= 61.5 && p[0] <= 147.5 && p[1] >= 41.5 && p[1] <= 117.5));
        assert!(detect(&gray, Some(&Plane::<u8>::new(200, 160)), &Settings::default()).points.is_empty());
        assert_eq!(detect(&gray, None, &Settings { maximum: 10, ..Settings::default() }).points.len(), 10);
    }
}
