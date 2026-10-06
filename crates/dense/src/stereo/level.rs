//! Pyramid levels: port of `Stereo.build_level`, including Pillow's resampling.
//!
//! The reference crops each view to the common canvas (zero outside the photo)
//! and resizes grey with `Image.Resampling.BILINEAR` and the mask with `BOX`
//! on 32-bit float images. [`resize`] follows Pillow's `Resample.c` for that
//! mode: per output pixel a normalised filter of support `support * scale`,
//! accumulated in double precision and stored as float after each of the two
//! passes (rows first, then columns).

#![allow(clippy::needless_range_loop)]

use crate::inputs::{parallel_map, round_half_even, Camera, Inputs, Plane};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Filter {
    Box,
    Bilinear,
    Bicubic,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Box => 0.5,
            Filter::Bilinear => 1.0,
            Filter::Bicubic => 2.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Filter::Box => (x > -0.5 && x <= 0.5) as u8 as f64,
            Filter::Bilinear => {
                let x = x.abs();
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Filter::Bicubic => {
                const A: f64 = -0.5;
                let x = x.abs();
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
        }
    }
}

/// Pillow's `precompute_coeffs`: first source index and normalised weights per output index.
fn coefficients(in_size: usize, out_size: usize, filter: Filter) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = filter.support() * filter_scale;
    let inverse = 1.0 / filter_scale;
    (0..out_size)
        .map(|n| {
            let centre = (n as f64 + 0.5) * scale;
            let first = ((centre - support + 0.5) as i64).max(0) as usize;
            let last = ((centre + support + 0.5) as i64).min(in_size as i64) as usize;
            let mut weights: Vec<f64> = (first..last).map(|x| filter.weight((x as f64 - centre + 0.5) * inverse)).collect();
            let sum: f64 = weights.iter().sum();
            if sum != 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
            (first, weights)
        })
        .collect()
}

/// `Image.resize((width, height), filter)` for a mode "F" image.
pub fn resize(source: &Plane<f32>, width: usize, height: usize, filter: Filter) -> Plane<f32> {
    if (source.width, source.height) == (width, height) {
        return source.clone();
    }
    let mut rows = Plane::<f32>::new(width, source.height);
    if source.width == width {
        rows.data.copy_from_slice(&source.data);
    } else {
        let horizontal = coefficients(source.width, width, filter);
        for y in 0..source.height {
            let line = &source.data[y * source.width..(y + 1) * source.width];
            for (x, (first, weights)) in horizontal.iter().enumerate() {
                let mut sum = 0.0f64;
                for (n, w) in weights.iter().enumerate() {
                    sum += line[first + n] as f64 * w;
                }
                rows.data[y * width + x] = sum as f32;
            }
        }
    }
    if source.height == height {
        return rows;
    }
    let vertical = coefficients(source.height, height, filter);
    let mut out = Plane::<f32>::new(width, height);
    for (y, (first, weights)) in vertical.iter().enumerate() {
        for x in 0..width {
            let mut sum = 0.0f64;
            for (n, w) in weights.iter().enumerate() {
                sum += rows.data[(first + n) * width + x] as f64 * w;
            }
            out.data[y * width + x] = sum as f32;
        }
    }
    out
}

/// `Image.crop(box)`: the box may leave the image; outside is zero.
pub fn crop<T: Copy + Default>(source: &Plane<T>, bounds: [i64; 4]) -> Plane<T> {
    let (width, height) = ((bounds[2] - bounds[0]) as usize, (bounds[3] - bounds[1]) as usize);
    let mut out = Plane::<T>::new(width, height);
    for y in 0..height {
        let sy = y as i64 + bounds[1];
        if sy < 0 || sy >= source.height as i64 {
            continue;
        }
        for x in 0..width {
            let sx = x as i64 + bounds[0];
            if sx >= 0 && sx < source.width as i64 {
                out.data[y * width + x] = source.data[sy as usize * source.width + sx as usize];
            }
        }
    }
    out
}

/// One view at one pyramid level: the common canvas scaled to the level size.
#[derive(Debug, Clone)]
pub struct LevelView {
    pub width: usize,
    pub height: usize,
    /// Rotation and translation of the view; intrinsics of the scaled crop.
    pub camera: Camera,
    pub gray: Plane<f32>,
    pub mask: Plane<u8>,
}

impl LevelView {
    /// `Stereo.project`: pixel coordinates and camera depth of a world point, float32.
    #[inline]
    pub fn project(&self, world: [f32; 3]) -> ([f32; 2], f32) {
        let p = self.camera.to_camera(world);
        let z = p[2].max(1e-6);
        let k = &self.camera.k;
        ([p[0] / z * k[0] + k[2] - 0.5, p[1] / z * k[1] + k[3] - 0.5], p[2])
    }

    /// World point of pixel (x, y) at camera depth `depth`: `(rays * depth - t) @ R`.
    #[inline]
    pub fn unproject(&self, x: usize, y: usize, depth: f32) -> [f32; 3] {
        let (k, r, t) = (&self.camera.k, &self.camera.rotation, &self.camera.translation);
        let ray = [(x as f32 + 0.5 - k[2]) / k[0], (y as f32 + 0.5 - k[3]) / k[1], 1.0];
        let c = [ray[0] * depth - t[0], ray[1] * depth - t[1], ray[2] * depth - t[2]];
        [
            c[0] * r[0][0] + c[1] * r[1][0] + c[2] * r[2][0],
            c[0] * r[0][1] + c[1] * r[1][1] + c[2] * r[2][1],
            c[0] * r[0][2] + c[1] * r[1][2] + c[2] * r[2][2],
        ]
    }
}

/// `Stereo.build_level(size)`.
pub fn build_level(inputs: &Inputs, size: i64) -> Vec<LevelView> {
    let factor = (size as f64 / inputs.longest as f64).min(1.0);
    parallel_map(inputs.count(), |n| {
        let bounds = inputs.boxes[n];
        let (bw, bh) = (bounds[2] - bounds[0], bounds[3] - bounds[1]);
        let width = (round_half_even(bw as f64 * factor) as i64).max(8) as usize;
        let height = (round_half_even(bh as f64 * factor) as i64).max(8) as usize;
        let gray = resize(&crop(&inputs.gray[n], bounds), width, height, Filter::Bilinear);
        let mask_crop = crop(&inputs.masks[n], bounds);
        let mask_float =
            Plane { width: mask_crop.width, height: mask_crop.height, data: mask_crop.data.iter().map(|&m| m as f32).collect() };
        let box_mask = resize(&mask_float, width, height, Filter::Box);
        let mask = Plane { width, height, data: box_mask.data.iter().map(|&m| (m >= 0.5) as u8).collect() };
        let (sx, sy) = ((width as f64 / bw as f64) as f32, (height as f64 / bh as f64) as f32);
        let k = inputs.cameras[n].k;
        let camera = Camera {
            rotation: inputs.cameras[n].rotation,
            translation: inputs.cameras[n].translation,
            k: [k[0] * sx, k[1] * sy, (k[2] - bounds[0] as f32) * sx, (k[3] - bounds[1] as f32) * sy],
        };
        LevelView { width, height, camera, gray, mask }
    })
}

/// Pyramid sizes capped at the canvas; a cap may merge levels.
pub fn level_sizes(sizes: &[i64], longest: i64) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::new();
    for size in sizes.iter().map(|&s| s.min(longest)) {
        if !out.contains(&size) {
            out.push(size);
        }
    }
    out
}

/// Fraction of mask pixels with depth, per view: median, minimum, maximum (`coverage`).
pub fn coverage(level: &[LevelView], depths: &[Plane<f32>]) -> serde_json::Value {
    let mut values: Vec<f64> = level
        .iter()
        .zip(depths)
        .map(|(view, depth)| {
            let both = view.mask.data.iter().zip(&depth.data).filter(|(&m, &d)| m != 0 && d > 0.0).count();
            let mask = view.mask.data.iter().filter(|&&m| m != 0).count();
            (both as f32 / mask as f32) as f64
        })
        .collect();
    let (minimum, maximum) = (values.iter().cloned().fold(f64::MAX, f64::min), values.iter().cloned().fold(f64::MIN, f64::max));
    serde_json::json!({"median": crate::inputs::median_f64(&mut values), "minimum": minimum, "maximum": maximum})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values are Pillow 11.2 outputs for mode "F" images, compared bit for bit.
    #[test]
    fn reductions_and_enlargements_follow_pillow() {
        let row = |values: &[f32]| Plane { width: values.len(), height: 1, data: values.to_vec() };
        let ramp = row(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(resize(&ramp, 4, 1, Filter::Box).data, [0.5, 2.5, 4.5, 6.5]);
        assert_eq!(resize(&ramp, 4, 1, Filter::Bilinear).data, [0.714_285_73, 2.5, 4.5, 6.285_714]);
        assert_eq!(resize(&ramp, 8, 1, Filter::Bilinear), ramp);
        let five = row(&[1.0, 2.0, 4.0, 8.0, 16.0]);
        assert_eq!(resize(&five, 2, 1, Filter::Box).data, [2.333_333_3, 12.0]);
        assert_eq!(resize(&five, 2, 1, Filter::Bilinear).data, [2.409_091, 9.363_636]);
        assert_eq!(resize(&five, 2, 1, Filter::Bicubic).data, [1.868_695_5, 9.863_162]);
        assert_eq!(
            resize(&five, 7, 1, Filter::Bicubic).data,
            [0.941_747_55, 1.426_592_8, 2.440_233_2, 4.0, 6.507_288_5, 11.789_474, 16.466_019]
        );
        // Two passes: ((arange(35).reshape(5, 7) ** 2) % 11) resized to 3 x 2.
        let grid = Plane { width: 7, height: 5, data: (0..35).map(|n| ((n * n) % 11) as f32).collect() };
        let expected = [4.457_792_3, 3.812_834_3, 3.529_220_8, 3.529_220_8, 4.106_951_7, 3.814_935];
        assert_eq!(resize(&grid, 3, 2, Filter::Bilinear).data, expected);
    }

    #[test]
    fn crop_pads_with_zero_and_sizes_merge() {
        let source = Plane { width: 2, height: 2, data: vec![1u8, 2, 3, 4] };
        let out = crop(&source, [-1, 1, 2, 3]);
        assert_eq!((out.width, out.height), (3, 2));
        assert_eq!(out.data, vec![0, 3, 4, 0, 0, 0]);
        assert_eq!(level_sizes(&[256, 512, 1024], 876), vec![256, 512, 876]);
        assert_eq!(level_sizes(&[64, 128], 60), vec![60]);
    }
}
