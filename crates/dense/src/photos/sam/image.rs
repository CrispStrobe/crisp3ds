//! What happens to pixels around the network, done as PyTorch does it so
//! that the masks agree with the reference route: the antialiased bilinear
//! resize of the photo to the model's square input (`torchvision`'s `Resize`
//! on a float tensor) with normalisation, and the plain bilinear enlargement
//! of the low-resolution mask logits to the photo (`F.interpolate`,
//! `align_corners=False`).

use crate::inputs::Plane;

/// Weights of one output sample of the antialiased resize: first input index and the normalised triangle weights.
fn antialias_weights(input: usize, output: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = input as f32 / output as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let inverse = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    (0..output)
        .map(|i| {
            let center = scale * (i as f32 + 0.5);
            let low = ((center - support + 0.5) as i64).max(0) as usize;
            let high = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let mut weights: Vec<f32> = (low..high)
                .map(|j| {
                    let x = ((j as f32 - center + 0.5) * inverse).abs();
                    if x < 1.0 {
                        1.0 - x
                    } else {
                        0.0
                    }
                })
                .collect();
            let total: f32 = weights.iter().sum();
            if total != 0.0 {
                weights.iter_mut().for_each(|w| *w /= total);
            }
            (low, weights)
        })
        .collect()
}

/// The network input from 8-bit RGB (`width * height * 3`, row by row): values
/// over 255, resized to `size` x `size` with an antialiased triangle filter
/// (columns first, then rows), then `(v - mean) / std`. Planar output: all of
/// red, then green, then blue.
pub fn model_input(rgb: &[u8], width: usize, height: usize, size: usize, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let across = antialias_weights(width, size);
    let down = antialias_weights(height, size);
    let mut out = vec![0f32; 3 * size * size];
    let mut rows = vec![0f32; height * size];
    for channel in 0..3 {
        for y in 0..height {
            let line = &rgb[y * width * 3..(y + 1) * width * 3];
            for (x, (low, weights)) in across.iter().enumerate() {
                let mut sum = 0f32;
                for (j, weight) in weights.iter().enumerate() {
                    sum += line[(low + j) * 3 + channel] as f32 / 255.0 * weight;
                }
                rows[y * size + x] = sum;
            }
        }
        let plane = &mut out[channel * size * size..(channel + 1) * size * size];
        for (y, (low, weights)) in down.iter().enumerate() {
            let target = &mut plane[y * size..(y + 1) * size];
            for (j, weight) in weights.iter().enumerate() {
                let source = &rows[(low + j) * size..(low + j + 1) * size];
                for (t, s) in target.iter_mut().zip(source) {
                    *t += s * weight;
                }
            }
            for t in target.iter_mut() {
                *t = (*t - mean[channel]) / std[channel];
            }
        }
    }
    out
}

/// Source positions of a bilinear enlargement: lower index, upper index, weight of the upper one.
fn bilinear_taps(input: usize, output: usize) -> Vec<(usize, usize, f32)> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|i| {
            let source = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
            let low = (source as usize).min(input - 1);
            (low, (low + 1).min(input - 1), source - low as f32)
        })
        .collect()
}

/// The mask `logits > 0` after bilinear resampling of a `side` x `side` map of logits to `width` x `height`.
pub fn mask_from_logits(logits: &[f32], side: usize, width: usize, height: usize) -> Plane<u8> {
    let across = bilinear_taps(side, width);
    let down = bilinear_taps(side, height);
    let mut mask = Plane::<u8>::new(width, height);
    let mut upper = vec![0f32; width];
    let mut lower = vec![0f32; width];
    let fill = |row: usize, target: &mut [f32]| {
        let line = &logits[row * side..(row + 1) * side];
        for (value, &(x0, x1, weight)) in target.iter_mut().zip(&across) {
            *value = line[x0] + weight * (line[x1] - line[x0]);
        }
    };
    let mut held = (usize::MAX, usize::MAX);
    for (y, &(y0, y1, weight)) in down.iter().enumerate() {
        if held != (y0, y1) {
            fill(y0, &mut upper);
            fill(y1, &mut lower);
            held = (y0, y1);
        }
        let target = &mut mask.data[y * width..(y + 1) * width];
        for ((m, &a), &b) in target.iter_mut().zip(&upper).zip(&lower) {
            *m = (a + weight * (b - a) > 0.0) as u8;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_resize_weights_are_those_of_pytorch() {
        // F.interpolate(arange(7.)[None, None, None], size=(1, 3), mode="bilinear", antialias=True) = [0.8571428, 3.0, 5.1428571]
        let rgb: Vec<u8> = (0..7u8).flat_map(|v| [v * 30, 0, 255]).collect();
        let out = model_input(&rgb, 7, 1, 3, [0.0; 3], [1.0; 3]);
        // A 7 x 1 photo to 3 x 3: every row equal.
        for (got, expected) in out[..3].iter().zip([0.857_142_8_f32, 3.0, 5.142_857]) {
            assert!((got * 255.0 / 30.0 - expected).abs() < 2e-5, "{got} {expected}");
        }
        assert_eq!(out[..3], out[6..9]);
        assert!(out[9..18].iter().all(|&v| v == 0.0) && out[18..].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // Enlarging uses the plain two-tap weights; normalisation is applied per channel.
        let out = model_input(&[0, 0, 0, 255, 255, 255], 2, 1, 4, [0.5, 0.0, 0.0], [0.5, 1.0, 1.0]);
        for (got, expected) in out[..4].iter().zip([-1.0f32, -0.5, 0.5, 1.0]) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn logits_are_enlarged_like_interpolate_without_aligned_corners() {
        // F.interpolate(tensor([[-1., 1.], [1., 3.]])[None, None], (4, 4), mode="bilinear") > 0
        let mask = mask_from_logits(&[-1.0, 1.0, 1.0, 3.0], 2, 4, 4);
        // Values of the first row: -1, -0.5, 0.5, 1; second: -0.5, 0, 1, 1.5.
        assert_eq!(mask.data, vec![0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        let same = mask_from_logits(&[1.0, -1.0, -1.0, -1.0], 2, 2, 2);
        assert_eq!(same.data, vec![1, 0, 0, 0]);
    }
}
