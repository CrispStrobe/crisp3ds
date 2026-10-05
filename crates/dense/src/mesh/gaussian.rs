//! Gaussian smoothing with the semantics of `scipy.ndimage.gaussian_filter` on `float32` data.
//!
//! Separable; the kernel is sampled at integer offsets, cut at `truncate` standard deviations
//! (SciPy's default 4) and normalised; the boundary is `reflect` (`d c b a | a b c d | d c b a`).
//! Each line is filtered in double precision and stored as `float32` before the next axis,
//! axes in the order x, y, z, exactly as SciPy does for a `float32` array.

use super::grid::{for_each_line, Dims};

pub const TRUNCATE: f64 = 4.0;

/// Normalised weights for offsets `-radius..=radius`, with `radius = int(truncate * sigma + 0.5)`.
pub fn kernel(sigma: f64) -> Vec<f64> {
    let radius = (TRUNCATE * sigma + 0.5) as i64;
    let mut weights: Vec<f64> = (-radius..=radius).map(|x| (-0.5 / (sigma * sigma) * (x * x) as f64).exp()).collect();
    let sum: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= sum);
    weights
}

/// Index into a line of `length` elements for position `i` outside it, under `reflect`.
fn reflect(i: i64, length: usize) -> usize {
    let n = length as i64;
    let period = 2 * n;
    let m = i.rem_euclid(period);
    (if m < n { m } else { period - 1 - m }) as usize
}

/// One line: `ext` and `sum` are scratch buffers.
fn filter_line(line: &mut [f32], weights: &[f64], ext: &mut Vec<f64>, sum: &mut Vec<f64>) {
    let radius = weights.len() / 2;
    let n = line.len();
    ext.clear();
    ext.extend((0..n + 2 * radius).map(|i| f64::from(line[reflect(i as i64 - radius as i64, n)])));
    // Same order of additions as SciPy's symmetric correlation: the centre first, then
    // pairs from the outermost inwards.
    sum.clear();
    sum.extend(ext[radius..radius + n].iter().map(|centre| centre * weights[radius]));
    for j in (1..=radius).rev() {
        let weight = weights[radius - j];
        let (left, right) = (&ext[radius - j..radius - j + n], &ext[radius + j..radius + j + n]);
        for ((total, a), b) in sum.iter_mut().zip(left).zip(right) {
            *total += (a + b) * weight;
        }
    }
    for (value, total) in line.iter_mut().zip(sum.iter()) {
        *value = *total as f32;
    }
}

/// Smooths the grid in place. `sigma <= 0` leaves it unchanged.
pub fn gaussian_filter(data: &mut [f32], dims: Dims, sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let weights = kernel(sigma);
    for axis in 0..3 {
        for_each_line(data, dims, axis, || (Vec::new(), Vec::new()), |line, (ext, sum)| filter_line(line, &weights, ext, sum));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::testdata;

    #[test]
    fn reflect_repeats_the_edge_element() {
        let picked: Vec<usize> = (-6..10).map(|i| reflect(i, 4)).collect();
        assert_eq!(picked, [2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1]);
        assert_eq!(reflect(-3, 1), 0);
    }

    #[test]
    fn kernel_is_normalised_and_cut_like_scipy() {
        assert_eq!(kernel(1.0).len(), 9);
        assert_eq!(kernel(0.6).len(), 5);
        assert_eq!(kernel(4.0).len(), 33);
        assert_eq!(kernel(0.1).len(), 1);
        assert!((kernel(2.0).iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn a_constant_stays_constant() {
        let dims = [5, 4, 3];
        let mut data = vec![0.75f32; 60];
        gaussian_filter(&mut data, dims, 4.0);
        assert!(data.iter().all(|&v| (v - 0.75).abs() < 1e-6));
    }

    /// Against `scipy.ndimage.gaussian_filter` on float32 input for the sigmas the mesher
    /// uses, including a kernel wider than the array; see `tests/fixtures/dense-native/make_fixtures.py`.
    #[test]
    fn matches_scipy() {
        let reference = testdata::scipy_reference();
        let case = &reference["gaussian_filter"];
        let dims = testdata::dims(&case["shape"]);
        let input: Vec<f32> = testdata::numbers(&case["input"]).iter().map(|&v| v as f32).collect();
        let outputs = case["outputs"].as_object().unwrap();
        assert!(outputs.len() >= 4);
        for (sigma, expected) in outputs {
            let mut data = input.clone();
            gaussian_filter(&mut data, dims, sigma.parse().unwrap());
            let expected = testdata::numbers(expected);
            let worst = data.iter().zip(&expected).map(|(&ours, &theirs)| (f64::from(ours) - theirs).abs()).fold(0.0, f64::max);
            // Values are of order one; float32 results differ by at most a unit in the last place.
            assert!(worst <= 2.4e-7, "sigma {sigma}: largest difference {worst}");
        }
    }
}
