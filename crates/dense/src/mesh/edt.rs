//! Exact Euclidean distance transform on a voxel grid.
//!
//! The separable lower-envelope algorithm of Felzenszwalb and Huttenlocher ("Distance
//! Transforms of Sampled Functions", 2012) on squared distances. Squared distances between
//! voxels are integers, and they are kept as integers here, so the result is exact; its square
//! root equals `scipy.ndimage.distance_transform_edt` with unit sampling.

use super::grid::{for_each_line, Dims};

/// "No feature seen yet"; larger than any squared distance in a grid the pipeline can hold.
pub const FAR: i32 = i32::MAX;

/// Squared distance from every voxel to the nearest voxel where `feature` is true
/// (0 at the features themselves, `FAR` everywhere if there is none).
pub fn squared_distance(feature: impl Fn(usize) -> bool + Sync, dims: Dims) -> Vec<i32> {
    assert!(dims.iter().all(|&n| n < 20_000), "grid too large for 32-bit squared distances");
    let mut grid: Vec<i32> = (0..dims[0] * dims[1] * dims[2]).map(|i| if feature(i) { 0 } else { FAR }).collect();
    // Along z the features are points: two sweeps give the distance to the nearest one.
    for_each_line(
        &mut grid,
        dims,
        2,
        || (),
        |line, _| {
            let mut since: Option<i32> = None;
            for value in line.iter_mut() {
                since = if *value == 0 { Some(0) } else { since.map(|d| d + 1) };
                if let Some(d) = since {
                    *value = d * d;
                }
            }
            since = None;
            for value in line.iter_mut().rev() {
                since = if *value == 0 { Some(0) } else { since.map(|d| d + 1) };
                if let Some(d) = since {
                    *value = (*value).min(d * d);
                }
            }
        },
    );
    for axis in [1, 0] {
        let length = dims[axis];
        for_each_line(&mut grid, dims, axis, || Envelope::new(length), |line, envelope| envelope.transform(line));
    }
    grid
}

/// Scratch space of the one-dimensional transform `d(p) = min_q f(q) + (p - q)^2`.
pub struct Envelope {
    apex: Vec<usize>,
    from: Vec<f64>,
    source: Vec<i64>,
}

impl Envelope {
    pub fn new(length: usize) -> Self {
        Envelope { apex: vec![0; length], from: vec![0.0; length + 1], source: vec![0; length] }
    }

    pub fn transform(&mut self, line: &mut [i32]) {
        let mut parabolas = 0usize;
        for (q, &value) in line.iter().enumerate() {
            if value == FAR {
                continue;
            }
            let height = i64::from(value) + (q * q) as i64;
            self.source[q] = i64::from(value);
            // Intersections are ratios of integers below 2^53 with denominators below 2^16.
            // Distinct ones differ by far more than the rounding of one division and equal
            // ones round identically, so comparing them as f64 decides exactly; where two
            // parabolas tie at a voxel either gives the same distance.
            let mut start = f64::NEG_INFINITY;
            while parabolas > 0 {
                let v = self.apex[parabolas - 1];
                let other = self.source[v] + (v * v) as i64;
                start = (height - other) as f64 / (2 * (q - v)) as f64;
                if start > self.from[parabolas - 1] {
                    break;
                }
                parabolas -= 1;
                start = f64::NEG_INFINITY;
            }
            self.apex[parabolas] = q;
            self.from[parabolas] = start;
            parabolas += 1;
        }
        if parabolas == 0 {
            return;
        }
        let mut k = 0;
        for (p, value) in line.iter_mut().enumerate() {
            while k + 1 < parabolas && self.from[k + 1] < p as f64 {
                k += 1;
            }
            let v = self.apex[k];
            let offset = p as i64 - v as i64;
            *value = (self.source[v] + offset * offset) as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::testdata;

    #[test]
    fn one_dimensional_envelope() {
        let mut line = [FAR, 0, FAR, FAR, 9, 1, FAR];
        Envelope::new(7).transform(&mut line);
        assert_eq!(line, [1, 0, 1, 4, 2, 1, 2]);
        let mut empty = [FAR; 4];
        Envelope::new(4).transform(&mut empty);
        assert_eq!(empty, [FAR; 4]);
    }

    #[test]
    fn matches_brute_force() {
        let dims = [7, 6, 9];
        let n = dims[0] * dims[1] * dims[2];
        // A deterministic scatter of about one feature in eleven voxels.
        let feature: Vec<bool> = (0..n).map(|i| (i as u64 * 2654435761) % 97 < 9).collect();
        let fast = squared_distance(|i| feature[i], dims);
        let coordinates = |i: usize| [(i / (dims[1] * dims[2])) as i32, ((i / dims[2]) % dims[1]) as i32, (i % dims[2]) as i32];
        for (i, &found) in fast.iter().enumerate() {
            let a = coordinates(i);
            let brute = (0..n)
                .filter(|&j| feature[j])
                .map(|j| coordinates(j).iter().zip(&a).map(|(p, q)| (p - q) * (p - q)).sum::<i32>())
                .min()
                .unwrap();
            assert_eq!(found, brute, "voxel {i}");
        }
    }

    /// Against `scipy.ndimage.distance_transform_edt`; see `tests/fixtures/dense-native/make_fixtures.py`.
    #[test]
    fn matches_scipy() {
        let reference = testdata::scipy_reference();
        let case = &reference["distance_transform_edt"];
        let dims = testdata::dims(&case["shape"]);
        let input = testdata::numbers(&case["input"]);
        let expected = testdata::numbers(&case["output"]);
        // SciPy measures, at every non-zero element, the distance to the nearest zero.
        let squared = squared_distance(|i| input[i] == 0.0, dims);
        assert_eq!(squared.len(), expected.len());
        for (i, (&ours, &theirs)) in squared.iter().zip(&expected).enumerate() {
            assert!((f64::from(ours).sqrt() - theirs).abs() < 1e-12, "voxel {i}: {} against {theirs}", f64::from(ours).sqrt());
        }
    }
}
