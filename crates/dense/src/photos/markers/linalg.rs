//! Small dense linear algebra for the marker and turntable providers: 3-vectors
//! and rotations, a symmetric eigen solver, linear solves and the lens model.
//! Plain `f64` code without dependencies, so it builds for every target.

// Matrix code reads best with indices.
#![allow(clippy::needless_range_loop)]

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

pub const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn unit(a: V3) -> V3 {
    scale(a, 1.0 / norm(a).max(1e-300))
}

pub fn mul(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub fn apply(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

pub fn transpose(m: &M3) -> M3 {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

pub fn determinant(m: &M3) -> f64 {
    dot(m[0], cross(m[1], m[2]))
}

pub fn inverse(m: &M3) -> Option<M3> {
    let d = determinant(m);
    if d == 0.0 || !d.is_finite() {
        return None;
    }
    let (a, b, c) = (cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]));
    Some([[a[0] / d, b[0] / d, c[0] / d], [a[1] / d, b[1] / d, c[1] / d], [a[2] / d, b[2] / d, c[2] / d]])
}

/// Rotation matrix of an axis-angle vector (Rodrigues).
pub fn rodrigues(w: V3) -> M3 {
    let angle = norm(w);
    if angle < 1e-12 {
        return [[1.0, -w[2], w[1]], [w[2], 1.0, -w[0]], [-w[1], w[0], 1.0]];
    }
    let k = scale(w, 1.0 / angle);
    let (s, c) = angle.sin_cos();
    let v = 1.0 - c;
    [
        [c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s],
        [k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s],
        [k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v],
    ]
}

/// Angle in degrees of the rotation `a * b^T`.
pub fn rotation_angle_deg(a: &M3, b: &M3) -> f64 {
    let r = mul(a, &transpose(b));
    (((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
}

/// The rotation nearest to a matrix with positive determinant (polar factor by Newton's iteration).
pub fn nearest_rotation(m: &M3) -> M3 {
    let mut r = *m;
    for _ in 0..60 {
        let Some(inverse) = inverse(&r) else { break };
        let it = transpose(&inverse);
        let mut next = [[0.0; 3]; 3];
        let mut change = 0f64;
        for i in 0..3 {
            for j in 0..3 {
                next[i][j] = 0.5 * (r[i][j] + it[i][j]);
                change = change.max((next[i][j] - r[i][j]).abs());
            }
        }
        r = next;
        if change < 1e-15 {
            break;
        }
    }
    r
}

/// Eigenvalues (ascending) and eigenvectors (as rows) of a symmetric matrix, by cyclic Jacobi rotations.
pub fn symmetric_eigen(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| (i == j) as u8 as f64).collect()).collect();
    for _ in 0..100 {
        let off: f64 = (0..n).map(|i| (0..n).filter(|&j| j != i).map(|j| a[i][j] * a[i][j]).sum::<f64>()).sum();
        let diagonal: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-30 * diagonal.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[x][x].total_cmp(&a[y][y]));
    (order.iter().map(|&i| a[i][i]).collect(), order.iter().map(|&i| (0..n).map(|k| v[k][i]).collect()).collect())
}

/// Solves `a x = b` for a small dense system by elimination with partial pivoting.
pub fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for column in 0..n {
        let pivot = (column..n).max_by(|&x, &y| a[x][column].abs().total_cmp(&a[y][column].abs()))?;
        if a[pivot][column].abs() < 1e-300 || !a[pivot][column].is_finite() {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        for row in column + 1..n {
            let factor = a[row][column] / a[column][column];
            if factor != 0.0 {
                for k in column..n {
                    a[row][k] -= factor * a[column][k];
                }
                b[row] -= factor * b[column];
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let sum: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - sum) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// The homography `H` with `target ~ H * (source, 1)` from four or more point pairs (normalised DLT).
pub fn homography(source: &[[f64; 2]], target: &[[f64; 2]]) -> Option<M3> {
    if source.len() < 4 || source.len() != target.len() {
        return None;
    }
    let normalise = |points: &[[f64; 2]]| -> (M3, Vec<[f64; 2]>) {
        let n = points.len() as f64;
        let (mx, my) = (points.iter().map(|p| p[0]).sum::<f64>() / n, points.iter().map(|p| p[1]).sum::<f64>() / n);
        let spread = points.iter().map(|p| ((p[0] - mx).powi(2) + (p[1] - my).powi(2)).sqrt()).sum::<f64>() / n;
        let s = if spread > 0.0 { std::f64::consts::SQRT_2 / spread } else { 1.0 };
        ([[s, 0.0, -s * mx], [0.0, s, -s * my], [0.0, 0.0, 1.0]], points.iter().map(|p| [s * (p[0] - mx), s * (p[1] - my)]).collect())
    };
    let (ts, s) = normalise(source);
    let (tt, t) = normalise(target);
    let mut ata = vec![vec![0.0; 9]; 9];
    for (p, q) in s.iter().zip(&t) {
        let rows = [
            [-p[0], -p[1], -1.0, 0.0, 0.0, 0.0, q[0] * p[0], q[0] * p[1], q[0]],
            [0.0, 0.0, 0.0, -p[0], -p[1], -1.0, q[1] * p[0], q[1] * p[1], q[1]],
        ];
        for row in rows {
            for i in 0..9 {
                for j in 0..9 {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }
    }
    let (_, vectors) = symmetric_eigen(&ata);
    let h = &vectors[0];
    let normalised = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]];
    let result = mul(&inverse(&tt)?, &mul(&normalised, &ts));
    result.iter().flatten().all(|v| v.is_finite()).then_some(result)
}

pub fn map_point(h: &M3, p: [f64; 2]) -> [f64; 2] {
    let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    [(h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w, (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w]
}

/// Pinhole camera with radial k1, k2, k3; pixel (0, 0) is the centre of the top-left pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k: [f64; 3],
}

impl Camera {
    pub fn from_lens(lens: &crate::photos::solution::Lens) -> Self {
        let [fx, fy, cx, cy] = lens.pixels;
        Camera { fx, fy, cx, cy, k: lens.k }
    }

    /// Normalised undistorted coordinates to pixels.
    pub fn distort(&self, n: [f64; 2]) -> [f64; 2] {
        let rr = n[0] * n[0] + n[1] * n[1];
        let gain = 1.0 + rr * (self.k[0] + rr * (self.k[1] + rr * self.k[2]));
        [self.fx * n[0] * gain + self.cx, self.fy * n[1] * gain + self.cy]
    }

    /// Pixels to normalised undistorted coordinates (Newton's iteration on the radius).
    pub fn undistort(&self, pixel: [f64; 2]) -> [f64; 2] {
        let d = [(pixel[0] - self.cx) / self.fx, (pixel[1] - self.cy) / self.fy];
        let rd = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if rd < 1e-12 {
            return d;
        }
        let k = self.k;
        let mut r = rd;
        for _ in 0..30 {
            let r2 = r * r;
            let value = r * (1.0 + r2 * (k[0] + r2 * (k[1] + r2 * k[2]))) - rd;
            let slope = 1.0 + r2 * (3.0 * k[0] + r2 * (5.0 * k[1] + r2 * 7.0 * k[2]));
            if slope.abs() < 1e-9 {
                break;
            }
            let step = value / slope;
            r -= step;
            if step.abs() < 1e-14 {
                break;
            }
        }
        [d[0] * r / rd, d[1] * r / rd]
    }

    /// A point in the camera frame to pixels; `None` behind the camera.
    pub fn project(&self, p: V3) -> Option<[f64; 2]> {
        (p[2] > 1e-9).then(|| self.distort([p[0] / p[2], p[1] / p[2]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_and_polar_factor() {
        let r = rodrigues([0.3, -0.2, 0.9]);
        assert!((determinant(&r) - 1.0).abs() < 1e-12);
        let rt = mul(&r, &transpose(&r));
        assert!((0..3).all(|i| (0..3).all(|j| (rt[i][j] - (i == j) as u8 as f64).abs() < 1e-12)));
        assert!((rotation_angle_deg(&r, &IDENTITY) - norm([0.3, -0.2, 0.9]).to_degrees()).abs() < 1e-9);
        let mut skewed = r;
        skewed[0][1] += 0.05;
        skewed[2][2] *= 1.1;
        let fixed = nearest_rotation(&skewed);
        assert!((determinant(&fixed) - 1.0).abs() < 1e-12 && rotation_angle_deg(&fixed, &r) < 4.0);
        let inv = inverse(&skewed).unwrap();
        let id = mul(&inv, &skewed);
        assert!((id[0][0] - 1.0).abs() < 1e-12 && id[0][1].abs() < 1e-12);
    }

    #[test]
    fn eigen_solve_and_homography() {
        let m = vec![vec![4.0, 1.0, 0.5], vec![1.0, 3.0, 0.2], vec![0.5, 0.2, 1.0]];
        let (values, vectors) = symmetric_eigen(&m);
        assert!(values[0] < values[1] && values[1] < values[2]);
        for (value, vector) in values.iter().zip(&vectors) {
            for i in 0..3 {
                let mv: f64 = (0..3).map(|j| m[i][j] * vector[j]).sum();
                assert!((mv - value * vector[i]).abs() < 1e-10);
            }
        }
        let x = solve(m.clone(), vec![1.0, 2.0, 3.0]).unwrap();
        for i in 0..3 {
            assert!(((0..3).map(|j| m[i][j] * x[j]).sum::<f64>() - (i + 1) as f64).abs() < 1e-12);
        }
        assert!(solve(vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![1.0, 1.0]).is_none());
        let truth = [[1.2, 0.1, 5.0], [-0.2, 0.9, 3.0], [0.001, 0.002, 1.0]];
        let source: Vec<[f64; 2]> = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 7.0], [0.0, 7.0], [4.0, 3.0], [8.0, 1.0]];
        let target: Vec<[f64; 2]> = source.iter().map(|p| map_point(&truth, *p)).collect();
        let h = homography(&source, &target).unwrap();
        for p in &source {
            let (a, b) = (map_point(&h, *p), map_point(&truth, *p));
            assert!((a[0] - b[0]).abs() < 1e-8 && (a[1] - b[1]).abs() < 1e-8);
        }
    }

    #[test]
    fn lens_round_trip() {
        let camera = Camera { fx: 2328.0, fy: 2330.0, cx: 874.0, cy: 556.0, k: [-0.1718, 0.4123, -3.717] };
        for pixel in [[0.0, 0.0], [1748.0, 1154.0], [874.0, 556.0], [300.5, 900.25]] {
            let back = camera.distort(camera.undistort(pixel));
            assert!((back[0] - pixel[0]).abs() < 1e-8 && (back[1] - pixel[1]).abs() < 1e-8, "{pixel:?} {back:?}");
        }
        assert!(camera.project([0.0, 0.0, -1.0]).is_none());
        assert_eq!(camera.project([0.0, 0.0, 2.0]), Some([874.0, 556.0]));
    }
}
