//! Two-view geometry on normalised image coordinates: the essential matrix
//! by the eight-point algorithm inside RANSAC, its decomposition into a
//! rotation and a translation direction, Sampson distances and linear
//! triangulation.

use crate::photos::markers::linalg::{
    add, apply, cross, determinant, dot, mul, norm, scale, sub, symmetric_eigen, transpose, unit, M3, V3,
};

pub type P2 = [f64; 2];

/// Deterministic generator (SplitMix64) for the random samples.
pub struct Random(pub u64);

impl Random {
    pub fn draw(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.draw() % bound.max(1) as u64) as usize
    }
}

pub fn skew(v: V3) -> M3 {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

/// The essential matrix of a motion `x2 = R x1 + t`.
pub fn essential_of(rotation: &M3, translation: V3) -> M3 {
    mul(&skew(translation), rotation)
}

/// Sampson distance (not squared) of a pair under `b' E a = 0`, in normalised units.
pub fn sampson(e: &M3, a: P2, b: P2) -> f64 {
    let (ah, bh) = ([a[0], a[1], 1.0], [b[0], b[1], 1.0]);
    let ea = apply(e, ah);
    let etb = apply(&transpose(e), bh);
    let value = dot(bh, ea);
    (value * value / (ea[0] * ea[0] + ea[1] * ea[1] + etb[0] * etb[0] + etb[1] * etb[1] + 1e-18)).sqrt()
}

/// Singular value decomposition of a 3 x 3 matrix: `m = u * diag(s) * v^T`, values descending,
/// `u` and `v` as matrices with the vectors in their columns. The third left vector completes a
/// right-handed set when its singular value vanishes.
pub fn svd3(m: &M3) -> (M3, V3, M3) {
    let mtm = mul(&transpose(m), m);
    let (values, vectors) = symmetric_eigen(&mtm.iter().map(|r| r.to_vec()).collect::<Vec<_>>());
    // Ascending eigenvalues; take them in descending order.
    let v: Vec<V3> = (0..3).map(|k| [vectors[2 - k][0], vectors[2 - k][1], vectors[2 - k][2]]).collect();
    let s = [values[2].max(0.0).sqrt(), values[1].max(0.0).sqrt(), values[0].max(0.0).sqrt()];
    let u0 = unit(apply(m, v[0]));
    let mut u1 = apply(m, v[1]);
    u1 = unit(sub(u1, scale(u0, dot(u0, u1))));
    let u2 = if s[2] > 1e-9 * s[0].max(1e-300) {
        let raw = apply(m, v[2]);
        let rest = sub(sub(raw, scale(u0, dot(u0, raw))), scale(u1, dot(u1, raw)));
        if norm(rest) > 1e-12 {
            unit(rest)
        } else {
            cross(u0, u1)
        }
    } else {
        cross(u0, u1)
    };
    let columns = |c: [V3; 3]| [[c[0][0], c[1][0], c[2][0]], [c[0][1], c[1][1], c[2][1]], [c[0][2], c[1][2], c[2][2]]];
    (columns([u0, u1, u2]), s, columns([v[0], v[1], v[2]]))
}

/// The essential matrix closest to a 3 x 3 matrix: two equal singular values and a zero.
fn project_essential(m: &M3) -> M3 {
    let (u, _, v) = svd3(m);
    let d = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
    mul(&u, &mul(&d, &transpose(&v)))
}

/// Eight-point estimate from at least eight pairs (Hartley normalisation).
pub fn eight_point(a: &[P2], b: &[P2]) -> Option<M3> {
    if a.len() < 8 || a.len() != b.len() {
        return None;
    }
    let normalise = |points: &[P2]| -> (M3, Vec<P2>) {
        let n = points.len() as f64;
        let (mx, my) = (points.iter().map(|p| p[0]).sum::<f64>() / n, points.iter().map(|p| p[1]).sum::<f64>() / n);
        let spread = points.iter().map(|p| (p[0] - mx).hypot(p[1] - my)).sum::<f64>() / n;
        let s = if spread > 0.0 { std::f64::consts::SQRT_2 / spread } else { 1.0 };
        ([[s, 0.0, -s * mx], [0.0, s, -s * my], [0.0, 0.0, 1.0]], points.iter().map(|p| [s * (p[0] - mx), s * (p[1] - my)]).collect())
    };
    let (ta, na) = normalise(a);
    let (tb, nb) = normalise(b);
    let mut ata = vec![vec![0.0; 9]; 9];
    for (p, q) in na.iter().zip(&nb) {
        let row = [q[0] * p[0], q[0] * p[1], q[0], q[1] * p[0], q[1] * p[1], q[1], p[0], p[1], 1.0];
        for i in 0..9 {
            for j in i..9 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..9 {
        for j in 0..i {
            ata[i][j] = ata[j][i];
        }
    }
    let (_, vectors) = symmetric_eigen(&ata);
    let f = &vectors[0];
    let normalised = [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]];
    let e = mul(&transpose(&tb), &mul(&normalised, &ta));
    let e = project_essential(&e);
    e.iter().flatten().all(|v| v.is_finite()).then_some(e)
}

/// Essential matrix by RANSAC over eight-point samples, refitted on its inliers.
/// `bound` is the largest Sampson distance of an inlier. Returns the matrix and the inlier flags.
pub fn essential_ransac(a: &[P2], b: &[P2], bound: f64, random: &mut Random) -> Option<(M3, Vec<bool>)> {
    let n = a.len();
    if n < 8 {
        return None;
    }
    let count = |e: &M3| (0..n).filter(|&k| sampson(e, a[k], b[k]) < bound).count();
    let (mut best, mut best_count) = (None, 0usize);
    let mut needed = 2000usize;
    let mut iteration = 0;
    while iteration < needed {
        iteration += 1;
        let mut picks = [0usize; 8];
        let mut filled = 0;
        while filled < 8 {
            let candidate = random.below(n);
            if !picks[..filled].contains(&candidate) {
                picks[filled] = candidate;
                filled += 1;
            }
        }
        let (sa, sb): (Vec<P2>, Vec<P2>) = picks.iter().map(|&k| (a[k], b[k])).unzip();
        let Some(e) = eight_point(&sa, &sb) else { continue };
        let inliers = count(&e);
        if inliers > best_count {
            (best, best_count) = (Some(e), inliers);
            // Enough samples for one all-inlier sample with 99.9 % confidence.
            let chance = (inliers as f64 / n as f64).powi(8);
            let estimate = if chance >= 1.0 - 1e-12 { 1.0 } else { 0.001f64.ln() / (1.0 - chance).ln() };
            needed = needed.min((estimate.ceil() as usize).max(50));
        }
    }
    let mut e = best?;
    // Refit on the inliers, twice, each time with the inliers of the last fit.
    for _ in 0..2 {
        let flags: Vec<bool> = (0..n).map(|k| sampson(&e, a[k], b[k]) < bound).collect();
        let (ia, ib): (Vec<P2>, Vec<P2>) = (0..n).filter(|&k| flags[k]).map(|k| (a[k], b[k])).unzip();
        match eight_point(&ia, &ib) {
            Some(refit) if count(&refit) >= count(&e) => e = refit,
            _ => break,
        }
    }
    let flags: Vec<bool> = (0..n).map(|k| sampson(&e, a[k], b[k]) < bound).collect();
    Some((e, flags))
}

/// Linear triangulation of a point from normalised observations in several cameras (`x = R X + t`).
pub fn triangulate(observations: &[(M3, V3, P2)]) -> Option<V3> {
    let mut ata = vec![vec![0.0; 4]; 4];
    for (r, t, p) in observations {
        let row = |k: usize| [r[k][0], r[k][1], r[k][2], t[k]];
        let (p0, p1, p2) = (row(0), row(1), row(2));
        for line in [[0, 1, 2, 3].map(|c| p[0] * p2[c] - p0[c]), [0, 1, 2, 3].map(|c| p[1] * p2[c] - p1[c])] {
            for i in 0..4 {
                for j in 0..4 {
                    ata[i][j] += line[i] * line[j];
                }
            }
        }
    }
    let (_, vectors) = symmetric_eigen(&ata);
    let x = &vectors[0];
    (x[3].abs() > 1e-12).then(|| [x[0] / x[3], x[1] / x[3], x[2] / x[3]]).filter(|p| p.iter().all(|v| v.is_finite()))
}

/// Rotation and unit translation of an essential matrix (`x2 = R x1 + t`): of the four
/// decompositions the one that puts most inlier points in front of both cameras.
pub fn recover_pose(e: &M3, a: &[P2], b: &[P2], inliers: &[bool]) -> Option<(M3, V3)> {
    let (mut u, _, mut v) = svd3(e);
    if determinant(&u) < 0.0 {
        for row in u.iter_mut() {
            row[2] = -row[2];
        }
    }
    if determinant(&v) < 0.0 {
        for row in v.iter_mut() {
            row[2] = -row[2];
        }
    }
    let w = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let t = [u[0][2], u[1][2], u[2][2]];
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut best: Option<(usize, M3, V3)> = None;
    for rotation in [mul(&u, &mul(&w, &transpose(&v))), mul(&u, &mul(&transpose(&w), &transpose(&v)))] {
        for translation in [t, scale(t, -1.0)] {
            let mut front = 0;
            for k in (0..a.len()).filter(|&k| inliers[k]) {
                let Some(x) = triangulate(&[(identity, [0.0; 3], a[k]), (rotation, translation, b[k])]) else { continue };
                let second = add(apply(&rotation, x), translation);
                front += (x[2] > 0.0 && second[2] > 0.0) as usize;
            }
            if best.as_ref().is_none_or(|held| front > held.0) {
                best = Some((front, rotation, translation));
            }
        }
    }
    best.filter(|b| b.0 > 0).map(|b| (b.1, b.2))
}

/// Axis-angle vector of a rotation matrix.
pub fn rotation_vector(r: &M3) -> V3 {
    let cosine = ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0);
    let angle = cosine.acos();
    let axis = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    let length = norm(axis);
    if length < 1e-12 {
        return [0.0; 3];
    }
    scale(axis, angle / length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photos::markers::linalg::{rodrigues, rotation_angle_deg};

    fn scene(rotation: &M3, translation: V3, outliers: usize) -> (Vec<P2>, Vec<P2>) {
        let mut random = Random(5);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for n in 0..200 {
            let mut u = || random.draw() as f64 / u64::MAX as f64;
            let x = [u() * 2.0 - 1.0, u() * 2.0 - 1.0, 4.0 + 2.0 * u()];
            let y = add(apply(rotation, x), translation);
            a.push([x[0] / x[2], x[1] / x[2]]);
            b.push(if n < outliers { [u() - 0.5, u() - 0.5] } else { [y[0] / y[2], y[1] / y[2]] });
        }
        (a, b)
    }

    #[test]
    fn essential_matrix_recovers_a_small_turntable_step() {
        // Five degrees about an axis five units in front of the camera.
        let rotation = rodrigues([0.0, 5f64.to_radians(), 0.0]);
        let centre = [0.0, 0.0, 5.0];
        let translation = sub(centre, apply(&rotation, centre));
        let (a, b) = scene(&rotation, translation, 60);
        let truth = essential_of(&rotation, translation);
        assert!(sampson(&truth, a[100], b[100]) < 1e-12);
        let (e, inliers) = essential_ransac(&a, &b, 1e-3, &mut Random(1)).unwrap();
        assert!(inliers.iter().filter(|&&f| f).count() >= 140 && inliers[..60].iter().filter(|&&f| f).count() < 10);
        let (r, t) = recover_pose(&e, &a, &b, &inliers).unwrap();
        assert!(rotation_angle_deg(&r, &rotation) < 0.05, "{}", rotation_angle_deg(&r, &rotation));
        assert!(dot(t, unit(translation)) > 0.999, "{t:?}");
        let vector = rotation_vector(&r);
        assert!((norm(vector).to_degrees() - 5.0).abs() < 0.05 && vector[1] > 0.0);
    }

    #[test]
    fn svd_and_triangulation() {
        let m = [[2.0, 0.5, -1.0], [0.3, 1.5, 0.2], [-0.7, 0.1, 0.9]];
        let (u, s, v) = svd3(&m);
        let d = [[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, s[2]]];
        let back = mul(&u, &mul(&d, &transpose(&v)));
        assert!((0..3).all(|i| (0..3).all(|j| (back[i][j] - m[i][j]).abs() < 1e-9)), "{back:?}");
        assert!(s[0] >= s[1] && s[1] >= s[2]);
        let rotation = rodrigues([0.1, -0.2, 0.05]);
        let translation = [0.4, 0.1, -0.2];
        let x = [0.3, -0.2, 5.0];
        let y = add(apply(&rotation, x), translation);
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let found =
            triangulate(&[(identity, [0.0; 3], [x[0] / x[2], x[1] / x[2]]), (rotation, translation, [y[0] / y[2], y[1] / y[2]])]).unwrap();
        assert!(norm(sub(found, x)) < 1e-9);
    }
}
