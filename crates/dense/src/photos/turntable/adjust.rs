//! Bundle adjustment of all poses and points with the lens fixed:
//! Levenberg-Marquardt with Huber weights, the points eliminated by a Schur
//! complement so that only a `6 * photos` square system is solved. Every pose
//! has six free parameters; the first is held to fix the frame. Residuals are
//! in pixels (normalised coordinates times the focal length).

use crate::photos::markers::linalg::{add, apply, inverse, mul, rodrigues, solve, sub, M3, V3};

/// One observation: camera, point and the normalised image position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub camera: usize,
    pub point: usize,
    pub target: [f64; 2],
}

const HUBER: f64 = 1.0;

/// Residual vectors (pixels) of all observations, and the Huber cost.
pub fn residuals(rotations: &[M3], translations: &[V3], points: &[V3], observations: &[Observation], focal: f64) -> (Vec<[f64; 2]>, f64) {
    let mut cost = 0.0;
    let out = observations
        .iter()
        .map(|o| {
            let q = add(apply(&rotations[o.camera], points[o.point]), translations[o.camera]);
            let r = [(q[0] / q[2] - o.target[0]) * focal, (q[1] / q[2] - o.target[1]) * focal];
            let e = r[0].hypot(r[1]);
            cost += if !e.is_finite() || q[2] <= 0.0 {
                1e12
            } else if e <= HUBER {
                0.5 * e * e
            } else {
                HUBER * (e - 0.5 * HUBER)
            };
            r
        })
        .collect();
    (out, cost)
}

/// Adjusts poses and points in place; returns the cost before and after.
pub fn adjust(
    rotations: &mut Vec<M3>,
    translations: &mut Vec<V3>,
    points: &mut Vec<V3>,
    observations: &[Observation],
    focal: f64,
    iterations: usize,
) -> (f64, f64) {
    let (n, m) = (rotations.len(), points.len());
    // Observations of every point, for the elimination.
    let mut of_point: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (k, o) in observations.iter().enumerate() {
        of_point[o.point].push(k);
    }
    let (mut r, mut cost) = residuals(rotations, translations, points, observations, focal);
    let first = cost;
    let mut damping = 1e-3;
    for _ in 0..iterations {
        // Per observation: 2 x 6 pose block, 2 x 3 point block, weight.
        let mut jc = vec![[[0.0f64; 6]; 2]; observations.len()];
        let mut jp = vec![[[0.0f64; 3]; 2]; observations.len()];
        let mut weight = vec![1.0f64; observations.len()];
        let mut u = vec![[[0.0f64; 6]; 6]; n];
        let mut bc = vec![[0.0f64; 6]; n];
        let mut v = vec![[[0.0f64; 3]; 3]; m];
        let mut bp = vec![[0.0f64; 3]; m];
        for (k, o) in observations.iter().enumerate() {
            let rotation = &rotations[o.camera];
            let rx = apply(rotation, points[o.point]);
            let q = add(rx, translations[o.camera]);
            let iz = 1.0 / q[2];
            // d residual / d camera-frame point
            let d = [[focal * iz, 0.0, -focal * q[0] * iz * iz], [0.0, focal * iz, -focal * q[1] * iz * iz]];
            // d(exp(w) R X) / dw = -[R X]x
            let k_ = [[0.0, rx[2], -rx[1]], [-rx[2], 0.0, rx[0]], [rx[1], -rx[0], 0.0]];
            for a in 0..2 {
                for c in 0..3 {
                    jc[k][a][c] = (0..3).map(|i| d[a][i] * k_[i][c]).sum();
                    jc[k][a][3 + c] = d[a][c];
                    jp[k][a][c] = (0..3).map(|i| d[a][i] * rotation[i][c]).sum();
                }
            }
            if o.camera == 0 {
                jc[k] = [[0.0; 6]; 2];
            }
            let e = r[k][0].hypot(r[k][1]);
            weight[k] = if e <= HUBER { 1.0 } else { HUBER / e };
            for a in 0..2 {
                for i in 0..6 {
                    bc[o.camera][i] += weight[k] * jc[k][a][i] * r[k][a];
                    for j in 0..6 {
                        u[o.camera][i][j] += weight[k] * jc[k][a][i] * jc[k][a][j];
                    }
                }
                for i in 0..3 {
                    bp[o.point][i] += weight[k] * jp[k][a][i] * r[k][a];
                    for j in 0..3 {
                        v[o.point][i][j] += weight[k] * jp[k][a][i] * jp[k][a][j];
                    }
                }
            }
        }
        // W = Jc^T w Jp per observation, 6 x 3.
        let w: Vec<[[f64; 3]; 6]> = (0..observations.len())
            .map(|k| {
                let mut block = [[0.0; 3]; 6];
                for (i, row) in block.iter_mut().enumerate() {
                    for (j, value) in row.iter_mut().enumerate() {
                        *value = weight[k] * (jc[k][0][i] * jp[k][0][j] + jc[k][1][i] * jp[k][1][j]);
                    }
                }
                block
            })
            .collect();
        let mut improved = false;
        for _ in 0..8 {
            let mut vinv = vec![[[0.0f64; 3]; 3]; m];
            for (p, block) in v.iter().enumerate() {
                let mut damped = *block;
                for i in 0..3 {
                    damped[i][i] += damping * block[i][i] + 1e-12;
                }
                vinv[p] = inverse(&damped).unwrap_or([[0.0; 3]; 3]);
            }
            let mut s = vec![vec![0.0f64; 6 * n]; 6 * n];
            let mut rhs = vec![0.0f64; 6 * n];
            for c in 0..n {
                for i in 0..6 {
                    rhs[6 * c + i] = -bc[c][i];
                    for j in 0..6 {
                        s[6 * c + i][6 * c + j] = u[c][i][j];
                    }
                    s[6 * c + i][6 * c + i] += damping * u[c][i][i] + 1e-9;
                }
            }
            for (p, list) in of_point.iter().enumerate() {
                for &k1 in list {
                    // Y = W_k1 * Vinv, 6 x 3
                    let mut y = [[0.0f64; 3]; 6];
                    for i in 0..6 {
                        for j in 0..3 {
                            y[i][j] = (0..3).map(|l| w[k1][i][l] * vinv[p][l][j]).sum();
                        }
                    }
                    let c1 = observations[k1].camera;
                    for i in 0..6 {
                        rhs[6 * c1 + i] += (0..3).map(|j| y[i][j] * bp[p][j]).sum::<f64>();
                    }
                    for &k2 in list {
                        let c2 = observations[k2].camera;
                        for i in 0..6 {
                            for j in 0..6 {
                                s[6 * c1 + i][6 * c2 + j] -= (0..3).map(|l| y[i][l] * w[k2][j][l]).sum::<f64>();
                            }
                        }
                    }
                }
            }
            let Some(dc) = solve(s, rhs) else {
                damping *= 10.0;
                continue;
            };
            // Back substitution: dp = -Vinv (bp + W^T dc).
            let mut sums = bp.clone();
            for (k, o) in observations.iter().enumerate() {
                for j in 0..3 {
                    sums[o.point][j] += (0..6).map(|i| w[k][i][j] * dc[6 * o.camera + i]).sum::<f64>();
                }
            }
            let new_points: Vec<V3> = (0..m).map(|p| sub(points[p], apply(&vinv[p], sums[p]))).collect();
            let new_rotations: Vec<M3> =
                (0..n).map(|c| mul(&rodrigues([dc[6 * c], dc[6 * c + 1], dc[6 * c + 2]]), &rotations[c])).collect();
            let new_translations: Vec<V3> = (0..n).map(|c| add(translations[c], [dc[6 * c + 3], dc[6 * c + 4], dc[6 * c + 5]])).collect();
            let (new_r, new_cost) = residuals(&new_rotations, &new_translations, &new_points, observations, focal);
            if new_cost.is_finite() && new_cost < cost {
                let gain = cost - new_cost;
                (*rotations, *translations, *points, r, cost) = (new_rotations, new_translations, new_points, new_r, new_cost);
                damping = (damping / 3.0).max(1e-9);
                improved = gain > 1e-6 * cost;
                break;
            }
            damping *= 10.0;
        }
        if !improved {
            break;
        }
    }
    for rotation in rotations.iter_mut() {
        *rotation = crate::photos::markers::linalg::nearest_rotation(rotation);
    }
    (first, cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photos::markers::linalg::{norm, rotation_angle_deg};

    #[test]
    fn perturbed_ring_returns_to_the_truth() {
        let cameras = 8;
        let (mut rotations, mut translations) = (Vec::new(), Vec::new());
        for c in 0..cameras {
            let rotation = rodrigues([0.0, 0.25 * c as f64, 0.0]);
            let centre = [0.0, 0.0, 6.0];
            rotations.push(rotation);
            translations.push(sub(centre, apply(&rotation, centre)));
        }
        let mut points: Vec<V3> = (0..60).map(|n| [((n * 7) as f64).sin(), ((n * 3) as f64).cos(), 6.0 + ((n * 5) as f64).sin()]).collect();
        let mut observations = Vec::new();
        for (p, x) in points.iter().enumerate() {
            for c in 0..cameras {
                let q = add(apply(&rotations[c], *x), translations[c]);
                observations.push(Observation { camera: c, point: p, target: [q[0] / q[2], q[1] / q[2]] });
            }
        }
        let (true_rotations, true_points) = (rotations.clone(), points.clone());
        for c in 1..cameras {
            rotations[c] = mul(&rodrigues([0.01, -0.012 * c as f64 / 8.0, 0.008]), &rotations[c]);
            translations[c] = add(translations[c], [0.03, -0.02, 0.04]);
        }
        for (p, x) in points.iter_mut().enumerate() {
            *x = add(*x, [0.02 * (p as f64).sin(), 0.02 * (p as f64).cos(), -0.03]);
        }
        let (before, after) = adjust(&mut rotations, &mut translations, &mut points, &observations, 1000.0, 50);
        assert!(before > 1e3 && after < 1e-6, "{before} -> {after}");
        // The first camera is held, so the frame is the true one up to the free scale; the scale stays near 1.
        let scale = norm(points[0]) / norm(true_points[0]);
        assert!((scale - 1.0).abs() < 0.05, "{scale}");
        for c in 0..cameras {
            assert!(rotation_angle_deg(&rotations[c], &true_rotations[c]) < 1e-3);
        }
    }
}
