//! Joint image-space fit of matches on a rotating planar support. Uses no
//! supplied poses, board dimensions, grid labels or reference geometry.
use super::solver::Matches;
use crate::photos::markers::linalg::{apply, cross, dot, norm, rodrigues, scale, solve, sub, unit, Camera, M3, V3};
use anyhow::{ensure, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Seed {
    pub axis: V3,
    pub centre: V3,
    pub steps: Vec<f64>,
    pub pairs: usize,
    pub inlier_fraction: f64,
}
#[derive(Clone)]
struct Pair {
    i: usize,
    gap: usize,
    a: Vec<V3>,
    b: Vec<[f64; 2]>,
}
fn frame(p: &[f64; 5]) -> (V3, V3) {
    let n = [p[1].sin(), p[1].cos() * p[0].cos(), p[1].cos() * p[0].sin()];
    let u = unit(sub([1., 0., 0.], scale(n, n[0])));
    let v = cross(n, u);
    (n, std::array::from_fn(|a| n[a] + p[2] * u[a] + p[3] * v[a]))
}
fn h(n: V3, c: V3, theta: f64) -> M3 {
    let r = rodrigues(scale(n, theta));
    let t = sub(c, apply(&r, c));
    std::array::from_fn(|i| std::array::from_fn(|j| r[i][j] + t[i] * n[j]))
}
fn errors(p: &[f64; 5], pairs: &[Pair], nominal: f64, focal: f64) -> Vec<f64> {
    let (n, c) = frame(p);
    let hs: [M3; 4] = std::array::from_fn(|gap| h(n, c, nominal * p[4] * (gap + 1) as f64));
    let mut out = Vec::new();
    for pair in pairs {
        for (&a, &b) in pair.a.iter().zip(&pair.b) {
            let q = apply(&hs[pair.gap - 1], a);
            out.push((q[0] / q[2] - b[0]) * focal);
            out.push((q[1] / q[2] - b[1]) * focal);
        }
    }
    out
}
fn cost(r: &[f64]) -> f64 {
    r.iter().map(|v| ((v / 2.).powi(2)).atan()).sum()
}
fn fit(mut p: [f64; 5], pairs: &[Pair], nominal: f64, focal: f64) -> (f64, [f64; 5]) {
    let sign = p[4];
    let low = [0.1, -0.2, -1., -4., sign - 0.1];
    let high = [1.5, 0.2, 1., -0.05, sign + 0.1];
    let mut damping = 1e-3;
    for _ in 0..80 {
        let r = errors(&p, pairs, nominal, focal);
        let before = cost(&r);
        let derivatives: [Vec<f64>; 5] = std::array::from_fn(|a| {
            let mut q = p;
            q[a] += 1e-5;
            errors(&q, pairs, nominal, focal).iter().zip(&r).map(|(x, y)| (x - y) / 1e-5).collect()
        });
        let mut a = vec![vec![0.; 5]; 5];
        let mut b = vec![0.; 5];
        for k in 0..r.len() {
            let weight = 1. / (1. + (r[k] / 2.).powi(4));
            for i in 0..5 {
                b[i] -= weight * derivatives[i][k] * r[k];
                for j in 0..5 {
                    a[i][j] += weight * derivatives[i][k] * derivatives[j][k];
                }
            }
        }
        for i in 0..5 {
            a[i][i] += damping * (a[i][i] + 1e-6);
        }
        let Some(delta) = solve(a, b) else { break };
        let q = std::array::from_fn(|i| (p[i] + delta[i]).clamp(low[i], high[i]));
        let after = cost(&errors(&q, pairs, nominal, focal));
        if after < before {
            p = q;
            damping = (damping * 0.3).max(1e-8);
            if before - after < 1e-7 {
                break;
            }
        } else {
            damping *= 10.;
            if damping > 1e8 {
                break;
            }
        }
    }
    (cost(&errors(&p, pairs, nominal, focal)), p)
}
fn step_cost(pair: &Pair, n: V3, c: V3, theta: f64, focal: f64) -> f64 {
    let matrix = h(n, c, theta);
    pair.a
        .iter()
        .zip(&pair.b)
        .map(|(&a, &b)| {
            let q = apply(&matrix, a);
            let x = (q[0] / q[2] - b[0]) * focal / 2.;
            let y = (q[1] / q[2] - b[1]) * focal / 2.;
            (x * x).atan() + (y * y).atan()
        })
        .sum()
}

pub fn estimate(matches: &Matches, inside: &[Vec<bool>], camera: &Camera) -> Result<Seed> {
    let count = matches.keypoints.len();
    let focal = (camera.fx + camera.fy) / 2.;
    let nominal = std::f64::consts::TAU / count as f64;
    let mut pairs = Vec::new();
    for (i, j, list) in &matches.pairs {
        let gap = (j + count - i) % count;
        if gap == 0 || gap > 4 {
            continue;
        }
        let kept: Vec<_> = list.iter().filter(|&&(a, b)| !inside[*i][a as usize] && !inside[*j][b as usize]).collect();
        if kept.len() < 20 {
            continue;
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        let stride = (kept.len() / 100).max(1);
        for &&(ia, ib) in kept.iter().step_by(stride) {
            let p = camera.undistort(matches.keypoints[*i][ia as usize]);
            a.push([p[0], p[1], 1.]);
            b.push(camera.undistort(matches.keypoints[*j][ib as usize]));
        }
        pairs.push(Pair { i: *i, gap, a, b });
    }
    let longer: Vec<_> = pairs.iter().filter(|p| p.gap >= 2).cloned().collect();
    ensure!(longer.len() >= 8, "too few rotating support matches");
    let mut best = (f64::INFINITY, [0.; 5]);
    for sign in [1., -1.] {
        for tilt in [30f64, 45., 60., 75.] {
            for ct in [-0.5, -1.] {
                let candidate = fit([tilt.to_radians(), 0., 0., ct, sign], &longer, nominal, focal);
                if candidate.0 < best.0 {
                    best = candidate;
                }
            }
        }
    }
    let p = best.1;
    let (n, c) = frame(&p);
    let residual = errors(&p, &longer, nominal, focal);
    let inlier_fraction = residual.as_chunks::<2>().0.iter().filter(|r| r[0].hypot(r[1]) < 2.).count() as f64 / (residual.len() / 2) as f64;
    ensure!(inlier_fraction >= 0.25, "rotating plane fit has too little image support ({:.1}%)", inlier_fraction * 100.);
    let typical = nominal * p[4];
    let mut steps = vec![typical; count];
    let mut supported = 0;
    for pair in pairs.iter().filter(|p| p.gap == 1) {
        // Refine only correspondences supported by the joint motion. A
        // repeated checkerboard can otherwise pull a small-angle fit sideways.
        let matrix = h(n, c, typical);
        let mut supported_pair = pair.clone();
        let kept: Vec<_> = pair
            .a
            .iter()
            .zip(&pair.b)
            .filter_map(|(&a, &b)| {
                let q = apply(&matrix, a);
                (((q[0] / q[2] - b[0]) * focal).hypot((q[1] / q[2] - b[1]) * focal) < 4.).then_some((a, b))
            })
            .collect();
        if kept.len() < 12 {
            continue;
        }
        (supported_pair.a, supported_pair.b) = kept.into_iter().unzip();
        let pair = &supported_pair;
        let sign = typical.signum();
        let mut best = (f64::INFINITY, typical.abs());
        for k in 0..33 {
            let theta = nominal * (0.2 + 1.6 * k as f64 / 32.);
            let v = step_cost(pair, n, c, theta * sign, focal);
            if v < best.0 {
                best = (v, theta);
            }
        }
        let mut lo = best.1 - nominal * 0.05;
        let mut hi = best.1 + nominal * 0.05;
        for _ in 0..24 {
            let l = lo + (hi - lo) / 3.;
            let r = hi - (hi - lo) / 3.;
            if step_cost(pair, n, c, l * sign, focal) < step_cost(pair, n, c, r * sign, focal) {
                hi = r
            } else {
                lo = l
            }
        }
        steps[pair.i] = (lo + hi) * 0.5 * sign;
        supported += 1;
    }
    ensure!(supported >= count * 3 / 4, "too few consecutive rotating-plane estimates");
    let centre = unit(sub(c, scale(n, dot(c, n))));
    ensure!(norm(centre) > 0., "degenerate rotating support centre");
    Ok(Seed { axis: n, centre, steps, pairs: longer.len(), inlier_fraction })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_plane_fit_recovers_steep_motion_with_wrong_matches() {
        let camera = Camera { fx: 1000., fy: 1000., cx: 500., cy: 400., k: [0.; 3] };
        let p = [55f64.to_radians(), -0.03, 0.02, -0.7, 1.];
        let (n, c) = frame(&p);
        let count = 16;
        let base: Vec<_> = (0..80)
            .map(|i| {
                let x = (i % 10) as f64 * 0.025 - 0.12;
                let y = (i / 10) as f64 * 0.025 - 0.1;
                let u = unit(sub([1., 0., 0.], scale(n, n[0])));
                let v = cross(n, u);
                std::array::from_fn(|a| c[a] + x * u[a] + y * v[a])
            })
            .collect();
        let keypoints: Vec<Vec<[f64; 2]>> = (0..count)
            .map(|i| {
                let r = rodrigues(scale(n, std::f64::consts::TAU * i as f64 / count as f64));
                base.iter()
                    .map(|x| {
                        let q = std::array::from_fn(|a| c[a] + apply(&r, sub(*x, c))[a]);
                        camera.project(q).unwrap()
                    })
                    .collect()
            })
            .collect();
        let pairs = super::super::matching::ring_pairs(count, 4, false)
            .into_iter()
            .map(|(i, j)| (i, j, (0..80).map(|k| (k, if k % 7 == 0 { (k + 23) % 80 } else { k })).collect()))
            .collect();
        let matches = Matches { keypoints, pairs };
        let seed = estimate(&matches, &vec![vec![false; 80]; count], &camera).unwrap();
        assert!(dot(seed.axis, n) > 0.99999);
        assert!(seed.inlier_fraction > 0.8);
        assert!((seed.steps.iter().sum::<f64>() - std::f64::consts::TAU).abs() < 0.001, "{seed:?}");
    }
}
