//! Cameras of an ordered turntable capture from feature matches
//! (`docs/TURNTABLE-SOLVER.md`): one repeated rotation about a fixed axis as
//! the starting point, then tracks and a bundle adjustment with free poses.
//! Port of `crates/dense/tools/turntable_prototype.py`.

use std::collections::HashMap;

use anyhow::bail;
use serde_json::{json, Value};

use crate::inputs::median_f64;
use crate::photos::markers::linalg::{add, apply, dot, mul, norm, rodrigues, scale, sub, transpose, unit, Camera, M3, V3};
use crate::photos::solution::{Landmark, Lens, Solution, View};

use super::adjust::{adjust, residuals, Observation};
use super::geometry::{essential_of, sampson, triangulate, P2};

/// `(first photo, second photo, [(feature in first, feature in second)])`.
pub type PairMatches = (usize, usize, Vec<(u32, u32)>);

/// Features of every photo (pixels, centre of the top-left pixel at (0, 0)) and matches between photo pairs.
#[derive(Debug, Clone, Default)]
pub struct Matches {
    pub keypoints: Vec<Vec<P2>>,
    /// `(first photo, second photo, [(feature in first, feature in second)])`.
    pub pairs: Vec<PairMatches>,
}

impl Matches {
    /// The file the prototype writes with `--export-matches` (`crisp3ds_turntable_matches_v1`).
    pub fn from_json(value: &Value) -> anyhow::Result<Matches> {
        if value["schema"] != "crisp3ds_turntable_matches_v1" {
            bail!("not a turntable matches file");
        }
        let keypoints: Vec<Vec<P2>> = serde_json::from_value(value["keypoints"].clone())?;
        let mut pairs = Vec::new();
        for pair in value["pairs"].as_array().ok_or_else(|| anyhow::anyhow!("matches pairs must be an array"))? {
            let list: Vec<(u32, u32)> = serde_json::from_value(pair["matches"].clone())?;
            let first =
                pair["first"].as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| anyhow::anyhow!("invalid first view index"))?;
            let second = pair["second"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| anyhow::anyhow!("invalid second view index"))?;
            pairs.push((first, second, list));
        }
        if keypoints.iter().flatten().flatten().any(|v| !v.is_finite())
            || pairs.iter().any(|(i, j, list)| {
                *i >= keypoints.len()
                    || *j >= keypoints.len()
                    || list.iter().any(|&(a, b)| a as usize >= keypoints[*i].len() || b as usize >= keypoints[*j].len())
            })
        {
            bail!("matches contain invalid coordinates or feature indices");
        }
        Ok(Matches { keypoints, pairs })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// The photos do not close a full turn: do not scale the steps to 360 degrees.
    pub open_turn: bool,
    /// Expanded search for downward views, used only with experimental surface features.
    pub wide_axis: bool,
    pub planar: Option<super::planar::Seed>,
}

/// Turns that the steps of a closed capture may add up to before they are scaled to one. Measured on
/// correct solutions: 0.78 (the Happy Buddha, rotation and translation hard to tell apart) to 1.07 (25
/// photos at 15 degrees); a wrong axis gave 1.19 (a smooth bottle seen from a steep camera).
const CLOSING_TURNS: (f64, f64) = (0.75, 1.15);

/// The step motion of the turntable model: rotation by `angle` about the axis through `centre`.
fn step_motion(axis: V3, centre: V3, angle: f64) -> (M3, V3) {
    let rotation = rodrigues(scale(axis, angle));
    (rotation, sub(centre, apply(&rotation, centre)))
}

/// Minimum of a function on an interval by golden-section search.
fn minimise(f: &dyn Fn(f64) -> f64, mut low: f64, mut high: f64) -> f64 {
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (high - ratio * (high - low), low + ratio * (high - low));
    let (mut fa, mut fb) = (f(a), f(b));
    for _ in 0..40 {
        if fa < fb {
            (high, b, fb) = (b, a, fa);
            a = high - ratio * (high - low);
            fa = f(a);
        } else {
            (low, a, fa) = (a, b, fb);
            b = low + ratio * (high - low);
            fb = f(b);
        }
    }
    (low + high) / 2.0
}

struct Tracks {
    points: Vec<V3>,
    /// `(photo, point, feature)`.
    observations: Vec<(usize, usize, u32)>,
    verified: usize,
}

/// Matches that agree with the poses (Sampson distance below `bound` pixels), joined into tracks of
/// at least three photos and triangulated.
fn build(matches: &Matches, normalised: &[Vec<P2>], rotations: &[M3], translations: &[V3], focal: f64, bound: f64) -> Tracks {
    let mut ids: HashMap<(usize, u32), usize> = HashMap::new();
    let mut parent: Vec<usize> = Vec::new();
    let mut nodes: Vec<(usize, u32)> = Vec::new();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut verified = 0;
    for (i, j, list) in &matches.pairs {
        let relative = mul(&rotations[*j], &transpose(&rotations[*i]));
        let translation = sub(translations[*j], apply(&relative, translations[*i]));
        if norm(translation) < 1e-6 {
            continue;
        }
        let e = essential_of(&relative, translation);
        for &(a, b) in list {
            if sampson(&e, normalised[*i][a as usize], normalised[*j][b as usize]) * focal >= bound {
                continue;
            }
            verified += 1;
            let mut node = |key: (usize, u32)| {
                *ids.entry(key).or_insert_with(|| {
                    parent.push(parent.len());
                    nodes.push(key);
                    parent.len() - 1
                })
            };
            let (x, y) = (node((*i, a)), node((*j, b)));
            let (rx, ry) = (find(&mut parent, x), find(&mut parent, y));
            parent[rx] = ry;
        }
    }
    let mut groups: HashMap<usize, Vec<(usize, u32)>> = HashMap::new();
    for (index, key) in nodes.iter().enumerate() {
        groups.entry(find(&mut parent, index)).or_default().push(*key);
    }
    let mut tracks: Vec<Vec<(usize, u32)>> = groups.into_values().collect();
    for track in &mut tracks {
        track.sort_unstable();
    }
    // A fixed order, whatever the hash map did.
    tracks.sort_unstable();
    let (mut points, mut observations) = (Vec::new(), Vec::new());
    for track in tracks {
        let distinct = track.windows(2).all(|w| w[0].0 != w[1].0);
        if track.len() < 3 || !distinct {
            continue;
        }
        let seen: Vec<(M3, V3, P2)> = track.iter().map(|&(v, f)| (rotations[v], translations[v], normalised[v][f as usize])).collect();
        let Some(x) = triangulate(&seen) else { continue };
        let mut worst = 0f64;
        let mut front = true;
        for (r, t, p) in &seen {
            let q = add(apply(r, x), *t);
            front &= q[2] > 0.0;
            worst = worst.max((q[0] / q[2] - p[0]).hypot(q[1] / q[2] - p[1]) * focal);
        }
        if front && worst < 2.0 * bound {
            observations.extend(track.iter().map(|&(v, f)| (v, points.len(), f)));
            points.push(x);
        }
    }
    Tracks { points, observations, verified }
}

/// What the solver found, with the numbers for the report.
pub struct Solved {
    pub rotations: Vec<M3>,
    pub translations: Vec<V3>,
    pub points: Vec<V3>,
    /// `(photo, point, feature)`.
    pub observations: Vec<(usize, usize, u32)>,
    pub report: Value,
}

/// Solves the cameras of `matches` for a lens `camera`. The frame is the first camera's; the
/// distance from the camera to the turntable axis is 1 before the adjustment.
pub fn solve(matches: &Matches, camera: &Camera, settings: &Settings) -> anyhow::Result<Solved> {
    let n = matches.keypoints.len();
    if n < 8 {
        bail!("the turntable solver needs at least 8 photos, found {n}");
    }
    let focal = (camera.fx + camera.fy) / 2.0;
    let normalised: Vec<Vec<P2>> = matches.keypoints.iter().map(|list| list.iter().map(|p| camera.undistort(*p)).collect()).collect();
    let pair = |i: usize, j: usize| matches.pairs.iter().find(|p| p.0 == i && p.1 == j).map(|p| p.2.as_slice()).unwrap_or(&[]);
    let consecutive: Vec<(Vec<P2>, Vec<P2>)> = (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            pair(i, j).iter().map(|&(a, b)| (normalised[i][a as usize], normalised[j][b as usize])).unzip()
        })
        .collect();
    let flow = |a: &[P2], b: &[P2]| -> f64 {
        if a.is_empty() {
            return 0.0;
        }
        median_f64(&mut a.iter().zip(b).map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1])).collect::<Vec<_>>()) * focal
    };

    // One motion for every step: a rotation about a fixed axis. Its direction, the direction from the
    // camera to it (at distance 1, which fixes the scale) and the angle of every step are fitted to the
    // matches of all consecutive pairs together. A single pair of a small object seen through a narrow
    // field barely tells rotation from translation; all pairs sharing one axis do.
    let moving: Vec<usize> = (0..n).filter(|&i| consecutive[i].0.len() >= 30 && flow(&consecutive[i].0, &consecutive[i].1) > 2.0).collect();
    if moving.len() < 3 {
        bail!("only {} consecutive photo pairs show motion; too few matches between neighbours", moving.len());
    }
    let mut moved = vec![false; n];
    for &i in &moving {
        moved[i] = true;
    }
    // At most 400 matches per pair for the fit; all of them for the angles afterwards.
    let sample: Vec<Vec<(P2, P2)>> = moving
        .iter()
        .map(|&i| {
            let (a, b) = &consecutive[i];
            let stride = a.len().div_ceil(400).max(1);
            (0..a.len()).step_by(stride).map(|k| (a[k], b[k])).collect()
        })
        .collect();
    let pair_cost = |pairs: &[(P2, P2)], axis: V3, centre: V3, angle: f64| -> f64 {
        let (rotation, t) = step_motion(axis, centre, angle);
        let e = essential_of(&rotation, t);
        pairs.iter().map(|(p, q)| (sampson(&e, *p, *q) * focal).min(3.0)).sum()
    };
    let nominal = 2.0 * std::f64::consts::PI / n as f64;
    // The best angle of one pair for a given axis: a coarse scan, then a fine search.
    let best_angle = |pairs: &[(P2, P2)], axis: V3, centre: V3, typical: f64, fine: bool| -> (f64, f64) {
        let mut best = (f64::INFINITY, typical);
        for k in 0..24 {
            let angle = typical * (0.2 + 2.3 * k as f64 / 23.0);
            let value = pair_cost(pairs, axis, centre, angle);
            if value < best.0 {
                best = (value, angle);
            }
        }
        if fine {
            let reach = 0.1 * typical.abs();
            let angle = minimise(&|x| pair_cost(pairs, axis, centre, x), best.1 - reach, best.1 + reach);
            best = (pair_cost(pairs, axis, centre, angle), angle);
        }
        best
    };
    let total_cost =
        |axis: V3, centre: V3, typical: f64| -> f64 { sample.iter().map(|pairs| best_angle(pairs, axis, centre, typical, false).0).sum() };
    // The axis is roughly upright in the photos and the object in front of the camera: start from the
    // best of a few tilts and both turning directions, towards where the features are.
    let mean = {
        let all: Vec<P2> = moving.iter().flat_map(|&i| consecutive[i].0.iter().copied()).collect();
        let count = all.len().max(1) as f64;
        [all.iter().map(|p| p[0]).sum::<f64>() / count, all.iter().map(|p| p[1]).sum::<f64>() / count, 1.0]
    };
    let across = |axis: V3, v: V3| unit(sub(v, scale(axis, dot(axis, v))));
    let mut start = (f64::INFINITY, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], nominal);
    for tilt in [-80.0f64, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0] {
        if !settings.wide_axis && tilt.abs() > 50.0 {
            continue;
        }
        let axis = [0.0, tilt.to_radians().cos(), tilt.to_radians().sin()];
        let centre = across(axis, mean);
        for typical in [nominal, -nominal] {
            let value = total_cost(axis, centre, typical);
            if value < start.0 {
                start = (value, axis, centre, typical);
            }
        }
    }
    if let Some(seed) = &settings.planar {
        start = (0.0, seed.axis, seed.centre, median_f64(&mut seed.steps.clone()));
    }
    let (_, mut axis, mut centre, signed) = start;
    // Refinement: the axis direction (two angles) and the direction to the axis (one angle about it),
    // one after the other over shrinking ranges, the step angles re-fitted inside every evaluation.
    for range in [12.0f64, 6.0, 3.0, 1.5, 0.7, 0.3, 0.1] {
        if settings.planar.is_some() {
            break;
        }
        let range = range.to_radians();
        for parameter in 0..3 {
            let moved_to = |x: f64| -> (V3, V3) {
                let turn = match parameter {
                    0 => rodrigues(scale(centre, x)),
                    1 => rodrigues(scale(crate::photos::markers::linalg::cross(axis, centre), x)),
                    _ => rodrigues(scale(axis, x)),
                };
                let new_axis = if parameter < 2 { unit(apply(&turn, axis)) } else { axis };
                (new_axis, across(new_axis, apply(&turn, centre)))
            };
            let evaluate = |x: f64| {
                let (a, c) = moved_to(x);
                total_cost(a, c, signed)
            };
            let candidate = moved_to(minimise(&evaluate, -range, range));
            if total_cost(candidate.0, candidate.1, signed) < total_cost(axis, centre, signed) {
                (axis, centre) = candidate;
            }
        }
    }
    let mut angles: Vec<f64> = sample.iter().map(|pairs| best_angle(pairs, axis, centre, signed, true).1).collect();
    let typical = if let Some(seed) = &settings.planar { median_f64(&mut seed.steps.clone()) } else { median_f64(&mut angles) };
    let axes = &moving;

    // The angle of every step with the axis fixed, from all its matches.
    let mut steps = Vec::with_capacity(n);
    for (i, (a, b)) in consecutive.iter().enumerate() {
        if a.len() < 12 {
            steps.push(typical);
            continue;
        }
        // A step of zero has no epipolar geometry; it shows as no image motion.
        if !moved[i] && flow(a, b) <= 2.0 {
            steps.push(0.0);
            continue;
        }
        let pairs: Vec<(P2, P2)> = a.iter().copied().zip(b.iter().copied()).collect();
        steps.push(best_angle(&pairs, axis, centre, typical, true).1);
    }
    if let Some(seed) = &settings.planar {
        steps = seed.steps.clone();
    }
    // Work with positive steps: an axis pointing the other way turns the same way.
    if typical < 0.0 {
        axis = scale(axis, -1.0);
        steps.iter_mut().for_each(|s| *s = -*s);
    }
    let typical = typical.abs();
    // A step far from the others is a failed search, not a jump of the turntable.
    let usual = median_f64(&mut steps.iter().copied().filter(|s| *s > 0.0).collect::<Vec<_>>());
    for step in &mut steps {
        if *step > 1.6 * typical || (*step < 0.4 * typical && *step > 0.0) {
            *step = usual;
        }
    }
    let raw_sum: f64 = steps.iter().sum();
    let turns = raw_sum / (2.0 * std::f64::consts::PI);
    // A full turn closes: all steps, the one back to the first photo included, add up to 360 degrees.
    // Steps found from the matches add up to close to one turn when the axis is right; far from it
    // the motion was not found, and scaling the steps to 360 degrees would only hide that (the
    // adjustment then settles on wrong cameras that pass every later gate).
    if !settings.open_turn && !(CLOSING_TURNS.0..=CLOSING_TURNS.1).contains(&turns) {
        bail!(
            "the steps found add up to {:.0} degrees, not about one turn ({:.0} to {:.0}): the turntable's motion was not found \
             (an axis tilted further than the search reaches, or too few features on the object); pass --open-turn if the \
             photos are not one closed turn",
            raw_sum.to_degrees(),
            360.0 * CLOSING_TURNS.0,
            360.0 * CLOSING_TURNS.1
        );
    }
    let closed = !settings.open_turn;
    if closed {
        for step in &mut steps {
            *step /= turns;
        }
    }
    let (mut rotations, mut translations) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut total = 0.0;
    for step in &steps {
        let (rotation, t) = step_motion(axis, centre, total);
        rotations.push(rotation);
        translations.push(t);
        total += step;
    }

    // Tracks, then adjustment rounds: once more tracks with the adjusted poses, then twice without the far observations.
    let mut tracks = build(matches, &normalised, &rotations, &translations, focal, 4.0);
    let first_tracks = (tracks.verified, tracks.points.len(), tracks.observations.len());
    let mut rounds = Vec::new();
    let mut errors: Vec<f64> = Vec::new();
    for round in 0..4 {
        if tracks.points.len() < 20 {
            bail!("only {} points could be triangulated from the matches", tracks.points.len());
        }
        let observations: Vec<Observation> =
            tracks.observations.iter().map(|&(v, p, f)| Observation { camera: v, point: p, target: normalised[v][f as usize] }).collect();
        let (before, after) = adjust(&mut rotations, &mut translations, &mut tracks.points, &observations, focal, 40);
        let (r, _) = residuals(&rotations, &translations, &tracks.points, &observations, focal);
        errors = r.iter().map(|d| d[0].hypot(d[1])).collect();
        let median = median_f64(&mut errors.clone());
        rounds.push(json!({"observations": observations.len(), "points": tracks.points.len(), "cost_before": before, "cost_after": after, "median_px": median}));
        if round == 3 {
            break;
        }
        if round == 0 {
            tracks = build(matches, &normalised, &rotations, &translations, focal, 3.0);
            continue;
        }
        let bound = (3.0 * median).max(2.0);
        let kept: Vec<(usize, usize, u32)> =
            tracks.observations.iter().zip(&errors).filter(|(_, e)| **e < bound).map(|(o, _)| *o).collect();
        let mut count = vec![0usize; tracks.points.len()];
        for o in &kept {
            count[o.1] += 1;
        }
        let mut renumber = vec![usize::MAX; tracks.points.len()];
        let mut points = Vec::new();
        for (p, x) in tracks.points.iter().enumerate() {
            if count[p] >= 2 {
                renumber[p] = points.len();
                points.push(*x);
            }
        }
        tracks.observations = kept.into_iter().filter(|o| renumber[o.1] != usize::MAX).map(|(v, p, f)| (v, renumber[p], f)).collect();
        tracks.points = points;
    }
    let mut sorted = errors.clone();
    sorted.sort_by(f64::total_cmp);
    let report = json!({
        "planar_seed_pairs": settings.planar.as_ref().map(|s|s.pairs), "planar_seed_inlier_fraction": settings.planar.as_ref().map(|s|s.inlier_fraction), "photos": n, "pairs_with_motion": axes.len(), "axis": axis, "direction_to_axis": centre,
        "step_median_deg": typical.to_degrees(), "turn_sum_deg": raw_sum.to_degrees(), "closed_to_full_turn": closed,
        "steps_deg": steps.iter().map(|s| (s.to_degrees() * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        "first_tracks": {"verified_matches": first_tracks.0, "points": first_tracks.1, "observations": first_tracks.2},
        "adjustments": rounds, "points": tracks.points.len(), "observations": tracks.observations.len(),
        "reprojection_px": {"median": median_f64(&mut errors), "p95": crate::inputs::percentile_sorted_f64(&sorted, 95.0)},
    });
    Ok(Solved { rotations, translations, points: tracks.points, observations: tracks.observations, report })
}

/// The neutral solution of a solved capture; `names[i]` is the photo of camera `i`.
pub fn solution(lens: &Lens, names: &[String], matches: &Matches, solved: &Solved) -> Solution {
    let views = names
        .iter()
        .enumerate()
        .map(|(n, name)| {
            let (rotation, translation) = (solved.rotations[n], solved.translations[n]);
            let centre = scale(apply(&transpose(&rotation), translation), -1.0);
            View { id: n.to_string(), source: name.clone(), rotation, centre, translation }
        })
        .collect();
    let mut landmarks: Vec<Landmark> = solved.points.iter().map(|p| Landmark { position: *p, observations: Vec::new() }).collect();
    for &(view, point, feature) in &solved.observations {
        landmarks[point].observations.push((view, matches.keypoints[view][feature as usize]));
    }
    Solution { lens: *lens, views, unregistered: Vec::new(), landmarks, lens_locked: Some(true), scale: None, object_points: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photos::markers::linalg::rotation_angle_deg;

    #[test]
    fn malformed_match_indices_are_refused_before_indexing() {
        let valid = json!({"schema":"crisp3ds_turntable_matches_v1", "keypoints":[[[1.,2.]],[[3.,4.]]], "pairs":[{"first":0,"second":1,"matches":[[0,0]]}]});
        assert!(Matches::from_json(&valid).is_ok());
        for (field, value) in [("first", json!(-1)), ("second", json!(9)), ("matches", json!([[0, 8]]))] {
            let mut bad = valid.clone();
            bad["pairs"][0][field] = value;
            assert!(Matches::from_json(&bad).is_err());
        }
        let mut bad = valid;
        bad["pairs"][0].as_object_mut().unwrap().remove("first");
        assert!(Matches::from_json(&bad).is_err());
    }

    /// A turntable capture as matches: a blob of points about a tilted axis, uneven steps (one of them
    /// zero), a distorting lens, one wrong match in ten.
    fn capture(photos: usize, turn: f64) -> (Matches, Camera, Vec<M3>, Vec<V3>) {
        capture_tilt(photos, turn, 20.0)
    }

    fn capture_tilt(photos: usize, turn: f64, tilt: f64) -> (Matches, Camera, Vec<M3>, Vec<V3>) {
        let camera = Camera { fx: 2300.0, fy: 2302.0, cx: 870.0, cy: 560.0, k: [-0.1, 0.2, -0.5] };
        let axis = unit([0.02, tilt.to_radians().cos(), tilt.to_radians().sin()]);
        let centre = [0.01, 0.05, 1.0];
        let nominal = turn * 2.0 * std::f64::consts::PI / photos as f64;
        let mut steps: Vec<f64> = (0..photos).map(|i| nominal * (1.0 + 0.1 * ((i * 7) as f64).sin())).collect();
        steps[5] = 0.0;
        let (mut rotations, mut translations, mut total) = (Vec::new(), Vec::new(), 0.0);
        for step in &steps {
            let (r, t) = step_motion(axis, centre, total);
            rotations.push(r);
            translations.push(t);
            total += step;
        }
        let points: Vec<V3> = (0..400)
            .map(|k| {
                let f = k as f64;
                [0.01 + 0.07 * (f * 1.7).sin(), 0.05 + 0.08 * (f * 2.3).cos(), 1.0 + 0.07 * (f * 3.1).sin() * (f * 0.37).cos()]
            })
            .collect();
        let keypoints: Vec<Vec<P2>> = (0..photos)
            .map(|v| points.iter().map(|x| camera.project(add(apply(&rotations[v], *x), translations[v])).unwrap()).collect())
            .collect();
        let mut pairs = Vec::new();
        for (i, j) in crate::photos::turntable::matching::ring_pairs(photos, 3, turn < 0.99) {
            let list = (0..400u32).map(|k| if (k as usize + i).is_multiple_of(10) { (k, (k * 37 + 11) % 400) } else { (k, k) }).collect();
            pairs.push((i, j, list));
        }
        (Matches { keypoints, pairs }, camera, rotations, translations)
    }

    #[test]
    fn steep_capture_recovers_without_relaxing_closure() {
        let (matches, camera, truth, _) = capture_tilt(24, 1.0, 65.0);
        let solved = solve(&matches, &camera, &Settings { open_turn: false, wide_axis: true, planar: None }).unwrap();
        assert!(solved.report["reprojection_px"]["median"].as_f64().unwrap() < 0.01);
        for (actual, expected) in solved.rotations.iter().zip(truth) {
            assert!(rotation_angle_deg(actual, &expected) < 0.01);
        }
    }

    #[test]
    fn turntable_capture_is_recovered_from_its_matches() {
        for (turn, open) in [(1.0, false), (0.6, true)] {
            let (matches, camera, rotations, translations) = capture(24, turn);
            let solved = solve(&matches, &camera, &Settings { open_turn: open, wide_axis: false, planar: None }).unwrap();
            assert_eq!(solved.report["closed_to_full_turn"], !open);
            assert!(solved.points.len() > 300 && solved.report["reprojection_px"]["median"].as_f64().unwrap() < 0.01, "{}", solved.report);
            // Both in the first camera's frame; the scale is free.
            let scale_of = |t: &[V3]| t.iter().map(|v| norm(*v)).sum::<f64>();
            let factor = scale_of(&translations) / scale_of(&solved.translations);
            for v in 0..24 {
                assert!(rotation_angle_deg(&solved.rotations[v], &rotations[v]) < 0.01, "view {v}");
                assert!(norm(sub(scale(solved.translations[v], factor), translations[v])) < 1e-3, "view {v}");
            }
            // The step that did not move is found as such.
            assert!(norm(sub(solved.translations[6], solved.translations[5])) < 1e-4);
        }
        let (matches, camera, _, _) = capture(6, 1.0);
        assert!(solve(&matches, &camera, &Settings { open_turn: false, wide_axis: false, planar: None }).is_err());
    }
}
