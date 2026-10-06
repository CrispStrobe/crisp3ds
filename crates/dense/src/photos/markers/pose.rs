//! Camera poses from the mat's markers.
//!
//! The mat is rigid and its geometry is known, so every photo gives the
//! transform from the mat's frame to its camera on its own: a plane-to-image
//! homography of all visible marker corners gives a first pose, a robust
//! Levenberg-Marquardt refinement of the reprojection error in the photo
//! (lens fixed) the final one. This covers the turntable (the mat turns with
//! the object under a fixed camera) and a camera walking around a mat at
//! rest alike: in both the scene's frame is the mat's.
//!
//! A plane seen from far away has two poses that explain it almost equally
//! well (tilted towards or away from the camera). Both are refined and the
//! better one is kept; where they are close, the camera height above the mat
//! that the confident photos agree on decides.
//!
//! What all photos share is the print: a printer may scale the two page axes
//! differently. That one number can be estimated from all photos together.

use serde_json::{json, Value};

use crate::inputs::median_f64;
use crate::photos::solution::{Landmark, Lens, Solution, View};

use super::detect::Detection;
use super::linalg::{
    add, apply, cross, homography, mul, nearest_rotation, norm, rodrigues, scale, solve, sub, transpose, unit, Camera, M3, V3,
};
use super::mat::Mat;

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// A photo with fewer decoded markers gets no pose.
    pub minimum_markers: usize,
    /// Corners farther than this from their reprojection are dropped (pixels).
    pub outlier_pixels: f64,
    /// A photo whose remaining corners have a larger root-mean-square error gets no pose (pixels).
    pub maximum_rms_pixels: f64,
    /// Page height over page width as printed, relative to the design: 1 for a true print.
    pub aspect: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { minimum_markers: 3, outlier_pixels: 3.0, maximum_rms_pixels: 2.0, aspect: 1.0 }
    }
}

/// World-to-camera pose: `camera = rotation * mat + translation`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub rotation: M3,
    pub translation: V3,
}

impl Pose {
    pub fn centre(&self) -> V3 {
        scale(apply(&transpose(&self.rotation), self.translation), -1.0)
    }
}

/// A corner of the mat with where it was seen.
#[derive(Debug, Clone, Copy)]
struct Corner {
    /// Marker id * 4 + corner number.
    key: u32,
    mat: V3,
    pixel: [f64; 2],
}

fn residuals(camera: &Camera, pose: &Pose, corners: &[Corner]) -> Vec<f64> {
    let mut out = Vec::with_capacity(corners.len() * 2);
    for corner in corners {
        match camera.project(add(apply(&pose.rotation, corner.mat), pose.translation)) {
            Some(p) => out.extend([p[0] - corner.pixel[0], p[1] - corner.pixel[1]]),
            None => out.extend([1e3, 1e3]),
        }
    }
    out
}

/// Sum of Huber losses of the corner distances (pixels), the quantity the refinement lowers.
fn robust_cost(r: &[f64]) -> f64 {
    const DELTA: f64 = 1.5;
    r.as_chunks::<2>().0.iter().map(|d| d[0].hypot(d[1])).map(|e| if e <= DELTA { 0.5 * e * e } else { DELTA * (e - 0.5 * DELTA) }).sum()
}

/// Levenberg-Marquardt on the six pose parameters with Huber weights.
fn refine(camera: &Camera, start: Pose, corners: &[Corner]) -> (Pose, f64) {
    const DELTA: f64 = 1.5;
    let mut pose = start;
    let mut cost = robust_cost(&residuals(camera, &pose, corners));
    let mut damping = 1e-3;
    let moved = |pose: &Pose, d: &[f64]| Pose {
        rotation: mul(&rodrigues([d[0], d[1], d[2]]), &pose.rotation),
        translation: add(pose.translation, [d[3], d[4], d[5]]),
    };
    for _ in 0..60 {
        let r = residuals(camera, &pose, corners);
        // Numeric Jacobian; steps scaled to the size of the scene.
        let reach = norm(pose.translation).max(1.0);
        let mut jacobian = vec![[0.0; 6]; r.len()];
        for parameter in 0..6 {
            let step = if parameter < 3 { 1e-6 } else { 1e-6 * reach };
            let mut d = [0.0; 6];
            d[parameter] = step;
            let plus = residuals(camera, &moved(&pose, &d), corners);
            d[parameter] = -step;
            let minus = residuals(camera, &moved(&pose, &d), corners);
            for (row, (p, m)) in jacobian.iter_mut().zip(plus.iter().zip(&minus)) {
                row[parameter] = (p - m) / (2.0 * step);
            }
        }
        let (mut jtj, mut jtr) = (vec![vec![0.0; 6]; 6], vec![0.0; 6]);
        for (point, pair) in r.as_chunks::<2>().0.iter().enumerate() {
            let error = pair[0].hypot(pair[1]);
            let weight = if error <= DELTA { 1.0 } else { DELTA / error };
            for axis in 0..2 {
                let row = &jacobian[2 * point + axis];
                for i in 0..6 {
                    jtr[i] += weight * row[i] * pair[axis];
                    for j in 0..6 {
                        jtj[i][j] += weight * row[i] * row[j];
                    }
                }
            }
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut system = jtj.clone();
            for (i, row) in system.iter_mut().enumerate() {
                row[i] += damping * jtj[i][i].max(1e-12);
            }
            let Some(delta) = solve(system, jtr.iter().map(|v| -v).collect()) else {
                damping *= 10.0;
                continue;
            };
            let candidate = moved(&pose, &delta);
            let candidate_cost = robust_cost(&residuals(camera, &candidate, corners));
            if candidate_cost < cost {
                let gain = cost - candidate_cost;
                pose = Pose { rotation: nearest_rotation(&candidate.rotation), translation: candidate.translation };
                cost = candidate_cost;
                damping = (damping * 0.3).max(1e-9);
                improved = gain > 1e-12 * cost.max(1e-12);
                break;
            }
            damping *= 10.0;
        }
        if !improved {
            break;
        }
    }
    (pose, cost)
}

/// First pose of a plane from the homography between mat and normalised image coordinates.
fn planar_pose(camera: &Camera, corners: &[Corner]) -> Option<Pose> {
    let source: Vec<[f64; 2]> = corners.iter().map(|c| [c.mat[0], c.mat[1]]).collect();
    let target: Vec<[f64; 2]> = corners.iter().map(|c| camera.undistort(c.pixel)).collect();
    let h = homography(&source, &target)?;
    let column = |n: usize| [h[0][n], h[1][n], h[2][n]];
    let (mut h1, mut h2, mut t) = (column(0), column(1), column(2));
    let lambda = 2.0 / (norm(h1) + norm(h2));
    let sign = if t[2] < 0.0 { -lambda } else { lambda };
    (h1, h2, t) = (scale(h1, sign), scale(h2, sign), scale(t, sign));
    let h3 = cross(h1, h2);
    let rotation = nearest_rotation(&[[h1[0], h2[0], h3[0]], [h1[1], h2[1], h3[1]], [h1[2], h2[2], h3[2]]]);
    rotation.iter().flatten().chain(&t).all(|v| v.is_finite()).then_some(Pose { rotation, translation: t })
}

/// The other pose of the planar ambiguity: the plane tilted the opposite way about the line of sight.
fn mirrored(pose: &Pose, corners: &[Corner]) -> Pose {
    let n = corners.len() as f64;
    let centroid = scale(corners.iter().fold([0.0; 3], |sum, c| add(sum, c.mat)), 1.0 / n);
    let seen = add(apply(&pose.rotation, centroid), pose.translation);
    let about_sight = rodrigues(scale(unit(seen), std::f64::consts::PI));
    let about_normal = rodrigues([0.0, 0.0, std::f64::consts::PI]);
    let rotation = mul(&about_sight, &mul(&pose.rotation, &about_normal));
    Pose { rotation, translation: sub(seen, apply(&rotation, centroid)) }
}

/// What the markers of one photo gave.
#[derive(Debug, Clone)]
pub struct ViewResult {
    pub markers: usize,
    pub pose: Option<Pose>,
    /// Corners kept: `(marker id * 4 + corner, pixel)`.
    pub inliers: Vec<(u32, [f64; 2])>,
    pub rms_pixels: f64,
    /// Cost of the better pose over that of the mirrored one; near 1 means the tilt was hard to tell.
    pub ambiguity: f64,
    pub reason: Option<String>,
    /// The mirrored pose, kept while the tilt is open.
    alternative: Option<(Pose, f64)>,
}

fn corners_of(mat: &Mat, detections: &[Detection], aspect: f64) -> Vec<Corner> {
    let mut out = Vec::new();
    for detection in detections {
        let Some(marker) = mat.markers.iter().find(|m| m.id == detection.id) else { continue };
        for k in 0..4 {
            out.push(Corner {
                key: detection.id * 4 + k as u32,
                mat: [marker.corners[k][0], marker.corners[k][1] * aspect, 0.0],
                pixel: detection.corners[k],
            });
        }
    }
    out
}

fn rms(camera: &Camera, pose: &Pose, corners: &[Corner]) -> f64 {
    let r = residuals(camera, pose, corners);
    (r.iter().map(|v| v * v).sum::<f64>() / (r.len() as f64 / 2.0).max(1.0)).sqrt()
}

/// The pose of one photo from its detections.
pub fn solve_view(camera: &Camera, mat: &Mat, detections: &[Detection], settings: &Settings) -> ViewResult {
    let mut result = ViewResult {
        markers: detections.len(),
        pose: None,
        inliers: Vec::new(),
        rms_pixels: f64::NAN,
        ambiguity: f64::NAN,
        reason: None,
        alternative: None,
    };
    if detections.len() < settings.minimum_markers.max(1) {
        result.reason = Some(format!("{} markers decoded; {} needed", detections.len(), settings.minimum_markers.max(1)));
        return result;
    }
    let mut corners = corners_of(mat, detections, settings.aspect);
    let fit = |corners: &[Corner]| -> Option<((Pose, f64), (Pose, f64))> {
        let first = planar_pose(camera, corners)?;
        let a = refine(camera, first, corners);
        let b = refine(camera, mirrored(&a.0, corners), corners);
        // A mat is seen from above: its normal points towards the camera.
        let above = |p: &Pose| p.centre()[2] > 0.0;
        Some(match (above(&a.0), above(&b.0)) {
            (true, false) => (a, (b.0, f64::INFINITY)),
            (false, true) => (b, (a.0, f64::INFINITY)),
            _ if b.1 < a.1 => (b, a),
            _ => (a, b),
        })
    };
    let Some(mut best) = fit(&corners) else {
        result.reason = Some("no pose explains the marker corners".to_string());
        return result;
    };
    // Corners far from their reprojection belong to a misread marker or a damaged corner.
    for _ in 0..2 {
        let r = residuals(camera, &best.0 .0, &corners);
        let kept: Vec<Corner> = corners
            .iter()
            .zip(r.as_chunks::<2>().0)
            .filter(|(_, d)| d[0].hypot(d[1]) <= settings.outlier_pixels)
            .map(|(c, _)| *c)
            .collect();
        if kept.len() == corners.len() || kept.len() < 8 {
            break;
        }
        corners = kept;
        match fit(&corners) {
            Some(again) => best = again,
            None => break,
        }
    }
    let ((pose, cost), (other, other_cost)) = best;
    result.rms_pixels = rms(camera, &pose, &corners);
    result.ambiguity = if other_cost.is_finite() { (cost / other_cost.max(1e-12)).min(1.0) } else { 0.0 };
    let markers_left = {
        let mut ids: Vec<u32> = corners.iter().map(|c| c.key / 4).collect();
        ids.dedup();
        ids.len()
    };
    if markers_left < settings.minimum_markers.max(1) || corners.len() < 8 {
        result.reason = Some(format!("{markers_left} markers agree with one pose; {} needed", settings.minimum_markers.max(2)));
    } else if result.rms_pixels > settings.maximum_rms_pixels || result.rms_pixels.is_nan() {
        result.reason =
            Some(format!("marker corners reproject with {:.2} px RMS; limit {}", result.rms_pixels, settings.maximum_rms_pixels));
    } else if pose.centre()[2] <= 0.0 {
        result.reason = Some("the pose puts the camera below the mat".to_string());
    } else {
        result.pose = Some(pose);
        result.inliers = corners.iter().map(|c| (c.key, c.pixel)).collect();
        if other_cost.is_finite() && other.centre()[2] > 0.0 {
            result.alternative = Some((other, other_cost));
        }
    }
    result
}

fn elevation_deg(pose: &Pose) -> f64 {
    let c = pose.centre();
    (c[2] / norm(c).max(1e-300)).clamp(-1.0, 1.0).asin().to_degrees()
}

/// Solves every photo. Where the tilt of the mat was hard to tell (the two
/// poses within a factor of two in cost), the pose whose camera elevation is
/// nearer to the median of the confident photos is taken.
pub fn solve_views(camera: &Camera, mat: &Mat, detections: &[Vec<Detection>], settings: &Settings) -> Vec<ViewResult> {
    let mut results: Vec<ViewResult> = detections.iter().map(|d| solve_view(camera, mat, d, settings)).collect();
    let mut confident: Vec<f64> = results.iter().filter(|r| r.ambiguity < 0.5).filter_map(|r| r.pose.as_ref().map(elevation_deg)).collect();
    if confident.len() >= 3 {
        let median = median_f64(&mut confident);
        for (result, detected) in results.iter_mut().zip(detections) {
            let (Some(pose), Some((other, _))) = (result.pose, result.alternative) else { continue };
            if result.ambiguity >= 0.5 && (elevation_deg(&other) - median).abs() < (elevation_deg(&pose) - median).abs() {
                let corners: Vec<Corner> = corners_of(mat, detected, settings.aspect)
                    .into_iter()
                    .filter(|c| result.inliers.iter().any(|(key, _)| *key == c.key))
                    .collect();
                let other_rms = rms(camera, &other, &corners);
                if other_rms <= settings.maximum_rms_pixels {
                    result.pose = Some(other);
                    result.rms_pixels = other_rms;
                }
            }
        }
    }
    results
}

/// Total robust cost of all photos for a print whose height is `aspect` times the design's, relative to its width.
fn aspect_cost(camera: &Camera, mat: &Mat, detections: &[Vec<Detection>], results: &[ViewResult], aspect: f64) -> f64 {
    let mut total = 0.0;
    for (detected, result) in detections.iter().zip(results) {
        let Some(pose) = result.pose else { continue };
        let corners: Vec<Corner> =
            corners_of(mat, detected, aspect).into_iter().filter(|c| result.inliers.iter().any(|(key, _)| *key == c.key)).collect();
        total += refine(camera, pose, &corners).1;
    }
    total
}

/// The print's height-to-width scale that all photos together explain best
/// (golden-section search over 0.97 to 1.03), with the cost at 1 and at the optimum.
pub fn estimate_aspect(camera: &Camera, mat: &Mat, detections: &[Vec<Detection>], results: &[ViewResult]) -> (f64, f64, f64) {
    let cost = |aspect: f64| aspect_cost(camera, mat, detections, results, aspect);
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let (mut low, mut high) = (0.97, 1.03);
    let (mut a, mut b) = (high - ratio * (high - low), low + ratio * (high - low));
    let (mut fa, mut fb) = (cost(a), cost(b));
    for _ in 0..22 {
        if fa < fb {
            (high, b, fb) = (b, a, fa);
            a = high - ratio * (high - low);
            fa = cost(a);
        } else {
            (low, a, fa) = (a, b, fb);
            b = low + ratio * (high - low);
            fb = cost(b);
        }
    }
    let best = (low + high) / 2.0;
    (best, cost(1.0), cost(best))
}

/// Points of the space above the mat that every mask allows: a coarse visual
/// hull, used only to tell the dense stages where the object is. `inside(view,
/// pixel)` says whether a pixel of a registered view belongs to the object
/// (`None` beyond the photo). Falls back to a column over the object zone.
pub fn object_points(camera: &Camera, mat: &Mat, poses: &[Pose], inside: &dyn Fn(usize, [f64; 2]) -> Option<bool>) -> Vec<V3> {
    let half = 0.5 * mat.page[0].min(mat.page[1]);
    let height = 2.0 * mat.page[0].max(mat.page[1]);
    let (steps_xy, steps_z) = (40usize, 80usize);
    let allowed_misses = poses.len() / 12;
    let mut points = Vec::new();
    for k in 0..steps_z {
        let z = (k as f64 + 0.5) * height / steps_z as f64;
        for j in 0..steps_xy {
            for i in 0..steps_xy {
                let p =
                    [((i as f64 + 0.5) / steps_xy as f64 * 2.0 - 1.0) * half, ((j as f64 + 0.5) / steps_xy as f64 * 2.0 - 1.0) * half, z];
                let (mut seen, mut misses) = (0, 0);
                for (view, pose) in poses.iter().enumerate() {
                    let Some(pixel) = camera.project(add(apply(&pose.rotation, p), pose.translation)) else { continue };
                    match inside(view, pixel) {
                        Some(true) => seen += 1,
                        Some(false) => misses += 1,
                        None => {}
                    }
                    if misses > allowed_misses {
                        break;
                    }
                }
                if misses <= allowed_misses && seen >= poses.len().div_ceil(2) {
                    points.push(p);
                }
            }
        }
    }
    if points.len() < 8 {
        let r = mat.object_radius.max(10.0);
        points = (0..64).map(|n| [r * (n as f64).cos() * 0.7, r * (n as f64).sin() * 0.7, r * (n % 8) as f64 / 4.0]).collect();
    }
    points
}

/// The neutral solution of a set of photos: registered views with their mat-to-camera poses, and the
/// marker corners as landmarks with their observations. `names[i]` is the photo of `results[i]`.
pub fn solution(lens: &Lens, mat: &Mat, names: &[String], results: &[ViewResult], aspect: f64) -> Solution {
    let (mut views, mut unregistered) = (Vec::new(), Vec::new());
    let mut index_of = vec![usize::MAX; results.len()];
    for (n, (name, result)) in names.iter().zip(results).enumerate() {
        match result.pose {
            Some(pose) => {
                index_of[n] = views.len();
                views.push(View {
                    id: n.to_string(),
                    source: name.clone(),
                    rotation: pose.rotation,
                    centre: pose.centre(),
                    translation: pose.translation,
                });
            }
            None => unregistered.push(name.clone()),
        }
    }
    let mut landmarks = Vec::new();
    for marker in &mat.markers {
        for k in 0..4u32 {
            let key = marker.id * 4 + k;
            let observations: Vec<(usize, [f64; 2])> = results
                .iter()
                .enumerate()
                .filter(|(n, _)| index_of[*n] != usize::MAX)
                .filter_map(|(n, r)| r.inliers.iter().find(|(seen, _)| *seen == key).map(|(_, pixel)| (index_of[n], *pixel)))
                .collect();
            if observations.len() >= 2 {
                let corner = marker.corners[k as usize];
                landmarks.push(Landmark { position: [corner[0], corner[1] * aspect, 0.0], observations });
            }
        }
    }
    Solution {
        lens: *lens,
        views,
        unregistered,
        landmarks,
        lens_locked: Some(true),
        scale: Some(("mm".to_string(), "markers".to_string())),
        object_points: None,
    }
}

/// Per-photo numbers for the report.
pub fn report(names: &[String], results: &[ViewResult]) -> Value {
    let rows: Vec<Value> = names
        .iter()
        .zip(results)
        .map(|(name, r)| {
            json!({
                "photo": name, "markers": r.markers, "corners_used": r.inliers.len(), "registered": r.pose.is_some(),
                "rms_pixels": if r.rms_pixels.is_finite() { json!(r.rms_pixels) } else { Value::Null },
                "tilt_ambiguity": if r.ambiguity.is_finite() { json!(r.ambiguity) } else { Value::Null },
                "elevation_deg": r.pose.as_ref().map(elevation_deg), "reason": r.reason,
            })
        })
        .collect();
    let mut counts: Vec<f64> = results.iter().map(|r| r.markers as f64).collect();
    let mut errors: Vec<f64> = results.iter().filter(|r| r.pose.is_some()).map(|r| r.rms_pixels).collect();
    json!({
        "photos": rows.len(), "registered": results.iter().filter(|r| r.pose.is_some()).count(),
        "markers_per_photo": {"min": counts.iter().cloned().fold(f64::INFINITY, f64::min), "median": if counts.is_empty() { 0.0 } else { median_f64(&mut counts) }},
        "rms_pixels_median": if errors.is_empty() { Value::Null } else { json!(median_f64(&mut errors)) },
        "views": rows,
    })
}

/// Rotation and position differences between two pose lists in one frame: degrees and frame units per view.
pub fn compare(a: &[Pose], b: &[Pose]) -> Vec<(f64, f64)> {
    a.iter().zip(b).map(|(p, q)| (super::linalg::rotation_angle_deg(&p.rotation, &q.rotation), norm(sub(p.centre(), q.centre())))).collect()
}

/// Paints the visible parts of the mat's markers white in a grey photo, so that a dark-object
/// threshold does not take them for part of the object. Every marker is projected with the photo's
/// pose and compared with the photo cell by cell (its 6 x 6 cells and the ring of white paper around
/// them). A cell is painted only where it and all its neighbours look as printed: where the object
/// covers part of a marker, the covered cells and those next to them stay as they are, and the
/// object is not cut. Returns the number of markers painted at least in part.
pub fn hide_markers(gray: &mut crate::inputs::Plane<u8>, mat: &Mat, camera: &Camera, pose: &Pose) -> usize {
    let cells = mat.bits + 2;
    let grid = cells + 2; // with the ring of paper
    let (w, h) = (gray.width as f64, gray.height as f64);
    let project = |x: f64, y: f64| camera.project(add(apply(&pose.rotation, [x, y, 0.0]), pose.translation));
    let to_mat = transpose(&pose.rotation);
    let origin = pose.centre();
    let size = mat.marker_size;
    let cell = size / cells as f64;
    let mut painted = 0;
    for marker in &mat.markers {
        let [left, top] = marker.corners[0];
        // Grid cell (row, column), ring included, to mat coordinates of its centre.
        let centre_of = |row: usize, column: usize| (left + (column as f64 - 0.5) * cell, top - (row as f64 - 0.5) * cell);
        let white = |row: usize, column: usize| -> bool {
            let ring = row == 0 || column == 0 || row == grid - 1 || column == grid - 1;
            let code = (2..=mat.bits + 1).contains(&row) && (2..=mat.bits + 1).contains(&column);
            ring || (code && mat.white(marker, row - 2, column - 2))
        };
        let mut values = vec![f64::NAN; grid * grid];
        for row in 0..grid {
            for column in 0..grid {
                let (x, y) = centre_of(row, column);
                if let Some(p) = project(x, y).filter(|p| p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= w - 1.0 && p[1] <= h - 1.0) {
                    values[row * grid + column] = gray.data[p[1].round() as usize * gray.width + p[0].round() as usize] as f64;
                }
            }
        }
        let seen: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if seen.len() < grid * grid / 2 {
            continue;
        }
        let (low, high) = seen.iter().fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        if high - low < 30.0 {
            continue;
        }
        let agrees: Vec<bool> =
            (0..grid * grid).map(|i| values[i].is_finite() && (values[i] > (low + high) / 2.0) == white(i / grid, i % grid)).collect();
        let paintable: Vec<bool> = (0..grid * grid)
            .map(|i| {
                let (row, column) = ((i / grid) as i64, (i % grid) as i64);
                (-1..=1).all(|dr| {
                    (-1..=1).all(|dc| {
                        let (r, c) = (row + dr, column + dc);
                        r < 0 || c < 0 || r >= grid as i64 || c >= grid as i64 || agrees[r as usize * grid + c as usize]
                    })
                })
            })
            .collect();
        if !paintable.iter().any(|&p| p) {
            continue;
        }
        painted += 1;
        let corners = [centre_of(0, 0), centre_of(0, grid - 1), centre_of(grid - 1, grid - 1), centre_of(grid - 1, 0)];
        let outline: Vec<[f64; 2]> = corners.iter().filter_map(|c| project(c.0, c.1)).collect();
        if outline.len() < 4 {
            continue;
        }
        let bound = |axis: usize, limit: f64| -> (usize, usize) {
            let low = outline.iter().map(|p| p[axis]).fold(f64::MAX, f64::min).floor().clamp(0.0, limit);
            let high = outline.iter().map(|p| p[axis]).fold(f64::MIN, f64::max).ceil().clamp(0.0, limit);
            (low as usize, high as usize)
        };
        let ((x0, x1), (y0, y1)) = (bound(0, w - 1.0), bound(1, h - 1.0));
        for y in y0..=y1 {
            for x in x0..=x1 {
                // The pixel's ray, met with the mat's plane.
                let n = camera.undistort([x as f64, y as f64]);
                let direction = apply(&to_mat, [n[0], n[1], 1.0]);
                if direction[2].abs() < 1e-12 {
                    continue;
                }
                let t = -origin[2] / direction[2];
                let (column, row) = ((origin[0] + t * direction[0] - left) / cell + 1.0, (top - origin[1] - t * direction[1]) / cell + 1.0);
                if t > 0.0
                    && column >= 0.0
                    && row >= 0.0
                    && (column as usize) < grid
                    && (row as usize) < grid
                    && paintable[row as usize * grid + column as usize]
                {
                    gray.data[y * gray.width + x] = 255;
                }
            }
        }
    }
    painted
}

/// The grey level that separates the object from the paper in one photo, for a dark-object
/// threshold: Otsu's level of the pixels that look at the mat's object zone (the paper there, and
/// the object standing on it), so that whatever lies around the mat has no say. Also the median
/// grey of the surface just beyond the page, where the photo shows it. `None` when the zone covers
/// too few pixels.
pub fn object_level(gray: &crate::inputs::Plane<u8>, mat: &Mat, camera: &Camera, pose: &Pose) -> Option<(u32, Option<f64>)> {
    let to_mat = transpose(&pose.rotation);
    let origin = pose.centre();
    let radius = mat.object_radius.max(mat.marker_size);
    let project = |x: f64, y: f64| camera.project(add(apply(&pose.rotation, [x, y, 0.0]), pose.translation));
    // Bounding box of the zone in the photo, then every pixel's ray against the plane.
    let rim: Vec<[f64; 2]> =
        (0..32).filter_map(|k| project(radius * (k as f64 * 0.19635).cos(), radius * (k as f64 * 0.19635).sin())).collect();
    if rim.len() < 8 {
        return None;
    }
    let bound = |axis: usize, limit: usize| -> (usize, usize) {
        let low = rim.iter().map(|p| p[axis]).fold(f64::MAX, f64::min).floor().clamp(0.0, limit as f64 - 1.0);
        let high = rim.iter().map(|p| p[axis]).fold(f64::MIN, f64::max).ceil().clamp(0.0, limit as f64 - 1.0);
        (low as usize, high as usize)
    };
    let ((x0, x1), (y0, y1)) = (bound(0, gray.width), bound(1, gray.height));
    let mut inside = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let n = camera.undistort([x as f64, y as f64]);
            let direction = apply(&to_mat, [n[0], n[1], 1.0]);
            if direction[2].abs() < 1e-12 {
                continue;
            }
            let t = -origin[2] / direction[2];
            let (mx, my) = (origin[0] + t * direction[0], origin[1] + t * direction[1]);
            if t > 0.0 && mx * mx + my * my <= radius * radius {
                inside.push(gray.data[y * gray.width + x]);
            }
        }
    }
    if inside.len() < 400 {
        return None;
    }
    let level = crate::photos::coarse::otsu_threshold(inside.into_iter());
    // The surface beyond the page: a band 8 to 30 mm outside its edges.
    let (half_w, half_h) = (mat.page[0] / 2.0, mat.page[1] / 2.0);
    let mut around = Vec::new();
    for k in 0..200 {
        let (u, d) = ((k as f64 + 0.5) / 200.0 * 2.0 - 1.0, 8.0 + (k % 5) as f64 * 5.5);
        for (x, y) in [(u * half_w, half_h + d), (u * half_w, -half_h - d), (half_w + d, u * half_h), (-half_w - d, u * half_h)] {
            if let Some(p) =
                project(x, y).filter(|p| p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= gray.width as f64 - 1.0 && p[1] <= gray.height as f64 - 1.0)
            {
                around.push(gray.data[p[1].round() as usize * gray.width + p[0].round() as usize] as f64);
            }
        }
    }
    let surface = (around.len() >= 20).then(|| median_f64(&mut around));
    Some((level, surface))
}
