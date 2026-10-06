//! Ring sanity of the recovered cameras and the gate decision:
//! `scene_cameras`, `ring_statistics` and `decide_gates` of `photos_to_inputs.py`.

use anyhow::{anyhow, bail};
use serde_json::{json, Value};

use crate::inputs::median_f64;

use super::util::{self, python_number};

type Vector = [f64; 3];

fn sub(a: Vector, b: Vector) -> Vector {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Vector, b: Vector) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vector, b: Vector) -> Vector {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: Vector) -> f64 {
    dot(a, a).sqrt()
}

/// Singular values (descending) and right singular vectors of an N x 3 matrix
/// by one-sided Jacobi rotations, which keeps small singular values accurate.
pub fn singular_values(rows: &[Vector]) -> ([f64; 3], [Vector; 3]) {
    let mut a: Vec<Vector> = rows.to_vec();
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..60 {
        let mut rotated = false;
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
            for row in &a {
                alpha += row[p] * row[p];
                beta += row[q] * row[q];
                gamma += row[p] * row[q];
            }
            if gamma == 0.0 || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                continue;
            }
            rotated = true;
            let zeta = (beta - alpha) / (2.0 * gamma);
            let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
            let t = if zeta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = c * t;
            for row in a.iter_mut().chain(v.iter_mut()) {
                let (x, y) = (row[p], row[q]);
                row[p] = c * x - s * y;
                row[q] = s * x + c * y;
            }
        }
        if !rotated {
            break;
        }
    }
    let lengths = [0, 1, 2].map(|column| a.iter().map(|row| row[column] * row[column]).sum::<f64>().sqrt());
    let mut order = [0, 1, 2];
    order.sort_by(|&x, &y| lengths[y].total_cmp(&lengths[x]));
    // `v` holds the vectors as columns; return them as rows, largest singular value first.
    (order.map(|n| lengths[n]), order.map(|n| [v[0][n], v[1][n], v[2][n]]))
}

/// Solves a 3 x 3 linear system by elimination with partial pivoting.
fn solve3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for column in 0..3 {
        let pivot = (column..3).max_by(|&a, &b| m[a][column].abs().total_cmp(&m[b][column].abs()))?;
        if m[pivot][column] == 0.0 || !m[pivot][column].is_finite() {
            return None;
        }
        m.swap(column, pivot);
        for row in 0..3 {
            if row != column {
                let (factor, pivot_row) = (m[row][column] / m[column][column], m[column]);
                for (value, pivot_value) in m[row].iter_mut().zip(pivot_row).skip(column) {
                    *value -= factor * pivot_value;
                }
            }
        }
    }
    Some([m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]])
}

/// `np.unwrap` of angles in radians.
fn unwrap(angles: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut out = Vec::with_capacity(angles.len());
    let mut correction = 0.0;
    for (n, &angle) in angles.iter().enumerate() {
        if n > 0 {
            let step = angle - angles[n - 1];
            let mut wrapped = (step + PI).rem_euclid(2.0 * PI) - PI;
            if wrapped == -PI && step > 0.0 {
                wrapped = PI;
            }
            if step.abs() >= PI {
                correction += wrapped - step;
            }
        }
        out.push(angle + correction);
    }
    out
}

fn summary(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    let median = median_f64(&mut sorted);
    json!({"min": sorted[0], "median": median, "max": sorted[sorted.len() - 1]})
}

/// A registered view: capture position, file name, world-to-camera rotation (rows) and centre.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneCamera {
    pub index: usize,
    pub name: String,
    pub rotation: [Vector; 3],
    pub centre: Vector,
}

fn numbers<const N: usize>(value: &Value) -> anyhow::Result<[f64; N]> {
    let items = value.as_array().filter(|a| a.len() == N).ok_or_else(|| anyhow!("invalid native numeric array"))?;
    let mut out = [0.0; N];
    for (target, item) in out.iter_mut().zip(items) {
        *target = util::number(item)?;
    }
    Ok(out)
}

/// A pose's rotation, stored column by column, as world-to-camera rows.
pub fn pose_rotation(transform: &Value) -> anyhow::Result<[Vector; 3]> {
    let r: [f64; 9] = numbers(&transform["rotation"])?;
    Ok([[r[0], r[3], r[6]], [r[1], r[4], r[7]], [r[2], r[5], r[8]]])
}

pub fn pose_centre(transform: &Value) -> anyhow::Result<Vector> {
    numbers(&transform["center"])
}

/// The registered views of a scene whose photos are named `capture_NNNN.png`, in capture order.
pub fn scene_cameras(scene: &Value) -> anyhow::Result<Vec<SceneCamera>> {
    let empty = Vec::new();
    let poses: std::collections::HashMap<String, &Value> = scene
        .get("poses")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .map(|p| (util::identifier(&p["poseId"]), &p["pose"]["transform"]))
        .collect();
    let mut rows = Vec::new();
    for view in scene.get("views").and_then(Value::as_array).ok_or_else(|| anyhow!("the scene has no views"))? {
        let Some(pose) = poses.get(&util::identifier(&view["poseId"])) else { continue };
        let name = super::staging::file_name(std::path::Path::new(view["path"].as_str().unwrap_or_default()));
        let index = name
            .strip_prefix("capture_")
            .and_then(|rest| rest.strip_suffix(".png"))
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|digits| digits.parse::<usize>().ok());
        let Some(index) = index else { bail!("view is not a capture_NNNN.png photo: {name}") };
        rows.push(SceneCamera { index, name, rotation: pose_rotation(pose)?, centre: pose_centre(pose)? });
    }
    rows.sort_by_key(|row| row.index);
    Ok(rows)
}

/// How well the camera centres form one planar circle walked in capture order.
///
/// `indices` are capture positions (gaps allowed where views failed to
/// register). Photo-only: nothing here knows the true turntable angles.
pub fn ring_statistics(cameras: &[SceneCamera], duplicate_step_deg: f64) -> Value {
    let mut cameras: Vec<&SceneCamera> = cameras.iter().collect();
    cameras.sort_by_key(|camera| camera.index);
    let count = cameras.len();
    let degenerate = json!({"registered": count, "degenerate": true});
    if count < 4 {
        return degenerate;
    }
    let mut mean = [0.0; 3];
    for camera in &cameras {
        for (sum, value) in mean.iter_mut().zip(camera.centre) {
            *sum += value / count as f64;
        }
    }
    let centred: Vec<Vector> = cameras.iter().map(|camera| sub(camera.centre, mean)).collect();
    let (singular, basis) = singular_values(&centred);
    let normal = basis[2];
    let planar: Vec<[f64; 2]> = centred.iter().map(|&c| [dot(c, basis[0]), dot(c, basis[1])]).collect();
    if singular[1] < 1e-9 * singular[0].max(1e-300) || singular[1].is_nan() {
        return degenerate;
    }
    // Least-squares circle: 2 a x + 2 b y + c = x^2 + y^2, through the normal equations.
    let mut system = [[0.0; 4]; 3];
    for p in &planar {
        let row = [2.0 * p[0], 2.0 * p[1], 1.0];
        let target = p[0] * p[0] + p[1] * p[1];
        for i in 0..3 {
            for j in 0..3 {
                system[i][j] += row[i] * row[j];
            }
            system[i][3] += row[i] * target;
        }
    }
    let Some(solution) = solve3(system) else { return degenerate };
    let middle = [solution[0], solution[1]];
    let radius = (solution[2] + middle[0] * middle[0] + middle[1] * middle[1]).max(0.0).sqrt();
    if !radius.is_finite() || radius <= 0.0 {
        return degenerate;
    }
    let distance: Vec<f64> = planar.iter().map(|p| ((p[0] - middle[0]).powi(2) + (p[1] - middle[1]).powi(2)).sqrt()).collect();
    let raw: Vec<f64> = planar.iter().map(|p| (p[1] - middle[1]).atan2(p[0] - middle[0])).collect();
    let mut angle: Vec<f64> = unwrap(&raw).into_iter().map(f64::to_degrees).collect();
    let mut steps: Vec<f64> = (1..count).map(|n| (angle[n] - angle[n - 1]) / (cameras[n].index - cameras[n - 1].index) as f64).collect();
    if median_f64(&mut steps.clone()) < 0.0 {
        angle.iter_mut().for_each(|a| *a = -*a);
        steps.iter_mut().for_each(|s| *s = -*s);
    }
    let mut circular: Vec<f64> = angle.iter().map(|a| (a - angle[0]).rem_euclid(360.0)).collect();
    circular.sort_by(f64::total_cmp);
    circular.push(360.0);
    let largest_gap = circular.windows(2).map(|w| w[1] - w[0]).fold(f64::NEG_INFINITY, f64::max);
    let centre = [0, 1, 2].map(|axis| mean[axis] + middle[0] * basis[0][axis] + middle[1] * basis[1][axis]);
    let (mut miss, mut outward) = (Vec::with_capacity(count), 0);
    for camera in &cameras {
        let mut inward = sub(centre, camera.centre);
        let along = dot(inward, normal);
        inward = [inward[0] - along * normal[0], inward[1] - along * normal[1], inward[2] - along * normal[2]];
        // Each optical axis should pass close to the ring axis (any elevation is fine) and point towards it.
        let forward = camera.rotation[2];
        let across = cross(forward, normal);
        let length = norm(across);
        let offset = if length > 1e-9 { dot(inward, across).abs() / length.max(1e-9) } else { norm(inward) };
        miss.push(offset / radius * 100.0);
        outward += (dot(forward, inward) <= 0.0) as usize;
    }
    let pairs: Vec<[usize; 2]> =
        (1..count).filter(|&n| steps[n - 1].abs() < duplicate_step_deg).map(|n| [cameras[n - 1].index, cameras[n].index]).collect();
    let (nearest, farthest) = distance.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &d| (lo.min(d), hi.max(d)));
    json!({
        "registered": count, "degenerate": false, "fitted_radius": radius,
        "radius_spread_percent": (farthest - nearest) / radius * 100.0,
        "out_of_plane_percent": centred.iter().map(|&c| dot(c, normal).abs()).fold(0.0, f64::max) / radius * 100.0,
        "largest_angular_gap_deg": largest_gap, "unwrapped_sweep_deg": angle[count - 1] - angle[0],
        "step_deg_per_capture": summary(&steps),
        "reversed_steps": steps.iter().filter(|&&s| s < -duplicate_step_deg).count(),
        "duplicate_pose_pairs": pairs, "optical_axis_miss_percent": summary(&miss), "cameras_looking_outward": outward,
    })
}

/// `(passed, reasons)`. `audit` is the camera audit report, `ring` the ring statistics, `limits` the gate limits.
pub fn decide_gates(audit: &Value, ring: &Value, limits: &Value) -> (bool, Vec<String>) {
    const BEHIND: &str = "landmark behind camera";
    let audit_reasons: Vec<&str> =
        audit.get("reasons").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let fraction = audit.get("positive_depth_fraction").and_then(Value::as_f64);
    let minimum_fraction = limits["minimum_positive_depth_fraction"].as_f64().unwrap_or(1.0);
    let tolerated = fraction.is_some_and(|f| f >= minimum_fraction);
    let behind = audit_reasons.contains(&BEHIND);
    let mut reasons: Vec<String> =
        audit_reasons.iter().filter(|&&reason| reason != BEHIND).map(|reason| format!("camera audit: {reason}")).collect();
    if behind && !tolerated {
        let share = fraction.map(|f| format!("{f:.6}")).unwrap_or_else(|| "none".into());
        reasons.push(format!(
            "only {share} of the sparse observations lie in front of their camera; need {}",
            python_number(&limits["minimum_positive_depth_fraction"])
        ));
    }
    if audit.get("passed").and_then(Value::as_bool) != Some(true) && audit_reasons.is_empty() {
        reasons.push("camera audit failed".into());
    }
    if audit.get("coverage").and_then(Value::as_f64).unwrap_or(0.0) < limits["minimum_registered_fraction"].as_f64().unwrap_or(1.0) {
        let shown = |name: &str| audit.get(name).map(python_number).unwrap_or_else(|| "None".into());
        reasons.push(format!(
            "registered {} of {} photos; need a fraction of {}",
            shown("registered_cameras"),
            shown("input_images"),
            python_number(&limits["minimum_registered_fraction"])
        ));
    }
    let finish = |mut reasons: Vec<String>| {
        reasons.sort();
        reasons.dedup();
        reasons
    };
    if ring.get("degenerate").and_then(Value::as_bool).unwrap_or(true) {
        reasons.push("camera centres do not define a ring".into());
        return (false, finish(reasons));
    }
    let value = |name: &str| ring[name].as_f64().unwrap_or(f64::NAN);
    let limit = |name: &str| limits[name].as_f64().unwrap_or(f64::NAN);
    let shown = |name: &str| python_number(&limits[name]);
    if value("radius_spread_percent") > limit("maximum_radius_spread_percent") {
        let (v, l) = (value("radius_spread_percent"), shown("maximum_radius_spread_percent"));
        reasons.push(format!("ring radius spread {v:.2}% exceeds {l}%"));
    }
    if value("out_of_plane_percent") > limit("maximum_out_of_plane_percent") {
        let (v, l) = (value("out_of_plane_percent"), shown("maximum_out_of_plane_percent"));
        reasons.push(format!("camera centres leave the ring plane by {v:.2}% of the radius; limit {l}%"));
    }
    if value("largest_angular_gap_deg") > limit("maximum_angular_gap_deg") {
        let (v, l) = (value("largest_angular_gap_deg"), shown("maximum_angular_gap_deg"));
        reasons.push(format!("largest gap between neighbouring cameras is {v:.1} degrees; limit {l}"));
    }
    if value("reversed_steps") > limit("maximum_reversed_steps") {
        let (v, l) = (python_number(&ring["reversed_steps"]), shown("maximum_reversed_steps"));
        reasons.push(format!("{v} capture steps run against the turning direction; limit {l}"));
    }
    let miss = ring["optical_axis_miss_percent"]["max"].as_f64().unwrap_or(f64::NAN);
    if miss > limit("maximum_optical_axis_miss_percent") {
        let l = shown("maximum_optical_axis_miss_percent");
        reasons.push(format!("a camera's optical axis misses the ring axis by {miss:.1}% of the radius; limit {l}%"));
    }
    let outward = ring["cameras_looking_outward"].as_u64().unwrap_or(0);
    if outward > 0 {
        reasons.push(format!("{outward} cameras look away from the ring axis"));
    }
    let reasons = finish(reasons);
    (reasons.is_empty(), reasons)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn limits() -> Value {
        json!({
            "minimum_registered_fraction": 0.8, "minimum_observations_per_view": 20, "maximum_view_reprojection_p95_pixels": 4.0,
            "minimum_positive_depth_fraction": 1.0, "maximum_radius_spread_percent": 5.0, "maximum_out_of_plane_percent": 5.0,
            "maximum_angular_gap_deg": 30.0, "maximum_reversed_steps": 0, "maximum_optical_axis_miss_percent": 25.0,
            "duplicate_step_deg": 0.5,
        })
    }

    fn good_audit() -> Value {
        json!({"passed": true, "reasons": [], "coverage": 1.0, "registered_cameras": 36, "input_images": 36})
    }

    fn with(base: &Value, changes: Value) -> Value {
        let mut out = base.clone();
        out.as_object_mut().unwrap().extend(changes.as_object().unwrap().clone());
        out
    }

    /// Cameras on a circle around the y axis looking at the origin.
    pub fn orbit(angles_deg: &[f64], noise: Option<&[Vector]>) -> Vec<SceneCamera> {
        let (radius, height) = (2.0, 0.3);
        angles_deg
            .iter()
            .enumerate()
            .map(|(n, angle)| {
                let angle = angle.to_radians();
                let mut centre = [radius * angle.sin(), height, radius * angle.cos()];
                if let Some(noise) = noise {
                    centre = [centre[0] + noise[n][0], centre[1] + noise[n][1], centre[2] + noise[n][2]];
                }
                let length = norm(centre);
                let forward = centre.map(|v| -v / length);
                let right = cross([0.0, 1.0, 0.0], forward);
                let right = right.map(|v| v / norm(right));
                SceneCamera { index: n, name: format!("capture_{n:04}.png"), rotation: [right, cross(forward, right), forward], centre }
            })
            .collect()
    }

    fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
        (0..).map(|n| start + n as f64 * step).take_while(|v| if step > 0.0 { *v < stop } else { *v > stop }).collect()
    }

    fn close(value: &Value, expected: f64, tolerance: f64) -> bool {
        (value.as_f64().unwrap() - expected).abs() < tolerance
    }

    #[test]
    fn clean_full_turn_passes() {
        let ring = ring_statistics(&orbit(&arange(0.0, 360.0, 10.0), None), 0.5);
        assert_eq!(ring["degenerate"], false);
        assert!(close(&ring["fitted_radius"], 2.0, 1e-9));
        assert!(ring["radius_spread_percent"].as_f64().unwrap() < 1e-6);
        assert!(close(&ring["largest_angular_gap_deg"], 10.0, 1e-6));
        assert!(close(&ring["step_deg_per_capture"]["median"], 10.0, 1e-6));
        assert!(close(&ring["unwrapped_sweep_deg"], 350.0, 1e-6));
        // Elevated cameras looking down still hit the axis.
        assert!(ring["optical_axis_miss_percent"]["max"].as_f64().unwrap() < 1e-6);
        assert_eq!(ring["cameras_looking_outward"], 0);
        assert_eq!((ring["reversed_steps"].as_u64(), ring["duplicate_pose_pairs"].as_array().unwrap().len()), (Some(0), 0));
        assert_eq!(decide_gates(&good_audit(), &ring, &limits()), (true, vec![]));
    }

    #[test]
    fn turning_direction_does_not_matter() {
        let ring = ring_statistics(&orbit(&arange(0.0, -360.0, -10.0), None), 0.5);
        assert!(close(&ring["step_deg_per_capture"]["median"], 10.0, 1e-6));
        assert!(decide_gates(&good_audit(), &ring, &limits()).0);
    }

    #[test]
    fn duplicate_frame_is_reported_but_not_a_failure() {
        let mut angles = arange(0.0, 130.0, 10.0);
        angles.extend(arange(120.0, 360.0, 10.0)); // 120 twice
        let ring = ring_statistics(&orbit(&angles, None), 0.5);
        assert_eq!(ring["duplicate_pose_pairs"], json!([[12, 13]]));
        assert_eq!(ring["reversed_steps"], 0);
        assert!(decide_gates(&good_audit(), &ring, &limits()).0);
    }

    #[test]
    fn missing_views_use_capture_positions() {
        let cameras: Vec<_> = orbit(&arange(0.0, 360.0, 10.0), None).into_iter().filter(|c| c.index != 3 && c.index != 4).collect();
        let ring = ring_statistics(&cameras, 0.5);
        assert!(close(&ring["step_deg_per_capture"]["max"], 10.0, 1e-6));
        assert!(close(&ring["largest_angular_gap_deg"], 30.0, 1e-6));
    }

    #[test]
    fn half_turn_and_reversal_and_scatter_fail_with_reasons() {
        let half = ring_statistics(&orbit(&arange(0.0, 180.0, 5.0), None), 0.5);
        let (passed, reasons) = decide_gates(&good_audit(), &half, &limits());
        assert!(!passed);
        assert_eq!(reasons, ["largest gap between neighbouring cameras is 185.0 degrees; limit 30.0"]);
        assert!(decide_gates(&good_audit(), &half, &with(&limits(), json!({"maximum_angular_gap_deg": 360}))).0);
        let mut angles = arange(0.0, 360.0, 10.0);
        angles.swap(20, 21); // two photos swapped
        let swapped = ring_statistics(&orbit(&angles, None), 0.5);
        assert_eq!(swapped["reversed_steps"], 1);
        let reasons = decide_gates(&good_audit(), &swapped, &limits()).1;
        assert_eq!(reasons, ["1 capture steps run against the turning direction; limit 0"]);
        // Deterministic scatter of about 0.15 in every coordinate.
        let noise: Vec<Vector> =
            (0..36).map(|n| [0.2 * ((n * 7) as f64).sin(), 0.2 * ((n * 13) as f64).cos(), 0.2 * ((n * 5) as f64).sin()]).collect();
        let scattered = ring_statistics(&orbit(&arange(0.0, 360.0, 10.0), Some(&noise)), 0.5);
        let (passed, reasons) = decide_gates(&good_audit(), &scattered, &limits());
        assert!(!passed);
        assert!(reasons.iter().any(|r| r.contains("radius spread") || r.contains("ring plane")), "{reasons:?}");
    }

    #[test]
    fn outward_looking_cameras_fail() {
        let cameras = orbit(&arange(0.0, 360.0, 10.0), None);
        let flipped: Vec<_> = cameras
            .iter()
            .map(|c| SceneCamera { rotation: [c.rotation[0].map(|v| -v), c.rotation[1], c.rotation[2].map(|v| -v)], ..c.clone() })
            .collect();
        let ring = ring_statistics(&flipped, 0.5);
        assert_eq!(ring["cameras_looking_outward"], 36);
        assert_eq!(decide_gates(&good_audit(), &ring, &limits()), (false, vec!["36 cameras look away from the ring axis".to_string()]));
        // Every camera turned 20 degrees sideways: the axes miss the turntable axis.
        let yaw = 20f64.to_radians();
        let turn = [[yaw.cos(), 0.0, yaw.sin()], [0.0, 1.0, 0.0], [-yaw.sin(), 0.0, yaw.cos()]];
        let turned: Vec<_> = cameras
            .iter()
            .map(|c| {
                let rotation = [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| turn[i][k] * c.rotation[k][j]).sum::<f64>()));
                SceneCamera { rotation, ..c.clone() }
            })
            .collect();
        let ring = ring_statistics(&turned, 0.5);
        assert!(ring["optical_axis_miss_percent"]["min"].as_f64().unwrap() > 25.0);
        assert!(decide_gates(&good_audit(), &ring, &limits()).1.iter().any(|r| r.contains("misses the ring axis")));
    }

    #[test]
    fn audit_failures_and_degenerate_rings_fail() {
        let ring = ring_statistics(&orbit(&arange(0.0, 360.0, 10.0), None), 0.5);
        let audit = with(&good_audit(), json!({"passed": false, "reasons": ["per-view reprojection exceeds declared bound"]}));
        assert_eq!(
            decide_gates(&audit, &ring, &limits()),
            (false, vec!["camera audit: per-view reprojection exceeds declared bound".to_string()])
        );
        let low = with(&good_audit(), json!({"coverage": 0.5, "registered_cameras": 18}));
        assert_eq!(decide_gates(&low, &ring, &limits()).1, ["registered 18 of 36 photos; need a fraction of 0.8"]);
        assert_eq!(
            decide_gates(&with(&good_audit(), json!({"passed": false})), &ring, &limits()),
            (false, vec!["camera audit failed".to_string()])
        );
        let stray =
            with(&good_audit(), json!({"passed": false, "reasons": ["landmark behind camera"], "positive_depth_fraction": 0.99997}));
        let (passed, reasons) = decide_gates(&stray, &ring, &limits()); // a limit of 1.0 rejects one stray point
        assert!(!passed);
        assert_eq!(reasons, ["only 0.999970 of the sparse observations lie in front of their camera; need 1.0"]);
        let tolerant = with(&limits(), json!({"minimum_positive_depth_fraction": 0.999}));
        assert_eq!(decide_gates(&stray, &ring, &tolerant), (true, vec![]));
        assert!(!decide_gates(&with(&stray, json!({"positive_depth_fraction": 0.9})), &ring, &tolerant).0);
        let few = ring_statistics(&orbit(&[0.0, 10.0, 20.0], None), 0.5);
        assert_eq!(few, json!({"registered": 3, "degenerate": true}));
        assert_eq!(decide_gates(&good_audit(), &few, &limits()), (false, vec!["camera centres do not define a ring".to_string()]));
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let line: Vec<_> =
            (0..6).map(|i| SceneCamera { index: i, name: String::new(), rotation: identity, centre: [i as f64, 0.0, 0.0] }).collect();
        assert_eq!(ring_statistics(&line, 0.5)["degenerate"], true);
        // The same line in a general direction: the small singular value must not be lost to rounding.
        let line: Vec<_> = (0..6)
            .map(|i| SceneCamera {
                index: i,
                name: String::new(),
                rotation: identity,
                centre: [0.3 * i as f64, 0.7 * i as f64 + 1.0, -1.1 * i as f64],
            })
            .collect();
        assert_eq!(ring_statistics(&line, 0.5)["degenerate"], true);
    }

    #[test]
    fn scene_cameras_read_poses_in_capture_order() {
        let pose = |id: &str, centre: [f64; 3]| json!({"poseId": id, "pose": {"transform": {"rotation": ["1", "2", "3", "4", "5", "6", "7", "8", "9"], "center": centre}}});
        let scene = json!({
            "views": [{"viewId": "9", "poseId": "9", "path": "/x/capture_0010.png"}, {"viewId": "4", "poseId": "4", "path": "/x/capture_0002.png"},
                      {"viewId": "5", "poseId": "5", "path": "/x/capture_0003.png"}],
            "poses": [pose("9", [1.0, 2.0, 3.0]), pose("4", [4.0, 5.0, 6.0])],
        });
        let cameras = scene_cameras(&scene).unwrap();
        assert_eq!(cameras.iter().map(|c| c.index).collect::<Vec<_>>(), [2, 10]);
        assert_eq!(cameras[1].centre, [1.0, 2.0, 3.0]);
        // Column-major storage: the first row of the matrix is elements 0, 3, 6.
        assert_eq!(cameras[0].rotation[0], [1.0, 4.0, 7.0]);
        assert_eq!(cameras[0].rotation[2], [3.0, 6.0, 9.0]);
    }

    #[test]
    fn unwrap_follows_numpy() {
        // np.unwrap([0, 3, -3, 0.5, 6.5, -6]) == [0, 3, 3.28318531, 0.5, 0.21681469, 0.28318531]
        let out = unwrap(&[0.0, 3.0, -3.0, 0.5, 6.5, -6.0]);
        for (ours, expected) in out.iter().zip([0.0, 3.0, 3.283185307179586, 0.5, 0.21681469282041377, 0.28318530717958623]) {
            assert!((ours - expected).abs() < 1e-12, "{out:?}");
        }
    }
}
