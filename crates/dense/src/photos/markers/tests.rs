//! Detection and poses on rendered photos of the mat, against the exact truth.

use super::detect::{detect, Settings as DetectSettings};
use super::linalg::{add, apply, norm, rotation_angle_deg, sub};
use super::mat::Mat;
use super::pose::{self, estimate_aspect, solve_views};
use super::synth::Capture;
use crate::inputs::Plane;

fn small() -> Capture {
    Capture { views: 5, width: 560, height: 420, turn: 0.8, ..Capture::default() }
}

fn photos(capture: &Capture, mat: &Mat) -> Vec<Plane<u8>> {
    (0..capture.views).map(|n| Plane { width: capture.width, height: capture.height, data: capture.render(mat, n).0 }).collect()
}

#[test]
fn markers_are_found_where_the_truth_puts_them() {
    let (capture, mat) = (small(), Mat::standard("a4").unwrap());
    let camera = capture.camera();
    let mut worst = 0f64;
    let mut total = 0;
    for (n, gray) in photos(&capture, &mat).iter().enumerate() {
        let found = detect(gray, &mat, Some(&camera), &DetectSettings::default());
        assert!(found.len() >= 10, "view {n}: {} markers", found.len());
        let truth = capture.pose(n);
        for detection in &found {
            let marker = mat.markers.iter().find(|m| m.id == detection.id).unwrap();
            for k in 0..4 {
                let expected = camera
                    .project(add(apply(&truth.rotation, [marker.corners[k][0], marker.corners[k][1], 0.0]), truth.translation))
                    .unwrap();
                let error = (expected[0] - detection.corners[k][0]).hypot(expected[1] - detection.corners[k][1]);
                assert!(error < 1.5, "view {n} marker {} corner {k}: {error:.2} px", detection.id);
                worst = worst.max(error);
            }
            total += 1;
        }
    }
    assert!(total >= 60 && worst < 1.5, "{total} markers, worst corner {worst:.2} px");
}

#[test]
fn poses_come_out_in_the_mats_millimetres() {
    let (capture, mat) = (small(), Mat::standard("a4").unwrap());
    let camera = capture.camera();
    let detections: Vec<_> =
        photos(&capture, &mat).iter().map(|gray| detect(gray, &mat, Some(&camera), &DetectSettings::default())).collect();
    let results = solve_views(&camera, &mat, &detections, &pose::Settings::default());
    for (n, result) in results.iter().enumerate() {
        let (found, truth) = (result.pose.unwrap_or_else(|| panic!("view {n}: {:?}", result.reason)), capture.pose(n));
        let angle = rotation_angle_deg(&found.rotation, &truth.rotation);
        let offset = norm(sub(found.centre(), truth.centre()));
        // No alignment between the two: the frame, the scale and the handedness are the mat's.
        assert!(angle < 0.25 && offset < 0.006 * capture.distance, "view {n}: {angle:.3} deg, {offset:.2} mm");
        assert!(result.rms_pixels < 0.6, "view {n}: {:.2} px", result.rms_pixels);
    }
    let names: Vec<String> = (0..capture.views).map(|n| format!("capture_{n:04}.png")).collect();
    let lens = crate::photos::solution::Lens {
        width: capture.width as u32,
        height: capture.height as u32,
        pixels: [camera.fx, camera.fy, camera.cx, camera.cy],
        k: camera.k,
    };
    let solution = pose::solution(&lens, &mat, &names, &results, 1.0);
    assert_eq!((solution.views.len(), solution.unregistered.len()), (capture.views, 0));
    assert_eq!(solution.scale, Some(("mm".to_string(), "markers".to_string())));
    assert!(solution.landmarks.len() >= 40 && solution.landmarks.iter().all(|l| l.position[2] == 0.0 && l.observations.len() >= 2));
    // The shared audit accepts it: every observation reprojects through the declared lens.
    let audit = crate::photos::audit::audit_solution(
        &solution,
        &crate::photos::audit::Policy { minimum_observations_per_view: 12, ..Default::default() },
    )
    .unwrap();
    assert_eq!(audit["passed"], true, "{}", audit["reasons"]);
    assert!(audit["reprojection_pixels"]["p95"].as_f64().unwrap() < 1.5);
    // A print stretched by 1 % along the page is noticed from all photos together.
    let (estimate, at_one, at_best) = estimate_aspect(&camera, &mat, &detections, &results);
    assert!((estimate - 1.0).abs() < 0.004 && at_best <= at_one, "aspect {estimate}");
    let mut stretched = mat.clone();
    for marker in &mut stretched.markers {
        for corner in &mut marker.corners {
            corner[1] /= 1.01;
        }
    }
    let results = solve_views(&camera, &stretched, &detections, &pose::Settings::default());
    let (estimate, _, _) = estimate_aspect(&camera, &stretched, &detections, &results);
    assert!((estimate - 1.01).abs() < 0.004, "aspect {estimate}");
}

#[test]
fn too_few_markers_give_no_pose_and_a_reason() {
    let (capture, mat) = (Capture { views: 1, ..small() }, Mat::standard("a4").unwrap());
    let camera = capture.camera();
    let gray = &photos(&capture, &mat)[0];
    let found = detect(gray, &mat, Some(&camera), &DetectSettings::default());
    let results = solve_views(&camera, &mat, &[found[..2].to_vec(), Vec::new()], &pose::Settings::default());
    assert!(results.iter().all(|r| r.pose.is_none()));
    assert_eq!(results[0].reason.as_deref(), Some("2 markers decoded; 3 needed"));
    assert_eq!(results[1].reason.as_deref(), Some("0 markers decoded; 3 needed"));
    // A blank photo has no markers; a mirrored photo has none either (the codes do not read backwards).
    let blank = Plane { width: 200, height: 150, data: vec![200u8; 200 * 150] };
    assert!(detect(&blank, &mat, None, &DetectSettings::default()).is_empty());
    let mut mirrored = gray.clone();
    for row in mirrored.data.chunks_mut(gray.width) {
        row.reverse();
    }
    let wrong = detect(&mirrored, &mat, None, &DetectSettings::default());
    // A mirrored code can fall within one cell of another marker's; never enough of them for a pose.
    assert!(wrong.len() < 3, "{} markers read in a mirrored photo", wrong.len());
}
