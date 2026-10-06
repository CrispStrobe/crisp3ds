//! Detection of the mat's square markers in a photo, in the photo's own
//! (distorted) frame.
//!
//! 1. Dark regions by comparing every pixel with the mean of its surroundings
//!    (integral image, two window sizes).
//! 2. A quadrilateral per region from four extreme points of its outline.
//! 3. Corners to sub-pixel accuracy: edge points along each side from the grey
//!    gradient, straight lines fitted to them after removing the lens
//!    distortion, corners where neighbouring lines meet.
//! 4. The code: cell means sampled through the quadrilateral's homography,
//!    black border checked, bits compared with every marker of the mat in the
//!    four rotations, with error correction up to the dictionary's capacity.
//!
//! Written for this crate; no detector library is used.

use crate::inputs::Plane;
use crate::photos::coarse::label;

use super::linalg::{homography, map_point, Camera};
use super::mat::Mat;

/// A decoded marker: corners top-left, top-right, bottom-right, bottom-left
/// of the printed marker, in pixels of the photo ((0, 0) is the centre of the
/// top-left pixel).
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub id: u32,
    pub corners: [[f64; 2]; 4],
    /// Code cells that had to be corrected.
    pub corrected: u32,
    /// Grey difference between the white and the black cells.
    pub contrast: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// Smallest side of a marker in pixels.
    pub minimum_side: f64,
    /// How much darker than its surroundings a marker pixel must be, in grey levels.
    pub darker_by: f64,
    /// Smallest grey difference between white and black cells.
    pub minimum_contrast: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { minimum_side: 10.0, darker_by: 12.0, minimum_contrast: 30.0 }
    }
}

fn bilinear(gray: &Plane<u8>, x: f64, y: f64) -> f64 {
    let (x, y) = (x.clamp(0.0, gray.width as f64 - 1.0), y.clamp(0.0, gray.height as f64 - 1.0));
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(gray.width - 1), (y0 + 1).min(gray.height - 1));
    let (tx, ty) = (x - x0 as f64, y - y0 as f64);
    let at = |u: usize, v: usize| gray.data[v * gray.width + u] as f64;
    (1.0 - ty) * ((1.0 - tx) * at(x0, y0) + tx * at(x1, y0)) + ty * ((1.0 - tx) * at(x0, y1) + tx * at(x1, y1))
}

/// Pixels darker than the mean of the `2 * radius + 1` square around them by `darker_by`.
fn dark_regions(gray: &Plane<u8>, radius: usize, darker_by: f64) -> Plane<u8> {
    let (w, h) = (gray.width, gray.height);
    let mut integral = vec![0u64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0u64;
        for x in 0..w {
            row += gray.data[y * w + x] as u64;
            integral[(y + 1) * (w + 1) + x + 1] = integral[y * (w + 1) + x + 1] + row;
        }
    }
    let mut out = Plane::<u8>::new(w, h);
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(radius), (y + radius + 1).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(radius), (x + radius + 1).min(w));
            let sum = integral[y1 * (w + 1) + x1] + integral[y0 * (w + 1) + x0] - integral[y0 * (w + 1) + x1] - integral[y1 * (w + 1) + x0];
            let mean = sum as f64 / ((x1 - x0) * (y1 - y0)) as f64;
            out.data[y * w + x] = ((gray.data[y * w + x] as f64) < mean - darker_by) as u8;
        }
    }
    out
}

/// Four corners of a convex region from its outline pixels: the two ends of
/// its longest diagonal and the farthest point on either side of it.
fn extreme_quad(outline: &[[f64; 2]]) -> Option<[[f64; 2]; 4]> {
    let n = outline.len() as f64;
    let centre = [outline.iter().map(|p| p[0]).sum::<f64>() / n, outline.iter().map(|p| p[1]).sum::<f64>() / n];
    let far = |from: [f64; 2]| {
        outline.iter().copied().max_by(|a, b| (a[0] - from[0]).hypot(a[1] - from[1]).total_cmp(&(b[0] - from[0]).hypot(b[1] - from[1])))
    };
    let a = far(centre)?;
    let c = far(a)?;
    let a = far(c)?;
    let side = |p: [f64; 2]| (c[0] - a[0]) * (p[1] - a[1]) - (c[1] - a[1]) * (p[0] - a[0]);
    let b = outline.iter().copied().max_by(|p, q| side(*p).total_cmp(&side(*q)))?;
    let d = outline.iter().copied().min_by(|p, q| side(*p).total_cmp(&side(*q)))?;
    if side(b) <= 0.0 || side(d) >= 0.0 {
        return None;
    }
    // With y pointing down, positive `side` is clockwise from the diagonal a -> c: a, d, c, b runs clockwise on screen.
    Some([a, d, c, b])
}

fn quad_area(q: &[[f64; 2]; 4]) -> f64 {
    0.5 * (0..4).map(|i| q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1]).sum::<f64>()
}

/// A line through points as `(point, unit direction)` by principal axis, refitted once without the worst outliers.
fn fit_line(points: &[[f64; 2]]) -> Option<([f64; 2], [f64; 2])> {
    let fit = |points: &[[f64; 2]]| -> Option<([f64; 2], [f64; 2])> {
        if points.len() < 2 {
            return None;
        }
        let n = points.len() as f64;
        let (mx, my) = (points.iter().map(|p| p[0]).sum::<f64>() / n, points.iter().map(|p| p[1]).sum::<f64>() / n);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for p in points {
            sxx += (p[0] - mx).powi(2);
            sxy += (p[0] - mx) * (p[1] - my);
            syy += (p[1] - my).powi(2);
        }
        let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        Some(([mx, my], [angle.cos(), angle.sin()]))
    };
    let (point, direction) = fit(points)?;
    let distance = |p: &[f64; 2]| ((p[0] - point[0]) * direction[1] - (p[1] - point[1]) * direction[0]).abs();
    let mut residuals: Vec<f64> = points.iter().map(distance).collect();
    residuals.sort_by(f64::total_cmp);
    let bound = (2.5 * residuals[residuals.len() / 2]).max(0.02 * residuals[residuals.len() - 1].max(1e-9));
    let kept: Vec<[f64; 2]> = points.iter().copied().filter(|p| distance(p) <= bound).collect();
    if kept.len() >= 4 && kept.len() < points.len() {
        fit(&kept)
    } else {
        Some((point, direction))
    }
}

fn intersect(a: ([f64; 2], [f64; 2]), b: ([f64; 2], [f64; 2])) -> Option<[f64; 2]> {
    let denominator = a.1[0] * b.1[1] - a.1[1] * b.1[0];
    if denominator.abs() < 1e-9 {
        return None;
    }
    let t = ((b.0[0] - a.0[0]) * b.1[1] - (b.0[1] - a.0[1]) * b.1[0]) / denominator;
    Some([a.0[0] + t * a.1[0], a.0[1] + t * a.1[1]])
}

/// Corners where the four straight sides meet. Edge points are taken where
/// the grey gradient across a side is largest and are brought into the
/// undistorted frame (`straighten`) before lines are fitted; the result is in
/// that frame as well.
fn refine(gray: &Plane<u8>, quad: &[[f64; 2]; 4], straighten: &dyn Fn([f64; 2]) -> [f64; 2]) -> Option<[[f64; 2]; 4]> {
    let mut lines = Vec::with_capacity(4);
    for side in 0..4 {
        let (p, q) = (quad[side], quad[(side + 1) % 4]);
        let length = (q[0] - p[0]).hypot(q[1] - p[1]);
        if length < 4.0 {
            return None;
        }
        let along = [(q[0] - p[0]) / length, (q[1] - p[1]) / length];
        // Clockwise corners with y down: the outside is to the left of the direction of travel on screen.
        let outward = [along[1], -along[0]];
        let reach = (0.12 * length).clamp(2.0, 12.0);
        let samples = (length / 2.0).clamp(6.0, 40.0) as usize;
        let mut points = Vec::with_capacity(samples);
        for n in 0..samples {
            let t = 0.12 + 0.76 * (n as f64 + 0.5) / samples as f64;
            let base = [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])];
            // Gradient outwards (dark inside, light outside) at half-pixel steps.
            let steps = (reach * 2.0) as i64;
            let value = |s: f64| bilinear(gray, base[0] + s * outward[0], base[1] + s * outward[1]);
            let (mut best, mut best_at, mut around) = (0.0, 0.0, [0.0; 2]);
            for step in -steps..=steps {
                let s = step as f64 * 0.5;
                let gradient = value(s + 0.75) - value(s - 0.75);
                if gradient > best {
                    (best, best_at) = (gradient, s);
                    around = [value(s + 0.25) - value(s - 1.25), value(s + 1.25) - value(s - 0.25)];
                }
            }
            if best < 4.0 {
                continue;
            }
            // Parabola through the gradient at -0.5, 0, +0.5 around the peak.
            let curvature = around[0] - 2.0 * best + around[1];
            let shift = if curvature < -1e-9 { (0.25 * (around[0] - around[1]) / curvature).clamp(-0.5, 0.5) } else { 0.0 };
            let s = best_at + shift;
            points.push(straighten([base[0] + s * outward[0], base[1] + s * outward[1]]));
        }
        if points.len() < 4 {
            return None;
        }
        lines.push(fit_line(&points)?);
    }
    let mut corners = [[0.0; 2]; 4];
    for (index, corner) in corners.iter_mut().enumerate() {
        *corner = intersect(lines[(index + 3) % 4], lines[index])?;
    }
    Some(corners)
}

/// Reads the code of a quadrilateral (corners clockwise on screen, in the
/// straightened frame; `bend` maps back into the photo) and matches it against the mat.
fn decode(
    gray: &Plane<u8>,
    corners: &[[f64; 2]; 4],
    bend: &dyn Fn([f64; 2]) -> [f64; 2],
    mat: &Mat,
    settings: &Settings,
) -> Option<(u32, usize, u32, f64)> {
    let cells = mat.bits + 2;
    let unit = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let h = homography(&unit, corners)?;
    let mut values = vec![0.0; cells * cells];
    for row in 0..cells {
        for column in 0..cells {
            let mut sum = 0.0;
            for dy in [0.3, 0.5, 0.7] {
                for dx in [0.3, 0.5, 0.7] {
                    let p = bend(map_point(&h, [(column as f64 + dx) / cells as f64, (row as f64 + dy) / cells as f64]));
                    sum += bilinear(gray, p[0], p[1]);
                }
            }
            values[row * cells + column] = sum / 9.0;
        }
    }
    let border = |row: usize, column: usize| row == 0 || column == 0 || row == cells - 1 || column == cells - 1;
    let dark: f64 = (0..cells * cells).filter(|&i| border(i / cells, i % cells)).map(|i| values[i]).sum::<f64>() / (4 * cells - 4) as f64;
    let light = values.iter().cloned().fold(f64::MIN, f64::max);
    if light - dark < settings.minimum_contrast {
        return None;
    }
    let threshold = (light + dark) / 2.0;
    let wrong_border = (0..cells * cells).filter(|&i| border(i / cells, i % cells) && values[i] > threshold).count();
    if wrong_border > cells / 2 {
        return None;
    }
    let bit = |row: usize, column: usize| values[(row + 1) * cells + column + 1] > threshold;
    let n = mat.bits;
    // The code as seen, and turned so that each corner in turn is the marker's top-left.
    let mut seen = [0u64; 4];
    for (turn, code) in seen.iter_mut().enumerate() {
        for row in 0..n {
            for column in 0..n {
                let (r, c) = match turn {
                    0 => (row, column),
                    1 => (column, n - 1 - row),
                    2 => (n - 1 - row, n - 1 - column),
                    _ => (n - 1 - column, row),
                };
                *code = (*code << 1) | bit(r, c) as u64;
            }
        }
    }
    // A dictionary of minimum distance d corrects (d - 1) / 2 cells; DICT_4X4_50 has d = 4.
    let capacity = if mat.bits == 4 { 1 } else { 2 };
    let mut best: Option<(u32, usize, u32)> = None;
    for marker in &mat.markers {
        for (turn, code) in seen.iter().enumerate() {
            let distance = (code ^ marker.code).count_ones();
            if distance <= capacity && best.is_none_or(|b| distance < b.2) {
                best = Some((marker.id, turn, distance));
            }
        }
    }
    best.map(|(id, turn, distance)| (id, turn, distance, light - dark))
}

/// All markers of `mat` found in a photo. `camera` removes the lens distortion
/// where straight lines are needed; without it the photo is taken as undistorted.
pub fn detect(gray: &Plane<u8>, mat: &Mat, camera: Option<&Camera>, settings: &Settings) -> Vec<Detection> {
    let (w, h) = (gray.width, gray.height);
    let straighten = |p: [f64; 2]| match camera {
        Some(camera) => {
            let n = camera.undistort(p);
            [camera.fx * n[0] + camera.cx, camera.fy * n[1] + camera.cy]
        }
        None => p,
    };
    let bend = |p: [f64; 2]| match camera {
        Some(camera) => camera.distort([(p[0] - camera.cx) / camera.fx, (p[1] - camera.cy) / camera.fy]),
        None => p,
    };
    let mut found: Vec<Detection> = Vec::new();
    let longest = w.max(h);
    for radius in [longest / 48, longest / 14] {
        let regions = dark_regions(gray, radius.max(4), settings.darker_by);
        let (labels, count) = label(&regions, false);
        // Outline pixels (a neighbour outside the region) and sizes per region.
        let mut outlines: Vec<Vec<[f64; 2]>> = vec![Vec::new(); count + 1];
        let mut sizes = vec![0usize; count + 1];
        for y in 0..h {
            for x in 0..w {
                let id = labels.data[y * w + x];
                if id == 0 {
                    continue;
                }
                sizes[id as usize] += 1;
                let edge = x == 0
                    || y == 0
                    || x + 1 == w
                    || y + 1 == h
                    || labels.data[y * w + x - 1] != id
                    || labels.data[y * w + x + 1] != id
                    || labels.data[(y - 1) * w + x] != id
                    || labels.data[(y + 1) * w + x] != id;
                if edge {
                    outlines[id as usize].push([x as f64, y as f64]);
                }
            }
        }
        for (outline, &size) in outlines.iter().zip(&sizes) {
            let smallest = settings.minimum_side * settings.minimum_side * 0.4;
            if (size as f64) < smallest || size > w * h / 6 || outline.len() < 12 {
                continue;
            }
            let Some(quad) = extreme_quad(outline) else { continue };
            let area = quad_area(&quad);
            // The black part of a marker covers between a good half and all of its square.
            if area < smallest || (size as f64) < 0.35 * area || (size as f64) > 1.25 * area {
                continue;
            }
            let shortest =
                (0..4).map(|i| (quad[i][0] - quad[(i + 1) % 4][0]).hypot(quad[i][1] - quad[(i + 1) % 4][1])).fold(f64::MAX, f64::min);
            if shortest < 0.35 * settings.minimum_side {
                continue;
            }
            let Some(corners) = refine(gray, &quad, &straighten) else { continue };
            if quad_area(&corners) < 0.5 * area || quad_area(&corners) > 2.0 * area {
                continue;
            }
            let Some((id, turn, corrected, contrast)) = decode(gray, &corners, &bend, mat, settings) else { continue };
            let ordered = [0, 1, 2, 3].map(|k| bend(corners[(k + turn) % 4]));
            let detection = Detection { id, corners: ordered, corrected, contrast };
            match found.iter_mut().find(|d| d.id == id) {
                Some(existing) => {
                    if (detection.corrected, -detection.contrast) < (existing.corrected, -existing.contrast) {
                        *existing = detection;
                    }
                }
                None => found.push(detection),
            }
        }
    }
    found.sort_by_key(|d| d.id);
    found
}
