//! Automatic prompts from the coarse dark-object masks: a port of
//! `frozen_prompts`, `foreground_distance`, `interior_positive_points` and
//! `exterior_negative_points` of `scripts/turntable_mesh/segment.py`.
//!
//! Every photo gets the same box (the union of the coarse masks' bounds, not
//! dilated) and one positive point, the pixel of its coarse mask farthest from
//! the mask's edge. With automatic cues it also gets up to four more positives
//! (one per quadrant of the mask's box) and up to eight negatives on a frame
//! around the common box.

use anyhow::bail;

use crate::inputs::Plane;

/// The prompts of one photo, in photo pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// The primary positive point (x, y): the cleaned mask must contain it.
    pub point: [usize; 2],
    /// All points, the primary one first.
    pub points: Vec<[usize; 2]>,
    /// 1: object, 0: background; one per point.
    pub labels: Vec<u8>,
    /// x0, y0, x1, y1 with exclusive upper bounds, the same for every photo.
    pub box_xyxy: [usize; 4],
}

/// What one coarse mask contributes before the common box is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Support {
    /// Bounds of the mask: x0, y0, x1, y1, upper bounds exclusive.
    pub bounds: [usize; 4],
    pub point: [usize; 2],
    /// Further positive points (automatic cues), the primary one not included.
    pub interior: Vec<[usize; 2]>,
}

/// Bounds of the non-zero pixels: x0, y0, x1, y1 with exclusive upper bounds.
pub fn bounds(support: &Plane<u8>) -> Option<[usize; 4]> {
    let (width, height) = (support.width, support.height);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..height {
        let row = &support.data[y * width..(y + 1) * width];
        let Some(first) = row.iter().position(|&v| v != 0) else { continue };
        let last = row.iter().rposition(|&v| v != 0).unwrap_or(first);
        (x0, x1) = (x0.min(first), x1.max(last + 1));
        (y0, y1) = (y0.min(y), y + 1);
    }
    (x1 > 0).then_some([x0, y0, x1, y1])
}

/// Squared Euclidean distance of every object pixel to the nearest background
/// pixel, exact (Meijster's two passes in integers), zero on the background.
/// Like the reference it looks only at the mask's box and the one-pixel ring
/// of real pixels around it, so an image edge is not background.
pub fn foreground_distance_squared(support: &Plane<u8>, bounds: [usize; 4]) -> Plane<u32> {
    let (width, height) = (support.width, support.height);
    let (x0, y0) = (bounds[0].saturating_sub(1), bounds[1].saturating_sub(1));
    let (x1, y1) = ((bounds[2] + 1).min(width), (bounds[3] + 1).min(height));
    let (m, n) = (x1 - x0, y1 - y0);
    let mut distance = Plane::<u32>::new(width, height);
    if m == 0 || n == 0 {
        return distance;
    }
    let infinite = (m + n) as i64;
    // Distance to the nearest background pixel in the same column.
    let mut column = vec![0i64; m * n];
    for x in 0..m {
        let at = |y: usize| support.data[(y + y0) * width + x + x0] != 0;
        let mut run = infinite;
        for y in 0..n {
            run = if at(y) { (run + 1).min(infinite) } else { 0 };
            column[y * m + x] = run;
        }
        for y in (0..n.saturating_sub(1)).rev() {
            let below = column[(y + 1) * m + x];
            if below + 1 < column[y * m + x] {
                column[y * m + x] = below + 1;
            }
        }
    }
    // Lower envelope of the parabolas of a row.
    let (mut s, mut t) = (vec![0i64; m], vec![0i64; m]);
    for y in 0..n {
        let g = &column[y * m..(y + 1) * m];
        let f = |x: i64, i: i64| (x - i) * (x - i) + g[i as usize] * g[i as usize];
        let mut q = 0i64;
        (s[0], t[0]) = (0, 0);
        for u in 1..m as i64 {
            while q >= 0 && f(t[q as usize], s[q as usize]) > f(t[q as usize], u) {
                q -= 1;
            }
            if q < 0 {
                q = 0;
                s[0] = u;
            } else {
                let i = s[q as usize];
                let (gu, gi) = (g[u as usize], g[i as usize]);
                let w = 1 + (u * u - i * i + gu * gu - gi * gi).div_euclid(2 * (u - i));
                if w < m as i64 {
                    q += 1;
                    (s[q as usize], t[q as usize]) = (u, w);
                }
            }
        }
        for u in (0..m as i64).rev() {
            distance.data[(y + y0) * width + u as usize + x0] = f(u, s[q as usize]) as u32;
            if u == t[q as usize] {
                q -= 1;
            }
        }
    }
    distance
}

/// The first largest value in row order inside a window, as `numpy.argmax` finds it; `keep` filters pixels.
fn farthest(distance: &Plane<u32>, window: [usize; 4], keep: impl Fn(usize, usize, u32) -> bool) -> Option<([usize; 2], u32)> {
    let mut best: Option<([usize; 2], u32)> = None;
    for y in window[1]..window[3] {
        for x in window[0]..window[2] {
            let value = distance.data[y * distance.width + x];
            if value > 0 && best.is_none_or(|(_, most)| value > most) && keep(x, y, value) {
                best = Some(([x, y], value));
            }
        }
    }
    best
}

/// Up to four more positive points, one per quadrant of the mask's box: the
/// pixel farthest from the edge there, if it is at least `max(2, a quarter of
/// the largest distance)` inside and at least `max(2, a fifth of the smaller
/// box side)` away from the primary point.
fn interior_positive_points(distance: &Plane<u32>, bounds: [usize; 4], primary: [usize; 2], largest: u32) -> Vec<[usize; 2]> {
    let [x0, y0, x1, y1] = bounds;
    let (mx, my) = ((x0 + x1) / 2, (y0 + y1) / 2);
    let clearance = (0.25 * (largest as f64).sqrt()).max(2.0);
    let separation = (0.2 * (x1 - x0).min(y1 - y0) as f64).max(2.0);
    let mut points: Vec<[usize; 2]> = Vec::new();
    for window in [[x0, y0, mx, my], [mx, y0, x1, my], [x0, my, mx, y1], [mx, my, x1, y1]] {
        let found = farthest(distance, window, |x, y, value| {
            let (dx, dy) = (x as f64 - primary[0] as f64, y as f64 - primary[1] as f64);
            (value as f64).sqrt() >= clearance && dx * dx + dy * dy >= separation * separation
        });
        if let Some((point, _)) = found {
            if point != primary && !points.contains(&point) {
                points.push(point);
            }
        }
    }
    points
}

/// Bounds, primary point and (with `automatic_cues`) the further positives of one coarse mask (0 / non-zero).
pub fn support(mask: &Plane<u8>, automatic_cues: bool) -> anyhow::Result<Support> {
    let Some(bounds) = bounds(mask) else { bail!("coarse mask is empty") };
    if mask.data.iter().all(|&v| v != 0) {
        bail!("coarse mask covers the whole photo");
    }
    let distance = foreground_distance_squared(mask, bounds);
    let Some((point, largest)) = farthest(&distance, bounds, |_, _, _| true) else { bail!("coarse mask is empty") };
    let interior = if automatic_cues { interior_positive_points(&distance, bounds, point, largest) } else { Vec::new() };
    Ok(Support { bounds, point, interior })
}

/// Background points on a frame around the box, `max(24, a fifth of its
/// longer side)` pixels out: the middle of each side, then the corners; those
/// outside the photo are left out.
pub fn exterior_negative_points(box_xyxy: [usize; 4], width: usize, height: usize) -> Vec<[usize; 2]> {
    let [x0, y0, x1, y1] = box_xyxy.map(|v| v as i64);
    let margin = ((0.2 * (x1 - x0).max(y1 - y0) as f64).ceil() as i64).max(24);
    let (left, top, right, bottom) = (x0 - margin - 1, y0 - margin - 1, x1 + margin, y1 + margin);
    let (mx, my) = ((x0 + x1 - 1).div_euclid(2), (y0 + y1 - 1).div_euclid(2));
    [[mx, top], [right, my], [mx, bottom], [left, my], [left, top], [right, top], [left, bottom], [right, bottom]]
        .into_iter()
        .filter(|&[x, y]| (0..width as i64).contains(&x) && (0..height as i64).contains(&y))
        .map(|[x, y]| [x as usize, y as usize])
        .collect()
}

/// The prompts of all photos from what their coarse masks contribute.
pub fn frozen_prompts(supports: &[Support], width: usize, height: usize, automatic_cues: bool) -> Vec<Prompt> {
    let common = supports.iter().fold([usize::MAX, usize::MAX, 0, 0], |b, s| {
        [b[0].min(s.bounds[0]), b[1].min(s.bounds[1]), b[2].max(s.bounds[2]), b[3].max(s.bounds[3])]
    });
    let negatives = if automatic_cues { exterior_negative_points(common, width, height) } else { Vec::new() };
    supports
        .iter()
        .map(|support| {
            let mut points = vec![support.point];
            points.extend(&support.interior);
            let mut labels = vec![1u8; points.len()];
            points.extend(&negatives);
            labels.resize(points.len(), 0);
            Prompt { point: support.point, points, labels, box_xyxy: common }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(rows: &[&str]) -> Plane<u8> {
        let width = rows[0].len();
        Plane { width, height: rows.len(), data: rows.iter().flat_map(|r| r.bytes().map(|b| (b == b'#') as u8)).collect() }
    }

    /// Squared distance by trying every background pixel of the same window.
    fn brute(support: &Plane<u8>, bounds: [usize; 4]) -> Vec<u32> {
        let (width, height) = (support.width, support.height);
        let (x0, y0) = (bounds[0].saturating_sub(1), bounds[1].saturating_sub(1));
        let (x1, y1) = ((bounds[2] + 1).min(width), (bounds[3] + 1).min(height));
        let mut out = vec![0u32; width * height];
        for y in y0..y1 {
            for x in x0..x1 {
                if support.data[y * width + x] == 0 {
                    continue;
                }
                let mut best = u32::MAX;
                for v in y0..y1 {
                    for u in x0..x1 {
                        if support.data[v * width + u] == 0 {
                            best = best.min(((x as i64 - u as i64).pow(2) + (y as i64 - v as i64).pow(2)) as u32);
                        }
                    }
                }
                out[y * width + x] = best;
            }
        }
        out
    }

    #[test]
    fn distance_is_exact_and_an_image_edge_is_not_background() {
        let mask = plane(&["........", ".#####..", ".#####..", ".##.##..", ".#####..", "........"]);
        let b = bounds(&mask).unwrap();
        assert_eq!(b, [1, 1, 6, 5]);
        assert_eq!(foreground_distance_squared(&mask, b).data, brute(&mask, b));
        // The object touches the left and the lower edge: distances grow towards them.
        let edge = plane(&["....", "##..", "###.", "###."]);
        let b = bounds(&edge).unwrap();
        let d = foreground_distance_squared(&edge, b);
        assert_eq!(d.data, brute(&edge, b));
        assert_eq!(d.data[3 * 4], 8);
        // Pseudo-random blobs against the brute-force answer.
        let mut state = 12345u32;
        for _ in 0..20 {
            let (width, height) = (23, 17);
            let mut data = vec![0u8; width * height];
            for value in data.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *value = (state >> 24 > 60) as u8;
            }
            let blob = Plane { width, height, data };
            let b = bounds(&blob).unwrap();
            assert_eq!(foreground_distance_squared(&blob, b).data, brute(&blob, b));
        }
    }

    #[test]
    fn prompts_follow_the_reference_recipe() {
        // A 40 x 30 photo with a 20 x 12 block at (10, 8): values checked against segment.py.
        let (width, height) = (40, 30);
        let mut mask = Plane::<u8>::new(width, height);
        for y in 8..20 {
            for x in 10..30 {
                mask.data[y * width + x] = 1;
            }
        }
        let one = support(&mask, true).unwrap();
        assert_eq!(one.bounds, [10, 8, 30, 20]);
        // The first pixel six from the edge in row order.
        assert_eq!(one.point, [15, 13]);
        // Clearance max(2, 6/4) = 2, separation max(2, 0.2*12) = 2.4: per quadrant the first farthest pixel that far away.
        assert_eq!(one.interior, vec![[18, 13], [20, 13], [18, 14], [20, 14]]);
        // An L with a hole: one quadrant is empty, another repeats nothing.
        let (w, h) = (120, 90);
        let mut shape = Plane::<u8>::new(w, h);
        for y in 10..80 {
            for x in 20..100 {
                shape.data[y * w + x] = ((x < 50 || y >= 50) && !((60..66).contains(&y) && (30..36).contains(&x))) as u8;
            }
        }
        let l = support(&shape, true).unwrap();
        assert_eq!((l.point, l.interior.clone()), ([34, 24], vec![[34, 38], [34, 45], [60, 64]]));
        assert!(exterior_negative_points(l.bounds, w, h).is_empty());
        let prompts = frozen_prompts(std::slice::from_ref(&one), width, height, true);
        assert_eq!(prompts[0].box_xyxy, [10, 8, 30, 20]);
        // Margin max(24, ceil(0.2 * 20)) = 24: every negative falls outside this small photo.
        assert_eq!(prompts[0].points.len(), 1 + one.interior.len());
        assert!(prompts[0].labels.iter().all(|&l| l == 1));
        assert_eq!(
            exterior_negative_points([100, 80, 300, 200], 640, 480),
            vec![[199, 39], [340, 139], [199, 240], [59, 139], [59, 39], [340, 39], [59, 240], [340, 240]]
        );
        assert_eq!(exterior_negative_points([100, 30, 300, 200], 640, 480).len(), 5);
        // Without cues: one point, the box of all photos.
        let mut other = Plane::<u8>::new(width, height);
        other.data[2 * width + 3] = 1;
        let two = support(&other, false).unwrap();
        let prompts = frozen_prompts(&[support(&mask, false).unwrap(), two], width, height, false);
        assert_eq!((prompts[1].points.clone(), prompts[1].box_xyxy), (vec![[3, 2]], [3, 2, 30, 20]));
        assert!(support(&Plane::<u8>::new(4, 4), true).is_err());
    }
}
