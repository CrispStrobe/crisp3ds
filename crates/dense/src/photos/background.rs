//! Photo-only foreground separation for a centred object against a backdrop.
//! No reference masks, camera poses, learned weights or external dependencies.
use super::coarse::{label, CoarseInfo};
use crate::inputs::Plane;
use anyhow::{ensure, Result};

fn chromatic(p: [u8; 3]) -> [f64; 3] {
    let sum = p.iter().map(|&v| v as f64).sum::<f64>() + 1.0;
    p.map(|v| v as f64 / sum)
}
fn median(v: &mut [f64]) -> f64 {
    crate::inputs::median_f64(v)
}

/// The peripheral colour mode describes the backdrop. For a uniform backdrop,
/// RGB contrast retains white/light and dark foreground alike. With large
/// peripheral lighting variation, chromatic contrast avoids lighting gradients.
/// The strongest component near the frame centre is selected. This is a
/// capture assumption, not general-purpose segmentation of arbitrary scenes.
pub fn mask(photo: &image::RgbImage, envelope: [usize; 4]) -> Result<(Plane<u8>, CoarseInfo)> {
    let (w, h) = (photo.width() as usize, photo.height() as usize);
    ensure!(w > 0 && h > 0, "empty photo");
    let [x0, y0, x1, y1] = envelope;
    let mut samples = Vec::new();
    let step = (w.max(h) / 400).max(1);
    let mut bins = vec![0usize; 1024];
    for y in (y0..y1).step_by(step) {
        for x in (x0..x1).step_by(step) {
            if x > x0 + (x1 - x0) / 3 && x < x1 - (x1 - x0) / 3 && y > y0 + (y1 - y0) / 3 && y < y1 - (y1 - y0) / 3 {
                continue;
            }
            let p = photo.get_pixel(x as u32, y as u32).0;
            if *p.iter().max().unwrap() <= 20 {
                continue;
            }
            let c = chromatic(p);
            let key = ((c[0] * 32.0) as usize).min(31) * 32 + ((c[1] * 32.0) as usize).min(31);
            bins[key] += 1;
            samples.push((p, c, key));
        }
    }
    ensure!(!samples.is_empty(), "no usable backdrop samples; choose another mask provider");
    let mode = (0..1024).max_by_key(|&k| (bins[k], std::cmp::Reverse(k))).unwrap();
    let bg: [f64; 3] = std::array::from_fn(|a| median(&mut samples.iter().filter(|s| s.2 == mode).map(|s| s.1[a]).collect::<Vec<_>>()));
    let rgb: [f64; 3] = std::array::from_fn(|a| median(&mut samples.iter().map(|s| s.0[a] as f64).collect::<Vec<_>>()));
    let spread =
        std::array::from_fn::<_, 3, _>(|a| median(&mut samples.iter().map(|s| (s.0[a] as f64 - rgb[a]).abs()).collect::<Vec<_>>()));
    let uniform = spread.iter().all(|&v| v < 8.0);
    let tolerance: [f64; 3] = spread.map(|v| (v * 6.0).max(12.0));
    // A light support surface can differ slightly from the backdrop. Learn
    // additional darker peripheral modes, rather than incorporating the entire
    // turntable into the object's silhouette. Bright modes are not removed:
    // a white foreground may be brighter than the surrounding backdrop.
    let mut rgb_bins = vec![0usize; 4096];
    for (p, _, _) in &samples {
        let k = (p[0] as usize / 16) * 256 + (p[1] as usize / 16) * 16 + p[2] as usize / 16;
        rgb_bins[k] += 1;
    }
    let mut backdrops = vec![rgb];
    if uniform {
        for (key, &count) in rgb_bins.iter().enumerate() {
            if count * 50 < samples.len() {
                continue;
            }
            let values: Vec<_> = samples
                .iter()
                .filter(|(p, _, _)| (p[0] as usize / 16) * 256 + (p[1] as usize / 16) * 16 + p[2] as usize / 16 == key)
                .collect();
            let color: [f64; 3] = std::array::from_fn(|a| median(&mut values.iter().map(|s| s.0[a] as f64).collect::<Vec<_>>()));
            if color.iter().sum::<f64>() < rgb.iter().sum::<f64>() - 24.0 && (0..3).all(|a| (color[a] - rgb[a]).abs() < 64.0) {
                backdrops.push(color);
            }
        }
    }
    // Remove only near-black padding connected to the frame. Interior dark
    // object pixels remain valid, including a black object on a light backdrop.
    let mut padding = Plane::<u8>::new(w, h);
    let mut stack = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if (x == 0 || y == 0 || x + 1 == w || y + 1 == h) && photo.get_pixel(x as u32, y as u32).0.iter().all(|&v| v <= 8) {
                let at = y * w + x;
                padding.data[at] = 1;
                stack.push(at);
            }
        }
    }
    while let Some(at) = stack.pop() {
        let (x, y) = (at % w, at / w);
        for (nx, ny) in [(x.saturating_sub(1), y), ((x + 1).min(w - 1), y), (x, y.saturating_sub(1)), (x, (y + 1).min(h - 1))] {
            let n = ny * w + nx;
            if padding.data[n] == 0 && photo.get_pixel(nx as u32, ny as u32).0.iter().all(|&v| v <= 8) {
                padding.data[n] = 1;
                stack.push(n);
            }
        }
    }
    let mut support = Plane::<u8>::new(w, h);
    for y in y0..y1 {
        for x in x0..x1 {
            let p = photo.get_pixel(x as u32, y as u32).0;
            let c = chromatic(p);
            let different = if uniform {
                backdrops.iter().all(|color| (0..3).any(|a| (p[a] as f64 - color[a]).abs() > tolerance[a]))
            } else {
                c.iter().zip(bg).map(|(a, b)| (a - b).powi(2)).sum::<f64>() > 0.12f64.powi(2)
                    || p.iter().map(|&v| v as f64).sum::<f64>() < rgb.iter().sum::<f64>() * 0.35
            };
            support.data[y * w + x] = (different && padding.data[y * w + x] == 0) as u8;
        }
    }
    if !uniform {
        // A narrow pale connector can separate two strong foreground regions
        // (e.g. a drill's silver collar and dark chuck). Grow only nearby colour
        // contrast, with a fixed short reach, rather than flooding cast shadows.
        let reach = (w.max(h) / 200).clamp(1, 12) as u16;
        let mut distance = vec![u16::MAX; w * h];
        let mut queue = Vec::new();
        for (at, &v) in support.data.iter().enumerate() {
            if v != 0 {
                distance[at] = 0;
                queue.push(at);
            }
        }
        let mut head = 0;
        while head < queue.len() {
            let at = queue[head];
            head += 1;
            if distance[at] >= reach {
                continue;
            }
            let (x, y) = (at % w, at / w);
            for (nx, ny) in [(x.saturating_sub(1), y), ((x + 1).min(w - 1), y), (x, y.saturating_sub(1)), (x, (y + 1).min(h - 1))] {
                let n = ny * w + nx;
                if nx < x0 || nx >= x1 || ny < y0 || ny >= y1 || distance[n] != u16::MAX || padding.data[n] != 0 {
                    continue;
                }
                let p = photo.get_pixel(nx as u32, ny as u32).0;
                let c = chromatic(p);
                let weak = c.iter().zip(bg).map(|(a, b)| (a - b).powi(2)).sum::<f64>() > 0.06f64.powi(2)
                    && (0..3).any(|a| (p[a] as f64 - rgb[a]).abs() > 24.0);
                if weak {
                    support.data[n] = 1;
                    distance[n] = distance[at] + 1;
                    queue.push(n);
                }
            }
        }
    }
    let (labels, count) = label(&support, true);
    ensure!(count > 0, "no foreground contrast against backdrop; choose another mask provider");
    let mut stats = vec![(0usize, 0f64, 0f64); count + 1];
    for (at, &l) in labels.data.iter().enumerate() {
        if l > 0 {
            let s = &mut stats[l as usize];
            s.0 += 1;
            s.1 += (at % w) as f64;
            s.2 += (at / w) as f64;
        }
    }
    let score = |i: usize| {
        let (area, sx, sy) = stats[i];
        let d = ((sx / area as f64 / w as f64 - 0.5).powi(2) + (sy / area as f64 / h as f64 - 0.5).powi(2)).sqrt();
        area as f64 / (1.0 + (d / 0.12).powi(4))
    };
    let best = (1..=count).max_by(|&a, &b| score(a).total_cmp(&score(b))).unwrap() as u32;
    let mut result = Plane::<u8>::new(w, h);
    for (m, &l) in result.data.iter_mut().zip(&labels.data) {
        *m = (l == best) as u8;
    }
    // Fill enclosed appearance holes (labels/printing) only. Open concavities
    // remain background because the flood can reach them from the frame.
    let mut exterior = vec![false; w * h];
    let mut queue = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if (x == 0 || y == 0 || x + 1 == w || y + 1 == h) && result.data[y * w + x] == 0 {
                exterior[y * w + x] = true;
                queue.push(y * w + x);
            }
        }
    }
    while let Some(at) = queue.pop() {
        let (x, y) = (at % w, at / w);
        for (nx, ny) in [(x.saturating_sub(1), y), ((x + 1).min(w - 1), y), (x, y.saturating_sub(1)), (x, (y + 1).min(h - 1))] {
            let n = ny * w + nx;
            if !exterior[n] && result.data[n] == 0 {
                exterior[n] = true;
                queue.push(n);
            }
        }
    }
    let mut bbox = [w, h, 0, 0];
    let mut pixels = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let at = y * w + x;
            let p = photo.get_pixel(x as u32, y as u32).0;
            let material = backdrops.iter().all(|color| (0..3).any(|a| (p[a] as f64 - color[a]).abs() > tolerance[a].min(24.0)));
            if result.data[at] != 0 || (!exterior[at] && material) {
                result.data[at] = 1;
                pixels += 1;
                bbox = [bbox[0].min(x), bbox[1].min(y), bbox[2].max(x + 1), bbox[3].max(y + 1)];
            }
        }
    }
    Ok((
        result,
        CoarseInfo {
            threshold: 0,
            foreground_pixels: pixels,
            other_dark_pixels_in_envelope: 0,
            dark_pixels_outside_envelope: 0,
            bbox_xyxy: bbox,
            touches_envelope: bbox[0] <= x0 || bbox[1] <= y0 || bbox[2] >= x1 || bbox[3] >= y1,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retains_light_parts_and_rejects_black_padding() {
        let mut p = image::RgbImage::from_pixel(100, 100, image::Rgb([210, 210, 210]));
        for y in 0..100 {
            for x in 0..100 {
                if x < 5 || y < 5 || x >= 95 || y >= 95 {
                    p.put_pixel(x, y, image::Rgb([0, 0, 0]));
                }
            }
        }
        for y in 30..70 {
            for x in 35..65 {
                p.put_pixel(x, y, image::Rgb(if y < 50 { [250, 250, 250] } else { [30, 30, 30] }));
            }
        }
        let (m, info) = mask(&p, [0, 0, 100, 100]).unwrap();
        assert_eq!(info.foreground_pixels, 1200);
        assert_eq!(m.at(50, 40), 1);
        assert_eq!(m.at(50, 60), 1);
        assert_eq!(m.at(0, 50), 0);
    }
    #[test]
    fn preserves_a_real_backdrop_hole_inside_foreground() {
        let mut p = image::RgbImage::from_pixel(100, 100, image::Rgb([210, 210, 210]));
        for y in 30..70 {
            for x in 30..70 {
                p.put_pixel(x, y, image::Rgb([30, 30, 30]));
            }
        }
        for y in 40..60 {
            for x in 40..60 {
                p.put_pixel(x, y, image::Rgb([210, 210, 210]));
            }
        }
        let (m, _) = mask(&p, [0, 0, 100, 100]).unwrap();
        assert_eq!(m.at(50, 50), 0);
        assert_eq!(m.at(35, 50), 1);
    }
    #[test]
    fn keeps_a_pale_connector_and_dark_tip_against_varied_backdrop() {
        let mut p = image::RgbImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let s = 0.7 + 0.6 * y as f64 / 100.;
                p.put_pixel(x, y, image::Rgb([(150. * s) as u8, (170. * s) as u8, (210. * s).min(255.) as u8]));
            }
        }
        for y in 40..60 {
            for x in 35..50 {
                p.put_pixel(x, y, image::Rgb([180, 130, 30]));
            }
            for x in 50..52 {
                p.put_pixel(x, y, image::Rgb([100, 100, 100]));
            }
            for x in 52..65 {
                p.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        let (m, _) = mask(&p, [0, 0, 100, 100]).unwrap();
        assert_eq!(m.at(50, 50), 1);
        assert_eq!(m.at(60, 50), 1);
        assert_eq!(m.at(5, 50), 0);
    }
    #[test]
    fn ambiguous_flat_frame_is_refused() {
        let p = image::RgbImage::from_pixel(32, 32, image::Rgb([210, 210, 210]));
        assert!(mask(&p, [0, 0, 32, 32]).is_err());
    }
}
