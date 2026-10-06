//! Small CPU raster helpers for the photo check: 8-bit RGB images, bilinear
//! resizing and warping with OpenCV's pixel-centre convention, polygon
//! silhouettes as `cv2.fillPoly` draws them, a z-buffered flat-shaded mesh
//! view (port of `stl_compare_render.py`) and a few label glyphs.
//!
//! These produce pictures for people. They follow the reference's geometry
//! (centres, scales, crops, lighting, colours); resampling is done in float
//! where OpenCV uses fixed point, and labels use a built-in 5x7 font instead
//! of OpenCV's Hershey strokes.

#![allow(clippy::needless_range_loop)]

use std::path::Path;

use anyhow::Context;

/// Interleaved 8-bit RGB image.
#[derive(Debug, Clone, PartialEq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Rgb {
    pub fn filled(width: usize, height: usize, colour: [u8; 3]) -> Self {
        Rgb { width, height, data: colour.iter().copied().cycle().take(width * height * 3).collect() }
    }

    /// A photo as 8-bit RGB (alpha dropped), like `cv2.imread` up to channel order.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let decoded = image::open(path).with_context(|| path.display().to_string())?.to_rgb8();
        Ok(Rgb { width: decoded.width() as usize, height: decoded.height() as usize, data: decoded.into_raw() })
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        image::RgbImage::from_raw(self.width as u32, self.height as u32, self.data.clone()).expect("image size").save(path)?;
        Ok(())
    }

    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let at = (y * self.width + x) * 3;
        [self.data[at], self.data[at + 1], self.data[at + 2]]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, colour: [u8; 3]) {
        let at = (y * self.width + x) * 3;
        self.data[at..at + 3].copy_from_slice(&colour);
    }

    /// Bilinear sample at continuous pixel-index coordinates, edge pixels repeated.
    fn sample(&self, x: f64, y: f64) -> [f64; 3] {
        let (x, y) = (x.clamp(0.0, self.width as f64 - 1.0), y.clamp(0.0, self.height as f64 - 1.0));
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (tx, ty) = (x - x0 as f64, y - y0 as f64);
        let (a, b, c, d) = (self.pixel(x0, y0), self.pixel(x1, y0), self.pixel(x0, y1), self.pixel(x1, y1));
        [0, 1, 2].map(|n| (1.0 - ty) * ((1.0 - tx) * a[n] as f64 + tx * b[n] as f64) + ty * ((1.0 - tx) * c[n] as f64 + tx * d[n] as f64))
    }

    /// `cv2.resize(image, (width, height))` with the default bilinear interpolation (no antialiasing).
    pub fn resize(&self, width: usize, height: usize) -> Rgb {
        let (sx, sy) = (self.width as f64 / width as f64, self.height as f64 / height as f64);
        let mut out = Rgb::filled(width, height, [0; 3]);
        for y in 0..height {
            for x in 0..width {
                let value = self.sample((x as f64 + 0.5) * sx - 0.5, (y as f64 + 0.5) * sy - 0.5);
                out.set(x, y, value.map(|v| (v + 0.5) as u8));
            }
        }
        out
    }

    /// The rows `y0..y1` and columns `x0..x1`.
    pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Rgb {
        let mut out = Rgb::filled(x1 - x0, y1 - y0, [0; 3]);
        for y in y0..y1 {
            let from = (y * self.width + x0) * 3;
            let to = (y - y0) * out.width * 3;
            out.data[to..to + out.width * 3].copy_from_slice(&self.data[from..from + out.width * 3]);
        }
        out
    }

    /// Copies `tile` with its top-left corner at (x, y), clipped to this image.
    pub fn paste(&mut self, tile: &Rgb, x: usize, y: usize) {
        for row in 0..tile.height.min(self.height.saturating_sub(y)) {
            let count = tile.width.min(self.width.saturating_sub(x)) * 3;
            let to = ((y + row) * self.width + x) * 3;
            self.data[to..to + count].copy_from_slice(&tile.data[row * tile.width * 3..row * tile.width * 3 + count]);
        }
    }

    /// `cv2.warpAffine` for a uniform scale about `centre` into a `width` x `height`
    /// view whose middle shows `centre`: bilinear, `border` outside the photo.
    pub fn scaled_view(&self, centre: [f64; 2], scale: f64, width: usize, height: usize, border: [u8; 3]) -> Rgb {
        let mut out = Rgb::filled(width, height, border);
        for y in 0..height {
            for x in 0..width {
                let sx = (x as f64 - width as f64 / 2.0) / scale + centre[0];
                let sy = (y as f64 - height as f64 / 2.0) / scale + centre[1];
                if sx >= -0.5 && sy >= -0.5 && sx <= self.width as f64 - 0.5 && sy <= self.height as f64 - 0.5 {
                    out.set(x, y, self.sample(sx, sy).map(|v| (v + 0.5) as u8));
                }
            }
        }
        out
    }

    /// Text in the built-in font, glyph cells of 5x7 dots drawn `dot` pixels
    /// large with their baseline at `y`. Characters without a glyph are skipped.
    pub fn text(&mut self, text: &str, x: usize, y: usize, dot: usize, colour: [u8; 3]) {
        let top = y.saturating_sub(7 * dot);
        for (n, character) in text.chars().enumerate() {
            let Some(rows) = glyph(character) else { continue };
            for (row, bits) in rows.iter().enumerate() {
                for column in 0..5 {
                    if bits & (0b10000 >> column) != 0 {
                        for dy in 0..dot {
                            for dx in 0..dot {
                                let (px, py) = (x + n * 6 * dot + column * dot + dx, top + row * dot + dy);
                                if px < self.width && py < self.height {
                                    self.set(px, py, colour);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 5x7 glyphs for the labels the pipeline writes.
fn glyph(character: char) -> Option<[u8; 7]> {
    Some(match character {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'a' => [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
        'c' => [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110],
        'e' => [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
        'g' => [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110],
        'h' => [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
        'i' => [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
        'l' => [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'n' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
        'o' => [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
        'p' => [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000],
        'r' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
        's' => [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
        't' => [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
        'u' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b00100, 0b01000],
        _ => return None,
    })
}

/// `1,044,840`.
pub fn with_thousands(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (n, c) in digits.chars().enumerate() {
        if n > 0 && (digits.len() - n).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Silhouette of triangles the way `cv2.fillPoly(image, points, 255, shift=4)`
/// draws a list of them. `points` are pixel coordinates times 16, rounded
/// (three per triangle).
///
/// OpenCV draws every edge as an 8-connected line between the vertices'
/// rounded pixels and then fills the whole edge collection at once with an
/// even-odd scanline rule: edges are paired left to right on every row and the
/// span between the two edges of a pair is drawn from `round(x)` to
/// `round(x)`. On a closed mesh every edge occurs twice, so the fill adds
/// little more than the edges themselves; the same collection-wide rule is
/// applied here, so that the pixel sets agree.
pub fn fill_triangles(points: &[[[i64; 2]; 3]], width: usize, height: usize) -> Vec<u8> {
    const ONE: i64 = 1 << 16;
    struct Edge {
        y0: i64,
        y1: i64,
        x: i64,
        dx: i64,
    }
    let mut image = vec![0u8; width * height];
    let mut edges: Vec<Edge> = Vec::with_capacity(points.len() * 3);
    {
        let mut put = |x: i64, y: i64| {
            if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                image[y as usize * width + x as usize] = 255;
            }
        };
        for triangle in points {
            // x in 16.16 fixed point, y rounded to the pixel row, as CollectPolyEdges does.
            let fixed = triangle.map(|p| (p[0] << 12, (p[1] + 8) >> 4));
            for n in 0..3 {
                let (a, b) = (fixed[(n + 2) % 3], fixed[n]);
                line(((a.0 + ONE / 2) >> 16, a.1), ((b.0 + ONE / 2) >> 16, b.1), &mut put);
                if a.1 == b.1 {
                    continue;
                }
                let dx = (b.0 - a.0) / (b.1 - a.1);
                let (top, bottom) = if a.1 < b.1 { (a, b) } else { (b, a) };
                edges.push(Edge { y0: top.1, y1: bottom.1, x: top.0, dx });
            }
        }
    }
    edges.sort_by_key(|e| (e.y0, e.x, e.dx));
    let Some(first) = edges.first().map(|e| e.y0) else { return image };
    let last = edges.iter().map(|e| e.y1).max().unwrap().min(height as i64);
    let mut next = 0;
    let mut active: Vec<usize> = Vec::new();
    let mut merged: Vec<usize> = Vec::new();
    for y in first..last {
        // Drop edges that end on this row, then merge in the edges that start on it (they come sorted by x).
        active.retain(|&e| edges[e].y1 != y);
        merged.clear();
        let mut old = 0;
        while old < active.len() || (next < edges.len() && edges[next].y0 == y) {
            let starting = next < edges.len() && edges[next].y0 == y;
            if old < active.len() && (!starting || edges[active[old]].x < edges[next].x) {
                merged.push(active[old]);
                old += 1;
            } else {
                merged.push(next);
                next += 1;
            }
        }
        std::mem::swap(&mut active, &mut merged);
        for pair in active.as_chunks::<2>().0 {
            let (a, b) = (edges[pair[0]].x, edges[pair[1]].x);
            if y >= 0 {
                let (x1, x2) = ((a.min(b) + ONE / 2) >> 16, (a.max(b) + ONE / 2) >> 16);
                if x1 < width as i64 && x2 >= 0 {
                    let row = y as usize * width;
                    for x in x1.max(0)..=x2.min(width as i64 - 1) {
                        image[row + x as usize] = 255;
                    }
                }
            }
            for &e in pair {
                edges[e].x += edges[e].dx;
            }
        }
        // OpenCV re-sorts the active edges by x with a bubble sort, which is stable.
        active.sort_by_key(|&e| edges[e].x);
    }
    image
}

/// OpenCV's 8-connected line between integer pixels, drawn left to right.
fn line(mut from: (i64, i64), mut to: (i64, i64), put: &mut impl FnMut(i64, i64)) {
    if to.0 < from.0 {
        std::mem::swap(&mut from, &mut to);
    }
    let (mut dx, mut dy) = (to.0 - from.0, (to.1 - from.1).abs());
    let step_y = if to.1 < from.1 { -1 } else { 1 };
    // The major axis advances every step; the minor one when the error turns negative.
    let steep = dy > dx;
    if steep {
        std::mem::swap(&mut dx, &mut dy);
    }
    let mut error = dx - 2 * dy;
    let (mut x, mut y) = from;
    for _ in 0..=dx {
        put(x, y);
        let both = error < 0;
        error += if both { 2 * dx - 2 * dy } else { -2 * dy };
        if steep {
            y += step_y;
            x += both as i64;
        } else {
            x += 1;
            y += if both { step_y } else { 0 };
        }
    }
}

/// Z-buffered flat shading (`rasterise`): `pixels` are triangle corners in
/// pixel-index coordinates, `depth` their camera depths, `values` one grey
/// per triangle. Returns the grey per pixel, NaN where no triangle covers
/// the pixel centre.
pub fn rasterise(pixels: &[[[f64; 2]; 3]], depth: &[[f64; 3]], values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut nearest = vec![f64::INFINITY; width * height];
    let mut image = vec![f64::NAN; width * height];
    for ((corners, z), &value) in pixels.iter().zip(depth).zip(values) {
        let [a, b, c] = *corners;
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if area.abs() <= 1e-12 {
            continue;
        }
        let low = [a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize, a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize];
        let high = [a[0].max(b[0]).max(c[0]).ceil().min(width as f64 - 1.0), a[1].max(b[1]).max(c[1]).ceil().min(height as f64 - 1.0)];
        if high[0] < 0.0 || high[1] < 0.0 {
            continue;
        }
        for py in low[1]..=high[1] as usize {
            for px in low[0]..=high[0] as usize {
                let (x, y) = (px as f64, py as f64);
                let w0 = ((b[0] - x) * (c[1] - y) - (b[1] - y) * (c[0] - x)) / area;
                let w1 = ((c[0] - x) * (a[1] - y) - (c[1] - y) * (a[0] - x)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 >= -1e-6 && w1 >= -1e-6 && w2 >= -1e-6 {
                    let at = py * width + px;
                    let depth_here = w0 * z[0] + w1 * z[1] + w2 * z[2];
                    if depth_here < nearest[at] {
                        nearest[at] = depth_here;
                        image[at] = value;
                    }
                }
            }
        }
    }
    image
}

/// Camera of a view in double precision, as the check reads it from `cameras.json`.
#[derive(Debug, Clone)]
pub struct ViewCamera {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    pub k: [f64; 4],
}

impl ViewCamera {
    #[inline]
    pub fn to_camera(&self, p: [f64; 3]) -> [f64; 3] {
        let (r, t) = (&self.rotation, &self.translation);
        [0, 1, 2].map(|n| r[n][0] * p[0] + r[n][1] * p[1] + r[n][2] * p[2] + t[n])
    }
}

/// Shaded view of a mesh through one camera (`shade`): the region around
/// `centre` at `scale`, rendered at twice the size and averaged down.
pub fn shade(triangles: &[[[f32; 3]; 3]], camera: &ViewCamera, centre: [f64; 2], scale: f64, width: usize, height: usize) -> Rgb {
    const BACKGROUND: [f64; 3] = [246.0, 247.0, 249.0];
    const SURFACE: [f64; 3] = [170.0, 186.0, 204.0];
    let (w, h) = (2 * width, 2 * height);
    let light = {
        let l = [-0.35f64, -0.5, -1.0];
        let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        l.map(|v| v / n)
    };
    let (mut pixels, mut depths, mut values) = (Vec::new(), Vec::new(), Vec::new());
    for triangle in triangles {
        let cam = triangle.map(|p| camera.to_camera(p.map(|v| v as f64)));
        if cam.iter().any(|p| p[2] <= 0.0) {
            continue;
        }
        let corners = cam.map(|p| {
            let xy = [p[0] / p[2] * camera.k[0] + camera.k[2], p[1] / p[2] * camera.k[1] + camera.k[3]];
            [
                ((xy[0] - centre[0]) * scale + width as f64 / 2.0) * 2.0 - 0.5,
                ((xy[1] - centre[1]) * scale + height as f64 / 2.0) * 2.0 - 0.5,
            ]
        });
        let visible = (0..2).all(|n| {
            corners.iter().map(|c| c[n]).fold(f64::MIN, f64::max) >= 0.0
                && corners.iter().map(|c| c[n]).fold(f64::MAX, f64::min) < [w, h][n] as f64
        });
        if !visible {
            continue;
        }
        let (u, v) = ([0, 1, 2].map(|n| cam[1][n] - cam[0][n]), [0, 1, 2].map(|n| cam[2][n] - cam[0][n]));
        let normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt().max(1e-20);
        let facing = (normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]) / length;
        pixels.push(corners);
        depths.push(cam.map(|p| p[2]));
        values.push(0.30 + 0.65 * facing.max(0.0));
    }
    let grey = rasterise(&pixels, &depths, &values, w, h);
    let mut out = Rgb::filled(width, height, [0; 3]);
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0f64; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let value = grey[(2 * y + dy) * w + 2 * x + dx];
                for n in 0..3 {
                    // Each sample is truncated to 8 bits before averaging, as the reference's panel is.
                    sum[n] += if value.is_nan() { BACKGROUND[n] } else { (value * SURFACE[n]).floor().min(255.0) };
                }
            }
            out.set(x, y, sum.map(|s| (s / 4.0 + 0.5) as u8));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_follow_opencv_and_triangles_fill() {
        let mut drawn = Vec::new();
        line((0, 0), (4, 2), &mut |x, y| drawn.push((x, y)));
        assert_eq!(drawn, vec![(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)]);
        drawn.clear();
        line((3, 5), (1, 0), &mut |x, y| drawn.push((x, y)));
        assert_eq!(drawn.len(), 6);
        assert_eq!((drawn[0], drawn[5]), ((1, 0), (3, 5)));
        // A triangle with corners at pixels (2,1), (10,1), (2,9) covers its outline and interior.
        let image = fill_triangles(&[[[32, 16], [160, 16], [32, 144]]], 16, 12);
        let at = |x: usize, y: usize| image[y * 16 + x] != 0;
        assert!(at(2, 1) && at(10, 1) && at(2, 9) && at(4, 4) && at(6, 5));
        assert!(!at(9, 8) && !at(1, 1) && !at(11, 1) && !at(2, 10));
        assert_eq!(fill_triangles(&[[[-1600, 16], [-1500, 16], [-1600, 160]]], 4, 4), vec![0; 16]);
    }

    #[test]
    fn rasteriser_keeps_the_nearest_triangle() {
        let far = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let near = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let image = rasterise(&[near, far], &[[1.0; 3], [2.0; 3]], &[0.9, 0.4], 8, 8);
        assert_eq!(image[8 + 1], 0.9);
        assert_eq!(image[5 * 8 + 1], 0.4);
        assert!(image[7 * 8 + 7].is_nan());
    }

    #[test]
    fn resize_crop_paste_and_labels() {
        let mut image = Rgb::filled(4, 2, [10, 20, 30]);
        image.set(3, 1, [200, 100, 50]);
        assert_eq!(image.resize(4, 2), image);
        assert_eq!(image.resize(2, 1).pixel(0, 0), [10, 20, 30]);
        assert_eq!(image.crop(2, 1, 4, 2).pixel(1, 0), [200, 100, 50]);
        let mut sheet = Rgb::filled(6, 3, [0; 3]);
        sheet.paste(&image, 3, 2);
        assert_eq!(sheet.pixel(3, 2), [10, 20, 30]);
        assert_eq!(sheet.pixel(2, 2), [0; 3]);
        let view = image.scaled_view([2.0, 1.0], 1.0, 8, 6, [1, 2, 3]);
        assert_eq!(view.pixel(0, 0), [1, 2, 3]);
        assert_eq!(view.pixel(4, 3), [10, 20, 30]);
        let mut label = Rgb::filled(80, 20, [255; 3]);
        label.text("Input 1,044 (9)", 2, 16, 1, [0; 3]);
        assert!(label.data.iter().filter(|&&v| v == 0).count() > 100);
        assert_eq!(with_thousands(1044840), "1,044,840");
        assert_eq!(with_thousands(999), "999");
    }
}
