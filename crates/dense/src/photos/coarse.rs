//! Coarse dark-object masks: `resolve_envelope`, `otsu_threshold` and
//! `coarse_mask` of `photos_to_inputs.py`, and connected components labelled
//! like `scipy.ndimage.label`.

use anyhow::{anyhow, bail};
use serde_json::{json, Value};

use crate::inputs::{round_half_even, Plane};

/// Grey level below which a pixel is dark, or Otsu's threshold of the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Threshold {
    Level(u32),
    Otsu,
}

impl Threshold {
    /// `--dark-threshold`: a grey level 1..255 or `otsu`.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let text = text.trim().to_lowercase();
        if text == "otsu" {
            return Ok(Threshold::Otsu);
        }
        match text.parse::<u32>() {
            Ok(level) if (1..=255).contains(&level) && text.bytes().all(|b| b.is_ascii_digit()) => Ok(Threshold::Level(level)),
            _ => bail!("--dark-threshold must be a grey level 1..255 or 'otsu'"),
        }
    }

    pub fn to_json(self) -> Value {
        match self {
            Threshold::Level(level) => json!(level),
            Threshold::Otsu => json!("otsu"),
        }
    }
}

/// Search window for the object as `[x0, y0, x1, y1]`.
///
/// `auto` is the whole frame. Otherwise four comma separated numbers: all at
/// most 1 are fractions of width and height, anything else is pixels.
pub fn resolve_envelope(spec: &str, width: usize, height: usize) -> anyhow::Result<[usize; 4]> {
    if spec.trim().eq_ignore_ascii_case("auto") {
        return Ok([0, 0, width, height]);
    }
    let invalid = || anyhow!("envelope must be 'auto' or x0,y0,x1,y1");
    let mut values = Vec::new();
    for part in spec.split(',') {
        values.push(part.trim().parse::<f64>().map_err(|_| invalid())?);
    }
    if values.len() != 4 || !values.iter().all(|v| v.is_finite() && *v >= 0.0) {
        return Err(invalid());
    }
    if values.iter().all(|v| *v <= 1.0) {
        values = vec![values[0] * width as f64, values[1] * height as f64, values[2] * width as f64, values[3] * height as f64];
    }
    let rounded: Vec<usize> = values.iter().map(|v| round_half_even(*v) as usize).collect();
    let (x0, y0, x1, y1) = (rounded[0], rounded[1], rounded[2].min(width), rounded[3].min(height));
    if !(x0 < x1 && y0 < y1) {
        bail!("envelope is empty or outside the photo");
    }
    Ok([x0, y0, x1, y1])
}

/// Checks the form of an envelope before any photo is read.
pub fn validate_envelope(spec: &str) -> anyhow::Result<()> {
    resolve_envelope(spec, 1 << 40, 1 << 40).map(|_| ())
}

/// Otsu's threshold of 8-bit values: pixels strictly below the result are the dark class.
pub fn otsu_threshold(values: impl Iterator<Item = u8>) -> u32 {
    let mut histogram = [0f64; 256];
    for value in values {
        histogram[value as usize] += 1.0;
    }
    let total: f64 = histogram.iter().sum();
    let (mut cumulative, mut weighted) = ([0f64; 256], [0f64; 256]);
    let (mut count, mut sum) = (0.0, 0.0);
    for level in 0..256 {
        count += histogram[level];
        sum += histogram[level] * level as f64;
        cumulative[level] = count;
        weighted[level] = sum;
    }
    let (mut best, mut best_level) = (f64::NEG_INFINITY, 0);
    for level in 0..256 {
        let mean_dark = weighted[level] / cumulative[level];
        let mean_light = (weighted[255] - weighted[level]) / (total - cumulative[level]);
        let mut between = cumulative[level] * (total - cumulative[level]) * (mean_dark - mean_light).powi(2);
        if !between.is_finite() {
            between = -1.0;
        }
        if between > best {
            (best, best_level) = (between, level);
        }
    }
    best_level as u32 + 1
}

/// Connected components of the non-zero pixels, numbered from 1 in the order
/// their first pixel appears row by row, as `scipy.ndimage.label` numbers them.
/// `eight` joins diagonal neighbours (`structure=np.ones((3, 3))`); otherwise
/// only edge neighbours are joined (SciPy's default structure).
pub fn label(support: &Plane<u8>, eight: bool) -> (Plane<u32>, usize) {
    let (width, height) = (support.width, support.height);
    let mut labels = Plane::<u32>::new(width, height);
    let mut count = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..width * height {
        if support.data[start] == 0 || labels.data[start] != 0 {
            continue;
        }
        count += 1;
        labels.data[start] = count as u32;
        stack.push(start);
        while let Some(index) = stack.pop() {
            let (x, y) = (index % width, index / width);
            let mut visit = |nx: usize, ny: usize| {
                let neighbour = ny * width + nx;
                if support.data[neighbour] != 0 && labels.data[neighbour] == 0 {
                    labels.data[neighbour] = count as u32;
                    stack.push(neighbour);
                }
            };
            let (left, right, up, down) = (x > 0, x + 1 < width, y > 0, y + 1 < height);
            if left {
                visit(x - 1, y);
            }
            if right {
                visit(x + 1, y);
            }
            if up {
                visit(x, y - 1);
            }
            if down {
                visit(x, y + 1);
            }
            if eight {
                if left && up {
                    visit(x - 1, y - 1);
                }
                if right && up {
                    visit(x + 1, y - 1);
                }
                if left && down {
                    visit(x - 1, y + 1);
                }
                if right && down {
                    visit(x + 1, y + 1);
                }
            }
        }
    }
    (labels, count)
}

/// Numbers that `coarse_mask` reports next to the mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoarseInfo {
    pub threshold: u32,
    pub foreground_pixels: usize,
    pub other_dark_pixels_in_envelope: usize,
    pub dark_pixels_outside_envelope: usize,
    pub bbox_xyxy: [usize; 4],
    pub touches_envelope: bool,
}

/// Largest 8-connected dark component inside the envelope (0/1 mask).
pub fn coarse_mask(gray: &Plane<u8>, threshold: Threshold, envelope: [usize; 4]) -> anyhow::Result<(Plane<u8>, CoarseInfo)> {
    let (width, height) = (gray.width, gray.height);
    let [x0, y0, x1, y1] = envelope;
    if !(x0 < x1 && y0 < y1 && x1 <= width && y1 <= height) {
        bail!("envelope is empty or outside the photo");
    }
    let level = match threshold {
        Threshold::Level(level) => level,
        Threshold::Otsu => otsu_threshold((y0..y1).flat_map(|y| gray.data[y * width + x0..y * width + x1].iter().copied())),
    };
    let mut support = Plane::<u8>::new(width, height);
    let (mut inside, mut everywhere) = (0usize, 0usize);
    for y in 0..height {
        for x in 0..width {
            let dark = (gray.data[y * width + x] as u32) < level;
            everywhere += dark as usize;
            if dark && (x0..x1).contains(&x) && (y0..y1).contains(&y) {
                support.data[y * width + x] = 1;
                inside += 1;
            }
        }
    }
    let (labels, count) = label(&support, true);
    if count == 0 {
        bail!("no dark pixels inside the envelope; is the object dark on a light backdrop?");
    }
    let mut sizes = vec![0usize; count + 1];
    for &value in &labels.data {
        sizes[value as usize] += 1;
    }
    sizes[0] = 0;
    // np.argmax: the first of equally large components.
    let largest = (1..=count).fold(1, |best, n| if sizes[n] > sizes[best] { n } else { best }) as u32;
    let mut mask = Plane::<u8>::new(width, height);
    let mut bbox = [usize::MAX, usize::MAX, 0, 0];
    for y in 0..height {
        for x in 0..width {
            if labels.data[y * width + x] == largest {
                mask.data[y * width + x] = 1;
                bbox = [bbox[0].min(x), bbox[1].min(y), bbox[2].max(x + 1), bbox[3].max(y + 1)];
            }
        }
    }
    let foreground = sizes[largest as usize];
    let info = CoarseInfo {
        threshold: level,
        foreground_pixels: foreground,
        other_dark_pixels_in_envelope: inside - foreground,
        dark_pixels_outside_envelope: everywhere - inside,
        bbox_xyxy: bbox,
        touches_envelope: bbox[0] <= x0 || bbox[1] <= y0 || bbox[2] >= x1 || bbox[3] >= y1,
    };
    Ok((mask, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(width: usize, height: usize, value: u8) -> Plane<u8> {
        Plane { width, height, data: vec![value; width * height] }
    }

    fn paint(plane: &mut Plane<u8>, rows: std::ops::Range<usize>, columns: std::ops::Range<usize>, value: u8) {
        for y in rows {
            for x in columns.clone() {
                plane.data[y * plane.width + x] = value;
            }
        }
    }

    #[test]
    fn envelope_forms() {
        assert_eq!(resolve_envelope("auto", 1749, 1155).unwrap(), [0, 0, 1749, 1155]);
        assert_eq!(resolve_envelope("200,250,1520,1150", 1749, 1155).unwrap(), [200, 250, 1520, 1150]);
        assert_eq!(resolve_envelope("0.1,0.2,0.9,1", 1000, 500).unwrap(), [100, 100, 900, 500]);
        assert_eq!(resolve_envelope("10,10,5000,5000", 100, 80).unwrap(), [10, 10, 100, 80]);
        for bad in ["1,2,3", "a,b,c,d", "50,50,40,90", "-1,0,5,5"] {
            assert!(resolve_envelope(bad, 100, 80).is_err(), "{bad}");
        }
    }

    #[test]
    fn coarse_mask_keeps_largest_dark_component_inside_envelope() {
        let mut gray = filled(80, 60, 200);
        paint(&mut gray, 20..40, 30..50, 30); // object
        paint(&mut gray, 5..8, 5..8, 10); // speck inside the envelope
        paint(&mut gray, 50..60, 0..80, 0); // dark border, larger than the object
        let (mask, info) = coarse_mask(&gray, Threshold::Level(70), [0, 0, 80, 48]).unwrap();
        assert_eq!(mask.data.iter().map(|&m| m as usize).sum::<usize>(), 400);
        assert!(mask.at(35, 25) == 1 && mask.at(6, 6) == 0 && mask.at(10, 55) == 0);
        assert_eq!(info.bbox_xyxy, [30, 20, 50, 40]);
        assert_eq!(info.other_dark_pixels_in_envelope, 9);
        assert_eq!(info.dark_pixels_outside_envelope, 800);
        assert!(!info.touches_envelope);
        // Whole frame: the border wins and touches the frame, which the caller reports.
        let (mask, info) = coarse_mask(&gray, Threshold::Level(70), resolve_envelope("auto", 80, 60).unwrap()).unwrap();
        assert!(mask.at(10, 55) == 1 && info.touches_envelope);
        assert!(coarse_mask(&gray, Threshold::Level(70), [0, 0, 80, 38]).unwrap().1.touches_envelope);
        assert!(coarse_mask(&filled(10, 10, 200), Threshold::Level(70), [0, 0, 10, 10]).is_err());
    }

    #[test]
    fn threshold_is_strict_and_otsu_separates_two_levels() {
        let mut gray = filled(20, 20, 180);
        paint(&mut gray, 5..10, 5..10, 70);
        assert!(coarse_mask(&gray, Threshold::Level(70), [0, 0, 20, 20]).is_err());
        assert_eq!(coarse_mask(&gray, Threshold::Level(71), [0, 0, 20, 20]).unwrap().1.foreground_pixels, 25);
        let level = otsu_threshold(gray.data.iter().copied());
        assert!(70 < level && level <= 180);
        let (mask, info) = coarse_mask(&gray, Threshold::Otsu, [0, 0, 20, 20]).unwrap();
        assert_eq!((mask.data.iter().filter(|&&m| m == 1).count(), info.threshold), (25, level));
        // np.argmax takes the first maximum: every level from 70 to 179 separates the two classes equally well.
        assert_eq!(level, 71);
    }

    #[test]
    fn threshold_option() {
        assert_eq!(Threshold::parse("70").unwrap(), Threshold::Level(70));
        assert_eq!(Threshold::parse(" OTSU ").unwrap(), Threshold::Otsu);
        for bad in ["0", "256", "300", "-3", "7.5", "+7", ""] {
            assert!(Threshold::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn labels_follow_scipy_numbering_and_connectivity() {
        // scipy.ndimage.label of this array: default structure -> 4 components numbered
        // [[1,0,2],[0,3,0],[4,0,0]]... written out below; np.ones((3,3)) -> one component.
        let support = Plane { width: 3, height: 3, data: vec![1, 0, 1, 0, 1, 0, 1, 0, 0] };
        let (four, count) = label(&support, false);
        assert_eq!((four.data, count), (vec![1, 0, 2, 0, 3, 0, 4, 0, 0], 4));
        let (eight, count) = label(&support, true);
        assert_eq!((eight.data, count), (vec![1, 0, 1, 0, 1, 0, 1, 0, 0], 1));
        // A U shape: its two arms meet only at the bottom, and still get one label.
        let support = Plane { width: 3, height: 3, data: vec![1, 0, 1, 1, 0, 1, 1, 1, 1] };
        assert_eq!(label(&support, false).0.data, vec![1, 0, 1, 1, 0, 1, 1, 1, 1]);
    }
}
