//! Contrast images for feature detection: `gamma_table` and `contrast_image`
//! of `photos_to_inputs.py` (gamma, then CLAHE on the Lab lightness).
//!
//! The reference calls OpenCV: `cvtColor(RGB2LAB)`, `createCLAHE().apply` and
//! `cvtColor(LAB2RGB)` on 8-bit data. All three are integer or single-precision
//! algorithms and are reimplemented here to give the same bytes:
//!
//! - RGB to Lab: OpenCV's `RGB2Lab_b` (sRGB gamma table at 3 extra bits, cube
//!   root table at 15 bits, D65 matrix at 12 bits);
//! - CLAHE: histogram per tile of the image padded to a multiple of the grid
//!   (`BORDER_REFLECT_101`), clipping with the excess spread evenly and its
//!   remainder at regular bin steps, cumulative look-up tables scaled in
//!   `float`, bilinear blending of the four surrounding tiles in `float`;
//! - Lab to RGB: OpenCV's `Lab2RGBinteger` (14-bit lightness and a/b tables,
//!   12-bit inverse gamma table).
//!
//! `tests/fixtures/dense-native/opencv-contrast.json` holds values produced by
//! OpenCV 4.10 (`make_photos_fixtures.py`); the conversions were also compared
//! with it for all 2^24 colours.

use std::sync::OnceLock;

use crate::inputs::round_half_even;

/// 8-bit look-up table of a gamma curve; 0.5 is the square root used for dark objects.
pub fn gamma_table(gamma: f64) -> [u8; 256] {
    let mut table = [0u8; 256];
    for (value, entry) in table.iter_mut().enumerate() {
        let x = value as f64 / 255.0;
        let curve = if gamma == 0.5 { x.sqrt() } else { x.powf(gamma) };
        *entry = round_half_even(255.0 * curve).clamp(0.0, 255.0) as u8;
    }
    table
}

const LAB_SHIFT: u32 = 12;
const GAMMA_SHIFT: u32 = 3;
const LAB_SHIFT2: u32 = LAB_SHIFT + GAMMA_SHIFT;
/// Entries of the cube root table that 8-bit input can reach (the matrix rows sum to one).
const CBRT_SIZE: usize = 255 * (1 << GAMMA_SHIFT) + 1;
const BASE: i64 = 1 << 14;
const MINIMUM_AB: i64 = -8145;
const AB_SIZE: usize = (BASE * 9 / 4) as usize;
const INVERSE_GAMMA_SIZE: usize = 1 << 12;

/// D65 sRGB to XYZ, each row divided by its white point component, at 12 bits.
const TO_XYZ: [[i64; 3]; 3] = [[1777, 1541, 778], [871, 2929, 296], [73, 448, 3575]];
/// XYZ to sRGB, each column multiplied by its white point component, at 12 bits.
const FROM_XYZ: [[i64; 3]; 3] = [[12615, -6296, -2223], [-3773, 7684, 185], [217, -836, 4715]];

pub struct LabTables {
    /// sRGB decoding at 255 * 8 levels.
    pub gamma: [u16; 256],
    /// The Lab transfer function `f(t)` at 15 bits, for `t` in steps of 1 / (255 * 8).
    pub cbrt: Vec<u16>,
    /// Per 8-bit lightness: `Y` and `f(Y)` at 14 bits.
    pub lightness: [[i64; 2]; 256],
    /// Inverse of `f` at 14 bits, indexed from `MINIMUM_AB`.
    pub ab: Vec<i32>,
    /// sRGB encoding of a 12-bit linear value.
    pub inverse_gamma: Vec<u8>,
}

/// OpenCV's `cv::cbrt` for single precision: a rational approximation on the
/// mantissa evaluated in double precision. Only +, * and / are used, so the
/// result does not depend on the platform's maths library.
fn opencv_cbrt(value: f32) -> f32 {
    let bits = value.to_bits() as i32;
    let ix = bits & 0x7fff_ffff;
    if ix == 0 {
        return 0.0;
    }
    let exponent = ((bits >> 23) & 255) - 127;
    let mut shift = exponent % 3;
    if shift >= 0 {
        shift -= 3;
    }
    let exponent = (exponent - shift) / 3;
    let fraction = f32::from_bits(((ix & ((1 << 23) - 1)) | ((shift + 127) << 23)) as u32) as f64;
    let numerator = (((45.254_833_975_680_3 * fraction + 192.279_836_835_506_1) * fraction + 119.165_482_428_558_16) * fraction
        + 13.432_501_390_862_399)
        * fraction
        + 0.163_616_122_658_575_42;
    let denominator = (((14.808_840_932_191_346 * fraction + 151.971_405_104_443_56) * fraction + 168.525_441_410_156_83) * fraction
        + 33.990_594_135_021_56)
        * fraction
        + 1.0;
    let root = (numerator / denominator) as f32;
    f32::from_bits((root.to_bits() as i32 + exponent * (1 << 23)) as u32)
}

fn build_tables() -> LabTables {
    let mut gamma = [0u16; 256];
    for (value, entry) in gamma.iter_mut().enumerate() {
        let x = value as f64 / 255.0;
        let linear = if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) };
        *entry = round_half_even(255.0 * (1 << GAMMA_SHIFT) as f64 * linear) as u16;
    }
    let (threshold, slope, offset) = (216f32 / 24389f32, 841f32 / 108f32, 16f32 / 116f32);
    let mut cbrt: Vec<u16> = (0..CBRT_SIZE)
        .map(|index| {
            let x = index as f32 / (255 * (1 << GAMMA_SHIFT)) as f32;
            let value = if x < threshold { x * slope + offset } else { opencv_cbrt(x) };
            round_half_even(((1 << LAB_SHIFT2) as f32 * value) as f64) as u16
        })
        .collect();
    // The only entry where this single-precision evaluation lands on a tie that OpenCV 4.10
    // resolves the other way (found by comparing all 2^24 colours).
    cbrt[324] = 17745;
    let mut lightness = [[0i64; 2]; 256];
    for (index, entry) in lightness.iter_mut().enumerate() {
        let i = index as i64;
        if index <= 20 {
            // Below L* = 8: Y = L / 903.3 and f(Y) = 7.787 Y + 16 / 116.
            let y = round_half_even(((i * BASE * 20 * 9) as f32 / (17 * 29 * 29 * 29) as f32) as f64) as i64;
            let fy = round_half_even((BASE as f32 * (16f32 / 116f32 + (i * 5) as f32 / (3 * 17 * 29) as f32)) as f64) as i64;
            *entry = [y, fy];
        } else {
            let fy = (i * 100 * BASE) as f32 / (255 * 116) as f32 + (16 * BASE) as f32 / 116f32;
            let y = round_half_even((fy * fy * fy / (BASE * BASE) as f32) as f64) as i64;
            *entry = [y, round_half_even(fy as f64) as i64];
        }
    }
    let ab = (0..AB_SIZE as i64)
        .map(|index| {
            let i = index + MINIMUM_AB;
            // C integer arithmetic: division truncates towards zero.
            let value = if i <= 3390 { i * 108 / 841 - BASE * 16 / 116 * 108 / 841 } else { i * i / BASE * i / BASE };
            value as i32
        })
        .collect();
    let inverse_gamma = (0..INVERSE_GAMMA_SIZE)
        .map(|index| {
            let x = index as f64 / INVERSE_GAMMA_SIZE as f64;
            let encoded = if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
            round_half_even(255.0 * encoded) as u8
        })
        .collect();
    LabTables { gamma, cbrt, lightness, ab, inverse_gamma }
}

pub fn tables() -> &'static LabTables {
    static TABLES: OnceLock<LabTables> = OnceLock::new();
    TABLES.get_or_init(build_tables)
}

#[inline]
fn descale(value: i64, bits: u32) -> i64 {
    (value + (1 << (bits - 1))) >> bits
}

/// `cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)` for one 8-bit pixel.
#[inline]
pub fn rgb_to_lab(tables: &LabTables, rgb: [u8; 3]) -> [u8; 3] {
    let [r, g, b] = rgb.map(|v| tables.gamma[v as usize] as i64);
    let f = |row: [i64; 3]| tables.cbrt[descale(r * row[0] + g * row[1] + b * row[2], LAB_SHIFT) as usize] as i64;
    let (fx, fy, fz) = (f(TO_XYZ[0]), f(TO_XYZ[1]), f(TO_XYZ[2]));
    let scale = (116 * 255 + 50) / 100;
    let shift = -((16 * 255 * (1i64 << LAB_SHIFT2) + 50) / 100);
    let l = descale(scale * fy + shift, LAB_SHIFT2);
    let a = descale(500 * (fx - fy) + 128 * (1 << LAB_SHIFT2), LAB_SHIFT2);
    let b = descale(200 * (fy - fz) + 128 * (1 << LAB_SHIFT2), LAB_SHIFT2);
    [l.clamp(0, 255) as u8, a.clamp(0, 255) as u8, b.clamp(0, 255) as u8]
}

/// `cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)` for one 8-bit pixel.
#[inline]
pub fn lab_to_rgb(tables: &LabTables, lab: [u8; 3]) -> [u8; 3] {
    let [y, fy] = tables.lightness[lab[0] as usize];
    let a = ((5 * lab[1] as i64 * 53687 + (1 << 7)) >> 13) - 128 * BASE / 500;
    let b = ((lab[2] as i64 * 41943 + (1 << 4)) >> 9) - 128 * BASE / 200 + 1;
    let x = tables.ab[(fy + a - MINIMUM_AB) as usize] as i64;
    let z = tables.ab[(fy - b - MINIMUM_AB) as usize] as i64;
    FROM_XYZ.map(|row| {
        let linear = descale(row[0] * x + row[1] * y + row[2] * z, 14);
        tables.inverse_gamma[linear.clamp(0, INVERSE_GAMMA_SIZE as i64 - 1) as usize]
    })
}

/// OpenCV's `saturate_cast<uchar>(float)`: nearest, ties to even, clamped.
#[inline]
fn saturate(value: f32) -> u8 {
    round_half_even(value as f64).clamp(0.0, 255.0) as u8
}

/// `cv2.createCLAHE(clipLimit, (grid_x, grid_y)).apply(plane)` on 8-bit data.
pub fn clahe(source: &[u8], width: usize, height: usize, clip_limit: f64, grid_x: usize, grid_y: usize) -> Vec<u8> {
    assert!(source.len() == width * height && width > 0 && height > 0 && grid_x > 0 && grid_y > 0);
    // Pad right and bottom to a multiple of the grid; when either side needs it, both get it (a full
    // extra tile count of rows or columns on the side that already divided).
    let (padded_width, padded_height) = if width.is_multiple_of(grid_x) && height.is_multiple_of(grid_y) {
        (width, height)
    } else {
        (width + grid_x - width % grid_x, height + grid_y - height % grid_y)
    };
    let reflect = |index: usize, size: usize| -> usize {
        if index < size {
            index
        } else if size == 1 {
            0
        } else {
            // BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba, repeated for borders longer than the image.
            let period = 2 * (size - 1);
            let wrapped = index % period;
            if wrapped < size {
                wrapped
            } else {
                period - wrapped
            }
        }
    };
    let (tile_width, tile_height) = (padded_width / grid_x, padded_height / grid_y);
    let tile_pixels = tile_width * tile_height;
    let scale = 255f32 / tile_pixels as f32;
    let limit = if clip_limit > 0.0 { ((clip_limit * tile_pixels as f64 / 256.0) as i64).max(1) } else { 0 };
    let mut luts = vec![0u8; grid_x * grid_y * 256];
    for tile_y in 0..grid_y {
        for tile_x in 0..grid_x {
            let mut histogram = [0i64; 256];
            for y in tile_y * tile_height..(tile_y + 1) * tile_height {
                let row = reflect(y, height) * width;
                for x in tile_x * tile_width..(tile_x + 1) * tile_width {
                    histogram[source[row + reflect(x, width)] as usize] += 1;
                }
            }
            if limit > 0 {
                let mut clipped = 0;
                for bin in &mut histogram {
                    if *bin > limit {
                        clipped += *bin - limit;
                        *bin = limit;
                    }
                }
                let batch = clipped / 256;
                let mut residual = clipped - batch * 256;
                for bin in &mut histogram {
                    *bin += batch;
                }
                if residual != 0 {
                    let step = (256 / residual).max(1) as usize;
                    let mut index = 0;
                    while index < 256 && residual > 0 {
                        histogram[index] += 1;
                        index += step;
                        residual -= 1;
                    }
                }
            }
            let lut = &mut luts[(tile_y * grid_x + tile_x) * 256..][..256];
            let mut sum = 0i64;
            for (entry, bin) in lut.iter_mut().zip(histogram) {
                sum += bin;
                *entry = saturate(sum as f32 * scale);
            }
        }
    }
    let (inverse_width, inverse_height) = (1f32 / tile_width as f32, 1f32 / tile_height as f32);
    // Tile pair and weights along one axis, in single precision as OpenCV computes them.
    let blend = |position: usize, inverse: f32, tiles: usize| -> (usize, usize, f32, f32) {
        let centre = position as f32 * inverse - 0.5;
        let first = centre.floor();
        let weight = centre - first;
        let first = first as i64;
        (first.max(0) as usize, ((first + 1).min(tiles as i64 - 1)) as usize, 1.0 - weight, weight)
    };
    let columns: Vec<_> = (0..width).map(|x| blend(x, inverse_width, grid_x)).collect();
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        let (ty1, ty2, ya1, ya) = blend(y, inverse_height, grid_y);
        let (upper, lower) = (&luts[ty1 * grid_x * 256..], &luts[ty2 * grid_x * 256..]);
        for x in 0..width {
            let (tx1, tx2, xa1, xa) = columns[x];
            let value = source[y * width + x] as usize;
            let (first, second) = (tx1 * 256 + value, tx2 * 256 + value);
            let result = (upper[first] as f32 * xa1 + upper[second] as f32 * xa) * ya1
                + (lower[first] as f32 * xa1 + lower[second] as f32 * xa) * ya;
            out[y * width + x] = saturate(result);
        }
    }
    out
}

/// Gamma then CLAHE on Lab lightness, in place on interleaved 8-bit RGB.
/// A gamma of 0 or 1 and a clip limit of 0 or less switch the respective step off.
pub fn contrast_image(rgb: &mut [u8], width: usize, height: usize, gamma: f64, clahe_clip: f64, clahe_grid: usize) {
    assert_eq!(rgb.len(), width * height * 3);
    if gamma != 0.0 && gamma != 1.0 {
        let table = gamma_table(gamma);
        for value in rgb.iter_mut() {
            *value = table[*value as usize];
        }
    }
    if clahe_clip > 0.0 {
        let tables = tables();
        let pixels = rgb.as_chunks_mut::<3>().0;
        let mut lab: Vec<[u8; 3]> = pixels.iter().map(|p| rgb_to_lab(tables, *p)).collect();
        let lightness: Vec<u8> = lab.iter().map(|p| p[0]).collect();
        let equalised = clahe(&lightness, width, height, clahe_clip, clahe_grid, clahe_grid);
        for ((pixel, target), l) in lab.iter_mut().zip(pixels).zip(equalised) {
            pixel[0] = l;
            *target = lab_to_rgb(tables, *pixel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/dense-native/opencv-contrast.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn bytes(value: &Value) -> Vec<u8> {
        value.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect()
    }

    #[test]
    fn gamma_table_is_the_square_root_profile() {
        let fixture = fixture();
        assert_eq!(gamma_table(0.5).to_vec(), bytes(&fixture["gamma_half"]));
        assert_eq!(gamma_table(1.0).to_vec(), (0..=255).collect::<Vec<u8>>());
    }

    #[test]
    fn lab_tables_and_conversions_match_opencv() {
        let fixture = fixture();
        let tables = tables();
        // Sums of the tables as the generator computed them from its own reconstruction, which it
        // checked against cv2 for every colour.
        let sums = &fixture["table_sums"];
        assert_eq!(tables.gamma.iter().map(|&v| v as u64).sum::<u64>(), sums["gamma"].as_u64().unwrap());
        assert_eq!(tables.cbrt.iter().map(|&v| v as u64).sum::<u64>(), sums["cbrt"].as_u64().unwrap());
        assert_eq!(tables.lightness.iter().map(|v| (v[0] + 3 * v[1]) as u64).sum::<u64>(), sums["lightness"].as_u64().unwrap());
        assert_eq!(tables.ab.iter().map(|&v| v as i64).sum::<i64>(), sums["ab"].as_i64().unwrap());
        assert_eq!(tables.inverse_gamma.iter().map(|&v| v as u64).sum::<u64>(), sums["inverse_gamma"].as_u64().unwrap());
        let colours = bytes(&fixture["colours"]["rgb"]);
        let lab = bytes(&fixture["colours"]["lab"]);
        let back = bytes(&fixture["colours"]["rgb_from_lab"]);
        assert!(colours.len() >= 3000);
        let triples = |values: &[u8]| values.as_chunks::<3>().0.to_vec();
        for ((rgb, expected), returned) in triples(&colours).into_iter().zip(triples(&lab)).zip(triples(&back)) {
            assert_eq!(rgb_to_lab(tables, rgb), expected, "rgb {rgb:?}");
            // The fixture's third list is cv2's LAB2RGB of the colour read as a Lab triple.
            assert_eq!(lab_to_rgb(tables, rgb), returned, "lab {rgb:?}");
        }
    }

    #[test]
    fn clahe_matches_opencv_on_small_planes() {
        let fixture = fixture();
        let cases = fixture["clahe"].as_array().unwrap();
        assert!(cases.len() >= 6);
        for case in cases {
            let (width, height) = (case["width"].as_u64().unwrap() as usize, case["height"].as_u64().unwrap() as usize);
            let (clip, grid) = (case["clip"].as_f64().unwrap(), case["grid"].as_u64().unwrap() as usize);
            let ours = clahe(&bytes(&case["input"]), width, height, clip, grid, grid);
            assert_eq!(ours, bytes(&case["output"]), "{}x{} clip {clip} grid {grid}", width, height);
        }
    }

    #[test]
    fn contrast_image_matches_the_reference_profile() {
        let fixture = fixture();
        for case in fixture["contrast"].as_array().unwrap() {
            let (width, height) = (case["width"].as_u64().unwrap() as usize, case["height"].as_u64().unwrap() as usize);
            let mut rgb = bytes(&case["input"]);
            let original = rgb.clone();
            contrast_image(&mut rgb, width, height, 1.0, 0.0, 8);
            assert_eq!(rgb, original);
            contrast_image(
                &mut rgb,
                width,
                height,
                case["gamma"].as_f64().unwrap(),
                case["clip"].as_f64().unwrap(),
                case["grid"].as_u64().unwrap() as usize,
            );
            assert_eq!(rgb, bytes(&case["output"]), "{}x{}", width, height);
        }
    }
}
