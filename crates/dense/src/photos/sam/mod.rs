//! Native SAM 2.1 masks (`--masks sam`): what `scripts/turntable_mesh/segment.py`
//! does, without Python.
//!
//! | Module | Content |
//! | --- | --- |
//! | `prompts` | the automatic prompts from the coarse masks: common box, farthest interior point, further positives, negatives around the box |
//! | `image` | the photo as network input (antialiased resize, normalisation); mask logits back at photo size |
//! | `select` | cleaning the candidate masks and choosing one |
//! | `backend` | the model directory (`model.json`) and the interface to the network |
//! | `onnx` | ONNX Runtime backend (cargo feature `sam-onnx`) |
//! | `ggml` | CrispEmbed's ggml engine through `libcrispembed-sam2` (cargo feature `sam-ggml`) |
//! | `fetch` | the default model, downloaded on first use and checked against pinned SHA-256 (`sam-onnx`) |
//! | `provider` | the mask provider: files, progress, report |
//!
//! Only `onnx` and `provider` depend on the platform; the rest is plain Rust
//! and builds for every target.

pub mod backend;
#[cfg(feature = "sam-onnx")]
pub mod fetch;
#[cfg(feature = "sam-ggml")]
pub mod ggml;
pub mod image;
#[cfg(feature = "sam-onnx")]
pub mod onnx;
pub mod prompts;
pub mod provider;
pub mod select;

use anyhow::bail;
use serde_json::{json, Value};

use crate::inputs::Plane;

use backend::{ModelInfo, SamBackend};
use prompts::Prompt;

/// What `segment.py` takes as `--multimask` and `--preserve-holes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
    /// Let the network propose three masks and take the best that agrees with the points.
    pub multimask: bool,
    /// Keep background enclosed by the chosen region instead of filling it.
    pub preserve_holes: bool,
}

/// The silhouette of one photo.
#[derive(Debug, Clone)]
pub struct Segmented {
    /// The chosen candidate as the network gave it (0/1).
    pub raw: Plane<u8>,
    /// Its region connected to the primary point (0/1).
    pub clean: Plane<u8>,
    pub metrics: Value,
    /// Seconds for the image (resize and encoder) and for the prompts (decoder, enlargement, selection).
    pub seconds: [f64; 2],
}

/// The prompts as the network takes them: the two box corners, then the points, in pixels of its input.
pub fn model_prompts(prompt: &Prompt, width: usize, height: usize, size: usize) -> (Vec<[f32; 2]>, Vec<i64>) {
    let scaled = |x: usize, y: usize| [x as f32 / width as f32 * size as f32, y as f32 / height as f32 * size as f32];
    let [x0, y0, x1, y1] = prompt.box_xyxy;
    let mut points = vec![scaled(x0, y0), scaled(x1, y1)];
    let mut labels = vec![2i64, 3];
    points.extend(prompt.points.iter().map(|p| scaled(p[0], p[1])));
    labels.extend(prompt.labels.iter().map(|&l| l as i64));
    (points, labels)
}

/// One photo (8-bit RGB, row by row) through the network and the selection.
pub fn segment(
    backend: &mut dyn SamBackend,
    model: &ModelInfo,
    rgb: &[u8],
    width: usize,
    height: usize,
    prompt: &Prompt,
    settings: Settings,
) -> anyhow::Result<Segmented> {
    let started = web_time::Instant::now();
    let input = image::model_input(rgb, width, height, model.image_size, model.mean, model.std);
    backend.set_image(&input)?;
    drop(input);
    let embedded = started.elapsed().as_secs_f64();
    let (points, labels) = model_prompts(prompt, width, height, model.image_size);
    let prediction = backend.predict(&points, &labels)?;
    let per_mask = model.mask_size * model.mask_size;
    let tokens = prediction.scores.len();
    if tokens == 0 || prediction.logits.len() != tokens * per_mask || prediction.logits.iter().any(|v| !v.is_finite()) {
        bail!("SAM returned malformed masks or scores");
    }
    let enlarged = |token: usize| {
        image::mask_from_logits(&prediction.logits[token * per_mask..(token + 1) * per_mask], model.mask_size, width, height)
    };
    let (raw, clean, metrics) = if settings.multimask && tokens > 1 {
        let masks: Vec<Plane<u8>> = (1..tokens).map(enlarged).collect();
        let selection = select::select_prediction(&masks, &prediction.scores[1..], prompt, settings.preserve_holes);
        if !selection.passed {
            bail!("no SAM mask candidate satisfies all photographic point cues after cleanup");
        }
        (masks.into_iter().nth(selection.selected).unwrap_or_else(|| selection.clean.clone()), selection.clean, selection.metrics)
    } else {
        let token = match model.stability {
            Some((delta, threshold)) => select::single_mask_token(&prediction.logits, &prediction.scores, delta, threshold),
            None => 0,
        };
        let raw = enlarged(token);
        let cleaned =
            select::clean_prediction(&raw, prompt.point, settings.preserve_holes).map_err(|reason| anyhow::anyhow!("{reason}"))?;
        if !select::points_agree(&cleaned.clean, prompt) {
            bail!("cleaned silhouette violates an explicit positive/negative photo point");
        }
        let metrics = json!({
            "predicted_iou": prediction.scores[token], "raw_components": cleaned.raw_components,
            "raw_foreground_pixels": cleaned.raw_foreground_pixels, "foreground_pixels": cleaned.foreground_pixels,
            "holes_filled_pixels": cleaned.holes_filled_pixels, "output_token": token,
        });
        (raw, cleaned.clean, metrics)
    };
    Ok(Segmented { raw, clean, metrics, seconds: [embedded, started.elapsed().as_secs_f64() - embedded] })
}

#[cfg(test)]
mod tests {
    use super::backend::{Prediction, SamBackend};
    use super::*;

    /// A stand-in network: every token is a disc around the first object point, of 9, 3, 6 and 30 mask cells.
    struct Discs {
        size: usize,
        mask: usize,
        seen: Option<usize>,
    }

    impl SamBackend for Discs {
        fn description(&self) -> String {
            "discs".into()
        }

        fn set_image(&mut self, input: &[f32]) -> anyhow::Result<()> {
            self.seen = Some(input.len());
            Ok(())
        }

        fn predict(&mut self, points: &[[f32; 2]], labels: &[i64]) -> anyhow::Result<Prediction> {
            assert_eq!(self.seen, Some(3 * self.size * self.size));
            assert_eq!(labels[..3], [2, 3, 1]);
            let scale = self.mask as f32 / self.size as f32;
            let (cx, cy) = (points[2][0] * scale, points[2][1] * scale);
            let mut logits = Vec::new();
            for radius in [9.0f32, 3.0, 6.0, 30.0] {
                for y in 0..self.mask {
                    for x in 0..self.mask {
                        logits.push(radius - ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt());
                    }
                }
            }
            Ok(Prediction { logits, scores: vec![0.5, 0.6, 0.9, 0.95] })
        }
    }

    #[test]
    fn a_photo_goes_through_prompts_network_and_selection() {
        let description = backend::tests::description();
        let mut model = ModelInfo::parse(std::path::Path::new("."), &description).unwrap();
        (model.image_size, model.mask_size) = (64, 16);
        let (width, height) = (96, 48);
        let rgb = vec![200u8; width * height * 3];
        // An object point in the middle, a background point 42 pixels to its right (7 mask cells).
        let prompt = Prompt { point: [48, 24], points: vec![[48, 24], [90, 24]], labels: vec![1, 0], box_xyxy: [30, 10, 66, 38] };
        let (points, labels) = model_prompts(&prompt, width, height, 64);
        assert_eq!((points[0], points[1], points[2]), ([20.0, 64.0 * 10.0 / 48.0], [44.0, 64.0 * 38.0 / 48.0], [32.0, 32.0]));
        assert_eq!(labels, vec![2, 3, 1, 0]);
        let mut network = Discs { size: 64, mask: 16, seen: None };
        // Three candidates: the largest (best scored) covers the background point, so the middle one is taken.
        let many = segment(&mut network, &model, &rgb, width, height, &prompt, Settings { multimask: true, preserve_holes: true }).unwrap();
        assert_eq!(many.metrics["selected_index"], 1);
        assert_eq!(many.raw.data, many.clean.data);
        assert!(many.clean.data[24 * width + 48] == 1 && many.clean.data[24 * width + 90] == 0);
        // A single mask (the fallback to the other tokens switched off): token 0 covers the background point,
        // which is an error as in the reference.
        model.stability = None;
        let one = segment(&mut network, &model, &rgb, width, height, &prompt, Settings { multimask: false, preserve_holes: true });
        assert!(one.unwrap_err().to_string().contains("violates"));
        let lone = Prompt { points: vec![[48, 24]], labels: vec![1], ..prompt };
        let one = segment(&mut network, &model, &rgb, width, height, &lone, Settings { multimask: false, preserve_holes: false }).unwrap();
        assert_eq!(one.metrics["output_token"], 0);
    }
}
