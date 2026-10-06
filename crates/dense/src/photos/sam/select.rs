//! From the network's candidate masks to one silhouette: a port of
//! `clean_prediction` and `select_prediction` of
//! `scripts/turntable_mesh/segment.py`, and of the mask decoder's choice
//! between its single-mask and multi-mask outputs.
//!
//! A candidate is cleaned by keeping the 4-connected region that holds the
//! primary point (holes are filled unless they are to be preserved). Among
//! the cleaned candidates that agree with every point (object points inside,
//! background points outside) the one the network scores highest is taken.

use serde_json::{json, Value};

use crate::inputs::Plane;

use super::super::coarse::label;
use super::prompts::Prompt;

/// One cleaned candidate.
#[derive(Debug, Clone)]
pub struct Cleaned {
    pub clean: Plane<u8>,
    pub raw_components: usize,
    pub raw_foreground_pixels: usize,
    pub foreground_pixels: usize,
    pub holes_filled_pixels: usize,
}

/// The chosen silhouette and how it was chosen.
#[derive(Debug, Clone)]
pub struct Selection {
    pub clean: Plane<u8>,
    /// Index among the candidates given.
    pub selected: usize,
    /// Whether the chosen candidate agrees with every point after cleaning.
    pub passed: bool,
    pub metrics: Value,
}

/// Background regions that do not reach the border become object (`scipy.ndimage.binary_fill_holes`).
fn fill_holes(mask: &Plane<u8>) -> Plane<u8> {
    let (width, height) = (mask.width, mask.height);
    let background = Plane { width, height, data: mask.data.iter().map(|&m| (m == 0) as u8).collect() };
    let (labels, count) = label(&background, false);
    let mut outside = vec![false; count + 1];
    for x in 0..width {
        outside[labels.data[x] as usize] = true;
        outside[labels.data[(height - 1) * width + x] as usize] = true;
    }
    for y in 0..height {
        outside[labels.data[y * width] as usize] = true;
        outside[labels.data[y * width + width - 1] as usize] = true;
    }
    Plane { width, height, data: mask.data.iter().zip(&labels.data).map(|(&m, &l)| (m != 0 || !outside[l as usize]) as u8).collect() }
}

/// The region of `raw` (0/1) connected to `point`, holes filled unless preserved.
pub fn clean_prediction(raw: &Plane<u8>, point: [usize; 2], preserve_holes: bool) -> Result<Cleaned, &'static str> {
    let at = point[1] * raw.width + point[0];
    if raw.data[at] == 0 {
        return Err("SAM prediction does not contain its positive prompt");
    }
    let (labels, count) = label(raw, false);
    let chosen = labels.data[at];
    let connected = Plane { width: raw.width, height: raw.height, data: labels.data.iter().map(|&l| (l == chosen) as u8).collect() };
    let connected_pixels = connected.data.iter().filter(|&&m| m != 0).count();
    let clean = if preserve_holes { connected } else { fill_holes(&connected) };
    let foreground_pixels = clean.data.iter().filter(|&&m| m != 0).count();
    if foreground_pixels == clean.data.len() {
        return Err("SAM returned an unbounded all-foreground silhouette");
    }
    Ok(Cleaned {
        clean,
        raw_components: count,
        raw_foreground_pixels: raw.data.iter().filter(|&&m| m != 0).count(),
        foreground_pixels,
        holes_filled_pixels: foreground_pixels - connected_pixels,
    })
}

/// Whether every point lies on the side its label says.
pub fn points_agree(mask: &Plane<u8>, prompt: &Prompt) -> bool {
    prompt.points.iter().zip(&prompt.labels).all(|(p, &expected)| (mask.data[p[1] * mask.width + p[0]] != 0) == (expected != 0))
}

/// Chooses among candidate masks (0/1) with their predicted scores.
pub fn select_prediction(masks: &[Plane<u8>], scores: &[f32], prompt: &Prompt, preserve_holes: bool) -> Selection {
    let mut cleaned: Vec<Option<Cleaned>> = Vec::new();
    let mut rows = Vec::new();
    let mut passed = Vec::new();
    for (index, raw) in masks.iter().enumerate() {
        let mut row = json!({"index": index, "predicted_iou": scores[index], "passed": false});
        match clean_prediction(raw, prompt.point, preserve_holes) {
            Ok(candidate) => {
                let agrees = points_agree(&candidate.clean, prompt);
                row["passed"] = json!(agrees);
                if !agrees {
                    row["reason"] = json!("cleaned silhouette violates photographic point cues");
                }
                passed.push(agrees);
                cleaned.push(Some(candidate));
            }
            Err(reason) => {
                row["reason"] = json!(reason);
                passed.push(false);
                cleaned.push(None);
            }
        }
        rows.push(row);
    }
    // The best positive component stays available as evidence when nothing passes.
    let valid: Vec<usize> = (0..masks.len()).filter(|&i| passed[i]).collect();
    let usable: Vec<usize> = (0..masks.len()).filter(|&i| cleaned[i].is_some()).collect();
    let eligible = if !valid.is_empty() {
        valid.clone()
    } else if !usable.is_empty() {
        usable
    } else {
        (0..masks.len()).collect()
    };
    // Highest score; the earlier candidate on a tie.
    let selected = eligible.iter().copied().fold(eligible[0], |best, i| if scores[i] > scores[best] { i } else { best });
    let (clean, mut metrics) = match cleaned[selected].take() {
        Some(candidate) => (
            candidate.clean,
            json!({
                "predicted_iou": scores[selected], "raw_components": candidate.raw_components,
                "raw_foreground_pixels": candidate.raw_foreground_pixels, "foreground_pixels": candidate.foreground_pixels,
                "holes_filled_pixels": candidate.holes_filled_pixels,
            }),
        ),
        None => {
            let pixels = masks[selected].data.iter().filter(|&&m| m != 0).count();
            (
                masks[selected].clone(),
                json!({
                    "predicted_iou": scores[selected], "raw_foreground_pixels": pixels, "foreground_pixels": pixels,
                    "holes_filled_pixels": 0, "cleanup_unavailable": true,
                }),
            )
        }
    };
    metrics["selected_index"] = json!(selected);
    metrics["candidate_count"] = json!(masks.len());
    metrics["selection_passed"] = json!(!valid.is_empty());
    metrics["candidates"] = json!(rows);
    Selection { clean, selected, passed: !valid.is_empty(), metrics }
}

/// Which output token to use when a single mask is asked for: token 0, unless
/// its logits are unstable (the share of pixels above `delta` among those
/// above `-delta` is below `threshold`); then the best scored of tokens 1 to 3.
/// `logits` holds the tokens' low-resolution masks one after another.
pub fn single_mask_token(logits: &[f32], scores: &[f32], delta: f32, threshold: f32) -> usize {
    let tokens = scores.len();
    let first = &logits[..logits.len() / tokens.max(1)];
    let inner = first.iter().filter(|&&v| v > delta).count() as f32;
    let outer = first.iter().filter(|&&v| v > -delta).count() as f32;
    let stability = if outer > 0.0 { inner / outer } else { 1.0 };
    if stability >= threshold || tokens < 2 {
        return 0;
    }
    (1..tokens).fold(1, |best, i| if scores[i] > scores[best] { i } else { best })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(width: usize, height: usize, boxes: &[(std::ops::Range<usize>, std::ops::Range<usize>, u8)]) -> Plane<u8> {
        let mut plane = Plane::<u8>::new(width, height);
        for (ys, xs, value) in boxes {
            for y in ys.clone() {
                for x in xs.clone() {
                    plane.data[y * width + x] = *value;
                }
            }
        }
        plane
    }

    fn area(mask: &Plane<u8>) -> usize {
        mask.data.iter().filter(|&&m| m != 0).count()
    }

    /// The cases and expected numbers are those `select_prediction` of segment.py gives.
    #[test]
    fn selection_follows_the_reference() {
        let (w, h) = (12, 8);
        // A block with a one-pixel hole and a stray pixel; a block that covers the background point; one without the point.
        let a = block(w, h, &[(1..7, 1..7, 1), (3..4, 3..4, 0), (1..2, 10..11, 1)]);
        let b = block(w, h, &[(2..6, 2..11, 1)]);
        let c = block(w, h, &[(0..2, 0..2, 1)]);
        let masks = [a.clone(), b, c.clone()];
        let scores = [0.5, 0.9, 0.99];
        let prompt = Prompt { point: [2, 2], points: vec![[2, 2], [5, 5], [9, 3]], labels: vec![1, 1, 0], box_xyxy: [0, 0, w, h] };
        let kept = select_prediction(&masks, &scores, &prompt, true);
        assert_eq!((kept.selected, kept.passed, area(&kept.clean)), (0, true, 35));
        assert_eq!((kept.metrics["raw_components"].as_u64(), kept.metrics["raw_foreground_pixels"].as_u64()), (Some(2), Some(36)));
        let filled = select_prediction(&masks, &scores, &prompt, false);
        assert_eq!((filled.selected, area(&filled.clean), filled.metrics["holes_filled_pixels"].as_u64()), (0, 36, Some(1)));
        // A background point in the hole: filling the hole violates it, and then nothing passes.
        let hole = Prompt { point: [2, 2], points: vec![[2, 2], [3, 3]], labels: vec![1, 0], box_xyxy: [0, 0, w, h] };
        assert_eq!(select_prediction(&masks, &scores, &hole, true).selected, 0);
        let none = select_prediction(&masks, &scores, &hole, false);
        assert_eq!((none.selected, none.passed, area(&none.clean)), (1, false, 36));
        assert_eq!(none.metrics["candidates"][2]["reason"], "SAM prediction does not contain its positive prompt");
        // Equal scores: the earlier candidate.
        let tie = select_prediction(&[a.clone(), a, c], &[0.7, 0.7, 0.99], &prompt, true);
        assert_eq!(tie.selected, 0);
        // A mask that covers everything is refused.
        assert!(clean_prediction(&block(4, 4, &[(0..4, 0..4, 1)]), [1, 1], true).is_err());
    }

    #[test]
    fn an_unstable_single_mask_gives_way_to_the_best_of_the_others() {
        // Four tokens of four pixels each.
        let stable = [1.0, 1.0, -1.0, -1.0];
        let unstable = [1.0, 0.01, -1.0, -1.0];
        let rest = [0.0f32; 12];
        let scores = [0.9, 0.2, 0.8, 0.8];
        assert_eq!(single_mask_token(&[&stable[..], &rest[..]].concat(), &scores, 0.05, 0.98), 0);
        assert_eq!(single_mask_token(&[&unstable[..], &rest[..]].concat(), &scores, 0.05, 0.98), 2);
        assert_eq!(single_mask_token(&[-1.0; 16], &scores, 0.05, 0.98), 0);
    }
}
