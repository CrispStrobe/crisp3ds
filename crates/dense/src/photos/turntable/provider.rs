//! The `turntable` camera provider inside the `photos` command.

use crate::photos::fs::Stored as _;
use anyhow::{anyhow, bail};
use serde_json::json;

use crate::photos::cameras::CameraProvider;
use crate::photos::markers::linalg::Camera;
use crate::photos::options::Options;
use crate::photos::providers::{Provider, CAMERAS_TURNTABLE};
use crate::photos::run::Run;
use crate::photos::solution::{Lens, Solution};
use crate::photos::staging::capture_name;
use crate::photos::util;

use super::solver::{self, Matches};
use super::{features, matching};

pub struct Turntable;

impl CameraProvider for Turntable {
    fn info(&self) -> &'static Provider {
        &CAMERAS_TURNTABLE
    }

    fn check(&self, options: &Options) -> anyhow::Result<()> {
        if let Some(path) = &options.turntable.matches {
            if !path.stored_file() {
                bail!("--turntable-matches {}: not found", path.display());
            }
        }
        Ok(())
    }

    fn recover(&self, run: &mut Run, lens: Option<&Lens>) -> anyhow::Result<Solution> {
        let lens = *lens.ok_or_else(|| anyhow!("the turntable provider needs the declared lens (--calibration)"))?;
        let (options, out) = (run.options, run.out);
        let camera = Camera::from_lens(&lens);
        let (count, threads, timeouts) = (options.photo_count, options.threads, &options.timeouts);
        let names: Vec<String> = (0..count).map(capture_name).collect();
        let matches = match &options.turntable.matches {
            // Debug input: features and matches made elsewhere (the prototype's `--export-matches`).
            Some(path) => {
                let loaded = Matches::from_json(&util::read_json(path)?)?;
                if loaded.keypoints.len() != count {
                    bail!("--turntable-matches lists {} photos, the capture has {count}", loaded.keypoints.len());
                }
                loaded
            }
            None => {
                let settings = features::Settings { maximum: options.turntable.features, ..features::Settings::default() };
                let found =
                    run.internal("cameras", "turntable-features", timeouts.features, 0.04, 0.4, "Detecting features", count, |watch| {
                        let work = |n: usize| -> anyhow::Result<features::Features> {
                            let photo = crate::photos::staging::open_photo(&out.join("work/contrast").join(&names[n]))?;
                            let mask = crate::scene::read_gray(&out.join("masks").join(&names[n]))?;
                            Ok(features::detect(&photo.gray(), Some(&mask), &settings))
                        };
                        util::parallel(count, threads, work, watch)
                    })?;
                let span = options.turntable.span;
                let pairs = matching::ring_pairs(count, span, options.open_turn);
                let matched = run.internal(
                    "cameras",
                    "turntable-matching",
                    timeouts.matching,
                    0.4,
                    0.6,
                    "Matching features",
                    pairs.len(),
                    |watch| {
                        let work = |k: usize| -> anyhow::Result<Vec<(u32, u32)>> {
                            Ok(matching::match_pair(&found[pairs[k].0], &found[pairs[k].1], 0.8))
                        };
                        util::parallel(pairs.len(), threads, work, watch)
                    },
                )?;
                Matches {
                    keypoints: found.iter().map(|f| f.points.clone()).collect(),
                    pairs: pairs.iter().zip(matched).map(|(&(i, j), list)| (i, j, list)).collect(),
                }
            }
        };
        let counted: Vec<(usize, usize, usize)> = matches.pairs.iter().map(|p| (p.0, p.1, p.2.len())).collect();
        let by_distance = matching::matches_by_distance(count, &counted);
        run.note("turntable_order", json!({"median_matches_by_distance": by_distance}));
        matching::check_order(&by_distance).map_err(|reason| anyhow!(reason))?;
        let settings = solver::Settings { open_turn: options.open_turn };
        let solved =
            run.internal("cameras", "turntable-solve", timeouts.sfm, 0.6, 0.82, "Recovering cameras (turntable solver)", 1, |_| {
                solver::solve(&matches, &camera, &settings)
            })?;
        let folder = out.join("sfm/turntable");
        crate::photos::fs::create_dir_all(&folder)?;
        let mut report = solved.report.clone();
        let mut counts: Vec<f64> = matches.keypoints.iter().map(|k| k.len() as f64).collect();
        let total: usize = matches.pairs.iter().map(|p| p.2.len()).sum();
        report["features_per_photo"] =
            json!({"min": counts.iter().cloned().fold(f64::INFINITY, f64::min), "median": crate::inputs::median_f64(&mut counts)});
        report["matches"] = json!({"pairs": matches.pairs.len(), "total": total});
        util::write_json(&folder.join("report.json"), &report, 1)?;
        if let Some(object) = report.as_object_mut() {
            object.remove("steps_deg");
            object.remove("adjustments");
        }
        run.note("turntable", report);
        Ok(solver::solution(&lens, &names, &matches, &solved))
    }

    fn intermediates(&self) -> &'static [&'static str] {
        &[]
    }
}
